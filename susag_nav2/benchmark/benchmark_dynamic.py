#!/usr/bin/env python3
# Copyright 2026 SLIP project
#
# Dynamic-obstacle benchmark orchestrator (TG-MPPI thesis, 2026-09-17).
#
# Sibling of benchmark_barn.py (same lifecycle pattern: one fresh child process
# per trial, launches as SIGINT-able process groups), built for the polynomial-
# track worlds from make_open_world.py (world_idx dyn1..dynN). What makes the
# trials IDENTICAL, which manual runs could not guarantee:
#   * Gazebo is restarted for every trial, so sim time -- and with it every
#     obstacle's position -- starts from zero each time;
#   * the first goal is sent at a FIXED SIM TIME (--start-sim-time), not "as
#     soon as Nav2 happens to be ready", so every trial meets the same scene;
#     a trial whose stack was not ready by then is flagged late_start;
#   * the goal sequence is fixed (the generator's keep-out points), each goal is
#     sent only after the previous one has finished -- nothing is ever preempted;
#   * every trial runs on its own ROS_DOMAIN_ID and ROS_LOG_DIR, and records a
#     bag plus the git revisions of the code it ran.
# Scoring is ground truth (footprint vs obstacle discs, walls), per leg, with
# who-closed-the-gap classification of the FIRST contact -- position-forced
# obstacles can shove the robot unphysically, so later contacts are not trusted.
#
# Conditions (see navigation.launch.py for the switches):
#   A   stock Nav2 MPPI (navigation_sim_tight.yaml, unmodified baseline)
#   B   TG-MPPI full, legacy sample split
#   Bp  TG-MPPI full, equal sample split (amoeba_sandbox allocation)
#   C   TG-MPPI without space-time
#   D   plain MPPI + predicted-obstacle critic (TG-MPPI with the bias off)
#
# Usage:
#   python3 benchmark_dynamic.py --conditions A B Bp D --worlds dyn1 dyn2 dyn3 --reps 3
#   python3 benchmark_dynamic.py --dry-run --conditions B --worlds dyn1
#   python3 benchmark_dynamic.py --report                 # latest run
#   python3 benchmark_dynamic.py --report --run 20260917_1700
#   python3 benchmark_dynamic.py --score-bag <bag dir> [--log <controller log>]
# Close any Gazebo / Nav2 you have open first: trials kill leftover processes.

import argparse
import csv
import datetime
import glob
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time

WS = os.path.expanduser("~/robohouse_ws")
PARAM_DIR = f"{WS}/src/susag_nav2/param"
TG_YAML = f"{PARAM_DIR}/navigation_tgmppi_tight.yaml"
SIM_YAML = f"{PARAM_DIR}/navigation_sim_tight.yaml"
WORLD_DIR = f"{WS}/src/BARN_dataset/scaled_1/world_files"
RESULTS_ROOT = f"{WS}/benchmark_results/dynamic"
# The package lives in src/susag_new_model but package.xml names it this:
GZ_PKG = "susag_updated_model_description"

START = (0.0, 1.0, math.pi / 2)
# make_open_world.py keeps every obstacle track 1.2 m clear of these points and
# requires each track to pass within 2 m of the legs between them.
POINT_A = (2.5, 11.3)
POINT_B = (-2.8, 0.5)
ROUTE = [POINT_A, POINT_B, POINT_A, POINT_B]

ROOM_X, ROOM_Y0, ROOM_Y1 = 4.0, -0.5, 12.5
BAG_REGEX = ("^/(ground_truth/odom|odom|cmd_vel|cmd_vel_nav|rosout|plan|clock|tf|tf_static|"
             "moving_obstacle_[0-9]+/.*)$")
RECOVERY_RE = re.compile(r"fail to compute path|Failed to make progress|Running (backup|wait)")

TG_COMMON = ["dynamic_obstacles:=true", "backend:=cpu"]
CONDITIONS = {
    "A":  ("stock Nav2 MPPI", SIM_YAML, []),
    "B":  ("TG-MPPI full, legacy split", TG_YAML,
           TG_COMMON + ["ablation:=full", "group_allocation:=legacy"]),
    "E":  ("TG-MPPI + space-time blob", TG_YAML,
           TG_COMMON + ["ablation:=full", "group_allocation:=legacy", "spacetime_blob:=true"]),
    "F":  ("TG-MPPI + space-time blob pods", TG_YAML,
           TG_COMMON + ["ablation:=full", "group_allocation:=legacy", "spacetime_blob:=pods"]),
    "Bp": ("TG-MPPI full, equal split", TG_YAML,
           TG_COMMON + ["ablation:=full", "group_allocation:=equal"]),
    "C":  ("TG-MPPI without space-time", TG_YAML,
           TG_COMMON + ["ablation:=no_spacetime", "group_allocation:=legacy"]),
    "D":  ("plain MPPI + prediction critic", TG_YAML, TG_COMMON + ["ablation:=no_topology"]),
}
STRAY_PATTERNS = ("gzserver", "gzclient", "component_container_isolated", "robot_state_publisher",
                  "spawn_entity.py", "ground_truth_localizer.py", "tracked_obstacle_scan_filter.py",
                  "obstacle_tracking_noise.py", "bag record")


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------

class Launch:
    """A subprocess we can SIGINT cleanly as a process group (from benchmark_barn.py)."""

    def __init__(self, args, log_path, env):
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT,
                                     preexec_fn=os.setsid, env=env)

    def alive(self):
        return self.proc.poll() is None

    def stop(self, timeout=20.0):
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                self.proc.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.log.close()


def _ancestors():
    """PIDs of this process and every parent up to init."""
    pids, pid = set(), os.getpid()
    while pid > 1:
        pids.add(pid)
        try:
            with open(f"/proc/{pid}/stat") as f:
                pid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def kill_strays():
    """Nothing from a previous trial may survive.

    Matches by command line, but never touches this process's own ancestry: a
    plain `pkill -f gzserver` also kills any shell whose command text happens to
    contain "gzserver" -- including the one that started this benchmark
    (2026-09-17, first smoke test died that way with exit 144)."""
    keep = _ancestors()
    victims = set()
    for pat in STRAY_PATTERNS:
        out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout
        victims.update(int(x) for x in out.split() if x.strip().isdigit())
    victims -= keep
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if victims:
        time.sleep(2.0)


def trial_env(domain, log_dir):
    env = dict(os.environ)
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env["ROS_DOMAIN_ID"] = str(domain)
    env["ROS_LOG_DIR"] = log_dir
    uri = os.path.expanduser("~/.config/cyclonedds/local_sim.xml")
    if os.path.exists(uri):
        env["CYCLONEDDS_URI"] = "file://" + uri
    return env


def git_rev(path):
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return rev + ("-dirty" if dirty else "") if rev else "unknown"
    except Exception:
        return "unknown"


def yaw_between(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


# ---------------------------------------------------------------------------
# the trial node (only imported inside the per-trial child process)
# ---------------------------------------------------------------------------

def make_node():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from nav2_msgs.action import NavigateToPose
    from lifecycle_msgs.srv import GetState

    class DynBench(Node):
        def __init__(self):
            super().__init__("dynamic_benchmark",
                             parameter_overrides=[Parameter("use_sim_time", value=True)])
            self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
            self.pose = None
            self.create_subscription(Odometry, "/ground_truth/odom", self._gt, qos_profile_sensor_data)

        def _gt(self, m):
            q = m.pose.pose.orientation
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, 2 * math.atan2(q.z, q.w))

        def sim_now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def wait_for(self, cond, timeout, what):
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < timeout:
                if cond():
                    return True
                rclpy.spin_once(self, timeout_sec=0.2)
            print(f"  !! timed out waiting for {what}", flush=True)
            return False

        def wait_node_active(self, name, timeout=90.0):
            cli = self.create_client(GetState, f"/{name}/get_state")
            t0 = time.time()
            try:
                while rclpy.ok() and time.time() - t0 < timeout:
                    if cli.wait_for_service(timeout_sec=1.0):
                        fut = cli.call_async(GetState.Request())
                        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
                        res = fut.result()
                        if res is not None and res.current_state.label == "active":
                            return True
                    rclpy.spin_once(self, timeout_sec=0.2)
            finally:
                self.destroy_client(cli)
            print(f"  !! {name} never became active", flush=True)
            return False

        def run_leg(self, goal_xy, goal_yaw, timeout_sim):
            goal = NavigateToPose.Goal()
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x, ps.pose.position.y = float(goal_xy[0]), float(goal_xy[1])
            ps.pose.orientation.z = math.sin(goal_yaw / 2.0)
            ps.pose.orientation.w = math.cos(goal_yaw / 2.0)
            goal.pose = ps
            leg = dict(goal=list(goal_xy), goal_yaw=goal_yaw, start_pose=list(self.pose or (0, 0, 0)))
            handle, t_retry = None, time.time()
            while rclpy.ok() and time.time() - t_retry < 30.0:
                ps.header.stamp = self.get_clock().now().to_msg()
                fut = self.nav.send_goal_async(goal)
                rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
                handle = fut.result()
                if handle is not None and handle.accepted:
                    break
                handle = None
                time.sleep(2.0)
            leg.update(t0_wall=time.time(), t0_sim=self.sim_now())
            if handle is None:
                leg.update(status="rejected", t1_wall=time.time(), t1_sim=self.sim_now())
                return leg
            res = handle.get_result_async()
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
                if res.done():
                    code = res.result().status
                    leg["status"] = {4: "succeeded", 5: "canceled", 6: "aborted"}.get(code, f"status{code}")
                    break
                if self.sim_now() - leg["t0_sim"] > timeout_sim:
                    cfut = handle.cancel_goal_async()
                    rclpy.spin_until_future_complete(self, cfut, timeout_sec=5.0)
                    self.wait_for(lambda: res.done(), 10.0, "cancel")
                    leg["status"] = "timeout"
                    break
            leg.update(t1_wall=time.time(), t1_sim=self.sim_now(), end_pose=list(self.pose or (0, 0, 0)))
            return leg

    return rclpy, DynBench


# ---------------------------------------------------------------------------
# one trial (fresh child process)
# ---------------------------------------------------------------------------

def run_trial(cond, world, rep, a):
    label, params, extra = CONDITIONS[cond]
    tdir = os.path.join(RESULTS_ROOT, a.run_id, world, cond, f"rep{rep}")
    os.makedirs(tdir, exist_ok=True)
    env = trial_env(a.domain, os.path.join(tdir, "ros_log"))
    os.environ.update(env)          # this process's own rclpy must join the same domain
    kill_strays()
    rclpy, DynBench = make_node()
    rclpy.init()
    node = DynBench()
    gz = nav = bag = None
    record = dict(condition=cond, condition_label=label, world=world, rep=rep,
                  max_speed=a.max_speed, tracking_quality=a.tracking_quality,
                  start_sim_time=a.start_sim_time, legs=[], started=datetime.datetime.now().isoformat(),
                  git_slip=git_rev(f"{WS}/src"), git_sandbox=git_rev(f"{WS}/amoeba_sandbox"))
    try:
        gz = Launch(["ros2", "launch", GZ_PKG, "gazebo_barn.launch.py", f"world_idx:={world}",
                     f"x_pose:={START[0]}", f"y_pose:={START[1]}", f"yaw:={START[2]:.4f}", "gui:=false"],
                    os.path.join(tdir, "gazebo.log"), env)
        ok = node.wait_for(lambda: node.pose is not None, 90.0, "gazebo ground truth")
        if ok:
            nav = Launch(["ros2", "launch", "susag_nav2", "navigation.launch.py", "sim:=true", "rviz:=false",
                          f"world_idx:={world}", f"max_speed:={a.max_speed}",
                          f"tracking_quality:={a.tracking_quality}", f"nav2_params:={params}"] + extra,
                         os.path.join(tdir, "nav2.log"), env)
            ok = (node.wait_node_active("controller_server") and node.wait_node_active("bt_navigator")
                  and node.wait_for(lambda: node.nav.server_is_ready(), 30.0, "navigate_to_pose server"))
        record["stack_ready_sim"] = round(node.sim_now(), 2)
        if ok:
            bag = Launch(["ros2", "bag", "record", "-o", os.path.join(tdir, "bag"), "-e", BAG_REGEX],
                         os.path.join(tdir, "bag.log"), env)
            t_end = time.time() + 3.0          # give the recorder time to discover topics
            while time.time() < t_end:
                rclpy.spin_once(node, timeout_sec=0.1)
            record["late_start"] = node.sim_now() > a.start_sim_time
            node.wait_for(lambda: node.sim_now() >= a.start_sim_time, 300.0, "start sim time")
            prev, fails = START[:2], 0
            for gxy in ROUTE[:a.legs]:
                leg = node.run_leg(gxy, yaw_between(prev, gxy), a.leg_timeout)
                record["legs"].append(leg)
                print(f"  leg -> ({gxy[0]:+.1f},{gxy[1]:.1f}): {leg['status']} in "
                      f"{leg['t1_sim'] - leg['t0_sim']:.1f} s sim", flush=True)
                fails = 0 if leg["status"] == "succeeded" else fails + 1
                if fails >= a.max_consecutive_failures:
                    record["ended_early"] = True
                    break
                prev = gxy
        else:
            record["error"] = "stack did not come up"
    finally:
        for p in (bag, nav, gz):
            if p is not None:
                p.stop()
        node.destroy_node()
        rclpy.shutdown()
        kill_strays()
    record["finished"] = datetime.datetime.now().isoformat()
    if record["legs"] and os.path.isdir(os.path.join(tdir, "bag")):
        try:
            record["score"] = score_bag(os.path.join(tdir, "bag"), record["legs"], world,
                                        find_controller_log(os.path.join(tdir, "ros_log")))
        except Exception as e:     # never lose the raw record because scoring failed
            record["score_error"] = repr(e)
    with open(os.path.join(tdir, "trial.json"), "w") as f:
        json.dump(record, f, indent=2)
    return 0 if "error" not in record else 1


# ---------------------------------------------------------------------------
# scoring (ground truth)
# ---------------------------------------------------------------------------

def load_footprint():
    import yaml
    fp = yaml.safe_load(open(TG_YAML))["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]
    foot = yaml.safe_load(fp) if isinstance(fp, str) else fp
    return foot, max(abs(p[0]) for p in foot), max(abs(p[1]) for p in foot)


def world_radii(world):
    txt = open(os.path.join(WORLD_DIR, f"world_{world}.world")).read()
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(
        r"<model name='moving_obstacle_(\d+)'>.*?<radius>([0-9.]+)</radius>", txt, re.S)}


def find_controller_log(log_dir):
    # Nav2 runs composed, so the controller writes into the container's log; launch.log
    # also mentions controller_server, so look at container logs first.
    containers = sorted(glob.glob(os.path.join(log_dir, "**", "component_container_isolated_*.log"),
                                  recursive=True))
    for f in containers + sorted(glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True)):
        if os.path.basename(f) == "launch.log":
            continue
        try:
            if "[controller_server]" in open(f, errors="ignore").read(400000):
                return f
        except OSError:
            pass
    return None


def controller_stats(log_path):
    if not log_path or not os.path.exists(log_path):
        return {}
    txt = open(log_path, errors="ignore").read()
    cyc = [float(x) for x in re.findall(r"cycle time \(compute_backend=\w+\): ([0-9.]+) ms", txt)]
    st = re.findall(r"crossing detected (\d+).*?modes appended (\d+), non-distinct (\d+)", txt)
    return dict(
        cycle_ms_mean=round(statistics.mean(cyc), 2) if cyc else None,
        cycle_ms_max_window=round(max(cyc), 2) if cyc else None,
        missed_deadlines=txt.count("missed its desired rate"),
        optimizer_resets=txt.count("Optimizer reset"),
        fail_to_compute=txt.count("fail to compute path"),
        mode_switches=txt.count("mode switch"),
        spacetime_selections=len(re.findall(r"mode switch -?\d+ -> 10\d\d", txt)),
        spacetime_detected=sum(int(x[0]) for x in st),
        spacetime_appended=sum(int(x[1]) for x in st),
        spacetime_non_distinct=sum(int(x[2]) for x in st),
        corrupt_or_quarantine=txt.count("TGMPPI diag] quarantin") + txt.count("sampled control corrupt"),
        alloc_example=(re.findall(r"TGMPPI alloc\] (.*)", txt) or [None])[0],
    )


def score_bag(bag_dir, legs, world, ctrl_log=None):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    foot, hx, hy = load_footprint()
    radii = world_radii(world)
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    obs_topics = [t for t in types if re.match(r"^/moving_obstacle_\d+/ground_truth/odom$", t)]
    want = [t for t in ("/ground_truth/odom", "/rosout") if t in types] + obs_topics
    r.set_filter(rosbag2_py.StorageFilter(topics=want))
    rob, obs, logs = [], {}, []
    while r.has_next():
        tp, data, tn = r.read_next()
        t = tn * 1e-9
        m = deserialize_message(data, get_message(types[tp]))
        if tp == "/ground_truth/odom":
            q = m.pose.pose.orientation
            rob.append((t, m.pose.pose.position.x, m.pose.pose.position.y, 2 * math.atan2(q.z, q.w)))
        elif tp == "/rosout":
            logs.append((t, m.msg))
        else:
            i = int(re.search(r"obstacle_(\d+)", tp).group(1))
            obs.setdefault(i, []).append((t, m.pose.pose.position.x, m.pose.pose.position.y))
    import bisect
    rk = [x[0] for x in rob]

    def robot_at(t):
        return rob[min(bisect.bisect_left(rk, t), len(rob) - 1)]

    def vel(series, k):
        a_, b_ = max(0, k - 5), min(len(series) - 1, k + 1)
        dt = series[b_][0] - series[a_][0]
        return ((series[b_][1] - series[a_][1]) / dt, (series[b_][2] - series[a_][2]) / dt) if dt > 1e-6 else (0.0, 0.0)

    def clearance(rx, ry, ryaw, ox, oy, orad):
        c, s = math.cos(-ryaw), math.sin(-ryaw)
        px, py = c * (ox - rx) - s * (oy - ry), s * (ox - rx) + c * (oy - ry)
        dx, dy = abs(px) - hx, abs(py) - hy
        return math.hypot(max(dx, 0), max(dy, 0)) + min(max(dx, dy), 0) - orad

    out = dict(obstacle_topics=len(obs_topics), legs=[])
    first_contact = None
    for li, L in enumerate(legs):
        t0, t1 = L["t0_wall"], L["t1_wall"]
        a_, b_ = bisect.bisect_left(rk, t0), bisect.bisect_left(rk, t1)
        seg = rob[a_:b_]
        driven = sum(math.hypot(seg[k][1] - seg[k - 1][1], seg[k][2] - seg[k - 1][2]) for k in range(1, len(seg)))
        sx, sy = L["start_pose"][0], L["start_pose"][1]
        straight = math.hypot(L["goal"][0] - sx, L["goal"][1] - sy)
        dur = L["t1_sim"] - L["t0_sim"]
        minc, contacts = 9.9, []
        for i, series in obs.items():
            last = -9.0
            for k, (t, ox, oy) in enumerate(series):
                if not t0 <= t <= t1:
                    continue
                _, rx, ry, ryaw = robot_at(t)
                c = clearance(rx, ry, ryaw, ox, oy, radii.get(i, 0.25))
                minc = min(minc, c)
                if c < 0.02 and t - last > 0.5:
                    j = min(bisect.bisect_left(rk, t), len(rob) - 1)
                    rvx, rvy = vel(rob, j)
                    ovx, ovy = vel(series, k)
                    nx, ny = ox - rx, oy - ry
                    nn = math.hypot(nx, ny) or 1.0
                    rc = (rvx * nx + rvy * ny) / nn
                    oc = -(ovx * nx + ovy * ny) / nn
                    spd = math.hypot(rvx, rvy)
                    who = ("obstacle_hit_stopped_robot" if spd < 0.05 else "robot_drove_in" if rc > max(oc, 0.05)
                           else "obstacle_ran_in" if oc > 0.05 else "graze")
                    contacts.append(dict(t_rel=round(t - t0, 2), obstacle=i, robot_speed=round(spd, 2), who=who))
                    if first_contact is None or t < first_contact[0]:
                        first_contact = (t, li, i, who)
                if c < 0.02:
                    last = t
        walls = 0
        for (t, rx, ry, ryaw) in seg:
            for p in foot:
                wx = rx + math.cos(ryaw) * p[0] - math.sin(ryaw) * p[1]
                wy = ry + math.sin(ryaw) * p[0] + math.cos(ryaw) * p[1]
                if abs(wx) > ROOM_X - 0.02 or wy < ROOM_Y0 + 0.02 or wy > ROOM_Y1 - 0.02:
                    walls += 1
                    break
        recov = sum(1 for (t, msg) in logs if t0 <= t <= t1 and RECOVERY_RE.search(msg))
        ok = L["status"] == "succeeded"
        out["legs"].append(dict(
            leg=li, status=L["status"], success=int(ok), duration_sim=round(dur, 2),
            straight_m=round(straight, 2), driven_m=round(driven, 2),
            path_ratio=round(driven / straight, 3) if straight > 0.1 else None,
            effective_speed=round(straight / dur, 3) if ok and dur > 0.1 else None,
            min_clearance=round(minc, 3) if minc < 9 else None,
            contacts=contacts, n_contacts=len(contacts), wall_samples=walls, recoveries=recov))
    if first_contact is not None:
        out["first_contact"] = dict(leg=first_contact[1], obstacle=first_contact[2], who=first_contact[3])
    out["controller"] = controller_stats(ctrl_log)
    return out


def legs_from_rosout(bag_dir):
    """For re-scoring bags that were NOT recorded by this tool (manual runs)."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=["/rosout"]))
    legs, cur, last_t = [], None, None
    while r.has_next():
        _, data, tn = r.read_next()
        t = tn * 1e-9
        last_t = t
        msg = deserialize_message(data, get_message(types["/rosout"])).msg
        g = re.search(r"Begin navigating from current location \(([-0-9.]+), ([-0-9.]+)\) to "
                      r"\(([-0-9.]+), ([-0-9.]+)\)", msg)
        if g:
            if cur:
                cur.update(status="preempted", t1_wall=t, t1_sim=t - cur["t0_wall"] + cur["t0_sim"])
                legs.append(cur)
            x0, y0, x1, y1 = map(float, g.groups())
            cur = dict(goal=[x1, y1], start_pose=[x0, y0, 0.0], t0_wall=t, t0_sim=0.0)
        elif cur and ("Goal succeeded" in msg or "Goal failed" in msg or "Goal canceled" in msg):
            cur.update(status="succeeded" if "succeeded" in msg else "aborted", t1_wall=t,
                       t1_sim=t - cur["t0_wall"])
            legs.append(cur)
            cur = None
    if cur:
        cur.update(status="unfinished", t1_wall=last_t, t1_sim=last_t - cur["t0_wall"])
        legs.append(cur)
    return legs


# ---------------------------------------------------------------------------
# matrix + report
# ---------------------------------------------------------------------------

def preflight():
    """Fail in seconds, not after a 90 s Gazebo timeout, if the shell is not set up."""
    missing = []
    for pkg in (GZ_PKG, "susag_nav2", "nav2_tgmppi_controller", "susag_gazebo_plugins"):
        r = subprocess.run(["ros2", "pkg", "prefix", pkg], capture_output=True, text=True)
        if r.returncode != 0:
            missing.append(pkg)
    if missing:
        sys.exit("ROS packages not found: %s\nSource the workspace in THIS terminal first:\n"
                 "  source /opt/ros/humble/setup.bash && source ~/robohouse_ws/install/setup.bash\n"
                 "(from zsh use the setup.zsh files instead)" % ", ".join(missing))
    for world in WORLDS_CHECK:
        if not os.path.exists(os.path.join(WORLD_DIR, f"world_{world}.world")):
            sys.exit(f"world_{world}.world not found in {WORLD_DIR} -- generate it with "
                     f"make_open_world.py --motion polynomial --scenario {world} ...")


WORLDS_CHECK = []


def run_matrix(a):
    WORLDS_CHECK[:] = a.worlds
    preflight()
    if subprocess.run(["pgrep", "-x", "gzserver"], capture_output=True).returncode == 0 and not a.force:
        sys.exit("A Gazebo server is already running. Close your own sim first (trials kill leftover "
                 "Gazebo/Nav2 processes), or pass --force.")
    trials = [(w, c, r) for w in a.worlds for c in a.conditions for r in range(a.reps)]
    print(f"run {a.run_id}: {len(trials)} trials -> {os.path.join(RESULTS_ROOT, a.run_id)}", flush=True)
    for n, (w, c, r) in enumerate(trials, 1):
        tj = os.path.join(RESULTS_ROOT, a.run_id, w, c, f"rep{r}", "trial.json")
        if os.path.exists(tj) and not a.overwrite:
            print(f"[{n}/{len(trials)}] {w} {c} rep{r}: done already, skipping", flush=True)
            continue
        cmd = [sys.executable, os.path.abspath(__file__), "--_trial", c, w, str(r), "--run", a.run_id,
               "--max-speed", str(a.max_speed), "--tracking-quality", a.tracking_quality,
               "--start-sim-time", str(a.start_sim_time), "--legs", str(a.legs),
               "--leg-timeout", str(a.leg_timeout), "--domain", str(a.domain),
               "--max-consecutive-failures", str(a.max_consecutive_failures)]
        print(f"[{n}/{len(trials)}] {w} {c} ({CONDITIONS[c][0]}) rep{r}", flush=True)
        if a.dry_run:
            print("   " + " ".join(cmd))
            continue
        subprocess.run(cmd)
    if not a.dry_run:
        kill_strays()
        report(a.run_id)


def fmt(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "--"
    return f"{statistics.mean(xs):.{nd}f} ± {statistics.stdev(xs):.{nd}f}" if len(xs) > 1 else f"{xs[0]:.{nd}f}"


def report(run_id=None):
    runs = ([d for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
            if os.path.isdir(RESULTS_ROOT) else [])
    if not runs:
        sys.exit(f"no runs under {RESULTS_ROOT}")
    # newest by modification time, not by name (a named run must not shadow dated ones)
    run_id = run_id or max(runs, key=lambda d: os.path.getmtime(os.path.join(RESULTS_ROOT, d)))
    root = os.path.join(RESULTS_ROOT, run_id)
    trials = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(root, "*", "*", "rep*", "trial.json")))]
    rows = []
    for t in trials:
        for L in (t.get("score") or {}).get("legs", []):
            rows.append(dict(world=t["world"], condition=t["condition"], rep=t["rep"],
                             late_start=int(bool(t.get("late_start"))), **{k: v for k, v in L.items() if k != "contacts"}))
    if rows:
        with open(os.path.join(root, "legs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    lines = [f"# Dynamic benchmark {run_id}", "",
             f"{len(trials)} trials, {len(rows)} legs. Effective speed = straight-line distance / time on "
             "successful legs. Contacts use ground truth (footprint vs obstacle discs).", "",
             "| cond | description | trials | legs ok | eff. speed (m/s) | legs w/ contact | "
             "first contact: robot drove in | min clearance (m) | recoveries/leg | wall legs | "
             "cycle ms | missed deadl./trial | late starts |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        ts = [t for t in trials if t["condition"] == c]
        if not ts:
            continue
        lr = [r for r in rows if r["condition"] == c]
        ok = sum(r["success"] for r in lr)
        contact_legs = sum(1 for r in lr if r["n_contacts"])
        drove = sum(1 for t in ts if (t.get("score") or {}).get("first_contact", {}).get("who") == "robot_drove_in")
        fc = sum(1 for t in ts if (t.get("score") or {}).get("first_contact"))
        ctrl = [(t.get("score") or {}).get("controller", {}) for t in ts]
        lines.append("| %s | %s | %d | %d/%d | %s | %d/%d | %d/%d | %s | %s | %d | %s | %s | %d |" % (
            c, CONDITIONS[c][0], len(ts), ok, len(lr),
            fmt([r["effective_speed"] for r in lr]), contact_legs, len(lr), drove, fc,
            fmt([r["min_clearance"] for r in lr], 3), fmt([r["recoveries"] for r in lr]),
            sum(1 for r in lr if r["wall_samples"]),
            fmt([x.get("cycle_ms_mean") for x in ctrl], 1), fmt([x.get("missed_deadlines") for x in ctrl], 1),
            sum(1 for t in ts if t.get("late_start"))))
    revs = sorted({(t.get("git_slip"), t.get("git_sandbox")) for t in trials})
    lines += ["", f"Code revisions (SLIP, sandbox): {revs}"]
    txt = "\n".join(lines) + "\n"
    open(os.path.join(root, "report.md"), "w").write(txt)
    print(txt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conditions", nargs="+", default=["A", "B", "Bp", "D"], choices=list(CONDITIONS))
    p.add_argument("--worlds", nargs="+", default=["dyn1", "dyn2", "dyn3", "dyn4", "dyn5"])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--legs", type=int, default=4, help="goals per trial along A,B,A,B")
    p.add_argument("--max-speed", type=float, default=1.5)
    p.add_argument("--tracking-quality", default="perfect", choices=["perfect", "good", "poor"])
    p.add_argument("--start-sim-time", type=float, default=30.0,
                   help="first goal is sent exactly at this sim time in every trial")
    p.add_argument("--leg-timeout", type=float, default=90.0, help="sim seconds per goal")
    p.add_argument("--max-consecutive-failures", type=int, default=2)
    p.add_argument("--domain", type=int, default=42, help="ROS_DOMAIN_ID used only for benchmark trials")
    p.add_argument("--run", dest="run_id", default=None, help="run folder name (resume by reusing it)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--score-bag", default=None, help="score a manually recorded bag")
    p.add_argument("--world", default="dyn1", help="world for --score-bag")
    p.add_argument("--log", default=None, help="controller log for --score-bag")
    p.add_argument("--_trial", nargs=3, default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a._trial:
        c, w, r = a._trial
        sys.exit(run_trial(c, w, int(r), a))
    if a.report:
        return report(a.run_id)
    if a.score_bag:
        s = score_bag(a.score_bag, legs_from_rosout(a.score_bag), a.world, a.log)
        print(json.dumps(s, indent=2))
        return
    a.run_id = a.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_matrix(a)


if __name__ == "__main__":
    main()

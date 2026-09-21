#!/usr/bin/env python3
"""Build a minimal open-room world (no static clutter) with 3 moving
obstacles that are DESIGNED to cross the robot's straight-line path,
per professor's guidance to decouple space-time topology testing from
BARN's narrow-corridor confound. Reuses the exact plugin blocks already
proven working in world_48_dynamic.world.
"""

WORLD_HEADER = """<sdf version='1.6'>
  <world name='default'>
    <light name='sun' type='directional'>
      <cast_shadows>1</cast_shadows>
      <pose frame=''>0.000000 0.000000 10 0 -0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.5 -1</direction>
    </light>
    <model name='ground_plane'>
      <static>1</static>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <surface>
            <friction>
              <ode>
                <mu>100</mu>
                <mu2>50</mu2>
              </ode>
              <torsional>
                <ode/>
              </torsional>
            </friction>
            <contact>
              <ode/>
            </contact>
            <bounce/>
          </surface>
          <max_contacts>10</max_contacts>
        </collision>
        <visual name='visual'>
          <cast_shadows>0</cast_shadows>
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
          <material>
            <script>
              <uri>file://media/materials/scripts/gazebo.material</uri>
              <name>Gazebo/Grey</name>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <kinematic>0</kinematic>
        <gravity>1</gravity>
      </link>
    </model>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>
    <physics name='default_physics' default='0' type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
"""

WALL_TEMPLATE = """    <model name='{name}'>
      <static>1</static>
      <pose frame=''>{x} {y} 0.5 0 0 {yaw}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry>
            <box><size>{length} 0.1 1.0</size></box>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <box><size>{length} 0.1 1.0</size></box>
          </geometry>
          <material>
            <ambient>0.6 0.6 0.6 1</ambient>
            <diffuse>0.6 0.6 0.6 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
"""

OBSTACLE_TEMPLATE = """    <!-- {comment} -->
    <model name='moving_obstacle_{idx}'>
      <static>false</static>
      <pose frame=''>{cx} {cy} 0.3 0 0 0</pose>
      <link name='link'>
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <collision name='collision'>
          <geometry>
            <cylinder><radius>{radius}</radius><length>0.6</length></cylinder>
          </geometry>
        </collision>
        <visual name='visual'>
          <geometry>
            <cylinder><radius>{radius}</radius><length>0.6</length></cylinder>
          </geometry>
          <material>
            <ambient>{color} 1</ambient>
            <diffuse>{color} 1</diffuse>
          </material>
        </visual>
      </link>
      <plugin name='oscillating_obstacle_plugin' filename='liboscillating_obstacle_plugin.so'>
        <center_x>{cx}</center_x>
        <center_y>{cy}</center_y>
        <z>0.3</z>
{motion_block}
      </plugin>
      <plugin name="moving_obstacle_p3d" filename="libgazebo_ros_p3d.so">
        <ros>
          <namespace>/moving_obstacle_{idx}</namespace>
          <remapping>odom:=ground_truth/odom</remapping>
        </ros>
        <body_name>link</body_name>
        <frame_name>map</frame_name>
        <update_rate>30.0</update_rate>
        <gaussian_noise>0.0</gaussian_noise>
      </plugin>
    </model>
"""

# Interior room is x in [-4,4], y in [-0.5,12.5]; the robot runs roughly
# y=1 -> y=11. Obstacles oscillate along x (axis='x'), so every one sweeps
# across the robot's route once per period. Different omega per obstacle =>
# they never stay in phase => distinct crossing timings along one run.
#
# Obstacle COUNT is a command-line option (2026-09-16). Two rules the
# geometry must respect, both measured from bag analysis, not guessed:
#
#  1. Lane spacing >= 2 * (contact 0.65 m + DynamicObstacleCritic
#     soft_distance) ~= 1.6 m. Tighter lanes leave no cost-free corridor
#     between them and the robot reverses and loops instead of driving
#     (bag tgmppi_dyn_20260915_234156, ~55 s lost to exactly that).
#     So add obstacles PER LANE, not more lanes, once lanes are at 2.0 m.
#  2. Peak speed = |amplitude| * omega must stay under the robot's vx_max,
#     or the obstacle is undodgeable by construction (an earlier pass had
#     one at 0.98 m/s vs vx_max 0.35).
#
# With --per-lane 2 each lane gets a pair placed left and right of centre, the
# right one with a NEGATIVE amplitude (= an exact 180 deg phase shift), so a
# pair opens a gap on one side while closing it on the other. --seed adds a
# random per-obstacle <phase> on top (plugin support added 2026-09-16); without
# it every run replays the identical scene, so repeated runs of an experiment
# are not independent samples.
import argparse   # noqa: E402
import math      # noqa: E402  (kept next to the knobs it drives)
import random    # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--lanes", type=int, default=5, help="y-lanes of obstacles")
parser.add_argument("--per-lane", type=int, default=2, choices=(1, 2),
                    help="obstacles per lane (2 = anti-phased pair)")
parser.add_argument("--y-start", type=float, default=2.5)
parser.add_argument("--y-step", type=float, default=2.0, help="lane spacing (m)")
parser.add_argument("--amplitude", type=float, default=1.3)
parser.add_argument("--cx-offset", type=float, default=1.6,
                    help="pair centres at -offset / +offset (--per-lane 2)")
parser.add_argument("--peak-min", type=float, default=0.13,
                    help="slowest obstacle's peak speed (m/s)")
parser.add_argument("--peak-max", type=float, default=0.30,
                    help="fastest obstacle's peak speed (m/s)")
parser.add_argument("--vx-max", type=float, default=0.35,
                    help="robot vx_max; keep in sync with navigation_tgmppi_tight.yaml")
parser.add_argument("--motion", choices=("oscillate", "polynomial"), default="oscillate",
                    help="oscillate = lanes of sinusoids; polynomial = DynaBARN-style "
                         "random polynomial tracks (their worlds are Gazebo 9 binaries "
                         "with no source, so the METHOD is reproduced, not the files)")
parser.add_argument("--obstacles", type=int, default=10,
                    help="polynomial mode: number of obstacles")
parser.add_argument("--speed-min", type=float, default=0.30, help="polynomial mode (m/s)")
parser.add_argument("--speed-max", type=float, default=1.00, help="polynomial mode (m/s)")
parser.add_argument("--radius-scale", type=float, default=1.0,
                    help="polynomial mode: scale the obstacle cylinder radius (0.25/0.20 m) by this "
                         "factor. Tracks are still fitted with the UNSCALED radius, so the paths are "
                         "identical to the scale-1.0 world with the same seed; only the body shrinks.")
parser.add_argument("--scenario", default=None,
                    help="write world_<name>.world and matching map yaml links, so the "
                         "launch can select it with world_idx:=<name>")
parser.add_argument("--seed", type=int, default=None,
                    help="randomize obstacle phases; omit for all-zero phases "
                         "(every run then replays the identical scene, so repeats "
                         "of an experiment are not independent samples)")
parser.add_argument("--out", default="/home/saran/robohouse_ws/src/BARN_dataset/"
                                     "scaled_1/world_files/world_open_dynamic.world")
args = parser.parse_args()

COLORS = ["0.9 0.2 0.1", "0.1 0.3 0.9", "0.2 0.8 0.3", "0.9 0.6 0.1", "0.6 0.2 0.9",
          "0.1 0.7 0.7", "0.8 0.4 0.6", "0.5 0.5 0.1", "0.3 0.3 0.8", "0.7 0.7 0.2"]
ROOM_HALF_X = 4.0
ROOM_Y_MIN = -0.5         # inner faces of wall_south / wall_north
ROOM_Y_MAX = 12.5
SAFE_MARGIN = 0.05        # keep the swept edge off the wall
MIN_LANE_SPACING = 1.6    # see rule 1 above

rng = random.Random(args.seed) if args.seed is not None else None
problems = []
if args.lanes > 1 and args.y_step < MIN_LANE_SPACING:
    problems.append(f"lane spacing {args.y_step:.2f} m < {MIN_LANE_SPACING} m: "
                    "no cost-free corridor between lanes")

OSC_BLOCK = ("        <amplitude>{amplitude}</amplitude>\n"
             "        <omega>{omega}</omega>\n"
             "        <phase>{phase}</phase>\n"
             "        <axis>x</axis>")
TRACK_BLOCK = ("        <waypoints>{waypoints}</waypoints>\n"
               "        <speed>{speed}</speed>")

# Robot start and the two goals used in every dynamic run: a track that passes
# through them would put an obstacle on top of the robot before it moves.
KEEPOUTS = [(0.0, 1.0), (2.5, 11.3), (-2.8, 0.5)]
KEEPOUT_R = 1.2
# The two legs the robot actually drives; tracks must interact with one of them.
ROUTES = [((0.0, 1.0), (2.5, 11.3)), ((2.5, 11.3), (-2.8, 0.5))]


def polynomial_track(rng, radius):
    """DynaBARN polynomial_fit.py, adapted to this room: random waypoints, a
    polynomial fitted through them, sampled and kept inside the walls."""
    import numpy as np
    x_lo = -ROOM_HALF_X + radius + SAFE_MARGIN
    x_hi = ROOM_HALF_X - radius - SAFE_MARGIN
    y_lo = ROOM_Y_MIN + radius + SAFE_MARGIN
    y_hi = ROOM_Y_MAX - radius - SAFE_MARGIN
    for _ in range(400):
        order = rng.choice([2, 3])
        xs = [rng.uniform(x_lo, x_hi) for _ in range(order + 1)]
        ys = [rng.uniform(y_lo, y_hi) for _ in range(order + 1)]
        if len(set(round(v, 3) for v in xs)) != len(xs):
            continue                      # duplicate x -> polyfit is ill-posed
        poly = np.poly1d(np.polyfit(xs, ys, order))
        sample_x = np.linspace(min(xs), max(xs), 24)
        pts = [(float(a), float(poly(a))) for a in sample_x]
        pts = [(a, b) for a, b in pts if y_lo <= b <= y_hi and x_lo <= a <= x_hi]
        if len(pts) < 6:
            continue
        if any(math.hypot(a - kx, b - ky) < KEEPOUT_R
               for a, b in pts for kx, ky in KEEPOUTS):
            continue
        length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                     for i in range(1, len(pts)))
        if length < 3.0 or length > 40.0:
            continue
        # Resample by ARC LENGTH. Sampling uniformly in x puts waypoints metres
        # apart wherever the polynomial is steep, and the plugin interpolates
        # linearly between them, so the obstacle would cut long straight chords
        # instead of following the curve.
        pts = densify(pts, 0.25)
        # An obstacle that never comes near the route cannot influence the run;
        # DynaBARN's obstacles cross the arena, so require the same here.
        if min(point_segment_distance(pt, a, b) for pt in pts for a, b in ROUTES) > 2.0:
            continue
        return pts, length
    return None, 0.0


def densify(pts, spacing):
    out = [pts[0]]
    for i in range(1, len(pts)):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(seg / spacing))
        for k in range(1, n + 1):
            f = k / float(n)
            out.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0)))
    return out[:400]


def point_segment_distance(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den < 1e-9 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / den))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


obstacles = []
if args.motion == "polynomial":
    if rng is None:
        rng = random.Random(0)
    count = args.obstacles
    for i in range(count):
        pts, length = polynomial_track(rng, 0.25 if i % 2 == 0 else 0.20)
        if pts is None:
            problems.append("could not fit a track for obstacle %d inside the room" % (i + 1))
            continue
        speed = round(rng.uniform(args.speed_min, args.speed_max), 3)
        if speed >= args.vx_max:
            problems.append("obstacle %d speed %.2f >= vx_max %.2f" % (i + 1, speed, args.vx_max))
        radius = round((0.25 if i % 2 == 0 else 0.20) * args.radius_scale, 3)
        obstacles.append(dict(
            idx=i + 1, cx=round(pts[0][0], 3), cy=round(pts[0][1], 3), radius=radius,
            color=COLORS[i % len(COLORS)], phase=0.0, speed=speed,
            motion_block=TRACK_BLOCK.format(
                waypoints=" ".join("%.3f %.3f" % pt for pt in pts), speed=speed),
            comment="DynaBARN-style polynomial track, %d pts, %.1f m long, %.2f m/s"
                    % (len(pts), length, speed)))
else:
  count = args.lanes * args.per_lane
  for lane in range(args.lanes):
    cy = args.y_start + lane * args.y_step
    for k in range(args.per_lane):
        i = len(obstacles)
        peak = args.peak_min if count == 1 else (
            args.peak_min + (args.peak_max - args.peak_min) * i / (count - 1))
        # sign flips per obstacle within a lane: -A == 180 deg phase shift
        amplitude = args.amplitude if k == 0 else -args.amplitude
        cx = 0.0 if args.per_lane == 1 else (-args.cx_offset if k == 0 else args.cx_offset)
        radius = 0.25 if i % 2 == 0 else 0.20
        reach = abs(cx) + abs(amplitude) + radius + SAFE_MARGIN
        if reach > ROOM_HALF_X:
            problems.append(f"obstacle {i + 1} sweeps to {reach:.2f} m > room half-width "
                            f"{ROOM_HALF_X} m")
        if not (ROOM_Y_MIN + radius + SAFE_MARGIN <= cy <= ROOM_Y_MAX - radius - SAFE_MARGIN):
            problems.append(f"obstacle {i + 1} lane y={cy:.2f} is outside the room "
                            f"(y in [{ROOM_Y_MIN}, {ROOM_Y_MAX}]) -- it would spawn "
                            "inside or past a wall")
        if peak >= args.vx_max:
            problems.append(f"obstacle {i + 1} peak speed {peak:.3f} >= vx_max {args.vx_max}")
        phase = round(rng.uniform(0.0, 6.2832), 4) if rng is not None else 0.0
        omega = round(peak / abs(amplitude), 4)
        obstacles.append(dict(
            idx=i + 1, cx=cx, cy=cy, radius=radius, amplitude=amplitude, phase=phase,
            omega=omega, color=COLORS[i % len(COLORS)],
            motion_block=OSC_BLOCK.format(amplitude=amplitude, omega=omega, phase=phase),
            comment=f"lane y={cy:.1f}, sweeps x in "
                    f"[{cx - abs(amplitude):.2f}, {cx + abs(amplitude):.2f}], "
                    f"peak speed {peak:.3f} m/s"))

if problems:
    raise SystemExit("refusing to write an unusable world:\n  - " + "\n  - ".join(problems))

walls = [
    dict(name="wall_south", x=0.0, y=-0.55, yaw=0.0, length=8.1),
    dict(name="wall_north", x=0.0, y=12.55, yaw=0.0, length=8.1),
    dict(name="wall_west", x=-4.05, y=6.0, yaw=1.5708, length=13.1),
    dict(name="wall_east", x=4.05, y=6.0, yaw=1.5708, length=13.1),
]

parts = [WORLD_HEADER]
for w in walls:
    parts.append(WALL_TEMPLATE.format(**w))
for o in obstacles:
    parts.append(OBSTACLE_TEMPLATE.format(**o))
parts.append("  </world>\n</sdf>\n")

import os          # noqa: E402

out_path = args.out
if args.scenario:
    out_path = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "world_%s.world" % args.scenario)
with open(out_path, "w") as f:
    f.write("".join(parts))
if args.motion == "polynomial":
    print("wrote %s: %d obstacles on DynaBARN-style polynomial tracks (seed %s)"
          % (out_path, len(obstacles), args.seed))
else:
    print("wrote %s: %d obstacles in %d lane(s) (%d per lane, %.2f m apart)"
          % (out_path, len(obstacles), args.lanes, args.per_lane, args.y_step))

# The room geometry is identical across scenarios, so every scenario reuses the
# open_dynamic map; the launch resolves yaml_<world_idx>.yaml, hence one link per
# scenario name.
if args.scenario:
    base = os.path.dirname(os.path.dirname(os.path.abspath(args.out)))
    for mdir in ("map_files", "map_files_fine"):
        src = os.path.join(base, mdir, "yaml_open_dynamic.yaml")
        dst = os.path.join(base, mdir, "yaml_%s.yaml" % args.scenario)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink("yaml_open_dynamic.yaml", dst)
            print("  map link: %s -> yaml_open_dynamic.yaml" % dst)
    print("  launch with: world_idx:=%s" % args.scenario)

print(f"\n--- speed check (must stay < vx_max={args.vx_max}) ---")
if args.motion == "polynomial":
    for o in obstacles:
        print('  obstacle %2d: start (%+.2f,%.2f)  speed %.3f m/s  %s'
              % (o["idx"], o["cx"], o["cy"], o["speed"], o["comment"]))
for o in obstacles if args.motion != "polynomial" else []:
    peak = abs(o["amplitude"]) * o["omega"]
    print(f'  obstacle {o["idx"]:>2}: y={o["cy"]:5.1f}  x in '
          f'[{o["cx"] - abs(o["amplitude"]):+.2f}, {o["cx"] + abs(o["amplitude"]):+.2f}]  '
          f'peak {peak:.3f} m/s  omega {o["omega"]:.4f}  phase {o["phase"]:.2f}'
          f'{"  (anti-phase)" if o["amplitude"] < 0 else ""}')

print("\n--- paste this into navigation_tgmppi_tight.yaml's "
      "tgmppi_spacetime_obstacle_topics ---")
print("      tgmppi_spacetime_obstacle_topics:")
for o in obstacles:
    print(f'        - "/moving_obstacle_{o["idx"]}/ground_truth/odom"')

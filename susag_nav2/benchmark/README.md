# TgMppi vs vanilla MPPI benchmark

Paired A/B harness. It drives a fixed start→goal repeatedly, spawning a
different obstacle each scenario, and logs one CSV row per trial. Run it once
per controller (you swap the controller by launching nav2 with a different
params file); both runs append to the same `benchmark_results.csv`.

## 0. Tune the course FIRST (required)

Open `benchmark_tgmppi.py` and set `START` / `GOAL` to a **straight,
obstacle-free corridor** in your map (`maze_map_new`: x ∈ [-8.51, 9.99],
y ∈ [-9.85, 8.15]). The default is (0,0)→(4,0); confirm in RViz that the
global path between them is a clean straight line with free space either side,
otherwise the spawned obstacle isn't the only thing in the way and the result
is meaningless. The obstacles are auto-placed on the midpoint of that segment.

## 1. Vanilla run

```bash
# terminal A – sim + nav2 with the VANILLA params
ros2 launch susag_nav2 navigation.launch.py sim:=true \
     nav2_params:=$(ros2 pkg prefix susag_nav2)/share/susag_nav2/param/navigation_sim.yaml
# In RViz, give a 2D Pose Estimate so AMCL is localized (map frame appears).
```

```bash
# terminal B
cd src/susag_nav2/benchmark
python3 benchmark_tgmppi.py --controller vanilla --reps 10
```

## 2. TgMppi run

Ctrl-C nav2 in terminal A, relaunch with the tgmppi params:

```bash
ros2 launch susag_nav2 navigation.launch.py sim:=true \
     nav2_params:=$(ros2 pkg prefix susag_nav2)/share/susag_nav2/param/navigation_tgmppi.yaml
# 2D Pose Estimate again.
python3 benchmark_tgmppi.py --controller tgmppi --reps 10
```

## 3. Compare

```bash
python3 benchmark_tgmppi.py --report
```

Prints success-rate / time / path / clearance / stall per (scenario × controller).

## What to read

- **Headline = `succ%`.** Expect S0 ≈ tie; S1–S3 tgmppi higher; S4 tgmppi
  recovers where vanilla wanders. `time_s` is secondary (successful runs only)
  and vanilla will often win it on S0 — that's expected, not a loss.
- **`stall_s`** is the mode-averaging proxy: seconds spent creeping (|v|<0.03)
  while still far from goal. Vanilla should stall hard on S1/S3 if the trap bites.
- `min_clear_m` is nearest-anything from /scan (includes maze walls), so read it
  as a safety floor, not obstacle-specific distance.

## Quick single-scenario smoke test

```bash
python3 benchmark_tgmppi.py --controller tgmppi --reps 2 --scenarios S1_pillar
```

## Notes / limits

- Resets by **navigating home** (empty world has no `set_entity_state`), so a
  failed trial that leaves the robot stuck costs one extra home leg.
- No control-rate logging here; check that separately with
  `... | grep -i "missed its desired rate"` on the nav2 launch output.
- Uses the `navigate_to_pose` action and `map`-frame goals, so AMCL must be
  localized before you start (step 1's 2D Pose Estimate).

---

# Dynamic-obstacle benchmark (`benchmark_dynamic.py`, 2026-09-17)

Thesis ablation on the polynomial-track worlds (`world_idx` dyn1..dynN, built by
`BARN_dataset/scaled_1/world_files/make_open_world.py --motion polynomial`). It owns
the whole lifecycle of every trial, so runs are identical in the ways manual runs
cannot be:

- Gazebo restarts per trial, so sim time and every obstacle's position start from zero;
- the first goal is sent at a **fixed sim time** (`--start-sim-time`, default 30 s), so
  every trial meets the same scene (a stack not ready by then is flagged `late_start`);
- a fixed goal sequence (start (0, 1) → A (2.5, 11.3) → B (−2.8, 0.5) → A → B), each goal
  sent only after the previous one finished — nothing is preempted;
- its own `ROS_DOMAIN_ID` (default 42) and `ROS_LOG_DIR` per trial, a bag per trial, and
  the git revisions of both repos recorded in `trial.json`.

| cond | what runs |
|---|---|
| A  | stock Nav2 MPPI (`navigation_sim_tight.yaml`, unmodified) |
| B  | TG-MPPI full, legacy sample split |
| Bp | TG-MPPI full, equal sample split (amoeba_sandbox allocation) |
| C  | TG-MPPI without space-time |
| D  | plain MPPI + predicted-obstacle critic (TG-MPPI with the bias off) |

All TG conditions use `backend:=cpu` (identical shared-critic code to the baseline) and
`dynamic_obstacles:=true`; every condition uses `max_speed` (default 1.5).

## Run

Close any Gazebo / Nav2 you have open first — trials kill leftover processes.

```bash
source /opt/ros/humble/setup.bash && source ~/robohouse_ws/install/setup.bash
cd ~/robohouse_ws/src/susag_nav2/benchmark
python3 benchmark_dynamic.py --dry-run --conditions A B Bp D --worlds dyn1 dyn2 dyn3 --reps 3
python3 benchmark_dynamic.py --conditions A B Bp D --worlds dyn1 dyn2 dyn3 --reps 3
```

A trial takes roughly 2–4 minutes. Interrupted? Re-run with the same `--run <name>`:
finished trials (those with a `trial.json`) are skipped.

## Results

`~/robohouse_ws/benchmark_results/dynamic/<run>/<world>/<cond>/rep<k>/` holds
`trial.json`, `bag/`, `gazebo.log`, `nav2.log`, `ros_log/`. Per run: `legs.csv` (one row per
leg) and `report.md` (per-condition table), regenerated with

```bash
python3 benchmark_dynamic.py --report [--run <name>]
```

Scoring is ground truth: robot footprint vs obstacle discs, and the room walls.
**Effective speed** = straight-line distance / time, on successful legs only (fair
across different start poses). For each contact the tool records who closed the gap
(`robot_drove_in`, `obstacle_ran_in`, `obstacle_hit_stopped_robot`, `graze`). Only the
**first** contact of a trial is trustworthy: the obstacles are position-forced and can
shove the robot unphysically afterwards (seen on 2026-09-17: a sandwiched robot skidded
3 m into a wall with zero commanded velocity).

Re-score a manually recorded bag with the same metrics:

```bash
python3 benchmark_dynamic.py --score-bag ~/robohouse_ws/bags/<bag> --world dyn1 \
    --log ~/.ros/log/component_container_isolated_<pid>_<stamp>.log
```

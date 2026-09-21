# Space-time blob (branch `space-time-blob`)

Status (2026-09-21): Phase 1 and Phase 2 implemented, compiled (syntax-only against the package flags), NOT yet run in Gazebo. Nothing here is merged to `main`
until the verification gates at the bottom pass.

## What the current controller does

* **Static blob** (`FlowField::build`): robot-centred geodesic ball `B_rho` over the local
  costmap's free cells, outer membrane, static water level `D` (distance-to-goal flooded from
  the membrane), up to 3 pseudopods = geodesic-tree paths from the robot to separated
  membrane exits. A pseudopod is a 2-D polyline.
* **Moving obstacles** are NOT part of the blob. They enter through (a) `DynamicObstacleCritic`
  (a per-rollout cost) and (b) `trySpacetimeAlternatives`, a separate wait/detour Dijkstra that
  is triggered by a detected crossing and adds at most 2 extra modes. Measured: inert.

## What "true space-time blob" means here

Lift the body, membrane and pseudopods into (x, y, t):

| static blob                                   | space-time blob                                               |
|-----------------------------------------------|---------------------------------------------------------------|
| body = free cells within geodesic radius rho  | body = free (x,y,t) states reachable from (robot, 0) at grid speed `res/dt`, avoiding static AND predicted-moving obstacles |
| membrane = outer wavefront of the ball        | membrane = reachable states on the last time layer (the light-cone rim) |
| promise on membrane = distance to goal        | promise = `G(exit)` (space-time cost) + `w * D_static(exit_xy)` — the static water level is the terminal cost-to-go |
| pseudopod = geodesic path to a separated exit | pseudopod = min-cost (x,y,t) route to an exit, **distinct by homotopy class** (winding-number signature over all obstacles), then by exit separation |

Why this is principled: the static flood already IS the value function beyond the horizon, so
`G + D_static` is the exact cost-to-go through the space-time body. Pass-before vs pass-behind
routes are different exits/classes of the same body, found in ONE flood over ALL obstacles —
no trigger, no single-crossing budget, no `sideOk` heuristic.

## Algorithm (`SpaceTimeBody::build`)

1. Grid = the light cone: `(2K+1)^2` cells (`K = horizon/dt`), one cell of motion per layer,
   8-neighbourhood + wait. Static free mask computed once, dynamic free per layer
   (`|cell - predicted obstacle(t)| > r_obs + r_robot`).
2. Layered DP (every edge advances one layer, so no priority queue): `G[k+1][n] =
   min_m G[k][m] + step(m->n)`, step = `res*|d|` for a move, `wait_epsilon` for a wait (time is
   priced by the terminal water level not decreasing, not by a tuned wait cost). Parent
   pointers kept for route reconstruction.
3. Membrane = reached cells on layer K. Rank by promise. For each candidate reconstruct the
   route and its signature (per obstacle: sign of winding number if `|lambda| >= 1/(4 pi)`,
   else 0; same rule as `SpaceTimeSearch::routesAreDistinct`).
4. Pick pseudopods: best exit; then best exit of each NEW signature (min exit separation 0.3 m,
   promise within `max_regret` of the best); fill remaining slots by exit separation 0.65 m
   (the static rule). Max 3.

Output routes use the existing `SpaceTimeRoute` (one (x,y) per layer), so they feed the
existing `resample()` -> (v, w, x, y) mode builder unchanged.

## Phases

* **Phase 1** — `SpaceTimeBody` class (ROS-free, unit-tested with plain `g++`), wired in behind
  `tgmppi_spacetime_blob` (default false) as the generator of the space-time modes, replacing
  `twoRouteSearch`. Slot bookkeeping (3 pods + 2 extras) unchanged. Ablation-friendly:
  flag off == current behaviour byte for byte.
* **Phase 2** — promote space-time routes to the 3 pseudopod slots when moving obstacles are
  present (tracking by signature instead of spatial overlap); static blob otherwise.
* **Phase 3** — benchmarks: `dyn1..5` (10 obstacles) and the density worlds `d15_*`, `d20_*`,
  conditions B (current), D (plain prediction), and the new blob.

## As implemented

Switches (launch `spacetime_blob:=`): `false` = wait/detour search (main behaviour);
`true` = Phase 1 (2 extra modes from the blob); `pods` = Phase 2.

Phase 2 (`buildSpacetimePods`, per cycle): flood with all obstacles + the same flood without;
`delay = best promise (with) - best promise (without)`. Space-time pods are active from
`delay >= tgmppi_spacetime_blob_gate` (0.2 m) until `delay < gate/2` (hysteresis). While active the 3
pod slots hold the blob routes (own tracker: time-aligned route overlap, own key range
`kSpacetimePodKeyBase = 100000`), the mode reference is the resampled route instead of a
pure-pursuit polyline, with the same footprint collision check; the wait/detour extras are off.
Inactive = static pseudopods, unchanged. Benchmark conditions: `E` (phase 1), `F` (phase 2).

## Verification gates before merging to `main`

1. Unit tests pass, including a check that every returned route respects static + dynamic free.
2. Flag off: controller behaviour identical to `main` (same build, canary trial matches).
3. Cycle time on `d20_*`: max < 50 ms, missed deadlines not worse than condition B.
4. `dyn1..5` x 3 reps: legs-ok not worse than B (95%); density sweep reported either way.

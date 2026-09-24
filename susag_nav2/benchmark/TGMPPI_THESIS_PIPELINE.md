# TGMPPI thesis evaluation pipeline

Prepared 2026-09-24 from the current controller, launch configuration, benchmark runner, and selected local papers. This is an experimental plan, not a statement of achieved results. Controller implementation is frozen for this plan; configuration changes, scenario generation, recording, and offline analysis remain allowed.

## 1. Research question and contribution

**Research question:** Under matched observations, objectives, dynamics, and rollout budgets, when does topology-guided, mode-selective MPPI improve collision-free navigation, and what computation and prediction limits constrain that improvement?

Candidate contribution statement:

> We develop and evaluate a topology-guided proposal mechanism and a mode-selective MPPI update for SLIP navigation. We characterize their effects on collision-free completion, sample-budget sensitivity, and behavior in dynamic environments, including conditions where the method fails.

Treat this as a claim to test. Do not claim global optimality, guaranteed safety, superiority to all controllers, or novelty of every component. Establish the distinction from prior work through the literature discussion. A negative result can support a useful, carefully delimited characterization, but does not automatically establish an improved algorithm.

Describe the implemented method accurately: proposals shift sampling means; samples receive local exponential weights; a group is selected by free energy with switching stabilizers. The retained Nav2 gamma term is not a general mixture proposal-density correction. Document it as part of the implemented objective and explain the departure from the Biased-MPPI derivation. No new importance-sampling theorem is needed for an empirical evaluation, but the mathematical description must match the code.

## 2. Lessons from the local papers

The following experiment/results sections were reviewed in `/home/saran/sampling efficient MPPI for SLIP/`:

| Paper | Relevant evidence and practice | Adaptation for this thesis |
|---|---|---|
| `3. Biased MPPI.pdf`, Secs. V–VII, Tables I–II | Sweeps sample counts, separates collisions/deadlocks from successful-run travel metrics, and discusses cases where guidance is safer without being faster. | Test useful proposals at several sample budgets. Report the safety–time tradeoff explicitly. |
| `6. Spline-Interpolated MPPI.pdf`, simulation evaluation and Table I | Compares the full method, an ablation without SVGD, and ordinary MPPI at different sample counts; failure includes collisions and getting stuck. | Remove individual TG components to test their contribution. Do not call an indefinitely stopped robot successful. |
| `24. Topology-Driven Parallel Trajectory Optimization in Dynamic.pdf`, Sec. V and Sec. VII | Studies the number of guidance trajectories and consistency behavior; investigates prediction mismatch after observing collisions. | Evaluate guidance and commitment separately. Explain remaining collisions through evidence rather than hiding them or assuming more pods will solve them. |
| `1. LP-MPPI.pdf`, Sec. IV | Gives competing methods a tuning budget and evaluates horizon/sample-count sensitivity and computational overhead. | Separate development tuning from final testing, and distinguish sample efficiency from compute efficiency. |

These are related methodological examples, not numerical benchmarks for SLIP. Their robots, dynamics, objectives, sensors, and scenes differ. A direct superiority claim over one of these methods requires a faithful implementation and matched experiments.

## 3. Freeze the experiment before final trials

Use existing/manual runs as development data. Keep unseen scenario seeds for final tests; do not tune on them after inspecting outcomes. Give the tracked-MPPI baseline a comparable development tuning budget. Use the same robot limits, footprint, tracking input, prediction model, goal, common critics, sample budget, and control frequency in the controlled comparison.

Choose one primary robot speed during development and freeze it. If 1.5 m/s makes every controller fail nearly every trial, a predeclared slower operating point is legitimate; retain 1.5 m/s as a stress test. Do not choose a speed by looking at held-out comparative outcomes. Report every tested development setting separately from the final results.

For each trial save:

- World SDF and map YAML/image, or their hashes with archived copies.
- Source configuration, exact launch command, code revision **and dirty diff**, and built-library identifiers. Git revision alone is insufficient in this workspace.
- Runtime parameter dumps for controller, costmaps, and velocity smoother after all overrides and before the goal.
- Start/goal, obstacle motion seed, goal-send simulation time, repeat ID, timeout, backend, machine, and visualization setting.
- Bag, Nav2/Gazebo logs, and trial outcome, including runs that never produced a scored leg.

Current launch/runner behavior to account for:

1. The default navigation YAML is `navigation_tgmppi_tight.yaml`, not the experiment YAML. Select the file explicitly.
2. `max_speed` rewrites the space-time horizon to 3 s, time layer to 0.20 s, resolution to `0.20 * max_speed`, and several other settings. A YAML horizon of 4 s alone is not effective under that launch. For the unchanged automatic runner, use its effective 3 s as the declared configuration. A 4 s manual experiment requires applying and recording the parameter before the goal on every relevant trial.
3. Condition `Be` forces legacy allocation and CPU, even though its YAML requests equal allocation and CUDA. The experiment YAML is not the complete experiment definition.
4. Condition `D` aliases `A`; it is not an independent algorithm. `A` is plain MPPI with tracking and the added critic/safety handling; it is not unmodified upstream Nav2. `A0` is the costmap-only reference in this workspace.
5. `B`, `C`, and `Be` currently load different configurations or overrides. They are not automatically a clean full/static ablation pair.
6. `flow_max_pseudopods` is capped at five. The blob pod path uses its five-slot capacity directly; changing the static flow pod-count parameter does not provide a clean blob-count ablation.
7. Guidance/group statistics are not all recorded by the current bag topic regex. Add `/tgmppi/.*` and `/tgmppi_debug` to a separate diagnostic recording when needed; keep equivalent recording overhead across compared methods.

This plan does not change controller or runner code. The existing runner cannot select arbitrary new YAML paths through a general CLI option. Use controlled manual launches with copied YAML configurations for the exact arms below, or arrange runner configuration support separately before automating them. Do not silently relabel existing conditions as these arms.

## 4. Define the comparison arms

Create immutable YAML copies from one frozen TG base configuration for the three TG arms. Use matching common settings for plain MPPI. Keep `flow_critic_enabled: false` across the TG arms if it is false in the base. Keep obstacle tracking, the dynamic critic, and safety checks enabled in all primary arms.

| Arm | Configuration | Purpose |
|---|---|---|
| M | Plain Nav2-derived MPPI with shared tracking and prediction critic; matching common settings, visualization off | Practical baseline: does the complete proposal method help? |
| U | Same TG implementation/configuration as F, with `tgmppi_bias_enabled: false` | Controls for differences between controller implementations; unguided update within the same codebase. |
| S | Same as F, with `tgmppi_spacetime_enabled: false`, both blob flags false; static flow proposals and grouped update enabled | Isolates the value of space-time guidance while retaining dynamic-obstacle scoring. |
| F | Frozen full experiment configuration: bias, grouped update, space-time, blob, and blob pods enabled | Full proposed method. |

Optional diagnostic G: copy F, set `tgmppi_grouped_update: false`. For a comparison focused on averaging versus selection, disable mode warm-start in BOTH G and its paired grouped arm; otherwise warm-start behavior changes too. Group commitment/hysteresis are inherently part of the selected-group policy. Label this an ablation of that policy, not a proof that every difference is caused by one softmax operation.

Costmap-only A0 is a useful deployment reference, but it cannot isolate topology's benefit when the main method has obstacle prediction that A0 lacks. A faithful Biased-MPPI baseline is optional unless claiming improvement over that paper. Do not label U or G as a reproduction of Biased-MPPI without checking its objective and sampler against the paper.

For the mechanism comparison, use CPU for all arms, or explicitly disclose unequal backends. If CPU cannot meet the selected control rate, that is a measured implementation limitation. A separate practical deployment comparison can use TG CUDA and MPPI CPU, but it tests the complete system and cannot attribute speed differences solely to guidance. Do not describe the current GPU implementation as a completed end-to-end CUDA pipeline.

## 5. Scenario suite: mechanism, performance, limits

### A. Mechanism demonstrations

Use a small set of interpretable cases, with several preselected obstacle phases for each:

1. Open route: establishes overhead and whether guidance causes unnecessary detours.
2. Static bottleneck or two routes around an obstacle: tests useful geometric alternatives. Select several representative BARN geometries; do not infer dynamic performance from them.
3. Crossing obstacle: tests passing before/after and waiting decisions.
4. Two moving obstacles forming an opening: tests whether guidance proposes and the robot actually executes a feasible passage.

Show robot footprint, obstacle trajectories, proposed paths, executed path, and timestamps. Display both a representative success and a failure. Predeclare how representative cases are selected, such as the first held-out disagreement between M and F, rather than choosing only the most attractive clip.

### B. Controlled dynamic-density sweep

Use one arena, one start/goal pair, one radius/speed distribution, and polynomial tracks at N = 10, 20, 30, 40, 50. Keep world generation seeds paired between methods. Generate new worlds for held-out seeds; the current `d30_1` is an elongated sinusoidal-lane world and must not be inserted into this density curve.

Example generator command (configuration/data generation, not a controller change):

```bash
python3 /home/saran/robohouse_ws/src/BARN_dataset/scaled_1/world_files/make_open_world.py \
  --motion polynomial --obstacles 20 --seed 11 \
  --speed-min 0.30 --speed-max 1.00 --vx-max 1.5 \
  --room-y-max 12.5 --radius-scale 1.0 --scenario thesis_n20_s11
```

Use the same explicit arguments for other counts/seeds, changing only those two fields and the scenario name. Verify map bounds, actual obstacle count, full swept tracks, and start/goal clearance. Record whether obstacle-obstacle intersections occur; these prescribed motions are not an interactive pedestrian model. The same generator seed should preserve earlier obstacle tracks as N increases; verify that property before calling the sweep nested.

Practical first final-test matrix: 5 unseen world seeds × 2 fixed goal-send times (e.g. 30 and 45 simulation seconds) × 5 counts × 4 arms = **200 trials**. Five independent world seeds give limited generalization; report that limitation and uncertainty. If time permits, expand to 10 unseen seeds, giving 400 trials. This is a planning budget, not a statistical power guarantee.

Start with one outbound goal per fresh simulation, equivalent to `--legs 1`. Four sequential goals are correlated, encounter different obstacle phases after different controller outcomes, and can be contaminated by earlier contact. Multi-leg missions can be a separate robustness study after the primary single-goal study.

### C. Sample-budget sweep

Use a preselected moderate density, such as N=20, and the same held-out seed/time pairs. Compare M and F at K = 250, 500, 1000, 2000 total rollout samples, with the same model steps and one optimization iteration. These are candidate test budgets; confirm they are valid and stable during development. Relaunch for each saved configuration.

The K=2000 points already exist in the density matrix. Five seeds × two times × three additional budgets × two arms adds **60 trials**, for 260 primary trials in the smaller design.

Plot collision-free completion against K and separately against measured computation. Fewer rollout samples do not establish lower total compute because flood/blob construction costs extra. More pods also divide a fixed K among more groups; they do not create free additional coverage.

If sample-budget performance does not improve, remove the sample-efficiency claim. An improvement in behavior or robustness may still be supportable with its overhead clearly stated.

### D. Limits and stress tests

Keep current 40/50-obstacle worlds and `d30_1` as individually labeled stress cases, including their differing speed/motion/layout. Report failed cases alongside successful ones. Optional sensitivity: change the effective space-time horizon 3→4 s with all other parameters fixed, or perfect→degraded tracking using existing presets. Do not combine every sensitivity into a huge grid; use one predeclared medium/hard scene subset.

Order of priority if time is short: complete the matched main comparison; add the sample-budget sweep if claiming sampling efficiency; add one mechanism ablation; then optional sensitivities. Estimate runtime from a pilot's observed wall time per trial, including resets, rather than assuming simulation runs at real-time speed.

## 6. Trial procedure and data integrity

For each scenario/time pair, run all arms from a fresh simulation with the same initial robot pose, route, obstacle trajectories, target simulation start time, and timeout. Balance or randomize controller order across pairs to reduce thermal/load/order effects. Use the same visualization, bag recording, and background workload.

A world seed and start time define an environment case. A repeated launch of the same deterministic case is a repeatability check, not automatically a new independent scene. The noise generator uses xtensor sampling, but the runner has no explicit per-repeat sampler-seed argument. Do not claim independent random seeds merely because `--reps` was increased. Record what is controlled and what is not.

If startup misses the target simulation time, flag the run as a protocol deviation. Define ahead of time whether it is rerun; preserve the failed attempt and apply the same rule to every method. Distinguish environment/setup failure from a controller crash after navigation starts. Preserve all records and report exclusion counts and reasons.

At first contact, classify the navigation episode as failed. Position-forced obstacles can shove the robot afterward: later speed, path length, or collisions are not evidence of normal controller behavior. Count the contact regardless of the attribution label; an obstacle hitting a stopped robot is still an unsafe mission outcome. Diagnose controllability only with additional evidence, not that label alone.

## 7. Metrics and current report limitations

Primary outcome per valid planned episode:

**Collision-free completion = goal reached within the fixed timeout, with no obstacle contact/clearance violation and no wall contact before arrival.**

Report the outcome categories: collision-free completion; contact; timeout/stall without contact; controller/navigation failure; setup/protocol invalid. Define precedence so a contacted robot that later reaches the goal is still a failed episode. Keep every valid started episode in the denominator.

Current scoring details that must be disclosed or handled in offline analysis:

- `success` currently means only that the Nav2 action succeeded. Combine it with zero contacts and zero wall samples for the primary outcome; do not copy the current success column directly into the thesis.
- The contact detector uses signed clearance **< 0.02 m**, so its count includes a 2 cm clearance buffer. Label it a contact/clearance violation proxy, or separately analyze penetration below zero. Do not present it as a perfect physical contact sensor.
- Robot/obstacle pose alignment is approximate. Inspect near-threshold examples and report the sampling/alignment limitation. Keep frame/time conventions consistent in offline analysis.
- The report's speed metrics include action-successful runs even if they had contacts. Filter to collision-free successes. Show per-method success rates alongside time metrics and, for a paired speed comparison, report the subset completed safely by both methods with its sample count.
- The dynamic scorer assumes room bounds x=±4, y=−0.5…12.5. Its wall scores are invalid for the extended `d30_1` without world-specific offline correction. Use the standard room for the primary density suite.
- A trial without a scored leg can disappear from the report's leg denominator. Reconcile against a manifest of all intended/started trials and classify missing data explicitly.
- TG cycle logs contain averages over multiple cycles and stop timing before final smoothing/safety processing. They are not per-cycle end-to-end latencies. Do not compute a purported p95 latency from these averages; report the available statistic accurately. Deadline-warning counts are useful but are not a full latency distribution, and a missing timing metric for plain MPPI is not zero cost.

Secondary metrics: safe-success travel time, traveled distance/path ratio, minimum footprint clearance up to first contact, stopped time while the goal remains distant, recoveries, mode-switch behavior, observed computation statistics, and deadline warnings. Proposal-selection counts are diagnostic and require checking whether the relevant existing topics/logs actually expose them.

Report counts and proportions by obstacle density and by method. Show paired differences on identical cases. For confidence intervals use an analysis appropriate to clustered world seeds (for example, bootstrap entire world-seed blocks, keeping phases/methods paired); avoid treating every odometry sample or repeated identical run as independent. With only five seed clusters, emphasize descriptive results and wide uncertainty. Predeclare one primary endpoint to avoid searching many metrics for a favorable result.

## 8. Failure analysis without controller changes

For each representative collision examine the interval leading to first contact:

| Candidate explanation | Evidence to seek |
|---|---|
| Observation missing/stale | Topic coverage, message timestamps, tracking fault logs, transform availability |
| Prediction mismatch | Compare positions predicted from the last observed velocity with subsequent ground-truth positions at 0.5, 1, and 2 s using a consistent time base |
| No useful proposal | Available ancillary paths and their relationship to the opening, obstacle paths, and executed controls |
| Commitment or switching problem | Group-selection logs and repeated reversals/delayed mode changes |
| Computation/control delay | Deadline warnings, command timestamps, and timing data actually available |
| Tracking/dynamics mismatch | Commanded versus measured robot motion before contact |
| Very constrained encounter | Geometric/temporal reconstruction; do not call it unavoidable without testing feasible alternatives |

An offline reconstruction supports an explanation; it is not a causal proof that fixing that one factor would prevent the collision. Count both the frequency and uncertainty of failure categories, allowing multiple contributing factors.

## 9. Figures, tables, and conclusion structure

Prepare these outputs:

1. Architecture figure: obstacle observations + costmap/global path → flow/blob proposals → grouped sampling → scoring → mode selection → executed control. Label the shared prediction/safety components.
2. Paired trajectory panels for the interpretable scenes, including one failure case.
3. Collision-free completion versus density, with counts/uncertainty; separate contact and timeout outcomes.
4. Collision-free completion versus rollout budget; pair it with available compute/deadline data.
5. Ablation table M/U/S/F: safe completion, safe-success time, stalls, contact count, and timing limitations.
6. Failure analysis for the high-density boundary, including prediction error examples.

Write results in the order: tested claim → observed effect with counts/uncertainty → mechanism evidence → cost/tradeoff → limits. Example templates, to fill only with actual measurements:

> At a fixed budget of K rollouts, F completed X/N matched episodes without contact compared with Y/N for M. The improvement was concentrated in [scenario class]; it was [absent/reversed/uncertain] in [other class].

> Removing space-time guidance changed collision-free completion by [difference], supporting/weakening the hypothesis that temporal proposals explain the observed benefit.

> Comparable completion was reached with [budgets], while computation changed by [measured statistic]. This supports sample efficiency but does/does not demonstrate compute efficiency.

> At 40–50 obstacles, completion deteriorated. Recorded failures were associated with [evidence], so the demonstrated benefit is limited to [tested operating conditions].

If everything ties within uncertainty, say the experiment did not establish a benefit at the tested budgets. If TG is safer but slower, report the tradeoff. If the static proposal arm matches the full arm, the added blob has not earned a performance claim in those tests. A contribution does not require winning every comparison; its actual advantage and operating limits must be evidenced.

## 10. Immediate next session

1. Treat the recently edited YAML as a development candidate, not the final tuned configuration. Several parameters changed together, so those trials cannot isolate which change helped.
2. Run a short pilot on existing development worlds to verify goals, contact scoring, parameter overrides, topic coverage, and timing.
3. Choose/freeze the operating speed, effective horizon, allocation, margins, and backend. Prepare M/U/S/F copies and archive effective runtime parameters.
4. Generate/validate the held-out matched worlds and create the trial manifest before running them.
5. Run the main matrix, reconcile every trial, then run the sample-budget additions if resources allow.
6. Produce the figures and write the limitations together with the positive findings.

No compilation is required by this plan. Do not start the final matrix until its arms and effective parameters are verified. Do not use the old benchmark README condition descriptions as the source of truth: several describe an earlier version of the runner.

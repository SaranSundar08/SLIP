// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_TGMPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
#define NAV2_TGMPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

#include <cstddef>
#include <string>
#include <vector>
#include "nav2_tgmppi_controller/models/constraints.hpp"

namespace tgmppi::models
{

/**
 * @struct tgmppi::models::OptimizerSettings
 * @brief Settings for the optimizer to use
 */
struct OptimizerSettings
{
  models::ControlConstraints base_constraints{0, 0, 0, 0};
  models::ControlConstraints constraints{0, 0, 0, 0};
  models::SamplingStd sampling_std{0, 0, 0};
  float model_dt{0};
  float temperature{0};
  float gamma{0};
  unsigned int batch_size{0};
  unsigned int time_steps{0};
  unsigned int iteration_count{0};
  bool shift_control_sequence{false};
  size_t retry_attempt_limit{0};

  // "cpu" (xtensor, default -- always available) or "cuda" (LibTorch GPU
  // rollout, only exists if built with -DTGMPPI_WITH_CUDA=ON; silently
  // falls back to "cpu" if requested but unavailable at build or run time).
  std::string compute_backend{"cpu"};

  // --- TgMppi shape-conditioned sampling (this controller's core behavior) ---
  float tgmppi_bias_strength{0.6f};   // fraction of the batch seeded onto the wrap modes
  float tgmppi_bias_gain{1.5f};       // heading P-gain: tangent bearing error -> yaw rate
  float tgmppi_lookahead_dist{0.6f};  // path lookahead for the "toward-goal" bearing (m)
  float tgmppi_goal_dist{1.0f};       // within this range of the goal, disable the tgmppi
                                      // Let plain MPPI + GoalCritic dock near the goal.
  // Floor on the heading-error speed factor (cos(err), clamped) used when
  // building each pseudopod's ancillary reference (v,w) sequence -- 0.15
  // is the long-standing default (a big turn slows the reference to 15% of
  // cruise speed, never fully stopping it). Ported from the amoeba_sandbox
  // finding that this floor being LOW makes different pseudopods' references
  // barely distinguishable within the MPPI horizon (see amoeba_sandbox
  // docs/experiments/REFERENCE_DISTINCTNESS.md, "nominal_fb"): raising it
  // toward 1.0 removes heading-based slowdown, making each pseudopod commit
  // to its own direction faster and diverge from the others sooner. Unlike
  // the sandbox's branch_to_control_sequence(), this reference step has no
  // curvature- or clearance-based slowdown term to also disable -- there
  // wasn't one here to begin with, so raising this floor is the whole of
  // this controller's nominal_fb analog.
  float tgmppi_reference_min_speed_ratio{0.15f};
  // When true (sandbox's reference_infeasible_fallback): if a pseudopod's
  // ancillary rollout built with tgmppi_reference_min_speed_ratio turns out
  // invalid (mode_valid[m]==false, e.g. it collides), that ONE pseudopod is
  // regenerated using the safe 0.15 floor instead before being scored, same
  // as the sandbox falling back to its "shaped" reference per-branch.
  // Meaningless (never triggers) when tgmppi_reference_min_speed_ratio is
  // already <= 0.15.
  bool tgmppi_reference_infeasible_fallback{false};

  // amoeba_sandbox grouped_sampling.py port (2026-09-13), V3 ("uncorrected
  // within-mode") only -- V4's importance-mixture correction + ESS guard is
  // deliberately NOT ported yet, a declared future step, not an oversight.
  // Default false reproduces the existing single-shared-softmax update
  // exactly. When true: each pseudopod's row block (plus the "wait" block
  // and the remaining unbiased rows) gets its OWN local softmax over just
  // its own rows instead of one softmax over the whole batch, each group's
  // free energy is compared, and the single lowest-free-energy group's
  // local weighted mean becomes the new control sequence -- the sandbox's
  // core fix for samples random-walking across modes instead of committing
  // to one. No cross-cycle hysteresis (sandbox's select_mode() switch_margin)
  // is implemented: that needs stable pseudopod identity across reflood
  // cycles, which is itself an explicitly unsolved, deferred sandbox
  // problem ("topological branch re-ID", see thesis-october-green-light-
  // plan). Only meaningful when tgmppi_bias_enabled=true and
  // tgmppi_shadow_mode=false -- silently behaves exactly like the default
  // otherwise, since there's only one group (the whole batch) in that case.
  bool tgmppi_grouped_update{false};

  // 2026-09-15 stabilizer port. The port had the sandbox's PROPOSAL half
  // (pseudopod ancillary modes) but none of what makes the sandbox stick to
  // a branch instead of re-deciding every cycle. Each knob mirrors one
  // sandbox mechanism; every default below reproduces the pre-port
  // behaviour bit-for-bit, navigation_tgmppi_tight.yaml turns them on with
  // the frozen sandbox values (v7_benchmark.py amoeba config).
  //  - pseudopods.py BranchTracker: greedy identity matching of this
  //    reflood's pseudopods against the previous ones (0.5*mean resampled
  //    centerline distance + 0.5*endpoint distance <= match_distance).
  //    A tracked pseudopod keeps its ancillary SLOT (row block, published
  //    path index, group key) across refloods instead of being re-sorted
  //    by promise every rebuild.
  bool tgmppi_pod_tracking{false};
  float tgmppi_pod_match_distance{1.0f};
  //  - mppi.py _select_committed_mode(): the grouped update keeps the
  //    currently selected group for at least min_dwell cycles, and only
  //    switches to a challenger that beats it by switch_margin (free-energy
  //    units) for confirm_cycles consecutive cycles. Defaults (0 / 1 / 0)
  //    reduce to "best group every cycle" = today's V3 behaviour.
  //    NOTE: the sandbox's 0.3 margin is in ITS cost units (lam=0.3); Nav2
  //    critic costs are on a different scale -- calibrate from the
  //    "[TGMPPI] mode switch" debug log before trusting a value.
  int tgmppi_mode_min_dwell{0};
  int tgmppi_mode_confirm_cycles{1};
  float tgmppi_mode_switch_margin{0.0f};
  //  - mppi.py mode_warm_start: each pseudopod's ancillary reference is
  //    blended with that SAME mode's local weighted mean from the previous
  //    cycle (shifted one step, sandbox mode_nominals), so a mode's sampling
  //    centre moves continuously instead of jumping to a fresh reference.
  //    Needs tgmppi_grouped_update (that's where per-group means exist) and
  //    stable keys (tgmppi_pod_tracking) to mean anything.
  float tgmppi_mode_warm_start{0.0f};
  //  - AmoebaHybrid gate_rate: the seeded fraction ramps in at this rate
  //    on entering ASSIST (level += rate*(target-level)) instead of
  //    switching on at full strength in one cycle. 1.0 = instant = today.
  float tgmppi_assist_ramp_rate{1.0f};
  //  - AmoebaHybrid bias_deadband: pseudopod-reference heading errors
  //    smaller than this (rad) are treated as noise and produce no yaw
  //    command, so a few degrees of grid jitter can't flip w's sign.
  float tgmppi_bias_deadband{0.0f};
  // Floor of a pseudopod reference's cruise speed (m/s, clamped to vx_max):
  // cruise = max(this, |current nominal vx|). 0.18 = the original hard-coded
  // floor. With the floor below vx_max, a mode slowed once (e.g. by an
  // obstacle) keeps referencing its own low speed -- and tgmppi_mode_warm_start
  // blends it with its own slow previous mean -- so it creeps (bag
  // tgmppi_dyn_20260916_001024: mode 0 mean 0.15-0.22 m/s vs fallback 0.32-0.36).
  float tgmppi_pod_cruise_speed{0.18f};
  // How the batch is split between guided groups and the unguided fallback
  // (2026-09-17). "legacy": only tgmppi_bias_strength of the batch is recentred on
  // the pseudopod / space-time / wait groups and the rest (80% at 0.2) forms one
  // large fallback group. A group's free energy is bounded below by its minimum
  // cost, whose expectation falls with group size, so that big group wins the
  // grouped selection by sample count alone (about 0.9 sigma at 1600 vs 80 rows).
  // "equal": the amoeba_sandbox rule (grouped_sampling.py allocate_group_counts) --
  // every group, the fallback included, gets a near-equal share, and
  // tgmppi_bias_strength is not used. Only the assist ramp scales the guided part.
  std::string tgmppi_group_allocation{"legacy"};

  // amoeba_sandbox spacetime.py Phase 1 port (2026-09-13): opt-in extra
  // sampling modes ("wait"/"detour") built from a time-expanded (x,y,t)
  // search against a real moving obstacle, alongside the ordinary
  // pseudopod modes. Default false: zero behavior change, no subscription
  // to any obstacle topic, tgmppi_spacetime_obstacle_topics ignored.
  bool tgmppi_spacetime_enabled{false};
  // Ground-truth nav_msgs/Odometry topics (position + twist), one per
  // moving obstacle -- e.g. susag_gazebo_plugins' OscillatingObstaclePlugin
  // + libgazebo_ros_p3d on the same model, matching amoeba_sandbox's own
  // ground-truth-oracle shortcut for predicted_moving_obs() (see
  // moving_obstacle_driver.py's docstring history / PROJECT_STATUS.md,
  // 2026-09-13): this project doesn't have or need real obstacle-tracking
  // perception to port the search algorithm faithfully.
  std::vector<std::string> tgmppi_spacetime_obstacle_topics{};
  float tgmppi_spacetime_obstacle_radius{0.25f};    // matches the test world's cylinder
  float tgmppi_spacetime_horizon{3.0f};             // search horizon (s), sandbox default
  float tgmppi_spacetime_dt_layer{0.25f};           // search time-layer spacing (s)
  float tgmppi_spacetime_res{0.10f};                // search grid resolution (m)
  float tgmppi_spacetime_window{2.5f};              // search grid lateral room (m)
  float tgmppi_spacetime_relevance{0.0f};           // crossing-trigger margin (m); 0 =
                                                     // only a genuine predicted collision
  // Space-time blob (branch space-time-blob, docs/space_time_blob_design.md). false = the
  // trigger-based wait/detour search above, unchanged. true = the geodesic blob lifted into
  // (x, y, t): one flood over ALL predicted obstacles, routes to distinct-homotopy exits
  // of its membrane become the space-time modes. Read once at configure.
  bool tgmppi_spacetime_blob{false};
  // Phase 2 (implies the blob): while moving obstacles matter (gate above, with hysteresis) the
  // 3 pseudopod slots are the blob's own (x, y, t) routes instead of the static pseudopods,
  // rebuilt every cycle and tracked by time-aligned route overlap; otherwise the static
  // pseudopods are used unchanged. No wait/detour extras in this mode.
  bool tgmppi_spacetime_blob_pods{false};
  float tgmppi_spacetime_blob_robot_radius{0.50f};  // m; clearance radius the blob keeps from moving
                                                     // obstacles (must satisfy DynamicObstacleCritic's
                                                     // two-disc geometry, see space_time_body.hpp)
  float tgmppi_spacetime_blob_gate{0.2f};           // m; extras only when the moving obstacles make the
                                                     // best space-time route at least this much more
                                                     // expensive than in an obstacle-free flood (the
                                                     // blob's analogue of a genuine predicted crossing)
  float tgmppi_spacetime_blob_max_regret{1.0f};     // m; an alternative homotopy class is kept
                                                     // only if within this of the best route
  bool tgmppi_debug{false};           // publish /tgmppi_debug markers (scan rays + wrap arrows)
  bool tgmppi_ancillary_debug{false};   // publish up to 3 lightweight Path candidates

  // Observe and visualize tgmppi state without changing MPPI samples or costs.
  // This is the safe first Nav2 integration phase and defaults to enabled.
  bool tgmppi_shadow_mode{true};
  // Independent Phase 3 authority gates. Shadow mode remains the master
  // override: when true, neither sampling nor critic scoring may affect MPPI.
  bool tgmppi_bias_enabled{false};
  bool flow_critic_enabled{false};
  float tgmppi_body_radius{2.5f};
  bool tgmppi_debug_grid{false};
  float tgmppi_debug_arrow_spacing{0.40f};
  float tgmppi_debug_arrow_length{0.18f};
  float tgmppi_debug_arrow_width{0.015f};

  // --- TG-MPPI flow field: the water field over the local costmap ----------
  int flow_reflood_every{1};          // rebuild the field every N control cycles
  bool flow_path_seed{true};          // window-boundary promise from the plan
                                      // (dist-to-plan + remaining length) vs
                                      // straight-line-to-plan-end (planner-free)
  float flow_viscosity{1.5f};         // water physics: flood resistance near
                                      // walls (0 = pure geodesic); wide
                                      // channels flow freely, pinches drag
  float flow_promise_temperature{0.75f};  // softmax temperature (m) used to
                                          // allocate samples among pseudopods
  // Cap on spatially-distinct pseudopod exits kept per flood (2026-09-22).
  // Legacy/sandbox default 3. Each accepted pseudopod is its own MPPI sample
  // group + ancillary rollout + collision check, so raising this trades
  // real-time headroom for more simultaneously-representable route
  // alternatives -- not yet validated above 3 in a real run.
  int flow_max_pseudopods{3};
  bool ancillary_collision_check{true};   // validate ancillary mean rollouts with the robot footprint
  int ancillary_collision_stride{1};      // footprint-check every Nth horizon pose (1 = every pose)
  bool flow_assist_only_when_path_blocked{true};  // preserve the global path in NORMAL mode
  float flow_path_check_distance{1.5f};           // local plan distance inspected by the gate (m)
  float flow_path_blocked_ratio{0.07f};           // enter ASSIST above this invalid-point ratio
  int flow_clear_confirm_cycles{3};               // clear cycles required before leaving ASSIST
  bool flow_wait_enabled{true};                   // offer an explicit zero-control sample population
  float flow_wait_fraction{0.20f};                // fraction of the biased budget reserved for waiting
  float flow_rejoin_lateral_weight{3.0f};         // penalize pseudopods far from the global path
  float flow_rejoin_remaining_weight{2.0f};       // penalize modes that lose path progress
};

}  // namespace tgmppi::models

#endif  // NAV2_TGMPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

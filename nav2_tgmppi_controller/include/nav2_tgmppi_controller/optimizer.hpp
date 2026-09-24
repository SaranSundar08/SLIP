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

#ifndef NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_
#define NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_

#include <array>
#include <limits>
#include <map>
#include <string>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <chrono>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "rclcpp_lifecycle/lifecycle_node.hpp"

#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/footprint_collision_checker.hpp"
#include "nav2_core/goal_checker.hpp"

#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "nav2_tgmppi_controller/models/optimizer_settings.hpp"
#include "nav2_tgmppi_controller/motion_models.hpp"
#include "nav2_tgmppi_controller/critic_manager.hpp"
#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"
#include "nav2_tgmppi_controller/tools/flow_field.hpp"
#include "nav2_tgmppi_controller/tools/space_time_body.hpp"
#include "nav2_tgmppi_controller/tools/space_time_search.hpp"
#include "nav2_tgmppi_controller/tools/noise_generator.hpp"
#include "nav2_tgmppi_controller/tools/parameters_handler.hpp"
#include "nav2_tgmppi_controller/tools/utils.hpp"
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#endif

#ifdef __APPLE__
  #include "nav2_tgmppi_controller/tools/apple_utils.hpp"
#endif

namespace tgmppi
{

/**
 * @class tgmppi::Optimizer
 * @brief Main algorithm optimizer of the MPPI Controller
 */
class Optimizer
{
public:
  /**
    * @brief Constructor for tgmppi::Optimizer
    */
  Optimizer() = default;

  /**
   * @brief Destructor for tgmppi::Optimizer
   */
  ~Optimizer() {shutdown();}


  /**
   * @brief Initializes optimizer on startup
   * @param parent WeakPtr to node
   * @param name Name of plugin
   * @param costmap_ros Costmap2DROS object of environment
   * @param dynamic_parameter_handler Parameter handler object
   */
  void initialize(
    rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
    ParametersHandler * dynamic_parameters_handler);

  /**
   * @brief Shutdown for optimizer at process end
   */
  void shutdown();

  /**
   * @brief Compute control using MPPI algorithm
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   * @return TwistStamped of the MPPI control
   */
  geometry_msgs::msg::TwistStamped evalControl(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed, const nav_msgs::msg::Path & plan,
    nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Get the trajectories generated in a cycle for visualization
   * @return Set of trajectories evaluated in cycle
   */
  models::Trajectories & getGeneratedTrajectories();

  /**
   * @brief Get the optimal trajectory for a cycle for visualization
   * @return Optimal trajectory
   */
  xt::xtensor<float, 2> getOptimizedTrajectory();

  /**
   * @brief Set the maximum speed based on the speed limits callback
   * @param speed_limit Limit of the speed for use
   * @param percentage Whether the speed limit is absolute or relative
   */
  void setSpeedLimit(double speed_limit, bool percentage);

  /**
   * @brief Reset the optimization problem to initial conditions
   */
  void reset();

protected:
  /**
   * @brief Main function to generate, score, and return trajectories
   */
  void optimize();

  /**
   * @brief Prepare state information on new request for trajectory rollouts
   * @param robot_pose Pose of the robot at given time
   * @param robot_speed Speed of the robot at given time
   * @param plan Path plan to track
   * @param goal_checker Object to check if goal is completed
   */
  void prepare(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const geometry_msgs::msg::Twist & robot_speed,
    const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker);

  /**
   * @brief Obtain the main controller's parameters
   */
  void getParams();

  /**
   * @brief Set the motion model of the vehicle platform
   * @param model Model string to use
   */
  void setMotionModel(const std::string & model);

  /**
   * @brief Shift the optimal control sequence after processing for
   * next iterations initial conditions after execution
   */
  void shiftControlSequence();

  /**
   * @brief updates generated trajectories with noised trajectories
   * from the last cycle's optimal control
   */
  void generateNoisedTrajectories();

  /**
   * @brief Flow-mode bias: re-flood the water field
   * from the local costmap (every flow_reflood_every cycles), roll the
   * nominal control sequence out once, and at each horizon point shift the
   * sampling mean's yaw rate toward the field's downhill direction — the
   * whole horizon bends into the water channel, not just the first step,
   * and not just line-of-sight like the ray scan. The nominal
   * control_sequence_ is untouched so the gamma control-cost term supplies
   * the importance correction, same as the ray mode.
   */
  void applyFlowBias();

  /** @brief Forget the previous sampling iteration's ancillary row allocation. */
  void clearAncillaryModeState();

  /** @brief Allocate validated proposals and shift their rows' sampling means. */
  void allocateModeSamples(
    const std::vector<std::vector<float>> & mode_v,
    const std::vector<std::vector<float>> & mode_w,
    const std::vector<bool> & mode_valid,
    const std::vector<float> & promises, float fraction);

  /** @brief Add a constant global-path rejoin prior to each pseudopod row group. */
  void applyTgMppiModePriors();

  /** @brief True when the upcoming transformed global path is obstructed in
   * the local costmap. Used to keep TgMppi dormant during normal tracking. */
  bool isLocalPathBlocked() const;

  /**
   * @brief Publish the water field as RViz markers on /tgmppi_debug:
   * downhill direction arrows on a coarse subsample of wet cells plus a
   * state label with the re-flood time.
   */
  void publishFlowDebug();

  /**
   * @brief Publish up to three geodesic pseudopod candidates as lightweight
   * nav_msgs/Path topics, independently of dense MarkerArray visualization.
   */
  void publishAncillaryPaths();

  /**
   * @brief Publish the dynamically rolled-out, collision-validated ancillary
   * means. Empty paths clear rejected or inactive modes in RViz.
   */
  void publishAncillaryRollouts();

  /**
   * @brief Apply hard vehicle constraints on control sequence
   */
  void applyControlSequenceConstraints();

  /**
   * @brief  Update velocities in state
   * @param state fill state with velocities on each step
   */
  void updateStateVelocities(models::State & state) const;

  /**
   * @brief  Update initial velocity in state
   * @param state fill state
   */
  void updateInitialStateVelocities(models::State & state) const;

  /**
   * @brief predict velocities in state using model
   * for time horizon equal to timesteps
   * @param state fill state
   */
  void propagateStateVelocitiesFromInitials(models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    models::Trajectories & trajectories,
    const models::State & state) const;

  /**
   * @brief Rollout velocities in state to poses
   * @param trajectories to rollout
   * @param state fill state
   */
  void integrateStateVelocities(
    xt::xtensor<float, 2> & trajectories,
    const xt::xtensor<float, 2> & state) const;

  /**
   * @brief Update control sequence with state controls weighted by costs
   * using softmax function
   */
  void updateControlSequence();

  /** @brief Reject dynamically colliding rows; return true when none is safe. */
  bool enforceDynamicCollisionSafety();

  /** @brief Check the final smoothed sequence with DynamicObstacleCritic geometry. */
  bool finalSequenceIsDynamicallySafe();

  /** @brief Clear the command and its smoother/mode state after a safety veto. */
  void stopForDynamicSafety(const char * reason);

  /**
   * @brief Convert control sequence to a twist commant
   * @param stamp Timestamp to use
   * @return TwistStamped of command to send to robot base
   */
  geometry_msgs::msg::TwistStamped
  getControlFromSequenceAsTwist(const builtin_interfaces::msg::Time & stamp);

  /**
   * @brief Whether the motion model is holonomic
   * @return Bool if holonomic to populate `y` axis of state
   */
  bool isHolonomic() const;

  /**
   * @brief Using control frequence and time step size, determine if trajectory
   * offset should be used to populate initial state of the next cycle
   */
  void setOffset(double controller_frequency);

  /**
   * @brief Perform fallback behavior to try to recover from a set of trajectories in collision
   * @param fail Whether the system failed to recover from
   */
  bool fallback(bool fail);

protected:
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D * costmap_;
  std::string name_;

  std::shared_ptr<MotionModel> motion_model_;

  ParametersHandler * parameters_handler_;
  CriticManager critic_manager_;
  NoiseGenerator noise_generator_;

  models::OptimizerSettings settings_;

  models::State state_;
  models::ControlSequence control_sequence_;
  // Flow mode: the water field over the local costmap + re-flood cycle count.
  FlowField flow_field_;
  unsigned int flow_cycle_{0};
#ifdef TGMPPI_WITH_CUDA
  // compute_backend:"cuda" -- see generateNoisedTrajectories(). Absent
  // entirely (not just inert) from a default build; no LibTorch dependency
  // unless built with -DTGMPPI_WITH_CUDA=ON.
  GpuBatch gpu_batch_;
#endif
  rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>::SharedPtr
    tgmppi_debug_pub_;
  // Five tracked pseudopods plus up to two space-time alternatives.
  // Keep row bookkeeping and publishers large enough for every proposal.
  static constexpr std::size_t kPodSlots = 5;
  static constexpr std::size_t kMaxAncillaryModes = kPodSlots + 2;
  std::array<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr,
    kMaxAncillaryModes> ancillary_path_pubs_;
  std::array<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>::SharedPtr,
    kMaxAncillaryModes> ancillary_rollout_pubs_;
  std::array<std::vector<float>, kMaxAncillaryModes> ancillary_rollout_x_;
  std::array<std::vector<float>, kMaxAncillaryModes> ancillary_rollout_y_;
  std::array<bool, kMaxAncillaryModes> ancillary_mode_valid_{};
  std::array<unsigned int, kMaxAncillaryModes> ancillary_mode_samples_{};
  std::array<unsigned int, kMaxAncillaryModes> ancillary_mode_row_start_{};
  std::array<float, kMaxAncillaryModes> ancillary_mode_rejoin_prior_{};
  bool tgmppi_assist_active_{false};
  bool flow_path_blocked_now_{false};
  unsigned int flow_clear_cycles_{0u};
  unsigned int flow_wait_samples_{0u};

  // --- 2026-09-15 stabilizer port (see tgmppi_pod_tracking & friends in
  // optimizer_settings.hpp). Group keys: pseudopod slots carry the tracked
  // pseudopod id (or the slot index when tracking is off), spacetime extras
  // kSpacetimeKeyBase+slot, the wait block kWaitModeKey, the unbiased
  // remainder kFallbackModeKey (sandbox key -1 = path fallback).
  // 2026-09-22: raised from 3 to test flow_max_pseudopods > 3 in Gazebo.
  // pod_slot_ids_/st_slot_ids_ are reset via .fill(-1) in reset() (never a
  // hardcoded {-1,-1,-1}-style literal) so this scales safely on its own --
  // see reset() and the two new_slots assignments in trackPseudopods() /
  // trackSpacetimeRoutes()-equivalent for the sentinel-fill pattern this
  // constant depends on.
  static constexpr int kNoModeKey = std::numeric_limits<int>::min();
  static constexpr int kFallbackModeKey = -1;
  static constexpr int kWaitModeKey = -2;
  static constexpr int kSpacetimeKeyBase = 1000;
  struct TrackedPod
  {
    int id;
    std::vector<std::pair<float, float>> centerline;
    float promise;
  };
  std::vector<TrackedPod> tracked_pods_prev_;
  // Sentinel-filled in reset(), not here -- a hardcoded {-1,-1,-1} literal
  // silently zero-pads instead of erroring when kPodSlots != 3.
  std::array<int, kPodSlots> pod_slot_ids_{};
  int next_pod_id_{0};
  // Slot-ordered pseudopods actually used for ancillary modes + publishing:
  // == flow_field_.pseudopods() when tracking is off; with tracking on, a
  // fixed kPodSlots-long vector where an empty polyline means "no pseudopod
  // in this slot this reflood" (its mode stays invalid, its path publishes
  // empty), so a surviving pseudopod never changes slot.
  std::vector<std::vector<std::pair<float, float>>> display_pods_;
  std::vector<float> display_promises_;
  std::array<int, kMaxAncillaryModes> ancillary_mode_key_{};
  std::map<int, std::pair<std::vector<float>, std::vector<float>>> mode_nominals_;
  int selected_mode_key_{kNoModeKey};
  unsigned int selected_mode_age_{0u};
  int pending_mode_key_{kNoModeKey};
  unsigned int pending_mode_count_{0u};
  unsigned int mode_switch_count_{0u};
  float assist_level_{0.0f};
  // Runs right after every flow_field_.build(): BranchTracker matching +
  // slot assignment, fills display_pods_/display_promises_/pod_slot_ids_.
  void trackPseudopods();

  // amoeba_sandbox spacetime.py Phase 1 port: ground-truth moving-obstacle
  // subscriptions (see tgmppi_spacetime_obstacle_topics's docstring in
  // optimizer_settings.hpp) and the latest state read from them.
  // spacetime_obstacles_mutex_ guards spacetime_obstacles_ since callbacks
  // may run on a different thread than evalControl() depending on the
  // node's executor/callback-group configuration -- cheap correctness
  // insurance, this isn't a hot path.
  std::vector<rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr>
  spacetime_obstacle_subs_;
  std::mutex spacetime_obstacles_mutex_;
  std::vector<SpaceTimeObstacle> spacetime_obstacles_;
  std::vector<rclcpp::Time> spacetime_obstacle_stamps_;
  // Whether each configured obstacle topic has delivered at least one message.
  // Before that its entry is not a real obstacle and must not be used.
  std::vector<bool> spacetime_obstacle_received_;
  // Fresh observations only, projected to the current cycle time under the
  // mutex in prepare(); shared by space-time search and DynamicObstacleCritic.
  std::vector<SpaceTimeObstacle> tracked_obstacles_snapshot_;
  bool obstacle_tracking_fault_{false};
  void snapshotTrackedObstacles();
  void snapshotTrackedObstaclesAt(const rclcpp::Time & now);
  void spacetimeObstacleCallback(std::size_t index, const nav_msgs::msg::Odometry & msg);
  // Checks each valid pseudopod for a predicted moving-obstacle crossing
  // and, if found, appends up to 2 extra modes (wait/detour) into
  // mode_v/w/x/y/valid and promises_local at indices mode_count and
  // mode_count+1 -- called from applyFlowBias() after the ordinary
  // pseudopod loop, before row allocation.
  void trySpacetimeAlternatives(
    const std::vector<std::vector<std::pair<float, float>>> & pods, float rx, float ry,
    std::vector<std::vector<float>> & mode_v, std::vector<std::vector<float>> & mode_w,
    std::vector<std::vector<float>> & mode_x, std::vector<std::vector<float>> & mode_y,
    std::vector<bool> & mode_valid, std::vector<float> & promises_local);
  // Resample a (x, y)-per-dt_layer space-time route onto this controller's (time_steps,
  // model_dt) grid as a (v, w, x, y) mode; false if it needs more than wz_max for >25% of the
  // horizon (an unfollowable reference). Shared by the wait/detour search and the blob.
  bool resampleSpaceTimeRoute(
    const SpaceTimeRoute & route, float rx, float ry,
    std::vector<float> & v, std::vector<float> & w,
    std::vector<float> & x, std::vector<float> & y);
  // tgmppi_spacetime_blob: the space-time blob (SpaceTimeBody) as the generator of the extra
  // modes, in place of the crossing-triggered wait/detour search. Same slot contract as
  // trySpacetimeAlternatives(): appends at most 2 modes past mode_count.
  void trySpacetimeBlob(
    const std::vector<std::vector<std::pair<float, float>>> & pods, float rx, float ry,
    std::vector<std::vector<float>> & mode_v, std::vector<std::vector<float>> & mode_w,
    std::vector<std::vector<float>> & mode_x, std::vector<std::vector<float>> & mode_y,
    std::vector<bool> & mode_valid, std::vector<float> & promises_local);
  SpaceTimeBody spacetime_body_;
  // ---- Phase 2: blob routes as the pseudopod slots (tgmppi_spacetime_blob_pods) ----
  // Own tracker + key range: static pseudopods are rebuilt every reflood_every cycles, these
  // every cycle (they depend on where the obstacles are NOW), and the two id spaces must not mix.
  static constexpr int kSpacetimePodKeyBase = 100000;
  struct StTrackedPod
  {
    int id;
    SpaceTimeRoute route;
    float promise;
  };
  std::vector<StTrackedPod> st_tracked_prev_;
  // Sentinel-filled in reset(), not here -- see pod_slot_ids_ above.
  std::array<int, kPodSlots> st_slot_ids_{};
  int next_st_id_{0};
  unsigned int st_unfollowable_count_{0u};   // routes rejected as unfollowable (log counter)
  bool st_pods_active_{false};   // gate state with hysteresis, persists across cycles
  // Slot-ordered, like display_pods_: st_display_pods_[m] is slot m's (x, y) polyline (empty =
  // no route in this slot), st_display_routes_[m] the same route with its layer timing.
  std::vector<std::vector<std::pair<float, float>>> st_display_pods_;
  std::vector<float> st_display_promises_;
  std::vector<SpaceTimeRoute> st_display_routes_;
  // Builds this cycle's space-time pods; true if they replace the static pseudopods this cycle.
  bool buildSpacetimePods(float rx, float ry);
  SpaceTimeBody spacetime_body_free_;   // same flood without obstacles: the gate's reference
  nav2_costmap_2d::FootprintCollisionChecker<nav2_costmap_2d::Costmap2D *>
  ancillary_collision_checker_{nullptr};
  std::array<tgmppi::models::Control, 4> control_history_;
  models::Trajectories generated_trajectories_;
  // Reused one-rollout buffers for the post-smoothing dynamic collision veto.
  // Allocated only in reset(), never in the controller hot path.
  models::State final_safety_state_;
  models::Trajectories final_safety_trajectory_;
  models::Path path_;
  xt::xtensor<float, 1> costs_;
  std::vector<uint8_t> dynamic_collision_rows_;

  CriticData critics_data_ =
  {state_, generated_trajectories_, path_, costs_, settings_.model_dt, false, nullptr, nullptr,
    std::nullopt, std::nullopt, nullptr, settings_.compute_backend,
    nullptr
    , nullptr, &dynamic_collision_rows_, std::nullopt
  };  /// Caution, keep references

  rclcpp::Logger logger_{rclcpp::get_logger("TgMppiController")};

  // Real end-to-end cycle timing (2026-09-11): logs a rolling average of
  // evalControl()'s prepare()+optimize() wall time every kCycleLogEvery
  // calls, tagged with compute_backend, so a live Nav2 run gives a real
  // number for the actual full cycle (noise gen + tgmppi bias + rollout +
  // every critic + softmax) instead of a synthetic component slice --
  // see PROJECT_STATUS.md 2026-09-11's closing entry on why every number
  // up to this point was a benchmark, not the real thing.
  static constexpr unsigned int kCycleLogEvery = 50;
  unsigned int cycle_log_count_{0};
  double cycle_log_sum_ms_{0.0};
};

template<typename E>
inline auto cumsum_1d(const E & expression)
{
  #ifdef __APPLE__
  return utils::manual_cumsum_1d(expression);
  #else
  return xt::cumsum(expression, 0);
  #endif
}

template<typename E>
inline auto cumsum_2d(const E & expression, int axis)
{
 #ifdef __APPLE__
  return utils::manual_cumsum_2d(expression, axis);
 #else
  return xt::cumsum(expression, axis);
 #endif
}

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__OPTIMIZER_HPP_

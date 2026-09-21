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

#include "nav2_tgmppi_controller/optimizer.hpp"

#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <utility>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_costmap_2d/cost_values.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace tgmppi
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

namespace
{
// Point at arc-length `s` along a polyline, clamped to the final vertex
// beyond the polyline's own length -- the C++ equivalent of
// amoeba_sandbox proposals.py/spacetime.py's `_interpolate`/`_arc_length`
// helpers, reimplemented locally (small, self-contained) rather than
// pulled into utils.hpp for one caller.
std::pair<float, float> pointAtArcLength(
  const std::vector<std::pair<float, float>> & polyline, float s)
{
  if (polyline.empty()) {return {0.0f, 0.0f};}
  if (polyline.size() == 1 || s <= 0.0f) {return polyline.front();}
  float remaining = s;
  for (std::size_t k = 1; k < polyline.size(); ++k) {
    const float dx = polyline[k].first - polyline[k - 1].first;
    const float dy = polyline[k].second - polyline[k - 1].second;
    const float seg_len = std::sqrt(dx * dx + dy * dy);
    if (remaining <= seg_len || seg_len < 1e-9f) {
      const float frac = seg_len < 1e-9f ? 0.0f : remaining / seg_len;
      return {
        polyline[k - 1].first + frac * dx,
        polyline[k - 1].second + frac * dy};
    }
    remaining -= seg_len;
  }
  return polyline.back();
}
}  // namespace

void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  ancillary_collision_checker_.setCostmap(costmap_);
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();

  // Debug-viz publisher. Activated here (not via the wrapper's on_activate) so
  // it is ready before the first control cycle; it only emits while the
  // controller is active, since computeTgMppiModes() runs only from evalControl.
  tgmppi_debug_pub_ =
    node->create_publisher<visualization_msgs::msg::MarkerArray>("/tgmppi_debug", 1);
  tgmppi_debug_pub_->on_activate();
  for (std::size_t i = 0; i < ancillary_path_pubs_.size(); ++i) {
    ancillary_path_pubs_[i] = node->create_publisher<nav_msgs::msg::Path>(
      "/tgmppi/ancillary_path_" + std::to_string(i + 1), 1);
    ancillary_path_pubs_[i]->on_activate();
    ancillary_rollout_pubs_[i] = node->create_publisher<nav_msgs::msg::Path>(
      "/tgmppi/ancillary_rollout_" + std::to_string(i + 1), 1);
    ancillary_rollout_pubs_[i]->on_activate();
  }

  // One ground-truth Odometry subscription per configured obstacle topic, used by
  // the space-time search AND DynamicObstacleCritic (2026-09-15: subscribed whenever
  // topics are configured, not only with tgmppi_spacetime_enabled). No-op when the
  // list is empty (the default). Entries are value-initialised and only used once
  // their topic has delivered a message -- previously resize() left them
  // uninitialised and the space-time search could read garbage obstacles.
  if (!settings_.tgmppi_spacetime_obstacle_topics.empty()) {
    spacetime_obstacles_.assign(settings_.tgmppi_spacetime_obstacle_topics.size(), SpaceTimeObstacle{});
    spacetime_obstacle_received_.assign(settings_.tgmppi_spacetime_obstacle_topics.size(), false);
    for (std::size_t i = 0; i < settings_.tgmppi_spacetime_obstacle_topics.size(); ++i) {
      spacetime_obstacle_subs_.push_back(
        node->create_subscription<nav_msgs::msg::Odometry>(
          settings_.tgmppi_spacetime_obstacle_topics[i], rclcpp::SensorDataQoS(),
          [this, i](const nav_msgs::msg::Odometry & msg) {spacetimeObstacleCallback(i, msg);}));
    }
  }

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

  reset();
}

void Optimizer::shutdown()
{
  noise_generator_.shutdown();
}

void Optimizer::getParams()
{
  std::string motion_model_name;

  auto & s = settings_;
  auto getParam = parameters_handler_->getParamGetter(name_);
  auto getParentParam = parameters_handler_->getParamGetter("");
  getParam(s.model_dt, "model_dt", 0.05f);
  getParam(s.time_steps, "time_steps", 56);
  getParam(s.batch_size, "batch_size", 1000);
  getParam(s.iteration_count, "iteration_count", 1);
  getParam(s.temperature, "temperature", 0.3f);
  getParam(s.gamma, "gamma", 0.015f);
  getParam(s.base_constraints.vx_max, "vx_max", 0.5);
  getParam(s.base_constraints.vx_min, "vx_min", -0.35);
  getParam(s.base_constraints.vy, "vy_max", 0.5);
  getParam(s.base_constraints.wz, "wz_max", 1.9);
  getParam(s.sampling_std.vx, "vx_std", 0.2);
  getParam(s.sampling_std.vy, "vy_std", 0.2);
  getParam(s.sampling_std.wz, "wz_std", 0.4);
  getParam(s.retry_attempt_limit, "retry_attempt_limit", 1);

  getParam(s.tgmppi_bias_strength, "tgmppi_bias_strength", 0.6f);
  getParam(s.tgmppi_bias_gain, "tgmppi_bias_gain", 1.5f);
  getParam(s.tgmppi_lookahead_dist, "tgmppi_lookahead_dist", 0.6f);
  getParam(s.tgmppi_goal_dist, "tgmppi_goal_dist", 1.0f);
  getParam(s.tgmppi_reference_min_speed_ratio, "tgmppi_reference_min_speed_ratio", 0.15f);
  getParam(
    s.tgmppi_reference_infeasible_fallback, "tgmppi_reference_infeasible_fallback", false);
  getParam(s.tgmppi_grouped_update, "tgmppi_grouped_update", false);
  getParam(s.tgmppi_pod_tracking, "tgmppi_pod_tracking", false);
  getParam(s.tgmppi_pod_match_distance, "tgmppi_pod_match_distance", 1.0f);
  getParam(s.tgmppi_mode_min_dwell, "tgmppi_mode_min_dwell", 0);
  getParam(s.tgmppi_mode_confirm_cycles, "tgmppi_mode_confirm_cycles", 1);
  getParam(s.tgmppi_mode_switch_margin, "tgmppi_mode_switch_margin", 0.0f);
  getParam(s.tgmppi_mode_warm_start, "tgmppi_mode_warm_start", 0.0f);
  getParam(s.tgmppi_assist_ramp_rate, "tgmppi_assist_ramp_rate", 1.0f);
  getParam(s.tgmppi_bias_deadband, "tgmppi_bias_deadband", 0.0f);
  getParam(s.tgmppi_pod_cruise_speed, "tgmppi_pod_cruise_speed", 0.18f);
  getParam(s.tgmppi_group_allocation, "tgmppi_group_allocation", std::string("legacy"));
  if (s.tgmppi_group_allocation != "legacy" && s.tgmppi_group_allocation != "equal") {
    RCLCPP_WARN(
      logger_, "[TG-MPPI] tgmppi_group_allocation '%s' not recognized, using 'legacy'",
      s.tgmppi_group_allocation.c_str());
    s.tgmppi_group_allocation = "legacy";
  }
  getParam(s.tgmppi_spacetime_enabled, "tgmppi_spacetime_enabled", false);
  getParam(
    s.tgmppi_spacetime_obstacle_topics, "tgmppi_spacetime_obstacle_topics",
    std::vector<std::string>{});
  getParam(s.tgmppi_spacetime_obstacle_radius, "tgmppi_spacetime_obstacle_radius", 0.25f);
  getParam(s.tgmppi_spacetime_horizon, "tgmppi_spacetime_horizon", 3.0f);
  getParam(s.tgmppi_spacetime_dt_layer, "tgmppi_spacetime_dt_layer", 0.25f);
  getParam(s.tgmppi_spacetime_res, "tgmppi_spacetime_res", 0.10f);
  getParam(s.tgmppi_spacetime_window, "tgmppi_spacetime_window", 2.5f);
  getParam(s.tgmppi_spacetime_relevance, "tgmppi_spacetime_relevance", 0.0f);
  getParam(s.tgmppi_spacetime_blob, "tgmppi_spacetime_blob", false);
  getParam(s.tgmppi_spacetime_blob_max_regret, "tgmppi_spacetime_blob_max_regret", 1.0f);
  getParam(s.tgmppi_debug, "tgmppi_debug", false);
  getParam(s.tgmppi_ancillary_debug, "tgmppi_ancillary_debug", false);
  getParam(s.tgmppi_shadow_mode, "tgmppi_shadow_mode", true);
  getParam(s.tgmppi_bias_enabled, "tgmppi_bias_enabled", false);
  getParam(s.flow_critic_enabled, "flow_critic_enabled", false);
  getParam(s.tgmppi_body_radius, "tgmppi_body_radius", 2.5f);
  getParam(s.tgmppi_debug_grid, "tgmppi_debug_grid", false);
  getParam(s.tgmppi_debug_arrow_spacing, "tgmppi_debug_arrow_spacing", 0.40f);
  getParam(s.tgmppi_debug_arrow_length, "tgmppi_debug_arrow_length", 0.18f);
  getParam(s.tgmppi_debug_arrow_width, "tgmppi_debug_arrow_width", 0.015f);
  getParam(s.flow_reflood_every, "flow_reflood_every", 1);
  getParam(s.flow_path_seed, "flow_path_seed", true);
  getParam(s.flow_viscosity, "flow_viscosity", 1.5f);
  getParam(s.flow_promise_temperature, "flow_promise_temperature", 0.75f);
  getParam(s.ancillary_collision_check, "ancillary_collision_check", true);
  getParam(s.ancillary_collision_stride, "ancillary_collision_stride", 1);
  getParam(
    s.flow_assist_only_when_path_blocked, "flow_assist_only_when_path_blocked", true);
  getParam(s.flow_path_check_distance, "flow_path_check_distance", 1.5f);
  getParam(s.flow_path_blocked_ratio, "flow_path_blocked_ratio", 0.07f);
  getParam(s.flow_clear_confirm_cycles, "flow_clear_confirm_cycles", 3);
  getParam(s.flow_wait_enabled, "flow_wait_enabled", true);
  getParam(s.flow_wait_fraction, "flow_wait_fraction", 0.20f);
  getParam(s.flow_rejoin_lateral_weight, "flow_rejoin_lateral_weight", 3.0f);
  getParam(s.flow_rejoin_remaining_weight, "flow_rejoin_remaining_weight", 2.0f);

  getParam(s.compute_backend, "compute_backend", std::string("cpu"));
  if (s.compute_backend != "cpu" && s.compute_backend != "cuda") {
    RCLCPP_WARN(
      logger_, "[TG-MPPI] compute_backend '%s' not recognized, using 'cpu'",
      s.compute_backend.c_str());
    s.compute_backend = "cpu";
  }
#ifndef TGMPPI_WITH_CUDA
  if (s.compute_backend == "cuda") {
    RCLCPP_WARN(
      logger_,
      "[TG-MPPI] compute_backend:'cuda' requested but this build has no "
      "LibTorch/CUDA support (built without -DTGMPPI_WITH_CUDA=ON) -- "
      "falling back to 'cpu'");
    s.compute_backend = "cpu";
  }
#endif
  RCLCPP_INFO(
    logger_,
    "[TG-MPPI] flow mode: water field over local costmap (reflood every %d "
    "cycle(s), %s-seeded membrane, radius=%.2fm, bias=%.2f gain=%.2f, "
    "shadow=%s, sampling=%s, critic=%s, ancillary collision check=%s)",
    s.flow_reflood_every, s.flow_path_seed ? "plan" : "euclidean",
    s.tgmppi_body_radius, s.tgmppi_bias_strength, s.tgmppi_bias_gain,
    s.tgmppi_shadow_mode ? "true" : "false",
    s.tgmppi_bias_enabled ? "enabled" : "disabled",
    s.flow_critic_enabled ? "enabled" : "disabled",
    s.ancillary_collision_check ? "enabled" : "disabled");

  getParam(motion_model_name, "motion_model", std::string("DiffDrive"));

  s.constraints = s.base_constraints;
  setMotionModel(motion_model_name);
  parameters_handler_->addPostCallback([this]() {reset();});

  double controller_frequency;
  getParentParam(controller_frequency, "controller_frequency", 0.0, ParameterType::Static);
  setOffset(controller_frequency);
}

void Optimizer::setOffset(double controller_frequency)
{
  const double controller_period = 1.0 / controller_frequency;
  constexpr double eps = 1e-6;

  if ((controller_period + eps) < settings_.model_dt) {
    RCLCPP_WARN(
      logger_,
      "Controller period is less then model dt, consider setting it equal");
  } else if (abs(controller_period - settings_.model_dt) < eps) {
    RCLCPP_INFO(
      logger_,
      "Controller period is equal to model dt. Control sequence "
      "shifting is ON");
    settings_.shift_control_sequence = true;
  } else {
    throw std::runtime_error(
            "Controller period more then model dt, set it equal to model dt");
  }
}

void Optimizer::reset()
{
  state_.reset(settings_.batch_size, settings_.time_steps);
  control_sequence_.reset(settings_.time_steps);
  control_history_[0] = {0.0, 0.0, 0.0};
  control_history_[1] = {0.0, 0.0, 0.0};
  control_history_[2] = {0.0, 0.0, 0.0};
  control_history_[3] = {0.0, 0.0, 0.0};

  settings_.constraints = settings_.base_constraints;

  // 2026-09-15 stabilizer state: a new plan/goal starts with no tracked
  // pseudopods, no committed group, no per-mode memory, assist fully off.
  tracked_pods_prev_.clear();
  pod_slot_ids_.fill(-1);
  display_pods_.clear();
  display_promises_.clear();
  ancillary_mode_key_.fill(kNoModeKey);
  mode_nominals_.clear();
  selected_mode_key_ = kNoModeKey;
  selected_mode_age_ = 0u;
  pending_mode_key_ = kNoModeKey;
  pending_mode_count_ = 0u;
  mode_switch_count_ = 0u;
  assist_level_ = 0.0f;

  costs_ = xt::zeros<float>({settings_.batch_size});
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());

#ifdef TGMPPI_WITH_CUDA
  if (settings_.compute_backend == "cuda") {
    gpu_rollout_.initialize(settings_.batch_size, settings_.time_steps);
    if (!gpu_rollout_.ready()) {
      RCLCPP_WARN(
        logger_,
        "[TG-MPPI] compute_backend:'cuda' requested but no CUDA device is "
        "available at runtime -- falling back to 'cpu' for this session");
      settings_.compute_backend = "cpu";
    }
  }
#endif
  RCLCPP_INFO(
    logger_, "Optimizer reset (compute_backend=%s)", settings_.compute_backend.c_str());
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  const auto cycle_t0 = std::chrono::steady_clock::now();

  prepare(robot_pose, robot_speed, plan, goal_checker);

  do {
    optimize();
  } while (fallback(critics_data_.fail_flag));

  cycle_log_sum_ms_ += std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - cycle_t0).count();
  if (++cycle_log_count_ >= kCycleLogEvery) {
    RCLCPP_INFO(
      logger_, "[TG-MPPI] cycle time (compute_backend=%s): %.3f ms/cycle avg over %u cycles",
      settings_.compute_backend.c_str(), cycle_log_sum_ms_ / cycle_log_count_, cycle_log_count_);
    cycle_log_count_ = 0;
    cycle_log_sum_ms_ = 0.0;
  }

  const auto sequence_finite = [this]() {
      for (unsigned int t = 0; t < settings_.time_steps; ++t) {
        if (utils::isBadFloat(control_sequence_.vx(t)) ||
          utils::isBadFloat(control_sequence_.wz(t)) ||
          (isHolonomic() && utils::isBadFloat(control_sequence_.vy(t))))
        {
          return false;
        }
      }
      return true;
    };
  const bool finite_before_smoothing = sequence_finite();
  utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);

  // 2026-09-16 fail-safe: a non-finite control sequence (or smoother history)
  // publishes NaN -- velocity_smoother rejects it, the robot freezes -- and
  // re-seeds every later cycle with NaN (bag tgmppi_dyn_20260916_002534).
  // Restart from rest instead.
  {
    static unsigned int nan_logs = 0u;
    bool finite = sequence_finite();
    for (const auto & h : control_history_) {
      finite = finite && !utils::isBadFloat(h.vx) && !utils::isBadFloat(h.vy) &&
        !utils::isBadFloat(h.wz);
    }
    if (!finite) {
      if (nan_logs++ < 20u) {
        RCLCPP_ERROR(
          logger_,
          "[TGMPPI diag] non-finite control sequence (%s smoothing) -- resetting to rest",
          finite_before_smoothing ? "introduced by" : "already before");
      }
      control_sequence_.reset(settings_.time_steps);
      for (auto & h : control_history_) {
        h = {0.0, 0.0, 0.0};
      }
      mode_nominals_.clear();
    }
  }
  auto control = getControlFromSequenceAsTwist(plan.header.stamp);

  if (settings_.shift_control_sequence) {
    shiftControlSequence();
  }

  return control;
}

void Optimizer::optimize()
{
  const auto implausible = [](float c) {
      return utils::isBadFloat(c) || c < -1.0e3f || c > 1.0e12f;
    };
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    generateNoisedTrajectories();

    // 2026-09-16 diagnostics (first offenders, capped): sampled controls after
    // the TG-MPPI bias and the rollouts, before any critic scores them.
    {
      static unsigned int sample_logs = 0u;
      const auto & s = settings_;
      const float px = static_cast<float>(state_.pose.pose.position.x);
      const float py = static_cast<float>(state_.pose.pose.position.y);
      for (unsigned int r = 0; r < s.batch_size && sample_logs < 10u; ++r) {
        for (unsigned int t = 0; t < s.time_steps; ++t) {
          const float v = state_.cvx(r, t);
          const float w = state_.cwz(r, t);
          const float x = generated_trajectories_.x(r, t);
          const float y = generated_trajectories_.y(r, t);
          const bool bad_control = utils::isBadFloat(v) || utils::isBadFloat(w) ||
            std::fabs(v) > 10.0f || std::fabs(w) > 50.0f;
          const bool bad_rollout = utils::isBadFloat(x) || utils::isBadFloat(y) ||
            std::fabs(x - px) > 50.0f || std::fabs(y - py) > 50.0f;
          if (!bad_control && !bad_rollout) {continue;}
          int slot = -1;
          for (std::size_t m = 0; m < ancillary_mode_samples_.size(); ++m) {
            if (ancillary_mode_valid_[m] && r >= ancillary_mode_row_start_[m] &&
              r < ancillary_mode_row_start_[m] + ancillary_mode_samples_[m])
            {
              slot = static_cast<int>(m);
              break;
            }
          }
          ++sample_logs;
          RCLCPP_WARN(
            logger_,
            "[TGMPPI diag] %s corrupt: row %u t %u cvx %g cwz %g x %g y %g "
            "(u_vx %g u_wz %g, pod slot %d, assist %d)",
            bad_control ? "sampled control" : "rollout", r, t, v, w, x, y,
            control_sequence_.vx(t), control_sequence_.wz(t), slot,
            tgmppi_assist_active_ ? 1 : 0);
          break;
        }
      }
    }

    critic_manager_.evalTrajectoriesScores(critics_data_);

    static unsigned int prior_logs = 0u;
    std::vector<uint8_t> corrupt_after_critics;
    if (prior_logs < 10u) {
      corrupt_after_critics.resize(settings_.batch_size);
      for (unsigned int r = 0; r < settings_.batch_size; ++r) {
        corrupt_after_critics[r] = implausible(costs_(r)) ? 1u : 0u;
      }
    }
    applyTgMppiModePriors();
    for (unsigned int r = 0; r < corrupt_after_critics.size(); ++r) {
      if (corrupt_after_critics[r] || !implausible(costs_(r))) {continue;}
      std::string priors;
      for (std::size_t m = 0; m < ancillary_mode_rejoin_prior_.size(); ++m) {
        priors += std::to_string(m) + ":" + std::to_string(ancillary_mode_rejoin_prior_[m]) +
          "[" + std::to_string(ancillary_mode_row_start_[m]) + "+" +
          std::to_string(ancillary_mode_samples_[m]) + "] ";
      }
      ++prior_logs;
      RCLCPP_WARN(
        logger_, "[TGMPPI diag] rejoin prior made row %u corrupt: cost %g; priors %s",
        r, costs_(r), priors.c_str());
      break;
    }

    updateControlSequence();
  }
}

bool Optimizer::fallback(bool fail)
{
  static size_t counter = 0;

  if (!fail) {
    counter = 0;
    return false;
  }

  reset();

  if (++counter > settings_.retry_attempt_limit) {
    counter = 0;
    throw std::runtime_error("Optimizer fail to compute path");
  }

  return true;
}

void Optimizer::prepare(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  state_.pose = robot_pose;
  state_.speed = robot_speed;
  path_ = utils::toTensor(plan);
  costs_.fill(0);

  critics_data_.fail_flag = false;
  critics_data_.goal_checker = goal_checker;
  critics_data_.motion_model = motion_model_;
  critics_data_.furthest_reached_path_point.reset();
  critics_data_.path_pts_valid.reset();
  critics_data_.flow_field =
    (settings_.tgmppi_shadow_mode || !settings_.flow_critic_enabled) ?
    nullptr : &flow_field_;
  snapshotTrackedObstacles();
  critics_data_.tracked_obstacles =
    tracked_obstacles_snapshot_.empty() ? nullptr : &tracked_obstacles_snapshot_;
}

void Optimizer::shiftControlSequence()
{
  using namespace xt::placeholders;  // NOLINT
  control_sequence_.vx = xt::roll(control_sequence_.vx, -1);
  control_sequence_.wz = xt::roll(control_sequence_.wz, -1);


  xt::view(control_sequence_.vx, -1) =
    xt::view(control_sequence_.vx, -2);

  xt::view(control_sequence_.wz, -1) =
    xt::view(control_sequence_.wz, -2);


  if (isHolonomic()) {
    control_sequence_.vy = xt::roll(control_sequence_.vy, -1);
    xt::view(control_sequence_.vy, -1) =
      xt::view(control_sequence_.vy, -2);
  }
}

void Optimizer::generateNoisedTrajectories()
{
  noise_generator_.setNoisedControls(state_, control_sequence_);
  noise_generator_.generateNextNoises();
  if (settings_.tgmppi_shadow_mode || !settings_.tgmppi_bias_enabled) {
    const auto & s = settings_;
    const bool rebuild = flow_cycle_++ %
      static_cast<unsigned int>(std::max(1, s.flow_reflood_every)) == 0;
    if (rebuild) {
      flow_field_.build(
        *costmap_, path_, s.flow_path_seed, s.flow_viscosity,
        static_cast<float>(state_.pose.pose.position.x),
        static_cast<float>(state_.pose.pose.position.y), s.tgmppi_body_radius);
      trackPseudopods();
      if (s.tgmppi_ancillary_debug) {
        publishAncillaryPaths();
      }
      if (s.tgmppi_debug) {
        publishFlowDebug();
      }
    }
  } else {
    applyFlowBias();  // water field: bend the sampling mean downhill
  }

#ifdef TGMPPI_WITH_CUDA
  if (settings_.compute_backend == "cuda" && gpu_rollout_.ready()) {
    // state_.cvx/cvy/cwz already carry this cycle's noise (+ TG-MPPI bias,
    // if applied above) -- predict + integrate run on the GPU, then write
    // the same state_.vx/vy/wz + generated_trajectories_ every critic and
    // the visualizer already read, unchanged either way.
    gpu_rollout_.rollout(state_, state_, generated_trajectories_, settings_.model_dt, isHolonomic());
    return;
  }
#endif
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
}

void Optimizer::spacetimeObstacleCallback(std::size_t index, const nav_msgs::msg::Odometry & msg)
{
  std::lock_guard<std::mutex> lock(spacetime_obstacles_mutex_);
  if (index >= spacetime_obstacles_.size()) {return;}
  spacetime_obstacles_[index].x = static_cast<float>(msg.pose.pose.position.x);
  spacetime_obstacles_[index].y = static_cast<float>(msg.pose.pose.position.y);
  spacetime_obstacles_[index].vx = static_cast<float>(msg.twist.twist.linear.x);
  spacetime_obstacles_[index].vy = static_cast<float>(msg.twist.twist.linear.y);
  spacetime_obstacles_[index].radius = settings_.tgmppi_spacetime_obstacle_radius;
  spacetime_obstacle_received_[index] = true;
}

void Optimizer::snapshotTrackedObstacles()
{
  tracked_obstacles_snapshot_.clear();
  std::lock_guard<std::mutex> lock(spacetime_obstacles_mutex_);
  for (std::size_t i = 0; i < spacetime_obstacles_.size(); ++i) {
    if (i < spacetime_obstacle_received_.size() && spacetime_obstacle_received_[i]) {
      tracked_obstacles_snapshot_.push_back(spacetime_obstacles_[i]);
    }
  }
}

// Resample a raw (x, y)-per-dt_layer path onto this controller's own
// (time_steps, model_dt) grid via linear interpolation, holding the
// final position beyond the path's own duration (preserves a wait
// tail as a held position, matching spacetime_path_to_proposal()'s
// documented intent). Simplified relative to the sandbox: no
// acceleration-limit reprojection or re-rollout-based refeasibility
// check -- a documented scoping simplification given this project's
// final-porting-day timeline, not an oversight.
bool Optimizer::resampleSpaceTimeRoute(
  const SpaceTimeRoute & route, float rx, float ry,
  std::vector<float> & v, std::vector<float> & w,
  std::vector<float> & x, std::vector<float> & y)
{
  const auto & s = settings_;
  v.assign(s.time_steps, 0.0f);
  w.assign(s.time_steps, 0.0f);
  x.reserve(s.time_steps);
  y.reserve(s.time_steps);
  float prev_x = rx, prev_y = ry, prev_yaw = 0.0f;
  bool have_prev = false;
  for (unsigned int t = 0; t < s.time_steps; ++t) {
    const float layer_f = static_cast<float>(t) * s.model_dt / s.tgmppi_spacetime_dt_layer;
    const int layer = std::min(
      static_cast<int>(route.path.size()) - 1, static_cast<int>(std::floor(layer_f)));
    const float frac = std::min(1.0f, layer_f - static_cast<float>(layer));
    const auto p0 = route.path[static_cast<std::size_t>(std::max(0, layer))];
    const auto p1 = route.path[static_cast<std::size_t>(
      std::min(static_cast<int>(route.path.size()) - 1, layer + 1))];
    const float px = p0.first + frac * (p1.first - p0.first);
    const float py = p0.second + frac * (p1.second - p0.second);
    x.push_back(px);
    y.push_back(py);
    if (have_prev) {
      const float dx = px - prev_x, dy = py - prev_y;
      v[t] = std::clamp(
        std::sqrt(dx * dx + dy * dy) / s.model_dt, 0.0f, s.constraints.vx_max);
    }
    prev_x = px;
    prev_y = py;
    have_prev = true;
  }
  (void)prev_yaw;

  // Headings come from a LOOKAHEAD along the resampled path, not from
  // consecutive points: two consecutive points of an 8-connected grid
  // path (cell size = spacetime_res) can differ in direction by 180 deg,
  // and that divided by model_dt gave |wz| ~ pi/model_dt = 63 rad/s --
  // 33x wz_max, an unfollowable reference that still entered the sampling
  // (observed 2026-09-16, log 36452..., pod slot 3). A route needing more
  // than wz_max for a large share of the horizon is not a proposal this
  // robot can act on, so it is rejected rather than clamped into nonsense.
  const std::size_t lookahead = std::max<std::size_t>(
    1u, static_cast<std::size_t>(std::lround(
      0.3 / std::max(0.01f, s.constraints.vx_max * s.model_dt))));
  float heading = static_cast<float>(tf2::getYaw(state_.pose.pose.orientation));
  unsigned int over_limit = 0u;
  for (unsigned int t = 0; t < s.time_steps; ++t) {
    const std::size_t j = std::min<std::size_t>(t + lookahead, s.time_steps - 1);
    const float dx = x[j] - x[t], dy = y[j] - y[t];
    const float yaw = (std::sqrt(dx * dx + dy * dy) > 1e-3f) ?
      std::atan2(dy, dx) : heading;
    const float raw_w =
      static_cast<float>(angles::shortest_angular_distance(heading, yaw)) / s.model_dt;
    if (std::fabs(raw_w) > s.constraints.wz) {++over_limit;}
    w[t] = std::clamp(raw_w, -s.constraints.wz, s.constraints.wz);
    heading = yaw;
  }
  return over_limit * 4u <= s.time_steps;   // reject if >25% of steps need more than wz_max
}

void Optimizer::trySpacetimeBlob(
  const std::vector<std::vector<std::pair<float, float>>> & pods, float rx, float ry,
  std::vector<std::vector<float>> & mode_v, std::vector<std::vector<float>> & mode_w,
  std::vector<std::vector<float>> & mode_x, std::vector<std::vector<float>> & mode_y,
  std::vector<bool> & mode_valid, std::vector<float> & promises_local)
{
  const auto & s = settings_;
  static unsigned int b_cycles = 0u, b_alt = 0u, b_appended = 0u, b_unfollowable = 0u;
  static double b_ms = 0.0, b_ms_max = 0.0;
  if (++b_cycles >= 100u) {
    RCLCPP_INFO(
      logger_,
      "[TGMPPI spacetime-blob] last %u cycles: alternative class found %u, modes appended %u, "
      "unfollowable %u, build %.2f ms mean / %.2f ms max",
      b_cycles, b_alt, b_appended, b_unfollowable, b_ms / b_cycles, b_ms_max);
    b_cycles = 0u; b_alt = 0u; b_appended = 0u; b_unfollowable = 0u;
    b_ms = 0.0; b_ms_max = 0.0;
  }

  const std::size_t mode_count = std::min(pods.size(), mode_valid.size());
  if (!flow_field_.ready()) {return;}

  // Promise scale: extras compete with the static pseudopods, whose promise is the water
  // level at a membrane exit ~body_radius away, while a blob exit is only horizon*speed away
  // (a different distance from the goal). So anchor the best blob route to the best static
  // pseudopod's promise and add each alternative's extra space-time cost in metres.
  float base = std::numeric_limits<float>::max();
  for (std::size_t m = 0; m < mode_count; ++m) {
    if (mode_valid[m]) {base = std::min(base, promises_local[m]);}
  }
  if (base >= std::numeric_limits<float>::max()) {return;}

  const float robot_r = static_cast<float>(costmap_ros_->getLayeredCostmap()->getInscribedRadius());
  const auto static_free = [this](float x, float y) {
      unsigned int mx, my;
      if (!costmap_->worldToMap(x, y, mx, my)) {return false;}
      const auto cost = costmap_->getCost(mx, my);
      return cost != nav2_costmap_2d::LETHAL_OBSTACLE &&
             cost != nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
    };
  SpaceTimeBodyParams bp;
  bp.horizon = s.tgmppi_spacetime_horizon;
  bp.dt_layer = s.tgmppi_spacetime_dt_layer;
  bp.res = s.tgmppi_spacetime_res;
  bp.robot_r = robot_r;
  bp.max_regret = s.tgmppi_spacetime_blob_max_regret;
  spacetime_body_.build(
    static_free, tracked_obstacles_snapshot_,
    [this](float x, float y) {return flow_field_.distAt(x, y);}, rx, ry, bp);
  b_ms += spacetime_body_.buildMs();
  b_ms_max = std::max(b_ms_max, spacetime_body_.buildMs());
  if (!spacetime_body_.ready() || spacetime_body_.pods().size() < 2u) {return;}

  // Extras only when the blob finds a genuinely different homotopy class (pass-before vs
  // pass-behind, above vs below ...): route 0 (best) and the best route of another class.
  const auto & sigs = spacetime_body_.signatures();
  std::size_t alt = 0;
  for (std::size_t i = 1; i < sigs.size(); ++i) {
    if (sigs[i] != sigs[0]) {alt = i; break;}
  }
  if (alt == 0) {return;}
  ++b_alt;

  const float best_cost = spacetime_body_.promises()[0];
  for (const std::size_t i : {std::size_t{0}, alt}) {
    std::vector<float> v, w, x, y;
    if (!resampleSpaceTimeRoute(spacetime_body_.pods()[i], rx, ry, v, w, x, y)) {
      ++b_unfollowable;
      continue;
    }
    ++b_appended;
    mode_v.push_back(std::move(v));
    mode_w.push_back(std::move(w));
    mode_x.push_back(std::move(x));
    mode_y.push_back(std::move(y));
    mode_valid.push_back(true);
    promises_local.push_back(base + (spacetime_body_.promises()[i] - best_cost));
  }
}

void Optimizer::trySpacetimeAlternatives(
  const std::vector<std::vector<std::pair<float, float>>> & pods, float rx, float ry,
  std::vector<std::vector<float>> & mode_v, std::vector<std::vector<float>> & mode_w,
  std::vector<std::vector<float>> & mode_x, std::vector<std::vector<float>> & mode_y,
  std::vector<bool> & mode_valid, std::vector<float> & promises_local)
{
  const auto & s = settings_;
  if (!s.tgmppi_spacetime_enabled || s.tgmppi_spacetime_obstacle_topics.empty()) {
    return;
  }

  // Received obstacles only, snapshotted in prepare() (same set the critic sees).
  const std::vector<SpaceTimeObstacle> & obstacles = tracked_obstacles_snapshot_;
  if (obstacles.empty()) {
    return;
  }
  if (s.tgmppi_spacetime_blob) {
    trySpacetimeBlob(pods, rx, ry, mode_v, mode_w, mode_x, mode_y, mode_valid, promises_local);
    return;
  }

  // 2026-09-16 diagnostics: bags _20260915_234156 and _20260916_001024 showed
  // zero space-time mode SELECTIONS, but nothing measured whether a crossing
  // was ever detected in the first place. Count each stage separately so the
  // blocking one is identifiable instead of inferred.
  static unsigned int st_cycles = 0u, st_gate_pass = 0u, st_feasible = 0u,
    st_appended = 0u, st_not_distinct = 0u, st_unfollowable = 0u;
  static float st_best_margin = std::numeric_limits<float>::max();
  static double st_search_ms = 0.0;
  if (++st_cycles >= 100u) {
    RCLCPP_INFO(
      logger_,
      "[TGMPPI spacetime] last %u cycles: crossing detected %u (best margin %.2f m vs "
      "relevance %.2f m), searches feasible %u, modes appended %u, non-distinct %u, "
      "unfollowable %u, search %.1f ms total",
      st_cycles, st_gate_pass, st_best_margin, s.tgmppi_spacetime_relevance,
      st_feasible, st_appended, st_not_distinct, st_unfollowable, st_search_ms);
    st_cycles = 0u; st_gate_pass = 0u; st_feasible = 0u; st_appended = 0u;
    st_not_distinct = 0u; st_search_ms = 0.0; st_unfollowable = 0u;
    st_best_margin = std::numeric_limits<float>::max();
  }

  const float search_speed = s.tgmppi_spacetime_res / s.tgmppi_spacetime_dt_layer;
  const int nk = static_cast<int>(std::lround(s.tgmppi_spacetime_horizon / s.tgmppi_spacetime_dt_layer)) + 1;
  // The robot's own safe circular radius -- NOT tgmppi_body_radius, which
  // is the flow field's flood extent (default 2.5m), an unrelated
  // parameter that happens to share the word "radius".
  const float robot_r = static_cast<float>(costmap_ros_->getLayeredCostmap()->getInscribedRadius());

  // Single-crossing budget (see trySpacetimeAlternatives()'s declaration
  // comment in optimizer.hpp): unlike the sandbox's per-branch loop, this
  // stops at the FIRST pseudopod with a detected crossing, bounded to at
  // most 2 extra modes total (wait + detour) to fit the fixed 5-slot
  // ancillary-mode bookkeeping without a larger refactor.
  const std::size_t mode_count = std::min(pods.size(), mode_valid.size());
  for (std::size_t m = 0; m < mode_count; ++m) {
    if (!mode_valid[m] || pods[m].size() < 2) {continue;}

    // Arc length along this pseudopod's own centerline, needed for both
    // the local goal and the synchronized check points below.
    float total_arc = 0.0f;
    std::vector<float> cum_arc{0.0f};
    for (std::size_t k = 1; k < pods[m].size(); ++k) {
      const float dx = pods[m][k].first - pods[m][k - 1].first;
      const float dy = pods[m][k].second - pods[m][k - 1].second;
      total_arc += std::sqrt(dx * dx + dy * dy);
      cum_arc.push_back(total_arc);
    }

    const auto local_goal = pointAtArcLength(
      pods[m], 0.9f * search_speed * s.tgmppi_spacetime_horizon);

    // nearestCrossingObstacle: worst (most negative) predicted clearance
    // margin, checked at the search's own synchronized time layers against
    // the centerline's OWN nominal-speed progression (not the ancillary
    // controller's achieved speed) -- direct port of
    // spacetime.py::nearest_crossing_obstacle().
    float worst_margin = std::numeric_limits<float>::max();
    std::size_t worst_obstacle = 0;
    for (int k = 0; k < nk; ++k) {
      const float t = static_cast<float>(k) * s.tgmppi_spacetime_dt_layer;
      const auto check_xy = pointAtArcLength(pods[m], search_speed * t);
      for (std::size_t oi = 0; oi < obstacles.size(); ++oi) {
        const auto pred = SpaceTimeSearch::predict(obstacles[oi], t, s.tgmppi_spacetime_horizon);
        const float dx = check_xy.first - pred.first, dy = check_xy.second - pred.second;
        const float margin = std::sqrt(dx * dx + dy * dy) -
          (obstacles[oi].radius + robot_r);
        if (margin < worst_margin) {worst_margin = margin; worst_obstacle = oi;}
      }
    }
    if (worst_margin < st_best_margin) {st_best_margin = worst_margin;}
    if (worst_margin > s.tgmppi_spacetime_relevance) {continue;}   // no genuine crossing here
    ++st_gate_pass;

    const SpaceTimeObstacle crossing_obstacle = obstacles[worst_obstacle];
    const auto static_free = [this](float x, float y) {
        unsigned int mx, my;
        if (!costmap_->worldToMap(x, y, mx, my)) {return false;}
        const auto cost = costmap_->getCost(mx, my);
        return cost != nav2_costmap_2d::LETHAL_OBSTACLE &&
        cost != nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
      };

    SpaceTimeRoute wait_route, detour_route;
    bool distinct = false;
    const auto search_t0 = std::chrono::steady_clock::now();
    SpaceTimeSearch::twoRouteSearch(
      static_free, {crossing_obstacle}, rx, ry, local_goal.first, local_goal.second,
      robot_r, s.tgmppi_spacetime_horizon, s.tgmppi_spacetime_dt_layer,
      s.tgmppi_spacetime_res, s.tgmppi_spacetime_window, s.tgmppi_spacetime_res * 1.5f,
      wait_route, detour_route, distinct);
    st_search_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - search_t0).count();
    if (wait_route.feasible || detour_route.feasible) {++st_feasible;}

    const auto resample = [&](const SpaceTimeRoute & route,
      std::vector<float> & v, std::vector<float> & w,
      std::vector<float> & x, std::vector<float> & y) -> bool {
        return resampleSpaceTimeRoute(route, rx, ry, v, w, x, y);
      };

    // twoRouteSearch already reports whether the two routes pass the crossing
    // obstacle on opposite sides (the whole point: wait-behind vs go-before).
    // That flag used to be ignored, so two identical routes could be appended
    // and burn both extra mode slots on the same behaviour. Keep both only
    // when they are genuinely distinct, otherwise keep the cheaper one.
    std::vector<std::reference_wrapper<const SpaceTimeRoute>> routes;
    if (distinct) {
      routes = {std::cref(wait_route), std::cref(detour_route)};
    } else {
      ++st_not_distinct;
      const bool prefer_wait = wait_route.feasible &&
        (!detour_route.feasible || wait_route.cost <= detour_route.cost);
      const SpaceTimeRoute & best = prefer_wait ? wait_route : detour_route;
      if (best.feasible) {routes.push_back(std::cref(best));}
    }
    for (const auto & route : routes) {
      if (!route.get().feasible) {continue;}
      std::vector<float> v, w, x, y;
      if (!resample(route.get(), v, w, x, y)) {
        ++st_unfollowable;
        continue;
      }
      ++st_appended;
      mode_v.push_back(std::move(v));
      mode_w.push_back(std::move(w));
      mode_x.push_back(std::move(x));
      mode_y.push_back(std::move(y));
      mode_valid.push_back(true);
      // Compete on equal footing with the triggering pseudopod's own
      // promise -- there is no independent "how good is this route"
      // score computed yet, so this is a reasonable, documented
      // simplification: an alternative to a mode is exactly as promising
      // as the mode it is an alternative to.
      promises_local.push_back(promises_local[m]);
      if (mode_v.size() >= mode_count + 2) {return;}   // budget exhausted
    }
    return;   // single-crossing budget: stop after the first triggering pseudopod
  }
}

void Optimizer::trackPseudopods()
{
  const auto & raw = flow_field_.pseudopods();
  const auto & raw_p = flow_field_.pseudopodPromises();
  const std::size_t raw_n = std::min(raw.size(), raw_p.size());
  if (!settings_.tgmppi_pod_tracking) {
    // Off: byte-identical to reading flow_field_ directly.
    display_pods_.assign(raw.begin(), raw.begin() + raw_n);
    display_promises_.assign(raw_p.begin(), raw_p.begin() + raw_n);
    return;
  }

  // amoeba_sandbox pseudopods.py BranchTracker.update(): resample both
  // centerlines to 32 arc-length points, metric = 0.5*mean point distance
  // + 0.5*endpoint distance, greedy best-unused match under match_distance,
  // otherwise a fresh id.
  constexpr std::size_t kResample = 32;
  auto resample = [](const std::vector<std::pair<float, float>> & poly) {
      std::vector<std::pair<float, float>> out(kResample);
      float length = 0.0f;
      for (std::size_t k = 1; k < poly.size(); ++k) {
        length += std::hypot(poly[k].first - poly[k - 1].first, poly[k].second - poly[k - 1].second);
      }
      for (std::size_t i = 0; i < kResample; ++i) {
        out[i] = pointAtArcLength(poly, length * static_cast<float>(i) / (kResample - 1));
      }
      return out;
    };
  std::vector<std::vector<std::pair<float, float>>> prev_samples;
  prev_samples.reserve(tracked_pods_prev_.size());
  for (const auto & old : tracked_pods_prev_) {prev_samples.push_back(resample(old.centerline));}
  std::vector<bool> used(tracked_pods_prev_.size(), false);
  std::vector<TrackedPod> tracked;
  for (std::size_t k = 0; k < raw_n; ++k) {
    if (raw[k].size() < 2) {continue;}
    const auto sample = resample(raw[k]);
    int best = -1;
    float best_metric = std::numeric_limits<float>::max();
    for (std::size_t j = 0; j < tracked_pods_prev_.size(); ++j) {
      if (used[j]) {continue;}
      float mean_d = 0.0f;
      for (std::size_t i = 0; i < kResample; ++i) {
        mean_d += std::hypot(
          sample[i].first - prev_samples[j][i].first, sample[i].second - prev_samples[j][i].second);
      }
      mean_d /= static_cast<float>(kResample);
      const auto & old_end = tracked_pods_prev_[j].centerline.back();
      const float end_d = std::hypot(raw[k].back().first - old_end.first, raw[k].back().second - old_end.second);
      const float metric = 0.5f * mean_d + 0.5f * end_d;
      if (metric < best_metric) {best_metric = metric; best = static_cast<int>(j);}
    }
    int id;
    if (best >= 0 && best_metric <= settings_.tgmppi_pod_match_distance) {
      used[best] = true;
      id = tracked_pods_prev_[best].id;
    } else {
      id = next_pod_id_++;
    }
    tracked.push_back({id, raw[k], raw_p[k]});
  }

  // Slot assignment: an id that already owns a slot keeps it; ids that
  // vanished free their slot; new ids take the first free slot (there are
  // never more than kPodSlots pseudopods, the flood extracts <= 3).
  std::array<int, kPodSlots> new_slots{{-1, -1, -1}};
  std::vector<bool> placed(tracked.size(), false);
  for (std::size_t m = 0; m < kPodSlots; ++m) {
    if (pod_slot_ids_[m] < 0) {continue;}
    for (std::size_t k = 0; k < tracked.size(); ++k) {
      if (!placed[k] && tracked[k].id == pod_slot_ids_[m]) {
        new_slots[m] = tracked[k].id;
        placed[k] = true;
        break;
      }
    }
  }
  for (std::size_t k = 0; k < tracked.size(); ++k) {
    if (placed[k]) {continue;}
    for (std::size_t m = 0; m < kPodSlots; ++m) {
      if (new_slots[m] < 0) {
        new_slots[m] = tracked[k].id;
        placed[k] = true;
        break;
      }
    }
  }
  pod_slot_ids_ = new_slots;
  display_pods_.assign(kPodSlots, std::vector<std::pair<float, float>>{});
  display_promises_.assign(kPodSlots, 0.0f);
  for (std::size_t m = 0; m < kPodSlots; ++m) {
    if (new_slots[m] < 0) {continue;}
    for (const auto & pod : tracked) {
      if (pod.id == new_slots[m]) {
        display_pods_[m] = pod.centerline;
        display_promises_[m] = pod.promise;
        break;
      }
    }
  }
  tracked_pods_prev_ = std::move(tracked);
}

void Optimizer::applyFlowBias()
{
  const auto & s = settings_;
  if (flow_cycle_++ %
    static_cast<unsigned int>(std::max(1, s.flow_reflood_every)) == 0)
  {
    flow_field_.build(
      *costmap_, path_, s.flow_path_seed, s.flow_viscosity,
      static_cast<float>(state_.pose.pose.position.x),
      static_cast<float>(state_.pose.pose.position.y), s.tgmppi_body_radius);
    trackPseudopods();
    if (s.tgmppi_ancillary_debug) {
      publishAncillaryPaths();
    }
  }
  if (s.tgmppi_debug) {
    publishFlowDebug();
  }
  if (!flow_field_.ready()) {
    return;  // no plan yet or window fully dry -> plain MPPI
  }

  flow_path_blocked_now_ = isLocalPathBlocked();
  if (!s.flow_assist_only_when_path_blocked) {
    tgmppi_assist_active_ = true;
    flow_clear_cycles_ = 0u;
  } else if (flow_path_blocked_now_) {
    tgmppi_assist_active_ = true;
    flow_clear_cycles_ = 0u;
  } else if (tgmppi_assist_active_) {
    ++flow_clear_cycles_;
    if (flow_clear_cycles_ >= static_cast<unsigned int>(
        std::max(1, s.flow_clear_confirm_cycles)))
    {
      tgmppi_assist_active_ = false;
      flow_clear_cycles_ = 0u;
    }
  }
  // Stabilizer: rate-limited assist level (sandbox gate_rate). ramp=1.0
  // makes this exactly 1 in ASSIST / 0 in NORMAL, i.e. today's behaviour.
  const float ramp = std::clamp(s.tgmppi_assist_ramp_rate, 0.0f, 1.0f);
  assist_level_ += ramp * ((tgmppi_assist_active_ ? 1.0f : 0.0f) - assist_level_);
  if (!tgmppi_assist_active_) {
    ancillary_mode_valid_.fill(false);
    ancillary_mode_samples_.fill(0u);
    ancillary_mode_rejoin_prior_.fill(0.0f);
    flow_wait_samples_ = 0u;
    for (auto & rollout : ancillary_rollout_x_) {
      rollout.clear();
    }
    for (auto & rollout : ancillary_rollout_y_) {
      rollout.clear();
    }
    if (s.tgmppi_ancillary_debug) {publishAncillaryRollouts();}
    if (s.tgmppi_debug) {publishFlowDebug();}
    return;  // NORMAL: protect ordinary global-path tracking
  }

  const bool equal_allocation = s.tgmppi_group_allocation == "equal";
  // "equal" allocation does not use bias_strength, so only the assist ramp can
  // switch guidance off there.
  const float frac = equal_allocation ? std::clamp(assist_level_, 0.0f, 1.0f) :
    std::clamp(s.tgmppi_bias_strength, 0.0f, 1.0f) * assist_level_;
  if (frac <= 0.0f) {
    return;
  }

  const float rx = static_cast<float>(state_.pose.pose.position.x);
  const float ry = static_cast<float>(state_.pose.pose.position.y);
  const float ryaw = static_cast<float>(tf2::getYaw(state_.pose.pose.orientation));

  // Endgame guard (same as ray mode): within tgmppi_goal_dist of the plan
  // end, drop the bias and let plain MPPI + GoalCritic dock.
  const std::size_t path_size = path_.x.shape(0);
  if (path_size >= 1) {
    const float ex = path_.x(path_size - 1) - rx;
    const float ey = path_.y(path_size - 1) - ry;
    if (ex * ex + ey * ey < s.tgmppi_goal_dist * s.tgmppi_goal_dist) {
      return;
    }
  }

  const auto & pods = display_pods_;          // slot-ordered (see trackPseudopods)
  const auto & promises = display_promises_;
  const std::size_t mode_count = std::min(pods.size(), promises.size());
  std::size_t nonempty_pods = 0;
  for (std::size_t m = 0; m < mode_count; ++m) {
    if (pods[m].size() >= 2) {++nonempty_pods;}
  }
  if (nonempty_pods == 0) {
    return;  // goal inside body or no distinct reachable membrane exit
  }
  for (std::size_t m = 0; m < ancillary_mode_key_.size(); ++m) {
    ancillary_mode_key_[m] = (m < kPodSlots) ?
      (s.tgmppi_pod_tracking ? pod_slot_ids_[m] : static_cast<int>(m)) :
      kSpacetimeKeyBase + static_cast<int>(m);
  }
  const float warm = std::clamp(s.tgmppi_mode_warm_start, 0.0f, 1.0f);

  // Convert each geodesic centerline into a finite-horizon diff-drive
  // ancillary controller. This is display-independent and produces a genuine
  // time-varying (v,w) sequence for every pseudopod.
  std::vector<std::vector<float>> mode_v(mode_count);
  std::vector<std::vector<float>> mode_w(mode_count);
  std::vector<std::vector<float>> mode_x(mode_count);
  std::vector<std::vector<float>> mode_y(mode_count);
  std::vector<bool> mode_valid(mode_count, true);
  ancillary_mode_valid_.fill(false);
  ancillary_mode_samples_.fill(0u);
  ancillary_mode_row_start_.fill(0u);
  ancillary_mode_rejoin_prior_.fill(0.0f);
  flow_wait_samples_ = 0u;
  for (auto & rollout : ancillary_rollout_x_) {
    rollout.clear();
  }
  for (auto & rollout : ancillary_rollout_y_) {
    rollout.clear();
  }
  const auto & footprint = costmap_ros_->getRobotFootprint();
  const bool tracking_unknown = costmap_ros_->getLayeredCostmap()->isTrackingUnknown();
  const unsigned int collision_stride =
    static_cast<unsigned int>(std::max(1, s.ancillary_collision_stride));
  // Builds one pseudopod's ancillary (v,w) reference sequence with a given
  // heading-error speed floor, filling mode_v/mode_w/mode_x/mode_y[m] and
  // returning whether the resulting rollout stays collision-free. Extracted
  // (2026-09-11, amoeba_sandbox nominal_fb port) so
  // tgmppi_reference_infeasible_fallback can re-run this for one mode with
  // the safe 0.15 floor if the configured floor produced an invalid
  // rollout -- see tgmppi_reference_min_speed_ratio's docstring in
  // optimizer_settings.hpp. Byte-identical to the pre-refactor loop body
  // when min_speed_ratio==0.15 (the default).
  auto generate_pod_reference = [&](std::size_t m, float min_speed_ratio) -> bool {
      const std::pair<std::vector<float>, std::vector<float>> * warm_prev = nullptr;
      if (warm > 0.0f && m < ancillary_mode_key_.size()) {
        const auto it = mode_nominals_.find(ancillary_mode_key_[m]);
        if (it != mode_nominals_.end() && it->second.first.size() == s.time_steps &&
          it->second.second.size() == s.time_steps)
        {
          warm_prev = &it->second;
        }
      }
      mode_v[m].assign(s.time_steps, 0.0f);
      mode_w[m].assign(s.time_steps, 0.0f);
      mode_x[m].clear();
      mode_x[m].reserve(s.time_steps);
      mode_y[m].clear();
      mode_y[m].reserve(s.time_steps);
      bool valid = true;
      float x = rx, y = ry, yaw = ryaw;
      std::size_t cursor = 0;
      for (unsigned int t = 0; t < s.time_steps; ++t) {
        // Advance to the nearest non-regressing point, then select a spatial
        // lookahead target. Restricting the search to [cursor,end) prevents the
        // controller from jumping backward on a curved pseudopod.
        float nearest_d2 = std::numeric_limits<float>::max();
        std::size_t nearest = cursor;
        for (std::size_t k = cursor; k < pods[m].size(); ++k) {
          const float ex = pods[m][k].first - x;
          const float ey = pods[m][k].second - y;
          const float d2 = ex * ex + ey * ey;
          if (d2 < nearest_d2) {nearest_d2 = d2; nearest = k;}
        }
        cursor = nearest;
        std::size_t target = cursor;
        float arc = 0.0f;
        while (target + 1 < pods[m].size() && arc < s.tgmppi_lookahead_dist) {
          arc += std::hypot(
            pods[m][target + 1].first - pods[m][target].first,
            pods[m][target + 1].second - pods[m][target].second);
          ++target;
        }

        const float ex = pods[m][target].first - x;
        const float ey = pods[m][target].second - y;
        const float want = std::atan2(ey, ex);
        float err = static_cast<float>(
          angles::shortest_angular_distance(yaw, want));
        if (std::fabs(err) < s.tgmppi_bias_deadband) {err = 0.0f;}  // stabilizer: deadband
        float w = std::clamp(
          s.tgmppi_bias_gain * err, -s.constraints.wz, s.constraints.wz);
        const float nominal_v = std::fabs(control_sequence_.vx(t));
        const float cruise = std::clamp(
          std::max(s.tgmppi_pod_cruise_speed, nominal_v), 0.0f, s.constraints.vx_max);
        const float turn_scale = std::clamp(std::cos(err), min_speed_ratio, 1.0f);
        const float endpoint_dist = std::hypot(
          pods[m].back().first - x, pods[m].back().second - y);
        const float stop_scale = std::clamp(endpoint_dist / 0.25f, 0.25f, 1.0f);
        float v = cruise * turn_scale * stop_scale;
        const float v_ref = v, w_ref = w;
        float warm_v = 0.0f, warm_w = 0.0f;
        if (warm_prev != nullptr) {
          // stabilizer: sandbox mode_warm_start against this mode's own
          // previous local mean, shifted one step (mode_nominals).
          const std::size_t tp = std::min<std::size_t>(t + 1, s.time_steps - 1);
          warm_v = warm_prev->first[tp];
          warm_w = warm_prev->second[tp];
          v = warm * warm_v + (1.0f - warm) * v;
          w = warm * warm_w + (1.0f - warm) * w;
        }
        // 2026-09-17: the warm-start blend above was never clamped, so whatever a
        // mode's stored mean held went straight into the reference -- log 37188
        // shows a whole pod block sampled at cwz ~ -9.9e15 rad/s (wz_max 1.9).
        // A reference outside the robot's own limits can never be followed, so
        // clamp it here, and log the inputs once so the source can be traced.
        const bool v_bad = utils::isBadFloat(v) ||
          v > s.constraints.vx_max * 1.01f + 1e-3f || v < s.constraints.vx_min * 1.01f - 1e-3f;
        const bool w_bad = utils::isBadFloat(w) ||
          std::fabs(w) > s.constraints.wz * 1.01f + 1e-3f;
        if (v_bad || w_bad) {
          static unsigned int ref_logs = 0u;
          if (ref_logs++ < 20u) {
            RCLCPP_WARN(
              logger_,
              "[TGMPPI diag] pod reference out of limits: slot %zu key %d t %u v %g w %g "
              "(pre-warm v %g w %g, warm v %g w %g, warm-start %s)",
              m, m < ancillary_mode_key_.size() ? ancillary_mode_key_[m] : -999, t, v, w,
              v_ref, w_ref, warm_v, warm_w, warm_prev != nullptr ? "on" : "off");
          }
          v = utils::isBadFloat(v) ? 0.0f :
            std::clamp(v, s.constraints.vx_min, s.constraints.vx_max);
          w = utils::isBadFloat(w) ? 0.0f : std::clamp(w, -s.constraints.wz, s.constraints.wz);
        }
        mode_v[m][t] = v;
        mode_w[m][t] = w;
        yaw += w * s.model_dt;
        x += v * std::cos(yaw) * s.model_dt;
        y += v * std::sin(yaw) * s.model_dt;
        mode_x[m].push_back(x);
        mode_y[m].push_back(y);

        const bool check_pose = t % collision_stride == 0 || t + 1 == s.time_steps;
        if (s.ancillary_collision_check && valid && check_pose) {
          unsigned int mx, my;
          if (!costmap_->worldToMap(x, y, mx, my)) {
            valid = false;
          } else {
            const double footprint_cost = ancillary_collision_checker_.footprintCostAtPose(
              x, y, yaw, footprint);
            const auto cell_cost = static_cast<unsigned char>(footprint_cost);
            valid = cell_cost != nav2_costmap_2d::LETHAL_OBSTACLE &&
              (cell_cost != nav2_costmap_2d::NO_INFORMATION || tracking_unknown);
          }
        }
      }
      return valid;
    };

  for (std::size_t m = 0; m < mode_count; ++m) {
    if (pods[m].size() < 2) {
      // empty tracked slot (tgmppi_pod_tracking): no pseudopod here this reflood
      mode_valid[m] = false;
      if (m < ancillary_mode_valid_.size()) {ancillary_mode_valid_[m] = false;}
      continue;
    }
    bool valid = generate_pod_reference(m, s.tgmppi_reference_min_speed_ratio);
    if (!valid && s.tgmppi_reference_infeasible_fallback &&
      s.tgmppi_reference_min_speed_ratio > 0.15f)
    {
      // Retry this one pseudopod with the safe floor -- the sandbox's
      // reference_infeasible_fallback, applied per-branch.
      valid = generate_pod_reference(m, 0.15f);
    }
    mode_valid[m] = valid;
    if (m < ancillary_mode_valid_.size()) {
      ancillary_mode_valid_[m] = mode_valid[m];
      if (mode_valid[m]) {
        ancillary_rollout_x_[m] = mode_x[m];
        ancillary_rollout_y_[m] = mode_y[m];

        // Rejoin prior: prefer a safe bypass that stays near the supplied
        // global path and has rejoined it by the end of the MPPI horizon.
        float lateral_sum = 0.0f;
        for (std::size_t t = 0; t < mode_x[m].size(); ++t) {
          float nearest_d2 = std::numeric_limits<float>::max();
          for (std::size_t k = 0; k < path_size; ++k) {
            const float dx = mode_x[m][t] - path_.x(k);
            const float dy = mode_y[m][t] - path_.y(k);
            nearest_d2 = std::min(nearest_d2, dx * dx + dy * dy);
          }
          lateral_sum += std::sqrt(nearest_d2);
        }
        const float mean_lateral = lateral_sum /
          static_cast<float>(std::max<std::size_t>(1, mode_x[m].size()));
        float progress_deficit = 0.0f;
        if (!mode_x[m].empty()) {
          float endpoint_d2 = std::numeric_limits<float>::max();
          std::size_t endpoint_index = 0;
          for (std::size_t k = 0; k < path_size; ++k) {
            const float dx = mode_x[m].back() - path_.x(k);
            const float dy = mode_y[m].back() - path_.y(k);
            const float d2 = dx * dx + dy * dy;
            if (d2 < endpoint_d2) {endpoint_d2 = d2; endpoint_index = k;}
          }
          float current_d2 = std::numeric_limits<float>::max();
          std::size_t current_index = 0;
          for (std::size_t k = 0; k < path_size; ++k) {
            const float dx = rx - path_.x(k);
            const float dy = ry - path_.y(k);
            const float d2 = dx * dx + dy * dy;
            if (d2 < current_d2) {current_d2 = d2; current_index = k;}
          }
          const auto remaining_path = [this, path_size](std::size_t begin) {
              float remaining = 0.0f;
              for (std::size_t k = begin + 1; k < path_size; ++k) {
                remaining += std::hypot(
                  path_.x(k) - path_.x(k - 1), path_.y(k) - path_.y(k - 1));
              }
              return remaining;
            };
          const float current_remaining = remaining_path(current_index);
          const float endpoint_remaining = remaining_path(endpoint_index);
          const float expected_progress = 0.18f * s.model_dt *
            static_cast<float>(s.time_steps);
          const float target_remaining = std::max(0.0f, current_remaining - expected_progress);
          progress_deficit = std::max(0.0f, endpoint_remaining - target_remaining);
        }
        ancillary_mode_rejoin_prior_[m] =
          s.flow_rejoin_lateral_weight * mean_lateral +
          s.flow_rejoin_remaining_weight * progress_deficit;
      }
    }
  }

  // amoeba_sandbox spacetime.py Phase 1: may append up to 2 extra modes
  // (wait/detour) to mode_v/w/x/y/valid past index mode_count-1. No-op
  // when tgmppi_spacetime_enabled is false. promises_local is a mutable
  // copy since flow_field_.pseudopodPromises() is a const reference into
  // FlowField's own storage -- extras need a promise value too (see
  // trySpacetimeAlternatives()'s docstring: they compete on equal footing
  // with the pseudopod that triggered them).
  std::vector<float> promises_local(promises.begin(), promises.end());
  trySpacetimeAlternatives(pods, rx, ry, mode_v, mode_w, mode_x, mode_y, mode_valid, promises_local);
  for (std::size_t m = mode_count; m < mode_v.size() && m < ancillary_mode_valid_.size(); ++m) {
    ancillary_mode_valid_[m] = mode_valid[m];
    ancillary_rollout_x_[m] = mode_x[m];
    ancillary_rollout_y_[m] = mode_y[m];
    ancillary_mode_rejoin_prior_[m] = 0.0f;   // not path-rejoin-scored, unlike ordinary pseudopods
    // Extras are appended at mode_count, which is < kPodSlots when tracking
    // is off and fewer than 3 pseudopods exist -- re-key by what the slot
    // actually holds so a space-time extra never shares a pseudopod's key
    // (that would corrupt hysteresis + warm-start memory).
    ancillary_mode_key_[m] = kSpacetimeKeyBase + static_cast<int>(m);
  }

  std::vector<std::size_t> active_modes;
  active_modes.reserve(mode_v.size());
  for (std::size_t m = 0; m < mode_v.size(); ++m) {
    if (mode_valid[m]) {active_modes.push_back(m);}
  }
  // Multimodal allocation. "legacy": promise-weighted, and only this leading
  // fraction of the batch is recentred -- all remaining rows keep vanilla MPPI
  // sampling as one fallback group. "equal": see tgmppi_group_allocation.
  const bool offer_wait = s.flow_wait_enabled && flow_path_blocked_now_;
  unsigned int n_biased = 0u;
  unsigned int n_wait = 0u;
  if (equal_allocation) {
    // amoeba_sandbox allocate_group_counts(): each active mode, the wait group if
    // offered, and the unguided fallback all get a near-equal share of the batch.
    const unsigned int n_groups =
      static_cast<unsigned int>(active_modes.size()) + (offer_wait ? 1u : 0u) + 1u;
    const unsigned int share = s.batch_size / n_groups;
    n_biased = static_cast<unsigned int>(frac * static_cast<float>(s.batch_size - share));
    n_wait = offer_wait ? static_cast<unsigned int>(frac * static_cast<float>(share)) : 0u;
    n_wait = std::min(n_wait, n_biased);
  } else {
    n_biased = std::min(
      s.batch_size,
      static_cast<unsigned int>(frac * static_cast<float>(s.batch_size)));
    n_wait = offer_wait ? static_cast<unsigned int>(std::lround(
        static_cast<float>(n_biased) * std::clamp(s.flow_wait_fraction, 0.0f, 1.0f))) : 0u;
    if (offer_wait) {n_wait = std::max(1u, n_wait);}
    n_wait = std::min(n_wait, n_biased);
    if (active_modes.empty()) {
      n_wait = offer_wait ? n_biased : 0u;
    }
  }
  if (n_biased == 0) {return;}
  const unsigned int pod_budget = n_biased - n_wait;
  if (active_modes.empty() && n_wait == 0u) {
    if (s.tgmppi_ancillary_debug) {publishAncillaryRollouts();}
    if (s.tgmppi_debug) {publishFlowDebug();}
    RCLCPP_DEBUG(logger_, "[TGMPPI] all ancillary means rejected; using vanilla MPPI");
    return;
  }
  const float temperature = std::max(0.05f, s.flow_promise_temperature);
  float best_promise = std::numeric_limits<float>::max();
  for (const auto m : active_modes) {
    best_promise = std::min(best_promise, promises_local[m]);
  }
  std::vector<float> weights(active_modes.size(), 0.0f);
  float weight_sum = 0.0f;
  for (std::size_t i = 0; i < active_modes.size(); ++i) {
    weights[i] = equal_allocation ? 1.0f :
      std::exp(-(promises_local[active_modes[i]] - best_promise) / temperature);
    weight_sum += weights[i];
  }

  unsigned int row = 0;
  for (std::size_t i = 0; i < active_modes.size() && row < pod_budget; ++i) {
    const std::size_t m = active_modes[i];
    const unsigned int modes_left = static_cast<unsigned int>(active_modes.size() - i);
    const unsigned int rows_left = pod_budget - row;
    unsigned int count = (i + 1 == active_modes.size()) ? rows_left :
      static_cast<unsigned int>(std::lround(
        static_cast<float>(pod_budget) * weights[i] / weight_sum));
    if (pod_budget >= active_modes.size()) {
      count = std::max(1u, count);
      count = std::min(count, rows_left - (modes_left - 1));
    } else {
      count = std::min(count, rows_left);
    }
    const unsigned int end = row + count;
    if (m < ancillary_mode_row_start_.size()) {ancillary_mode_row_start_[m] = row;}
    for (; row < end; ++row) {
      for (unsigned int t = 0; t < s.time_steps; ++t) {
        state_.cvx(row, t) += mode_v[m][t] - control_sequence_.vx(t);
        state_.cwz(row, t) += mode_w[m][t] - control_sequence_.wz(t);
      }
    }
    if (m < ancillary_mode_samples_.size()) {ancillary_mode_samples_[m] = count;}
  }
  // A stationary proposal lets MPPI wait briefly for a transient blockage.
  // It is not a forced command: obstacle, path and goal critics score these
  // rows against every ordinary and pseudopod rollout.
  flow_wait_samples_ = n_wait;
  const unsigned int wait_end = std::min(s.batch_size, row + n_wait);
  for (; row < wait_end; ++row) {
    for (unsigned int t = 0; t < s.time_steps; ++t) {
      state_.cvx(row, t) += -control_sequence_.vx(t);
      state_.cwz(row, t) += -control_sequence_.wz(t);
    }
  }
  if (s.tgmppi_ancillary_debug) {publishAncillaryRollouts();}
  if (s.tgmppi_debug) {publishFlowDebug();}
}

void Optimizer::applyTgMppiModePriors()
{
  if (settings_.tgmppi_shadow_mode || !settings_.tgmppi_bias_enabled ||
    !tgmppi_assist_active_)
  {
    return;
  }
  for (std::size_t m = 0; m < ancillary_mode_samples_.size(); ++m) {
    const unsigned int begin = ancillary_mode_row_start_[m];
    const unsigned int end = std::min(
      settings_.batch_size, begin + ancillary_mode_samples_[m]);
    for (unsigned int row = begin; row < end; ++row) {
      costs_(row) += ancillary_mode_rejoin_prior_[m];
    }
  }
}

bool Optimizer::isLocalPathBlocked() const
{
  const std::size_t path_size = path_.x.shape(0);
  if (path_size == 0) {return false;}

  const float rx = static_cast<float>(state_.pose.pose.position.x);
  const float ry = static_cast<float>(state_.pose.pose.position.y);
  std::size_t closest = 0;
  float closest_d2 = std::numeric_limits<float>::max();
  for (std::size_t i = 0; i < path_size; ++i) {
    const float dx = path_.x(i) - rx;
    const float dy = path_.y(i) - ry;
    const float d2 = dx * dx + dy * dy;
    if (d2 < closest_d2) {closest_d2 = d2; closest = i;}
  }

  std::size_t checked = 0;
  std::size_t invalid = 0;
  float arc = 0.0f;
  const bool tracking_unknown = costmap_ros_->getLayeredCostmap()->isTrackingUnknown();
  for (std::size_t i = closest; i < path_size; ++i) {
    if (i > closest) {
      arc += std::hypot(path_.x(i) - path_.x(i - 1), path_.y(i) - path_.y(i - 1));
      if (arc > settings_.flow_path_check_distance) {break;}
    }
    ++checked;
    unsigned int mx, my;
    if (!costmap_->worldToMap(path_.x(i), path_.y(i), mx, my)) {
      ++invalid;
      continue;
    }
    const unsigned char cost = costmap_->getCost(mx, my);
    if (cost == nav2_costmap_2d::LETHAL_OBSTACLE ||
      cost == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE ||
      (cost == nav2_costmap_2d::NO_INFORMATION && !tracking_unknown))
    {
      ++invalid;
    }
  }
  if (checked == 0 || invalid <= 2) {return false;}
  return static_cast<float>(invalid) / static_cast<float>(checked) >
         settings_.flow_path_blocked_ratio;
}

void Optimizer::publishAncillaryPaths()
{
  const auto & pods = display_pods_;  // slot-ordered: path i == ancillary slot i
  const std::string frame = costmap_ros_->getGlobalFrameID();
  for (std::size_t i = 0; i < ancillary_path_pubs_.size(); ++i) {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame;
    path.header.stamp = rclcpp::Time(0, 0);
    if (i < pods.size()) {
      path.poses.reserve(pods[i].size());
      for (const auto & point : pods[i]) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path.header;
        pose.pose.position.x = point.first;
        pose.pose.position.y = point.second;
        pose.pose.position.z = 0.08;
        pose.pose.orientation.w = 1.0;
        path.poses.push_back(std::move(pose));
      }
    }
    ancillary_path_pubs_[i]->publish(path);  // empty Path clears stale modes
  }
}

void Optimizer::publishAncillaryRollouts()
{
  const std::string frame = costmap_ros_->getGlobalFrameID();
  for (std::size_t i = 0; i < ancillary_rollout_pubs_.size(); ++i) {
    nav_msgs::msg::Path path;
    path.header.frame_id = frame;
    path.header.stamp = rclcpp::Time(0, 0);
    if (ancillary_mode_valid_[i]) {
      path.poses.reserve(ancillary_rollout_x_[i].size());
      for (std::size_t t = 0; t < ancillary_rollout_x_[i].size(); ++t) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header = path.header;
        pose.pose.position.x = ancillary_rollout_x_[i][t];
        pose.pose.position.y = ancillary_rollout_y_[i][t];
        pose.pose.position.z = 0.12;
        pose.pose.orientation.w = 1.0;
        path.poses.push_back(std::move(pose));
      }
    }
    ancillary_rollout_pubs_[i]->publish(path);
  }
}

void Optimizer::publishFlowDebug()
{
  if (!tgmppi_debug_pub_ || tgmppi_debug_pub_->get_subscription_count() == 0) {
    return;  // nobody listening -> skip the marker work
  }
  auto node = parent_.lock();
  if (!node) {
    return;
  }
  // Debug markers represent the current controller state, not historical data.
  // A zero stamp asks RViz to use the latest TF and avoids wall-time versus
  // simulation-time extrapolation errors during Gazebo runs.
  const rclcpp::Time stamp(0, 0);
  const std::string frame = costmap_ros_->getGlobalFrameID();
  constexpr double kZ = 0.05;

  visualization_msgs::msg::MarkerArray arr;
  visualization_msgs::msg::Marker clear;
  clear.header.frame_id = frame;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  arr.markers.push_back(clear);

  auto mk_point = [](double x, double y, double z) {
      geometry_msgs::msg::Point p;
      p.x = x; p.y = y; p.z = z;
      return p;
    };
  auto mk_color = [](float r, float g, float b, float a = 1.0f) {
      std_msgs::msg::ColorRGBA c;
      c.r = r; c.g = g; c.b = b; c.a = a;
      return c;
    };

  if (flow_field_.ready()) {
    // Finite robot-centred geodesic body. Overlapping spheres hide the square
    // cell lattice while preserving the exact collision-free support.
    visualization_msgs::msg::Marker body;
    body.header.frame_id = frame;
    body.header.stamp = stamp;
    body.ns = "tgmppi_body";
    body.id = 0;
    body.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    body.action = visualization_msgs::msg::Marker::ADD;
    body.pose.orientation.w = 1.0;
    const float res = flow_field_.resolution();
    body.scale.x = 1.35 * res;
    body.scale.y = 1.35 * res;
    body.scale.z = 0.018;
    body.color = mk_color(0.18f, 0.62f, 0.95f, 0.16f);

    visualization_msgs::msg::Marker membrane;
    membrane.header = body.header;
    membrane.ns = "tgmppi_membrane";
    membrane.id = 0;
    membrane.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    membrane.action = visualization_msgs::msg::Marker::ADD;
    membrane.pose.orientation.w = 1.0;
    membrane.scale.x = 1.85 * res;
    membrane.scale.y = 1.85 * res;
    membrane.scale.z = 0.024;
    membrane.color = mk_color(0.05f, 0.25f, 0.95f, 0.82f);

    for (int j = 0; j < flow_field_.sizeY(); ++j) {
      for (int i = 0; i < flow_field_.sizeX(); ++i) {
        const float cx = flow_field_.originX() + i * res;
        const float cy = flow_field_.originY() + j * res;
        if (flow_field_.cellBody(i, j)) {
          body.points.push_back(mk_point(cx, cy, kZ));
        }
        if (flow_field_.cellMembrane(i, j)) {
          membrane.points.push_back(mk_point(cx, cy, kZ + 0.015));
        }
      }
    }
    // A cheap display-only halo around the membrane provides the organic soft
    // edge used by the sandbox without duplicating the much larger body cloud.
    auto membrane_halo = membrane;
    membrane_halo.ns = "tgmppi_membrane_halo";
    membrane_halo.scale.x = 3.10 * res;
    membrane_halo.scale.y = 3.10 * res;
    membrane_halo.scale.z = 0.012;
    membrane_halo.color = mk_color(0.16f, 0.55f, 1.00f, 0.10f);

    arr.markers.push_back(body);
    arr.markers.push_back(membrane_halo);
    arr.markers.push_back(membrane);

    const std::array<std_msgs::msg::ColorRGBA, 3> pod_colors = {
      mk_color(0.10f, 0.85f, 0.35f, 1.0f),
      mk_color(1.00f, 0.55f, 0.05f, 1.0f),
      mk_color(0.75f, 0.20f, 0.95f, 1.0f)};
    const auto & pods = flow_field_.pseudopods();
    for (std::size_t p = 0; p < pods.size(); ++p) {
      visualization_msgs::msg::Marker pod;
      pod.header = body.header;
      pod.ns = "tgmppi_pseudopods";
      pod.id = static_cast<int>(p);
      pod.type = visualization_msgs::msg::Marker::LINE_STRIP;
      pod.action = visualization_msgs::msg::Marker::ADD;
      pod.pose.orientation.w = 1.0;
      pod.scale.x = 0.055;
      pod.color = pod_colors[p % pod_colors.size()];
      for (const auto & xy : pods[p]) {
        pod.points.push_back(mk_point(xy.first, xy.second, kZ + 0.04));
      }
      arr.markers.push_back(std::move(pod));
    }

    if (settings_.tgmppi_debug_grid) {
      // Downhill "water" arrows: smooth gradient directions (they visibly bend
      // away from walls with viscosity), colored by level -- bright toward the
      // goal, faint at the deep end.
      visualization_msgs::msg::Marker dirs;
      dirs.header.frame_id = frame;
      dirs.header.stamp = stamp;
      dirs.ns = "flow_dirs";
      dirs.id = 0;
      dirs.type = visualization_msgs::msg::Marker::LINE_LIST;
      dirs.action = visualization_msgs::msg::Marker::ADD;
      dirs.scale.x = settings_.tgmppi_debug_arrow_width;
      dirs.pose.orientation.w = 1.0;
      const float dmax = std::max(1e-3f, flow_field_.dmax());
      const int st = std::max(
        1, static_cast<int>(std::lround(settings_.tgmppi_debug_arrow_spacing / res)));
      for (int j = 0; j < flow_field_.sizeY(); j += st) {
        for (int i = 0; i < flow_field_.sizeX(); i += st) {
          if (!flow_field_.cellWet(i, j)) {
            continue;  // dry
          }
          const float d = flow_field_.cellDist(i, j);
          const float cx = flow_field_.originX() + i * res;
          const float cy = flow_field_.originY() + j * res;
          float dx = 0.0f, dy = 0.0f;
          if (!flow_field_.gradAt(cx, cy, dx, dy) &&
            !flow_field_.cellDir(i, j, dx, dy))
          {
            continue;
          }
          const float wn = 1.0f - std::clamp(d / dmax, 0.0f, 1.0f);
          const auto col = mk_color(
            0.15f + 0.25f * (1.0f - wn), 0.45f + 0.4f * wn, 0.95f,
            0.35f + 0.55f * wn);
          dirs.points.push_back(mk_point(cx, cy, kZ));
          dirs.points.push_back(
            mk_point(
              cx + settings_.tgmppi_debug_arrow_length * dx,
              cy + settings_.tgmppi_debug_arrow_length * dy, kZ));
          dirs.colors.push_back(col);
          dirs.colors.push_back(col);
        }
      }
      arr.markers.push_back(dirs);
    }
  }

  visualization_msgs::msg::Marker txt;
  txt.header.frame_id = frame;
  txt.header.stamp = stamp;
  txt.ns = "tgmppi_state";
  txt.id = 0;
  txt.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  txt.action = visualization_msgs::msg::Marker::ADD;
  txt.pose.position = mk_point(
    state_.pose.pose.position.x, state_.pose.pose.position.y, 0.7);
  txt.pose.orientation.w = 1.0;
  txt.scale.z = 0.18;
  txt.color = mk_color(1.0f, 1.0f, 1.0f);
  char buf[192];
  if (flow_field_.ready()) {
    const unsigned int safe_modes = static_cast<unsigned int>(std::count(
        ancillary_mode_valid_.begin(), ancillary_mode_valid_.end(), true));
    const unsigned int assigned = std::accumulate(
      ancillary_mode_samples_.begin(), ancillary_mode_samples_.end(), 0u);
    std::snprintf(
      buf, sizeof(buf),
      "TGMPPI %s: body %zu, membrane %zu, pods %zu, safe %u, biased %u, wait %u, sel %d age %u sw %u (%.2f ms)",
      tgmppi_assist_active_ ? "ASSIST" : "NORMAL",
      flow_field_.bodyCellCount(), flow_field_.membraneCellCount(),
      flow_field_.pseudopods().size(), safe_modes, assigned, flow_wait_samples_,
      selected_mode_key_, selected_mode_age_, mode_switch_count_,
      flow_field_.buildMs());
  } else {
    std::snprintf(buf, sizeof(buf), "FLOW: no field");
  }
  txt.text = buf;
  arr.markers.push_back(txt);

  tgmppi_debug_pub_->publish(arr);
}

bool Optimizer::isHolonomic() const {return motion_model_->isHolonomic();}

void Optimizer::applyControlSequenceConstraints()
{
  auto & s = settings_;

  if (isHolonomic()) {
    control_sequence_.vy = xt::clip(control_sequence_.vy, -s.constraints.vy, s.constraints.vy);
  }

  control_sequence_.vx = xt::clip(control_sequence_.vx, s.constraints.vx_min, s.constraints.vx_max);
  control_sequence_.wz = xt::clip(control_sequence_.wz, -s.constraints.wz, s.constraints.wz);

  motion_model_->applyConstraints(control_sequence_);
}

void Optimizer::updateStateVelocities(
  models::State & state) const
{
  updateInitialStateVelocities(state);
  propagateStateVelocitiesFromInitials(state);
}

void Optimizer::updateInitialStateVelocities(
  models::State & state) const
{
  xt::noalias(xt::view(state.vx, xt::all(), 0)) = state.speed.linear.x;
  xt::noalias(xt::view(state.wz, xt::all(), 0)) = state.speed.angular.z;

  if (isHolonomic()) {
    xt::noalias(xt::view(state.vy, xt::all(), 0)) = state.speed.linear.y;
  }
}

void Optimizer::propagateStateVelocitiesFromInitials(
  models::State & state) const
{
  motion_model_->predict(state);
}

void Optimizer::integrateStateVelocities(
  xt::xtensor<float, 2> & trajectory,
  const xt::xtensor<float, 2> & sequence) const
{
  float initial_yaw = tf2::getYaw(state_.pose.pose.orientation);

  const auto vx = xt::view(sequence, xt::all(), 0);
  const auto vy = xt::view(sequence, xt::all(), 2);
  const auto wz = xt::view(sequence, xt::all(), 1);

  auto traj_x = xt::view(trajectory, xt::all(), 0);
  auto traj_y = xt::view(trajectory, xt::all(), 1);
  auto traj_yaws = xt::view(trajectory, xt::all(), 2);

  xt::noalias(traj_yaws) = cumsum_1d(wz * settings_.model_dt) + initial_yaw;

  auto && yaw_cos = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 1>::from_shape(traj_yaws.shape());

  const auto yaw_offseted = xt::view(traj_yaws, xt::range(1, _));

  xt::noalias(xt::view(yaw_cos, 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::range(1, _))) = xt::cos(yaw_offseted);
  xt::noalias(xt::view(yaw_sin, xt::range(1, _))) = xt::sin(yaw_offseted);

  auto && dx = xt::eval(vx * yaw_cos);
  auto && dy = xt::eval(vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - vy * yaw_sin;
    dy = dy + vy * yaw_cos;
  }

  xt::noalias(traj_x) = state_.pose.pose.position.x + cumsum_1d(dx * settings_.model_dt);
  xt::noalias(traj_y) = state_.pose.pose.position.y + cumsum_1d(dy * settings_.model_dt);
}

void Optimizer::integrateStateVelocities(
  models::Trajectories & trajectories,
  const models::State & state) const
{
  const float initial_yaw = tf2::getYaw(state.pose.pose.orientation);

  xt::noalias(trajectories.yaws) =
    cumsum_2d(state.wz * settings_.model_dt, 1) + initial_yaw;

  const auto yaws_cutted = xt::view(trajectories.yaws, xt::all(), xt::range(0, -1));

  auto && yaw_cos = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  auto && yaw_sin = xt::xtensor<float, 2>::from_shape(trajectories.yaws.shape());
  xt::noalias(xt::view(yaw_cos, xt::all(), 0)) = cosf(initial_yaw);
  xt::noalias(xt::view(yaw_sin, xt::all(), 0)) = sinf(initial_yaw);
  xt::noalias(xt::view(yaw_cos, xt::all(), xt::range(1, _))) = xt::cos(yaws_cutted);
  xt::noalias(xt::view(yaw_sin, xt::all(), xt::range(1, _))) = xt::sin(yaws_cutted);

  auto && dx = xt::eval(state.vx * yaw_cos);
  auto && dy = xt::eval(state.vx * yaw_sin);

  if (isHolonomic()) {
    dx = dx - state.vy * yaw_sin;
    dy = dy + state.vy * yaw_cos;
  }

  xt::noalias(trajectories.x) = state.pose.pose.position.x +
    cumsum_2d(dx * settings_.model_dt, 1);
  xt::noalias(trajectories.y) = state.pose.pose.position.y +
    cumsum_2d(dy * settings_.model_dt, 1);
}

xt::xtensor<float, 2> Optimizer::getOptimizedTrajectory()
{
  auto && sequence =
    xt::xtensor<float, 2>::from_shape({settings_.time_steps, isHolonomic() ? 3u : 2u});
  auto && trajectories = xt::xtensor<float, 2>::from_shape({settings_.time_steps, 3});

  xt::noalias(xt::view(sequence, xt::all(), 0)) = control_sequence_.vx;
  xt::noalias(xt::view(sequence, xt::all(), 1)) = control_sequence_.wz;

  if (isHolonomic()) {
    xt::noalias(xt::view(sequence, xt::all(), 2)) = control_sequence_.vy;
  }

  integrateStateVelocities(trajectories, sequence);
  return std::move(trajectories);
}

void Optimizer::updateControlSequence()
{
  auto & s = settings_;
  // 2026-09-16 diagnostics: rows already corrupt before the control-cost term
  // (critics / priors) vs rows the term itself makes corrupt.
  static unsigned int control_cost_logs = 0u;
  const auto implausible = [](float c) {
      return utils::isBadFloat(c) || c < -1.0e3f || c > 1.0e12f;
    };
  std::vector<uint8_t> corrupt_before;
  if (control_cost_logs < 10u) {
    corrupt_before.resize(s.batch_size);
    for (unsigned int r = 0; r < s.batch_size; ++r) {
      corrupt_before[r] = implausible(costs_(r)) ? 1u : 0u;
    }
  }

  auto bounded_noises_vx = state_.cvx - control_sequence_.vx;
  auto bounded_noises_wz = state_.cwz - control_sequence_.wz;
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.vx, 2) * xt::sum(
    xt::view(control_sequence_.vx, xt::newaxis(), xt::all()) * bounded_noises_vx, 1, immediate);
  xt::noalias(costs_) +=
    s.gamma / powf(s.sampling_std.wz, 2) * xt::sum(
    xt::view(control_sequence_.wz, xt::newaxis(), xt::all()) * bounded_noises_wz, 1, immediate);

  if (isHolonomic()) {
    auto bounded_noises_vy = state_.cvy - control_sequence_.vy;
    xt::noalias(costs_) +=
      s.gamma / powf(s.sampling_std.vy, 2) * xt::sum(
      xt::view(control_sequence_.vy, xt::newaxis(), xt::all()) * bounded_noises_vy,
      1, immediate);
  }

  if (!corrupt_before.empty()) {
    // NaN-propagating running max (std::max would silently skip a NaN).
    const auto track = [](float & m, float x) {
        x = std::fabs(x);
        if (!(x <= m)) {m = x;}
      };
    for (unsigned int r = 0; r < s.batch_size; ++r) {
      if (corrupt_before[r] || !implausible(costs_(r))) {continue;}
      float max_cvx = 0.0f, max_cwz = 0.0f, max_u = 0.0f, max_w = 0.0f;
      for (unsigned int t = 0; t < s.time_steps; ++t) {
        track(max_cvx, state_.cvx(r, t));
        track(max_cwz, state_.cwz(r, t));
        track(max_u, control_sequence_.vx(t));
        track(max_w, control_sequence_.wz(t));
      }
      ++control_cost_logs;
      RCLCPP_WARN(
        logger_,
        "[TGMPPI diag] control-cost term made row %u corrupt: cost %g, max|cvx| %g "
        "max|cwz| %g, max|u_vx| %g max|u_wz| %g",
        r, costs_(r), max_cvx, max_cwz, max_u, max_w);
      break;
    }
  }

  // 2026-09-16 quarantine: a row whose cost is non-finite / implausible, or
  // whose sampled controls are non-finite, must never reach a softmax -- one
  // such row turns the weighted mean into NaN, which then feeds every later
  // cycle (bag tgmppi_dyn_20260916_002534: NaN commands from the second goal
  // on, robot frozen). Such rows get a cost just above the worst sane row and
  // the current nominal controls: zero weight, and harmless if ever averaged.
  {
    static unsigned int quarantine_logs = 0u;
    static unsigned int quarantine_cycles = 0u;
    static std::size_t quarantined_rows = 0u;
    std::vector<unsigned int> bad_rows;
    float worst_sane = 0.0f;
    for (unsigned int r = 0; r < s.batch_size; ++r) {
      // Finite-but-impossible controls count too: a space-time route once fed
      // |wz| ~ 63 rad/s (wz_max 1.9) into the batch. The bounds are generous
      // multiples of the sampling spread so ordinary MPPI exploration is kept.
      const float cvx_limit = s.constraints.vx_max + 5.0f * s.sampling_std.vx;
      const float cwz_limit = s.constraints.wz + 5.0f * s.sampling_std.wz;
      bool bad = implausible(costs_(r));
      for (unsigned int t = 0; t < s.time_steps && !bad; ++t) {
        bad = utils::isBadFloat(state_.cvx(r, t)) || utils::isBadFloat(state_.cwz(r, t)) ||
          std::fabs(state_.cvx(r, t)) > cvx_limit || std::fabs(state_.cwz(r, t)) > cwz_limit;
      }
      if (bad) {
        bad_rows.push_back(r);
      } else {
        worst_sane = std::max(worst_sane, costs_(r));
      }
    }
    if (!bad_rows.empty()) {
      if (quarantine_logs++ < 20u) {
        RCLCPP_WARN(
          logger_, "[TGMPPI diag] quarantining %zu row(s); first row %u cost %g",
          bad_rows.size(), bad_rows.front(), costs_(bad_rows.front()));
      }
      for (const unsigned int r : bad_rows) {
        costs_(r) = worst_sane + 1.0e4f;
        for (unsigned int t = 0; t < s.time_steps; ++t) {
          const float u = control_sequence_.vx(t);
          const float w = control_sequence_.wz(t);
          state_.cvx(r, t) = utils::isBadFloat(u) ? 0.0f : u;
          state_.cwz(r, t) = utils::isBadFloat(w) ? 0.0f : w;
          if (isHolonomic()) {
            const float vy = control_sequence_.vy(t);
            state_.cvy(r, t) = utils::isBadFloat(vy) ? 0.0f : vy;
          }
        }
      }
      quarantined_rows += bad_rows.size();
    }
    if (++quarantine_cycles >= 100u) {
      if (quarantined_rows > 0u) {
        RCLCPP_WARN(
          logger_, "[TGMPPI diag] quarantined %zu row(s) in the last 100 cycles",
          quarantined_rows);
      }
      quarantine_cycles = 0u;
      quarantined_rows = 0u;
    }
  }

  if (s.tgmppi_grouped_update && !s.tgmppi_shadow_mode && s.tgmppi_bias_enabled) {
    // amoeba_sandbox grouped_sampling.py port, V3 ("uncorrected within-
    // mode") only -- see tgmppi_grouped_update's docstring in
    // optimizer_settings.hpp. Reconstruct this cycle's row groups from the
    // same per-pseudopod bookkeeping applyFlowBias() just filled in: each
    // active pseudopod's contiguous block (rows [0, pod_budget) in
    // sub-blocks), the "wait" block if offered, then everything else (the
    // unbiased/nominal remainder) -- always present, covering the whole
    // batch by itself when bias produced no groups this cycle, which is
    // exactly the single-shared-softmax case below.
    struct Group {unsigned int start; unsigned int count; int key;};
    std::vector<Group> groups;
    unsigned int covered_end = 0;
    for (std::size_t m = 0; m < ancillary_mode_valid_.size(); ++m) {
      if (ancillary_mode_valid_[m] && ancillary_mode_samples_[m] > 0) {
        groups.push_back(
          {ancillary_mode_row_start_[m], ancillary_mode_samples_[m], ancillary_mode_key_[m]});
        covered_end = std::max(
          covered_end, ancillary_mode_row_start_[m] + ancillary_mode_samples_[m]);
      }
    }
    if (flow_wait_samples_ > 0) {
      groups.push_back({covered_end, flow_wait_samples_, kWaitModeKey});
      covered_end += flow_wait_samples_;
    }
    if (covered_end < s.batch_size) {
      groups.push_back({covered_end, s.batch_size - covered_end, kFallbackModeKey});
    }

    // 2026-09-15 guard: never index outside the batch (corrupt costs are
    // handled row-by-row by the quarantine above).
    {
      static unsigned int range_logs = 0u;
      std::vector<Group> sane;
      sane.reserve(groups.size());
      for (const auto & grp : groups) {
        if (grp.count == 0u || grp.start + grp.count > s.batch_size) {
          if (range_logs++ < 20u) {
            RCLCPP_WARN(
              logger_, "[TGMPPI] dropping out-of-range group key %d rows [%u, %u) batch %u",
              grp.key, grp.start, grp.start + grp.count, s.batch_size);
          }
          continue;
        }
        sane.push_back(grp);
      }
      if (sane.empty()) {
        sane.push_back({0u, s.batch_size, kFallbackModeKey});
      }
      groups.swap(sane);
    }

    // Periodic record of how the batch was actually split, so an allocation mode
    // can be verified from the log rather than assumed.
    {
      static unsigned int alloc_log_cycles = 0u;
      if (++alloc_log_cycles >= 200u) {
        alloc_log_cycles = 0u;
        std::string sizes;
        for (const auto & grp : groups) {
          sizes += std::to_string(grp.key) + ":" + std::to_string(grp.count) + " ";
        }
        RCLCPP_INFO(
          logger_, "[TGMPPI alloc] %s -- groups (key:rows): %s",
          s.tgmppi_group_allocation.c_str(), sizes.c_str());
      }
    }

    // Per-group local softmax + free energy (sandbox's mode_statistics()):
    // each group's weights are normalized against only its OWN rows, not
    // the whole batch.
    float best_free_energy = std::numeric_limits<float>::max();
    std::size_t best_group = 0;
    std::vector<xt::xtensor<float, 1>> group_softmax(groups.size());
    std::vector<float> group_free_energy(groups.size(), 0.0f);
    for (std::size_t g = 0; g < groups.size(); ++g) {
      auto && group_costs = xt::eval(
        xt::view(costs_, xt::range(groups[g].start, groups[g].start + groups[g].count)));
      const float cmin = xt::amin(group_costs, immediate)();
      auto && likelihood = xt::eval(xt::exp(-(group_costs - cmin) / s.temperature));
      const float likelihood_sum = xt::sum(likelihood, immediate)();
      group_softmax[g] = xt::eval(likelihood / likelihood_sum);
      const float mean_likelihood = likelihood_sum / static_cast<float>(groups[g].count);
      const float free_energy = cmin - s.temperature * std::log(mean_likelihood);
      group_free_energy[g] = free_energy;
      if (free_energy < best_free_energy) {
        best_free_energy = free_energy;
        best_group = g;
      }
    }

    // 2026-09-15 stabilizer port of the sandbox's _select_committed_mode():
    // dwell / switch margin / confirmation on top of the V3 best-group rule.
    // With the default knobs (dwell 0, confirm 1, margin 0) this is exactly
    // "best group every cycle" (ties keep the current group).
    constexpr std::size_t kNoGroup = std::numeric_limits<std::size_t>::max();
    std::size_t current_group = kNoGroup;
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (groups[g].key == selected_mode_key_) {current_group = g; break;}
    }
    std::size_t chosen_group = best_group;
    const char * reason = "best";
    const unsigned int min_dwell = static_cast<unsigned int>(std::max(0, s.tgmppi_mode_min_dwell));
    const unsigned int confirm_cycles =
      static_cast<unsigned int>(std::max(1, s.tgmppi_mode_confirm_cycles));
    if (current_group == kNoGroup) {
      chosen_group = best_group;
      reason = selected_mode_key_ == kNoModeKey ? "initial" : "current_unavailable";
    } else if (best_group == current_group) {
      chosen_group = current_group;
      reason = "best";
      pending_mode_key_ = kNoModeKey;
      pending_mode_count_ = 0u;
    } else if (selected_mode_age_ < min_dwell) {
      chosen_group = current_group;
      reason = "minimum_dwell";
    } else if (group_free_energy[best_group] + s.tgmppi_mode_switch_margin >=
      group_free_energy[current_group])
    {
      chosen_group = current_group;
      reason = "switch_margin";
      pending_mode_key_ = kNoModeKey;
      pending_mode_count_ = 0u;
    } else {
      if (pending_mode_key_ == groups[best_group].key) {
        ++pending_mode_count_;
      } else {
        pending_mode_key_ = groups[best_group].key;
        pending_mode_count_ = 1u;
      }
      if (pending_mode_count_ >= confirm_cycles) {
        chosen_group = best_group;
        reason = "confirmed_improvement";
      } else {
        chosen_group = current_group;
        reason = "awaiting_confirmation";
      }
    }
    const bool switched =
      selected_mode_key_ != kNoModeKey && groups[chosen_group].key != selected_mode_key_;
    if (switched) {
      ++mode_switch_count_;
      selected_mode_age_ = 0u;
      pending_mode_key_ = kNoModeKey;
      pending_mode_count_ = 0u;
      RCLCPP_INFO(
        logger_, "[TGMPPI] mode switch %d -> %d (%s), FE new %.3f vs old %.3f",
        selected_mode_key_, groups[chosen_group].key, reason,
        group_free_energy[chosen_group],
        current_group == kNoGroup ? 0.0f : group_free_energy[current_group]);
    } else {
      ++selected_mode_age_;
    }
    selected_mode_key_ = groups[chosen_group].key;
    (void)reason;

    // Per-mode memory for tgmppi_mode_warm_start (sandbox mode_nominals):
    // every group's own local weighted mean, keyed by group key; keys that
    // did not exist this cycle are forgotten. Skipped entirely when the
    // knob is off, so the default path does no extra work.
    if (s.tgmppi_mode_warm_start > 0.0f) {
      std::map<int, std::pair<std::vector<float>, std::vector<float>>> fresh;
      for (std::size_t g = 0; g < groups.size(); ++g) {
        const unsigned int g0 = groups[g].start;
        const unsigned int g1 = groups[g].start + groups[g].count;
        auto && sm = xt::eval(xt::view(group_softmax[g], xt::all(), xt::newaxis()));
        // 2026-09-17 ROOT CAUSE of the corrupt pod references (and, downstream,
        // the -1e19..-1e25 costs and the NaN freeze): these two were
        // `auto && x = xt::eval(xt::sum(..., immediate))`. With `immediate` the
        // sum is ALREADY an evaluated temporary container, and xt::eval() on a
        // container is the identity -- it returns a REFERENCE to that temporary
        // rather than a new object. Lifetime extension only applies when a
        // temporary binds directly to a reference, not through a function
        // returning one, so both references dangled at the end of their
        // statements. The second sum then reused the freed block, which is why
        // the stored means came back with vx == wz exactly (log 43409:
        // "warm v -1.72842 w -1.72842") or as garbage once the memory was reused
        // (-1.04e34). Owning containers make the lifetime explicit.
        const xt::xtensor<float, 1> mean_vx =
          xt::sum(xt::view(state_.cvx, xt::range(g0, g1), xt::all()) * sm, 0, immediate);
        const xt::xtensor<float, 1> mean_wz =
          xt::sum(xt::view(state_.cwz, xt::range(g0, g1), xt::all()) * sm, 0, immediate);
        fresh[groups[g].key] = {
          std::vector<float>(mean_vx.begin(), mean_vx.end()),
          std::vector<float>(mean_wz.begin(), mean_wz.end())};
      }
      mode_nominals_ = std::move(fresh);
    }

    const auto & chosen = groups[chosen_group];
    const unsigned int row0 = chosen.start;
    const unsigned int row1 = chosen.start + chosen.count;
    auto && softmax_extended = xt::eval(
      xt::view(group_softmax[chosen_group], xt::all(), xt::newaxis()));
    xt::noalias(control_sequence_.vx) = xt::sum(
      xt::view(state_.cvx, xt::range(row0, row1), xt::all()) * softmax_extended, 0, immediate);
    xt::noalias(control_sequence_.wz) = xt::sum(
      xt::view(state_.cwz, xt::range(row0, row1), xt::all()) * softmax_extended, 0, immediate);
    if (isHolonomic()) {
      xt::noalias(control_sequence_.vy) = xt::sum(
        xt::view(state_.cvy, xt::range(row0, row1), xt::all()) * softmax_extended, 0, immediate);
    }
    applyControlSequenceConstraints();
    return;
  }

  auto && costs_normalized = costs_ - xt::amin(costs_, immediate);
  auto && exponents = xt::eval(xt::exp(-1 / settings_.temperature * costs_normalized));
  auto && softmaxes = xt::eval(exponents / xt::sum(exponents, immediate));
  auto && softmaxes_extened = xt::eval(xt::view(softmaxes, xt::all(), xt::newaxis()));

  xt::noalias(control_sequence_.vx) = xt::sum(state_.cvx * softmaxes_extened, 0, immediate);
  xt::noalias(control_sequence_.wz) = xt::sum(state_.cwz * softmaxes_extened, 0, immediate);
  if (isHolonomic()) {
    xt::noalias(control_sequence_.vy) = xt::sum(state_.cvy * softmaxes_extened, 0, immediate);
  }

  applyControlSequenceConstraints();
}

geometry_msgs::msg::TwistStamped Optimizer::getControlFromSequenceAsTwist(
  const builtin_interfaces::msg::Time & stamp)
{
  unsigned int offset = settings_.shift_control_sequence ? 1 : 0;

  auto vx = control_sequence_.vx(offset);
  auto wz = control_sequence_.wz(offset);

  if (isHolonomic()) {
    auto vy = control_sequence_.vy(offset);
    return utils::toTwistStamped(vx, vy, wz, stamp, costmap_ros_->getBaseFrameID());
  }

  return utils::toTwistStamped(vx, wz, stamp, costmap_ros_->getBaseFrameID());
}

void Optimizer::setMotionModel(const std::string & model)
{
  if (model == "DiffDrive") {
    motion_model_ = std::make_shared<DiffDriveMotionModel>();
  } else if (model == "Omni") {
    motion_model_ = std::make_shared<OmniMotionModel>();
  } else if (model == "Ackermann") {
    motion_model_ = std::make_shared<AckermannMotionModel>(parameters_handler_, name_);
  } else {
    throw std::runtime_error(
            std::string(
              "Model " + model + " is not valid! Valid options are DiffDrive, Omni, "
              "or Ackermann"));
  }
}

void Optimizer::setSpeedLimit(double speed_limit, bool percentage)
{
  auto & s = settings_;
  if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    s.constraints.vx_max = s.base_constraints.vx_max;
    s.constraints.vx_min = s.base_constraints.vx_min;
    s.constraints.vy = s.base_constraints.vy;
    s.constraints.wz = s.base_constraints.wz;
  } else {
    if (percentage) {
      // Speed limit is expressed in % from maximum speed of robot
      double ratio = speed_limit / 100.0;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    } else {
      // Speed limit is expressed in absolute value
      double ratio = speed_limit / s.base_constraints.vx_max;
      s.constraints.vx_max = s.base_constraints.vx_max * ratio;
      s.constraints.vx_min = s.base_constraints.vx_min * ratio;
      s.constraints.vy = s.base_constraints.vy * ratio;
      s.constraints.wz = s.base_constraints.wz * ratio;
    }
  }
}

models::Trajectories & Optimizer::getGeneratedTrajectories()
{
  return generated_trajectories_;
}

}  // namespace tgmppi

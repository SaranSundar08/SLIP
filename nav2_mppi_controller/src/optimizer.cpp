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

#include "nav2_mppi_controller/optimizer.hpp"

#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>
#include <xtensor/xmath.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xnoalias.hpp>

#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_costmap_2d/cost_values.hpp"

namespace mppi
{

using namespace xt::placeholders;  // NOLINT
using xt::evaluation_strategy::immediate;

void Optimizer::initialize(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros,
  ParametersHandler * param_handler)
{
  parent_ = parent;
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  parameters_handler_ = param_handler;

  auto node = parent_.lock();
  logger_ = node->get_logger();

  getParams();

  if (require_obstacle_tracking_) {
    obstacles_.assign(obstacle_topics_.size(), tgmppi::SpaceTimeObstacle{});
    obstacle_stamps_.assign(obstacle_topics_.size(), rclcpp::Time(0, 0, RCL_ROS_TIME));
    obstacle_received_.assign(obstacle_topics_.size(), false);
    for (std::size_t i = 0; i < obstacle_topics_.size(); ++i) {
      obstacle_subs_.push_back(
        node->create_subscription<nav_msgs::msg::Odometry>(
          obstacle_topics_[i], rclcpp::SensorDataQoS(),
          [this, i](const nav_msgs::msg::Odometry & msg) {obstacleCallback(i, msg);}));
    }
  }

  critic_manager_.on_configure(parent_, name_, costmap_ros_, parameters_handler_);
  noise_generator_.initialize(settings_, isHolonomic(), name_, parameters_handler_);

  reset();
}

void Optimizer::shutdown()
{
  obstacle_subs_.clear();
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
  getParam(
    obstacle_topics_, "obstacle_topics", std::vector<std::string>{}, ParameterType::Static);
  getParam(obstacle_radius_, "obstacle_radius", 0.25f, ParameterType::Static);
  getParam(obstacle_timeout_, "obstacle_timeout", 0.5f, ParameterType::Static);
  getParam(require_obstacle_tracking_, "require_obstacle_tracking", false, ParameterType::Static);
  if (tgmppi::badObservationNumber(obstacle_timeout_) || obstacle_timeout_ < 0.05f) {
    RCLCPP_WARN(logger_, "[MPPI] invalid obstacle timeout; using 0.5 s");
    obstacle_timeout_ = 0.5f;
  }

  // --- SLIP MPPI variant selection + per-variant knobs ---
  std::string variant_name;
  getParam(variant_name, "mppi_variant", std::string("vanilla"));
  getParam(s.lognormal_sigma, "lognormal_sigma", 0.1f);
  getParam(s.lowpass_cutoff_hz, "lowpass_cutoff_hz", 2.0f);
  getParam(s.lowpass_order, "lowpass_order", 1);
  getParam(s.bias_strength, "bias_strength", 0.5f);
  getParam(s.bias_lookahead_dist, "bias_lookahead_dist", 0.6f);
  getParam(s.bias_gain, "bias_gain", 1.5f);
  std::string ancillary_name;
  getParam(ancillary_name, "ancillary_type", std::string("pursuit"));
  getParam(s.brake_gain, "brake_gain", 1.5f);
  getParam(s.brake_scan_dist, "brake_scan_dist", 0.3f);
  s.ancillary_type = (ancillary_name == "braking") ?
    models::AncillaryType::BRAKING : models::AncillaryType::PURSUIT;
  if (variant_name == "log") {
    s.variant = models::MPPIVariant::LOG;
  } else if (variant_name == "lowpass") {
    s.variant = models::MPPIVariant::LOWPASS;
  } else if (variant_name == "biased") {
    s.variant = models::MPPIVariant::BIASED;
  } else {
    s.variant = models::MPPIVariant::VANILLA;
    variant_name = "vanilla";
  }
  if (s.variant == models::MPPIVariant::BIASED) {
    RCLCPP_INFO(
      logger_, "[SLIP] MPPI variant = 'biased' (ancillary = '%s')", ancillary_name.c_str());
  } else {
    RCLCPP_INFO(logger_, "[SLIP] MPPI variant = '%s'", variant_name.c_str());
  }

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

  costs_ = xt::zeros<float>({settings_.batch_size});
  generated_trajectories_.reset(settings_.batch_size, settings_.time_steps);
  final_safety_state_.reset(1u, settings_.time_steps);
  final_safety_trajectory_.reset(1u, settings_.time_steps);

  noise_generator_.reset(settings_, isHolonomic());
  RCLCPP_INFO(logger_, "Optimizer reset");
}

geometry_msgs::msg::TwistStamped Optimizer::evalControl(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const geometry_msgs::msg::Twist & robot_speed,
  const nav_msgs::msg::Path & plan, nav2_core::GoalChecker * goal_checker)
{
  prepare(robot_pose, robot_speed, plan, goal_checker);

  if (obstacle_tracking_fault_) {
    stopForDynamicSafety("moving-obstacle tracking missing or stale");
    return getControlFromSequenceAsTwist(plan.header.stamp);
  }

  do {
    optimize();
  } while (fallback(critics_data_.fail_flag));

  utils::savitskyGolayFilter(control_sequence_, control_history_, settings_);
  bool sequence_valid = true;
  for (unsigned int t = 0; t < settings_.time_steps; ++t) {
    sequence_valid = sequence_valid &&
      !tgmppi::badObservationNumber(control_sequence_.vx(t)) &&
      !tgmppi::badObservationNumber(control_sequence_.wz(t)) &&
      (!isHolonomic() || !tgmppi::badObservationNumber(control_sequence_.vy(t)));
  }
  for (const auto & h : control_history_) {
    sequence_valid = sequence_valid && !tgmppi::badObservationNumber(h.vx) &&
      !tgmppi::badObservationNumber(h.vy) && !tgmppi::badObservationNumber(h.wz);
  }
  if (!sequence_valid) {
    stopForDynamicSafety("non-finite control sequence");
  }
  if (!finalSequenceIsDynamicallySafe()) {
    stopForDynamicSafety("smoothed command intersects a predicted obstacle");
  }
  auto control = getControlFromSequenceAsTwist(plan.header.stamp);

  if (settings_.shift_control_sequence) {
    shiftControlSequence();
  }

  return control;
}

void Optimizer::optimize()
{
  for (size_t i = 0; i < settings_.iteration_count; ++i) {
    generateNoisedTrajectories();
    critic_manager_.evalTrajectoriesScores(critics_data_);
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
  snapshotTrackedObstacles();
  critics_data_.tracked_obstacles =
    tracked_obstacles_snapshot_.empty() ? nullptr : &tracked_obstacles_snapshot_;
  critics_data_.dynamic_obstacle_params.reset();
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
  applyBiasedControls();  // SLIP Biased-MPPI: mean shift before motion-model predict
  updateStateVelocities(state_);
  integrateStateVelocities(generated_trajectories_, state_);
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

void Optimizer::computeAncillaryControl()
{
  // Biased-MPPI (SLIP): dispatch to the selected ancillary controller. Its
  // (constant over the horizon) command becomes the shifted sampling mean for
  // the biased fraction of samples; everything downstream is unchanged. This is
  // the experiment axis -- PURSUIT is geometric guidance (follow the plan),
  // BRAKING is reactive safety (avoid obstacles from the costmap).
  ancillary_vx_ = 0.0f;
  ancillary_wz_ = 0.0f;
  switch (settings_.ancillary_type) {
    case models::AncillaryType::BRAKING:
      computeBrakingAncillary();
      break;
    case models::AncillaryType::PURSUIT:
    default:
      computePursuitAncillary();
      break;
  }
}

void Optimizer::computePursuitAncillary()
{
  // Pursuit-to-path P-controller: steer toward a lookahead point on the
  // reference path. Biases samples toward FOLLOWING THE PLAN.
  const auto & s = settings_;

  const std::size_t path_size = path_.x.shape(0);
  if (path_size < 2) {
    return;  // no usable path -> no bias (biased rows fall back to nominal)
  }

  const float rx = static_cast<float>(state_.pose.pose.position.x);
  const float ry = static_cast<float>(state_.pose.pose.position.y);
  const float ryaw = tf2::getYaw(state_.pose.pose.orientation);

  // First path point at least bias_lookahead_dist away; else the last point.
  std::size_t idx = path_size - 1;
  const float ld2 = s.bias_lookahead_dist * s.bias_lookahead_dist;
  for (std::size_t i = 0; i < path_size; ++i) {
    const float dx = path_.x(i) - rx;
    const float dy = path_.y(i) - ry;
    if (dx * dx + dy * dy >= ld2) {
      idx = i;
      break;
    }
  }

  const float heading_err = static_cast<float>(
    angles::shortest_angular_distance(ryaw, std::atan2(path_.y(idx) - ry, path_.x(idx) - rx)));

  // Turn proportional to heading error; slow down when badly misaligned.
  ancillary_wz_ = std::clamp(s.bias_gain * heading_err, -s.constraints.wz, s.constraints.wz);
  ancillary_vx_ = std::clamp(
    s.constraints.vx_max * std::max(0.0f, std::cos(heading_err)),
    s.constraints.vx_min, s.constraints.vx_max);
}

void Optimizer::computeBrakingAncillary()
{
  // CBF/braking reactive controller (minimal CBF: a 1-D barrier on the costmap
  // proximity proxy). Biases samples toward SAFE MOTION:
  //   - forward speed scales with the safety margin  h = INSCRIBED - cost
  //     (full speed in open space, ~zero at an obstacle boundary),
  //   - yaw steers DOWN the cost gradient (away from the nearest obstacle).
  // Reads only the costmap + robot pose -- the same "world model" the critics
  // already use. Naturally commands stop near obstacles/goal (no pursuit swirl).
  const auto & s = settings_;
  if (costmap_ == nullptr) {
    return;  // no costmap -> no bias
  }

  const float rx = static_cast<float>(state_.pose.pose.position.x);
  const float ry = static_cast<float>(state_.pose.pose.position.y);
  const float ryaw = tf2::getYaw(state_.pose.pose.orientation);

  // Sample the costmap at a world point; unknown/off-map treated as free (0).
  auto cost_at = [this](float wx, float wy) -> float {
      unsigned int mx, my;
      if (!costmap_->worldToMap(wx, wy, mx, my)) {
        return 0.0f;  // off-map -> treat as open
      }
      const unsigned char c = costmap_->getCost(mx, my);
      if (c == nav2_costmap_2d::NO_INFORMATION) {
        return 0.0f;  // unknown -> keep moving rather than freeze
      }
      return static_cast<float>(c);
    };

  // Safety margin at the robot: 1 in open space, 0 at the inscribed boundary.
  const float kInscribed =
    static_cast<float>(nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE);
  const float safe = std::clamp((kInscribed - cost_at(rx, ry)) / kInscribed, 0.0f, 1.0f);
  ancillary_vx_ = std::clamp(safe * s.constraints.vx_max, 0.0f, s.constraints.vx_max);

  // Cost gradient via central differences; points TOWARD higher cost (obstacle).
  const float d = std::max(s.brake_scan_dist, static_cast<float>(costmap_->getResolution()));
  const float gx = cost_at(rx + d, ry) - cost_at(rx - d, ry);
  const float gy = cost_at(rx, ry + d) - cost_at(rx, ry - d);

  // Steer down-gradient (away from the obstacle); flat gradient -> go straight.
  if (gx * gx + gy * gy > 1e-6f) {
    const float heading_err = static_cast<float>(
      angles::shortest_angular_distance(ryaw, std::atan2(-gy, -gx)));
    ancillary_wz_ = std::clamp(s.brake_gain * heading_err, -s.constraints.wz, s.constraints.wz);
  }
}

void Optimizer::applyBiasedControls()
{
  // Biased-MPPI mean shift (SLIP): re-center the first n_biased samples onto the
  // ancillary command, REUSING their existing noise:
  //   cvx_biased = (cvx - nominal) + u_anc = u_anc + eps
  // The nominal control_sequence_ is intentionally left untouched so the gamma
  // control-cost term in updateControlSequence() already provides the (Tier-1)
  // importance-weight correction that keeps the estimator consistent.
  if (settings_.variant != models::MPPIVariant::BIASED) {
    return;
  }

  computeAncillaryControl();

  const auto & s = settings_;
  const float frac = std::clamp(s.bias_strength, 0.0f, 1.0f);
  const unsigned int n_biased =
    std::min(s.batch_size, static_cast<unsigned int>(frac * static_cast<float>(s.batch_size)));
  if (n_biased == 0) {
    return;
  }

  // Column mask: 1.0 for the first n_biased samples, 0.0 otherwise. Applying the
  // shift as (mask * shift) avoids row-slice views and stays fully vectorized;
  // mask is [batch,1] so it broadcasts against the [time_steps] shift to [batch,time].
  xt::xtensor<float, 2> mask = xt::zeros<float>({s.batch_size, 1u});
  for (unsigned int i = 0; i < n_biased; ++i) {
    mask(i, 0) = 1.0f;
  }

  // Shift so biased rows become u_anc + eps (nominal control_sequence_ untouched).
  const xt::xtensor<float, 1> shift_vx = ancillary_vx_ - control_sequence_.vx;
  const xt::xtensor<float, 1> shift_wz = ancillary_wz_ - control_sequence_.wz;
  state_.cvx += mask * shift_vx;
  state_.cwz += mask * shift_wz;

  if (isHolonomic()) {
    // No lateral ancillary; bias vy toward zero (straighten).
    const xt::xtensor<float, 1> shift_vy = -control_sequence_.vy;
    state_.cvy += mask * shift_vy;
  }
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

}  // namespace mppi

// Copyright 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0 (the "License");

#include "nav2_mppi_controller/optimizer.hpp"

#include <algorithm>
#include <mutex>

#include <xtensor/xnoalias.hpp>

namespace mppi
{

void Optimizer::obstacleCallback(std::size_t index, const nav_msgs::msg::Odometry & msg)
{
  auto node = parent_.lock();
  if (!node || !costmap_ros_ || msg.header.frame_id.empty() ||
    (msg.header.stamp.sec == 0 && msg.header.stamp.nanosec == 0))
  {
    return;
  }
  const rclcpp::Time stamp(msg.header.stamp, RCL_ROS_TIME);
  const double age = (node->now() - stamp).seconds();
  if (tgmppi::badObservationNumber(age) || age < -0.1 || age > obstacle_timeout_) {
    return;
  }

  const auto & target_frame = costmap_ros_->getGlobalFrameID();
  geometry_msgs::msg::TransformStamped transform;
  const geometry_msgs::msg::TransformStamped * transform_ptr = nullptr;
  if (msg.header.frame_id != target_frame) {
    try {
      transform = costmap_ros_->getTfBuffer()->lookupTransform(
        target_frame, msg.header.frame_id, stamp);
      transform_ptr = &transform;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        logger_, *node->get_clock(), 5000,
        "[MPPI] ignoring obstacle odometry: cannot transform %s to %s: %s",
        msg.header.frame_id.c_str(), target_frame.c_str(), ex.what());
      return;
    }
  }
  const auto observation = tgmppi::obstacleFromOdometry(
    msg, target_frame, transform_ptr, obstacle_radius_);
  if (!observation) {return;}

  std::lock_guard<std::mutex> lock(obstacles_mutex_);
  if (index >= obstacles_.size()) {return;}
  if (obstacle_received_[index] && stamp <= obstacle_stamps_[index]) {return;}
  obstacles_[index] = *observation;
  obstacle_stamps_[index] = stamp;
  obstacle_received_[index] = true;
}

void Optimizer::snapshotTrackedObstacles()
{
  tracked_obstacles_snapshot_.clear();
  obstacle_tracking_fault_ = false;
  auto node = parent_.lock();
  if (!node) {return;}
  const auto now = node->now();
  std::size_t stale_count = 0u;
  std::lock_guard<std::mutex> lock(obstacles_mutex_);
  for (std::size_t i = 0; i < obstacles_.size(); ++i) {
    if (!obstacle_received_[i]) {continue;}
    const double age = (now - obstacle_stamps_[i]).seconds();
    if (tgmppi::badObservationNumber(age) || age < -0.1 || age > obstacle_timeout_) {
      ++stale_count;
      continue;
    }
    auto obstacle = obstacles_[i];
    const float elapsed = static_cast<float>(std::max(0.0, age));
    obstacle.x += obstacle.vx * elapsed;
    obstacle.y += obstacle.vy * elapsed;
    tracked_obstacles_snapshot_.push_back(obstacle);
  }
  obstacle_tracking_fault_ = stale_count != 0u ||
    (require_obstacle_tracking_ && tracked_obstacles_snapshot_.empty());
}

bool Optimizer::finalSequenceIsDynamicallySafe()
{
  if (!critics_data_.dynamic_obstacle_params || tracked_obstacles_snapshot_.empty()) {
    return true;
  }
  final_safety_state_.pose = state_.pose;
  final_safety_state_.speed = state_.speed;
  xt::noalias(xt::view(final_safety_state_.cvx, 0, xt::all())) = control_sequence_.vx;
  xt::noalias(xt::view(final_safety_state_.cwz, 0, xt::all())) = control_sequence_.wz;
  if (isHolonomic()) {
    xt::noalias(xt::view(final_safety_state_.cvy, 0, xt::all())) = control_sequence_.vy;
  }
  updateStateVelocities(final_safety_state_);
  integrateStateVelocities(final_safety_trajectory_, final_safety_state_);
  auto params = *critics_data_.dynamic_obstacle_params;
  params.point_step = 1u;
  const auto result = tgmppi::scoreRolloutAgainstPredictions(
    final_safety_trajectory_.x.data(), final_safety_trajectory_.y.data(),
    final_safety_trajectory_.yaws.data(), settings_.time_steps,
    tracked_obstacles_snapshot_, params);
  return !result.collides;
}

void Optimizer::stopForDynamicSafety(const char * reason)
{
  RCLCPP_WARN_THROTTLE(
    logger_, *parent_.lock()->get_clock(), 5000,
    "[MPPI safety] stopping: %s", reason);
  control_sequence_.reset(settings_.time_steps);
  for (auto & h : control_history_) {
    h = {0.0, 0.0, 0.0};
  }
}

}  // namespace mppi

// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
// Copyright (c) 2023 Open Navigation LLC
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

#include "nav2_tgmppi_controller/critics/path_angle_critic.hpp"

#include <math.h>
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#endif

namespace tgmppi::critics
{

void PathAngleCritic::initialize()
{
  auto getParentParam = parameters_handler_->getParamGetter(parent_name_);
  float vx_min;
  getParentParam(vx_min, "vx_min", -0.35);
  if (fabs(vx_min) < 1e-6) {  // zero
    reversing_allowed_ = false;
  } else if (vx_min < 0.0) {   // reversing possible
    reversing_allowed_ = true;
  }

  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(offset_from_furthest_, "offset_from_furthest", 4);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 2.0);
  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 0.5);
  getParam(
    max_angle_to_furthest_,
    "max_angle_to_furthest", 1.2);
  getParam(
    forward_preference_,
    "forward_preference", true);

  if (!reversing_allowed_) {
    forward_preference_ = true;
  }

  RCLCPP_INFO(
    logger_,
    "PathAngleCritic instantiated with %d power and %f weight. Reversing %s",
    power_, weight_, reversing_allowed_ ? "allowed." : "not allowed.");

#ifdef TGMPPI_WITH_CUDA
  gpu_critics_.initialize();
#endif
}

void PathAngleCritic::score(CriticData & data)
{
  using xt::evaluation_strategy::immediate;
  if (!enabled_) {
    return;
  }

  if (utils::withinPositionGoalTolerance(threshold_to_consider_, data.state.pose.pose, data.path)) {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);

  auto offseted_idx = std::min(
    *data.furthest_reached_path_point + offset_from_furthest_, data.path.x.shape(0) - 1);

  const float goal_x = xt::view(data.path.x, offseted_idx);
  const float goal_y = xt::view(data.path.y, offseted_idx);

  if (utils::posePointAngle(
      data.state.pose.pose, goal_x, goal_y, forward_preference_) < max_angle_to_furthest_)
  {
    return;
  }

#ifdef TGMPPI_WITH_CUDA
  // GPU path only covers forward_preference:true (no reversing-correction
  // branch) -- the active campaign config's own setting, see
  // gpu_elementwise_critics.hpp's docstring.
  if (data.compute_backend == "cuda" && gpu_critics_.ready() &&
    !(reversing_allowed_ && !forward_preference_))
  {
    auto cost_out = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    if (data.gpu_batch != nullptr) {
      const auto * rollout = data.gpu_batch;
      auto cost_gpu = gpu_critics_.pathAngleCriticDevice(
        rollout->trajX(), rollout->trajY(), rollout->trajYaws(),
        goal_x, goal_y, weight_, power_);
      data.gpu_batch->addCosts(cost_gpu);
      return;
    } else {
      gpu_critics_.pathAngleCriticScore(
        data.trajectories, goal_x, goal_y, weight_, power_, cost_out);
    }
    data.costs += cost_out;
    return;
  }
  if (data.gpu_batch) {
    data.gpu_batch->materializeHost();
    data.gpu_batch->flushCosts(data.costs);
  }
#endif

  auto yaws_between_points = xt::atan2(
    goal_y - data.trajectories.y,
    goal_x - data.trajectories.x);

  auto yaws =
    xt::abs(utils::shortest_angular_distance(data.trajectories.yaws, yaws_between_points));

  if (reversing_allowed_ && !forward_preference_) {
    const auto yaws_between_points_corrected = xt::where(
      yaws < M_PI_2, yaws_between_points, utils::normalize_angles(yaws_between_points + M_PI));
    const auto corrected_yaws = xt::abs(
      utils::shortest_angular_distance(data.trajectories.yaws, yaws_between_points_corrected));
    data.costs += xt::pow(xt::mean(corrected_yaws, {1}, immediate) * weight_, power_);
  } else {
    data.costs += xt::pow(xt::mean(yaws, {1}, immediate) * weight_, power_);
  }
}

}  // namespace tgmppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  tgmppi::critics::PathAngleCritic,
  tgmppi::critics::CriticFunction)

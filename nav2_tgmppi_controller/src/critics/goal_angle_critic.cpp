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

#include "nav2_tgmppi_controller/critics/goal_angle_critic.hpp"
#include "angles/angles.h"
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#endif
namespace tgmppi::critics
{

void GoalAngleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.0);

  getParam(threshold_to_consider_, "threshold_to_consider", 0.5);
  getParam(symmetric_yaw_tolerance_, "symmetric_yaw_tolerance", false);

  RCLCPP_INFO(
    logger_,
    "GoalAngleCritic instantiated with %d power, %f weight, %f "
    "angular threshold and symmetric_yaw_tolerance %s",
    power_, weight_, threshold_to_consider_, symmetric_yaw_tolerance_ ? "enabled" : "disabled");

#ifdef TGMPPI_WITH_CUDA
  gpu_critics_.initialize();
#endif
}

void GoalAngleCritic::score(CriticData & data)
{
  if (!enabled_ || !utils::withinPositionGoalTolerance(
      threshold_to_consider_, data.state.pose.pose, data.path))
  {
    return;
  }

  const auto goal_idx = data.path.x.shape(0) - 1;
  const float goal_yaw = data.path.yaws(goal_idx);

#ifdef TGMPPI_WITH_CUDA
  if (data.compute_backend == "cuda" && gpu_critics_.ready() && !symmetric_yaw_tolerance_) {
    auto cost_out = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    if (data.gpu_batch != nullptr) {
      const auto * rollout = data.gpu_batch;
      auto cost_gpu = gpu_critics_.goalAngleCriticDevice(
        rollout->trajYaws(), goal_yaw, weight_, power_);
      data.gpu_batch->addCosts(cost_gpu);
      return;
    } else {
      gpu_critics_.goalAngleCriticScore(data.trajectories, goal_yaw, weight_, power_, cost_out);
    }
    data.costs += cost_out;
    return;
  }
  if (data.gpu_batch) {
    data.gpu_batch->materializeHost();
    data.gpu_batch->flushCosts(data.costs);
  }
#endif

  auto angular_distances =
    xt::eval(xt::fabs(utils::shortest_angular_distance(data.trajectories.yaws, goal_yaw)));

  if (symmetric_yaw_tolerance_) {
    // For symmetric robots: use minimum distance to either goal orientation or goal + 180°
    const float symmetric_goal_yaw = angles::normalize_angle(goal_yaw + M_PI);
    auto symmetric_distances =
      xt::eval(
      xt::fabs(
        utils::shortest_angular_distance(
          data.trajectories.yaws,
          symmetric_goal_yaw)));
    angular_distances = xt::eval(xt::minimum(angular_distances, symmetric_distances));
  }

  data.costs += xt::pow(
    xt::mean(angular_distances, {1}) * weight_, power_);
}

}  // namespace tgmppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  tgmppi::critics::GoalAngleCritic,
  tgmppi::critics::CriticFunction)

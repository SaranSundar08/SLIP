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

#include "nav2_tgmppi_controller/critics/path_follow_critic.hpp"

#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#endif

namespace tgmppi::critics
{

void PathFollowCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(
    threshold_to_consider_,
    "threshold_to_consider", 1.4);
  getParam(offset_from_furthest_, "offset_from_furthest", 6);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0);

#ifdef TGMPPI_WITH_CUDA
  gpu_critics_.initialize();
#endif
}

void PathFollowCritic::score(CriticData & data)
{
  if (!enabled_ || data.path.x.shape(0) < 2 ||
    utils::withinPositionGoalTolerance(threshold_to_consider_, data.state.pose.pose, data.path))
  {
    return;
  }

  utils::setPathFurthestPointIfNotSet(data);
  utils::setPathCostsIfNotSet(data, costmap_ros_);
  const size_t path_size = data.path.x.shape(0) - 1;

  auto offseted_idx = std::min(
    *data.furthest_reached_path_point + offset_from_furthest_, path_size);

  // Drive to the first valid path point, in case of dynamic obstacles on path
  // we want to drive past it, not through it
  bool valid = false;
  while (!valid && offseted_idx < path_size - 1) {
    valid = (*data.path_pts_valid)[offseted_idx];
    if (!valid) {
      offseted_idx++;
    }
  }

  const auto path_x = data.path.x(offseted_idx);
  const auto path_y = data.path.y(offseted_idx);

#ifdef TGMPPI_WITH_CUDA
  if (data.compute_backend == "cuda" && gpu_critics_.ready()) {
    auto cost_out = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    if (data.gpu_batch != nullptr) {
      const auto * rollout = data.gpu_batch;
      auto cost_gpu = gpu_critics_.pathFollowCriticDevice(
        rollout->trajX(), rollout->trajY(), path_x, path_y, weight_, power_);
      data.gpu_batch->addCosts(cost_gpu);
      return;
    } else {
      gpu_critics_.pathFollowCriticScore(
        data.trajectories, path_x, path_y, weight_, power_, cost_out);
    }
    data.costs += cost_out;
    return;
  }
  if (data.gpu_batch) {
    data.gpu_batch->materializeHost();
    data.gpu_batch->flushCosts(data.costs);
  }
#endif

  const auto last_x = xt::view(data.trajectories.x, xt::all(), -1);
  const auto last_y = xt::view(data.trajectories.y, xt::all(), -1);

  auto dists = xt::sqrt(
    xt::pow(last_x - path_x, 2) +
    xt::pow(last_y - path_y, 2));

  data.costs += xt::pow(weight_ * std::move(dists), power_);
}

}  // namespace tgmppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  tgmppi::critics::PathFollowCritic,
  tgmppi::critics::CriticFunction)

// Copyright (c) 2026 SLIP thesis fork
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

#include "nav2_tgmppi_controller/critics/flow_field_critic.hpp"

#include <cmath>
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#endif

#include "nav2_tgmppi_controller/tools/flow_field.hpp"

namespace tgmppi::critics
{

void FlowFieldCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);

  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 5.0f);
  getParam(running_weight_, "running_weight", 2.0f);

  RCLCPP_INFO(
    logger_, "FlowFieldCritic instantiated with %d power, %f weight, "
    "%f running weight.", power_, weight_, running_weight_);

#ifdef TGMPPI_WITH_CUDA
  gpu_critic_.initialize();
#endif
}

void FlowFieldCritic::score(CriticData & data)
{
  if (!enabled_ || data.flow_field == nullptr || !data.flow_field->ready()) {
    return;  // ray mode or no field yet -> defer to the other critics
  }
  const FlowField & ff = *data.flow_field;

#ifdef TGMPPI_WITH_CUDA
  if (data.compute_backend == "cuda" && gpu_critic_.ready()) {
    auto cost_out = xt::xtensor<float, 1>::from_shape({data.costs.shape(0)});
    if (data.gpu_batch != nullptr) {
      // Chained: reads the rollout's already-resident trajectory tensors
      // directly, uploading only the flow field's own grid (which the
      // rollout has no reason to already have).
      const auto * rollout = data.gpu_batch;
      auto grid = GpuFlowFieldCritic::uploadGrid(ff);
      auto cost_gpu = gpu_critic_.computeDevice(
        rollout->trajX(), rollout->trajY(), grid, ff.sizeX(), ff.sizeY(),
        ff.originX(), ff.originY(), ff.resolution(), ff.dmax(),
        weight_, running_weight_, power_);
      data.gpu_batch->addCosts(cost_gpu);
      return;
    } else {
      gpu_critic_.score(data.trajectories, ff, weight_, running_weight_, power_, cost_out);
    }
    data.costs += cost_out;
    return;
  }
  if (data.gpu_batch) {
    data.gpu_batch->materializeHost();
    data.gpu_batch->flushCosts(data.costs);
  }
#endif

  const auto & traj_x = data.trajectories.x;
  const auto & traj_y = data.trajectories.y;
  const std::size_t batch = traj_x.shape(0);
  const std::size_t steps = traj_x.shape(1);
  if (batch == 0 || steps == 0) {
    return;
  }

  for (std::size_t b = 0; b < batch; ++b) {
    float sum = 0.0f;
    for (std::size_t t = 0; t < steps; ++t) {
      sum += ff.distAt(traj_x(b, t), traj_y(b, t));
    }
    const float final_level = ff.distAt(traj_x(b, steps - 1), traj_y(b, steps - 1));
    const float cost = weight_ * final_level +
      running_weight_ * (sum / static_cast<float>(steps));
    data.costs(b) += (power_ > 1u) ?
      std::pow(cost, static_cast<float>(power_)) : cost;
  }
}

}  // namespace tgmppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(tgmppi::critics::FlowFieldCritic, tgmppi::critics::CriticFunction)

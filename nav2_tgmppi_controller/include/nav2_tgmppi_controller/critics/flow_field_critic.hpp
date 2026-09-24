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

#ifndef NAV2_TGMPPI_CONTROLLER__CRITICS__FLOW_FIELD_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__CRITICS__FLOW_FIELD_CRITIC_HPP_

#include "nav2_tgmppi_controller/critic_function.hpp"
#ifdef TGMPPI_WITH_CUDA
#include "nav2_tgmppi_controller/tools/gpu_flow_field_critic.hpp"
#endif

namespace tgmppi::critics
{

/**
 * @class tgmppi::critics::FlowFieldCritic
 * @brief Scores trajectories by the water level (geodesic distance-to-goal
 * flooded through robot-passable space) instead of Euclidean goal distance
 * or distance-to-path. Replaces GoalCritic + PathAlignCritic +
 * PathFollowCritic when tgmppi_mode is "flow": the whole field is the
 * attraction, so a sample threading a narrow-but-wet gap scores well and a
 * sample heading into a dry cul-de-sac scores badly — no single polyline to
 * be enslaved to. cost = weight * level(final point)
 *               + running_weight * mean(level over horizon).
 */
class FlowFieldCritic : public CriticFunction
{
public:
  void initialize() override;

  void score(CriticData & data) override;
  bool supportsGpuBatch() const override {return true;}

protected:
  unsigned int power_{1};
  float weight_{5.0f};
  float running_weight_{2.0f};

#ifdef TGMPPI_WITH_CUDA
  GpuFlowFieldCritic gpu_critic_;
#endif
};

}  // namespace tgmppi::critics

#endif  // NAV2_TGMPPI_CONTROLLER__CRITICS__FLOW_FIELD_CRITIC_HPP_

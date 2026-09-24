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

#ifndef NAV2_TGMPPI_CONTROLLER__CRITIC_DATA_HPP_
#define NAV2_TGMPPI_CONTROLLER__CRITIC_DATA_HPP_

#include <memory>
#include <optional>
#include <cstdint>
#include <vector>
#include <xtensor/xtensor.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"
#include "nav2_tgmppi_controller/motion_models.hpp"
#include "nav2_tgmppi_controller/tools/space_time_search.hpp"
#include "nav2_tgmppi_controller/tools/dynamic_obstacle_cost.hpp"


namespace tgmppi
{

class FlowField;
class GpuBatch;

/**
 * @struct tgmppi::CriticData
 * @brief Data to pass to critics for scoring, including state, trajectories, path, costs, and
 * important parameters to share
 */
struct CriticData
{
  const models::State & state;
  const models::Trajectories & trajectories;
  const models::Path & path;

  xt::xtensor<float, 1> & costs;
  float & model_dt;

  bool fail_flag;
  nav2_core::GoalChecker * goal_checker;
  std::shared_ptr<MotionModel> motion_model;
  std::optional<std::vector<bool>> path_pts_valid;
  std::optional<size_t> furthest_reached_path_point;

  // Water field built by the optimizer in flow mode; nullptr in ray mode.
  // FlowFieldCritic scores trajectories against it.
  const FlowField * flow_field{nullptr};

  // "cpu" (always) or "cuda" (only if the optimizer's GPU backend is built
  // AND ready at runtime -- see Optimizer::reset()). A critic with a GPU
  // implementation checks this and falls back to its CPU path whenever
  // it's "cpu", exactly like Optimizer::generateNoisedTrajectories() does.
  const std::string & compute_backend;

  // Current device batch, set only after a successful CUDA rollout.
  // Forward declaration keeps CPU builds independent of LibTorch.
  GpuBatch * gpu_batch{nullptr};

  // Tracked moving obstacles (received ground-truth states only), snapshotted
  // once per cycle in Optimizer::prepare(). nullptr when none have been
  // received. DynamicObstacleCritic scores rollouts against their predictions.
  const std::vector<SpaceTimeObstacle> * tracked_obstacles{nullptr};

  // Per-rollout contact result produced by DynamicObstacleCritic.  Costs alone
  // cannot represent infeasibility: grouped MPPI normalizes each mode locally,
  // so an all-colliding mode would otherwise still have a valid softmax.
  // The optimizer owns this buffer and resets it before every critic pass.
  std::vector<uint8_t> * dynamic_collision_rows{nullptr};

  // Geometry and prediction settings used by DynamicObstacleCritic this cycle.
  // The final command veto uses them with point_step=1 to catch contacts
  // between the critic's cost-sampling points.
  std::optional<DynamicObstacleCostParams> dynamic_obstacle_params;
};

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__CRITIC_DATA_HPP_

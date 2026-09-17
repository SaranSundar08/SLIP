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
#include <vector>
#include <xtensor/xtensor.hpp>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_core/goal_checker.hpp"
#include "nav2_tgmppi_controller/models/state.hpp"
#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"
#include "nav2_tgmppi_controller/motion_models.hpp"
#include "nav2_tgmppi_controller/tools/space_time_search.hpp"


namespace tgmppi
{

class FlowField;

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

  // Opaque tgmppi::GpuRollout* (cast only where TGMPPI_WITH_CUDA is
  // defined), non-null iff this cycle's rollout ran on the GPU. Lets a
  // GPU-capable critic (e.g. CostCritic) read the already-resident
  // trajectory tensors directly -- via GpuRollout::trajX()/trajY() -- and
  // chain its own compute onto them with no CPU round-trip in between,
  // instead of every GPU critic re-uploading trajectories.x/y itself.
  // Two additively-ported critics that each did their own independent
  // round trip (see PROJECT_STATUS.md 2026-09-10, slices 1 and 2) both
  // ended up SLOWER than CPU purely from that overhead -- this is the
  // fix, one shared upload consumed by every critic that can use it.
  const void * gpu_rollout{nullptr};

  // Tracked moving obstacles (received ground-truth states only), snapshotted
  // once per cycle in Optimizer::prepare(). nullptr when none have been
  // received. DynamicObstacleCritic scores rollouts against their predictions.
  const std::vector<SpaceTimeObstacle> * tracked_obstacles{nullptr};
};

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__CRITIC_DATA_HPP_

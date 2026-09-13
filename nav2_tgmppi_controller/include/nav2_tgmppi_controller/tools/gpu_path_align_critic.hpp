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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_PATH_ALIGN_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_PATH_ALIGN_CRITIC_HPP_

#ifdef TGMPPI_WITH_CUDA

#include <vector>

#include <torch/torch.h>

#include "nav2_tgmppi_controller/models/trajectories.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::GpuPathAlignCritic
 * @brief LibTorch/CUDA implementation of PathAlignCritic::score()'s hot
 * loop (use_path_orientations=false only -- not configured true anywhere
 * in this project's yaml, same "port what's active" scoping as
 * ConstraintCritic/GoalAngleCritic/PathAngleCritic).
 *
 * findClosestPathPt() (utils.hpp) is std::lower_bound over a sorted
 * distance array with a restricted search range [init, end) carried
 * across steps -- that's torch::searchsorted plus a hand-replicated
 * carry, NOT a single one-shot vectorized call, for two reasons:
 *
 * 1. Restricting the search to [init, end) only changes the result
 *    (vs. searching the full array) in one specific case: when nothing
 *    in [init, end) is < dist, the CPU function returns literal index 0
 *    rather than the true lower_bound answer. This is very likely an
 *    unintentional upstream quirk (a near-stationary candidate sample
 *    can trivially hit it -- MPPI batches always include some), but this
 *    project's parity rule is to replicate the real .cpp exactly, not a
 *    "corrected" version of it. That 0-vs-answer branch is per-row
 *    stateful (depends on each trajectory's own running `init`), so it's
 *    computed with a small host-side loop over the ~T/step sampled
 *    points (13 for this project's time_steps=56, trajectory_point_step
 *    =4), each iteration vectorized across the whole batch -- exactly
 *    the same loop-over-time/vectorize-over-batch shape GpuRollout
 *    itself already uses, not a new pattern here.
 * 2. When dist exceeds every value in the path's integrated-distance
 *    array, std::lower_bound returns vec.end() and the CPU code
 *    dereferences it unchecked (*iter) -- undefined behavior, not
 *    something with a well-defined "correct" answer to match. The GPU
 *    side clamps the index defensively instead of reproducing UB.
 */
class GpuPathAlignCritic
{
public:
  GpuPathAlignCritic() = default;

  void initialize() {ready_ = torch::cuda::is_available();}
  bool ready() const {return ready_;}

  /** @brief The small (O(path length), not O(batch)) per-call inputs a
   * GPU chained call still has to upload itself, exactly like
   * GpuCostCritic uploading its own costmap and GpuFlowFieldCritic its
   * own grid even when trajectories are already GPU-resident. */
  struct PathUpload
  {
    torch::Tensor path_x;
    torch::Tensor path_y;
    torch::Tensor path_valid;
    torch::Tensor path_integrated_distances;
  };

  /** @brief Uploads the path's x/y arrays (excluding the last point,
   * matching P_x/P_y in the CPU critic), the per-point validity mask,
   * and the CPU-computed integrated-distance array. */
  static PathUpload uploadPathInputs(
    const models::Path & path, const std::vector<bool> & path_pts_valid,
    size_t path_segments_count);

  /** @brief Standalone convenience: uploads trajectories.x/y, the path's
   * x/y arrays, the validity mask, and computes the (CPU-side, O(path
   * length) not O(batch)) integrated-distance array itself. */
  void score(
    const models::Trajectories & trajectories, const models::Path & path,
    const std::vector<bool> & path_pts_valid, size_t path_segments_count,
    int trajectory_point_step, float weight, unsigned int power,
    xt::xtensor<float, 1> & cost_out);

  /**
   * @brief Device-resident compute only: traj_x/traj_y are already CUDA
   * tensors (e.g. GpuRollout::trajX()/trajY()). path_x/path_y/path_valid
   * and path_integrated_distances are small (O(path length), not
   * O(batch)) CPU-side arrays uploaded fresh each call -- cheap relative
   * to the batch-sized trajectory tensors. Returns the [K] cost tensor,
   * still on the GPU.
   */
  torch::Tensor computeDevice(
    const torch::Tensor & traj_x, const torch::Tensor & traj_y,
    const torch::Tensor & path_x, const torch::Tensor & path_y,
    const torch::Tensor & path_valid, const torch::Tensor & path_integrated_distances,
    int trajectory_point_step, float weight, unsigned int power);

private:
  bool ready_{false};
};

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA
#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_PATH_ALIGN_CRITIC_HPP_

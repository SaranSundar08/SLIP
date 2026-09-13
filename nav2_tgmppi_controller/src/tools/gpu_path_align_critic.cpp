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

#include "nav2_tgmppi_controller/tools/gpu_path_align_critic.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cmath>
#include <cstring>
#include <vector>

namespace tgmppi
{

GpuPathAlignCritic::PathUpload GpuPathAlignCritic::uploadPathInputs(
  const models::Path & path, const std::vector<bool> & path_pts_valid,
  size_t path_segments_count)
{
  const int64_t path_len = static_cast<int64_t>(path.x.shape(0)) - 1;  // matches P_x/P_y's range(_, -1)
  const auto cpu_opts_f32 = torch::TensorOptions().dtype(torch::kFloat32);
  const auto cpu_opts_u8 = torch::TensorOptions().dtype(torch::kUInt8);

  auto path_x = torch::from_blob(
    const_cast<float *>(path.x.data()), {path_len}, cpu_opts_f32).clone().to(torch::kCUDA);
  auto path_y = torch::from_blob(
    const_cast<float *>(path.y.data()), {path_len}, cpu_opts_f32).clone().to(torch::kCUDA);

  // std::vector<bool> is bit-packed, not blob-able -- copy out to a real
  // contiguous byte buffer first.
  std::vector<uint8_t> valid_bytes(static_cast<size_t>(path_len), 0u);
  for (int64_t i = 0; i < path_len && static_cast<size_t>(i) < path_pts_valid.size(); ++i) {
    valid_bytes[static_cast<size_t>(i)] = path_pts_valid[static_cast<size_t>(i)] ? 1u : 0u;
  }
  auto path_valid = torch::from_blob(
    valid_bytes.data(), {path_len}, cpu_opts_u8).clone().to(torch::kCUDA);

  // Integrated distance along the path, up to path_segments_count -- an
  // O(path length) sequential prefix sum, the same as the CPU critic
  // computes it. Not worth parallelizing: it's tiny relative to the
  // O(batch) work below, and doing it on the GPU would just add a launch
  // for no benefit.
  std::vector<float> dist_cpu(path_segments_count, 0.0f);
  for (size_t i = 1; i < path_segments_count; ++i) {
    const float dx = path.x(i) - path.x(i - 1);
    const float dy = path.y(i) - path.y(i - 1);
    dist_cpu[i] = dist_cpu[i - 1] + std::sqrt(dx * dx + dy * dy);
  }
  auto path_integrated_distances = torch::from_blob(
    dist_cpu.data(), {static_cast<int64_t>(path_segments_count)}, cpu_opts_f32)
    .clone().to(torch::kCUDA);

  return {path_x, path_y, path_valid, path_integrated_distances};
}

torch::Tensor GpuPathAlignCritic::computeDevice(
  const torch::Tensor & traj_x, const torch::Tensor & traj_y,
  const torch::Tensor & path_x, const torch::Tensor & path_y,
  const torch::Tensor & path_valid, const torch::Tensor & path_integrated_distances,
  int trajectory_point_step, float weight, unsigned int power)
{
  using namespace torch::indexing;  // NOLINT

  const int64_t K = traj_x.size(0);
  const int64_t T = traj_x.size(1);
  const int64_t step = static_cast<int64_t>(trajectory_point_step);
  const int64_t path_len = path_integrated_distances.size(0);
  const auto long_opts = torch::TensorOptions().dtype(torch::kLong).device(traj_x.device());
  const auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(traj_x.device());

  // p values: step, 2*step, ... < T (matches the CPU for-loop exactly).
  // Prepend 0 so consecutive differences of this list give the same
  // per-segment displacements the CPU loop computes one at a time.
  std::vector<int64_t> sample_idx_host{0};
  for (int64_t p = step; p < T; p += step) {
    sample_idx_host.push_back(p);
  }
  const int64_t M = static_cast<int64_t>(sample_idx_host.size()) - 1;
  if (M <= 0) {
    return torch::zeros({K}, float_opts);
  }
  auto sample_idx = torch::from_blob(
    sample_idx_host.data(), {static_cast<int64_t>(sample_idx_host.size())},
    torch::TensorOptions().dtype(torch::kLong)).clone().to(traj_x.device());

  auto sampled_x = traj_x.index_select(1, sample_idx);  // [K, M+1]
  auto sampled_y = traj_y.index_select(1, sample_idx);

  auto seg_dx = sampled_x.index({Slice(), Slice(1, None)}) - sampled_x.index({Slice(), Slice(None, -1)});
  auto seg_dy = sampled_y.index({Slice(), Slice(1, None)}) - sampled_y.index({Slice(), Slice(None, -1)});
  auto seg_dist = torch::sqrt(seg_dx * seg_dx + seg_dy * seg_dy);          // [K, M]
  auto traj_integrated_distance_all = torch::cumsum(seg_dist, 1);         // [K, M]

  auto init = torch::zeros({K}, long_opts);
  auto num_samples = torch::zeros({K}, float_opts);
  auto summed_path_dist = torch::zeros({K}, float_opts);

  for (int64_t k = 0; k < M; ++k) {
    auto dist_k = traj_integrated_distance_all.select(1, k).contiguous();  // [K]

    // Full-array lower_bound; see the class docstring for why a full
    // search gives the same "advance" answer as the CPU's restricted
    // [init, end) search, and why the no-advance branch is handled
    // separately via an explicit gather+compare rather than folded into
    // the search itself.
    auto full_idx = torch::searchsorted(path_integrated_distances, dist_k);
    full_idx = full_idx.clamp(0, path_len - 1);  // defensive: CPU dereferences end() unchecked here (UB)

    auto vec_at_init = path_integrated_distances.index_select(0, init);
    auto no_advance = dist_k <= vec_at_init;

    auto idx_lo = (full_idx - 1).clamp(0, path_len - 1);
    auto idx_hi = full_idx;
    auto val_lo = path_integrated_distances.index_select(0, idx_lo);
    auto val_hi = path_integrated_distances.index_select(0, idx_hi);
    auto dist_to_lo = dist_k - val_lo;
    auto dist_to_hi = val_hi - dist_k;
    auto refined_idx = torch::where(dist_to_lo < dist_to_hi, idx_lo, idx_hi);

    auto path_pt = torch::where(no_advance, torch::zeros_like(refined_idx), refined_idx);

    auto valid_mask = path_valid.index_select(0, path_pt).to(torch::kBool);
    auto px_at = path_x.index_select(0, path_pt);
    auto py_at = path_y.index_select(0, path_pt);

    auto tx_k = sampled_x.select(1, k + 1);
    auto ty_k = sampled_y.select(1, k + 1);
    auto dx = px_at - tx_k;
    auto dy = py_at - ty_k;
    auto point_dist = torch::sqrt(dx * dx + dy * dy);

    summed_path_dist = summed_path_dist + torch::where(valid_mask, point_dist, torch::zeros_like(point_dist));
    num_samples = num_samples + valid_mask.to(torch::kFloat32);

    init = path_pt;
  }

  auto safe_denom = num_samples.clamp_min(1e-6f);
  auto raw_cost = summed_path_dist / safe_denom;
  auto cost = torch::where(num_samples > 0, raw_cost, torch::zeros_like(raw_cost));
  return torch::pow(cost * weight, static_cast<double>(power));
}

void GpuPathAlignCritic::score(
  const models::Trajectories & trajectories, const models::Path & path,
  const std::vector<bool> & path_pts_valid, size_t path_segments_count,
  int trajectory_point_step, float weight, unsigned int power,
  xt::xtensor<float, 1> & cost_out)
{
  const int64_t K = static_cast<int64_t>(trajectories.x.shape(0));
  const int64_t T = static_cast<int64_t>(trajectories.x.shape(1));
  const std::vector<int64_t> shape{K, T};
  const auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);

  auto traj_x = torch::from_blob(
    const_cast<float *>(trajectories.x.data()), shape, cpu_opts).to(torch::kCUDA);
  auto traj_y = torch::from_blob(
    const_cast<float *>(trajectories.y.data()), shape, cpu_opts).to(torch::kCUDA);

  auto up = uploadPathInputs(path, path_pts_valid, path_segments_count);
  auto cost = computeDevice(
    traj_x, traj_y, up.path_x, up.path_y, up.path_valid, up.path_integrated_distances,
    trajectory_point_step, weight, power);

  auto cost_cpu = cost.to(torch::kCPU).contiguous();
  std::memcpy(cost_out.data(), cost_cpu.data_ptr<float>(), static_cast<size_t>(K) * sizeof(float));
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

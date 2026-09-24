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

#include "nav2_tgmppi_controller/tools/gpu_rollout.hpp"

#ifdef TGMPPI_WITH_CUDA

#include <cmath>
#include <cstring>

namespace tgmppi
{

void GpuRollout::initialize(unsigned int batch_size, unsigned int time_steps)
{
  ready_ = false;
  if (!torch::cuda::is_available()) {
    return;  // caller (Optimizer) falls back to the CPU path with a warning
  }
  batch_size_ = batch_size;
  time_steps_ = time_steps;

  const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  vx_ = torch::zeros({batch_size_, time_steps_}, opts);
  vy_ = torch::zeros({batch_size_, time_steps_}, opts);
  wz_ = torch::zeros({batch_size_, time_steps_}, opts);
  traj_x_ = torch::zeros({batch_size_, time_steps_}, opts);
  traj_y_ = torch::zeros({batch_size_, time_steps_}, opts);
  traj_yaws_ = torch::zeros({batch_size_, time_steps_}, opts);
  ready_ = true;
}

void GpuRollout::rollout(
  const models::State & state_in, models::State & state_out,
  models::Trajectories & trajectories_out,
  float model_dt, bool is_holonomic)
{
  // Yaw from quaternion, computed directly instead of via tf2::getYaw().
  // tf2::getYaw()/tf2::fromMsg() are header-declared `inline`; at -O3 this
  // translation unit (uniquely among the ones that call tf2::getYaw()) did
  // not get the call inlined and no other .cpp in this .so happened to keep
  // an out-of-line copy of the vague-linkage symbol, producing an "undefined
  // symbol: tf2::fromMsg(...)" abort the first time a real control cycle hit
  // this line (2026-09-11, root-caused via gdb `break _exit`/`break abort`
  // on the live nav2_container process -- see PROJECT_STATUS.md). Avoiding
  // the tf2 inline entirely here removes the dependency on compiler inlining
  // decisions rather than papering over it with a linker anchor trick.
  const auto & q = state_in.pose.pose.orientation;
  const float initial_yaw = static_cast<float>(std::atan2(
    2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)));

  const auto cpu_opts = torch::TensorOptions().dtype(torch::kFloat32);
  const std::vector<int64_t> shape{
    static_cast<int64_t>(batch_size_), static_cast<int64_t>(time_steps_)};

  // Upload the already noised (+ TG-MPPI bias-shifted) commanded controls.
  // from_blob wraps the xtensor buffer without copying; .to(kCUDA) is the
  // actual (and only necessary) copy, taken before the CPU-side blob would
  // go out of scope.
  auto cvx = torch::from_blob(
    const_cast<float *>(state_in.cvx.data()), shape, cpu_opts).to(torch::kCUDA);
  auto cwz = torch::from_blob(
    const_cast<float *>(state_in.cwz.data()), shape, cpu_opts).to(torch::kCUDA);
  torch::Tensor cvy;
  if (is_holonomic) {
    cvy = torch::from_blob(
      const_cast<float *>(state_in.cvy.data()), shape, cpu_opts).to(torch::kCUDA);
  }

  computeDevice(
    cvx, cwz, cvy, model_dt, initial_yaw,
    state_in.pose.pose.position.x, state_in.pose.pose.position.y,
    static_cast<float>(state_in.speed.linear.x), static_cast<float>(state_in.speed.linear.y),
    static_cast<float>(state_in.speed.angular.z), is_holonomic);

  // Compatibility wrapper for callers explicitly requesting CPU output.
  // The optimizer uses GpuBatch::begin()/materializeHost() instead.
  auto copy_down = [](const torch::Tensor & src, float * dst) {
      auto cpu = src.to(torch::kCPU).contiguous();
      std::memcpy(dst, cpu.data_ptr<float>(), cpu.numel() * sizeof(float));
    };
  copy_down(vx_, state_out.vx.data());
  copy_down(wz_, state_out.wz.data());
  if (is_holonomic) {
    copy_down(vy_, state_out.vy.data());
  }
  copy_down(traj_x_, trajectories_out.x.data());
  copy_down(traj_y_, trajectories_out.y.data());
  copy_down(traj_yaws_, trajectories_out.yaws.data());
}

void GpuRollout::computeDevice(
  const torch::Tensor & cvx, const torch::Tensor & cwz, const torch::Tensor & cvy,
  float model_dt, float initial_yaw, double robot_x, double robot_y,
  float speed_vx, float speed_vy, float speed_wz, bool is_holonomic)
{
  using namespace torch::indexing;  // NOLINT

  // predict(): one-timestep actuation-delay shift, matching
  // MotionModel::predict() exactly -- state.v[:,1:] = state.cv[:,:-1],
  // state.v[:,0] = the robot's current measured speed. Every motion model
  // in this controller (DiffDrive, Omni, Ackermann) shares this one
  // predict() implementation; none override it.
  vx_.index_put_({Slice(), Slice(1, None)}, cvx.index({Slice(), Slice(0, -1)}));
  vx_.index_put_({Slice(), 0}, speed_vx);
  wz_.index_put_({Slice(), Slice(1, None)}, cwz.index({Slice(), Slice(0, -1)}));
  wz_.index_put_({Slice(), 0}, speed_wz);
  if (is_holonomic) {
    vy_.index_put_({Slice(), Slice(1, None)}, cvy.index({Slice(), Slice(0, -1)}));
    vy_.index_put_({Slice(), 0}, speed_vy);
  } else {
    vy_.zero_();
  }

  // integrateStateVelocities(): cumulative diff-drive kinematics. Matches
  // optimizer.cpp's xtensor formula element-for-element: yaw is the cumsum
  // of wz*dt offset by initial_yaw; the position increment at t uses the
  // yaw from t-1 (semi-implicit Euler), which is why yaw_cos/yaw_sin are
  // built from the yaws shifted right by one with column 0 seeded from
  // initial_yaw directly, not from traj_yaws_ column 0.
  traj_yaws_ = torch::cumsum(wz_ * model_dt, 1) + initial_yaw;
  auto yaw_cos = torch::empty_like(traj_yaws_);
  auto yaw_sin = torch::empty_like(traj_yaws_);
  yaw_cos.index_put_({Slice(), 0}, std::cos(initial_yaw));
  yaw_sin.index_put_({Slice(), 0}, std::sin(initial_yaw));
  const auto yaws_cut = traj_yaws_.index({Slice(), Slice(0, -1)});
  yaw_cos.index_put_({Slice(), Slice(1, None)}, torch::cos(yaws_cut));
  yaw_sin.index_put_({Slice(), Slice(1, None)}, torch::sin(yaws_cut));

  auto dx = vx_ * yaw_cos;
  auto dy = vx_ * yaw_sin;
  if (is_holonomic) {
    dx = dx - vy_ * yaw_sin;
    dy = dy + vy_ * yaw_cos;
  }
  traj_x_ = robot_x + torch::cumsum(dx * model_dt, 1);
  traj_y_ = robot_y + torch::cumsum(dy * model_dt, 1);
}

}  // namespace tgmppi

#endif  // TGMPPI_WITH_CUDA

// Copyright (c) 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0.
#include "nav2_tgmppi_controller/tools/gpu_batch.hpp"
#ifdef TGMPPI_WITH_CUDA
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace tgmppi
{
namespace
{
torch::Tensor hostTensor(const float * values, const std::vector<int64_t> & shape)
{
  return torch::from_blob(const_cast<float *>(values), shape, torch::kFloat32);
}
void download(const torch::Tensor & tensor, float * target)
{
  const auto cpu = tensor.to(torch::kCPU).contiguous();
  std::memcpy(target, cpu.data_ptr<float>(), cpu.numel() * sizeof(float));
}
}  // namespace

void GpuBatch::initialize(unsigned int batch, unsigned int steps)
{
  state_ = nullptr;
  trajectories_ = nullptr;
  host_current_ = false;
  pending_costs_ = false;
  dynamic_scored_ = false;
  batch_ = batch;
  steps_ = steps;
  rollout_.initialize(batch, steps);
  if (!ready()) {return;}
  const auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  controls_ = torch::zeros({3, batch, steps}, opts);
  costs_ = torch::zeros({batch}, opts);
  contacts_ = torch::zeros({batch}, opts.dtype(torch::kBool));
}

void GpuBatch::refreshControls()
{
  if (!state_) {throw std::logic_error("GPU batch has no current iteration");}
  const std::vector<int64_t> shape{batch_, steps_};
  controls_[0].copy_(hostTensor(state_->cvx.data(), shape));
  controls_[1].copy_(hostTensor(state_->cwz.data(), shape));
  if (omni_) {controls_[2].copy_(hostTensor(state_->cvy.data(), shape));}
  else {controls_[2].zero_();}
  ++transfers_.control_uploads;
}

void GpuBatch::begin(
  models::State & state, models::Trajectories & trajectories, float dt, bool omni)
{
  if (!ready() || state.cvx.shape(0) != batch_ || state.cvx.shape(1) != steps_) {
    throw std::logic_error("GPU batch shape does not match optimizer state");
  }
  torch::NoGradGuard guard;
  state_ = &state;
  trajectories_ = &trajectories;
  omni_ = omni;
  transfers_ = {};
  host_current_ = pending_costs_ = dynamic_scored_ = false;
  costs_.zero_();
  contacts_.zero_();
  refreshControls();
  const auto & q = state.pose.pose.orientation;
  const float yaw = std::atan2(
    2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
  rollout_.computeDevice(
    controls_[0], controls_[1], controls_[2], dt, yaw,
    state.pose.pose.position.x, state.pose.pose.position.y,
    state.speed.linear.x, state.speed.linear.y, state.speed.angular.z, omni);
}

void GpuBatch::materializeHost()
{
  if (!state_ || host_current_) {return;}
  // One packed transfer, only for an actual CPU consumer.
  auto snapshot = torch::stack({vx(), wz(), vy(), trajX(), trajY(), trajYaws()})
    .to(torch::kCPU).contiguous();
  const float * src = snapshot.data_ptr<float>();
  const size_t n = static_cast<size_t>(batch_) * steps_;
  float * dst[] = {state_->vx.data(), state_->wz.data(), state_->vy.data(),
    trajectories_->x.data(), trajectories_->y.data(), trajectories_->yaws.data()};
  for (size_t axis = 0; axis < 6; ++axis) {
    std::memcpy(dst[axis], src + axis * n, n * sizeof(float));
  }
  host_current_ = true;
  ++transfers_.host_snapshots;
}

void GpuBatch::addCosts(const torch::Tensor & costs)
{
  costs_.add_(costs);
  pending_costs_ = true;
}

void GpuBatch::flushCosts(xt::xtensor<float, 1> & costs)
{
  if (!pending_costs_) {return;}
  auto cpu = costs_.to(torch::kCPU).contiguous();
  const float * values = cpu.data_ptr<float>();
  for (unsigned int r = 0; r < batch_; ++r) {costs(r) += values[r];}
  costs_.zero_();
  pending_costs_ = false;
  ++transfers_.cost_downloads;
}

void GpuBatch::finishCritics(xt::xtensor<float, 1> & costs, std::vector<uint8_t> * contacts)
{
  flushCosts(costs);
  if (dynamic_scored_ && contacts) {
    auto cpu = contacts_.to(torch::kCPU, torch::kUInt8).contiguous();
    const auto * values = cpu.data_ptr<uint8_t>();
    contacts->assign(values, values + batch_);
  }
}

void GpuBatch::dynamicCosts(
  const std::vector<SpaceTimeObstacle> & obstacles, const DynamicObstacleCostParams & p,
  float weight, unsigned int power)
{
  using namespace torch::indexing;
  dynamic_scored_ = true;
  contacts_.zero_();
  if (obstacles.empty()) {return;}
  const int64_t stride = std::max<size_t>(1, p.point_step);
  const auto x = trajX().index({Slice(), Slice(0, None, stride)});
  const auto y = trajY().index({Slice(), Slice(0, None, stride)});
  const auto yaw = trajYaws().index({Slice(), Slice(0, None, stride)});
  const auto t = ((torch::arange(0, steps_, stride, x.options()) + 1) * p.model_dt)
    .clamp_max(p.max_prediction_time).view({1, -1, 1});
  const auto c = torch::cos(yaw).unsqueeze(2);
  const auto s = torch::sin(yaw).unsqueeze(2);
  auto clearance = torch::full_like(x, std::numeric_limits<float>::max());
  // Bound temporary K x T x obstacles tensors in dense scenes.
  constexpr size_t chunk_size = 16;
  for (size_t start = 0; start < obstacles.size(); start += chunk_size) {
    const size_t count = std::min(chunk_size, obstacles.size() - start);
    std::vector<float> packed(5 * count);
    for (size_t i = 0; i < count; ++i) {
      const auto & o = obstacles[start + i];
      packed[i] = o.x; packed[count + i] = o.y;
      packed[2 * count + i] = o.vx; packed[3 * count + i] = o.vy;
      packed[4 * count + i] = o.radius;
    }
    auto obs = hostTensor(packed.data(), {5, static_cast<int64_t>(count)}).to(x.device());
    auto ox = obs[0].view({1, 1, -1}) + obs[2].view({1, 1, -1}) * t;
    auto oy = obs[1].view({1, 1, -1}) + obs[3].view({1, 1, -1}) * t;
    for (float offset : {p.disc_offset, -p.disc_offset}) {
      auto dx = x.unsqueeze(2) + offset * c - ox;
      auto dy = y.unsqueeze(2) + offset * s - oy;
      auto d = torch::hypot(dx, dy) - (p.disc_radius + obs[4].view({1, 1, -1}));
      clearance = torch::minimum(clearance, std::get<0>(d.min(2)));
    }
  }
  contacts_ = (clearance < 0).any(1);
  const auto penetration = (-clearance).clamp_min(0).sum(1);
  auto repulsive = torch::zeros_like(penetration);
  if (p.soft_distance > 0) {
    const auto q = ((p.soft_distance - clearance) / p.soft_distance).clamp(0, 1);
    repulsive = torch::where(clearance >= 0, p.critical_cost * q.square(),
      torch::zeros_like(clearance)).sum(1);
  }
  auto score = torch::where(contacts_, p.collision_cost + p.penetration_cost * penetration,
    repulsive);
  addCosts(torch::pow(weight * score / static_cast<float>(steps_), power));
}

size_t GpuBatch::furthestPathPoint(const models::Path & path) const
{
  if (path.x.size() == 0) {return 0;}
  const std::vector<int64_t> shape{1, static_cast<int64_t>(path.x.size())};
  auto px = hostTensor(path.x.data(), shape).to(trajX().device());
  auto py = hostTensor(path.y.data(), shape).to(trajX().device());
  auto dx = trajX().select(1, steps_ - 1).unsqueeze(1) - px;
  auto dy = trajY().select(1, steps_ - 1).unsqueeze(1) - py;
  return (dx.square() + dy.square()).argmin(1).max().item<int64_t>();
}

size_t GpuBatch::initialPathPoint(const models::Path & path) const
{
  if (path.x.size() == 0) {return 0;}
  const std::vector<int64_t> shape{static_cast<int64_t>(path.x.size())};
  auto dx = hostTensor(path.x.data(), shape).to(trajX().device()) - trajX()[0][0];
  auto dy = hostTensor(path.y.data(), shape).to(trajY().device()) - trajY()[0][0];
  return (dx.square() + dy.square()).argmin().item<int64_t>();
}

xt::xtensor<float, 3> GpuBatch::weightedMeans(
  const std::vector<std::pair<unsigned int, unsigned int>> & groups,
  const std::vector<float> & weights)
{
  if (weights.size() != batch_ || groups.empty()) {
    throw std::invalid_argument("Invalid GPU update weights or groups");
  }
  auto w = hostTensor(weights.data(), {batch_}).to(controls_.device());
  std::vector<torch::Tensor> means;
  for (const auto & group : groups) {
    if (group.second == 0 || group.first + group.second > batch_) {
      throw std::invalid_argument("GPU update group outside batch");
    }
    auto local = w.narrow(0, group.first, group.second).view({1, -1, 1});
    means.push_back((controls_.narrow(1, group.first, group.second) * local).sum(1));
  }
  xt::xtensor<float, 3> result = xt::empty<float>({groups.size(), size_t{3}, size_t{steps_}});
  download(torch::stack(means), result.data());
  ++transfers_.mean_downloads;
  return result;
}
}  // namespace tgmppi
#endif

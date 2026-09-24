// Copyright (c) 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0.
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "nav2_tgmppi_controller/optimizer.hpp"
#include "nav2_tgmppi_controller/critics/goal_critic.hpp"
#include "nav2_tgmppi_controller/critics/constraint_critic.hpp"

namespace
{
class BatchOptimizer : public tgmppi::Optimizer
{
public:
  using Optimizer::state_;
  using Optimizer::settings_;
  using Optimizer::generated_trajectories_;
  using Optimizer::control_sequence_;
  using Optimizer::costs_;
  using Optimizer::critics_data_;
  using Optimizer::gpu_batch_;
  using Optimizer::path_;
  using Optimizer::updateControlSequence;
  using Optimizer::selected_mode_key_;
  using Optimizer::mode_nominals_;

  void configure(bool omni = false, unsigned int batch = 31, unsigned int steps = 17)
  {
    settings_.batch_size = batch;
    settings_.time_steps = steps;
    settings_.model_dt = 0.05f;
    settings_.temperature = 0.3f;
    settings_.gamma = 0.015f;
    settings_.constraints = {2.0f, -2.0f, 2.0f, 2.0f};
    settings_.sampling_std = {0.4f, 0.3f, 0.8f};
    settings_.tgmppi_shadow_mode = false;
    settings_.tgmppi_bias_enabled = true;
    settings_.tgmppi_mode_warm_start = 0.7f;
    if (omni) {motion_model_ = std::make_shared<tgmppi::OmniMotionModel>();}
    else {motion_model_ = std::make_shared<tgmppi::DiffDriveMotionModel>();}
    critics_data_.motion_model = motion_model_;
    state_.reset(batch, steps);
    generated_trajectories_.reset(batch, steps);
    control_sequence_.reset(steps);
    control_sequence_.vx.fill(0.3f);
    control_sequence_.wz.fill(-0.1f);
    costs_ = xt::zeros<float>({batch});
    state_.pose.pose.position.x = 0.3;
    state_.pose.pose.position.y = -0.2;
    state_.pose.pose.orientation.z = std::sin(0.21);
    state_.pose.pose.orientation.w = std::cos(0.21);
    state_.speed.linear.x = 0.2;
    state_.speed.linear.y = omni ? -0.1 : 0;
    state_.speed.angular.z = -0.3;
    for (unsigned int r = 0; r < batch; ++r) {
      costs_(r) = 0.03f * (r % 7);
      for (unsigned int t = 0; t < steps; ++t) {
        state_.cvx(r, t) = 0.6f * std::sin(0.3f * r + 0.1f * t);
        state_.cwz(r, t) = 0.7f * std::cos(0.2f * r - 0.05f * t);
        state_.cvy(r, t) = omni ? 0.2f * std::sin(0.13f * (r + t)) : 0;
      }
    }
    path_.reset(12);
    for (size_t i = 0; i < 12; ++i) {
      path_.x(i) = -0.5f + 0.2f * i;
      path_.y(i) = 0.1f * i;
    }
  }
  void cpuRollout()
  {
    updateStateVelocities(state_);
    integrateStateVelocities(generated_trajectories_, state_);
  }
  void deviceRollout()
  {
    settings_.compute_backend = "cuda";
    gpu_batch_.initialize(settings_.batch_size, settings_.time_steps);
    beginDevice();
  }
  void beginDevice()
  {
    gpu_batch_.begin(state_, generated_trajectories_, settings_.model_dt, isHolonomic());
    critics_data_.gpu_batch = &gpu_batch_;
  }
  void groups(bool enabled)
  {
    settings_.tgmppi_grouped_update = enabled;
    if (!enabled) {return;}
    for (size_t i = 0; i < 2; ++i) {
      ancillary_mode_valid_[i] = true;
      ancillary_mode_row_start_[i] = i * 10;
      ancillary_mode_samples_[i] = 10;
      ancillary_mode_key_[i] = i;
    }
  }
};

class Goal : public tgmppi::critics::GoalCritic
{
public:
  Goal() {enabled_ = true; weight_ = 2.0f; power_ = 1; threshold_to_consider_ = 100.0f;
    gpu_critic_.initialize();}
};
class Constraint : public tgmppi::critics::ConstraintCritic
{
public:
  Constraint() {enabled_ = true; weight_ = 3.0f; power_ = 1;
    max_vel_ = 0.4f; min_vel_ = -0.2f; gpu_critics_.initialize();}
};

class Manager : public tgmppi::CriticManager
{
public:
  void add(std::unique_ptr<tgmppi::critics::CriticFunction> critic)
  {
    critics_.push_back(std::move(critic));
  }
};

class CpuConsumer : public tgmppi::critics::CriticFunction
{
public:
  explicit CpuConsumer(const BatchOptimizer & expected) : expected_(expected) {}
  void initialize() override {}
  void score(tgmppi::CriticData & data) override
  {
    for (size_t r = 0; r < data.costs.size(); ++r) {
      EXPECT_NEAR(data.costs(r), expected_.costs_(r), 2e-5f);
      EXPECT_NEAR(data.trajectories.x(r, 8), expected_.generated_trajectories_.x(r, 8), 2e-5f);
    }
    data.costs += 3.0f;
  }
private:
  const BatchOptimizer & expected_;
};

template<class A, class B>
void expectTensorNear(const A & a, const B & b, float tolerance = 2e-5f)
{
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_NEAR(a.data()[i], b.data()[i], tolerance) << "element " << i;
  }
}

TEST(GpuBatch, RolloutMatchesCpuAndHostDownloadIsLazy)
{
  if (!torch::cuda::is_available()) {GTEST_SKIP() << "CUDA device unavailable";}
  for (bool omni : {false, true}) {
    BatchOptimizer cpu, gpu;
    cpu.configure(omni); gpu.configure(omni);
    cpu.cpuRollout(); gpu.deviceRollout();
    EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 0u);
    EXPECT_EQ(tgmppi::utils::findPathFurthestReachedPoint(cpu.critics_data_),
      tgmppi::utils::findPathFurthestReachedPoint(gpu.critics_data_));
    EXPECT_EQ(tgmppi::utils::findPathTrajectoryInitialPoint(cpu.critics_data_),
      tgmppi::utils::findPathTrajectoryInitialPoint(gpu.critics_data_));
    EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 0u);
    gpu.getGeneratedTrajectories();
    gpu.getGeneratedTrajectories();
    EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 1u);
    expectTensorNear(cpu.generated_trajectories_.x, gpu.generated_trajectories_.x);
    expectTensorNear(cpu.generated_trajectories_.y, gpu.generated_trajectories_.y);
    expectTensorNear(cpu.generated_trajectories_.yaws, gpu.generated_trajectories_.yaws);
    expectTensorNear(cpu.state_.vx, gpu.state_.vx);
    expectTensorNear(cpu.state_.vy, gpu.state_.vy);
  }
}

TEST(GpuBatch, DynamicCostsMatchCpuAcrossFiftyObstaclesAndReset)
{
  if (!torch::cuda::is_available()) {GTEST_SKIP() << "CUDA device unavailable";}
  BatchOptimizer cpu, gpu;
  cpu.configure(); gpu.configure(); cpu.cpuRollout(); gpu.deviceRollout();
  std::vector<tgmppi::SpaceTimeObstacle> obstacles;
  for (int i = 0; i < 50; ++i) {
    tgmppi::SpaceTimeObstacle o;
    o.x = 1.2f + 0.2f * (i % 10); o.y = -1.3f + 0.4f * (i / 10);
    o.vx = -0.3f; o.vy = 0.08f * (i % 3); o.radius = 0.2f;
    obstacles.push_back(o);
  }
  for (size_t stride : {size_t{1}, size_t{3}}) {
    gpu.beginDevice();
    tgmppi::DynamicObstacleCostParams p;
    p.point_step = stride; p.max_prediction_time = 0.4f;
    gpu.gpu_batch_.dynamicCosts(obstacles, p, 3.81f, 1);
    xt::xtensor<float, 1> result = xt::zeros<float>({31});
    std::vector<uint8_t> contacts;
    gpu.gpu_batch_.finishCritics(result, &contacts);
    for (size_t r = 0; r < 31; ++r) {
      const auto & tr = cpu.generated_trajectories_;
      auto score = tgmppi::scoreRolloutAgainstPredictions(
        tr.x.data() + r * 17, tr.y.data() + r * 17, tr.yaws.data() + r * 17,
        17, obstacles, p);
      const float expected = 3.81f * (score.collides ?
        p.collision_cost + p.penetration_cost * score.penetration : score.repulsive) / 17;
      EXPECT_EQ(contacts[r] != 0, score.collides);
      EXPECT_NEAR(result(r), expected, std::max(0.005f, std::fabs(expected) * 2e-5f));
    }
    EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 0u);
    EXPECT_EQ(gpu.gpu_batch_.transfers().cost_downloads, 1u);
  }
  gpu.beginDevice();
  gpu.gpu_batch_.dynamicCosts({}, {}, 3.81f, 1);
  xt::xtensor<float, 1> result = xt::zeros<float>({31});
  std::vector<uint8_t> contacts;
  gpu.gpu_batch_.finishCritics(result, &contacts);
  EXPECT_EQ(std::count(contacts.begin(), contacts.end(), uint8_t{1}), 0);
  EXPECT_EQ(xt::sum(result)(), 0.0f);
}

TEST(GpuBatch, CriticsAccumulateOnceAndGroupedUpdatesMatchCpu)
{
  if (!torch::cuda::is_available()) {GTEST_SKIP() << "CUDA device unavailable";}
  for (bool grouped : {false, true}) {
    for (bool corrupt : {false, true}) {
      BatchOptimizer cpu, gpu;
      cpu.configure(true); gpu.configure(true);
      if (corrupt) {
        cpu.state_.cvx(1, 3) = std::numeric_limits<float>::quiet_NaN();
        gpu.state_.cvx(1, 3) = std::numeric_limits<float>::quiet_NaN();
      }
      cpu.groups(grouped); gpu.groups(grouped);
      cpu.cpuRollout(); gpu.deviceRollout();
      Goal goal; Constraint constraint;
      goal.score(cpu.critics_data_); goal.score(gpu.critics_data_);
      constraint.score(cpu.critics_data_); constraint.score(gpu.critics_data_);
      EXPECT_EQ(gpu.gpu_batch_.transfers().cost_downloads, 0u);
      gpu.gpu_batch_.finishCritics(gpu.costs_, nullptr);
      EXPECT_EQ(gpu.gpu_batch_.transfers().cost_downloads, 1u);
      cpu.updateControlSequence(); gpu.updateControlSequence();
      expectTensorNear(cpu.control_sequence_.vx, gpu.control_sequence_.vx);
      expectTensorNear(cpu.control_sequence_.wz, gpu.control_sequence_.wz);
      expectTensorNear(cpu.control_sequence_.vy, gpu.control_sequence_.vy);
      EXPECT_EQ(cpu.selected_mode_key_, gpu.selected_mode_key_);
      ASSERT_EQ(cpu.mode_nominals_.size(), gpu.mode_nominals_.size());
      for (const auto & entry : cpu.mode_nominals_) {
        const auto & actual = gpu.mode_nominals_.at(entry.first);
        for (size_t t = 0; t < entry.second.first.size(); ++t) {
          EXPECT_NEAR(entry.second.first[t], actual.first[t], 2e-5f);
          EXPECT_NEAR(entry.second.second[t], actual.second[t], 2e-5f);
        }
      }
      EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 0u);
      EXPECT_EQ(gpu.gpu_batch_.transfers().mean_downloads, 1u);
      EXPECT_EQ(gpu.gpu_batch_.transfers().control_uploads, corrupt ? 2u : 1u);
    }
  }
}

TEST(GpuBatch, CpuPluginReceivesCurrentStateAndAccumulatedCosts)
{
  if (!torch::cuda::is_available()) {GTEST_SKIP() << "CUDA device unavailable";}
  BatchOptimizer cpu, gpu;
  cpu.configure(); gpu.configure(); cpu.cpuRollout(); gpu.deviceRollout();
  Goal goal;
  goal.score(cpu.critics_data_);
  Manager manager;
  manager.add(std::make_unique<Goal>());
  manager.add(std::make_unique<CpuConsumer>(cpu));
  manager.add(std::make_unique<Goal>());
  manager.evalTrajectoriesScores(gpu.critics_data_);
  cpu.costs_ += 3.0f;
  goal.score(cpu.critics_data_);
  expectTensorNear(cpu.costs_, gpu.costs_);
  EXPECT_EQ(gpu.gpu_batch_.transfers().host_snapshots, 1u);
  EXPECT_EQ(gpu.gpu_batch_.transfers().cost_downloads, 2u);
}
}  // namespace

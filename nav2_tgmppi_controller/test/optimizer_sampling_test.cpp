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

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "nav2_tgmppi_controller/optimizer.hpp"

namespace
{
struct Rows
{
  unsigned int start;
  unsigned int count;
  int key;
};

class SamplingOptimizer : public tgmppi::Optimizer
{
public:
  using Optimizer::settings_;
  using Optimizer::state_;
  using Optimizer::costs_;
  using Optimizer::control_sequence_;
  using Optimizer::selected_mode_key_;
  using Optimizer::pending_mode_count_;
  using Optimizer::mode_nominals_;
  using Optimizer::updateControlSequence;
  using Optimizer::generateNoisedTrajectories;
  using Optimizer::enforceDynamicCollisionSafety;
  using Optimizer::finalSequenceIsDynamicallySafe;
  using Optimizer::ancillary_mode_samples_;
  using Optimizer::flow_wait_samples_;
  using Optimizer::dynamic_collision_rows_;
  using Optimizer::tracked_obstacles_snapshot_;
  using Optimizer::critics_data_;

  void configure(unsigned int batch, const std::string & allocation, bool wait)
  {
    settings_.batch_size = batch;
    settings_.time_steps = 6;
    settings_.model_dt = 0.05f;
    settings_.temperature = 0.3f;
    settings_.gamma = 0.0f;
    settings_.base_constraints = {2.0f, -2.0f, 2.0f, 2.0f};
    settings_.sampling_std = {0.1f, 0.1f, 0.1f};
    settings_.tgmppi_group_allocation = allocation;
    settings_.tgmppi_grouped_update = true;
    settings_.tgmppi_shadow_mode = false;
    settings_.tgmppi_bias_enabled = true;
    settings_.tgmppi_mode_warm_start = 1.0f;
    settings_.tgmppi_debug = false;
    settings_.tgmppi_ancillary_debug = false;
    settings_.flow_wait_enabled = wait;
    settings_.flow_wait_fraction = 0.2f;
    settings_.flow_reflood_every = 1;
    motion_model_ = std::make_shared<tgmppi::DiffDriveMotionModel>();
    costmap_ = &map_;
    path_.reset(0);
    reset();
    state_.pose.pose.orientation.w = 1.0;
    flow_path_blocked_now_ = wait;
  }

  static float proposalV(std::size_t m, unsigned int t)
  {
    return 0.05f * (m + 1) + 0.001f * t;
  }
  static float proposalW(std::size_t m, unsigned int t)
  {
    return -0.03f * (m + 1) - 0.001f * t;
  }
  static float noise(unsigned int row) {return 0.0001f * (row + 1);}

  void sample(const std::vector<bool> & valid, float fraction)
  {
    clearAncillaryModeState();
    control_sequence_.vx.fill(0.4f);
    control_sequence_.wz.fill(-0.1f);
    for (unsigned int r = 0; r < settings_.batch_size; ++r) {
      for (unsigned int t = 0; t < settings_.time_steps; ++t) {
        state_.cvx(r, t) = 0.4f + noise(r);
        state_.cwz(r, t) = -0.1f - noise(r);
      }
    }
    std::vector<std::vector<float>> v(valid.size()), w(valid.size());
    std::vector<float> promises(valid.size());
    for (std::size_t m = 0; m < valid.size(); ++m) {
      ancillary_mode_valid_[m] = valid[m];
      ancillary_mode_key_[m] = m < kPodSlots ? static_cast<int>(m) :
        kSpacetimeKeyBase + static_cast<int>(m);
      promises[m] = 0.2f * m;
      for (unsigned int t = 0; t < settings_.time_steps; ++t) {
        v[m].push_back(proposalV(m, t));
        w[m].push_back(proposalW(m, t));
      }
    }
    allocateModeSamples(v, w, valid, promises, fraction);
  }

  std::vector<Rows> rows() const
  {
    std::vector<Rows> result;
    unsigned int end = 0;
    for (std::size_t m = 0; m < ancillary_mode_samples_.size(); ++m) {
      if (ancillary_mode_samples_[m] == 0) {continue;}
      result.push_back({ancillary_mode_row_start_[m], ancillary_mode_samples_[m],
        ancillary_mode_key_[m]});
      end = ancillary_mode_row_start_[m] + ancillary_mode_samples_[m];
    }
    if (flow_wait_samples_) {
      result.push_back({end, flow_wait_samples_, kWaitModeKey});
      end += flow_wait_samples_;
    }
    if (end < settings_.batch_size) {
      result.push_back({end, settings_.batch_size - end, kFallbackModeKey});
    }
    return result;
  }

  void expectAllocation(const std::vector<bool> & valid) const
  {
    unsigned int row = 0;
    for (std::size_t m = 0; m < ancillary_mode_samples_.size(); ++m) {
      const unsigned int count = ancillary_mode_samples_[m];
      if (m >= valid.size() || !valid[m]) {EXPECT_EQ(count, 0u);}
      if (!count) {continue;}
      ASSERT_EQ(ancillary_mode_row_start_[m], row);
      ASSERT_LE(count, settings_.batch_size - row);
      const unsigned int end = row + count;
      for (; row < end; ++row) {
        for (unsigned int t = 0; t < settings_.time_steps; ++t) {
          EXPECT_NEAR(state_.cvx(row, t), proposalV(m, t) + noise(row), 1e-6f);
          EXPECT_NEAR(state_.cwz(row, t), proposalW(m, t) - noise(row), 1e-6f);
        }
      }
    }
    ASSERT_LE(flow_wait_samples_, settings_.batch_size - row);
    const unsigned int wait_end = row + flow_wait_samples_;
    for (; row < wait_end; ++row) {
      for (unsigned int t = 0; t < settings_.time_steps; ++t) {
        EXPECT_NEAR(state_.cvx(row, t), noise(row), 1e-6f);
        EXPECT_NEAR(state_.cwz(row, t), -noise(row), 1e-6f);
      }
    }
    for (; row < settings_.batch_size; ++row) {
      for (unsigned int t = 0; t < settings_.time_steps; ++t) {
        EXPECT_NEAR(state_.cvx(row, t), 0.4f + noise(row), 1e-6f);
        EXPECT_NEAR(state_.cwz(row, t), -0.1f - noise(row), 1e-6f);
      }
    }
  }

  void favor(const Rows & group)
  {
    costs_.fill(100.0f);
    for (unsigned int r = group.start; r < group.start + group.count; ++r) {
      costs_(r) = 0.1f * (r - group.start);
    }
  }

  void enableDynamicSafety()
  {
    critics_data_.dynamic_obstacle_params = tgmppi::DynamicObstacleCostParams{};
    tracked_obstacles_snapshot_ = {{0.0f, 0.0f, 0.0f, 0.0f, 0.25f}};
    dynamic_collision_rows_.assign(settings_.batch_size, 0u);
  }

private:
  nav2_costmap_2d::Costmap2D map_{10, 10, 0.1, 0.0, 0.0, 0};
};

TEST(OptimizerSampling, ContiguousRowsPreserveNoiseForZeroThroughSevenModes)
{
  for (const std::string allocation : {"legacy", "equal"}) {
    for (bool wait : {false, true}) {
      for (unsigned int batch : {1u, 2u, 8u, 31u}) {
        SamplingOptimizer optimizer;
        optimizer.configure(batch, allocation, wait);
        for (std::size_t n = 0; n <= 7; ++n) {
          for (float fraction : {0.0f, 0.2f, 1.0f}) {
            SCOPED_TRACE(::testing::Message() << allocation << " wait=" << wait <<
              " batch=" << batch << " modes=" << n << " fraction=" << fraction);
            const std::vector<bool> valid(n, true);
            optimizer.sample(valid, fraction);
            optimizer.expectAllocation(valid);
          }
        }
      }
    }
  }
}

TEST(OptimizerSampling, RejectedAndEmptySlotsReceiveNoRows)
{
  for (const std::string allocation : {"legacy", "equal"}) {
    for (bool wait : {false, true}) {
      SamplingOptimizer optimizer;
      optimizer.configure(31, allocation, wait);
      for (const std::vector<bool> & valid : {
          std::vector<bool>{true, false, true, false, false, true, false},
          std::vector<bool>(7, false)})
      {
        optimizer.sample(valid, 0.6f);
        optimizer.expectAllocation(valid);
      }
    }
  }
}

TEST(OptimizerSampling, EqualAllocationIncludesFivePodsTwoExtrasWaitAndFallback)
{
  SamplingOptimizer optimizer;
  optimizer.configure(90, "equal", true);
  optimizer.sample(std::vector<bool>(7, true), 1.0f);
  const auto groups = optimizer.rows();
  ASSERT_EQ(groups.size(), 9u);
  for (const auto & group : groups) {EXPECT_EQ(group.count, 10u);}
}

TEST(OptimizerSampling, EqualAllocationBalancesIntegerRemainders)
{
  for (bool wait : {false, true}) {
    for (unsigned int batch : {1u, 2u, 8u, 31u, 100u}) {
      SamplingOptimizer optimizer;
      optimizer.configure(batch, "equal", wait);
      for (std::size_t n = 0; n <= 7; ++n) {
        SCOPED_TRACE(::testing::Message() << "wait=" << wait << " batch=" << batch <<
          " modes=" << n);
        optimizer.sample(std::vector<bool>(n, true), 1.0f);
        std::vector<unsigned int> counts;
        unsigned int used = optimizer.flow_wait_samples_;
        for (std::size_t m = 0; m < n; ++m) {
          counts.push_back(optimizer.ancillary_mode_samples_[m]);
          used += counts.back();
        }
        if (wait) {counts.push_back(optimizer.flow_wait_samples_);}
        ASSERT_LE(used, batch);
        counts.push_back(batch - used);
        const auto limits = std::minmax_element(counts.begin(), counts.end());
        EXPECT_LE(*limits.second - *limits.first, 1u);
      }
    }
  }
}

TEST(OptimizerSampling, EveryGroupCanWinAndUsesItsOwnWeightedMean)
{
  for (std::size_t winner = 0; winner < 9; ++winner) {
    SamplingOptimizer optimizer;
    optimizer.configure(90, "equal", true);
    optimizer.sample(std::vector<bool>(7, true), 1.0f);
    const auto groups = optimizer.rows();
    ASSERT_EQ(groups.size(), 9u);
    const auto & group = groups[winner];
    optimizer.favor(group);
    std::vector<double> vx(6, 0.0), wz(6, 0.0);
    double total = 0.0;
    for (unsigned int r = group.start; r < group.start + group.count; ++r) {
      const double weight = std::exp(-static_cast<double>(optimizer.costs_(r)) / 0.3);
      total += weight;
      for (unsigned int t = 0; t < 6; ++t) {
        vx[t] += weight * optimizer.state_.cvx(r, t);
        wz[t] += weight * optimizer.state_.cwz(r, t);
      }
    }
    optimizer.updateControlSequence();
    EXPECT_EQ(optimizer.selected_mode_key_, group.key);
    // The production update reconstructed all nine groups, including both extras.
    EXPECT_EQ(optimizer.mode_nominals_.size(), 9u);
    for (unsigned int t = 0; t < 6; ++t) {
      EXPECT_NEAR(optimizer.control_sequence_.vx(t), vx[t] / total, 1e-6);
      EXPECT_NEAR(optimizer.control_sequence_.wz(t), wz[t] / total, 1e-6);
    }
  }
}

TEST(OptimizerSampling, FreeEnergyNormalizesUnequalGroupSizes)
{
  SamplingOptimizer optimizer;
  optimizer.configure(100, "legacy", false);
  optimizer.sample({true}, 0.1f);  // Ten guided and ninety fallback samples.
  ASSERT_EQ(optimizer.rows().size(), 2u);
  optimizer.costs_.fill(5.0f);
  optimizer.selected_mode_key_ = 0;
  optimizer.updateControlSequence();
  EXPECT_EQ(optimizer.selected_mode_key_, 0);  // Equal costs: bigger group must not win.
}

TEST(OptimizerSampling, SwitchingRequiresConsecutiveConfirmation)
{
  SamplingOptimizer optimizer;
  optimizer.configure(30, "equal", false);
  optimizer.settings_.tgmppi_mode_confirm_cycles = 2;
  optimizer.sample({true, true}, 1.0f);
  const auto groups = optimizer.rows();
  ASSERT_EQ(groups.size(), 3u);
  optimizer.favor(groups[0]);
  optimizer.updateControlSequence();
  EXPECT_EQ(optimizer.selected_mode_key_, groups[0].key);
  optimizer.favor(groups[1]);
  optimizer.updateControlSequence();
  EXPECT_EQ(optimizer.selected_mode_key_, groups[0].key);
  EXPECT_EQ(optimizer.pending_mode_count_, 1u);
  optimizer.favor(groups[1]);
  optimizer.updateControlSequence();
  EXPECT_EQ(optimizer.selected_mode_key_, groups[1].key);
}

TEST(OptimizerSampling, GuidanceOffOrUnreadyFieldClearsActualCycleAllocation)
{
  for (bool bias_enabled : {false, true}) {
    SamplingOptimizer optimizer;
    optimizer.configure(31, "equal", true);
    optimizer.sample(std::vector<bool>(7, true), 1.0f);
    ASSERT_GT(optimizer.ancillary_mode_samples_[0], 0u);
    optimizer.settings_.tgmppi_bias_enabled = bias_enabled;
    // Empty plan: enabled guidance returns early; disabled guidance uses plain sampling.
    optimizer.generateNoisedTrajectories();
    for (auto count : optimizer.ancillary_mode_samples_) {EXPECT_EQ(count, 0u);}
    EXPECT_EQ(optimizer.flow_wait_samples_, 0u);
    optimizer.costs_.fill(0.0f);
    double expected = 0.0;
    for (unsigned int r = 0; r < 31; ++r) {expected += optimizer.state_.cvx(r, 0);}
    optimizer.updateControlSequence();
    EXPECT_NEAR(optimizer.control_sequence_.vx(0), expected / 31, 1e-6);
  }
}

TEST(OptimizerSampling, AllDynamicallyCollidingRowsStopInsteadOfSelectingLeastBadCollision)
{
  SamplingOptimizer optimizer;
  optimizer.configure(30, "equal", false);
  optimizer.sample({true}, 1.0f);
  optimizer.enableDynamicSafety();
  optimizer.control_sequence_.vx.fill(0.7f);
  optimizer.control_sequence_.wz.fill(0.2f);
  optimizer.costs_.fill(0.0f);
  std::fill(
    optimizer.dynamic_collision_rows_.begin(), optimizer.dynamic_collision_rows_.end(), 1u);

  EXPECT_TRUE(optimizer.enforceDynamicCollisionSafety());
  for (unsigned int t = 0; t < 6; ++t) {
    EXPECT_FLOAT_EQ(optimizer.control_sequence_.vx(t), 0.0f);
    EXPECT_FLOAT_EQ(optimizer.control_sequence_.wz(t), 0.0f);
  }
}

TEST(OptimizerSampling, AllCollidingGuidedModeCannotWinGroupedSelection)
{
  SamplingOptimizer optimizer;
  optimizer.configure(90, "equal", false);
  optimizer.sample({true, true}, 1.0f);
  const auto groups = optimizer.rows();
  ASSERT_EQ(groups.size(), 3u);
  optimizer.enableDynamicSafety();
  optimizer.favor(groups[0]);  // Would win without an explicit infeasibility signal.
  for (unsigned int r = groups[0].start; r < groups[0].start + groups[0].count; ++r) {
    optimizer.dynamic_collision_rows_[r] = 1u;
  }

  EXPECT_FALSE(optimizer.enforceDynamicCollisionSafety());
  optimizer.updateControlSequence();
  EXPECT_NE(optimizer.selected_mode_key_, groups[0].key);
}

TEST(OptimizerSampling, FinalSmoothedSequenceUsesTheDynamicCollisionGeometry)
{
  SamplingOptimizer optimizer;
  optimizer.configure(8, "equal", false);
  optimizer.enableDynamicSafety();
  optimizer.tracked_obstacles_snapshot_[0] = {0.3f, 0.0f, 0.0f, 0.0f, 0.25f};
  optimizer.control_sequence_.vx.fill(0.5f);
  optimizer.control_sequence_.wz.fill(0.0f);

  EXPECT_FALSE(optimizer.finalSequenceIsDynamicallySafe());
}

TEST(OptimizerSampling, FinalVetoCatchesContactBetweenCriticSamplingPoints)
{
  SamplingOptimizer optimizer;
  optimizer.configure(8, "equal", false);
  optimizer.enableDynamicSafety();
  auto & params = *optimizer.critics_data_.dynamic_obstacle_params;
  params.point_step = 3u;  // The production YAML's cost-sampling interval.
  params.disc_offset = 0.0f;
  params.disc_radius = 0.01f;
  optimizer.tracked_obstacles_snapshot_[0] = {0.05f, 0.0f, 0.0f, 0.0f, 0.01f};
  optimizer.state_.speed.linear.x = 0.5;
  optimizer.control_sequence_.vx.fill(0.5f);
  optimizer.control_sequence_.wz.fill(0.0f);

  // At model steps 1 and 4 the centre is 0.025 m and 0.10 m; at step 2 it
  // crosses the obstacle centre. A stride-three check misses that contact.
  const float x[] = {0.025f, 0.05f, 0.075f, 0.10f};
  const float y[] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float yaw[] = {0.0f, 0.0f, 0.0f, 0.0f};
  EXPECT_FALSE(tgmppi::scoreRolloutAgainstPredictions(
    x, y, yaw, 4u, optimizer.tracked_obstacles_snapshot_, params).collides);
  EXPECT_FALSE(optimizer.finalSequenceIsDynamicallySafe());
}
}  // namespace

#include <memory>

#include <gtest/gtest.h>

#include "nav2_tgmppi_controller/optimizer.hpp"

namespace
{
class TestOptimizer : public tgmppi::Optimizer
{
public:
  using Optimizer::clearAncillaryModeState;
  using Optimizer::fallback;

  void setUpRetry()
  {
    settings_.batch_size = 8;
    settings_.time_steps = 4;
    settings_.retry_attempt_limit = 1;
    motion_model_ = std::make_shared<tgmppi::DiffDriveMotionModel>();
  }

  void seedModes()
  {
    ancillary_mode_valid_.fill(true);
    ancillary_mode_samples_.fill(1);
    ancillary_mode_row_start_.fill(2);
    ancillary_mode_rejoin_prior_.fill(3.0f);
    flow_wait_samples_ = 1;
    for (auto & rollout : ancillary_rollout_x_) {rollout.push_back(1.0f);}
    for (auto & rollout : ancillary_rollout_y_) {rollout.push_back(1.0f);}
  }

  void expectModesCleared() const
  {
    for (std::size_t m = 0; m < kMaxAncillaryModes; ++m) {
      EXPECT_FALSE(ancillary_mode_valid_[m]);
      EXPECT_EQ(ancillary_mode_samples_[m], 0u);
      EXPECT_EQ(ancillary_mode_row_start_[m], 0u);
      EXPECT_FLOAT_EQ(ancillary_mode_rejoin_prior_[m], 0.0f);
      EXPECT_TRUE(ancillary_rollout_x_[m].empty());
      EXPECT_TRUE(ancillary_rollout_y_[m].empty());
    }
    EXPECT_EQ(flow_wait_samples_, 0u);
  }

  void setFailure() {critics_data_.fail_flag = true;}
  bool failed() const {return critics_data_.fail_flag;}
  static constexpr std::size_t podSlots() {return kPodSlots;}
  static constexpr std::size_t modeSlots() {return kMaxAncillaryModes;}
};

TEST(OptimizerState, ClearsEveryModeIncludingBothSpaceTimeExtras)
{
  static_assert(TestOptimizer::modeSlots() == TestOptimizer::podSlots() + 2);
  TestOptimizer optimizer;
  optimizer.seedModes();
  optimizer.clearAncillaryModeState();
  optimizer.expectModesCleared();
}

TEST(OptimizerState, FailedRetryClearsCriticFailure)
{
  TestOptimizer optimizer;
  optimizer.setUpRetry();
  optimizer.setFailure();
  EXPECT_TRUE(optimizer.fallback(true));
  EXPECT_FALSE(optimizer.failed());
  EXPECT_FALSE(optimizer.fallback(false));
}
}  // namespace

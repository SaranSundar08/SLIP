#include <memory>
#include <cmath>
#include <limits>

#include <gtest/gtest.h>

#include "nav2_tgmppi_controller/optimizer.hpp"
#include "nav2_tgmppi_controller/tools/obstacle_observation.hpp"

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

class ObstacleSnapshotOptimizer : public tgmppi::Optimizer
{
public:
  void configure(bool required)
  {
    settings_.tgmppi_obstacle_timeout = 0.5f;
    settings_.tgmppi_require_obstacle_tracking = required;
    spacetime_obstacles_ = {{1.0f, 2.0f, 2.0f, 0.0f, 0.25f}};
    spacetime_obstacle_stamps_ = {rclcpp::Time(0, 0, RCL_ROS_TIME)};
    spacetime_obstacle_received_ = {false};
  }

  void setObservation(const rclcpp::Time & stamp)
  {
    spacetime_obstacle_stamps_[0] = stamp;
    spacetime_obstacle_received_[0] = true;
  }

  void snapshot(const rclcpp::Time & now) {snapshotTrackedObstaclesAt(now);}
  bool trackingFault() const {return obstacle_tracking_fault_;}
  const std::vector<tgmppi::SpaceTimeObstacle> & obstacles() const
  {
    return tracked_obstacles_snapshot_;
  }
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

TEST(ObstacleObservation, ConvertsPoseAndChildFrameVelocityToCostmapFrame)
{
  nav_msgs::msg::Odometry msg;
  msg.header.frame_id = "odom";
  msg.child_frame_id = "body";
  msg.pose.pose.position.x = 1.0;
  msg.pose.pose.position.y = 2.0;
  msg.pose.pose.orientation.w = 1.0;
  msg.twist.twist.linear.x = 1.0;
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = "map";
  tf.child_frame_id = "odom";
  tf.transform.translation.x = 10.0;
  tf.transform.rotation.z = std::sqrt(0.5);
  tf.transform.rotation.w = std::sqrt(0.5);

  const auto observation = tgmppi::obstacleFromOdometry(msg, "map", &tf, 0.25f);
  ASSERT_TRUE(observation.has_value());
  EXPECT_NEAR(observation->x, 8.0f, 1e-5f);
  EXPECT_NEAR(observation->y, 1.0f, 1e-5f);
  EXPECT_NEAR(observation->vx, 0.0f, 1e-5f);
  EXPECT_NEAR(observation->vy, 1.0f, 1e-5f);
  EXPECT_FALSE(tgmppi::obstacleFromOdometry(msg, "map", nullptr, 0.25f).has_value());

  // Twist is expressed in the child frame: a turned child rotates its
  // forward speed even when its pose is already in the target map frame.
  msg.header.frame_id = "map";
  msg.pose.pose.orientation.z = std::sqrt(0.5);
  msg.pose.pose.orientation.w = std::sqrt(0.5);
  const auto turned = tgmppi::obstacleFromOdometry(msg, "map", nullptr, 0.25f);
  ASSERT_TRUE(turned.has_value());
  EXPECT_NEAR(turned->vx, 0.0f, 1e-5f);
  EXPECT_NEAR(turned->vy, 1.0f, 1e-5f);
}

TEST(ObstacleObservation, RejectsUnknownFramesAndNonfiniteStates)
{
  nav_msgs::msg::Odometry msg;
  msg.header.frame_id = "map";
  msg.pose.pose.orientation.w = 1.0;
  msg.twist.twist.linear.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(tgmppi::obstacleFromOdometry(msg, "map", nullptr, 0.25f).has_value());
  msg.twist.twist.linear.x = 0.0;
  msg.header.frame_id.clear();
  EXPECT_FALSE(tgmppi::obstacleFromOdometry(msg, "map", nullptr, 0.25f).has_value());
}

TEST(ObstacleObservation, FreshSnapshotAdvancesToControllerTimeAndStaleDataFaults)
{
  const rclcpp::Time now(10, 0, RCL_ROS_TIME);
  ObstacleSnapshotOptimizer optimizer;
  optimizer.configure(true);
  optimizer.snapshot(now);
  EXPECT_TRUE(optimizer.trackingFault());  // Dynamic mode waits for a live stream.

  optimizer.setObservation(now - rclcpp::Duration::from_seconds(0.2));
  optimizer.snapshot(now);
  ASSERT_EQ(optimizer.obstacles().size(), 1u);
  EXPECT_FALSE(optimizer.trackingFault());
  EXPECT_NEAR(optimizer.obstacles()[0].x, 1.4f, 0.05f);

  optimizer.setObservation(now - rclcpp::Duration::from_seconds(0.6));
  optimizer.snapshot(now);
  EXPECT_TRUE(optimizer.trackingFault());
  EXPECT_TRUE(optimizer.obstacles().empty());
}
}  // namespace

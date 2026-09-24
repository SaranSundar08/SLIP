#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__OBSTACLE_OBSERVATION_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__OBSTACLE_OBSERVATION_HPP_

#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include "nav2_tgmppi_controller/tools/space_time_search.hpp"

namespace tgmppi
{

// The package uses -ffast-math, under which std::isfinite may be optimized to
// true even for NaN. Inspect the IEEE-754 exponent instead.
inline bool badObservationNumber(double value)
{
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull;
}

inline bool badObservationNumber(float value)
{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7F800000u) == 0x7F800000u;
}

// Convert one Odometry observation to the costmap's global frame. Odometry
// defines pose in header.frame_id and twist in child_frame_id. The caller
// obtains target_from_header at the message stamp without blocking for TF.
inline std::optional<SpaceTimeObstacle> obstacleFromOdometry(
  const nav_msgs::msg::Odometry & msg, const std::string & target_frame,
  const geometry_msgs::msg::TransformStamped * target_from_header, float radius)
{
  if (msg.header.frame_id.empty() || target_frame.empty() ||
    badObservationNumber(radius) || radius <= 0.0f)
  {
    return std::nullopt;
  }
  if (msg.header.frame_id != target_frame &&
    (target_from_header == nullptr ||
    target_from_header->header.frame_id != target_frame ||
    target_from_header->child_frame_id != msg.header.frame_id))
  {
    return std::nullopt;
  }

  auto pose = msg.pose.pose;
  auto velocity = msg.twist.twist.linear;
  if (!msg.child_frame_id.empty() && msg.child_frame_id != msg.header.frame_id) {
    geometry_msgs::msg::TransformStamped child_to_header;
    child_to_header.transform.rotation = pose.orientation;
    geometry_msgs::msg::Vector3 rotated;
    tf2::doTransform(velocity, rotated, child_to_header);
    velocity = rotated;
  }
  if (msg.header.frame_id != target_frame) {
    geometry_msgs::msg::Pose transformed_pose;
    geometry_msgs::msg::Vector3 transformed_velocity;
    tf2::doTransform(pose, transformed_pose, *target_from_header);
    tf2::doTransform(velocity, transformed_velocity, *target_from_header);
    pose = transformed_pose;
    velocity = transformed_velocity;
  }
  if (badObservationNumber(pose.position.x) || badObservationNumber(pose.position.y) ||
    badObservationNumber(velocity.x) || badObservationNumber(velocity.y))
  {
    return std::nullopt;
  }
  return SpaceTimeObstacle{
    static_cast<float>(pose.position.x), static_cast<float>(pose.position.y),
    static_cast<float>(velocity.x), static_cast<float>(velocity.y), radius};
}

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__OBSTACLE_OBSERVATION_HPP_

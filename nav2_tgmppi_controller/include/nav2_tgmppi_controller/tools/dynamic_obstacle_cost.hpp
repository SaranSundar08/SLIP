// Time-indexed collision cost against predicted moving obstacles (2026-09-15).
//
// This is the MPPI form of T-MPC's dynamic collision constraint (de Groot et al.,
// Eq. 2d/14): rollout point j is compared with each obstacle's PREDICTED position
// at that point's own time t_j = (j + 1) * model_dt (constant-velocity model),
// not with a frozen costmap snapshot. Header-only and ROS-free so the critic and
// its unit test run exactly the same code.
#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__DYNAMIC_OBSTACLE_COST_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__DYNAMIC_OBSTACLE_COST_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "nav2_tgmppi_controller/tools/space_time_search.hpp"

namespace tgmppi
{

struct DynamicObstacleCostParams
{
  float model_dt{0.05f};
  // Robot footprint as two discs centred at +-disc_offset along the heading.
  // 0.84 x 0.68 m rectangle: offset 0.21, radius sqrt(0.21^2 + 0.34^2) ~= 0.40.
  float disc_offset{0.21f};
  float disc_radius{0.40f};
  float soft_distance{0.5f};      // repulsive band beyond contact (m)
  float critical_cost{300.0f};    // per-point cost at the contact edge (scaled by q^2)
  float collision_cost{1.0e6f};   // any predicted contact
  float penetration_cost{1.0e4f}; // per metre of summed penetration: grades colliding rollouts
  float max_prediction_time{10.0f};
  std::size_t point_step{1};
};

struct DynamicObstacleCostResult
{
  float repulsive{0.0f};
  bool collides{false};
  float penetration{0.0f};
};

// x/y/yaw: one rollout, T points; point j is the state after j+1 controls.
inline DynamicObstacleCostResult scoreRolloutAgainstPredictions(
  const float * x, const float * y, const float * yaw, std::size_t T,
  const std::vector<SpaceTimeObstacle> & obstacles, const DynamicObstacleCostParams & p)
{
  DynamicObstacleCostResult r;
  const std::size_t step = std::max<std::size_t>(1, p.point_step);
  for (std::size_t j = 0; j < T; j += step) {
    const float t = std::min(static_cast<float>(j + 1) * p.model_dt, p.max_prediction_time);
    const float c = std::cos(yaw[j]);
    const float s = std::sin(yaw[j]);
    float dmin = std::numeric_limits<float>::max();
    for (const auto & o : obstacles) {
      const float ox = o.x + o.vx * t;
      const float oy = o.y + o.vy * t;
      const float reach = p.disc_radius + o.radius;
      for (const float off : {p.disc_offset, -p.disc_offset}) {
        const float d = std::hypot(x[j] + c * off - ox, y[j] + s * off - oy) - reach;
        dmin = std::min(dmin, d);
      }
    }
    if (dmin < 0.0f) {
      r.collides = true;
      r.penetration += -dmin;       // keep going: total depth grades the least-bad escape
    } else if (dmin < p.soft_distance) {
      const float q = (p.soft_distance - dmin) / p.soft_distance;
      r.repulsive += p.critical_cost * q * q;
    }
  }
  return r;
}

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__DYNAMIC_OBSTACLE_COST_HPP_

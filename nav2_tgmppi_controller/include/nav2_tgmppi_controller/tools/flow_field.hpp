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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__FLOW_FIELD_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__FLOW_FIELD_HPP_

#include <cstdint>
#include <utility>
#include <vector>

#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"

namespace tgmppi
{

/**
 * @class tgmppi::FlowField
 * @brief The tgmppi "water level": geodesic distance-to-goal flooded through
 * the robot-passable free space of the LOCAL costmap (the tgmppi body -- the
 * rolling window IS the finite sensing diameter). Free = cost below
 * INSCRIBED_INFLATED_OBSTACLE, so the inflation layer's footprint inflation
 * provides a clearance proxy rather than an oriented polygon guarantee.
 * Diagonal transitions require both adjacent cardinal cells to be free.
 * Ancillary rollout validation separately checks the oriented footprint.
 *
 * Boundary condition when the plan's last transformed point sits at the
 * window edge (goal beyond the window): every free boundary cell is seeded
 * with promise = distance-to-plan + remaining plan length (path_seed=true,
 * the planner demoted from contract to boundary hint), or straight-line
 * distance to the plan end (path_seed=false, planner-free mode). A wrong
 * plan only mis-ranks where water enters; the flood inside runs over the
 * true costmap and never routes through dry gaps.
 */
class FlowField
{
public:
  FlowField() = default;

  /**
   * @brief Re-flood the field from the current local costmap.
   * @param costmap Local (rolling window) costmap, already inflated
   * @param path Transformed pruned plan in costmap frame; its last point is
   * the goal if inside the window, else the window-exit used for seeding
   * @param path_seed Seed boundary by distance-to-plan + remaining length
   * (true) or by Euclidean distance to the plan end (false)
   * @param viscosity Water physics: flood resistance multiplier near walls
   * (1 + viscosity at inscribed cost, 1 in free space). Water then "flows
   * more freely where there is space": levels prefer wide channels and
   * center in them, while narrow-but-necessary gaps still flood. 0 recovers
   * the pure-geodesic flood. Sandbox-validated ~1.5 for windowed fields.
   * @param max_pseudopods Cap on spatially-distinct membrane exits kept per
   * flood (legacy default 3, matching the sandbox). Each accepted pseudopod
   * becomes its own MPPI sample group + ancillary rollout + collision check,
   * so raising this trades real-time headroom for more simultaneously-
   * representable route alternatives -- not yet validated at values > 3.
   */
  void build(
    const nav2_costmap_2d::Costmap2D & costmap,
    const models::Path & path, bool path_seed, float viscosity,
    float robot_x, float robot_y, float body_radius,
    std::size_t max_pseudopods = 3);

  bool ready() const {return ready_;}

  /**
   * @brief Water level at a world point; dry or off-window cells return
   * dmax + penalty so they always score worse than any wet cell.
   */
  float distAt(float wx, float wy) const;

  /**
   * @brief Downhill unit vector at a world point.
   * @return false where undefined (dry cell) -- leave the sample unbiased
   */
  bool dirAt(float wx, float wy, float & dx, float & dy) const;

  /**
   * @brief Smooth downhill direction: normalized negative central-difference
   * gradient of the water level, sampled over +-3 cells. Continuous in angle
   * (unlike the 8-quantized dirAt), so a bias steering along it does not
   * flip between 45-degree steps cell to cell. Falls back to false where
   * the level is locally flat.
   */
  bool gradAt(float wx, float wy, float & dx, float & dy) const;

  double buildMs() const {return build_ms_;}

  // Dry/unreached cells hold a large FINITE sentinel (>= kWetLimit), never
  // IEEE inf: this package builds with -ffast-math, which strips isfinite()
  // and makes inf arithmetic produce NaNs.
  static constexpr float kWetLimit = 1.0e8f;

  // Grid accessors for debug visualization.
  int sizeX() const {return nx_;}
  int sizeY() const {return ny_;}
  float resolution() const {return res_;}
  float originX() const {return ox_;}
  float originY() const {return oy_;}
  float cellDist(int i, int j) const {return D_[idx(i, j)];}
  // Bulk accessor for GpuFlowFieldCritic: the whole [ny*nx] distance grid,
  // row-major (index = j*nx+i, matching idx()), for one-shot GPU upload
  // instead of a cellDist() call per point.
  const std::vector<float> & distGrid() const {return D_;}
  bool cellWet(int i, int j) const {return D_[idx(i, j)] < kWetLimit;}
  bool cellBody(int i, int j) const {return body_[idx(i, j)] != 0;}
  bool cellMembrane(int i, int j) const {return membrane_[idx(i, j)] != 0;}
  float dmax() const {return dmax_;}
  bool cellDir(int i, int j, float & dx, float & dy) const;
  const std::vector<std::vector<std::pair<float, float>>> & pseudopods() const
  {
    return pseudopods_;
  }
  const std::vector<float> & pseudopodPromises() const {return pseudopod_promises_;}
  std::size_t bodyCellCount() const {return body_cell_count_;}
  std::size_t membraneCellCount() const {return membrane_cell_count_;}

private:
  int idx(int i, int j) const {return j * nx_ + i;}
  bool worldToGridClamped(float wx, float wy, int & i, int & j) const;

  int nx_{0}, ny_{0};
  float res_{0.0f}, ox_{0.0f}, oy_{0.0f};
  float dmax_{0.0f};
  bool ready_{false};
  double build_ms_{0.0};
  std::vector<uint8_t> free_;
  std::vector<uint8_t> body_;
  std::vector<uint8_t> membrane_;
  std::vector<float> visc_;
  std::vector<float> body_dist_;
  std::vector<int> body_parent_;
  std::vector<float> D_;
  std::vector<int8_t> dir_;  // index into the 8-neighbor table, -1 = dead
  std::vector<std::vector<std::pair<float, float>>> pseudopods_;
  std::vector<float> pseudopod_promises_;
  std::size_t body_cell_count_{0};
  std::size_t membrane_cell_count_{0};
};

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__FLOW_FIELD_HPP_

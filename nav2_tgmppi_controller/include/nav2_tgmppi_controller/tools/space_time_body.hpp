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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_BODY_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_BODY_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "nav2_tgmppi_controller/tools/space_time_search.hpp"

namespace tgmppi
{

struct SpaceTimeBodyParams
{
  float horizon{3.0f};           // s; K = round(horizon / dt_layer) layers
  float dt_layer{0.25f};         // s
  float res{0.10f};              // m; grid speed is res / dt_layer (one cell per layer)
  float robot_r{0.34f};          // m; inscribed radius (obstacle clearance test)
  float terminal_weight{1.0f};   // weight of the static water level at the membrane
  float wait_epsilon{0.01f};     // m; per-waited-layer tie-break (time is priced by the
                                 // terminal water level, not by a tuned wait cost)
  float class_min_separation{0.3f};   // m; exits of two different homotopy classes
  float exit_separation{0.65f};       // m; exits filled by geometry (static blob rule)
  float max_regret{2.0f};        // m; another class must be within this promise of the best
  int max_pods{3};
};

/**
 * @class tgmppi::SpaceTimeBody
 * @brief The geodesic blob lifted into (x, y, t).
 *
 * BODY: free (x, y, t) states reachable from (robot, t=0) at grid speed
 * res/dt_layer, avoiding static obstacles and every predicted moving obstacle.
 * MEMBRANE: the reachable states on the last time layer (the light-cone rim).
 * PROMISE at a membrane exit: space-time cost G + terminal_weight * D_static(exit),
 * where D_static is the static water level (the flood of FlowField), i.e. the cost-to-go
 * beyond the horizon. PSEUDOPODS: min-cost (x, y, t) routes to exits that are distinct
 * by homotopy class (winding-number signature over all obstacles), then by exit
 * separation. Every edge advances one time layer, so the flood is a layered DP over a
 * DAG (no priority queue).
 *
 * ROS-free on purpose (takes std::function callbacks, like SpaceTimeSearch) so it can be
 * unit-tested with a plain compiler. Compiled with -ffast-math in the package: no
 * isfinite/inf; unreached cells hold the finite sentinel kUnreached.
 */
class SpaceTimeBody
{
public:
  using StaticFreeFn = std::function<bool (float x, float y)>;
  // Static water level at a world point (e.g. FlowField::distAt).
  using TerminalFn = std::function<float (float x, float y)>;

  static constexpr float kUnreached = 1.0e9f;

  void build(
    const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
    const TerminalFn & terminal, float start_x, float start_y, const SpaceTimeBodyParams & p);

  bool ready() const {return ready_;}

  // Pseudopods, best promise first. Routes hold one (x, y) per layer, so they feed the
  // existing resample() -> (v, w, x, y) mode builder unchanged. route.cost = promise.
  const std::vector<SpaceTimeRoute> & pods() const {return pods_;}
  const std::vector<float> & promises() const {return promises_;}
  // Per pseudopod: per relevant obstacle, -1 / 0 / +1 (winding sign if |lambda| >= 1/(4 pi)).
  const std::vector<std::vector<int8_t>> & signatures() const {return signatures_;}

  std::size_t bodyStates() const {return body_states_;}          // reached (cell, layer) states
  std::size_t membraneCells() const {return membrane_cells_;}    // reached cells on last layer
  std::size_t classesFound() const {return classes_found_;}      // distinct signatures on the rim
  double buildMs() const {return build_ms_;}

  int layers() const {return K_;}
  int side() const {return n_;}
  // Space-time cost G at grid cell (i, j) on layer k; kUnreached if not in the body.
  float cost(int i, int j, int k) const;
  float cellX(int i) const {return x0_ + static_cast<float>(i) * res_;}
  float cellY(int j) const {return y0_ + static_cast<float>(j) * res_;}

private:
  std::size_t at(int k, int i, int j) const
  {
    return (static_cast<std::size_t>(k) * n_ + j) * n_ + i;
  }

  bool ready_{false};
  int K_{0};
  int n_{0};
  float res_{0.1f};
  float x0_{0.0f};
  float y0_{0.0f};
  double build_ms_{0.0};
  std::size_t body_states_{0};
  std::size_t membrane_cells_{0};
  std::size_t classes_found_{0};
  std::vector<float> G_;
  std::vector<int8_t> parent_;   // offset index 0..8 of the predecessor; -1 = none
  std::vector<SpaceTimeRoute> pods_;
  std::vector<float> promises_;
  std::vector<std::vector<int8_t>> signatures_;
};

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_BODY_HPP_

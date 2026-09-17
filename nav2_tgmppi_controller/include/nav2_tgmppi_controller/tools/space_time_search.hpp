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

#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_SEARCH_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_SEARCH_HPP_

#include <functional>
#include <utility>
#include <vector>

namespace tgmppi
{

/**
 * @brief One moving obstacle: current ground-truth position + velocity,
 * constant-velocity extrapolated (see predict()). Multiple obstacles is a
 * small, safe generalization of amoeba_sandbox/spacetime.py's Phase 0
 * (single predict_obstacle() callback) -- that file's own docstring scopes
 * itself to "synthetic single-obstacle scenarios first", but dynamic_free()
 * checking against a list instead of one obstacle costs nothing extra to
 * implement and this project's test world already has two.
 */
struct SpaceTimeObstacle
{
  float x;
  float y;
  float vx;
  float vy;
  float radius;
};

/**
 * @brief One space-time route: (x, y) once per dt_layer-spaced time layer
 * from t=0 to arrival, or empty/infeasible if none was found within the
 * given horizon/window.
 */
struct SpaceTimeRoute
{
  std::vector<std::pair<float, float>> path;
  float cost{-1.0f};   // -1 sentinel for infeasible (avoids needing <limits> at call sites)
  bool feasible{false};
};

/**
 * @class tgmppi::SpaceTimeSearch
 * @brief Port of amoeba_sandbox/spacetime.py's time_expanded_search() /
 * two_route_search() -- a Dijkstra search over a discretized (x, y, t)
 * grid, avoiding static costmap obstacles and predicted moving obstacles,
 * used to construct genuine "wait for it to pass" vs "go around/before it
 * arrives" candidate routes past a moving obstacle.
 *
 * Deliberately CPU-only: this is a small (tens of thousands of nodes for
 * the sandbox's own default window=2.5m/res=0.10m/horizon=3.0s/
 * dt_layer=0.25s), inherently sequential priority-queue graph search, not
 * a batch of independent parallel trajectories like the MPPI critics --
 * there's no meaningful GPU parallelization for Dijkstra at this scale,
 * and every earlier lesson from this project's GPU port (small/serial
 * workloads lose to round-trip overhead) points the same way.
 *
 * Phase 0 scope, matching the sandbox's own staging ("this proves the
 * mechanism against synthetic scenarios first; integrating it into the
 * controller... is the next phase" -- spacetime.py's module docstring):
 * this class is the standalone search only. Wiring its output routes into
 * Optimizer's sampling as real extra pseudopod-like modes is a declared
 * next step, not done here.
 */
class SpaceTimeSearch
{
public:
  /** @brief Constant-velocity prediction, matching amoeba_sandbox/
   * dynabarn.py's predicted_moving_obs(): current position + velocity *
   * min(t, horizon) -- held at the horizon's value beyond it, not
   * extrapolated further. */
  static std::pair<float, float> predict(const SpaceTimeObstacle & obs, float t, float horizon);

  /** @brief True if a robot centered at (x, y) does not collide with a
   * static obstacle. Passed in rather than taking a Costmap2D directly so
   * this class (and its tests) don't need to depend on nav2_costmap_2d. */
  using StaticFreeFn = std::function<bool (float x, float y)>;

  /**
   * @brief Cheapest space-time path from (start_x, start_y) at t=0 to
   * within goal_tol of (goal_x, goal_y), through a local window, avoiding
   * static_free() violations and every obstacle in `obstacles`.
   * Direct port of time_expanded_search() (min_arrival_k/max_arrival_k
   * omitted: two_route_search(), the only caller this project wires up,
   * never overrides them from their full-horizon defaults).
   */
  static SpaceTimeRoute search(
    const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
    float start_x, float start_y, float goal_x, float goal_y, float robot_r,
    float horizon, float dt_layer, float res, float window, float goal_tol, float wait_cost,
    const SpaceTimeRoute * avoid_same_side_as = nullptr, float side_relevance = 0.8f);

  /**
   * @brief Port of two_route_search(): runs search() twice with a cheap
   * vs. expensive wait_cost so waiting-in-place and detouring-around each
   * become the true minimum-cost strategy in turn. Differing wait costs alone
   * was NOT enough -- both searches converged on the same passing class every
   * time (2026-09-16, log 36452: non-distinct 91/91), so the second search is
   * now additionally CONSTRAINED to the opposite side of the obstacle from the
   * first (avoid_same_side_as), which is what makes the two routes genuinely
   * distinct homotopy classes rather than two labels on the same path.
   * distinct is computed via routes_are_distinct()'s synchronized-time
   * side test (the space-time analogue of homotopy.py's mode_side_planes).
   */
  static void twoRouteSearch(
    const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
    float start_x, float start_y, float goal_x, float goal_y, float robot_r,
    float horizon, float dt_layer, float res, float window, float goal_tol,
    SpaceTimeRoute & wait_route, SpaceTimeRoute & detour_route, bool & distinct);

private:
  static bool routesAreDistinct(
    const SpaceTimeRoute & a, const SpaceTimeRoute & b,
    const std::vector<SpaceTimeObstacle> & obstacles, float dt_layer, float horizon,
    float relevance);
};

}  // namespace tgmppi

#endif  // NAV2_TGMPPI_CONTROLLER__TOOLS__SPACE_TIME_SEARCH_HPP_

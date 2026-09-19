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

#include "nav2_tgmppi_controller/tools/space_time_search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <tuple>

namespace tgmppi
{

namespace
{
using Cell = std::pair<int, int>;
using State = std::tuple<int, int, int>;   // (i, j, k)

// Grid bounds covering BOTH endpoints plus `window` of lateral room around
// each, not just `window` around the start alone -- a grid sized only
// around the start silently clips a distant goal to the window edge
// (toCell() clamps out-of-range indices), which looks like a normal
// search but quietly solves for the wrong goal. Direct port of
// spacetime.py's _grid_origin() docstring/logic.
// NOTE on float vs. the Python reference's double precision: with
// round-number inputs (this project's defaults include goal_tol=0.15,
// res=0.10), std::lround(goal_tol/res) can round to a DIFFERENT integer
// than Python/numpy's round(goal_tol/res) at exactly this kind of .5
// boundary (confirmed directly: float(0.15)/float(0.10) == 1.5 exactly,
// rounding up to 2; the same ratio computed in true double precision is
// 1.4999999999999998, rounding down to 1). Widening the float parameters
// to double inside this function does NOT fix it -- the imprecision is
// already baked into the caller's float literals before this function
// ever sees them (float(0.15) widened to double is 0.15000000596..., not
// 0.15), so the "fix" would just relocate an equally-arbitrary tie-break,
// not remove it. Documented rather than silently accepted: this is the
// same class of unavoidable float/double tie-break boundary already
// noted for GpuFlowFieldCritic's round vs. lround convention -- it only
// matters when a caller's parameters coincidentally land exactly on a
// half-cell boundary, not for the algorithm's correctness in general.
void gridOrigin(
  float start_x, float start_y, float goal_x, float goal_y, float window, float res,
  float & x0, float & y0, int & nx, int & ny)
{
  const float pad = 3.0f * res;
  x0 = std::min(start_x, goal_x) - window - pad;
  const float x1 = std::max(start_x, goal_x) + window + pad;
  y0 = std::min(start_y, goal_y) - window - pad;
  const float y1 = std::max(start_y, goal_y) + window + pad;
  nx = static_cast<int>(std::ceil((x1 - x0) / res)) + 1;
  ny = static_cast<int>(std::ceil((y1 - y0) / res)) + 1;
}

Cell toCell(float x, float y, float x0, float y0, float res, int nx, int ny)
{
  int i = static_cast<int>(std::lround((x - x0) / res));
  int j = static_cast<int>(std::lround((y - y0) / res));
  i = std::clamp(i, 0, nx - 1);
  j = std::clamp(j, 0, ny - 1);
  return {i, j};
}

// Winding-number homotopy distinctness (2026-09-19), replacing the earlier
// synchronized-time side test. That test required both routes to be within
// `relevance` of the SAME obstacle at the SAME time layer to say anything --
// but a pass-before route and a pass-behind route are close to the obstacle
// at DIFFERENT layers by construction, so the old test was vacuously
// unevaluated almost everywhere it mattered (live measurement: ~96% of
// ~11,000 detections flagged non-distinct; a 9-scenario synthetic harness of
// genuinely-distinct head-on crossings found it true in only 2/9). See
// docs/amoeba_mathematics_guide.tex, "Empirical distinctness rate and root
// cause", for the full diagnosis.
//
// This implements T-MPC's own winding-number homotopy test instead (de
// Groot et al., RA-L, Appendix B / Table VII: winding number and H-signature
// perform statistically identically there, winding number is ~3x cheaper to
// evaluate and needs no numerical integration). Each route's relative angle
// to the obstacle's predicted position is accumulated independently over the
// WHOLE route -- no synchronized-time requirement -- giving a normalized
// winding number lambda = (1/2*pi) * sum of wrapped angle differences. A
// route is considered to have genuinely passed the obstacle when
// |lambda| >= PASS_THRESHOLD (T-MPC's default, 1/(4*pi), a quarter turn);
// two routes are distinct when some obstacle is passed by both with
// opposite winding sign, i.e. on opposite sides.
constexpr float PASS_THRESHOLD = 1.0f / (4.0f * static_cast<float>(M_PI));

float windingNumber(
  const SpaceTimeRoute & route, const SpaceTimeObstacle & obs, float dt_layer, float horizon)
{
  const size_t n = route.path.size();
  if (n < 2) {return 0.0f;}
  const auto relativeAngle = [&](size_t k) {
      const auto & p = route.path[k];
      const float t = static_cast<float>(k) * dt_layer;
      const auto [ox, oy] = SpaceTimeSearch::predict(obs, t, horizon);
      return std::atan2(p.second - oy, p.first - ox);
    };
  float lambda = 0.0f;
  float prev_theta = relativeAngle(0);
  for (size_t k = 1; k < n; ++k) {
    const float theta = relativeAngle(k);
    float dtheta = theta - prev_theta;
    // wrap to (-pi, pi] so a step spanning the atan2 branch cut doesn't
    // register as a near-full spurious turn
    while (dtheta > static_cast<float>(M_PI)) {dtheta -= 2.0f * static_cast<float>(M_PI);}
    while (dtheta <= -static_cast<float>(M_PI)) {dtheta += 2.0f * static_cast<float>(M_PI);}
    lambda += dtheta;
    prev_theta = theta;
  }
  return lambda / (2.0f * static_cast<float>(M_PI));
}

}  // namespace

std::pair<float, float> SpaceTimeSearch::predict(
  const SpaceTimeObstacle & obs, float t, float horizon)
{
  const float tau = std::clamp(t, 0.0f, horizon);
  return {obs.x + tau * obs.vx, obs.y + tau * obs.vy};
}

SpaceTimeRoute SpaceTimeSearch::search(
  const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
  float start_x, float start_y, float goal_x, float goal_y, float robot_r,
  float horizon, float dt_layer, float res, float window, float goal_tol, float wait_cost,
  const SpaceTimeRoute * avoid_same_side_as, float side_relevance)
{
  SpaceTimeRoute out;

  float x0, y0;
  int nx, ny;
  gridOrigin(start_x, start_y, goal_x, goal_y, window, res, x0, y0, nx, ny);
  const int nk = static_cast<int>(std::lround(horizon / dt_layer)) + 1;
  const int max_arrival_k = nk - 1;
  const int goal_radius_cells = std::max(1, static_cast<int>(std::lround(goal_tol / res)));

  const auto [start_i, start_j] = toCell(start_x, start_y, x0, y0, res, nx, ny);
  const auto [goal_i, goal_j] = toCell(goal_x, goal_y, x0, y0, res, nx, ny);

  const auto isGoal = [&](int i, int j) {
      return std::abs(i - goal_i) <= goal_radius_cells && std::abs(j - goal_j) <= goal_radius_cells;
    };

  const auto cellXY = [&](int i, int j) {
      return std::make_pair(x0 + static_cast<float>(i) * res, y0 + static_cast<float>(j) * res);
    };

  const auto dynamicFree = [&](int i, int j, int k) {
      const auto [px, py] = cellXY(i, j);
      const float t = static_cast<float>(k) * dt_layer;
      for (const auto & obs : obstacles) {
        const auto [ox, oy] = predict(obs, t, horizon);
        const float dx = px - ox, dy = py - oy;
        if (std::sqrt(dx * dx + dy * dy) <= (obs.radius + robot_r)) {
          return false;
        }
      }
      return true;
    };

  // Homotopy constraint (2026-09-16): with a reference route given, a cell is
  // usable only if it lies on the OPPOSITE side of every nearby obstacle from
  // where the reference is at the SAME time layer. Differing wait costs alone
  // produced the same passing class in every cycle (log 36452: non-distinct
  // 91/91); this is what makes the second route a different homotopy class.
  // k == 0 is exempt: both routes start at the robot, so the test would
  // trivially report "same side" and block the search at its own start cell.
  const auto sideOk = [&](int i, int j, int k) {
      if (avoid_same_side_as == nullptr || avoid_same_side_as->path.empty() || k == 0) {
        return true;
      }
      const auto [px, py] = cellXY(i, j);
      const float t = static_cast<float>(k) * dt_layer;
      const auto & ref_path = avoid_same_side_as->path;
      const auto ref = ref_path[std::min(static_cast<std::size_t>(k), ref_path.size() - 1)];
      for (const auto & obs : obstacles) {
        const auto [ox, oy] = predict(obs, t, horizon);
        const float rx = ref.first - ox, ry = ref.second - oy;
        const float dx = px - ox, dy = py - oy;
        if (std::sqrt(rx * rx + ry * ry) > side_relevance ||
          std::sqrt(dx * dx + dy * dy) > side_relevance)
        {
          continue;   // constrain only where both routes are close to the obstacle
        }
        if (rx * dx + ry * dy > 0.0f) {return false;}   // same side as the reference
      }
      return true;
    };

  const auto free = [&](int i, int j, int k) {
      const auto [px, py] = cellXY(i, j);
      return static_free(px, py) && dynamicFree(i, j, k) && sideOk(i, j, k);
    };

  const State start_state{start_i, start_j, 0};
  if (!free(start_i, start_j, 0)) {
    return out;   // infeasible: robot's own current cell is already blocked
  }

  // 8-connected + (0,0) "wait" -- wait uses wait_cost, real moves use
  // res*hypot(di,dj) (diagonal steps cost more, matching real distance).
  struct Nbr {int di; int dj; float w;};
  std::vector<Nbr> nbrs;
  for (int di = -1; di <= 1; ++di) {
    for (int dj = -1; dj <= 1; ++dj) {
      nbrs.push_back({di, dj, res * std::sqrt(static_cast<float>(di * di + dj * dj))});
    }
  }

  std::map<State, float> g;
  std::map<State, State> came;
  std::set<State> visited;
  using PQItem = std::pair<float, State>;
  std::priority_queue<PQItem, std::vector<PQItem>, std::greater<>> pq;
  g[start_state] = 0.0f;
  pq.push({0.0f, start_state});

  bool found = false;
  State goal_state{};

  while (!pq.empty()) {
    auto [cost, state] = pq.top();
    pq.pop();
    if (visited.count(state)) {continue;}
    visited.insert(state);
    const auto [i, j, k] = state;
    if (isGoal(i, j) && k >= 0) {   // min_arrival_k always 0 -- see class docstring
      goal_state = state;
      found = true;
      break;
    }
    if (k >= max_arrival_k) {continue;}
    for (const auto & nbr : nbrs) {
      const int ni = i + nbr.di, nj = j + nbr.dj;
      if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) {continue;}
      // No corner-cutting: a diagonal move also needs both orthogonal
      // intermediate cells free at the arrival layer, matching astar.py's
      // rule (cited directly in spacetime.py's own comment).
      if (nbr.di != 0 && nbr.dj != 0 &&
        !(free(i + nbr.di, j, k + 1) && free(i, j + nbr.dj, k + 1)))
      {
        continue;
      }
      if (!free(ni, nj, k + 1)) {continue;}
      const float step_cost = (nbr.di == 0 && nbr.dj == 0) ? wait_cost : nbr.w;
      const float ncost = cost + step_cost;
      const State nstate{ni, nj, k + 1};
      const auto it = g.find(nstate);
      if (it == g.end() || ncost < it->second) {
        g[nstate] = ncost;
        came[nstate] = state;
        pq.push({ncost, nstate});
      }
    }
  }

  if (!found) {
    return out;
  }

  std::vector<State> cells{goal_state};
  State state = goal_state;
  while (!(state == start_state)) {
    state = came.at(state);
    cells.push_back(state);
  }
  std::reverse(cells.begin(), cells.end());

  out.path.reserve(cells.size());
  for (const auto & c : cells) {
    out.path.push_back(cellXY(std::get<0>(c), std::get<1>(c)));
  }
  out.cost = g.at(goal_state);
  out.feasible = true;
  return out;
}

bool SpaceTimeSearch::routesAreDistinct(
  const SpaceTimeRoute & a, const SpaceTimeRoute & b,
  const std::vector<SpaceTimeObstacle> & obstacles, float dt_layer, float horizon)
{
  for (const auto & obs : obstacles) {
    const float lambda_a = windingNumber(a, obs, dt_layer, horizon);
    const float lambda_b = windingNumber(b, obs, dt_layer, horizon);
    if (std::fabs(lambda_a) >= PASS_THRESHOLD && std::fabs(lambda_b) >= PASS_THRESHOLD &&
      std::signbit(lambda_a) != std::signbit(lambda_b))
    {
      return true;   // both routes genuinely pass this obstacle, on opposite sides
    }
  }
  return false;
}

void SpaceTimeSearch::twoRouteSearch(
  const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
  float start_x, float start_y, float goal_x, float goal_y, float robot_r,
  float horizon, float dt_layer, float res, float window, float goal_tol,
  SpaceTimeRoute & wait_route, SpaceTimeRoute & detour_route, bool & distinct)
{
  const float wait_cost_cheap = 0.25f * res;
  const float wait_cost_expensive = 5.0f * res;

  wait_route = search(
    static_free, obstacles, start_x, start_y, goal_x, goal_y, robot_r,
    horizon, dt_layer, res, window, goal_tol, wait_cost_cheap);
  // Second route must pass the obstacle on the other side from the first --
  // that constraint, not the wait cost, is what makes it a distinct class.
  detour_route = search(
    static_free, obstacles, start_x, start_y, goal_x, goal_y, robot_r,
    horizon, dt_layer, res, window, goal_tol, wait_cost_expensive,
    wait_route.feasible ? &wait_route : nullptr, 0.8f);

  distinct = wait_route.feasible && detour_route.feasible &&
    routesAreDistinct(wait_route, detour_route, obstacles, dt_layer, horizon);
}

}  // namespace tgmppi

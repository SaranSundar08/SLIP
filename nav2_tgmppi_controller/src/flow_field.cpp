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

#include "nav2_tgmppi_controller/tools/flow_field.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "nav2_costmap_2d/cost_values.hpp"

namespace tgmppi
{

namespace
{
// Large finite sentinel for unreached cells -- NOT infinity, because this
// package compiles with -ffast-math (isfinite is folded away, inf-inf=NaN).
constexpr float kUnreached = 1.0e9f;
constexpr float kDryPenalty = 2.0f;  // dry cells cost dmax + this (meters)
constexpr int kGoalEdgeMargin = 3;   // cells; closer than this to the window
                                     // edge counts as "goal beyond window"
// 8-neighbor offsets; dir_ stores an index into this table.
constexpr int kOff[8][2] =
{{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
}  // namespace

void FlowField::build(
  const nav2_costmap_2d::Costmap2D & costmap,
  const models::Path & path, bool path_seed, float viscosity,
  float robot_x, float robot_y, float body_radius,
  std::size_t max_pseudopods)
{
  const auto t0 = std::chrono::steady_clock::now();
  ready_ = false;
  pseudopods_.clear();
  pseudopod_promises_.clear();
  body_cell_count_ = 0;
  membrane_cell_count_ = 0;

  const std::size_t P = path.x.shape(0);
  if (P == 0) {
    return;  // nothing to flood toward
  }

  nx_ = static_cast<int>(costmap.getSizeInCellsX());
  ny_ = static_cast<int>(costmap.getSizeInCellsY());
  res_ = static_cast<float>(costmap.getResolution());
  ox_ = static_cast<float>(costmap.getOriginX());
  oy_ = static_cast<float>(costmap.getOriginY());
  const std::size_t n = static_cast<std::size_t>(nx_) * ny_;

  // Wet/dry classification straight off the inflated costmap: the inflation
  // layer already grew obstacles by the footprint's inscribed radius, so
  // this is a centre-cell clearance proxy, not an oriented polygon test. Unknown = free,
  // matching the optimistic convention of the rest of this controller.
  free_.assign(n, 0);
  visc_.assign(n, 1.0f);
  const unsigned char * chart = costmap.getCharMap();
  for (std::size_t k = 0; k < n; ++k) {
    const unsigned char c = chart[k];
    if (c < nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE) {
      free_[k] = 1;
      // Water viscosity: the inflation cost IS proximity-to-wall, so flood
      // resistance rises with it -- wide channels flow freely, pinches drag.
      visc_[k] = 1.0f + viscosity * static_cast<float>(c) / 252.0f;
    } else if (c == nav2_costmap_2d::NO_INFORMATION) {
      free_[k] = 1;  // unknown = free, resistance 1
    }
  }

  // One connectivity rule for body reachability, membrane adjacency, the
  // distance flood and direction lookup. A diagonal needs both side cells
  // free; checking only its destination lets paths squeeze through corners.
  const auto can_step = [this](int i, int j, int ni, int nj) {
      if (ni < 0 || ni >= nx_ || nj < 0 || nj >= ny_) {return false;}
      return free_[idx(ni, nj)] &&
             (i == ni || j == nj || (free_[idx(ni, j)] && free_[idx(i, nj)]));
    };

  using QItem = std::pair<float, int>;

  // --- Robot-centred finite geodesic body ---------------------------------
  body_.assign(n, 0);
  membrane_.assign(n, 0);
  body_dist_.assign(n, kUnreached);
  body_parent_.assign(n, -1);
  int ri = std::clamp(static_cast<int>(std::lround((robot_x - ox_) / res_)), 0, nx_ - 1);
  int rj = std::clamp(static_cast<int>(std::lround((robot_y - oy_) / res_)), 0, ny_ - 1);
  int root = idx(ri, rj);
  if (!free_[root]) {
    float nearest = kUnreached;
    int best_root = -1;
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        if (!free_[idx(i, j)]) {continue;}
        const float d2 = static_cast<float>((i - ri) * (i - ri) + (j - rj) * (j - rj));
        if (d2 < nearest) {nearest = d2; best_root = idx(i, j);}
      }
    }
    if (best_root < 0) {return;}
    root = best_root;
  }

  const float radius = std::max(res_, body_radius);
  const float diag = res_ * std::sqrt(2.0f);
  std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> body_q;
  body_dist_[root] = 0.0f;
  body_q.emplace(0.0f, root);
  while (!body_q.empty()) {
    const auto [d, k] = body_q.top();
    body_q.pop();
    if (d > body_dist_[k] || d > radius) {continue;}
    body_[k] = 1;
    const int i = k % nx_;
    const int j = k / nx_;
    for (int o = 0; o < 8; ++o) {
      const int ni = i + kOff[o][0];
      const int nj = j + kOff[o][1];
      if (!can_step(i, j, ni, nj)) {continue;}
      const int nk = idx(ni, nj);
      if (!free_[nk]) {continue;}
      const float step =
        (kOff[o][0] != 0 && kOff[o][1] != 0) ? diag : res_;
      const float nd = d + step;
      if (nd <= radius && nd < body_dist_[nk]) {
        body_dist_[nk] = nd;
        body_parent_[nk] = k;
        body_q.emplace(nd, nk);
      }
    }
  }

  // The membrane is the reachable outer wavefront, not obstacle boundaries
  // inside the body. A body cell is on the membrane when a free neighbor lies
  // beyond the finite geodesic radius.
  for (int j = 0; j < ny_; ++j) {
    for (int i = 0; i < nx_; ++i) {
      const int k = idx(i, j);
      if (!body_[k]) {continue;}
      ++body_cell_count_;
      bool outer = body_dist_[k] >= radius - diag;
      for (int o = 0; o < 8 && !outer; ++o) {
        const int ni = i + kOff[o][0];
        const int nj = j + kOff[o][1];
        if (!can_step(i, j, ni, nj)) {continue;}
        const int nk = idx(ni, nj);
        outer = !body_[nk];
      }
      if (outer) {
        membrane_[k] = 1;
        ++membrane_cell_count_;
      }
    }
  }

  D_.assign(n, kUnreached);
  std::priority_queue<QItem, std::vector<QItem>, std::greater<QItem>> pq;

  // --- Seeding -------------------------------------------------------------
  const float gxw = path.x(P - 1);
  const float gyw = path.y(P - 1);
  const int gi = static_cast<int>(std::lround((gxw - ox_) / res_));
  const int gj = static_cast<int>(std::lround((gyw - oy_) / res_));
  const bool goal_inside =
    gi >= kGoalEdgeMargin && gi < nx_ - kGoalEdgeMargin &&
    gj >= kGoalEdgeMargin && gj < ny_ - kGoalEdgeMargin && body_[idx(gi, gj)];

  if (goal_inside) {
    // Seed the (nearest free cell to the) goal with level 0.
    int bi = -1, bj = -1;
    float best = kUnreached;
    if (free_[idx(gi, gj)]) {
      bi = gi;
      bj = gj;
    } else {
      for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
          if (!body_[idx(i, j)]) {continue;}
          const float d2 = static_cast<float>((i - gi) * (i - gi) + (j - gj) * (j - gj));
          if (d2 < best) {best = d2; bi = i; bj = j;}
        }
      }
    }
    if (bi < 0) {
      return;  // window fully dry
    }
    D_[idx(bi, bj)] = 0.0f;
    pq.emplace(0.0f, idx(bi, bj));
  } else {
    // Goal beyond the finite body: promise is evaluated on the membrane.
    // Remaining plan length at each (subsampled) plan point:
    const std::size_t stride = std::max<std::size_t>(1, P / 64);
    std::vector<float> pxs, pys, rem;
    {
      std::vector<float> cum(P, 0.0f);
      for (std::size_t k = P - 1; k > 0; --k) {
        cum[k - 1] = cum[k] + std::hypot(
          path.x(k) - path.x(k - 1), path.y(k) - path.y(k - 1));
      }
      for (std::size_t k = 0; k < P; k += stride) {
        pxs.push_back(path.x(k));
        pys.push_back(path.y(k));
        rem.push_back(cum[k]);
      }
    }
    auto promise = [&](float wx, float wy) {
        if (!path_seed) {
          return std::hypot(wx - gxw, wy - gyw);
        }
        float best = kUnreached;
        for (std::size_t k = 0; k < pxs.size(); ++k) {
          best = std::min(best, std::hypot(wx - pxs[k], wy - pys[k]) + rem[k]);
        }
        return best;
      };
    std::vector<std::pair<float, int>> ranked_membrane;
    auto seed = [&](int i, int j) {
        if (!membrane_[idx(i, j)]) {return;}
        const float d0 = promise(ox_ + i * res_, oy_ + j * res_);
        ranked_membrane.emplace_back(d0, idx(i, j));
        if (d0 < D_[idx(i, j)]) {
          D_[idx(i, j)] = d0;
          pq.emplace(d0, idx(i, j));
        }
      };
    for (int j = 0; j < ny_; ++j) {
      for (int i = 0; i < nx_; ++i) {
        seed(i, j);
      }
    }
    if (pq.empty()) {
      return;  // no reachable membrane
    }

    // Select separated low-promise membrane exits and trace their geodesic
    // parents back to the robot. Phase 3 converts each selected pseudopod into
    // a distinct time-varying ancillary control proposal.
    std::sort(ranked_membrane.begin(), ranked_membrane.end());
    std::vector<int> selected;
    for (const auto & candidate : ranked_membrane) {
      const int endpoint = candidate.second;
      const int ei = endpoint % nx_;
      const int ej = endpoint / nx_;
      bool distinct = true;
      for (const int prior : selected) {
        const int pi = prior % nx_;
        const int pj = prior / nx_;
        if (std::hypot(static_cast<float>(ei - pi), static_cast<float>(ej - pj)) * res_ < 0.65f) {
          distinct = false;
          break;
        }
      }
      if (!distinct) {continue;}
      selected.push_back(endpoint);
      std::vector<std::pair<float, float>> pod;
      int k = endpoint;
      for (std::size_t guard = 0; guard < n && k >= 0; ++guard) {
        const int i = k % nx_;
        const int j = k / nx_;
        pod.emplace_back(ox_ + i * res_, oy_ + j * res_);
        if (k == root) {break;}
        k = body_parent_[k];
      }
      std::reverse(pod.begin(), pod.end());
      if (pod.size() >= 2) {
        pseudopods_.push_back(std::move(pod));
        pseudopod_promises_.push_back(candidate.first);
      }
      if (pseudopods_.size() >= max_pseudopods) {break;}
    }
  }

  // --- Dijkstra flood ------------------------------------------------------
  while (!pq.empty()) {
    const auto [d, k] = pq.top();
    pq.pop();
    if (d > D_[k]) {continue;}
    const int i = k % nx_;
    const int j = k / nx_;
    for (int o = 0; o < 8; ++o) {
      const int ni = i + kOff[o][0];
      const int nj = j + kOff[o][1];
      if (!can_step(i, j, ni, nj)) {continue;}
      const int nk = idx(ni, nj);
      if (!body_[nk]) {continue;}
      const float w =
        ((kOff[o][0] != 0 && kOff[o][1] != 0) ? diag : res_) * visc_[nk];
      if (d + w < D_[nk]) {
        D_[nk] = d + w;
        pq.emplace(d + w, nk);
      }
    }
  }

  // --- Steepest-descent direction table ------------------------------------
  dir_.assign(n, -1);
  dmax_ = 0.0f;
  for (int j = 0; j < ny_; ++j) {
    for (int i = 0; i < nx_; ++i) {
      const int k = idx(i, j);
      if (D_[k] >= kWetLimit) {continue;}
      dmax_ = std::max(dmax_, D_[k]);
      float best = kUnreached;
      int bo = -1;
      for (int o = 0; o < 8; ++o) {
        const int ni = i + kOff[o][0];
        const int nj = j + kOff[o][1];
        if (!can_step(i, j, ni, nj)) {continue;}
        const float dn = D_[idx(ni, nj)];
        if (dn < best) {best = dn; bo = o;}
      }
      if (bo >= 0 && best < kWetLimit) {
        dir_[k] = static_cast<int8_t>(bo);
      }
    }
  }

  ready_ = true;
  build_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
}

bool FlowField::worldToGridClamped(float wx, float wy, int & i, int & j) const
{
  if (nx_ == 0 || ny_ == 0) {
    return false;
  }
  i = std::clamp(static_cast<int>(std::lround((wx - ox_) / res_)), 0, nx_ - 1);
  j = std::clamp(static_cast<int>(std::lround((wy - oy_) / res_)), 0, ny_ - 1);
  return true;
}

float FlowField::distAt(float wx, float wy) const
{
  int i, j;
  if (!ready_ || !worldToGridClamped(wx, wy, i, j)) {
    return 0.0f;
  }
  const float d = D_[idx(i, j)];
  return d < kWetLimit ? d : dmax_ + kDryPenalty;
}

bool FlowField::dirAt(float wx, float wy, float & dx, float & dy) const
{
  int i, j;
  if (!ready_ || !worldToGridClamped(wx, wy, i, j)) {
    return false;
  }
  return cellDir(i, j, dx, dy);
}

bool FlowField::gradAt(float wx, float wy, float & dx, float & dy) const
{
  if (!ready_) {
    return false;
  }
  const float h = 3.0f * res_;
  const float gx = distAt(wx + h, wy) - distAt(wx - h, wy);
  const float gy = distAt(wx, wy + h) - distAt(wx, wy - h);
  const float n = std::hypot(gx, gy);
  if (n < 1e-4f) {
    return false;  // locally flat (e.g. deep inside a dry region)
  }
  dx = -gx / n;
  dy = -gy / n;
  return true;
}

bool FlowField::cellDir(int i, int j, float & dx, float & dy) const
{
  const int8_t o = dir_[idx(i, j)];
  if (o < 0) {
    return false;
  }
  const float norm = std::hypot(
    static_cast<float>(kOff[o][0]), static_cast<float>(kOff[o][1]));
  dx = static_cast<float>(kOff[o][0]) / norm;
  dy = static_cast<float>(kOff[o][1]) / norm;
  return true;
}

}  // namespace tgmppi

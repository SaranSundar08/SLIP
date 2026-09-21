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

#include "nav2_tgmppi_controller/tools/space_time_body.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace tgmppi
{

namespace
{
constexpr float kPi = 3.14159265358979323846f;
// T-MPC's default pass threshold, a quarter turn (same as space_time_search.cpp).
constexpr float kPassThreshold = 1.0f / (4.0f * kPi);
// Neighbour table: offset index = (di + 1) * 3 + (dj + 1); index 4 is "wait".
constexpr int kWait = 4;

inline void offsetOf(int o, int & di, int & dj)
{
  di = o / 3 - 1;
  dj = o % 3 - 1;
}

// Normalized winding number of a route around one obstacle (accumulated wrapped
// relative-angle change / 2 pi), independent of any time synchronisation.
float windingNumber(
  const std::vector<std::pair<float, float>> & path, const SpaceTimeObstacle & obs,
  float dt_layer, float horizon)
{
  const std::size_t n = path.size();
  if (n < 2) {return 0.0f;}
  const auto relativeAngle = [&](std::size_t k) {
      const auto pred = SpaceTimeSearch::predict(obs, static_cast<float>(k) * dt_layer, horizon);
      return std::atan2(path[k].second - pred.second, path[k].first - pred.first);
    };
  float lambda = 0.0f;
  float prev = relativeAngle(0);
  for (std::size_t k = 1; k < n; ++k) {
    const float theta = relativeAngle(k);
    float d = theta - prev;
    while (d > kPi) {d -= 2.0f * kPi;}
    while (d <= -kPi) {d += 2.0f * kPi;}
    lambda += d;
    prev = theta;
  }
  return lambda / (2.0f * kPi);
}
}  // namespace

float SpaceTimeBody::cost(int i, int j, int k) const
{
  if (!ready_ || i < 0 || i >= n_ || j < 0 || j >= n_ || k < 0 || k > K_) {
    return kUnreached;
  }
  return G_[at(k, i, j)];
}

void SpaceTimeBody::build(
  const StaticFreeFn & static_free, const std::vector<SpaceTimeObstacle> & obstacles,
  const TerminalFn & terminal, float start_x, float start_y, const SpaceTimeBodyParams & p)
{
  const auto t0 = std::chrono::steady_clock::now();
  ready_ = false;
  pods_.clear();
  promises_.clear();
  signatures_.clear();
  body_states_ = 0;
  membrane_cells_ = 0;
  classes_found_ = 0;

  K_ = std::max(1, static_cast<int>(std::lround(p.horizon / p.dt_layer)));
  n_ = 2 * K_ + 1;                 // the light cone: one cell of motion per layer
  res_ = p.res;
  x0_ = start_x - static_cast<float>(K_) * res_;
  y0_ = start_y - static_cast<float>(K_) * res_;
  const int c0 = K_;               // the robot's own cell
  const std::size_t plane = static_cast<std::size_t>(n_) * n_;

  // Obstacles that can block any cell of the grid at any layer. Farther ones are
  // dropped up front (they cannot change the flood or the signatures).
  const float grid_reach = static_cast<float>(K_) * res_ * 1.4143f;
  std::vector<SpaceTimeObstacle> rel;
  for (const auto & o : obstacles) {
    bool relevant = false;
    for (int k = 0; k <= K_ && !relevant; ++k) {
      const auto pr = SpaceTimeSearch::predict(o, static_cast<float>(k) * p.dt_layer, p.horizon);
      relevant = std::hypot(pr.first - start_x, pr.second - start_y) <=
        grid_reach + o.radius + p.robot_r + res_;
    }
    if (relevant) {rel.push_back(o);}
  }

  // Static free mask: layer-independent, evaluated once.
  std::vector<uint8_t> sfree(plane, 0);
  for (int j = 0; j < n_; ++j) {
    for (int i = 0; i < n_; ++i) {
      sfree[static_cast<std::size_t>(j) * n_ + i] = static_free(cellX(i), cellY(j)) ? 1 : 0;
    }
  }

  G_.assign(plane * (K_ + 1), kUnreached);
  parent_.assign(plane * (K_ + 1), -1);
  G_[at(0, c0, c0)] = 0.0f;   // the robot's own state is always the root, even if inflated

  std::vector<uint8_t> dfree(plane);
  for (int k = 0; k < K_; ++k) {
    // Free mask of layer k+1: static AND clear of every predicted obstacle at t_{k+1}.
    const float t = static_cast<float>(k + 1) * p.dt_layer;
    std::vector<std::pair<std::pair<float, float>, float>> pos;   // obstacle centre, clearance^2
    pos.reserve(rel.size());
    for (const auto & o : rel) {
      const float reach = o.radius + p.robot_r;
      pos.push_back({SpaceTimeSearch::predict(o, t, p.horizon), reach * reach});
    }
    for (int j = 0; j < n_; ++j) {
      for (int i = 0; i < n_; ++i) {
        const std::size_t c = static_cast<std::size_t>(j) * n_ + i;
        bool ok = sfree[c] != 0;
        if (ok) {
          const float px = cellX(i), py = cellY(j);
          for (const auto & q : pos) {
            const float dx = px - q.first.first, dy = py - q.first.second;
            if (dx * dx + dy * dy <= q.second) {ok = false; break;}
          }
        }
        dfree[c] = ok ? 1 : 0;
      }
    }

    // Layered DP: every edge advances exactly one layer.
    const int lo = std::max(0, c0 - k), hi = std::min(n_ - 1, c0 + k);   // reachable box at k
    for (int j = lo; j <= hi; ++j) {
      for (int i = lo; i <= hi; ++i) {
        const float g = G_[at(k, i, j)];
        if (g >= kUnreached) {continue;}
        for (int o = 0; o < 9; ++o) {
          int di, dj;
          offsetOf(o, di, dj);
          const int ni = i + di, nj = j + dj;
          if (ni < 0 || ni >= n_ || nj < 0 || nj >= n_) {continue;}
          if (!dfree[static_cast<std::size_t>(nj) * n_ + ni]) {continue;}
          const float step = (o == kWait) ? p.wait_epsilon :
            res_ * std::sqrt(static_cast<float>(di * di + dj * dj));
          const float ng = g + step;
          const std::size_t idx = at(k + 1, ni, nj);
          if (ng < G_[idx]) {
            G_[idx] = ng;
            parent_[idx] = static_cast<int8_t>(o);
          }
        }
      }
    }
  }

  for (int k = 0; k <= K_; ++k) {
    for (std::size_t c = 0; c < plane; ++c) {
      if (G_[static_cast<std::size_t>(k) * plane + c] < kUnreached) {++body_states_;}
    }
  }

  // Membrane = reached states on the last layer; rank by promise = G + w * D_static.
  struct Exit {float promise; int i; int j;};
  std::vector<Exit> exits;
  for (int j = 0; j < n_; ++j) {
    for (int i = 0; i < n_; ++i) {
      const float g = G_[at(K_, i, j)];
      if (g >= kUnreached) {continue;}
      exits.push_back({g + p.terminal_weight * terminal(cellX(i), cellY(j)), i, j});
    }
  }
  membrane_cells_ = exits.size();
  if (exits.empty()) {
    build_ms_ = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
    return;
  }
  std::sort(exits.begin(), exits.end(), [](const Exit & a, const Exit & b) {
      return a.promise < b.promise;
    });
  const float best = exits.front().promise;

  const auto reconstruct = [&](int ei, int ej) {
      std::vector<std::pair<float, float>> path(static_cast<std::size_t>(K_) + 1);
      int i = ei, j = ej;
      for (int k = K_; k >= 0; --k) {
        path[static_cast<std::size_t>(k)] = {cellX(i), cellY(j)};
        if (k == 0) {break;}
        int di, dj;
        offsetOf(parent_[at(k, i, j)], di, dj);
        i -= di;
        j -= dj;
      }
      return path;
    };
  const auto signatureOf = [&](const std::vector<std::pair<float, float>> & path) {
      std::vector<int8_t> sig;
      sig.reserve(rel.size());
      for (const auto & o : rel) {
        const float lam = windingNumber(path, o, p.dt_layer, p.horizon);
        sig.push_back(std::fabs(lam) < kPassThreshold ? int8_t{0} : (lam > 0 ? int8_t{1} : int8_t{-1}));
      }
      return sig;
    };
  const auto sepFromAccepted = [&](const std::vector<std::pair<int, int>> & acc, int i, int j) {
      float m = kUnreached;
      for (const auto & a : acc) {
        m = std::min(m, std::hypot(static_cast<float>(i - a.first), static_cast<float>(j - a.second)) * res_);
      }
      return m;
    };

  // Candidate pool: everything within max_regret of the best exit (capped).
  struct Cand {Exit e; std::vector<std::pair<float, float>> path; std::vector<int8_t> sig;};
  std::vector<Cand> pool;
  for (const auto & e : exits) {
    if (e.promise > best + p.max_regret || pool.size() >= 256u) {break;}
    auto path = reconstruct(e.i, e.j);
    auto sig = signatureOf(path);
    pool.push_back({e, std::move(path), std::move(sig)});
  }
  {
    std::vector<std::vector<int8_t>> seen;
    for (const auto & c : pool) {
      if (std::find(seen.begin(), seen.end(), c.sig) == seen.end()) {seen.push_back(c.sig);}
    }
    classes_found_ = seen.size();
  }

  std::vector<std::pair<int, int>> accepted;
  std::vector<uint8_t> taken(pool.size(), 0);
  const auto accept = [&](std::size_t idx) {
      taken[idx] = 1;
      accepted.emplace_back(pool[idx].e.i, pool[idx].e.j);
      SpaceTimeRoute r;
      r.path = pool[idx].path;
      r.cost = pool[idx].e.promise;
      r.feasible = true;
      pods_.push_back(std::move(r));
      promises_.push_back(pool[idx].e.promise);
      signatures_.push_back(pool[idx].sig);
    };

  // Pass 1: best exit overall, then the best exit of every NEW homotopy class.
  std::vector<std::vector<int8_t>> used_sigs;
  for (std::size_t c = 0; c < pool.size() && static_cast<int>(pods_.size()) < p.max_pods; ++c) {
    if (std::find(used_sigs.begin(), used_sigs.end(), pool[c].sig) != used_sigs.end()) {continue;}
    if (!accepted.empty() &&
      sepFromAccepted(accepted, pool[c].e.i, pool[c].e.j) < p.class_min_separation)
    {
      continue;
    }
    used_sigs.push_back(pool[c].sig);
    accept(c);
  }
  // Pass 2: fill remaining slots by exit separation (the static blob's rule).
  for (std::size_t c = 0; c < pool.size() && static_cast<int>(pods_.size()) < p.max_pods; ++c) {
    if (taken[c]) {continue;}
    if (sepFromAccepted(accepted, pool[c].e.i, pool[c].e.j) < p.exit_separation) {continue;}
    accept(c);
  }

  ready_ = true;
  build_ms_ = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
}

}  // namespace tgmppi

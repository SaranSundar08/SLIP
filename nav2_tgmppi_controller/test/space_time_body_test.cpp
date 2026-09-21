// Standalone test for tgmppi::SpaceTimeBody (no ROS, no gtest). Build and run from the
// package directory with the SAME flags as the package (-ffast-math matters: it strips
// isfinite):
//
//   g++ -std=c++17 -O2 -ffast-math -Wall -Wextra -Iinclude test/space_time_body_test.cpp src/tools/space_time_body.cpp src/tools/space_time_search.cpp -o /tmp/stb_test && /tmp/stb_test

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "nav2_tgmppi_controller/tools/space_time_body.hpp"

using tgmppi::SpaceTimeBody;
using tgmppi::SpaceTimeBodyParams;
using tgmppi::SpaceTimeObstacle;
using tgmppi::SpaceTimeSearch;

static int g_fail = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { \
      ++g_fail; std::printf("  FAIL %s:%d  %s  ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const SpaceTimeBody::StaticFreeFn kAllFree = [](float, float) {return true;};

static SpaceTimeBody::TerminalFn goalDistance(float gx, float gy)
{
  return [gx, gy](float x, float y) {return std::hypot(x - gx, y - gy);};
}

// Independent validity check of every returned route (does not reuse SpaceTimeBody internals).
static void checkRoutes(
  const char * tag, const SpaceTimeBody & b, const SpaceTimeBody::StaticFreeFn & sf,
  const std::vector<SpaceTimeObstacle> & obs, const SpaceTimeBodyParams & p, float sx, float sy)
{
  const int K = static_cast<int>(std::lround(p.horizon / p.dt_layer));
  for (std::size_t r = 0; r < b.pods().size(); ++r) {
    const auto & path = b.pods()[r].path;
    CHECK(static_cast<int>(path.size()) == K + 1, "%s pod %zu has %zu points", tag, r, path.size());
    if (static_cast<int>(path.size()) != K + 1) {continue;}
    CHECK(std::hypot(path[0].first - sx, path[0].second - sy) < 1e-4f, "%s pod %zu start", tag, r);
    for (int k = 1; k <= K; ++k) {
      const auto & a = path[k - 1];
      const auto & c = path[k];
      CHECK(std::hypot(c.first - a.first, c.second - a.second) <= p.res * 1.4143f + 1e-3f,
        "%s pod %zu step %d too long", tag, r, k);
      CHECK(sf(c.first, c.second), "%s pod %zu layer %d in static obstacle", tag, r, k);
      for (const auto & o : obs) {
        const auto q = SpaceTimeSearch::predict(o, k * p.dt_layer, p.horizon);
        CHECK(std::hypot(c.first - q.first, c.second - q.second) > o.radius + p.robot_r - 1e-3f,
          "%s pod %zu layer %d hits obstacle", tag, r, k);
      }
    }
  }
}

int main()
{
  SpaceTimeBodyParams p;   // horizon 3 s, dt 0.25, res 0.10 -> K = 12, reach 1.2 m

  std::printf("T1 free space\n");
  {
    SpaceTimeBody b;
    b.build(kAllFree, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    CHECK(b.ready(), "not ready");
    CHECK(b.layers() == 12, "layers %d", b.layers());
    CHECK(!b.pods().empty(), "no pods");
    if (!b.pods().empty()) {
      const auto & e = b.pods()[0].path.back();
      CHECK(std::fabs(e.first - 1.2f) < 0.06f && std::fabs(e.second) < 0.06f,
        "best exit (%.2f, %.2f), want ~(1.2, 0)", e.first, e.second);
      CHECK(std::fabs(b.promises()[0] - 4.0f) < 0.1f, "best promise %.3f want ~4.0", b.promises()[0]);
    }
    CHECK(b.pods().size() == 3u, "pods %zu want 3 (geometry fill)", b.pods().size());
    CHECK(b.classesFound() == 1u, "classes %zu want 1", b.classesFound());
    checkRoutes("T1", b, kAllFree, {}, p, 0.0f, 0.0f);
    std::printf("  body states %zu, membrane cells %zu, %.3f ms\n",
      b.bodyStates(), b.membraneCells(), b.buildMs());
  }

  std::printf("T2 static obstacle on the route -> above / below are different classes\n");
  {
    const std::vector<SpaceTimeObstacle> obs = {{0.9f, 0.0f, 0.0f, 0.0f, 0.25f}};
    SpaceTimeBody b0, b;
    b0.build(kAllFree, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    b.build(kAllFree, obs, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    CHECK(b.ready(), "not ready");
    CHECK(b.classesFound() >= 2u, "classes %zu want >= 2", b.classesFound());
    CHECK(b.pods().size() >= 2u, "pods %zu", b.pods().size());
    if (b.pods().size() >= 2u && !b.signatures()[0].empty()) {
      bool opposite = false;
      for (std::size_t i = 0; i < b.signatures().size(); ++i) {
        for (std::size_t j = i + 1; j < b.signatures().size(); ++j) {
          opposite |= (b.signatures()[i][0] * b.signatures()[j][0] == -1);
        }
      }
      CHECK(opposite, "no pair of pods passes the obstacle on opposite sides");
    }
    CHECK(b.promises()[0] > b0.promises()[0] + 0.01f, "obstacle should cost something: %.3f vs %.3f",
      b.promises()[0], b0.promises()[0]);
    checkRoutes("T2", b, kAllFree, obs, p, 0.0f, 0.0f);
  }

  std::printf("T3 moving obstacle crossing from above at 1.0 m/s -> valid routes\n");
  {
    SpaceTimeBodyParams q = p;
    const std::vector<SpaceTimeObstacle> obs = {{0.8f, 1.6f, 0.0f, -1.0f, 0.25f}};
    SpaceTimeBody b0, b;
    b0.build(kAllFree, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, q);
    b.build(kAllFree, obs, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, q);
    CHECK(b.ready() && !b.pods().empty(), "no pods");
    CHECK(b.promises()[0] >= b0.promises()[0] - 1e-3f, "obstacle cannot improve the best promise");
    checkRoutes("T3", b, kAllFree, obs, q, 0.0f, 0.0f);
  }

  std::printf("T4 start inside an inflated cell (robot already in the inflation zone)\n");
  {
    const SpaceTimeBody::StaticFreeFn sf = [](float x, float y) {
        return std::hypot(x, y) > 0.05f;   // only the root cell is 'blocked'
      };
    SpaceTimeBody b;
    b.build(sf, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    CHECK(b.ready() && !b.pods().empty(), "should still find routes out of a blocked root");
  }

  std::printf("T5 completely enclosed (nothing free) -> not ready, no crash\n");
  {
    const SpaceTimeBody::StaticFreeFn none = [](float, float) {return false;};
    SpaceTimeBody b;
    b.build(none, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    CHECK(!b.ready(), "should not be ready");
    CHECK(b.pods().empty(), "pods should be empty");
  }

  std::printf("T6 static wall: routes never enter it\n");
  {
    const SpaceTimeBody::StaticFreeFn wall = [](float x, float y) {
        return !(x > 0.5f && x < 0.7f && y < 0.4f);   // wall with a gap above y = 0.4
      };
    SpaceTimeBody b;
    b.build(wall, {}, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
    CHECK(b.ready() && !b.pods().empty(), "no pods");
    checkRoutes("T6", b, wall, {}, p, 0.0f, 0.0f);
  }

  std::printf("T7 timing: 20 moving obstacles scattered in a 6 x 6 m box around the robot\n");
  {
    std::vector<SpaceTimeObstacle> obs;
    unsigned s = 12345u;
    const auto rnd = [&s]() {s = s * 1664525u + 1013904223u; return static_cast<float>(s >> 8) / 16777216.0f;};
    while (obs.size() < 20u) {
      const float x = -3.0f + 6.0f * rnd(), y = -3.0f + 6.0f * rnd();
      if (std::hypot(x, y) < 1.0f) {continue;}   // keep the robot's own neighbourhood open
      obs.push_back({x, y, -0.5f + rnd(), -0.5f + rnd(), 0.25f});
    }
    double worst = 0.0;
    for (int rep = 0; rep < 20; ++rep) {
      SpaceTimeBody b;
      b.build(kAllFree, obs, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, p);
      worst = std::max(worst, b.buildMs());
      if (rep == 0) {
        CHECK(b.ready() && !b.pods().empty(), "no pods in a scattered scene");
        checkRoutes("T7", b, kAllFree, obs, p, 0.0f, 0.0f);
        std::printf("  pods %zu, classes %zu, states %zu\n", b.pods().size(), b.classesFound(), b.bodyStates());
      }
    }
    std::printf("  worst build over 20 reps: %.3f ms\n", worst);
    CHECK(worst < 20.0, "build too slow: %.3f ms", worst);
  }

  std::printf("T8 obstacle already inside the requested clearance at t = 0 -> still routes out\n");
  {
    SpaceTimeBodyParams q = p;
    q.robot_r = 0.5f;                                       // wants 0.75 m from a 0.25 m obstacle
    const std::vector<SpaceTimeObstacle> obs = {{0.5f, 0.0f, 0.0f, 0.0f, 0.25f}};   // only 0.5 m away
    SpaceTimeBody b;
    b.build(kAllFree, obs, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, q);
    CHECK(b.ready() && !b.pods().empty(), "robot inside the requested clearance has no way out");
    // Never allowed closer than it started (0.5 m): relaxed clearance is d0 - 0.02 = 0.48.
    for (const auto & r : b.pods()) {
      for (int k = 1; k < static_cast<int>(r.path.size()); ++k) {
        CHECK(std::hypot(r.path[k].first - 0.5f, r.path[k].second) > 0.48f - 1e-3f, "pod entered relaxed clearance");
      }
    }
  }

  std::printf("T9 larger clearance radius keeps more distance than the inscribed one\n");
  {
    const std::vector<SpaceTimeObstacle> obs = {{0.9f, 0.0f, 0.0f, 0.0f, 0.25f}};
    SpaceTimeBodyParams q = p;
    q.robot_r = 0.5f;
    SpaceTimeBody b;
    b.build(kAllFree, obs, goalDistance(4.0f, 0.0f), 0.0f, 0.0f, q);
    CHECK(b.ready() && !b.pods().empty(), "no pods");
    checkRoutes("T9", b, kAllFree, obs, q, 0.0f, 0.0f);   // checks clearance 0.25 + 0.5 = 0.75
  }

  std::printf(g_fail == 0 ? "\nALL PASSED\n" : "\n%d CHECK(S) FAILED\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}

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

#include <cmath>
#include <gtest/gtest.h>

#include "nav2_costmap_2d/cost_values.hpp"
#include "nav2_tgmppi_controller/tools/flow_field.hpp"

namespace
{
tgmppi::models::Path goal(float x, float y)
{
  tgmppi::models::Path path;
  path.reset(1);
  path.x(0) = x;
  path.y(0) = y;
  return path;
}

TEST(FlowField, BodyCannotSqueezePastEitherSideOfDiagonal)
{
  for (int dx : {-1, 1}) {
    for (int dy : {-1, 1}) {
      for (unsigned int blocked = 0; blocked < 4; ++blocked) {
        SCOPED_TRACE(::testing::Message() << dx << "," << dy << " mask=" << blocked);
        nav2_costmap_2d::Costmap2D map(15, 15, 1.0, 0.0, 0.0, 0);
        if (blocked & 1u) {map.setCost(6 + dx, 6, 254);}
        if (blocked & 2u) {map.setCost(6, 6 + dy, 253);}
        tgmppi::FlowField field;
        field.build(map, goal(14, 14), false, 0.0f, 6, 6, 1.5f);
        ASSERT_TRUE(field.ready());
        EXPECT_EQ(field.cellBody(6 + dx, 6 + dy), blocked == 0u);
      }
    }
  }
}

TEST(FlowField, DistanceFloodAndDirectionRespectBlockedCorner)
{
  for (int dx : {-1, 1}) {
    for (int dy : {-1, 1}) {
      nav2_costmap_2d::Costmap2D map(15, 15, 1.0, 0.0, 0.0, 0);
      map.setCost(6 + dx, 6, nav2_costmap_2d::LETHAL_OBSTACLE);
      tgmppi::FlowField field;
      field.build(map, goal(6 + dx, 6 + dy), false, 0.0f, 6, 6, 10.0f);
      ASSERT_TRUE(field.ready());
      // The legal shortest route takes two cardinal steps, not sqrt(2).
      EXPECT_NEAR(field.cellDist(6, 6), 2.0f, 1e-5f);
      float vx, vy;
      ASSERT_TRUE(field.cellDir(6, 6, vx, vy));
      EXPECT_FLOAT_EQ(vx, 0.0f);
      EXPECT_FLOAT_EQ(vy, static_cast<float>(dy));
    }
  }
}

TEST(FlowField, OpenDiagonalKeepsItsShorterDistance)
{
  nav2_costmap_2d::Costmap2D map(15, 15, 1.0, 0.0, 0.0, 0);
  tgmppi::FlowField field;
  field.build(map, goal(7, 7), false, 0.0f, 6, 6, 10.0f);
  ASSERT_TRUE(field.ready());
  EXPECT_NEAR(field.cellDist(6, 6), std::sqrt(2.0f), 1e-5f);
}

TEST(FlowField, ReturnedPseudopodsNeverCutBlockedCorners)
{
  nav2_costmap_2d::Costmap2D map(15, 15, 1.0, 0.0, 0.0, 0);
  map.setCost(7, 6, 254);
  map.setCost(6, 7, 253);
  tgmppi::FlowField field;
  field.build(map, goal(14, 14), false, 0.0f, 6, 6, 5.0f, 5);
  ASSERT_TRUE(field.ready());
  ASSERT_FALSE(field.pseudopods().empty());
  for (const auto & pod : field.pseudopods()) {
    for (std::size_t k = 1; k < pod.size(); ++k) {
      const int x0 = std::lround(pod[k - 1].first);
      const int y0 = std::lround(pod[k - 1].second);
      const int x1 = std::lround(pod[k].first);
      const int y1 = std::lround(pod[k].second);
      EXPECT_LT(map.getCost(x1, y1), 253);
      if (x0 != x1 && y0 != y1) {
        EXPECT_LT(map.getCost(x0, y1), 253);
        EXPECT_LT(map.getCost(x1, y0), 253);
      }
    }
  }
}

TEST(FlowField, DiagonallyTouchingRoomsStayDisconnected)
{
  nav2_costmap_2d::Costmap2D map(15, 15, 1.0, 0.0, 0.0, 254);
  for (unsigned int y = 3; y <= 5; ++y) {
    for (unsigned int x = 3; x <= 5; ++x) {map.setCost(x, y, 0);}
  }
  for (unsigned int y = 6; y <= 8; ++y) {
    for (unsigned int x = 6; x <= 8; ++x) {map.setCost(x, y, 0);}
  }
  tgmppi::FlowField field;
  field.build(map, goal(4, 4), false, 0.0f, 4, 4, 10.0f);
  ASSERT_TRUE(field.ready());
  EXPECT_EQ(field.bodyCellCount(), 9u);
  EXPECT_FALSE(field.cellBody(6, 6));
  EXPECT_FALSE(field.cellWet(6, 6));
  // A free diagonal across the blocked corner is not a membrane exit.
  EXPECT_FALSE(field.cellMembrane(5, 5));
}
}  // namespace

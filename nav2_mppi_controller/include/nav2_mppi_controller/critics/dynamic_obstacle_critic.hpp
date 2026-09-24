// Copyright 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0 (the "License");

#ifndef NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_
#define NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_

#include "nav2_mppi_controller/critic_function.hpp"
#include "nav2_tgmppi_controller/tools/dynamic_obstacle_cost.hpp"

namespace mppi::critics
{

class DynamicObstacleCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

private:
  unsigned int power_{1};
  float weight_{3.81f};
  float cull_distance_{4.0f};
  tgmppi::DynamicObstacleCostParams params_;
};

}  // namespace mppi::critics

#endif  // NAV2_MPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_

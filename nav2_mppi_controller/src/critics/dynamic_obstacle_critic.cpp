// Copyright 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0 (the "License");

#include "nav2_mppi_controller/critics/dynamic_obstacle_critic.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <xtensor/xmath.hpp>

namespace mppi::critics
{

void DynamicObstacleCritic::initialize()
{
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(power_, "cost_power", 1);
  getParam(weight_, "cost_weight", 3.81f);
  getParam(params_.critical_cost, "critical_cost", 300.0f);
  getParam(params_.collision_cost, "collision_cost", 1.0e6f);
  getParam(params_.penetration_cost, "penetration_cost", 1.0e4f);
  getParam(params_.soft_distance, "soft_distance", 0.5f);
  getParam(params_.disc_offset, "disc_offset", 0.21f);
  getParam(params_.disc_radius, "disc_radius", 0.40f);
  getParam(params_.max_prediction_time, "max_prediction_time", 10.0f);
  int point_step = 1;
  getParam(point_step, "trajectory_point_step", 1, ParameterType::Static);
  params_.point_step = static_cast<std::size_t>(std::max(1, point_step));
  getParam(cull_distance_, "cull_distance", 4.0f);
}

void DynamicObstacleCritic::score(CriticData & data)
{
  if (!enabled_ || data.tracked_obstacles == nullptr || data.tracked_obstacles->empty()) {
    return;
  }
  const auto & traj = data.trajectories;
  const std::size_t K = traj.x.shape(0);
  const std::size_t T = traj.x.shape(1);
  if (K == 0 || T == 0) {
    return;
  }
  params_.model_dt = data.model_dt;
  data.dynamic_obstacle_params = params_;

  const float rx = static_cast<float>(data.state.pose.pose.position.x);
  const float ry = static_cast<float>(data.state.pose.pose.position.y);
  std::vector<tgmppi::SpaceTimeObstacle> nearby;
  nearby.reserve(data.tracked_obstacles->size());
  for (const auto & o : *data.tracked_obstacles) {
    if (std::hypot(o.x - rx, o.y - ry) <= cull_distance_) {
      nearby.push_back(o);
    }
  }
  if (nearby.empty()) {
    return;
  }

  auto cost = xt::xtensor<float, 1>::from_shape({K});
  const float * xs = traj.x.data();
  const float * ys = traj.y.data();
  const float * yaws = traj.yaws.data();
  for (std::size_t i = 0; i < K; ++i) {
    const auto r = tgmppi::scoreRolloutAgainstPredictions(
      xs + i * T, ys + i * T, yaws + i * T, T, nearby, params_);
    cost(i) = r.collides ?
      params_.collision_cost + params_.penetration_cost * r.penetration : r.repulsive;
  }
  data.costs += xt::pow(weight_ * cost / static_cast<float>(T), power_);
}

}  // namespace mppi::critics

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(mppi::critics::DynamicObstacleCritic, mppi::critics::CriticFunction)

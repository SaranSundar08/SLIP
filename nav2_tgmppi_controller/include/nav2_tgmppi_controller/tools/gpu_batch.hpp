// Copyright (c) 2026 SLIP thesis fork
// Licensed under the Apache License, Version 2.0.
#ifndef NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_BATCH_HPP_
#define NAV2_TGMPPI_CONTROLLER__TOOLS__GPU_BATCH_HPP_

#ifdef TGMPPI_WITH_CUDA
#include <vector>
#include <utility>
#include "nav2_tgmppi_controller/tools/gpu_rollout.hpp"
#include "nav2_tgmppi_controller/tools/dynamic_obstacle_cost.hpp"
#include "nav2_tgmppi_controller/models/path.hpp"

namespace tgmppi
{

// One iteration's device data. Critics add costs here; only the supervisor's
// costs/contact flags and final small control sequences cross back to the CPU.
// CPU-only critic plugins and visualization explicitly request a lazy snapshot.
class GpuBatch
{
public:
  struct Transfers
  {
    unsigned int control_uploads{0}, host_snapshots{0}, cost_downloads{0}, mean_downloads{0};
  };
  void initialize(unsigned int batch, unsigned int steps);
  bool ready() const {return rollout_.ready();}
  void begin(models::State & state, models::Trajectories & trajectories, float dt, bool omni);
  void materializeHost();
  void addCosts(const torch::Tensor & costs);
  void flushCosts(xt::xtensor<float, 1> & costs);
  void finishCritics(xt::xtensor<float, 1> & costs, std::vector<uint8_t> * contacts);
  void dynamicCosts(
    const std::vector<SpaceTimeObstacle> & obstacles, const DynamicObstacleCostParams & params,
    float weight, unsigned int power);
  size_t furthestPathPoint(const models::Path & path) const;
  size_t initialPathPoint(const models::Path & path) const;
  // Synchronize controls only after the CPU quarantine actually changes rows.
  void refreshControls();
  // The groups are disjoint contiguous row intervals; weights use original row indices.
  // Result [group, vx/wz/vy, time]. CPU retains mode selection and safety policy.
  xt::xtensor<float, 3> weightedMeans(
    const std::vector<std::pair<unsigned int, unsigned int>> & groups,
    const std::vector<float> & weights);
  const Transfers & transfers() const {return transfers_;}
  const torch::Tensor & vx() const {return rollout_.vx();}
  const torch::Tensor & vy() const {return rollout_.vy();}
  const torch::Tensor & wz() const {return rollout_.wz();}
  const torch::Tensor & trajX() const {return rollout_.trajX();}
  const torch::Tensor & trajY() const {return rollout_.trajY();}
  const torch::Tensor & trajYaws() const {return rollout_.trajYaws();}

private:
  GpuRollout rollout_;
  models::State * state_{nullptr};
  models::Trajectories * trajectories_{nullptr};
  unsigned int batch_{0}, steps_{0};
  bool omni_{false}, host_current_{false}, pending_costs_{false}, dynamic_scored_{false};
  torch::Tensor controls_, costs_, contacts_;
  Transfers transfers_;
};
}  // namespace tgmppi
#endif
#endif

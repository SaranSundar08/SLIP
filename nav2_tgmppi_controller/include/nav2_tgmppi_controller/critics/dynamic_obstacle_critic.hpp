#ifndef NAV2_TGMPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_
#define NAV2_TGMPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_

#include "nav2_tgmppi_controller/critic_function.hpp"
#include "nav2_tgmppi_controller/tools/dynamic_obstacle_cost.hpp"

namespace tgmppi::critics
{

/**
 * @class tgmppi::critics::DynamicObstacleCritic
 * @brief Scores every rollout point against each tracked obstacle's predicted
 * position at that point's time (constant velocity). CostCritic scores a frozen
 * costmap snapshot for all horizon steps, i.e. it assumes moving obstacles stand
 * still for the whole 2.8 s horizon; this critic removes that assumption.
 * Inert when no tracked obstacle has been received (e.g. static BARN worlds).
 * Does not set fail_flag: colliding rollouts are graded by penetration depth so
 * MPPI can still pick the least-bad escape instead of aborting the controller.
 */
class DynamicObstacleCritic : public CriticFunction
{
public:
  void initialize() override;
  void score(CriticData & data) override;

protected:
  unsigned int power_{1};
  float weight_{3.81f};
  float cull_distance_{4.0f};
  DynamicObstacleCostParams params_;
};

}  // namespace tgmppi::critics

#endif  // NAV2_TGMPPI_CONTROLLER__CRITICS__DYNAMIC_OBSTACLE_CRITIC_HPP_

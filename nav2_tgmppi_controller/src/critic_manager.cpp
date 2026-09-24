// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
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

#include "nav2_tgmppi_controller/critic_manager.hpp"

#include <cstdint>
#include <vector>

#include "nav2_tgmppi_controller/tools/utils.hpp"

namespace tgmppi
{

void CriticManager::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, ParametersHandler * param_handler)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  name_ = name;
  auto node = parent_.lock();
  logger_ = node->get_logger();
  parameters_handler_ = param_handler;

  getParams();
  loadCritics();
}

void CriticManager::getParams()
{
  auto node = parent_.lock();
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(critic_names_, "critics", std::vector<std::string>{}, ParameterType::Static);
}

void CriticManager::loadCritics()
{
  if (!loader_) {
    loader_ = std::make_unique<pluginlib::ClassLoader<critics::CriticFunction>>(
      "nav2_tgmppi_controller", "tgmppi::critics::CriticFunction");
  }

  critics_.clear();
  for (auto name : critic_names_) {
    std::string fullname = getFullName(name);
    auto instance = std::unique_ptr<critics::CriticFunction>(
      loader_->createUnmanagedInstance(fullname));
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + name, costmap_ros_,
      parameters_handler_);
    RCLCPP_INFO(logger_, "Critic loaded : %s", fullname.c_str());
  }
}

std::string CriticManager::getFullName(const std::string & name)
{
  return "tgmppi::critics::" + name;
}

void CriticManager::evalTrajectoriesScores(
  CriticData & data) const
{
  // 2026-09-16 diagnostics: bags tgmppi_dyn_20260915_234156 / _20260916_001024 /
  // _20260916_002534 showed rollout costs of -1e19..-1e25 and then NaN commands.
  // Name the first critic that turns a row's cost non-finite or implausible
  // (no critic adds negative cost; none comes near 1e12).
  static unsigned int corrupt_logs = 0u;
  const auto implausible = [](float c) {
      return utils::isBadFloat(c) || c < -1.0e3f || c > 1.0e12f;
    };
  const std::size_t n = data.costs.shape(0);
  std::vector<uint8_t> bad;
  if (corrupt_logs < 20u && data.gpu_batch == nullptr) {
    bad.resize(n);
    for (std::size_t r = 0; r < n; ++r) {bad[r] = implausible(data.costs(r)) ? 1u : 0u;}
  }
  for (size_t q = 0; q < critics_.size(); q++) {
    if (data.fail_flag) {
      break;
    }
#ifdef TGMPPI_WITH_CUDA
    if (data.gpu_batch && !critics_[q]->supportsGpuBatch()) {
      data.gpu_batch->materializeHost();
      data.gpu_batch->flushCosts(data.costs);
    }
#endif
    critics_[q]->score(data);
    if (bad.empty() || corrupt_logs >= 20u) {
      continue;
    }
    std::size_t fresh = 0, first = 0;
    for (std::size_t r = 0; r < n; ++r) {
      if (!bad[r] && implausible(data.costs(r))) {
        bad[r] = 1u;
        if (fresh++ == 0) {first = r;}
      }
    }
    if (fresh > 0) {
      ++corrupt_logs;
      RCLCPP_WARN(
        logger_, "[TGMPPI diag] critic %s made %zu row(s) corrupt; first row %zu cost %g",
        q < critic_names_.size() ? critic_names_[q].c_str() : "?", fresh, first,
        data.costs(first));
    }
  }
#ifdef TGMPPI_WITH_CUDA
  if (data.gpu_batch) {
    data.gpu_batch->finishCritics(data.costs, data.dynamic_collision_rows);
    // Device critics accumulate together. Diagnose at the synchronization
    // boundary without falsely attributing earlier device costs to a CPU plugin.
    for (std::size_t r = 0; r < n && corrupt_logs < 20u; ++r) {
      if (implausible(data.costs(r))) {
        ++corrupt_logs;
        RCLCPP_WARN(logger_, "[TGMPPI diag] GPU critic batch corrupt at row %zu cost %g",
          r, data.costs(r));
        break;
      }
    }
  }
#endif
}

}  // namespace tgmppi

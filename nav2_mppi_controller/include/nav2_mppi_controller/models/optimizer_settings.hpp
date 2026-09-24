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

#ifndef NAV2_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
#define NAV2_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

#include <cstddef>
#include "nav2_mppi_controller/models/constraints.hpp"

namespace mppi::models
{

/**
 * @enum mppi::models::MPPIVariant
 * @brief Selects which MPPI sampling/weighting variant to run (SLIP study).
 *        Selected at runtime via the "mppi_variant" parameter.
 */
enum class MPPIVariant
{
  VANILLA,   // stock MPPI: i.i.d. Gaussian sampling noise + softmax weighting
  LOG,       // log-MPPI: Normal-Log-Normal (heavy-tailed) sampling noise
  LOWPASS,   // LP-MPPI: temporally low-pass filtered (colored) sampling noise
  BIASED     // Biased-MPPI: sampling mean shifted toward an ancillary controller
};

/**
 * @enum mppi::models::AncillaryType
 * @brief Which ancillary controller Biased-MPPI shifts samples toward (SLIP study).
 *        Selected at runtime via the "ancillary_type" parameter. Experiment axis:
 *        PURSUIT is guidance (follow the path), BRAKING is safety (avoid obstacles).
 */
enum class AncillaryType
{
  PURSUIT,   // pursuit-to-path P-controller: biases samples toward following the plan
  BRAKING    // CBF/braking reactive controller: biases samples toward safe motion
};

/**
 * @struct mppi::models::OptimizerSettings
 * @brief Settings for the optimizer to use
 */
struct OptimizerSettings
{
  models::ControlConstraints base_constraints{0, 0, 0, 0};
  models::ControlConstraints constraints{0, 0, 0, 0};
  models::SamplingStd sampling_std{0, 0, 0};
  float model_dt{0};
  float temperature{0};
  float gamma{0};
  unsigned int batch_size{0};
  unsigned int time_steps{0};
  unsigned int iteration_count{0};
  bool shift_control_sequence{false};
  size_t retry_attempt_limit{0};

  // --- SLIP MPPI variant selection + per-variant knobs ---
  MPPIVariant variant{MPPIVariant::VANILLA};
  float lognormal_sigma{0.1f};        // log-MPPI: std of the log-normal scale factor
  float lowpass_cutoff_hz{2.0f};      // LP-MPPI: -3dB cutoff (Hz); lower = smoother
  int lowpass_order{1};               // LP-MPPI: cascaded EMA passes; higher = steeper rolloff
  float bias_strength{0.5f};          // Biased-MPPI: fraction of samples shifted to ancillary
  float bias_lookahead_dist{0.6f};    // Biased-MPPI: pursuit lookahead distance (m)
  float bias_gain{1.5f};              // Biased-MPPI: pursuit heading P-gain
  AncillaryType ancillary_type{AncillaryType::PURSUIT};  // which ancillary to bias toward
  float brake_gain{1.5f};             // CBF/braking: obstacle-repulsion steering P-gain
  float brake_scan_dist{0.3f};        // CBF/braking: cost-gradient finite-difference offset (m)
};

}  // namespace mppi::models

#endif  // NAV2_MPPI_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

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

// Space-time topology validation obstacle (2026-09-13): a single-axis
// sinusoidal oscillator, the same DynaBARN/amoeba_sandbox dynabarn.py
// motion profile (constant center, amplitude, omega on one axis), reimplemented
// clean rather than reused from DynaBARN's own pre-built plugins -- those
// link against Gazebo 9 / ignition-common1 / boost 1.65, none of which
// exist on this Gazebo 11 machine, and DynaBARN ships no plugin source to
// rebuild, only the generator script that emits equally simple C++ (see
// create_plugin_polynomial.py's make_head/make_waypoint/make_tail, which
// use the same gazebo::common::PoseAnimation family of APIs this file's
// approach is a simpler alternative to -- direct SetWorldPose each tick
// needs no pre-baked keyframe list).
//
// Deliberately does NOT publish its own odometry: attach a second
// <plugin filename="libgazebo_ros_p3d.so"> to the same model/link (the
// same plugin this project already uses for the real robot's ground
// truth) to get position+velocity for free, regardless of what's driving
// the model's motion.

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <ignition/math/Vector2.hh>
#include <ignition/math/Vector3.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Pose3.hh>

namespace gazebo
{

class OscillatingObstaclePlugin : public ModelPlugin
{
public:
  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    center_x_ = sdf->HasElement("center_x") ? sdf->Get<double>("center_x") : 0.0;
    center_y_ = sdf->HasElement("center_y") ? sdf->Get<double>("center_y") : 0.0;
    z_ = sdf->HasElement("z") ? sdf->Get<double>("z") : 0.3;
    amplitude_ = sdf->HasElement("amplitude") ? sdf->Get<double>("amplitude") : 0.6;
    omega_ = sdf->HasElement("omega") ? sdf->Get<double>("omega") : 0.4;
    // Phase offset (rad), 2026-09-16: without it every obstacle starts at its
    // centre moving the same way, so repeated runs of an experiment are not
    // independent samples -- the whole scene replays identically.
    phase_ = sdf->HasElement("phase") ? sdf->Get<double>("phase") : 0.0;

    // Waypoint mode (2026-09-16): DynaBARN's released worlds ship one compiled
    // plugin per obstacle, built against Gazebo 9 and without source, so they
    // cannot run here (Gazebo 11 / ROS 2). Their generator (polynomial_fit.py)
    // IS public, so scenarios are reproduced by fitting a polynomial through
    // random waypoints and handing the sampled track to this plugin. With
    // <waypoints> present the obstacle ping-pongs along that track at <speed>;
    // without it the original sinusoid is used unchanged.
    if (sdf->HasElement("waypoints")) {
      std::istringstream ss(sdf->Get<std::string>("waypoints"));
      double wx = 0.0, wy = 0.0;
      while (ss >> wx >> wy) {
        track_.emplace_back(wx, wy);
      }
      speed_ = sdf->HasElement("speed") ? sdf->Get<double>("speed") : 0.5;
      arc_.assign(track_.size(), 0.0);
      for (std::size_t i = 1; i < track_.size(); ++i) {
        arc_[i] = arc_[i - 1] + track_[i - 1].Distance(track_[i]);
      }
    }
    axis_is_x_ = !(sdf->HasElement("axis") && sdf->Get<std::string>("axis") == "y");

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&OscillatingObstaclePlugin::OnUpdate, this));
  }

private:
  // Ping-pong along the track at constant speed: distance travelled folds into
  // [0, length] and reverses, so an obstacle sweeps its route back and forth
  // instead of teleporting to the start.
  void FollowTrack(double t)
  {
    // Trapezoidal speed profile (2026-09-17): accelerate from and decelerate to
    // a stop over distance d at each end, instead of reversing instantly. An
    // instant 180-degree reversal is not how people or carts move, and it is the
    // single worst case for a constant-velocity predictor, which keeps
    // extrapolating the old direction at full speed. <speed> is now the PEAK.
    const double length = arc_.back();
    const double v_peak = std::max(1e-3, speed_);
    const double d = std::min(1.0, 0.25 * length);
    const double accel = v_peak * v_peak / (2.0 * d);
    const double t_ramp = v_peak / accel;                       // = 2d / v_peak
    const double t_cruise = std::max(0.0, (length - 2.0 * d) / v_peak);
    const double t_half = 2.0 * t_ramp + t_cruise;
    const double period = 2.0 * t_half;
    double tau = std::fmod(t + phase_ / 6.2832 * period, period);
    if (tau < 0.0) {tau += period;}
    double dir = 1.0;
    if (tau > t_half) {
      tau -= t_half;
      dir = -1.0;
    }
    double s_fwd = 0.0, spd = 0.0;
    if (tau < t_ramp) {
      s_fwd = 0.5 * accel * tau * tau;
      spd = accel * tau;
    } else if (tau < t_ramp + t_cruise) {
      s_fwd = d + v_peak * (tau - t_ramp);
      spd = v_peak;
    } else {
      const double r = std::max(0.0, t_half - tau);
      s_fwd = length - 0.5 * accel * r * r;
      spd = accel * r;
    }
    const double s_pos = (dir > 0.0) ? s_fwd : length - s_fwd;
    std::size_t k = 1;
    while (k + 1 < arc_.size() && arc_[k] < s_pos) {++k;}
    const double seg = arc_[k] - arc_[k - 1];
    const double frac = (seg > 1e-9) ? (s_pos - arc_[k - 1]) / seg : 0.0;
    const auto p0 = track_[k - 1];
    const auto p1 = track_[k];
    const double x = p0.X() + frac * (p1.X() - p0.X());
    const double y = p0.Y() + frac * (p1.Y() - p0.Y());
    const double inv = (seg > 1e-9) ? 1.0 / seg : 0.0;
    const double vx = dir * spd * (p1.X() - p0.X()) * inv;
    const double vy = dir * spd * (p1.Y() - p0.Y()) * inv;
    model_->SetWorldPose(ignition::math::Pose3d(x, y, z_, 0, 0, 0));
    model_->SetLinearVel(ignition::math::Vector3d(vx, vy, 0));
    model_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
  }

  void OnUpdate()
  {
    const double t = model_->GetWorld()->SimTime().Double();
    if (track_.size() >= 2 && arc_.back() > 1e-6) {
      FollowTrack(t);
      return;
    }
    const double offset = amplitude_ * std::sin(omega_ * t + phase_);
    const double speed = amplitude_ * omega_ * std::cos(omega_ * t + phase_);
    const double x = axis_is_x_ ? center_x_ + offset : center_x_;
    const double y = axis_is_x_ ? center_y_ : center_y_ + offset;
    const double vx = axis_is_x_ ? speed : 0.0;
    const double vy = axis_is_x_ ? 0.0 : speed;
    model_->SetWorldPose(ignition::math::Pose3d(x, y, z_, 0, 0, 0));
    // SetWorldPose() alone teleports the pose without updating the physics
    // engine's own velocity state -- anything reading link velocity (e.g.
    // libgazebo_ros_p3d's twist field) sees stale/garbage values otherwise
    // (observed: near-zero x, a large bogus z from the discontinuous-
    // position artifact). Setting it explicitly to the same analytic
    // derivative used for the position keeps both consistent.
    model_->SetLinearVel(ignition::math::Vector3d(vx, vy, 0.0));
  }

  physics::ModelPtr model_;
  event::ConnectionPtr update_connection_;
  double center_x_{0.0};
  double center_y_{0.0};
  double z_{0.3};
  double amplitude_{0.6};
  double omega_{0.4};
  double phase_{0.0};
  double speed_{0.5};
  std::vector<ignition::math::Vector2d> track_;
  std::vector<double> arc_;
  bool axis_is_x_{true};
};

GZ_REGISTER_MODEL_PLUGIN(OscillatingObstaclePlugin)

}  // namespace gazebo

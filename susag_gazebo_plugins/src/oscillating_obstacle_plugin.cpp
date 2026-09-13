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
    axis_is_x_ = !(sdf->HasElement("axis") && sdf->Get<std::string>("axis") == "y");

    update_connection_ = event::Events::ConnectWorldUpdateBegin(
      std::bind(&OscillatingObstaclePlugin::OnUpdate, this));
  }

private:
  void OnUpdate()
  {
    const double t = model_->GetWorld()->SimTime().Double();
    const double offset = amplitude_ * std::sin(omega_ * t);
    const double speed = amplitude_ * omega_ * std::cos(omega_ * t);
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
  bool axis_is_x_{true};
};

GZ_REGISTER_MODEL_PLUGIN(OscillatingObstaclePlugin)

}  // namespace gazebo

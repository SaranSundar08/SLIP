#!/usr/bin/env python3
"""Vicon-like localization for Gazebo (2026-09-15).

The real trials localize with Vicon markers (near-exact pose), not AMCL.
This node gives the simulator the same property: it publishes map->odom
from Gazebo's /ground_truth/odom (libgazebo_ros_p3d, base_link in world,
no noise), so the robot's map pose IS its true pose. AMCL keeps running
(Nav2's localization lifecycle expects it) but navigation.launch.py sets
its tf_broadcast to false, so exactly one node owns map->odom.

Assumes the Gazebo world frame == map frame, which holds for the BARN
scaled_1 and junction/open worlds (maps are generated in world coordinates).

map->odom = T(map<-base, ground truth) * inverse(T(odom<-base, TF))
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def compose(a, b):
    ax, ay, at = a
    bx, by, bt = b
    return (ax + math.cos(at) * bx - math.sin(at) * by,
            ay + math.sin(at) * bx + math.cos(at) * by,
            math.atan2(math.sin(at + bt), math.cos(at + bt)))


def inverse(a):
    x, y, t = a
    c, s = math.cos(t), math.sin(t)
    return (-(c * x + s * y), -(-s * x + c * y), -t)


class GroundTruthLocalizer(Node):
    def __init__(self):
        super().__init__('ground_truth_localizer')
        self.gt_topic = self.declare_parameter('ground_truth_topic', '/ground_truth/odom').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        # Same as AMCL's transform_tolerance (1.0 s in our yamls): map->odom is
        # stamped at the NEWEST odom->base_link stamp plus this, so a consumer
        # asking for map->odom at its newest robot-pose time never extrapolates
        # into the future. 0.1 s off ground-truth time was too little: the
        # controller asked 0.27 s past it and aborted every goal (2026-09-15).
        self.future_dating = self.declare_parameter('future_dating', 1.0).value
        self.buffer = Buffer()
        # /tf (~540 Hz here) is received by a SEPARATE helper node on its own
        # thread, so the ground-truth callback can never starve it. It must be a
        # separate node: in Humble, TransformListener(spin_thread=True) adds the
        # node it is given to a second executor, and rclpy moves a node between
        # executors, so doing that with this node while rclpy.spin(this node)
        # runs stalls both (observed live 2026-09-15: all threads idle, no callbacks).
        use_sim_time = self.get_parameter('use_sim_time').value
        self.tf_node = rclpy.create_node(
            'ground_truth_localizer_tf',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)])
        self.listener = TransformListener(self.buffer, self.tf_node, spin_thread=True)
        self.broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/ground_truth_localization/pose', 10)
        self.create_subscription(Odometry, self.gt_topic, self.on_gt, 10)
        self.published = 0
        self.fallbacks = 0
        self.gt_received = 0
        self.last_error = ''
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'publishing {self.map_frame}->{self.odom_frame} from {self.gt_topic} '
            f'(AMCL tf_broadcast must be false)')

    def on_gt(self, msg):
        self.gt_received += 1
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        map_base = (p.x, p.y, yaw_of(q))
        stamp = Time.from_msg(msg.header.stamp)
        # Never block here. Exact stamp if the buffer already has it, else newest.
        try:
            tf = self.buffer.lookup_transform(self.odom_frame, self.base_frame, stamp)
        except Exception as exact_err:
            try:
                tf = self.buffer.lookup_transform(self.odom_frame, self.base_frame, Time())
                self.fallbacks += 1
            except Exception as latest_err:
                self.last_error = (f'exact-stamp lookup: {type(exact_err).__name__}: {exact_err} | '
                                   f'latest lookup: {type(latest_err).__name__}: {latest_err}')
                return  # no usable odom->base_link
        t = tf.transform
        odom_base = (t.translation.x, t.translation.y, yaw_of(t.rotation))
        mx, my, myaw = compose(map_base, inverse(odom_base))

        # Stamp off the newest odom data we know of, never older than ground truth.
        try:
            newest = self.buffer.lookup_transform(self.odom_frame, self.base_frame, Time())
            base_stamp = max(Time.from_msg(newest.header.stamp).nanoseconds, stamp.nanoseconds)
        except Exception:
            base_stamp = stamp.nanoseconds
        out = TransformStamped()
        out.header.stamp = (Time(nanoseconds=base_stamp, clock_type=stamp.clock_type)
                            + Duration(seconds=self.future_dating)).to_msg()
        out.header.frame_id = self.map_frame
        out.child_frame_id = self.odom_frame
        out.transform.translation.x, out.transform.translation.y = mx, my
        out.transform.rotation.z, out.transform.rotation.w = math.sin(myaw / 2.0), math.cos(myaw / 2.0)
        self.broadcaster.sendTransform(out)

        pose = PoseWithCovarianceStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.map_frame
        pose.pose.pose = msg.pose.pose
        self.pose_pub.publish(pose)
        self.published += 1

    def report(self):
        msg = (f'last 10 s: ground truth msgs {self.gt_received}, map->odom published '
               f'{self.published} ({self.fallbacks} via latest odom TF)')
        if self.gt_received == 0:
            self.get_logger().warn(msg + f' -- NO {self.gt_topic} received (Gazebo down, or '
                                   f'different RMW/ROS_DOMAIN_ID than the Gazebo terminal?)')
        elif self.published == 0:
            self.get_logger().warn(msg + f' -- odom->base_link lookup failing: {self.last_error}')
        else:
            self.get_logger().info(msg)
        self.published = 0
        self.fallbacks = 0
        self.gt_received = 0
        self.last_error = ''


def main():
    rclpy.init()
    node = GroundTruthLocalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C / launch shutdown: exit quietly
    finally:
        node.tf_node.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

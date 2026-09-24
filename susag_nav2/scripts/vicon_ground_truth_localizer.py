#!/usr/bin/env python3
"""Vicon ground-truth localization for the real robot (2026-09-22).

Real-robot counterpart to ground_truth_localizer.py (which does the same
thing in Gazebo from /ground_truth/odom). Here the source is the lab's
Vicon system (ros2-vicon-bridge), publishing PoseStamped on
/vicon/<subject>/<segment>/pose in the Vicon world frame.

FAST-LIO (fast_lio_robosenseAiry) already publishes camera_init->base_link
live (base_link_odom_en: true). This node fuses that with Vicon the same
way the Gazebo version fuses /ground_truth/odom with odom->base_link:

    map->camera_init = T(map<-base, Vicon) * inverse(T(camera_init<-base, TF))

IMPORTANT -- exactly one node may own map->camera_init at a time:
  - this node (real-robot trials, Vicon as ground truth), OR
  - slam_toolbox in mapping mode (building a map), OR
  - the static_transform_publisher in mapping_robosense_airy.launch.py
    (gravity-aligned map for RViz display only).
Running two of these together silently double-broadcasts the same TF edge
and interleaves on /tf -- this was diagnosed as the cause of unstable
SLAM registration on 2026-09-22. Don't launch slam_toolbox's mapping mode
or that static_transform_publisher while this node is running.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.executors import ExternalShutdownException
from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped, PoseStamped, PoseWithCovarianceStamped
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


class ViconGroundTruthLocalizer(Node):
    def __init__(self):
        super().__init__('vicon_ground_truth_localizer')
        self.vicon_topic = self.declare_parameter(
            'vicon_pose_topic', '/vicon/SLIP/SLIP/pose').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'camera_init').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        # Same rationale as ground_truth_localizer.py: stamp map->odom_frame
        # past the newest odom_frame->base_frame stamp so a consumer never
        # extrapolates into the future and aborts (0.1 s was too little there).
        self.future_dating = self.declare_parameter('future_dating', 1.0).value
        self.buffer = Buffer()
        # Separate node/thread for the TF listener so a slow Vicon callback
        # can never starve it -- same fix as ground_truth_localizer.py
        # (Humble: spin_thread=True moves the node to a second executor,
        # doing that with the node rclpy.spin()s stalls both).
        use_sim_time = self.get_parameter('use_sim_time').value
        self.tf_node = rclpy.create_node(
            'vicon_ground_truth_localizer_tf',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)])
        self.listener = TransformListener(self.buffer, self.tf_node, spin_thread=True)
        self.broadcaster = TransformBroadcaster(self)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/vicon_ground_truth_localization/pose', 10)
        self.create_subscription(PoseStamped, self.vicon_topic, self.on_vicon, 10)
        self.published = 0
        self.fallbacks = 0
        self.vicon_received = 0
        self.last_error = ''
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            f'publishing {self.map_frame}->{self.odom_frame} from {self.vicon_topic} '
            f'(nothing else may broadcast {self.map_frame}->{self.odom_frame} right now)')

    def on_vicon(self, msg):
        self.vicon_received += 1
        p, q = msg.pose.position, msg.pose.orientation
        map_base = (p.x, p.y, yaw_of(q))
        stamp = Time.from_msg(msg.header.stamp)
        try:
            tf = self.buffer.lookup_transform(self.odom_frame, self.base_frame, stamp)
        except Exception as exact_err:
            try:
                tf = self.buffer.lookup_transform(self.odom_frame, self.base_frame, Time())
                self.fallbacks += 1
            except Exception as latest_err:
                self.last_error = (f'exact-stamp lookup: {type(exact_err).__name__}: {exact_err} | '
                                   f'latest lookup: {type(latest_err).__name__}: {latest_err}')
                return  # no usable odom_frame->base_frame yet (is FAST-LIO running?)
        t = tf.transform
        odom_base = (t.translation.x, t.translation.y, yaw_of(t.rotation))
        mx, my, myaw = compose(map_base, inverse(odom_base))

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
        pose.pose.pose = msg.pose
        self.pose_pub.publish(pose)
        self.published += 1

    def report(self):
        msg = (f'last 10 s: vicon msgs {self.vicon_received}, {self.map_frame}->{self.odom_frame} '
               f'published {self.published} ({self.fallbacks} via latest odom TF)')
        if self.vicon_received == 0:
            self.get_logger().warn(msg + f' -- NO {self.vicon_topic} received (is vicon_bridge '
                                   f'running, and is the subject/segment name right?)')
        elif self.published == 0:
            self.get_logger().warn(msg + f' -- {self.odom_frame}->{self.base_frame} lookup '
                                   f'failing: {self.last_error} (is FAST-LIO running with '
                                   f'base_link_odom_en: true?)')
        else:
            self.get_logger().info(msg)
        self.published = 0
        self.fallbacks = 0
        self.vicon_received = 0
        self.last_error = ''


def main():
    rclpy.init()
    node = ViconGroundTruthLocalizer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.tf_node.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

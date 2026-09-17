#!/usr/bin/env python3
"""Sim-only: drop laser returns that hit TRACKED moving obstacles (2026-09-15).

With DynamicObstacleCritic scoring moving obstacles at their predicted positions,
the costmap must stop marking those same obstacles at their CURRENT positions,
otherwise CostCritic keeps penalising exactly the "pass behind" motion the
prediction says is safe (see PROJECT_STATUS.md, T-MPC analysis). Static geometry
is untouched: only beams whose endpoint lies within radius + margin of a tracked
obstacle are removed (set to inf; the costmaps use inf_is_valid: false, so those
beams neither mark nor clear).

Obstacles are discovered automatically: any topic matching
/moving_obstacle_*/ground_truth/odom. With none present (static BARN worlds)
the scan passes through unchanged.
"""
import math
import re

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

DEFAULT_TOPIC_RE = r"^/moving_obstacle_[^/]+/ground_truth/odom$"


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class TrackedObstacleScanFilter(Node):
    def __init__(self):
        super().__init__('tracked_obstacle_scan_filter')
        self.scan_in = self.declare_parameter('scan_in', '/front_scan').value
        self.scan_out = self.declare_parameter('scan_out', '/front_scan_filtered').value
        self.radius = self.declare_parameter('obstacle_radius', 0.25).value
        self.margin = self.declare_parameter('margin', 0.10).value
        self.fixed_frame = self.declare_parameter('fixed_frame', 'map').value
        # Which obstacle states to filter by. With tracking degradation active the
        # costmap must be cleared using the SAME (degraded) estimates the controller
        # plans on, not ground truth -- otherwise the scan filter quietly hands back
        # perfect knowledge through the back door.
        self.topic_re = re.compile(
            self.declare_parameter('odom_pattern', DEFAULT_TOPIC_RE).value)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)   # no spin_thread: same executor as this node
        self.obstacles = {}
        self.subs = {}
        self.pub = self.create_publisher(LaserScan, self.scan_out, qos_profile_sensor_data)
        self.create_subscription(LaserScan, self.scan_in, self.on_scan, qos_profile_sensor_data)
        self.create_timer(1.0, self.discover)
        self.stats = dict(scans=0, removed=0, passthrough_tf=0)
        self.create_timer(10.0, self.report)
        self.get_logger().info(f'filtering {self.scan_in} -> {self.scan_out}, obstacle radius {self.radius} + margin {self.margin}')

    def discover(self):
        for name, _ in self.get_topic_names_and_types():
            if self.topic_re.match(name) and name not in self.subs:
                self.subs[name] = self.create_subscription(
                    Odometry, name, lambda m, n=name: self.obstacles.__setitem__(n, (m.pose.pose.position.x, m.pose.pose.position.y)), 10)
                self.get_logger().info(f'tracking {name}')

    def on_scan(self, scan):
        self.stats['scans'] += 1
        out = scan
        if self.obstacles:
            try:
                tf = self.buffer.lookup_transform(self.fixed_frame, scan.header.frame_id, Time())
            except Exception:
                self.stats['passthrough_tf'] += 1
                self.pub.publish(scan)
                return
            tx, ty = tf.transform.translation.x, tf.transform.translation.y
            tyaw = yaw_of(tf.transform.rotation)
            ranges = np.asarray(scan.ranges, dtype=np.float32)
            angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment + tyaw
            valid = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
            ex = tx + ranges * np.cos(angles)
            ey = ty + ranges * np.sin(angles)
            hit = np.zeros(len(ranges), dtype=bool)
            limit = self.radius + self.margin
            for ox, oy in self.obstacles.values():
                hit |= valid & (np.hypot(ex - ox, ey - oy) <= limit)
            n = int(hit.sum())
            if n:
                ranges = ranges.copy()
                ranges[hit] = np.inf
                out = LaserScan()
                out.header = scan.header
                out.angle_min, out.angle_max, out.angle_increment = scan.angle_min, scan.angle_max, scan.angle_increment
                out.time_increment, out.scan_time = scan.time_increment, scan.scan_time
                out.range_min, out.range_max = scan.range_min, scan.range_max
                out.ranges = ranges.tolist()
                out.intensities = scan.intensities
                self.stats['removed'] += n
        self.pub.publish(out)

    def report(self):
        self.get_logger().info(
            f"last 10 s: {self.stats['scans']} scans, {self.stats['removed']} beams removed on "
            f"{len(self.obstacles)} tracked obstacles, {self.stats['passthrough_tf']} passed through (no TF)")
        self.stats = dict(scans=0, removed=0, passthrough_tf=0)


def main():
    rclpy.init()
    node = TrackedObstacleScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()

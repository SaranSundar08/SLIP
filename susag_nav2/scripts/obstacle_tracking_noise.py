#!/usr/bin/env python3
"""Degrade ground-truth obstacle states into realistic TRACKER output (sim only).

The dynamic-obstacle pipeline currently consumes Gazebo p3d ground truth: exact
position and velocity, every obstacle, no latency, no misses. That is perfect
tracking, and it is what T-MPC also had (motion capture + a Kalman filter), but
it is NOT what a lidar-based tracker on the real robot would deliver -- and the
honest question for the thesis is how good tracking has to be before the
prediction-based avoidance stops working.

This node sits between the two: it subscribes to every
/moving_obstacle_*/ground_truth/odom and republishes
/moving_obstacle_*/tracked/odom with configurable position noise, velocity
noise, latency, dropout and update rate. Point the controller (and the scan
filter) at the tracked topics and the whole stack runs unchanged, only on worse
information. With every knob at zero this is a pass-through, so "perfect" stays
available as the reference condition.
"""
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np

from nav_msgs.msg import Odometry

SRC_RE = re.compile(r"^/moving_obstacle_([^/]+)/ground_truth/odom$")


class ObstacleTrackingNoise(Node):
    def __init__(self):
        super().__init__('obstacle_tracking_noise')
        self.declare_parameter('pos_sigma', 0.0)      # m, per axis
        self.declare_parameter('vel_sigma', 0.0)      # m/s, per axis
        self.declare_parameter('latency', 0.0)        # s, reporting delay
        self.declare_parameter('dropout', 0.0)        # probability a message is lost
        self.declare_parameter('rate', 0.0)           # Hz, 0 = republish every input
        self.declare_parameter('seed', 0)
        g = self.get_parameter
        self.pos_sigma = float(g('pos_sigma').value)
        self.vel_sigma = float(g('vel_sigma').value)
        self.latency = float(g('latency').value)
        self.dropout = float(g('dropout').value)
        self.rate = float(g('rate').value)
        self.rng = np.random.default_rng(int(g('seed').value))

        self.subs = {}
        self.pubs = {}
        self.queues = {}        # name -> [(release_time, Odometry), ...]
        self.last_pub = {}
        self.n_in = self.n_out = self.n_dropped = 0
        self.create_timer(1.0, self.discover)
        self.create_timer(0.02, self.release)
        self.create_timer(10.0, self.report)
        self.get_logger().info(
            'tracking degradation: pos_sigma %.3f m, vel_sigma %.3f m/s, latency %.3f s, '
            'dropout %.2f, rate %.1f Hz'
            % (self.pos_sigma, self.vel_sigma, self.latency, self.dropout, self.rate))

    def discover(self):
        for topic, types in self.get_topic_names_and_types():
            m = SRC_RE.match(topic)
            if not m or topic in self.subs or 'nav_msgs/msg/Odometry' not in types:
                continue
            name = m.group(1)
            out = '/moving_obstacle_%s/tracked/odom' % name
            self.pubs[name] = self.create_publisher(Odometry, out, qos_profile_sensor_data)
            self.queues[name] = []
            self.subs[topic] = self.create_subscription(
                Odometry, topic, lambda msg, n=name: self.on_odom(n, msg),
                qos_profile_sensor_data)
            self.get_logger().info('tracking %s -> %s' % (topic, out))

    def on_odom(self, name, msg):
        self.n_in += 1
        if self.dropout > 0.0 and self.rng.random() < self.dropout:
            self.n_dropped += 1
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.rate > 0.0 and (now - self.last_pub.get(name, -1e9)) < (1.0 / self.rate):
            return
        self.last_pub[name] = now
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = msg.child_frame_id
        out.pose.pose.position.x = msg.pose.pose.position.x + self.noise(self.pos_sigma)
        out.pose.pose.position.y = msg.pose.pose.position.y + self.noise(self.pos_sigma)
        out.pose.pose.position.z = msg.pose.pose.position.z
        out.pose.pose.orientation = msg.pose.pose.orientation
        out.twist.twist.linear.x = msg.twist.twist.linear.x + self.noise(self.vel_sigma)
        out.twist.twist.linear.y = msg.twist.twist.linear.y + self.noise(self.vel_sigma)
        self.queues[name].append((now + self.latency, out))

    def noise(self, sigma):
        return float(self.rng.normal(0.0, sigma)) if sigma > 0.0 else 0.0

    def release(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        for name, q in self.queues.items():
            while q and q[0][0] <= now:
                self.pubs[name].publish(q.pop(0)[1])
                self.n_out += 1

    def report(self):
        self.get_logger().info(
            'last 10 s: %d in, %d out, %d dropped, %d obstacle(s)'
            % (self.n_in, self.n_out, self.n_dropped, len(self.pubs)))
        self.n_in = self.n_out = self.n_dropped = 0


def main():
    rclpy.init()
    node = ObstacleTrackingNoise()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

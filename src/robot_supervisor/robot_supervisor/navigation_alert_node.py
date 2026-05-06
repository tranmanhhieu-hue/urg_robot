#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

import tf2_ros


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class NavigationAlertNode(Node):
    def __init__(self):
        super().__init__('navigation_alert_node')

        self.declare_parameter('path_topic', '/plan')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('alert_topic', '/nav_alert_text')

        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('global_frame', 'map')

        self.declare_parameter('turn_lookahead_m', 1.0)
        self.declare_parameter('turn_after_m', 0.7)
        self.declare_parameter('turn_angle_threshold_deg', 35.0)

        self.declare_parameter('obstacle_distance_m', 1.0)
        self.declare_parameter('front_angle_deg', 25.0)

        self.declare_parameter('cooldown_sec', 5.0)

        self.path_topic = self.get_parameter('path_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.alert_topic = self.get_parameter('alert_topic').value

        self.robot_frame = self.get_parameter('robot_frame').value
        self.global_frame = self.get_parameter('global_frame').value

        self.turn_lookahead_m = self.get_parameter('turn_lookahead_m').value
        self.turn_after_m = self.get_parameter('turn_after_m').value
        self.turn_angle_threshold = math.radians(
            self.get_parameter('turn_angle_threshold_deg').value
        )

        self.obstacle_distance_m = self.get_parameter('obstacle_distance_m').value
        self.front_angle = math.radians(self.get_parameter('front_angle_deg').value)

        self.cooldown_sec = self.get_parameter('cooldown_sec').value

        self.latest_path = None
        self.last_alert_time = 0.0
        self.last_turn_alert = ''

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.alert_pub = self.create_publisher(String, self.alert_topic, 10)

        self.create_subscription(Path, self.path_topic, self.path_callback, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)

        self.create_timer(0.3, self.check_turn_alert)

        self.get_logger().info('Navigation alert node started')

    def path_callback(self, msg):
        self.latest_path = msg

    def can_alert(self):
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_alert_time >= self.cooldown_sec:
            self.last_alert_time = now
            return True
        return False

    def speak_alert(self, text):
        if not self.can_alert():
            return

        msg = String()
        msg.data = text
        self.alert_pub.publish(msg)
        self.get_logger().info(f'ALERT: {text}')

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_frame,
                rclpy.time.Time()
            )

            x = tf.transform.translation.x
            y = tf.transform.translation.y

            q = tf.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            return x, y, yaw

        except Exception as e:
            self.get_logger().warn(f'Cannot get TF {self.global_frame}->{self.robot_frame}: {e}')
            return None

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def find_nearest_index(self, path_points, rx, ry):
        min_dist = 999.0
        min_idx = 0

        for i, p in enumerate(path_points):
            d = math.hypot(p[0] - rx, p[1] - ry)
            if d < min_dist:
                min_dist = d
                min_idx = i

        return min_idx

    def find_index_by_distance(self, path_points, start_idx, target_dist):
        total = 0.0

        for i in range(start_idx, len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            total += self.distance(p1, p2)

            if total >= target_dist:
                return i + 1

        return None

    def check_turn_alert(self):
        if self.latest_path is None:
            return

        if len(self.latest_path.poses) < 5:
            return

        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return

        rx, ry, yaw = robot_pose

        path_points = [
            (
                pose.pose.position.x,
                pose.pose.position.y
            )
            for pose in self.latest_path.poses
        ]

        nearest_idx = self.find_nearest_index(path_points, rx, ry)

        idx_1m = self.find_index_by_distance(
            path_points,
            nearest_idx,
            self.turn_lookahead_m
        )

        if idx_1m is None:
            return

        idx_after = self.find_index_by_distance(
            path_points,
            idx_1m,
            self.turn_after_m
        )

        if idx_after is None:
            return

        p_now = path_points[nearest_idx]
        p_1m = path_points[idx_1m]
        p_after = path_points[idx_after]

        heading_before = math.atan2(
            p_1m[1] - p_now[1],
            p_1m[0] - p_now[0]
        )

        heading_after = math.atan2(
            p_after[1] - p_1m[1],
            p_after[0] - p_1m[0]
        )

        turn_angle = normalize_angle(heading_after - heading_before)

        if abs(turn_angle) < self.turn_angle_threshold:
            self.last_turn_alert = ''
            return

        if turn_angle > 0:
            alert = 'Còn 1 mét nữa rẽ trái'
        else:
            alert = 'Còn 1 mét nữa rẽ phải'

        if alert != self.last_turn_alert:
            self.speak_alert(alert)
            self.last_turn_alert = alert

    def scan_callback(self, msg):
        front_ranges = []

        angle = msg.angle_min

        for r in msg.ranges:
            if -self.front_angle <= angle <= self.front_angle:
                if math.isfinite(r) and msg.range_min < r < msg.range_max:
                    front_ranges.append(r)

            angle += msg.angle_increment

        if not front_ranges:
            return

        min_front = min(front_ranges)

        if min_front <= self.obstacle_distance_m:
            self.speak_alert('Phía trước cách 1 mét có chướng ngại vật')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationAlertNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
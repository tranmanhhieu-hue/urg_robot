import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_srvs.srv import Empty


class AutoLocalizer(Node):

    def __init__(self):
        super().__init__('auto_localizer')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

        self.cli = self.create_client(Empty, '/reinitialize_global_localization')

        self.pose_history = []
        self.latest_covariance = None

        self.localized = False
        self.global_localization_called = False
        self.localization_start_time = None
        self.stable_start_time = None

        self.rotate_speed = 0.2
        self.max_localizing_time = 45.0
        self.min_rotation_wait = 3.0
        self.required_stable_time = 3.0

        self.timer = self.create_timer(0.5, self.run)

        self.get_logger().info('Auto Localizer started...')

    def pose_callback(self, msg):
        pose = msg.pose.pose
        cov = msg.pose.covariance

        x = pose.position.x
        y = pose.position.y

        q = pose.orientation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        self.pose_history.append((x, y, yaw))
        if len(self.pose_history) > 15:
            self.pose_history.pop(0)

        self.latest_covariance = cov

    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def call_global_localization(self):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Service /reinitialize_global_localization not available')
            return False

        req = Empty.Request()
        self.cli.call_async(req)
        self.get_logger().info('Called global localization')
        return True

    def rotate(self, speed=None):
        twist = Twist()
        twist.angular.z = self.rotate_speed if speed is None else speed
        self.cmd_pub.publish(twist)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def is_pose_stable(self):
        if len(self.pose_history) < 10:
            return False

        xs = [p[0] for p in self.pose_history]
        ys = [p[1] for p in self.pose_history]

        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        stable_xy = dx < 0.08 and dy < 0.08

        cov_ok = False
        if self.latest_covariance is not None:
            cov_x = self.latest_covariance[0]
            cov_y = self.latest_covariance[7]
            cov_yaw = self.latest_covariance[35]
            cov_ok = (cov_x < 0.5 and cov_y < 0.5 and cov_yaw < 0.5)

        return stable_xy and cov_ok

    def run(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self.localized:
            self.stop()
            return

        if not self.global_localization_called:
            success = self.call_global_localization()
            if success:
                self.global_localization_called = True
                self.localization_start_time = now_sec
            return

        elapsed = now_sec - self.localization_start_time

        if elapsed < self.min_rotation_wait:
            self.rotate()
            self.get_logger().info('Rotating to collect scan data...')
            return

        if elapsed > self.max_localizing_time:
            self.stop()
            self.get_logger().warn('Localization timeout. Could not confidently determine pose.')
            return

        self.rotate()

        if self.is_pose_stable():
            if self.stable_start_time is None:
                self.stable_start_time = now_sec

            stable_elapsed = now_sec - self.stable_start_time
            if stable_elapsed >= self.required_stable_time:
                self.stop()
                self.localized = True
                self.get_logger().info('Localization DONE!')
        else:
            self.stable_start_time = None


def main(args=None):
    rclpy.init(args=args)
    node = AutoLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
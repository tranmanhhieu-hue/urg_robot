import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf_transformations import euler_from_quaternion


class PathCorridorNode(Node):
    def __init__(self):
        super().__init__('path_corridor_node')

        # ===== Parameters =====
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('left_path_topic', '/wheel_path_left')
        self.declare_parameter('right_path_topic', '/wheel_path_right')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('wheel_base', 0.36)      # khoảng cách 2 bánh
        self.declare_parameter('path_max_points', 1000)
        self.declare_parameter('min_point_distance', 0.01)

        self.odom_topic = self.get_parameter('odom_topic').value
        self.left_path_topic = self.get_parameter('left_path_topic').value
        self.right_path_topic = self.get_parameter('right_path_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.wheel_base = float(self.get_parameter('wheel_base').value)
        self.path_max_points = int(self.get_parameter('path_max_points').value)
        self.min_point_distance = float(self.get_parameter('min_point_distance').value)

        # ===== Publishers =====
        self.left_path_pub = self.create_publisher(Path, self.left_path_topic, 10)
        self.right_path_pub = self.create_publisher(Path, self.right_path_topic, 10)

        # ===== Subscriber =====
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            20
        )

        # ===== Path messages =====
        self.left_path = Path()
        self.left_path.header.frame_id = self.frame_id

        self.right_path = Path()
        self.right_path.header.frame_id = self.frame_id

        # lưu điểm trước đó để tránh publish quá dày
        self.last_left_x = None
        self.last_left_y = None
        self.last_right_x = None
        self.last_right_y = None

        self.get_logger().info('Path corridor node started.')
        self.get_logger().info(f'odom_topic         : {self.odom_topic}')
        self.get_logger().info(f'left_path_topic    : {self.left_path_topic}')
        self.get_logger().info(f'right_path_topic   : {self.right_path_topic}')
        self.get_logger().info(f'frame_id           : {self.frame_id}')
        self.get_logger().info(f'wheel_base         : {self.wheel_base}')
        self.get_logger().info(f'path_max_points    : {self.path_max_points}')
        self.get_logger().info(f'min_point_distance : {self.min_point_distance}')

    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        x = pose.position.x
        y = pose.position.y

        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        half_wheel_base = self.wheel_base / 2.0

        # ===== Tính vị trí line trái =====
        left_x = x - half_wheel_base * math.sin(yaw)
        left_y = y + half_wheel_base * math.cos(yaw)

        # ===== Tính vị trí line phải =====
        right_x = x + half_wheel_base * math.sin(yaw)
        right_y = y - half_wheel_base * math.cos(yaw)

        stamp = msg.header.stamp

        # thêm điểm trái nếu đủ xa điểm trước
        if self.should_add_point(
            left_x, left_y,
            self.last_left_x, self.last_left_y
        ):
            left_pose = PoseStamped()
            left_pose.header.stamp = stamp
            left_pose.header.frame_id = self.frame_id
            left_pose.pose.position.x = left_x
            left_pose.pose.position.y = left_y
            left_pose.pose.position.z = 0.0
            left_pose.pose.orientation.w = 1.0

            self.left_path.poses.append(left_pose)
            self.last_left_x = left_x
            self.last_left_y = left_y

        # thêm điểm phải nếu đủ xa điểm trước
        if self.should_add_point(
            right_x, right_y,
            self.last_right_x, self.last_right_y
        ):
            right_pose = PoseStamped()
            right_pose.header.stamp = stamp
            right_pose.header.frame_id = self.frame_id
            right_pose.pose.position.x = right_x
            right_pose.pose.position.y = right_y
            right_pose.pose.position.z = 0.0
            right_pose.pose.orientation.w = 1.0

            self.right_path.poses.append(right_pose)
            self.last_right_x = right_x
            self.last_right_y = right_y

        # giới hạn số điểm để không phình RAM
        if len(self.left_path.poses) > self.path_max_points:
            self.left_path.poses.pop(0)

        if len(self.right_path.poses) > self.path_max_points:
            self.right_path.poses.pop(0)

        # update header
        self.left_path.header.stamp = stamp
        self.left_path.header.frame_id = self.frame_id

        self.right_path.header.stamp = stamp
        self.right_path.header.frame_id = self.frame_id

        # publish
        self.left_path_pub.publish(self.left_path)
        self.right_path_pub.publish(self.right_path)

    def should_add_point(self, x, y, last_x, last_y):
        if last_x is None or last_y is None:
            return True

        dx = x - last_x
        dy = y - last_y
        dist = math.sqrt(dx * dx + dy * dy)
        return dist >= self.min_point_distance


def main(args=None):
    rclpy.init(args=args)
    node = PathCorridorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
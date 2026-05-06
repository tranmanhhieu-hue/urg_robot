#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Int32, Bool, UInt8


class HapticController(Node):
    def __init__(self):
        super().__init__('haptic_controller')

        # ===== Parameters =====
        self.declare_parameter('min_distance', 1.0)
        self.declare_parameter('max_distance', 1.4)
        self.declare_parameter('lost_person_stop', True)

        self.declare_parameter('vib_off', 0)
        self.declare_parameter('vib_light', 80)
        self.declare_parameter('vib_medium', 150)
        self.declare_parameter('vib_strong', 255)

        self.min_distance = float(self.get_parameter('min_distance').value)
        self.max_distance = float(self.get_parameter('max_distance').value)
        self.lost_person_stop = bool(self.get_parameter('lost_person_stop').value)

        self.vib_off = int(self.get_parameter('vib_off').value)
        self.vib_light = int(self.get_parameter('vib_light').value)
        self.vib_medium = int(self.get_parameter('vib_medium').value)
        self.vib_strong = int(self.get_parameter('vib_strong').value)

        # ===== State =====
        self.person_distance = None
        self.person_visible = False
        self.person_guidance = 0

        self.last_left = None
        self.last_center = None
        self.last_right = None

        # ===== Subscribers =====
        self.create_subscription(Float32, '/person_distance', self.distance_cb, 10)
        self.create_subscription(Bool, '/person_visible', self.visible_cb, 10)
        self.create_subscription(Int32, '/person_guidance', self.guidance_cb, 10)

        # ===== Publishers =====
        self.vib_left_pub = self.create_publisher(UInt8, '/vib_left', 10)
        self.vib_center_pub = self.create_publisher(UInt8, '/vib_center', 10)
        self.vib_right_pub = self.create_publisher(UInt8, '/vib_right', 10)

        # ===== Timer =====
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            'haptic_controller started | '
            f'min_distance={self.min_distance:.2f} | '
            f'max_distance={self.max_distance:.2f} | '
            f'light={self.vib_light} | medium={self.vib_medium} | strong={self.vib_strong}'
        )

    def distance_cb(self, msg: Float32):
        self.person_distance = msg.data

    def visible_cb(self, msg: Bool):
        self.person_visible = msg.data

    def guidance_cb(self, msg: Int32):
        self.person_guidance = msg.data

    def clamp_u8(self, value):
        value = int(value)
        if value < 0:
            return 0
        if value > 255:
            return 255
        return value

    def publish_vibration(self, left=0, center=0, right=0):
        left = self.clamp_u8(left)
        center = self.clamp_u8(center)
        right = self.clamp_u8(right)

        if (
            self.last_left == left and
            self.last_center == center and
            self.last_right == right
        ):
            return

        left_msg = UInt8()
        center_msg = UInt8()
        right_msg = UInt8()

        left_msg.data = left
        center_msg.data = center
        right_msg.data = right

        self.vib_left_pub.publish(left_msg)
        self.vib_center_pub.publish(center_msg)
        self.vib_right_pub.publish(right_msg)

        self.last_left = left
        self.last_center = center
        self.last_right = right

        self.get_logger().info(
            f'VIB -> L:{left} C:{center} R:{right} | '
            f'visible={self.person_visible} dist={self.person_distance} guide={self.person_guidance}'
        )

    def control_loop(self):
        # Không thấy người / chưa có distance
        if not self.person_visible or self.person_distance is None:
            if self.lost_person_stop:
                self.publish_vibration(self.vib_off, self.vib_off, self.vib_off)
            return

        # Quá gần: rung vừa phải cả 3 để báo đi chậm lại
        if self.person_distance < self.min_distance:
            self.publish_vibration(self.vib_medium, self.vib_medium, self.vib_medium)
            return

        # Quá xa: rung lần lượt trái -> giữa -> phải, mỗi motor 2 giây, mức medium
        if self.person_distance > self.max_distance:
            now = self.get_clock().now()
            elapsed = (now - self.far_last_switch_time).nanoseconds / 1e9

            if elapsed >= self.far_step_duration:
                self.far_step = (self.far_step + 1) % 3
                self.far_last_switch_time = now

            if self.far_step == 0:
                self.publish_vibration(self.vib_medium, self.vib_off, self.vib_off)

            elif self.far_step == 1:
                self.publish_vibration(self.vib_off, self.vib_medium, self.vib_off)

            else:
                self.publish_vibration(self.vib_off, self.vib_off, self.vib_medium)

            return

        # Trong khoảng an toàn: dẫn hướng
        # -1 = người lệch trái -> rung phải
        #  1 = người lệch phải -> rung trái
        #  0 = ở giữa          -> rung giữa vừa phải
        if self.person_guidance == -1:
            self.publish_vibration(self.vib_off, self.vib_off, self.vib_medium)

        elif self.person_guidance == 1:
            self.publish_vibration(self.vib_medium, self.vib_off, self.vib_off)

        else:
            self.publish_vibration(self.vib_off, self.vib_medium, self.vib_off)


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = HapticController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_vibration(0, 0, 0)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
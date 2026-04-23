#!/usr/bin/env python3
import time
import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

K_PULSE2V = 0.00013
K_PULSE2W = 0.0035

class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')

        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("wheel_separation", 0.345)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.L = float(self.get_parameter("wheel_separation").value)

        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)
        self.get_logger().info(f"Serial opened: {self.port} @ {self.baud}")

        self.sub_cmd = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_callback, 10
        )

        self.get_logger().info("Cmd_vel bridge is ready")

    def cmd_callback(self, msg: Twist):
        V = float(msg.linear.x)
        W = float(msg.angular.z)

        # động học vi sai
        v_left = V - (W * self.L * 0.5)
        v_right = V + (W * self.L * 0.5)

        # đổi m/s -> pulse, có dấu để thể hiện chiều
        pL = int(v_left / K_PULSE2V)
        pR = int(v_right / K_PULSE2V)

        # giới hạn
        pL = max(min(pL, 9999), -9999)
        pR = max(min(pR, 9999), -9999)

        # protocol: m,<pL>,<pR>
        cmd = f"m,{pL},{pR}\n"

        try:
            self.ser.write(cmd.encode("ascii", errors="ignore"))
            self.get_logger().info(
                f"[TX] {cmd.strip()} | V={V:.3f}, W={W:.3f}, "
                f"pL={pL}, pR={pR}"
            )
        except Exception as e:
            self.get_logger().warn(f"Serial write error: {e}")

    def destroy_node(self):
        try:
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import math
import time
import serial
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


def quaternion_from_yaw(yaw: float) -> Quaternion:
    qz = math.sin(yaw * 0.5)
    qw = math.cos(yaw * 0.5)
    return Quaternion(x=0.0, y=0.0, z=qz, w=qw)


class BridgeOdom(Node):
    def __init__(self):
        super().__init__('odom')

        # ===== Parameters =====
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("wheel_radius", 0.0425)      # bán kính bánh xe (m)
        self.declare_parameter("wheel_separation", 0.345)   # khoảng cách 2 bánh (m)
        self.declare_parameter("ticks_per_rev", 300.0)      # xung / vòng encoder
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("broadcast_tf", True)

        self.port = self.get_parameter("port").value
        self.baud = int(self.get_parameter("baud").value)
        self.R = float(self.get_parameter("wheel_radius").value)
        self.L = float(self.get_parameter("wheel_separation").value)
        self.ticks_per_rev = float(self.get_parameter("ticks_per_rev").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.broadcast_tf = bool(self.get_parameter("broadcast_tf").value)

        hz = float(self.get_parameter("publish_hz").value)
        self.dt = 1.0 / max(1.0, hz)

        # ===== Serial =====
        self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
        time.sleep(2.0)  # đợi Arduino reset
        self.get_logger().info(f"Serial opened: {self.port} @ {self.baud}")

        # ===== Odom state =====
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_l = None
        self.prev_r = None
        self.prev_time = self.get_clock().now()

        # ===== ROS I/O =====
        self.odom_pub = self.create_publisher(Odometry, "odom", 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(self.dt, self.loop)

        self.get_logger().info("Odom is ready")

    def parse_encoder_line(self, line: str):
        line = line.strip()

        # Format 1: E,<L>,<R>
        if line.startswith("E,"):
            try:
                _, l, r = line.split(",")
                return int(l), int(r)
            except Exception:
                return None

        # Format 2: <L>,<R>
        if "," in line and "ticksL=" not in line:
            try:
                l, r = line.split(",")
                return int(l), int(r)
            except Exception:
                return None

        # Format 3: ticksL=... ticksR=... | ...
        if "ticksL=" in line and "ticksR=" in line:
            try:
                tmp = line.split("|")[0].strip()
                tmp = tmp.replace("ticksL=", "").replace("ticksR=", "")
                parts = tmp.split()
                return int(parts[0]), int(parts[1])
            except Exception:
                return None

        return None

    def loop(self):
        latest = None

        try:
            while True:
                raw = self.ser.readline().decode("ascii", errors="ignore").strip()
                if not raw:
                    break

                parsed = self.parse_encoder_line(raw)
                if parsed is not None:
                    latest = parsed

        except Exception as e:
            self.get_logger().warn(f"Serial read error: {e}")
            return

        if latest is None:
            return

        l_now, r_now = latest

        # Lần đầu chỉ lưu mốc
        if self.prev_l is None:
            self.prev_l = l_now
            self.prev_r = r_now
            self.prev_time = self.get_clock().now()
            return

        now_time = self.get_clock().now()
        dt = (now_time - self.prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = self.dt
        self.prev_time = now_time

        dl_ticks = l_now - self.prev_l
        dr_ticks = r_now - self.prev_r
        self.prev_l = l_now
        self.prev_r = r_now

        meters_per_tick = (2.0 * math.pi * self.R) / self.ticks_per_rev
        d_left = dl_ticks * meters_per_tick
        d_right = dr_ticks * meters_per_tick

        d_s = 0.5 * (d_left + d_right)
        d_theta = (d_right - d_left) / self.L

        self.x += d_s * math.cos(self.theta + 0.5 * d_theta)
        self.y += d_s * math.sin(self.theta + 0.5 * d_theta)
        self.theta = (self.theta + d_theta + math.pi) % (2.0 * math.pi) - math.pi

        vx = d_s / dt
        wz = d_theta / dt

        odom = Odometry()
        odom.header.stamp = now_time.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = quaternion_from_yaw(self.theta)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = 0.0
        odom.twist.twist.angular.z = wz

        self.odom_pub.publish(odom)

        if self.broadcast_tf:
            t = TransformStamped()
            t.header.stamp = now_time.to_msg()
            t.header.frame_id = self.frame_id
            t.child_frame_id = self.child_frame_id
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation = quaternion_from_yaw(self.theta)
            self.tf_broadcaster.sendTransform(t)

    def destroy_node(self):
        try:
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BridgeOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3

import time
import serial
import serial.serialutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8


class ESP32Bridge(Node):
    def __init__(self):
        super().__init__('esp32_bridge')

        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('left_topic', '/vib_left')
        self.declare_parameter('center_topic', '/vib_center')
        self.declare_parameter('right_topic', '/vib_right')
        self.declare_parameter('send_period', 0.10)
        self.declare_parameter('only_send_on_change', False)

        self.port = str(self.get_parameter('port').value)
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.left_topic = str(self.get_parameter('left_topic').value)
        self.center_topic = str(self.get_parameter('center_topic').value)
        self.right_topic = str(self.get_parameter('right_topic').value)
        self.send_period = float(self.get_parameter('send_period').value)
        self.only_send_on_change = bool(self.get_parameter('only_send_on_change').value)

        self.left_state = 0
        self.center_state = 0
        self.right_state = 0

        self.last_sent_cmd = None
        self.serial_ok = False
        self.ser = None
        self.last_reconnect_time = 0.0
        self.reconnect_interval = 1.0

        self.create_subscription(UInt8, self.left_topic, self.left_callback, 10)
        self.create_subscription(UInt8, self.center_topic, self.center_callback, 10)
        self.create_subscription(UInt8, self.right_topic, self.right_callback, 10)

        self.timer = self.create_timer(self.send_period, self.timer_callback)

        self.connect_serial()
        self.get_logger().info('ESP32 bridge started.')

    def clamp_u8(self, value):
        value = int(value)
        if value < 0:
            return 0
        if value > 255:
            return 255
        return value

    def close_serial(self):
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        self.ser = None
        self.serial_ok = False

    def connect_serial(self):
        now = time.time()
        if now - self.last_reconnect_time < self.reconnect_interval:
            return

        self.last_reconnect_time = now
        self.close_serial()

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                write_timeout=1.0
            )

            time.sleep(1.5)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            self.serial_ok = True
            self.get_logger().info(f'Connected to ESP32 on {self.port} [{self.baudrate}]')

        except serial.serialutil.SerialException as e:
            self.serial_ok = False
            self.ser = None
            self.get_logger().error(f'Cannot open serial port {self.port}: {e}')

    def left_callback(self, msg: UInt8):
        self.left_state = self.clamp_u8(msg.data)

    def center_callback(self, msg: UInt8):
        self.center_state = self.clamp_u8(msg.data)

    def right_callback(self, msg: UInt8):
        self.right_state = self.clamp_u8(msg.data)

    def build_command(self):
        l = self.clamp_u8(self.left_state)
        c = self.clamp_u8(self.center_state)
        r = self.clamp_u8(self.right_state)
        return f'VIB,{l},{c},{r}\n'

    def read_feedback(self):
        if self.ser is None or not self.ser.is_open:
            return

        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.get_logger().info(f'ESP32 says: {line}')
        except Exception:
            pass

    def timer_callback(self):
        cmd = self.build_command()

        if self.only_send_on_change and cmd == self.last_sent_cmd:
            self.read_feedback()
            return

        if not self.serial_ok or self.ser is None or not self.ser.is_open:
            self.connect_serial()
            return

        try:
            self.ser.write(cmd.encode('utf-8'))
            self.ser.flush()
            self.last_sent_cmd = cmd
            self.get_logger().info(f'Sent: {cmd.strip()}')
            self.read_feedback()

        except serial.serialutil.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')
            self.close_serial()

        except Exception as e:
            self.get_logger().error(f'Unexpected serial error: {e}')
            self.close_serial()

    def destroy_node(self):
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.write(b'VIB,0,0,0\n')
                self.ser.flush()
        except Exception:
            pass

        self.close_serial()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ESP32Bridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
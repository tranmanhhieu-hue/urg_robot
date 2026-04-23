#!/usr/bin/env python3
import os
import time
import math
import cv2
import rclpy
import numpy as np
import face_recognition

from rclpy.node import Node
from std_msgs.msg import Float32, Bool, Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


class PersonDistanceEstimator(Node):
    def __init__(self):
        super().__init__('robot_camera')

        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()

        # ===== Parameters =====
        self.declare_parameter('camera_device', '/dev/video0')
        self.declare_parameter('publish_rate', 5.0)

        # distance_cm = (real_face_width_cm * focal_length_px) / face_width_px
        self.declare_parameter('real_face_width_cm', 14.0)
        self.declare_parameter('focal_length_px', 557.14)

        # EMA filter
        self.declare_parameter('ema_alpha', 0.6)

        # Nếu không thấy đúng chủ robot trong khoảng này thì coi như mất người
        self.declare_parameter('lost_timeout_sec', 3.0)

        # Camera capture size
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)

        # Resize frame trước khi nhận diện cho nhẹ máy
        self.declare_parameter('processing_scale', 0.5)

        # Thư mục chứa ảnh mẫu của chủ robot
        self.declare_parameter(
            'owner_data_dir',
            '/home/ubuntu/urg_robot/src/robot_camera/owner_data'
        )

        # Ngưỡng match face_recognition, nhỏ hơn thì chặt hơn
        self.declare_parameter('match_threshold', 0.45)

        # Hiện cửa sổ debug khi test trên laptop
        self.declare_parameter('show_debug_window', True)

        # Độ rộng cho phép quanh tâm ảnh, nằm trong vùng này thì coi là đi giữa
        self.declare_parameter('center_tolerance_px', 100.0)

        # ===== Path guide parameters =====
        self.declare_parameter('path_guide_enabled', True)
        self.declare_parameter('path_guide_half_width_px', 200.0)
        self.declare_parameter('path_guide_length_px', 350)
        self.declare_parameter('path_guide_origin_y_offset_px', 20)
        self.declare_parameter('path_curve_gain', 2.5)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('path_max_curve_strength', 1.2)
        self.declare_parameter('path_perspective_shrink', 0.35)

        camera_device = self.get_parameter('camera_device').value
        publish_rate = float(self.get_parameter('publish_rate').value)
        self.real_face_width_cm = float(self.get_parameter('real_face_width_cm').value)
        self.focal_length_px = float(self.get_parameter('focal_length_px').value)
        self.ema_alpha = float(self.get_parameter('ema_alpha').value)
        self.lost_timeout_sec = float(self.get_parameter('lost_timeout_sec').value)
        frame_width = int(self.get_parameter('frame_width').value)
        frame_height = int(self.get_parameter('frame_height').value)
        self.processing_scale = float(self.get_parameter('processing_scale').value)
        self.owner_data_dir = str(self.get_parameter('owner_data_dir').value)
        self.match_threshold = float(self.get_parameter('match_threshold').value)
        self.show_debug_window = bool(self.get_parameter('show_debug_window').value)
        self.center_tolerance_px = float(self.get_parameter('center_tolerance_px').value)

        self.path_guide_enabled = bool(self.get_parameter('path_guide_enabled').value)
        self.path_guide_half_width_px = float(self.get_parameter('path_guide_half_width_px').value)
        self.path_guide_length_px = int(self.get_parameter('path_guide_length_px').value)
        self.path_guide_origin_y_offset_px = int(self.get_parameter('path_guide_origin_y_offset_px').value)
        self.path_curve_gain = float(self.get_parameter('path_curve_gain').value)
        self.angular_deadband = float(self.get_parameter('angular_deadband').value)
        self.path_max_curve_strength = float(self.get_parameter('path_max_curve_strength').value)
        self.path_perspective_shrink = float(self.get_parameter('path_perspective_shrink').value)

        if self.processing_scale <= 0.0 or self.processing_scale > 1.0:
            self.get_logger().warn(
                f'processing_scale={self.processing_scale} không hợp lệ, dùng 0.5'
            )
            self.processing_scale = 0.5

        # ===== Publishers =====
        self.distance_pub = self.create_publisher(Float32, '/person_distance', 10)
        self.visible_pub = self.create_publisher(Bool, '/person_visible', 10)
        self.face_width_pub = self.create_publisher(Float32, '/person_face_width_px', 10)
        self.offset_x_pub = self.create_publisher(Float32, '/person_offset_x', 10)
        self.guidance_pub = self.create_publisher(Int32, '/person_guidance', 10)

        # ===== Subscribers =====
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # ===== Camera =====
        if isinstance(camera_device, str) and camera_device.isdigit():
            camera_device = int(camera_device)

        self.cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            self.get_logger().error(f'Cannot open camera: {camera_device}')
            raise RuntimeError(f'Cannot open camera: {camera_device}')

        # ===== Load owner face encodings =====
        self.owner_encodings = self.load_owner_encodings(self.owner_data_dir)
        if len(self.owner_encodings) == 0:
            self.get_logger().error(
                'No valid owner face encodings found. '
                'Please set owner_data_dir to a folder containing owner images.'
            )
            raise RuntimeError('No valid owner face encodings found.')

        # ===== State =====
        self.filtered_distance_cm = None
        self.last_seen_time = 0.0

        self.frame_idx = 0
        self.detect_every_n_frames = 2

        self.last_best_box = None          # lưu box theo face_locations format: top,right,bottom,left
        self.last_best_score = None
        self.last_face_width_px = None
        self.last_offset_x = 0.0
        self.last_guidance = 0
        self.last_tracking_mode = 'INIT'

        # odom state cho line cong
        self.current_linear_x = 0.0
        self.current_angular_z = 0.0

        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'person_distance_estimator started | '
            f'loaded_owner_faces={len(self.owner_encodings)} | '
            f'processing_scale={self.processing_scale} | '
            f'path_guides={self.path_guide_enabled}'
        )

    def odom_callback(self, msg):
        self.current_linear_x = float(msg.twist.twist.linear.x)
        ang = float(msg.twist.twist.angular.z)
        if abs(ang) < self.angular_deadband:
            ang = 0.0
        self.current_angular_z = ang

    def load_owner_encodings(self, owner_data_dir):
        encodings = []

        if not owner_data_dir:
            self.get_logger().error('Parameter owner_data_dir is empty.')
            return encodings

        if not os.path.isdir(owner_data_dir):
            self.get_logger().error(f'owner_data_dir does not exist: {owner_data_dir}')
            return encodings

        valid_exts = {'.jpg', '.jpeg', '.png'}

        for filename in sorted(os.listdir(owner_data_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_exts:
                continue

            path = os.path.join(owner_data_dir, filename)
            try:
                image = face_recognition.load_image_file(path)
                image_encodings = face_recognition.face_encodings(image)

                if len(image_encodings) == 0:
                    self.get_logger().warn(f'No face found in owner image: {path}')
                    continue

                if len(image_encodings) > 1:
                    self.get_logger().warn(
                        f'More than one face found in owner image, using first face: {path}'
                    )

                encodings.append(image_encodings[0])
                self.get_logger().info(f'Loaded owner face: {path}')

            except Exception as e:
                self.get_logger().warn(f'Failed to load {path}: {e}')

        return encodings

    def estimate_distance_cm(self, face_width_px: float):
        if face_width_px <= 1.0:
            return None
        return (self.real_face_width_cm * self.focal_length_px) / face_width_px

    def publish_not_visible(self):
        visible_msg = Bool()
        visible_msg.data = False
        self.visible_pub.publish(visible_msg)

    def reset_tracking(self):
        self.last_best_box = None
        self.last_best_score = None
        self.last_face_width_px = None
        self.last_offset_x = 0.0
        self.last_guidance = 0
        self.last_tracking_mode = 'LOST'

    def build_curved_guide_points(self, img_w, img_h):
        origin_x = img_w // 2
        origin_y = img_h - self.path_guide_origin_y_offset_px

        lane_half_width_px = self.path_guide_half_width_px
        lookahead_px = max(40, self.path_guide_length_px)

        yaw_rate = float(self.current_angular_z)

        left_pts = []
        right_pts = []
        center_pts = []

        curve_strength = yaw_rate * self.path_curve_gain
        curve_strength = max(-self.path_max_curve_strength, min(self.path_max_curve_strength, curve_strength))

        for i in range(0, lookahead_px + 1, 8):
            y = origin_y - i
            if y < 0:
                break

            t = i / float(lookahead_px)

            perspective = 1.0 - t * self.path_perspective_shrink
            perspective = max(0.2, perspective)

            dx = lane_half_width_px * perspective

            bend = curve_strength * (t ** 2) * 180.0
            center_x = origin_x + int(bend)

            left_pts.append([int(center_x - dx), int(y)])
            right_pts.append([int(center_x + dx), int(y)])
            center_pts.append([int(center_x), int(y)])

        return (
            np.array(left_pts, dtype=np.int32),
            np.array(right_pts, dtype=np.int32),
            np.array(center_pts, dtype=np.int32)
        )

    def draw_path_guides(self, display):
        if not self.path_guide_enabled:
            return

        h, w = display.shape[:2]
        left_pts, right_pts, center_pts = self.build_curved_guide_points(w, h)

        if len(left_pts) > 1:
            cv2.polylines(display, [left_pts], False, (0, 255, 255), 2)
        if len(right_pts) > 1:
            cv2.polylines(display, [right_pts], False, (0, 255, 255), 2)
        if len(center_pts) > 1:
            cv2.polylines(display, [center_pts], False, (255, 255, 0), 1)

    def show_debug(self, frame, text_lines=None, box=None):
        if not self.show_debug_window:
            return

        display = frame.copy()

        h, w = display.shape[:2]
        img_center = w // 2

        self.draw_path_guides(display)

        cv2.line(display, (img_center, 0), (img_center, h), (0, 255, 255), 1)

        if box is not None:
            left, top, right, bottom = box
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        if text_lines is not None:
            y = 30
            for line in text_lines:
                is_red = ('NOT FOUND' in line) or ('FAILED' in line) or ('INVALID' in line) or ('LOST' in line)
                cv2.putText(
                    display,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if is_red else (0, 255, 0),
                    2
                )
                y += 30

        cv2.putText(
            display,
            f'v={self.current_linear_x:.3f} m/s  w={self.current_angular_z:.3f} rad/s',
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow('person_distance_estimator', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info('Q pressed, shutting down node.')
            rclpy.shutdown()

    def publish_tracking_outputs(self, face_width_px, offset_x, guidance, visible=True):
        visible_msg = Bool()
        visible_msg.data = bool(visible)
        self.visible_pub.publish(visible_msg)

        if not visible:
            return

        distance_cm = self.estimate_distance_cm(face_width_px)
        if distance_cm is None:
            self.publish_not_visible()
            return

        if self.filtered_distance_cm is None:
            self.filtered_distance_cm = distance_cm
        else:
            self.filtered_distance_cm = (
                self.ema_alpha * distance_cm
                + (1.0 - self.ema_alpha) * self.filtered_distance_cm
            )

        dist_msg = Float32()
        dist_msg.data = float(self.filtered_distance_cm) / 100.0
        self.distance_pub.publish(dist_msg)

        face_msg = Float32()
        face_msg.data = float(face_width_px)
        self.face_width_pub.publish(face_msg)

        offset_msg = Float32()
        offset_msg.data = float(offset_x)
        self.offset_x_pub.publish(offset_msg)

        guidance_msg = Int32()
        guidance_msg.data = int(guidance)
        self.guidance_pub.publish(guidance_msg)

    def compute_guidance(self, offset_x):
        if offset_x < -self.center_tolerance_px:
            return -1
        elif offset_x > self.center_tolerance_px:
            return 1
        else:
            return 0

    def scale_box_to_full_frame(self, box):
        top, right, bottom, left = box
        scale_back = 1.0 / self.processing_scale
        top = int(top * scale_back)
        right = int(right * scale_back)
        bottom = int(bottom * scale_back)
        left = int(left * scale_back)
        return top, right, bottom, left

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to read frame from camera.')
            self.publish_not_visible()
            self.reset_tracking()
            self.show_debug(
                frame if frame is not None else self.blank_frame(),
                ['CAMERA READ FAILED']
            )
            return

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(img_msg)

        self.frame_idx += 1
        run_detection = (self.frame_idx % self.detect_every_n_frames == 0)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.processing_scale != 1.0:
            small_rgb = cv2.resize(
                rgb_frame,
                (0, 0),
                fx=self.processing_scale,
                fy=self.processing_scale
            )
        else:
            small_rgb = rgb_frame

        now = time.time()
        best_box = None
        best_distance_score = None
        tracking_mode = 'LOST'

        # ===== Chạy detect định kỳ hoặc khi chưa có box cũ =====
        if run_detection or self.last_best_box is None:
            face_locations = face_recognition.face_locations(small_rgb, model='hog')

            if len(face_locations) > 0:
                face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

                for i, face_encoding in enumerate(face_encodings):
                    distances = face_recognition.face_distance(self.owner_encodings, face_encoding)
                    if len(distances) == 0:
                        continue

                    min_dist = float(distances.min())

                    if min_dist <= self.match_threshold:
                        if best_distance_score is None or min_dist < best_distance_score:
                            best_distance_score = min_dist
                            best_box = face_locations[i]

            if best_box is not None:
                self.last_best_box = best_box
                self.last_best_score = best_distance_score
                self.last_seen_time = now
                tracking_mode = 'DETECTED'
            else:
                if self.last_best_box is not None and (now - self.last_seen_time) <= self.lost_timeout_sec:
                    best_box = self.last_best_box
                    best_distance_score = self.last_best_score
                    tracking_mode = 'HOLD'
                else:
                    self.publish_not_visible()
                    self.reset_tracking()
                    self.show_debug(frame, ['OWNER LOST'])
                    return
        else:
            if self.last_best_box is not None and (now - self.last_seen_time) <= self.lost_timeout_sec:
                best_box = self.last_best_box
                best_distance_score = self.last_best_score
                tracking_mode = 'HOLD'
            else:
                self.publish_not_visible()
                self.reset_tracking()
                self.show_debug(frame, ['OWNER LOST'])
                return

        if best_box is None:
            self.publish_not_visible()
            self.reset_tracking()
            self.show_debug(frame, ['OWNER LOST'])
            return

        top, right, bottom, left = self.scale_box_to_full_frame(best_box)

        face_width_px = float(right - left)
        if face_width_px <= 1.0:
            if self.last_face_width_px is not None and tracking_mode == 'HOLD':
                face_width_px = self.last_face_width_px
            else:
                self.publish_not_visible()
                self.reset_tracking()
                self.show_debug(frame, ['INVALID FACE WIDTH'])
                return

        self.last_face_width_px = face_width_px

        face_center_x = (left + right) / 2.0
        image_center_x = frame.shape[1] / 2.0
        offset_x = face_center_x - image_center_x
        guidance = self.compute_guidance(offset_x)

        self.last_offset_x = offset_x
        self.last_guidance = guidance
        self.last_tracking_mode = tracking_mode

        self.publish_tracking_outputs(face_width_px, offset_x, guidance, visible=True)

        score_text = (
            f'score={best_distance_score:.3f}'
            if best_distance_score is not None else
            'score=HOLD'
        )

        self.show_debug(
            frame,
            [
                f'{tracking_mode}',
                'OWNER matched',
                score_text,
                f'face_w_px={face_width_px:.1f}',
                f'dist={self.filtered_distance_cm / 100.0:.2f} m' if self.filtered_distance_cm is not None else 'dist=N/A',
                f'offset_x={offset_x:.1f}',
                f'guidance={guidance}'
            ],
            box=(left, top, right, bottom)
        )

    def blank_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def destroy_node(self):
        try:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PersonDistanceEstimator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
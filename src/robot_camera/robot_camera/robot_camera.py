#!/usr/bin/env python3
import os
import time
import threading
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
        self.declare_parameter('publish_rate', 30.0)          # tăng lên 30 Hz cho smooth
        self.declare_parameter('real_face_width_cm', 14.0)
        self.declare_parameter('focal_length_px', 751.0)
        self.declare_parameter('ema_alpha', 0.6)
        self.declare_parameter('lost_timeout_sec', 3.0)
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('processing_scale', 0.5)
        self.declare_parameter('owner_data_dir',
            '/home/ubuntu/urg_robot/src/robot_camera/owner_data')
        self.declare_parameter('match_threshold', 0.45)
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('center_tolerance_px', 100.0)
        self.declare_parameter('path_guide_enabled', True)
        self.declare_parameter('path_guide_half_width_px', 200.0)
        self.declare_parameter('path_guide_length_px', 350)
        self.declare_parameter('path_guide_origin_y_offset_px', 20)
        self.declare_parameter('path_curve_gain', 2.5)
        self.declare_parameter('angular_deadband', 0.03)
        self.declare_parameter('path_max_curve_strength', 1.2)
        self.declare_parameter('path_perspective_shrink', 0.35)

        camera_device             = self.get_parameter('camera_device').value
        publish_rate              = float(self.get_parameter('publish_rate').value)
        self.real_face_width_cm   = float(self.get_parameter('real_face_width_cm').value)
        self.focal_length_px      = float(self.get_parameter('focal_length_px').value)
        self.ema_alpha            = float(self.get_parameter('ema_alpha').value)
        self.lost_timeout_sec     = float(self.get_parameter('lost_timeout_sec').value)
        frame_width               = int(self.get_parameter('frame_width').value)
        frame_height              = int(self.get_parameter('frame_height').value)
        self.processing_scale     = float(self.get_parameter('processing_scale').value)
        self.owner_data_dir       = str(self.get_parameter('owner_data_dir').value)
        self.match_threshold      = float(self.get_parameter('match_threshold').value)
        self.show_debug_window    = bool(self.get_parameter('show_debug_window').value)
        self.center_tolerance_px  = float(self.get_parameter('center_tolerance_px').value)
        self.path_guide_enabled   = bool(self.get_parameter('path_guide_enabled').value)
        self.path_guide_half_width_px     = float(self.get_parameter('path_guide_half_width_px').value)
        self.path_guide_length_px         = int(self.get_parameter('path_guide_length_px').value)
        self.path_guide_origin_y_offset_px = int(self.get_parameter('path_guide_origin_y_offset_px').value)
        self.path_curve_gain              = float(self.get_parameter('path_curve_gain').value)
        self.angular_deadband             = float(self.get_parameter('angular_deadband').value)
        self.path_max_curve_strength      = float(self.get_parameter('path_max_curve_strength').value)
        self.path_perspective_shrink      = float(self.get_parameter('path_perspective_shrink').value)

        if self.processing_scale <= 0.0 or self.processing_scale > 1.0:
            self.get_logger().warn('processing_scale không hợp lệ, dùng 0.5')
            self.processing_scale = 0.5

        # ===== Publishers =====
        self.distance_pub  = self.create_publisher(Float32, '/person_distance', 10)
        self.visible_pub   = self.create_publisher(Bool,    '/person_visible', 10)
        self.face_width_pub = self.create_publisher(Float32, '/person_face_width_px', 10)
        self.offset_x_pub  = self.create_publisher(Float32, '/person_offset_x', 10)
        self.guidance_pub  = self.create_publisher(Int32,   '/person_guidance', 10)

        # ===== Subscriber =====
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

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

        # ===== Load owner encodings =====
        self.owner_encodings = self.load_owner_encodings(self.owner_data_dir)
        if not self.owner_encodings:
            raise RuntimeError('No valid owner face encodings found.')

        # ===== Shared state (protected by locks) =====

        # Camera thread → main thread
        self._frame_lock  = threading.Lock()
        self._latest_frame = None           # frame BGR mới nhất từ camera thread

        # Detection thread → main thread
        self._result_lock = threading.Lock()
        self._det_result = {                # kết quả detect mới nhất
            'box': None,                    # (top,right,bottom,left) full-res
            'score': None,
            'face_width_px': None,
            'offset_x': 0.0,
            'guidance': 0,
            'mode': 'INIT',
            'timestamp': 0.0,
        }

        # Detection thread ← camera thread: frame nhỏ để nhận diện
        self._detect_frame_lock  = threading.Lock()
        self._detect_frame_pending = None   # (small_rgb, full_rgb) chờ detect
        self._detect_event = threading.Event()  # báo có frame mới

        # Odom
        self.current_linear_x  = 0.0
        self.current_angular_z = 0.0

        # EMA distance
        self.filtered_distance_cm = None

        self._shutdown = False

        # ===== Khởi động threads =====
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name='camera_grab')
        self._camera_thread.start()

        self._detect_thread = threading.Thread(
            target=self._detection_loop, daemon=True, name='face_detect')
        self._detect_thread.start()

        # ROS timer chỉ publish + hiển thị
        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            f'Started | owner_faces={len(self.owner_encodings)} | '
            f'scale={self.processing_scale} | rate={publish_rate} Hz'
        )

    # ------------------------------------------------------------------
    # Thread 1: Liên tục grab frame, KHÔNG decode thừa
    # ------------------------------------------------------------------
    def _camera_loop(self):
        while not self._shutdown:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue

            # Luôn giữ frame mới nhất (drop frame cũ)
            with self._frame_lock:
                self._latest_frame = frame

            # Chuẩn bị frame cho detection thread
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.processing_scale != 1.0:
                small_rgb = cv2.resize(rgb, (0, 0),
                    fx=self.processing_scale, fy=self.processing_scale)
            else:
                small_rgb = rgb

            # Chỉ đưa vào queue nếu detection thread đang rảnh (drop nếu bận)
            if not self._detect_event.is_set():
                with self._detect_frame_lock:
                    self._detect_frame_pending = (small_rgb, frame.shape)
                self._detect_event.set()

    # ------------------------------------------------------------------
    # Thread 2: Nhận diện khuôn mặt (nặng, chạy riêng)
    # ------------------------------------------------------------------
    def _detection_loop(self):
        last_seen_time = 0.0
        last_box = None
        last_score = None

        while not self._shutdown:
            triggered = self._detect_event.wait(timeout=0.5)
            if not triggered or self._shutdown:
                continue
            self._detect_event.clear()

            with self._detect_frame_lock:
                pending = self._detect_frame_pending
                self._detect_frame_pending = None

            if pending is None:
                continue

            small_rgb, full_shape = pending
            full_h, full_w = full_shape[:2]
            now = time.time()

            # --- Face detection ---
            face_locations = face_recognition.face_locations(small_rgb, model='hog')
            found_box   = None
            found_score = None

            if face_locations:
                face_encodings = face_recognition.face_encodings(small_rgb, face_locations)
                for i, enc in enumerate(face_encodings):
                    distances = face_recognition.face_distance(self.owner_encodings, enc)
                    if len(distances) == 0:
                        continue
                    min_dist = float(distances.min())
                    if min_dist <= self.match_threshold:
                        if found_score is None or min_dist < found_score:
                            found_score = min_dist
                            found_box   = face_locations[i]

            if found_box is not None:
                # Scale box về full-res
                s = 1.0 / self.processing_scale
                top, right, bottom, left = found_box
                top    = int(top * s);    right  = int(right * s)
                bottom = int(bottom * s); left   = int(left * s)
                full_box = (top, right, bottom, left)

                face_width_px = float(right - left)
                face_center_x = (left + right) / 2.0
                offset_x  = face_center_x - full_w / 2.0
                guidance  = self._compute_guidance(offset_x)

                last_box        = full_box
                last_score      = found_score
                last_seen_time  = now
                mode            = 'DETECTED'

            else:
                # HOLD nếu chưa hết timeout
                if last_box is not None and (now - last_seen_time) <= self.lost_timeout_sec:
                    top, right, bottom, left = last_box
                    face_width_px = float(right - left)
                    face_center_x = (left + right) / 2.0
                    offset_x  = face_center_x - full_w / 2.0
                    guidance  = self._compute_guidance(offset_x)
                    found_score = last_score
                    mode        = 'HOLD'
                else:
                    last_box   = None
                    last_score = None
                    with self._result_lock:
                        self._det_result.update({
                            'box': None, 'score': None,
                            'face_width_px': None,
                            'offset_x': 0.0, 'guidance': 0,
                            'mode': 'LOST', 'timestamp': now,
                        })
                    continue

            with self._result_lock:
                self._det_result.update({
                    'box':          last_box,
                    'score':        found_score,
                    'face_width_px': face_width_px,
                    'offset_x':     offset_x,
                    'guidance':     guidance,
                    'mode':         mode,
                    'timestamp':    now,
                })

    # ------------------------------------------------------------------
    # ROS Timer: chỉ đọc kết quả + publish + hiển thị (nhẹ)
    # ------------------------------------------------------------------
    def timer_callback(self):
        # Lấy frame mới nhất
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return

        # Publish raw image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(img_msg)

        # Lấy kết quả detect
        with self._result_lock:
            result = dict(self._det_result)

        if result['box'] is None or result['mode'] == 'LOST':
            self._pub_not_visible()
            self.show_debug(frame, ['OWNER LOST'])
            return

        face_width_px = result['face_width_px']
        offset_x      = result['offset_x']
        guidance      = result['guidance']
        mode          = result['mode']
        score         = result['score']
        top, right, bottom, left = result['box']

        self._publish_tracking(face_width_px, offset_x, guidance)

        score_text = f'score={score:.3f}' if score is not None else 'score=HOLD'
        dist_text  = (f'dist={self.filtered_distance_cm / 100.0:.2f} m'
                      if self.filtered_distance_cm else 'dist=N/A')

        self.show_debug(
            frame,
            [mode, 'OWNER matched', score_text,
             f'face_w_px={face_width_px:.1f}', dist_text,
             f'offset_x={offset_x:.1f}', f'guidance={guidance}'],
            box=(left, top, right, bottom)
        )

    # ------------------------------------------------------------------
    # Helpers (giống code cũ, không đổi logic)
    # ------------------------------------------------------------------
    def odom_callback(self, msg):
        self.current_linear_x = float(msg.twist.twist.linear.x)
        ang = float(msg.twist.twist.angular.z)
        self.current_angular_z = 0.0 if abs(ang) < self.angular_deadband else ang

    def _compute_guidance(self, offset_x):
        if offset_x < -self.center_tolerance_px: return -1
        if offset_x >  self.center_tolerance_px: return  1
        return 0

    def _pub_not_visible(self):
        msg = Bool(); msg.data = False
        self.visible_pub.publish(msg)

    def _publish_tracking(self, face_width_px, offset_x, guidance):
        visible_msg = Bool(); visible_msg.data = True
        self.visible_pub.publish(visible_msg)

        if face_width_px <= 1.0:
            return

        dist_cm = (self.real_face_width_cm * self.focal_length_px) / face_width_px
        if self.filtered_distance_cm is None:
            self.filtered_distance_cm = dist_cm
        else:
            self.filtered_distance_cm = (self.ema_alpha * dist_cm
                + (1 - self.ema_alpha) * self.filtered_distance_cm)

        d = Float32(); d.data = float(self.filtered_distance_cm) / 100.0
        self.distance_pub.publish(d)

        fw = Float32(); fw.data = float(face_width_px)
        self.face_width_pub.publish(fw)

        ox = Float32(); ox.data = float(offset_x)
        self.offset_x_pub.publish(ox)

        gm = Int32(); gm.data = int(guidance)
        self.guidance_pub.publish(gm)

    def load_owner_encodings(self, owner_data_dir):
        encodings = []
        if not owner_data_dir or not os.path.isdir(owner_data_dir):
            self.get_logger().error(f'owner_data_dir không hợp lệ: {owner_data_dir}')
            return encodings
        for fn in sorted(os.listdir(owner_data_dir)):
            if os.path.splitext(fn)[1].lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            path = os.path.join(owner_data_dir, fn)
            try:
                img  = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if not encs:
                    self.get_logger().warn(f'Không tìm thấy mặt: {path}')
                    continue
                encodings.append(encs[0])
                self.get_logger().info(f'Loaded: {path}')
            except Exception as e:
                self.get_logger().warn(f'Load failed {path}: {e}')
        return encodings

    # Path guide (giữ nguyên từ code cũ)
    def build_curved_guide_points(self, img_w, img_h):
        origin_x = img_w // 2
        origin_y = img_h - self.path_guide_origin_y_offset_px
        lookahead_px = max(40, self.path_guide_length_px)
        curve_strength = self.current_angular_z * self.path_curve_gain
        curve_strength = max(-self.path_max_curve_strength,
                             min(self.path_max_curve_strength, curve_strength))
        left_pts, right_pts, center_pts = [], [], []
        for i in range(0, lookahead_px + 1, 8):
            y = origin_y - i
            if y < 0: break
            t = i / float(lookahead_px)
            perspective = max(0.2, 1.0 - t * self.path_perspective_shrink)
            dx = self.path_guide_half_width_px * perspective
            bend = curve_strength * (t ** 2) * 180.0
            cx = origin_x + int(bend)
            left_pts.append([int(cx - dx), int(y)])
            right_pts.append([int(cx + dx), int(y)])
            center_pts.append([int(cx), int(y)])
        return (np.array(left_pts,   dtype=np.int32),
                np.array(right_pts,  dtype=np.int32),
                np.array(center_pts, dtype=np.int32))

    def draw_path_guides(self, display):
        if not self.path_guide_enabled: return
        h, w = display.shape[:2]
        lp, rp, cp = self.build_curved_guide_points(w, h)
        if len(lp) > 1: cv2.polylines(display, [lp], False, (0, 255, 255), 2)
        if len(rp) > 1: cv2.polylines(display, [rp], False, (0, 255, 255), 2)
        if len(cp) > 1: cv2.polylines(display, [cp], False, (255, 255, 0), 1)

    def show_debug(self, frame, text_lines=None, box=None):
        if not self.show_debug_window: return
        display = frame.copy()
        h, w = display.shape[:2]
        self.draw_path_guides(display)
        cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
        if box is not None:
            l, t, r, b = box
            cv2.rectangle(display, (l, t), (r, b), (0, 255, 0), 2)
        y = 30
        for line in (text_lines or []):
            is_red = any(kw in line for kw in ('NOT FOUND','FAILED','INVALID','LOST'))
            cv2.putText(display, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255) if is_red else (0, 255, 0), 2)
            y += 30
        cv2.putText(display,
            f'v={self.current_linear_x:.3f} m/s  w={self.current_angular_z:.3f} rad/s',
            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('person_distance_estimator', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Q pressed, shutting down.')
            rclpy.shutdown()

    def destroy_node(self):
        self._shutdown = True
        self._detect_event.set()   # unblock detection thread
        try:
            self._camera_thread.join(timeout=1.0)
            self._detect_thread.join(timeout=1.0)
            if self.cap: self.cap.release()
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
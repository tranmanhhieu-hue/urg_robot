import os
import time
import atexit
import json
import re
from typing import Optional, List, Dict, Any

import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
import pygame

import face_recognition
import cv2
from pypdf import PdfReader
import numpy as np
from datetime import datetime
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ===== ROS2 =====
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ==============================
# ROS TOPICS 
# ==============================
CMD_TOPIC = "/robot_cmd"
STATUS_TOPIC = "/robot_status"

# ==============================
# CẤU HÌNH THƯ MỤC
# ==============================
BASE_DIR = "data"
FACE_DIR = os.path.join(BASE_DIR, "faces")
PROFILE_DIR = os.path.join(BASE_DIR, "profiles")
VOICE_DIR = os.path.join(BASE_DIR, "voices")
UNKNOWN_DIR = os.path.join(BASE_DIR, "unknown")

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)
os.makedirs(VOICE_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ===== HIỂN THỊ ẢNH VỪA CHỤP =====
SHOW_CAMERA_WINDOW = True

# ===== DỄ NHẬN HƠN (webcam) =====
FACE_TOLERANCE = 0.55  # tăng lên 0.60 nếu vẫn khó nhận (nhưng dễ nhầm hơn)

# ===== KHUNG TO HƠN ~ 1.5 LẦN =====
BOX_SCALE = 0.5          # mở rộng mỗi phía 50% => tổng thể ~1.5x
BOX_THICKNESS = 3
FONT_SCALE = 0.9
FONT_THICKNESS = 2

# ==============================
# OPENAI
# ==============================

client = OpenAI()

STOP_COMMANDS = ["tạm biệt", "dừng", "thoát", "bye"]
WAKE_WORDS = ["lumi", "lu mi", "lumi ơi", "ê lumi"]

# XÁC NHẬN
YES_WORDS = ["có", "đúng", "ok", "oke", "ừ", "yes", "chắc chắn", "đồng ý", "đi"]
NO_WORDS = ["không", "ko", "k", "thôi", "hủy", "huỷ", "no"]

# BẮT CÂU “đi/tới/đến vị trí A…”
GO_PATTERNS = [
    # ví dụ: "đưa tôi đến vị trí A", "dẫn tôi tới A", "cho tôi đến A"
    r"\b(đưa|dẫn|cho|chở)\s+(?:tôi|mình)?\s*(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
    # ví dụ: "đi tới vị trí A", "đến A"
    r"\b(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
    # ví dụ: "đến phòng kitchen"
    r"\b(đi|tới|đến)\s+phòng\s+([A-Za-z0-9_]+)\b",
]

robot_ear = sr.Recognizer()
pygame.mixer.init()

voice_files = []
voice_counter = 0


def cleanup_voices():
    for f in voice_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except:
            pass


atexit.register(cleanup_voices)

# ==============================
# LOAD FACE DATABASE
# ==============================
known_encodings = []
known_names = []


def load_faces():
    known_encodings.clear()
    known_names.clear()

    if not os.path.exists(FACE_DIR):
        return

    for file in os.listdir(FACE_DIR):
        if file.lower().endswith((".jpg", ".png")):
            path = os.path.join(FACE_DIR, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(file)[0])
            else:
                print("KHÔNG ENCODE ĐƯỢC:", file)


load_faces()

# ==============================
# ĐỌC PDF PROFILE
# ==============================
def read_profile(name):
    pdf_path = os.path.join(PROFILE_DIR, f"{name}.pdf")
    if not os.path.exists(pdf_path):
        return "Chưa có thông tin chi tiết."

    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        return " ".join(text_parts).strip() or "Chưa có thông tin chi tiết."
    except:
        return "Không thể đọc file thông tin."


# ==============================
# TIỆN ÍCH: PHÓNG KHUNG ~1.5 LẦN
# ==============================
def expand_box(top, right, bottom, left, img_h, img_w, scale=0.5):
    h = bottom - top
    w = right - left

    pad_h = int(h * scale)
    pad_w = int(w * scale)

    top2 = max(0, top - pad_h)
    bottom2 = min(img_h, bottom + pad_h)
    left2 = max(0, left - pad_w)
    right2 = min(img_w, right + pad_w)

    return top2, right2, bottom2, left2


# ==============================
# NHẬN DIỆN CAMERA (NHIỀU MẶT + THỨ TỰ 1,2,3)
# ==============================
def recognize_person(frame):

    if frame is None:
        print("Chưa có ảnh từ camera")
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_locations:
        if SHOW_CAMERA_WINDOW:
            display = frame.copy()
            cv2.putText(display, "Khong thay khuon mat", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Anh vua chup - Bam phim bat ky de dong", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return []

    items = list(zip(face_locations, face_encodings))
    items.sort(key=lambda x: (x[0][3], x[0][0]))

    names_found = []
    display = frame.copy()
    img_h, img_w = display.shape[0], display.shape[1]

    for order_idx, (loc, face_encoding) in enumerate(items, start=1):
        top, right, bottom, left = loc
        name = "Người lạ"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < FACE_TOLERANCE:
                name = known_names[best_idx]

        if name == "Người lạ":
            timestamp = int(time.time())
            filename = f"unknown_{timestamp}_{order_idx}.jpg"
            path = os.path.join(UNKNOWN_DIR, filename)
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(path, face_img)

        names_found.append(name)

        if SHOW_CAMERA_WINDOW:
            top2, right2, bottom2, left2 = expand_box(top, right, bottom, left, img_h, img_w, scale=BOX_SCALE)
            cv2.rectangle(display, (left2, top2), (right2, bottom2), (0, 255, 0), BOX_THICKNESS)

            label = f"{order_idx}. {name}"
            label_h = 32
            y1 = max(0, bottom2 - label_h)
            cv2.rectangle(display, (left2, y1), (right2, bottom2), (0, 255, 0), cv2.FILLED)

            cv2.putText(
                display, label, (left2 + 8, bottom2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS
            )

    if SHOW_CAMERA_WINDOW:
        cv2.imshow("Anh vua chup - Bam phim bat ky de dong", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return names_found


# ==============================
# VOICE FUNCTION
# ==============================
def speak(text):
    global voice_counter

    voice_counter += 1
    filename = os.path.join(VOICE_DIR, f"voice_{voice_counter}.mp3")

    try:
        tts = gTTS(text=text, lang="vi")
        tts.save(filename)
        voice_files.append(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception as e:
        print("Voice error:", e)


# ==============================
# ROS2 NODE: publish /robot_cmd, sub /robot_status
# ==============================
class VoiceRosBridge(Node):
    def __init__(self):
        super().__init__("voice_ros_bridge")
        self.cmd_pub = self.create_publisher(String, CMD_TOPIC, 10)
        self.status_sub = self.create_subscription(String, STATUS_TOPIC, self.on_status, 10)

        self.known_waypoints = []   # lấy từ commander khi event=ready
        self.busy = False           # dựa event busy/nav_start/nav_done...
        self.last_status = None

        self.bridge = CvBridge()
        self.latest_frame = None

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Convert lỗi: {e}')    

    def on_status(self, msg: String):
        self.last_status = msg.data
        try:
            data = json.loads(msg.data)
        except:
            return

        event = data.get("event", "")

        goal = data.get("goal")  # ✅ commander gửi kèm goal

        if event == "nav_succeeded":
            speak(f"Tôi đã đến vị trí {goal}." if goal else "Tôi đã đến nơi.")
            return
        if event == "nav_canceled":
            speak("Tôi đã hủy di chuyển.")
            return
        if event == "nav_aborted":
            speak("Tôi không thể đến đích.")
            return

        if event == "ready":
            wps = data.get("waypoints", [])
            if isinstance(wps, list):
                self.known_waypoints = wps

        if event in ("busy", "nav_start", "goal_accepted"):
            self.busy = True
        if event in ("nav_succeeded", "nav_canceled", "nav_aborted", "nav_done"):
            self.busy = False

    def send_go(self, goal_name: str):
        payload = {"type": "go", "goal": goal_name}
        m = String()
        m.data = json.dumps(payload, ensure_ascii=False)
        self.cmd_pub.publish(m)

    def send_cancel(self):
        payload = {"type": "cancel"}
        m = String()
        m.data = json.dumps(payload, ensure_ascii=False)
        self.cmd_pub.publish(m)


# ==============================
# NLP nhỏ: bắt ý định đi tới waypoint
# ==============================
def parse_go_intent(text: str) -> Optional[str]:
    t = text.strip()
    for pat in GO_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(m.lastindex)  # luôn lấy group cuối
    return None

def normalize_goal(goal: str) -> str:
    g = goal.lower().strip()

    mapping = {
        "a": "A",
        "b": "B", "bi": "B", "bê": "B",
        "c": "C", "xi": "C", "xê": "C",
        "d": "D", "đi": "D",
    }

    return mapping.get(g, goal.upper())

def is_yes(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in YES_WORDS)


def is_no(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in NO_WORDS)

def has_wake_word(text: str) -> bool:
    t = text.lower().strip()
    return any(w in t for w in WAKE_WORDS)


def remove_wake_word(text: str) -> str:
    cleaned = text
    for w in WAKE_WORDS:
        cleaned = re.sub(rf"\b{re.escape(w)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?:;")
    return cleaned

def get_vietnam_time_text() -> str:
    now = datetime.now()
    weekday_map = {
        0: "Thứ Hai",
        1: "Thứ Ba",
        2: "Thứ Tư",
        3: "Thứ Năm",
        4: "Thứ Sáu",
        5: "Thứ Bảy",
        6: "Chủ Nhật",
    }
    weekday = weekday_map[now.weekday()]
    return f"Hôm nay là {weekday}, ngày {now.day} tháng {now.month} năm {now.year}."
# ==============================
# MAIN LOOP
# ==============================
def main():
    # init ROS
    rclpy.init()
    bridge = VoiceRosBridge()

    pending_goal: Optional[str] = None  # chờ confirm

    # gợi ý cho user biết waypoint đang có (nếu commander đã chạy và publish ready)
    speak("Xin chào, tôi là Lumi, robot hỗ trợ người khiếm thị. Tôi có thể giúp gì cho bạn?")

    try:
        while True:
            # Cho ROS xử lý status giữa các vòng lặp 
            rclpy.spin_once(bridge, timeout_sec=0.1)

            with sr.Microphone() as mic:
                print("Robot: Tôi đang nghe...")
                try:
                    audio = robot_ear.listen(
                        mic,
                        timeout=0.8,
                        phrase_time_limit=6
                    )
                    you = robot_ear.recognize_google(
                        audio,
                        language="vi-VN"
                    )
                except sr.WaitTimeoutError:
                    # Không ai nói trong 1s
                    continue
                except sr.UnknownValueError:
                    # Không nhận diện được giọng nói
                    continue
                except Exception as e:
                    print("Voice error:", e)
                    continue

            print("You:", you)
            print("DEBUG raw:", repr(you))

            you_l_raw = you.lower()

            # STOP luôn được ưu tiên
            if any(cmd in you_l_raw for cmd in STOP_COMMANDS):
                speak("Vâng, tạm biệt bạn.")
                break           

            # Nếu đang chờ xác nhận thì không bắt buộc phải nói Lumi nữa
            if pending_goal is not None:
                you_l = you.lower()

                if is_yes(you):
                    if bridge.busy:
                        speak(f"Tôi đang bận, chưa thể đi đến {pending_goal}.")
                    else:
                        bridge.send_go(pending_goal)
                        speak(f"Ok. Tôi sẽ đi đến {pending_goal}.")
                    pending_goal = None
                    continue

                if is_no(you):
                    speak("Ok, tôi sẽ không di chuyển.")
                    pending_goal = None
                    continue

                speak(f"Bạn có chắc chắn muốn đi đến {pending_goal} không? Nói 'có' hoặc 'không'.")
                continue

            # Chỉ phản hồi khi có gọi tên Lumi
            if not has_wake_word(you):
                print("Bỏ qua vì không có wake word:", you)
                continue

            # Xóa từ gọi Lumi khỏi câu để phân tích lệnh
            you = remove_wake_word(you)
            you_l = you.lower()

            print("After wake-word cleanup:", you)

            goal_test = parse_go_intent(you)
            print("DEBUG goal:", goal_test)


            # ===== HỎI VỀ NGƯỜI =====
            if "đây là ai" in you_l:
                load_faces()
                if bridge.latest_frame is None:
                    speak("Tôi chưa nhận được hình ảnh từ camera.")
                    continue

                names = recognize_person(bridge.latest_frame.copy())

                if not names:
                    speak("Tôi không thấy ai trong khung hình.")
                    continue

                parts = []
                for i, name in enumerate(names, start=1):
                    if name == "Người lạ":
                        parts.append(f"{i} là người lạ. Tôi đã lưu ảnh lại.")
                    else:
                        parts.append(f"{i} là {name}. {read_profile(name)}")

                response = " ".join(parts)
                print("Robot:", response)
                speak(response)
                continue

            # ===== BẮT Ý ĐỊNH ĐI TỚI WAYPOINT =====
            goal = parse_go_intent(you)
            if goal:
                goal = normalize_goal(goal)
                # nếu commander đã báo danh sách waypoints, kiểm tra luôn để tránh sai
                if bridge.known_waypoints and goal not in bridge.known_waypoints:
                    speak(f"Tôi không thấy vị trí {goal}. Các vị trí có sẵn là: {', '.join(bridge.known_waypoints)}.")
                    continue

                pending_goal = goal
                speak(f"Bạn có chắc chắn muốn đi đến {goal} không?")
                continue

            # ===== LỆNH HỦY =====
            if "hủy" in you_l or "huỷ" in you_l or "cancel" in you_l:
                bridge.send_cancel()
                speak("Tôi đã gửi lệnh hủy.")
                continue

            # ===== HỎI NGÀY GIỜ =====
            if "hôm nay là thứ mấy" in you_l or "hôm nay thứ mấy" in you_l:
                response = get_vietnam_time_text()
                print("Robot:", response)
                speak(response)
                continue

            # ===== CHAT GPT (BÌNH THƯỜNG) =====
            if not client:
                # nếu không có api key
                speak("Tôi chưa được cấu hình OpenAI API key.")
                continue

            try:
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": 
                         "Bạn là Lumi, một robot hỗ trợ người khiếm thị. "
                         "Bạn nói chuyện thân thiện, rõ ràng, dễ hiểu. "
                         "Luôn xưng là 'Lumi'. "
                         "Ưu tiên mô tả bằng lời để hỗ trợ người không nhìn thấy. "
                         "Trả lời ngắn gọn, hữu ích. "
                         "Nếu người dùng muốn đi tới vị trí thì hướng dẫn họ nói 'đi tới vị trí A'."},
                        {"role": "user", "content": you},
                    ],
                    temperature=0.7,
                )
                robot_brain = completion.choices[0].message.content
            except Exception as e:
                print("OpenAI error:", e)
                robot_brain = "Tôi đang bận, vui lòng thử lại."

            print("Robot:", robot_brain)
            speak(robot_brain)

    finally:
        pygame.mixer.quit()
        cleanup_voices()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
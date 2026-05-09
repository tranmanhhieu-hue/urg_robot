    #!/usr/bin/env python3
    import os
    import time
    import atexit
    import json
    import re
    from typing import Optional
    from datetime import datetime

    import speech_recognition as sr
    from gtts import gTTS
    from openai import OpenAI
    import pygame

    # ===== ROS2 =====
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    # ==============================
    # ROS TOPICS
    # ==============================
    CMD_TOPIC = "/robot_cmd"
    STATUS_TOPIC = "/robot_status"
    ALERT_TOPIC = "/nav_alert_text"

    # ==============================
    # CẤU HÌNH THƯ MỤC
    # ==============================
    BASE_DIR = "data"
    VOICE_DIR = os.path.join(BASE_DIR, "voices")
    os.makedirs(VOICE_DIR, exist_ok=True)

    # ==============================
    # OPENAI
    # API key lấy tự động từ biến môi trường OPENAI_API_KEY
    # Nếu đã thêm key vào ~/.bashrc thì chỉ cần source ~/.bashrc hoặc mở terminal mới
    # ==============================
    client = OpenAI()

    # ==============================
    # TỪ KHÓA GIỌNG NÓI
    # ==============================
    STOP_COMMANDS = ["tạm biệt", "thoát", "bye"]
    WAKE_WORDS = ["lumi", "lu mi", "lumi ơi", "ê lumi"]

    YES_WORDS = ["có", "đúng", "ok", "oke", "ừ", "yes", "chắc chắn", "đồng ý", "đi"]
    NO_WORDS = ["không", "ko", "thôi", "hủy", "huỷ", "no"]

    GO_PATTERNS = [
        # Ví dụ: "đưa tôi đến vị trí A", "dẫn tôi tới A", "cho tôi đi đến B"
        r"\b(đưa|dẫn|cho|chở)\s+(?:tôi|mình)?\s*(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
        # Ví dụ: "đi tới vị trí A", "đến A"
        r"\b(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
        # Ví dụ: "đến phòng lab"
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
            except Exception:
                pass


    atexit.register(cleanup_voices)

    # ==============================
    # VOICE FUNCTION
    # ==============================
    def speak(text: str):
        global voice_counter

        print("Robot:", text)

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
    # ROS2 NODE
    # ==============================
    class VoiceRosBridge(Node):
        def __init__(self):
            super().__init__("voice_ros_bridge")
            self.cmd_pub = self.create_publisher(String, CMD_TOPIC, 10)
            self.status_sub = self.create_subscription(String, STATUS_TOPIC, self.on_status, 10)
            self.alert_sub = self.create_subscription(
                String,
                ALERT_TOPIC,
                self.on_nav_alert,
                10
            )    

            self.known_waypoints = []
            self.busy = False
            self.last_status = None

        def on_status(self, msg: String):
            self.last_status = msg.data

            try:
                data = json.loads(msg.data)
            except Exception:
                return

            event = data.get("event", "")
            goal = data.get("goal")

            if event == "nav_succeeded":
                speak(f"Tôi đã đến vị trí {goal}." if goal else "Tôi đã đến nơi.")
                self.busy = False
                return

            if event == "nav_canceled":
                speak("Tôi đã hủy di chuyển.")
                self.busy = False
                return

            if event == "nav_aborted":
                speak("Tôi không thể đến đích.")
                self.busy = False
                return

            if event == "ready":
                wps = data.get("waypoints", [])
                if isinstance(wps, list):
                    self.known_waypoints = wps

            if event in ("busy", "nav_start", "goal_accepted"):
                self.busy = True

            if event in ("nav_done",):
                self.busy = False

        def on_nav_alert(self, msg: String):
            text = msg.data.strip()

            if not text:
                return

            # Chỉ đọc cảnh báo khi robot đang di chuyển
            if self.busy:
                speak(text)                

        def send_go(self, goal_name: str):
            payload = {"type": "go", "goal": goal_name}
            m = String()
            m.data = json.dumps(payload, ensure_ascii=False)
            self.cmd_pub.publish(m)
            print("Published:", m.data)

        def send_cancel(self):
            payload = {"type": "cancel"}
            m = String()
            m.data = json.dumps(payload, ensure_ascii=False)
            self.cmd_pub.publish(m)
            print("Published:", m.data)


    # ==============================
    # NLP NHỎ CHO LỆNH ĐI TỚI VỊ TRÍ
    # ==============================
    def normalize_text(text: str) -> str:
        return text.lower().strip()


    def contains_word(text: str, words) -> bool:
        t = normalize_text(text)
        return any(re.search(rf"(^|\s){re.escape(w)}($|\s)", t) for w in words)


    def parse_go_intent(text: str) -> Optional[str]:
        t = text.strip()
        for pat in GO_PATTERNS:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                return m.group(m.lastindex)
        return None


    def normalize_goal(goal: str) -> str:
        g = goal.lower().strip()

        mapping = {
            "a": "A",
            "b": "B", "bi": "B", "bê": "B",
            "c": "C", "xi": "C", "xê": "C",
            "d": "D", "đê": "D",
            "e": "E", "i": "E",
        }

        return mapping.get(g, goal.upper())


    def is_yes(text: str) -> bool:
        return contains_word(text, YES_WORDS)


    def is_no(text: str) -> bool:
        return contains_word(text, NO_WORDS)


    def has_wake_word(text: str) -> bool:
        t = normalize_text(text)
        return any(w in t for w in WAKE_WORDS)


    def remove_wake_word(text: str) -> str:
        cleaned = text
        for w in WAKE_WORDS:
            cleaned = re.sub(rf"\b{re.escape(w)}\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?:;")
        return cleaned


    def is_cancel_command(text: str) -> bool:
        t = normalize_text(text)
        return any(w in t for w in ["hủy", "huỷ", "cancel", "dừng lại", "dừng di chuyển"])


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
    # MICROPHONE
    # ==============================
    def listen_voice() -> Optional[str]:
        try:
            with sr.Microphone() as mic:
                robot_ear.adjust_for_ambient_noise(mic, duration=0.5)
                print("Robot: Tôi đang nghe...")
                audio = robot_ear.listen(mic, timeout=1, phrase_time_limit=3)
                return robot_ear.recognize_google(audio, language="vi-VN")

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            print("Mic error: Không nhận diện được giọng nói")
            return None
        except sr.RequestError as e:
            print("Mic error: Lỗi dịch vụ Google Speech:", e)
            return None
        except Exception as e:
            print("Mic error:", e)
            return None


    # ==============================
    # CHAT GPT
    # ==============================
    def ask_openai(user_text: str) -> str:
        try:
            completion = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là Lumi, một robot hỗ trợ người khiếm thị. "
                            "Bạn nói chuyện thân thiện, rõ ràng, dễ hiểu. "
                            "Luôn xưng là 'Lumi'. "
                            "Trả lời ngắn gọn, hữu ích. "
                            "Nếu người dùng muốn robot di chuyển, hãy hướng dẫn họ nói: "
                            "'Lumi, đi tới vị trí A'."
                        ),
                    },
                    {"role": "user", "content": user_text},
                ],
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print("OpenAI error:", e)
            return "Lumi đang bị lỗi kết nối AI, bạn vui lòng thử lại."


    # ==============================
    # MAIN LOOP
    # ==============================
    def main():
        rclpy.init()
        bridge = VoiceRosBridge()

        pending_goal: Optional[str] = None

        speak("Xin chào, tôi là Lumi, robot hỗ trợ người khiếm thị. Tôi có thể giúp gì cho bạn?")

        try:
            while True:
                rclpy.spin_once(bridge, timeout_sec=0.1)

                you = listen_voice()
                if not you:
                    continue

                print("You:", you)
                print("DEBUG raw:", repr(you))

                you_l_raw = you.lower()

                # Thoát chương trình
                if any(cmd in you_l_raw for cmd in STOP_COMMANDS):
                    speak("Vâng, tạm biệt bạn.")
                    break

                # Nếu đang chờ xác nhận thì không bắt buộc phải gọi Lumi
                if pending_goal is not None:
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

                    speak(f"Bạn có chắc chắn muốn đi đến {pending_goal} không? Nói có hoặc không.")
                    continue

                # Chỉ phản hồi khi có gọi tên Lumi
                if not has_wake_word(you):
                    print("Bỏ qua vì không có wake word:", you)
                    continue

                # Xóa từ gọi Lumi để phân tích câu lệnh
                you = remove_wake_word(you)
                you_l = you.lower()
                print("After wake-word cleanup:", you)

                # Lệnh hủy di chuyển
                if is_cancel_command(you_l):
                    bridge.send_cancel()
                    speak("Tôi đã gửi lệnh hủy.")
                    continue

                # Hỏi ngày giờ
                if "hôm nay là thứ mấy" in you_l or "hôm nay thứ mấy" in you_l or "ngày bao nhiêu" in you_l:
                    speak(get_vietnam_time_text())
                    continue

                # Bắt ý định đi tới waypoint
                goal = parse_go_intent(you)
                print("DEBUG goal:", goal)

                if goal:
                    goal = normalize_goal(goal)

                    if bridge.known_waypoints and goal not in bridge.known_waypoints:
                        speak(f"Tôi không thấy vị trí {goal}. Các vị trí có sẵn là: {', '.join(bridge.known_waypoints)}.")
                        continue

                    pending_goal = goal
                    speak(f"Bạn có chắc chắn muốn đi đến {goal} không?")
                    continue

                # Giao tiếp bình thường bằng OpenAI
                robot_brain = ask_openai(you)
                speak(robot_brain)

        finally:
            pygame.mixer.quit()
            cleanup_voices()
            bridge.destroy_node()
            rclpy.shutdown()


    if __name__ == "__main__":
        main()
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
# FOLDER CONFIG
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

# ===== SHOW CAPTURED IMAGE =====
SHOW_CAMERA_WINDOW = True

# ===== EASIER RECOGNITION (webcam) =====
FACE_TOLERANCE = 0.55  # increase to 0.60 if still hard to recognize (but easier to confuse)

# ===== BIGGER BOX ~ 1.5x =====
BOX_SCALE = 0.5          # expand each side by 50% => total about 1.5x
BOX_THICKNESS = 3
FONT_SCALE = 0.9
FONT_THICKNESS = 2

# ==============================
# OPENAI
# ==============================
client = OpenAI()

STOP_COMMANDS = ["goodbye", "stop", "exit", "bye", "quit"]

# CONFIRMATION WORDS
YES_WORDS = ["yes", "yeah", "yep", "ok", "okay", "sure", "correct", "go ahead"]
NO_WORDS = ["no", "nope", "cancel", "don't", "do not"]

# COMMANDS LIKE: "go to A", "take me to B", "go to room kitchen"
GO_PATTERNS = [
    r"\b(take|bring|guide|lead)\s+(me\s+)?to\s+(location\s+)?([A-Za-z0-9_]+)\b",
    r"\b(go|move|head)\s+to\s+(location\s+)?([A-Za-z0-9_]+)\b",
    r"\b(go|move|head)\s+to\s+room\s+([A-Za-z0-9_]+)\b",
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
                print("CANNOT ENCODE:", file)


load_faces()

# ==============================
# READ PDF PROFILE
# ==============================
def read_profile(name):
    pdf_path = os.path.join(PROFILE_DIR, f"{name}.pdf")
    if not os.path.exists(pdf_path):
        return "No detailed information is available."

    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
        return " ".join(text_parts).strip() or "No detailed information is available."
    except:
        return "I cannot read the profile file."


# ==============================
# UTILITY: EXPAND BOX ~1.5x
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
# CAMERA RECOGNITION (MULTIPLE FACES + ORDER 1,2,3)
# ==============================
def recognize_person():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return []

    frame = None
    for _ in range(30):
        ret, tmp = cap.read()
        if ret and tmp is not None:
            frame = tmp

    cap.release()

    if frame is None:
        return []

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    if not face_locations:
        if SHOW_CAMERA_WINDOW:
            display = frame.copy()
            cv2.putText(display, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Captured image - Press any key to close", display)
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
        name = "Unknown person"

        if known_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances))
            best_dist = float(distances[best_idx])

            if best_dist < FACE_TOLERANCE:
                name = known_names[best_idx]

        if name == "Unknown person":
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
        cv2.imshow("Captured image - Press any key to close", display)
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
        tts = gTTS(text=text, lang="en")
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

        self.known_waypoints = []   # received from commander when event=ready
        self.busy = False           # based on event busy/nav_start/nav_done...
        self.last_status = None

    def on_status(self, msg: String):
        self.last_status = msg.data
        try:
            data = json.loads(msg.data)
        except:
            return

        event = data.get("event", "")
        goal = data.get("goal")

        if event == "nav_succeeded":
            speak(f"I have arrived at {goal}." if goal else "I have arrived.")
            return
        if event == "nav_canceled":
            speak("I have canceled the movement.")
            return
        if event == "nav_aborted":
            speak("I cannot reach the destination.")
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
# SMALL NLP: detect intent to go to waypoint
# ==============================
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

        "b": "B",
        "bee": "B",
        "be": "B",
        "bi": "B",

        "c": "C",
        "see": "C",
        "si": "C",
        "xi": "C",
    }

    return mapping.get(g, goal.upper())


def is_yes(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in YES_WORDS)


def is_no(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in NO_WORDS)


# ==============================
# MAIN LOOP
# ==============================
def main():
    # init ROS
    rclpy.init()
    bridge = VoiceRosBridge()

    pending_goal: Optional[str] = None

    speak("I'm Lumi, It's cloudy today, the temperature is 25 degrees.")

    try:
        while True:
            # Let ROS process status between loop iterations
            rclpy.spin_once(bridge, timeout_sec=0.1)

            with sr.Microphone() as mic:
                print("Robot: I'm listening...")
                try:
                    audio = robot_ear.listen(
                        mic,
                        timeout=0.8,
                        phrase_time_limit=6
                    )
                    you = robot_ear.recognize_google(
                        audio,
                        language="en-US"
                    )
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except Exception as e:
                    print("Voice error:", e)
                    continue

            you_l = you.lower()
            print("You:", you)
            print("DEBUG raw:", repr(you))
            goal_test = parse_go_intent(you)
            print("DEBUG goal:", goal_test)

            # STOP
            if any(cmd in you_l for cmd in STOP_COMMANDS):
                speak("Goodbye.")
                break

            # If waiting for confirmation to go to pending_goal
            if pending_goal is not None:
                if is_yes(you):
                    if bridge.busy:
                        speak(f"I am busy and cannot go to {pending_goal} right now.")
                    else:
                        bridge.send_go(pending_goal)
                        speak(f"Okay. I will go to {pending_goal}.")
                    pending_goal = None
                    continue

                if is_no(you):
                    speak("Okay. I will not move.")
                    pending_goal = None
                    continue

                speak(f"Are you sure you want me to go to {pending_goal}? Say yes or no.")
                continue

            # ===== ASK ABOUT A PERSON =====
            if "who is this" in you_l or "who is that" in you_l:
                load_faces()
                names = recognize_person()

                if not names:
                    speak("I cannot see anyone in the frame.")
                    continue

                parts = []
                for i, name in enumerate(names, start=1):
                    if name == "Unknown person":
                        parts.append(f"Person {i} is unknown. I have saved the image.")
                    else:
                        parts.append(f"Person {i} is {name}. {read_profile(name)}")

                response = " ".join(parts)
                print("Robot:", response)
                speak(response)
                continue

            # ===== DETECT INTENT TO GO TO WAYPOINT =====
            goal = parse_go_intent(you)
            if goal:
                goal = normalize_goal(goal)

                if bridge.known_waypoints and goal not in bridge.known_waypoints:
                    speak(f"I cannot find location {goal}. Available locations are: {', '.join(bridge.known_waypoints)}.")
                    continue

                pending_goal = goal
                speak(f"Are you sure you want me to go to {goal}?")
                continue

            # ===== CANCEL COMMAND =====
            if "cancel" in you_l:
                bridge.send_cancel()
                speak("I have sent the cancel command.")
                continue

            # ===== NORMAL CHATGPT CHAT =====
            if not client:
                speak("My OpenAI API key is not configured.")
                continue

            try:
                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a robot assistant. Reply briefly and clearly in English. "
                                "If the user wants to go somewhere, instruct them to say "
                                "'go to location A'."
                            ),
                        },
                        {"role": "user", "content": you},
                    ],
                    temperature=0.7,
                )
                robot_brain = completion.choices[0].message.content
            except Exception as e:
                print("OpenAI error:", e)
                robot_brain = "I am busy right now. Please try again."

            print("Robot:", robot_brain)
            speak(robot_brain)

    finally:
        pygame.mixer.quit()
        cleanup_voices()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
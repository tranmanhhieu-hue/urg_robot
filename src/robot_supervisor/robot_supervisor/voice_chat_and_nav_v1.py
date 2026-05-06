#!/usr/bin/env python3
import os
import time
import atexit
import json
import re
from typing import Optional

import speech_recognition as sr
from gtts import gTTS
from openai import OpenAI
import pygame

from datetime import datetime

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
# THƯ MỤC
# ==============================
VOICE_DIR = "data/voices"
os.makedirs(VOICE_DIR, exist_ok=True)

# ==============================
# OPENAI
# ==============================
client = OpenAI()

STOP_COMMANDS = ["tạm biệt", "dừng", "thoát", "bye"]
WAKE_WORDS = ["lumi", "lu mi", "lumi ơi", "ê lumi"]

YES_WORDS = ["có", "đúng", "ok", "oke", "ừ", "yes", "chắc chắn", "đồng ý", "đi"]
NO_WORDS = ["không", "ko", "k", "thôi", "hủy", "huỷ", "no"]

GO_PATTERNS = [
    r"\b(đưa|dẫn|cho|chở)\s+(?:tôi|mình)?\s*(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
    r"\b(đi|tới|đến)\s+(?:vị\s*trí\s*)?([A-Za-z0-9_]+)\b",
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
# VOICE
# ==============================
def speak(text):
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

        self.busy = False
        self.known_waypoints = []

    def on_status(self, msg: String):
        try:
            data = json.loads(msg.data)
        except:
            return

        event = data.get("event", "")
        goal = data.get("goal")

        if event == "nav_succeeded":
            speak(f"Tôi đã đến vị trí {goal}.")

        if event == "nav_canceled":
            speak("Tôi đã hủy di chuyển.")
        if event == "nav_aborted":
            speak("Tôi không thể đến đích.")

        if event == "ready":
            self.known_waypoints = data.get("waypoints", [])

        if event in ("busy", "nav_start", "goal_accepted"):
            self.busy = True

        if event in ("nav_succeeded", "nav_canceled", "nav_aborted", "nav_done"):
            self.busy = False

    def send_go(self, goal):
        msg = String()
        msg.data = json.dumps({"type": "go", "goal": goal})
        self.cmd_pub.publish(msg)

    def send_cancel(self):
        msg = String()
        msg.data = json.dumps({"type": "cancel"})
        self.cmd_pub.publish(msg)

# ==============================
# NLP
# ==============================
def parse_go_intent(text):
    for pat in GO_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(m.lastindex)
    return None

def is_yes(text):
    return any(w in text.lower() for w in YES_WORDS)

def is_no(text):
    return any(w in text.lower() for w in NO_WORDS)

def has_wake_word(text):
    return any(w in text.lower() for w in WAKE_WORDS)

def remove_wake_word(text):
    for w in WAKE_WORDS:
        text = re.sub(rf"\b{w}\b", "", text, flags=re.IGNORECASE)
    return text.strip()

# ==============================
# MAIN
# ==============================
def main():
    rclpy.init()
    bridge = VoiceRosBridge()

    pending_goal = None

    speak("Xin chào, tôi là Lumi. Tôi có thể giúp gì cho bạn?")

    try:
        while True:
            rclpy.spin_once(bridge, timeout_sec=0.1)

            with sr.Microphone() as mic:
                try:
                    audio = robot_ear.listen(mic, timeout=1, phrase_time_limit=5)
                    you = robot_ear.recognize_google(audio, language="vi-VN")
                except:
                    continue

            print("You:", you)

            if any(cmd in you.lower() for cmd in STOP_COMMANDS):
                speak("Tạm biệt.")
                break

            if pending_goal:
                if is_yes(you):
                    if bridge.busy:
                        bridge.send_cancel()
                        time.sleep(0.3)
                        bridge.busy = False

                    bridge.send_go(pending_goal)
                    speak(f"Tôi sẽ đi đến {pending_goal}.")
                    pending_goal = None
                    continue

                if is_no(you):
                    speak("Đã hủy.")
                    pending_goal = None
                    continue

            if not has_wake_word(you):
                continue

            you = remove_wake_word(you)

            goal = parse_go_intent(you)
            if goal:
                pending_goal = goal.upper()
                speak(f"Bạn có chắc muốn đi đến {pending_goal} không?")
                continue

    finally:
        pygame.mixer.quit()
        cleanup_voices()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
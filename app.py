import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time
import pygetwindow as gw
import subprocess
import os

# ---------------- SETTINGS ---------------- #
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

screen_w, screen_h = pyautogui.size()

frame_count = 0

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# ---------------- VARIABLES ---------------- #
prev_x, prev_y = 0, 0
smoothening = 5

click_delay = 0.3
last_click_time = 0
dragging = False

# Swipe (horizontal)
prev_hand_x = 0
swipe_threshold = 100
swipe_cooldown = 0.7
last_swipe_time = 0

# Scroll (vertical)
prev_hand_y = 0
scroll_threshold = 80
scroll_cooldown = 0.5
last_scroll_time = 0

# Air drawing
drawing = False
points = []
min_points = 25

# App detection
last_app_check = 0
app_name = "unknown"

# ---------------- FUNCTIONS ---------------- #

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_active_app():
    try:
        window = gw.getActiveWindow()
        if window:
            return window.title.lower()
    except:
        return "unknown"
    return "unknown"

def handle_swipe(direction, app):
    if "chrome" in app:
        pyautogui.hotkey('ctrl', 'tab' if direction=="right" else 'shift', 'tab')
    elif "code" in app:
        pyautogui.hotkey('ctrl', 'pagedown' if direction=="right" else 'pageup')
    else:
        pyautogui.press('right' if direction=="right" else 'left')

def detect_shape(points):
    if len(points) < min_points:
        return None

    pts = np.array(points)
    x, y = pts[:, 0], pts[:, 1]

    width = max(x) - min(x)
    height = max(y) - min(y)

    if width < 50 or height < 50:
        return None

    start, end = pts[0], pts[-1]
    mid = pts[len(pts)//2]

    if abs(start[0] - end[0]) > 40 and height > width * 0.5:
        return "C"

    if mid[1] > start[1] and mid[1] > end[1]:
        return "V"

    peaks = sum(1 for i in range(1, len(y)-1) if y[i] < y[i-1] and y[i] < y[i+1])
    if peaks >= 2:
        return "M"

    if width > 60 and height > 60 and abs(start[1] - end[1]) > 30:
        return "S"

    return None

def perform_action(shape):
    if shape == "C":
        subprocess.Popen("C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
    elif shape == "V":
        subprocess.Popen("C:\\Users\\DELL\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe")
    elif shape == "M":
        pyautogui.hotkey('win', 'down')
    elif shape == "S":
        os.system("shutdown /s /t 10")

# ---------------- MAIN LOOP ---------------- #

while True:
    success, frame = cap.read()
    if not success or frame is None:
        continue

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # Update app every 1 sec
    if time.time() - last_app_check > 1:
        app_name = get_active_app()
        last_app_check = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            # CURSOR
            screen_x = np.interp(index_tip[0], [0, w], [0, screen_w])
            screen_y = np.interp(index_tip[1], [0, h], [0, screen_h])

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            if abs(curr_x - prev_x) > 2 or abs(curr_y - prev_y) > 2:
                pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            thumb_index_dist = distance(thumb_tip, index_tip)

            # CLICK
            if thumb_index_dist < 30:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()

            # DRAG
            if thumb_index_dist < 25:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # HORIZONTAL SWIPE
            current_x = index_tip[0]
            if prev_hand_x != 0:
                dx = current_x - prev_hand_x
                if abs(dx) > swipe_threshold and time.time() - last_swipe_time > swipe_cooldown:
                    handle_swipe("right" if dx > 0 else "left", app_name)
                    last_swipe_time = time.time()
            prev_hand_x = current_x

            # 🔥 VERTICAL SCROLL (NEW)
            current_y = index_tip[1]
            if prev_hand_y != 0:
                dy = current_y - prev_hand_y
                if abs(dy) > scroll_threshold and time.time() - last_scroll_time > scroll_cooldown:
                    if dy > 0:
                        pyautogui.scroll(-200)  # down
                    else:
                        pyautogui.scroll(200)   # up
                    last_scroll_time = time.time()
            prev_hand_y = current_y

            # AIR DRAWING
            if thumb_index_dist < 30:
                if not drawing:
                    drawing = True
                    points = []

                if len(points) == 0 or distance(points[-1], index_tip) > 5:
                    points.append(index_tip)
            else:
                if drawing:
                    drawing = False
                    shape = detect_shape(points)
                    if shape:
                        perform_action(shape)

            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], (0,255,255), 2)

            cv2.putText(frame, f"App: {app_name[:20]}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.imshow("NeuroGesture FINAL", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
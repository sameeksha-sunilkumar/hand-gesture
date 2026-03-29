import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import time

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Smoothening
prev_x, prev_y = 0, 0
smoothening = 5

# Click control
click_delay = 0.3
last_click_time = 0
dragging = False

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            lm = hand_landmarks.landmark

            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
            middle_tip = (int(lm[12].x * w), int(lm[12].y * h))

            # Cursor movement
            screen_x = np.interp(index_tip[0], [0, w], [0, screen_w])
            screen_y = np.interp(index_tip[1], [0, h], [0, screen_h])

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Distances
            thumb_index_dist = distance(thumb_tip, index_tip)
            index_middle_dist = distance(index_tip, middle_tip)

            # Draw points
            cv2.circle(frame, index_tip, 8, (255, 0, 255), -1)
            cv2.circle(frame, thumb_tip, 8, (0, 255, 0), -1)

            # 🤏 CLICK (pinch)
            if thumb_index_dist < 30:
                if time.time() - last_click_time > click_delay:
                    pyautogui.click()
                    last_click_time = time.time()
                    cv2.putText(frame, "CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # ✊ DRAG (pinch hold)
            if thumb_index_dist < 25:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            # 🖖 SCROLL (index + middle close)
            if index_middle_dist < 40 and thumb_index_dist > 40:
                pyautogui.scroll(20)
                cv2.putText(frame, "SCROLL", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("NeuroGesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
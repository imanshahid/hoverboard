# main.py

import cv2
import numpy as np
from utils.handtracking import HandTracker

cap = cv2.VideoCapture(0)
hand_tracker = HandTracker()
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White canvas
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    landmarks = hand_tracker.find_hand_landmarks(frame)

    if landmarks:
        _, x, y = landmarks[8]  # Index finger tip
        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y

        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 1)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0  # Reset when finger not visible

    output = cv2.addWeighted(frame, 1, canvas, 0, 0)
    cv2.imshow("MindSketch", output)

    key = cv2.waitKey(1)

# Try checking window state safely
    try:
        if key == ord('q') or cv2.getWindowProperty("MindSketch", cv2.WND_PROP_AUTOSIZE) < 1:
            break
    except cv2.error:
    # Window was already closed by user
        break



cap.release()
cv2.destroyAllWindows()

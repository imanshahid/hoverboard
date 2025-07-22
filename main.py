# main.py

import cv2
import numpy as np
from utils.hand_tracking import HandTracker

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

        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 5)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0  # Reset when finger not visible

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("MindSketch", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

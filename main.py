import cv2
import numpy as np
from utils.handtracking import HandTracker
def draw_color_boxes(frame):
    box_colors = [("r", (0,0,255)), ("g", (0,255,0)), ("b", (255,0,0)), ("k", (0,0,0))]
    for i, (label, color) in enumerate(box_colors):
        x1, y1, x2, y2 = 10 + i*70, 10, 60 + i*70, 60
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label.upper(), (x1 + 15, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

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

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    draw_color_boxes(output)

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

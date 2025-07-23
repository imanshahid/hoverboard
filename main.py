import cv2
import numpy as np
from utils.handtracking import HandTracker

def draw_color_boxes(frame):
    box_colors = [("r", (0,0,255)), ("g", (0,255,0)), ("b", (255,0,0)), ("k", (0,0,0))]
    for i, (label, color) in enumerate(box_colors):
        x1, y1, x2, y2 = 10 + i*70, 10, 60 + i*70, 60
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, label.upper(), (x1 + 15, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

COLORS = {
    'r': (0, 0, 255),   # Red
    'g': (0, 255, 0),   # Green
    'b': (255, 0, 0),   # Blue
    'k': (0, 0, 0)      # Black
}

brush_color = COLORS['k']  # Default Black
eraser_mode = False

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
        # If fingertip in color box row
        if 10 < y < 60:
            if 10 < x < 60:
                brush_color, eraser_mode = COLORS['r'], False
            elif 80 < x < 130:
                brush_color, eraser_mode = COLORS['g'], False
            elif 150 < x < 200:
                brush_color, eraser_mode = COLORS['b'], False
            elif 220 < x < 270:
                brush_color, eraser_mode = COLORS['k'], False

        if prev_x == 0 and prev_y == 0:
            prev_x, prev_y = x, y

        color_to_use = (255, 255, 255) if eraser_mode else brush_color
        thickness = 20 if eraser_mode else 5
        cv2.line(canvas, (prev_x, prev_y), (x, y), color_to_use, thickness)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0  # Reset when finger not visible

    output = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    draw_color_boxes(output)

    # Show current brush color / eraser status
    cv2.circle(output, (600, 40), 20, brush_color if not eraser_mode else (255, 255, 255), -1)
    cv2.putText(output, 'Eraser' if eraser_mode else 'Brush', (520, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv2.imshow("MindSketch", output)

    key = cv2.waitKey(1)

    if key in [ord('r'), ord('g'), ord('b'), ord('k')]:
        brush_color = COLORS[chr(key)]
        eraser_mode = False  # Turn off eraser when color is picked

    elif key == ord('e'):
        eraser_mode = not eraser_mode

    try:
        if key == ord('q') or cv2.getWindowProperty("MindSketch", cv2.WND_PROP_AUTOSIZE) < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()

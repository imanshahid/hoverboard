# utils/hand_tracking.py

import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hand_landmarks(self, image, draw=True):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_img)
        landmarks = []

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(image, hand, self.hands_module.HAND_CONNECTIONS)
                for id, lm in enumerate(hand.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))
        return landmarks

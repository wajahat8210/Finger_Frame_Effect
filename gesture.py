import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18
def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5
def _is_l_shape(lm):
    wrist = lm[WRIST]
    thumb_extended = _dist(lm[THUMB_TIP], wrist) > _dist(lm[THUMB_IP], wrist)
    index_extended = _dist(lm[INDEX_TIP], wrist) > _dist(lm[INDEX_PIP], wrist)
    middle_curled = _dist(lm[MIDDLE_TIP], wrist) < _dist(lm[MIDDLE_PIP], wrist)
    ring_curled = _dist(lm[RING_TIP], wrist) < _dist(lm[RING_PIP], wrist)
    pinky_curled = _dist(lm[PINKY_TIP], wrist) < _dist(lm[PINKY_PIP], wrist)
    v1 = np.array([lm[THUMB_TIP].x - wrist.x, lm[THUMB_TIP].y - wrist.y])
    v2 = np.array([lm[INDEX_TIP].x - wrist.x, lm[INDEX_TIP].y - wrist.y])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    roughly_perpendicular = 35 < angle_deg < 150
    return (thumb_extended and index_extended and
            middle_curled and ring_curled and pinky_curled and
            roughly_perpendicular)
def _order_quad(points):
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts)
    return hull.reshape(-1, 2)
class GestureTracker:
    def __init__(self, model_path="hand_landmarker.task",
                 detection_width=480, smoothing=0.35, lost_grace_frames=5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.detection_width = detection_width
        self.smoothing = smoothing
        self.lost_grace_frames = lost_grace_frames
        self.smoothed_quad = None
        self.lost_frames = 0
    def update(self, frame, timestamp_ms):
        h, w, _ = frame.shape
        if w > self.detection_width:
            scale = self.detection_width / w
            small = cv2.resize(frame, (self.detection_width, int(h * scale)))
        else:
            small = frame

        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        frame_points = []
        if result.hand_landmarks:
            for lm in result.hand_landmarks:
                if _is_l_shape(lm):
                    thumb_px = (lm[THUMB_TIP].x * w, lm[THUMB_TIP].y * h)
                    index_px = (lm[INDEX_TIP].x * w, lm[INDEX_TIP].y * h)
                    frame_points.extend([thumb_px, index_px])
        if len(frame_points) == 4:
            self.lost_frames = 0
            new_quad = _order_quad(frame_points)
            if self.smoothed_quad is None:
                self.smoothed_quad = new_quad
            else:
                self.smoothed_quad = (self.smoothing * new_quad +
                                      (1 - self.smoothing) * self.smoothed_quad)
        else:
            self.lost_frames += 1
            if self.lost_frames > self.lost_grace_frames:
                self.smoothed_quad = None
        return self.smoothed_quad
    def close(self):
        self.detector.close()
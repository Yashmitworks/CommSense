"""
Gaze Tracker Module
Tracks eye contact percentage, gaze direction, and blink rate.
Uses landmark pts array (468x2) passed in from FaceAnalyzer.
No direct mediapipe dependency here — works with both old and new API.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class GazeResult:
    timestamp: float
    gaze_direction: str          # "center", "left", "right", "up", "down"
    eye_contact_pct: float       # rolling % of time spent in eye contact
    blink_rate: float            # blinks per minute
    left_ear: float              # Eye Aspect Ratio left
    right_ear: float             # Eye Aspect Ratio right
    is_blinking: bool


class GazeTracker:
    """
    Tracks gaze direction and eye contact using MediaPipe landmarks.
    Expects pts: np.ndarray of shape (468, 2) from FaceAnalyzer.
    """

    # MediaPipe landmark indices
    LEFT_EYE_VERT = [159, 145]   # top, bottom
    LEFT_EYE_HORIZ = [33, 133]   # left corner, right corner
    RIGHT_EYE_VERT = [386, 374]
    RIGHT_EYE_HORIZ = [362, 263]
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    EAR_BLINK_THRESHOLD = 0.21
    GAZE_CENTER_THRESHOLD = 0.15  # normalized offset to count as "center"

    def __init__(self, window_seconds: int = 30):
        self.window_seconds = window_seconds
        self.gaze_log = []          # list of (timestamp, is_center)
        self.blink_log = []         # list of timestamps when blink detected
        self._was_blinking = False

    def update(self, pts: np.ndarray) -> GazeResult:
        """Process landmarks and return gaze result."""
        timestamp = time.time()

        left_ear = self._eye_aspect_ratio(pts, self.LEFT_EYE_VERT, self.LEFT_EYE_HORIZ)
        right_ear = self._eye_aspect_ratio(pts, self.RIGHT_EYE_VERT, self.RIGHT_EYE_HORIZ)
        avg_ear = (left_ear + right_ear) / 2

        is_blinking = avg_ear < self.EAR_BLINK_THRESHOLD

        # Blink detection (rising edge)
        if is_blinking and not self._was_blinking:
            self.blink_log.append(timestamp)
        self._was_blinking = is_blinking

        # Gaze direction
        gaze_dir, is_center = self._compute_gaze_direction(pts)

        # Log gaze
        self.gaze_log.append((timestamp, is_center))

        # Prune old entries
        cutoff = timestamp - self.window_seconds
        self.gaze_log = [(t, v) for t, v in self.gaze_log if t > cutoff]
        self.blink_log = [t for t in self.blink_log if t > cutoff]

        # Eye contact percentage
        if self.gaze_log:
            eye_contact_pct = sum(1 for _, v in self.gaze_log if v) / len(self.gaze_log) * 100
        else:
            eye_contact_pct = 0.0

        # Blink rate (blinks per minute)
        blink_rate = len(self.blink_log) / (self.window_seconds / 60)

        return GazeResult(
            timestamp=timestamp,
            gaze_direction=gaze_dir,
            eye_contact_pct=round(eye_contact_pct, 1),
            blink_rate=round(blink_rate, 1),
            left_ear=round(left_ear, 3),
            right_ear=round(right_ear, 3),
            is_blinking=is_blinking
        )

    def _eye_aspect_ratio(self, pts: np.ndarray, vert_idx: list, horiz_idx: list) -> float:
        """EAR = vertical distance / horizontal distance."""
        try:
            vert = np.linalg.norm(pts[vert_idx[0]] - pts[vert_idx[1]])
            horiz = np.linalg.norm(pts[horiz_idx[0]] - pts[horiz_idx[1]])
            return float(vert / (horiz + 1e-6))
        except Exception:
            return 0.3

    def _compute_gaze_direction(self, pts: np.ndarray) -> Tuple[str, bool]:
        """
        Estimate gaze direction from iris position relative to eye corners.
        Returns (direction_string, is_center_bool).
        """
        try:
            left_iris = pts[self.LEFT_IRIS].mean(axis=0)
            right_iris = pts[self.RIGHT_IRIS].mean(axis=0)

            left_corner_l = pts[self.LEFT_EYE_HORIZ[0]]
            left_corner_r = pts[self.LEFT_EYE_HORIZ[1]]
            right_corner_l = pts[self.RIGHT_EYE_HORIZ[0]]
            right_corner_r = pts[self.RIGHT_EYE_HORIZ[1]]

            # Horizontal ratio: 0 = far left, 1 = far right
            left_ratio_h = (left_iris[0] - left_corner_l[0]) / (
                left_corner_r[0] - left_corner_l[0] + 1e-6)
            right_ratio_h = (right_iris[0] - right_corner_l[0]) / (
                right_corner_r[0] - right_corner_l[0] + 1e-6)
            avg_h = (left_ratio_h + right_ratio_h) / 2

            # Vertical ratio
            left_top = pts[self.LEFT_EYE_VERT[0]]
            left_bot = pts[self.LEFT_EYE_VERT[1]]
            left_ratio_v = (left_iris[1] - left_top[1]) / (
                left_bot[1] - left_top[1] + 1e-6)

            # Determine direction
            h_offset = abs(avg_h - 0.5)
            is_center = h_offset < self.GAZE_CENTER_THRESHOLD and (0.3 < left_ratio_v < 0.7)

            if is_center:
                direction = "center"
            elif avg_h < 0.35:
                direction = "left"
            elif avg_h > 0.65:
                direction = "right"
            elif left_ratio_v < 0.3:
                direction = "up"
            else:
                direction = "down"

            return direction, is_center

        except Exception:
            return "unknown", False

    def reset(self):
        self.gaze_log.clear()
        self.blink_log.clear()
        self._was_blinking = False

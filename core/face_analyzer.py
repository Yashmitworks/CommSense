"""
Face Analyzer Module
Handles real-time facial expression analysis using MediaPipe FaceLandmarker (Tasks API)
and DeepFace for emotion detection.
Compatible with mediapipe >= 0.10.14 (new Tasks API — mp.solutions removed).
"""

import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions


@dataclass
class FaceAnalysisResult:
    timestamp: float
    emotion: str
    emotion_scores: dict
    confidence_score: float       # 0-100
    nervousness_score: float      # 0-100
    trust_score: float            # 0-100
    eye_contact_score: float      # 0-100
    landmarks_detected: bool
    face_detected: bool
    brow_tension: float           # 0-1
    lip_compression: float        # 0-1
    jaw_set: float                # 0-1
    smile_genuine: float          # 0-1 (Duchenne marker)


# Landmark indices (same 468-point mesh, same indices as before)
LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LIPS_UPPER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_LOWER = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LEFT_BROW  = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
JAW        = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]

# Model path — downloaded by setup or on first run
_MODEL_PATH = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")


class FaceAnalyzer:
    def __init__(self):
        if not os.path.exists(_MODEL_PATH):
            self._download_model()

        options = FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # DeepFace interval (every N frames to save CPU)
        self.deepface_interval = 15
        self.frame_count = 0
        self.last_deepface_result = None

        # Smoothing buffers
        self.confidence_buffer = []
        self.nervousness_buffer = []
        self.trust_buffer = []
        self.buffer_size = 10

    def _download_model(self):
        import urllib.request
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print("  Downloading face_landmarker.task model (~3.6 MB)...")
        urllib.request.urlretrieve(url, _MODEL_PATH)
        print("  Model downloaded.")

    def analyze_frame(self, frame: np.ndarray) -> FaceAnalysisResult:
        """Main analysis function — call this per frame."""
        timestamp = time.time()
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return FaceAnalysisResult(
                timestamp=timestamp, emotion="unknown", emotion_scores={},
                confidence_score=0, nervousness_score=0, trust_score=0,
                eye_contact_score=0, landmarks_detected=False, face_detected=False,
                brow_tension=0, lip_compression=0, jaw_set=0, smile_genuine=0
            )

        # Convert normalized landmarks to pixel coords
        raw = detection_result.face_landmarks[0]
        pts = np.array([[lm.x * w, lm.y * h] for lm in raw])

        # Geometric features
        brow_tension    = self._compute_brow_tension(pts)
        lip_compression = self._compute_lip_compression(pts)
        jaw_set         = self._compute_jaw_set(pts)
        eye_contact     = self._compute_eye_contact(pts)
        smile_genuine   = self._compute_genuine_smile(pts)

        # DeepFace every N frames
        self.frame_count += 1
        emotion, emotion_scores = "neutral", {}

        if self.frame_count % self.deepface_interval == 0:
            try:
                analysis = DeepFace.analyze(
                    frame, actions=["emotion"],
                    enforce_detection=False, silent=True
                )
                if analysis:
                    self.last_deepface_result = analysis[0]
            except Exception:
                pass

        if self.last_deepface_result:
            emotion       = self.last_deepface_result.get("dominant_emotion", "neutral")
            emotion_scores = self.last_deepface_result.get("emotion", {})

        confidence  = self._compute_confidence_score(emotion, emotion_scores, brow_tension, lip_compression, jaw_set, eye_contact)
        nervousness = self._compute_nervousness_score(emotion, emotion_scores, brow_tension, lip_compression, eye_contact)
        trust       = self._compute_trust_score(emotion, emotion_scores, eye_contact, smile_genuine, brow_tension)

        confidence  = self._smooth(self.confidence_buffer,  confidence)
        nervousness = self._smooth(self.nervousness_buffer, nervousness)
        trust       = self._smooth(self.trust_buffer,       trust)

        return FaceAnalysisResult(
            timestamp=timestamp,
            emotion=emotion,
            emotion_scores=emotion_scores,
            confidence_score=round(confidence, 1),
            nervousness_score=round(nervousness, 1),
            trust_score=round(trust, 1),
            eye_contact_score=round(eye_contact, 1),
            landmarks_detected=True,
            face_detected=True,
            brow_tension=round(brow_tension, 3),
            lip_compression=round(lip_compression, 3),
            jaw_set=round(jaw_set, 3),
            smile_genuine=round(smile_genuine, 3)
        )

    def draw_overlay(self, frame: np.ndarray, result: FaceAnalysisResult) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        if not result.face_detected:
            cv2.putText(overlay, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return overlay
        self._draw_score_bar(overlay, "Confidence",  result.confidence_score,  (w - 220, 30),  (0, 200, 100))
        self._draw_score_bar(overlay, "Trust",       result.trust_score,       (w - 220, 80),  (100, 200, 255))
        self._draw_score_bar(overlay, "Nervousness", result.nervousness_score, (w - 220, 130), (0, 100, 255))
        self._draw_score_bar(overlay, "Eye Contact", result.eye_contact_score, (w - 220, 180), (200, 150, 50))
        ec = self._emotion_color(result.emotion)
        cv2.putText(overlay, f"Emotion: {result.emotion.upper()}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ec, 2)
        return overlay

    # ── Private helpers ──────────────────────────────────────────────────────

    def _compute_brow_tension(self, pts):
        left_brow_y  = np.mean(pts[LEFT_BROW, 1])
        right_brow_y = np.mean(pts[RIGHT_BROW, 1])
        left_eye_y   = np.mean(pts[LEFT_EYE, 1])
        right_eye_y  = np.mean(pts[RIGHT_EYE, 1])
        avg_gap      = ((abs(left_eye_y - left_brow_y) + abs(right_eye_y - right_brow_y)) / 2)
        face_height  = pts[:, 1].max() - pts[:, 1].min()
        return float(np.clip(1.0 - min(avg_gap / (face_height * 0.15), 1.0), 0, 1))

    def _compute_lip_compression(self, pts):
        lip_gap     = abs(np.mean(pts[LIPS_LOWER, 1]) - np.mean(pts[LIPS_UPPER, 1]))
        face_height = pts[:, 1].max() - pts[:, 1].min()
        return float(np.clip(1.0 - min(lip_gap / (face_height * 0.06), 1.0), 0, 1))

    def _compute_jaw_set(self, pts):
        jaw_width  = pts[JAW, 0].max() - pts[JAW, 0].min()
        face_width = pts[:, 0].max() - pts[:, 0].min()
        return float(np.clip((jaw_width / (face_width + 1e-6) - 0.5) * 2, 0, 1))

    def _compute_eye_contact(self, pts):
        try:
            left_iris_c  = pts[LEFT_IRIS].mean(axis=0)
            right_iris_c = pts[RIGHT_IRIS].mean(axis=0)
            left_eye_c   = pts[LEFT_EYE].mean(axis=0)
            right_eye_c  = pts[RIGHT_EYE].mean(axis=0)
            avg_offset   = (np.linalg.norm(left_iris_c - left_eye_c) +
                            np.linalg.norm(right_iris_c - right_eye_c)) / 2
            eye_width    = np.linalg.norm(pts[LEFT_EYE[0]] - pts[LEFT_EYE[8]])
            return float(np.clip((1.0 - min(avg_offset / (eye_width * 0.4 + 1e-6), 1.0)) * 100, 0, 100))
        except Exception:
            return 50.0

    def _compute_genuine_smile(self, pts):
        try:
            mouth_corners  = np.array([pts[61], pts[291]])
            mouth_center_y = np.mean(pts[LIPS_UPPER, 1])
            corner_raise   = mouth_center_y - np.mean(mouth_corners[:, 1])
            face_height    = pts[:, 1].max() - pts[:, 1].min()
            return float(np.clip(corner_raise / (face_height * 0.05 + 1e-6), 0, 1))
        except Exception:
            return 0.0

    def _compute_confidence_score(self, emotion, emotion_scores, brow_tension, lip_compression, jaw_set, eye_contact):
        score = 50.0
        for em, d in {"neutral": 10, "happy": 15, "surprise": -5}.items():
            score += d * emotion_scores.get(em, 0) / 100
        for em, d in {"fear": -25, "sad": -20, "angry": -10, "disgust": -15}.items():
            score += d * emotion_scores.get(em, 0) / 100
        score -= brow_tension * 20
        score -= lip_compression * 15
        score += eye_contact * 0.2
        score += jaw_set * 5
        return float(np.clip(score, 0, 100))

    def _compute_nervousness_score(self, emotion, emotion_scores, brow_tension, lip_compression, eye_contact):
        score = 20.0
        for em, d in {"fear": 40, "sad": 15, "disgust": 10}.items():
            score += d * emotion_scores.get(em, 0) / 100
        for em, d in {"neutral": -10, "happy": -15}.items():
            score += d * emotion_scores.get(em, 0) / 100
        score += brow_tension * 30
        score += lip_compression * 25
        score -= eye_contact * 0.15
        return float(np.clip(score, 0, 100))

    def _compute_trust_score(self, emotion, emotion_scores, eye_contact, smile_genuine, brow_tension):
        score = 50.0
        for em, d in {"neutral": 5, "happy": 20}.items():
            score += d * emotion_scores.get(em, 0) / 100
        for em, d in {"angry": -20, "disgust": -25, "fear": -15}.items():
            score += d * emotion_scores.get(em, 0) / 100
        score += eye_contact * 0.25
        score += smile_genuine * 20
        score -= brow_tension * 15
        return float(np.clip(score, 0, 100))

    def _smooth(self, buffer, value):
        buffer.append(value)
        if len(buffer) > self.buffer_size:
            buffer.pop(0)
        return float(np.mean(buffer))

    def _draw_score_bar(self, frame, label, score, pos, color):
        x, y = pos
        bar_w, bar_h = 180, 16
        filled = int(bar_w * score / 100)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (200, 200, 200), 1)
        cv2.putText(frame, f"{label}: {score:.0f}", (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    def _emotion_color(self, emotion):
        return {"happy": (0,255,100), "neutral": (200,200,200), "sad": (255,150,50),
                "angry": (0,50,255), "fear": (0,165,255), "surprise": (255,255,0),
                "disgust": (0,100,150)}.get(emotion, (200,200,200))

    def release(self):
        self.landmarker.close()

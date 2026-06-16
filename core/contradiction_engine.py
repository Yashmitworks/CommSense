"""
Contradiction Engine — The Core Differentiator
Detects semantic mismatches between what someone says and what their face shows.

Examples:
- Says "I'm confident" → face shows fear → CREDIBILITY GAP
- Says "This is straightforward" → face shows confusion → COMPLEXITY LEAK
- Says "I'm calm" → nervousness score spikes → STRESS LEAK
"""

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional
from core.face_analyzer import FaceAnalysisResult


# ── Keyword maps for verbal sentiment ────────────────────────────────────────

CONFIDENCE_WORDS = [
    "confident", "certain", "sure", "absolutely", "definitely", "clearly",
    "obviously", "without doubt", "i know", "i'm sure", "guaranteed",
    "proven", "established", "strong", "solid", "robust"
]

UNCERTAINTY_WORDS = [
    "maybe", "perhaps", "possibly", "i think", "i believe", "not sure",
    "might", "could be", "approximately", "roughly", "sort of", "kind of",
    "i guess", "probably", "uncertain", "unclear"
]

CALM_WORDS = [
    "calm", "relaxed", "comfortable", "fine", "okay", "no problem",
    "easy", "simple", "straightforward", "no issue", "handled"
]

STRESS_WORDS = [
    "difficult", "challenging", "worried", "concerned", "problem",
    "issue", "risk", "danger", "threat", "critical", "urgent", "serious"
]

POSITIVE_WORDS = [
    "great", "excellent", "perfect", "wonderful", "amazing", "fantastic",
    "good", "positive", "success", "achieve", "accomplish", "win"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "failure", "failed", "wrong", "mistake",
    "error", "loss", "defeat", "problem", "crisis"
]


@dataclass
class ContradictionEvent:
    timestamp: float
    contradiction_type: str      # e.g. "CREDIBILITY_GAP", "STRESS_LEAK"
    severity: str                # "low", "medium", "high"
    verbal_signal: str           # what was said
    facial_signal: str           # what face showed
    description: str             # human-readable explanation
    confidence_score: float      # face confidence at time of event
    nervousness_score: float     # face nervousness at time of event


@dataclass
class ContradictionReport:
    total_events: int
    high_severity: int
    medium_severity: int
    low_severity: int
    events: List[ContradictionEvent]
    summary: str


class ContradictionEngine:
    """
    Analyzes rolling window of face data + transcript segments
    to detect verbal-facial contradictions.
    """

    def __init__(self):
        self.events: List[ContradictionEvent] = []
        self.face_buffer: List[FaceAnalysisResult] = []
        self.buffer_window = 5.0  # seconds of face data to consider

    def update_face(self, result: FaceAnalysisResult):
        """Feed latest face analysis result."""
        self.face_buffer.append(result)
        cutoff = time.time() - self.buffer_window
        self.face_buffer = [r for r in self.face_buffer if r.timestamp > cutoff]

    def analyze_transcript(self, text: str, timestamp: float) -> List[ContradictionEvent]:
        """
        Analyze a new transcript segment against current face state.
        Returns list of detected contradictions (may be empty).
        """
        if not self.face_buffer or not text.strip():
            return []

        text_lower = text.lower()
        avg_confidence = sum(r.confidence_score for r in self.face_buffer) / len(self.face_buffer)
        avg_nervousness = sum(r.nervousness_score for r in self.face_buffer) / len(self.face_buffer)
        avg_trust = sum(r.trust_score for r in self.face_buffer) / len(self.face_buffer)
        dominant_emotion = self.face_buffer[-1].emotion

        new_events = []

        # ── Rule 1: Verbal confidence + facial anxiety ────────────────────
        verbal_confident = any(w in text_lower for w in CONFIDENCE_WORDS)
        if verbal_confident and avg_nervousness > 55 and avg_confidence < 45:
            event = ContradictionEvent(
                timestamp=timestamp,
                contradiction_type="CREDIBILITY_GAP",
                severity=self._severity(avg_nervousness, 55, 70, 85),
                verbal_signal=self._extract_trigger(text_lower, CONFIDENCE_WORDS),
                facial_signal=f"nervousness={avg_nervousness:.0f}, confidence={avg_confidence:.0f}",
                description=(
                    f"You said something confident but your face shows anxiety "
                    f"(nervousness {avg_nervousness:.0f}/100). "
                    f"Audiences detect this mismatch subconsciously."
                ),
                confidence_score=avg_confidence,
                nervousness_score=avg_nervousness
            )
            new_events.append(event)

        # ── Rule 2: Verbal calm + facial stress ───────────────────────────
        verbal_calm = any(w in text_lower for w in CALM_WORDS)
        if verbal_calm and avg_nervousness > 60:
            event = ContradictionEvent(
                timestamp=timestamp,
                contradiction_type="STRESS_LEAK",
                severity=self._severity(avg_nervousness, 60, 72, 85),
                verbal_signal=self._extract_trigger(text_lower, CALM_WORDS),
                facial_signal=f"nervousness={avg_nervousness:.0f}, emotion={dominant_emotion}",
                description=(
                    f"You projected calm verbally but stress is leaking through your face "
                    f"(nervousness {avg_nervousness:.0f}/100). "
                    f"Consider a breath reset before continuing."
                ),
                confidence_score=avg_confidence,
                nervousness_score=avg_nervousness
            )
            new_events.append(event)

        # ── Rule 3: Positive words + negative emotion ─────────────────────
        verbal_positive = any(w in text_lower for w in POSITIVE_WORDS)
        negative_face = dominant_emotion in ["sad", "angry", "disgust", "fear"]
        if verbal_positive and negative_face and avg_trust < 45:
            event = ContradictionEvent(
                timestamp=timestamp,
                contradiction_type="ENTHUSIASM_MISMATCH",
                severity="medium",
                verbal_signal=self._extract_trigger(text_lower, POSITIVE_WORDS),
                facial_signal=f"emotion={dominant_emotion}, trust={avg_trust:.0f}",
                description=(
                    f"Your words sound positive but your face shows {dominant_emotion}. "
                    f"This reduces perceived authenticity and trust."
                ),
                confidence_score=avg_confidence,
                nervousness_score=avg_nervousness
            )
            new_events.append(event)

        # ── Rule 4: Uncertainty words + high nervousness ──────────────────
        verbal_uncertain = any(w in text_lower for w in UNCERTAINTY_WORDS)
        if verbal_uncertain and avg_nervousness > 65:
            event = ContradictionEvent(
                timestamp=timestamp,
                contradiction_type="DOUBLE_UNCERTAINTY",
                severity=self._severity(avg_nervousness, 65, 75, 88),
                verbal_signal=self._extract_trigger(text_lower, UNCERTAINTY_WORDS),
                facial_signal=f"nervousness={avg_nervousness:.0f}",
                description=(
                    f"Both your words and face signal uncertainty simultaneously. "
                    f"This compounds perceived lack of authority. "
                    f"Replace hedging language with more definitive statements."
                ),
                confidence_score=avg_confidence,
                nervousness_score=avg_nervousness
            )
            new_events.append(event)

        # ── Rule 5: Low eye contact during key claims ─────────────────────
        avg_eye_contact = sum(r.eye_contact_score for r in self.face_buffer) / len(self.face_buffer)
        verbal_claim = any(w in text_lower for w in CONFIDENCE_WORDS + POSITIVE_WORDS)
        if verbal_claim and avg_eye_contact < 35:
            event = ContradictionEvent(
                timestamp=timestamp,
                contradiction_type="GAZE_AVOIDANCE",
                severity="medium",
                verbal_signal=self._extract_trigger(text_lower, CONFIDENCE_WORDS + POSITIVE_WORDS),
                facial_signal=f"eye_contact={avg_eye_contact:.0f}%",
                description=(
                    f"You made a strong verbal claim but eye contact dropped to {avg_eye_contact:.0f}%. "
                    f"Maintain direct gaze when making key assertions — it signals conviction."
                ),
                confidence_score=avg_confidence,
                nervousness_score=avg_nervousness
            )
            new_events.append(event)

        self.events.extend(new_events)
        return new_events

    def get_report(self) -> ContradictionReport:
        high = sum(1 for e in self.events if e.severity == "high")
        medium = sum(1 for e in self.events if e.severity == "medium")
        low = sum(1 for e in self.events if e.severity == "low")

        if not self.events:
            summary = "No significant verbal-facial contradictions detected. Strong alignment."
        elif high > 2:
            summary = f"Multiple high-severity contradictions detected ({high}). Significant credibility gaps present."
        elif high > 0 or medium > 2:
            summary = f"Some contradictions detected. Focus on aligning verbal confidence with facial expression."
        else:
            summary = "Minor contradictions only. Overall communication alignment is good."

        return ContradictionReport(
            total_events=len(self.events),
            high_severity=high,
            medium_severity=medium,
            low_severity=low,
            events=self.events.copy(),
            summary=summary
        )

    def reset(self):
        self.events.clear()
        self.face_buffer.clear()

    def _severity(self, value: float, low_thresh: float,
                  med_thresh: float, high_thresh: float) -> str:
        if value >= high_thresh:
            return "high"
        elif value >= med_thresh:
            return "medium"
        return "low"

    def _extract_trigger(self, text: str, word_list: list) -> str:
        for word in word_list:
            if word in text:
                # Return surrounding context (up to 8 words)
                idx = text.find(word)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(word) + 20)
                return f'...{text[start:end]}...'
        return text[:50]

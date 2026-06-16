"""
Feedback Generator Module
Uses Groq (fast LLaMA inference) to generate specific, actionable coaching feedback.
Groq uses an OpenAI-compatible API — pip install groq
"""

import os
import time
import logging
from typing import List, Optional
from dotenv import load_dotenv

from core.face_analyzer import FaceAnalysisResult
from core.contradiction_engine import ContradictionEvent, ContradictionReport

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.1-8b-instant"   # fast + free tier; alternatives: llama-3.3-70b-versatile


class FeedbackGenerator:
    """
    Generates LLM-powered coaching feedback using Groq.
    Falls back to rule-based tips if API key is not set.
    """

    def __init__(self):
        self._client = None
        self._init_client()

    def _init_client(self):
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key or api_key == "your_groq_api_key_here":
            logger.warning("GROQ_API_KEY not set. Using rule-based feedback fallback.")
            return
        try:
            from groq import Groq
            self._client = Groq(api_key=api_key)
            logger.info(f"Groq client initialized (model: {GROQ_MODEL}).")
        except ImportError:
            # Fallback: use openai package pointed at Groq endpoint
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
                logger.info("Groq client initialized via openai-compat.")
            except ImportError:
                logger.error("Neither groq nor openai package found. Run: pip install groq")

    def generate_realtime_tip(
        self,
        face_result: FaceAnalysisResult,
        recent_transcript: str,
        contradiction: Optional[ContradictionEvent] = None
    ) -> str:
        """Short real-time coaching tip (1-2 sentences)."""
        if not self._client:
            return self._fallback_tip(face_result, contradiction)

        context = self._build_realtime_context(face_result, recent_transcript, contradiction)
        prompt = f"""You are an expert communication coach for high-stakes scenarios 
(military briefings, diplomatic negotiations, courtroom arguments, executive presentations).

Current speaker state:
{context}

Give ONE specific, actionable coaching tip in 1-2 sentences. 
Be direct and precise. No fluff. Focus on what to do RIGHT NOW.
Start with an action verb."""

        return self._call_llm(prompt, max_tokens=120)

    def generate_session_report(
        self,
        face_timeline: List[FaceAnalysisResult],
        full_transcript: str,
        contradiction_report: ContradictionReport,
        session_duration: float,
        context: str = "general presentation"
    ) -> str:
        """Comprehensive post-session coaching report."""
        if not self._client:
            return self._fallback_report(contradiction_report, session_duration)

        if face_timeline:
            avg_confidence = sum(r.confidence_score for r in face_timeline) / len(face_timeline)
            avg_nervousness = sum(r.nervousness_score for r in face_timeline) / len(face_timeline)
            avg_trust = sum(r.trust_score for r in face_timeline) / len(face_timeline)
            avg_eye_contact = sum(r.eye_contact_score for r in face_timeline) / len(face_timeline)
            emotion_counts = {}
            for r in face_timeline:
                emotion_counts[r.emotion] = emotion_counts.get(r.emotion, 0) + 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral"
        else:
            avg_confidence = avg_nervousness = avg_trust = avg_eye_contact = 50
            dominant_emotion = "neutral"

        contradiction_text = ""
        for i, event in enumerate(contradiction_report.events[:5], 1):
            t = time.strftime('%M:%S', time.gmtime(event.timestamp))
            contradiction_text += (
                f"\n{i}. [{t}] {event.contradiction_type} ({event.severity.upper()})\n"
                f"   Said: {event.verbal_signal}\n"
                f"   Face: {event.facial_signal}\n"
            )

        prompt = f"""You are an expert communication coach analyzing a {context} session.

SESSION METRICS (duration: {session_duration:.0f}s):
- Average Confidence Score: {avg_confidence:.1f}/100
- Average Nervousness Score: {avg_nervousness:.1f}/100
- Average Trust Score: {avg_trust:.1f}/100
- Average Eye Contact: {avg_eye_contact:.1f}%
- Dominant Emotion: {dominant_emotion}

VERBAL-FACIAL CONTRADICTIONS ({contradiction_report.total_events} total):
- High severity: {contradiction_report.high_severity}
- Medium severity: {contradiction_report.medium_severity}
{contradiction_text}

TRANSCRIPT EXCERPT:
{full_transcript[:400]}

Write a structured coaching report with:
1. OVERALL ASSESSMENT (2-3 sentences, honest)
2. TOP 3 STRENGTHS (specific, evidence-based)
3. TOP 3 AREAS TO IMPROVE (with exact techniques)
4. KEY CONTRADICTION ANALYSIS (most significant mismatch)
5. DRILL FOR NEXT SESSION (one specific practice exercise)

Be specific and actionable. Write like a world-class coach."""

        return self._call_llm(prompt, max_tokens=700)

    def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        try:
            # Works for both groq.Groq and openai.OpenAI (same interface)
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return self._fallback_tip_text()

    def _build_realtime_context(self, face, transcript, contradiction):
        lines = [
            f"- Confidence: {face.confidence_score}/100",
            f"- Nervousness: {face.nervousness_score}/100",
            f"- Trust: {face.trust_score}/100",
            f"- Eye Contact: {face.eye_contact_score}%",
            f"- Dominant Emotion: {face.emotion}",
            f"- Recent speech: '{transcript[-150:]}'" if transcript else "- No speech yet",
        ]
        if contradiction:
            lines.append(f"- CONTRADICTION: {contradiction.contradiction_type}")
            lines.append(f"  {contradiction.description}")
        return "\n".join(lines)

    def _fallback_tip(self, face, contradiction):
        if contradiction:
            tips = {
                "CREDIBILITY_GAP": "Take a slow breath, lower your shoulders, and restate that point with a deliberate pause before the key claim.",
                "STRESS_LEAK": "Pause for 2 seconds, reset your posture — the pause reads as authority, not weakness.",
                "ENTHUSIASM_MISMATCH": "Let your face match your words — allow a genuine micro-smile when stating positive outcomes.",
                "DOUBLE_UNCERTAINTY": "Replace hedging language with 'Based on current data...' to project authority.",
                "GAZE_AVOIDANCE": "Look directly at the camera when making your key claim — gaze signals conviction.",
            }
            return tips.get(contradiction.contradiction_type, "Maintain steady eye contact and slow your speech rate slightly.")
        if face.nervousness_score > 70:
            return "Nervousness is high — take a deliberate breath and slow your speech rate by 20%."
        if face.eye_contact_score < 40:
            return "Eye contact is low — look directly at your audience when making key points."
        if face.confidence_score > 70:
            return "Strong confidence signals — maintain this posture and pace."
        return "Keep your chin level and shoulders back to project authority."

    def _fallback_tip_text(self):
        return "Maintain steady eye contact and speak at a measured pace to project confidence."

    def _fallback_report(self, report: ContradictionReport, duration: float) -> str:
        return f"""SESSION REPORT (Groq unavailable — rule-based analysis)

Duration: {duration:.0f} seconds
Contradictions detected: {report.total_events}
  High severity: {report.high_severity}
  Medium severity: {report.medium_severity}
  Low severity: {report.low_severity}

Summary: {report.summary}

To enable AI coaching, ensure GROQ_API_KEY is set in .env
Get your free key at: https://console.groq.com"""

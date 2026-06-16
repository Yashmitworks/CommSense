"""
Session Manager
Coordinates all modules, manages session state, and handles data persistence.
"""

import json
import os
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional
from pathlib import Path

from core.face_analyzer import FaceAnalysisResult
from core.contradiction_engine import ContradictionEvent

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    session_id: str
    start_time: float
    end_time: Optional[float]
    context: str                          # e.g. "job interview", "military briefing"
    face_timeline: List[dict] = field(default_factory=list)
    transcript_segments: List[dict] = field(default_factory=list)
    contradiction_events: List[dict] = field(default_factory=list)
    llm_report: str = ""
    duration: float = 0.0


class SessionManager:
    def __init__(self, log_dir: str = "data/session_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionData] = None
        self._face_buffer: List[FaceAnalysisResult] = []

    def start_session(self, context: str = "general presentation") -> str:
        session_id = f"session_{int(time.time())}"
        self.current_session = SessionData(
            session_id=session_id,
            start_time=time.time(),
            end_time=None,
            context=context
        )
        self._face_buffer.clear()
        logger.info(f"Session started: {session_id}")
        return session_id

    def log_face_result(self, result: FaceAnalysisResult):
        if not self.current_session:
            return
        self._face_buffer.append(result)
        # Store every 5th frame to keep log size manageable
        if len(self._face_buffer) % 5 == 0:
            self.current_session.face_timeline.append({
                "t": round(result.timestamp - self.current_session.start_time, 2),
                "confidence": result.confidence_score,
                "nervousness": result.nervousness_score,
                "trust": result.trust_score,
                "eye_contact": result.eye_contact_score,
                "emotion": result.emotion
            })

    def log_transcript(self, text: str, timestamp: float):
        if not self.current_session:
            return
        self.current_session.transcript_segments.append({
            "t": round(timestamp - self.current_session.start_time, 2),
            "text": text
        })

    def log_contradiction(self, event: ContradictionEvent):
        if not self.current_session:
            return
        self.current_session.contradiction_events.append({
            "t": round(event.timestamp - self.current_session.start_time, 2),
            "type": event.contradiction_type,
            "severity": event.severity,
            "verbal": event.verbal_signal,
            "facial": event.facial_signal,
            "description": event.description
        })

    def end_session(self, llm_report: str = "") -> SessionData:
        if not self.current_session:
            raise RuntimeError("No active session.")
        self.current_session.end_time = time.time()
        self.current_session.duration = (
            self.current_session.end_time - self.current_session.start_time
        )
        self.current_session.llm_report = llm_report
        self._save_session()
        session = self.current_session
        self.current_session = None
        return session

    def get_face_buffer(self) -> List[FaceAnalysisResult]:
        return self._face_buffer.copy()

    def get_elapsed(self) -> float:
        if not self.current_session:
            return 0.0
        return time.time() - self.current_session.start_time

    def is_active(self) -> bool:
        return self.current_session is not None

    def _save_session(self):
        if not self.current_session:
            return
        path = self.log_dir / f"{self.current_session.session_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self.current_session), f, indent=2)
        logger.info(f"Session saved: {path}")

    def load_session(self, session_id: str) -> Optional[SessionData]:
        path = self.log_dir / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return SessionData(**data)

    def list_sessions(self) -> List[str]:
        return [p.stem for p in sorted(self.log_dir.glob("session_*.json"), reverse=True)]

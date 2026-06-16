"""
Speech Processor Module
Handles real-time speech-to-text transcription using SpeechRecognition
with Google STT (online) or Whisper (offline fallback).
Runs in a background thread and provides a transcript queue.
"""

import threading
import queue
import time
import speech_recognition as sr
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    timestamp: float
    text: str
    duration: float
    is_final: bool


class SpeechProcessor:
    """
    Background speech processor. Continuously listens to microphone
    and pushes TranscriptSegment objects to an output queue.
    """

    def __init__(self, use_whisper: bool = False, language: str = "en-US"):
        self.use_whisper = use_whisper
        self.language = language
        self.transcript_queue = queue.Queue()
        self.full_transcript = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8

        if use_whisper:
            self._load_whisper()

    def _load_whisper(self):
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded.")
        except ImportError:
            logger.warning("Whisper not available, falling back to Google STT.")
            self.use_whisper = False

    def start(self):
        """Start background listening thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.info("Speech processor started.")

    def stop(self):
        """Stop background listening."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Speech processor stopped.")

    def get_latest_transcript(self) -> Optional[TranscriptSegment]:
        """Non-blocking get from queue."""
        try:
            return self.transcript_queue.get_nowait()
        except queue.Empty:
            return None

    def get_full_transcript_text(self) -> str:
        """Return all transcript text joined."""
        return " ".join([seg.text for seg in self.full_transcript])

    def _listen_loop(self):
        """Main listening loop running in background thread."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Microphone calibrated.")

            while self._running:
                try:
                    start_time = time.time()
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    duration = time.time() - start_time

                    text = self._transcribe(audio)
                    if text:
                        segment = TranscriptSegment(
                            timestamp=start_time,
                            text=text.strip(),
                            duration=duration,
                            is_final=True
                        )
                        self.full_transcript.append(segment)
                        self.transcript_queue.put(segment)
                        logger.debug(f"Transcribed: {text[:60]}...")

                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    logger.error(f"Speech error: {e}")
                    time.sleep(1)

    def _transcribe(self, audio: sr.AudioData) -> str:
        """Transcribe audio using selected backend."""
        if self.use_whisper:
            return self._transcribe_whisper(audio)
        else:
            return self._transcribe_google(audio)

    def _transcribe_google(self, audio: sr.AudioData) -> str:
        try:
            return self.recognizer.recognize_google(audio, language=self.language)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            logger.error(f"Google STT error: {e}")
            return ""

    def _transcribe_whisper(self, audio: sr.AudioData) -> str:
        try:
            import whisper
            import numpy as np
            import io
            import soundfile as sf

            wav_data = audio.get_wav_data()
            audio_array, sample_rate = sf.read(io.BytesIO(wav_data))
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)
            audio_float = audio_array.astype(np.float32)

            result = self.whisper_model.transcribe(audio_float, language="en")
            return result.get("text", "")
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return ""

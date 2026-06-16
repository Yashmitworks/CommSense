# CommSense — AI Communication Coach
### Psycho-Linguistic Readiness Assessment for High-Stakes Communication

> **DRDO Internship Project**  
> Real-time verbal-facial contradiction detection for military officers, diplomats, lawyers, and executives.

---

## What It Does

CommSense watches your face while you speak and tells you things a human coach can't catch in real-time:

- **When your confidence expression contradicts your words** — "I'm certain about this" while your face shows fear
- **When stress is leaking through** — saying "no problem" while nervousness spikes
- **When you're losing perceived trust** — positive words paired with negative micro-expressions
- **When gaze avoidance undermines your claims** — eye contact drops during key assertions

Then an LLM (Gemini/GPT) gives you specific, actionable coaching tips with timestamps.

---

## Architecture

```
CommSense/
├── core/
│   ├── face_analyzer.py        # MediaPipe 468-landmark face mesh + DeepFace emotions
│   ├── gaze_tracker.py         # Eye contact %, blink rate, gaze direction
│   ├── speech_processor.py     # Real-time STT (Google or Whisper)
│   ├── contradiction_engine.py # Verbal-facial mismatch detection (5 rule types)
│   ├── feedback_generator.py   # LLM coaching tips (Gemini/OpenAI)
│   └── session_manager.py      # Session state, logging, persistence
├── ui/
│   ├── dashboard.py            # Streamlit real-time dashboard
│   └── report.py               # PDF report generator
├── data/
│   ├── session_logs/           # JSON session recordings
│   └── reports/                # PDF reports
├── main.py                     # Entry point
├── setup.py                    # Dependency installer
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
python setup.py
```

### 2. Add API key (for AI coaching)
```bash
# Edit .env file
GROK_API_KEY=your_key_here   # Get free at console.x.ai
```

### 3. Run
```bash
python main.py
```
Open **http://localhost:8501** in your browser.

---

## Contradiction Types Detected

| Type | Trigger |
|------|---------|
| `CREDIBILITY_GAP` | Confident words + anxious face |
| `STRESS_LEAK` | Calm words + high nervousness |
| `ENTHUSIASM_MISMATCH` | Positive words + negative emotion |
| `DOUBLE_UNCERTAINTY` | Hedging language + nervous face |
| `GAZE_AVOIDANCE` | Strong claim + low eye contact |

---

## Scores Explained

- **Confidence (0-100)** — Composite of emotion, brow tension, jaw set, eye contact
- **Nervousness (0-100)** — Fear/stress emotion + brow furrow + lip compression
- **Trust (0-100)** — Eye contact + genuine smile (Duchenne marker) + emotion alignment
- **Eye Contact (0-100)** — Iris position relative to eye center (MediaPipe iris landmarks)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Face Detection & Landmarks | MediaPipe Face Mesh (468 points) |
| Emotion Recognition | DeepFace (FER+ model) |
| Gaze Tracking | MediaPipe Iris Landmarks |
| Speech-to-Text | Google STT / OpenAI Whisper |
| LLM Coaching | xAI Grok (grok-3-mini / grok-3) |
| Dashboard | Streamlit + Plotly |
| PDF Reports | FPDF2 |

---

## Use Cases (DRDO Relevance)

- **Pre-briefing preparation** for military officers and intelligence analysts
- **Diplomatic negotiation training** — detect stress leakage before high-stakes talks
- **Interrogation readiness** — understand your own tells before being questioned
- **Command presence training** — build authority in high-pressure communication

---

## Notes

- Works offline except for LLM feedback (Gemini/OpenAI require internet)
- Whisper mode enables fully offline speech transcription
- Session data stored locally in `data/session_logs/`
- No data sent anywhere except LLM API calls (transcript + face scores only, no video)

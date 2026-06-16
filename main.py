"""
CommSense — AI Communication Coach
Redesigned HUD: Dark (purple/blue) and Light (grey) themes.
Press T to toggle theme. Press Q to stop.
"""

import cv2
import numpy as np
import time
import threading
import sys
import math
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from core.face_analyzer import FaceAnalyzer
from core.contradiction_engine import ContradictionEngine
from core.feedback_generator import FeedbackGenerator
from core.session_manager import SessionManager
from core.speech_processor import SpeechProcessor

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
FRAME_W      = 1280
FRAME_H      = 720
TIP_INTERVAL = 30
CONTEXT      = "job interview"

# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "dark": {
        "bg_panel":      (18, 10, 35),
        "bg_card":       (28, 16, 52),
        "bg_header":     (35, 20, 65),
        "header_line":   (120, 60, 220),
        "accent1":       (160, 80, 255),   # purple
        "accent2":       (80, 140, 255),   # blue
        "accent_green":  (60, 220, 140),
        "accent_red":    (80, 70, 255),
        "accent_amber":  (60, 180, 255),
        "text_primary":  (235, 225, 255),
        "text_dim":      (130, 110, 180),
        "border":        (60, 40, 110),
        "meter_bg":      (40, 25, 75),
        "footer_bg":     (22, 12, 42),
        "name":          "DARK",
    },
    "light": {
        "bg_panel":      (230, 228, 235),
        "bg_card":       (215, 212, 222),
        "bg_header":     (200, 196, 215),
        "header_line":   (140, 120, 180),
        "accent1":       (100, 70, 160),   # purple
        "accent2":       (60, 100, 200),   # blue
        "accent_green":  (30, 150, 80),
        "accent_red":    (180, 40, 60),
        "accent_amber":  (180, 120, 20),
        "text_primary":  (30, 20, 50),
        "text_dim":      (100, 90, 130),
        "border":        (170, 160, 200),
        "meter_bg":      (195, 190, 210),
        "footer_bg":     (210, 206, 220),
        "name":          "LIGHT",
    }
}

current_theme = "dark"


def T():
    return THEMES[current_theme]


# ── Drawing helpers ───────────────────────────────────────────────────────────

def grad_rect(img, x1, y1, x2, y2, c1, c2, vertical=True):
    """Gradient filled rectangle."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    h, w = roi.shape[:2]
    steps = h if vertical else w
    for i in range(steps):
        t = i / max(steps - 1, 1)
        color = tuple(int(c1[j] * (1 - t) + c2[j] * t) for j in range(3))
        if vertical:
            cv2.line(img, (x1, y1 + i), (x2, y1 + i), color, 1)
        else:
            cv2.line(img, (x1 + i, y1), (x1 + i, y2), color, 1)


def glow_line(img, x1, y1, x2, y2, color, t=1):
    soft = tuple(max(0, c // 4) for c in color)
    cv2.line(img, (x1, y1), (x2, y2), soft, t + 3, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x2, y2), color, t, cv2.LINE_AA)


def corner_brackets(img, x1, y1, x2, y2, color, L=22, t=2):
    for (px, py), (dx, dy) in zip(
        [(x1,y1),(x2,y1),(x1,y2),(x2,y2)],
        [(1,1),(-1,1),(1,-1),(-1,-1)]
    ):
        cv2.line(img, (px, py), (px + dx*L, py), color, t, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy*L), color, t, cv2.LINE_AA)


def put_text_centered(img, text, cx, y, font, scale, color, thickness=1):
    tw = cv2.getTextSize(text, font, scale, thickness)[0][0]
    cv2.putText(img, text, (cx - tw // 2, y), font, scale, color, thickness, cv2.LINE_AA)


def wrap_text(text, font, scale, thickness, max_w):
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        if cv2.getTextSize(test, font, scale, thickness)[0][0] <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def score_color(value, invert=False):
    v = (100 - value) if invert else value
    th = T()
    if v >= 65:
        return th["accent_green"]
    elif v >= 38:
        return th["accent_amber"]
    return th["accent_red"]


def draw_arc_meter(img, cx, cy, r, value, label, invert=False):
    """Draw arc meter with label ABOVE the arc and value inside."""
    th = T()
    col = score_color(value, invert=invert)

    START, SWEEP = 150, 240

    # Background arc
    cv2.ellipse(img, (cx, cy), (r, r), 0, START, START + SWEEP,
                th["meter_bg"], 9, cv2.LINE_AA)
    # Value arc
    end_ang = START + int(SWEEP * max(0, min(value, 100)) / 100)
    if end_ang > START:
        cv2.ellipse(img, (cx, cy), (r, r), 0, START, end_ang,
                    col, 9, cv2.LINE_AA)
    # Glow dot at arc end
    rad = math.radians(end_ang)
    dot_x = int(cx + r * math.cos(rad))
    dot_y = int(cy + r * math.sin(rad))
    cv2.circle(img, (dot_x, dot_y), 5, col, -1, cv2.LINE_AA)

    # Value number inside arc
    val_str = f"{int(value)}"
    put_text_centered(img, val_str, cx, cy + 8,
                      cv2.FONT_HERSHEY_DUPLEX, 0.65, th["text_primary"], 1)

    # Label ABOVE the arc (clear, never overlapping camera)
    put_text_centered(img, label, cx, cy - r - 10,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.38, th["text_primary"], 1)


# ── Main HUD ──────────────────────────────────────────────────────────────────

PANEL_W  = 380
HEADER_H = 56
FOOTER_H = 50

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def draw_section_header(img, x, y, w, label, accent):
    """Draw a small labelled section divider."""
    cv2.putText(img, label, (x, y), FONT, 0.36,
                accent, 1, cv2.LINE_AA)
    glow_line(img, x, y + 5, x + w, y + 5, accent)


def draw_dashboard(frame, face_result, contradictions, tip, elapsed, transcript):
    th    = T()
    h, fw = frame.shape[:2]
    CAM_W = fw - PANEL_W

    # ══════════════════════════════════════════════════════════════
    # 1. PANEL BACKGROUND
    # ══════════════════════════════════════════════════════════════
    panel_overlay = frame.copy()
    cv2.rectangle(panel_overlay, (CAM_W, 0), (fw, h), th["bg_panel"], -1)
    cv2.addWeighted(panel_overlay, 0.95, frame, 0.05, 0, frame)

    # Panel left border glow
    glow_line(frame, CAM_W, 0, CAM_W, h, th["border"])

    # ══════════════════════════════════════════════════════════════
    # 2. HEADER BAR
    # ══════════════════════════════════════════════════════════════
    grad_rect(frame, 0, 0, fw, HEADER_H,
              th["bg_header"],
              tuple(max(0, c - 15) for c in th["bg_panel"]))
    glow_line(frame, 0, HEADER_H, fw, HEADER_H, th["header_line"])

    # Blinking REC dot
    rec_on = int(elapsed * 2) % 2 == 0
    cv2.circle(frame, (18, HEADER_H // 2), 7,
               th["accent_red"] if rec_on else th["meter_bg"], -1, cv2.LINE_AA)
    cv2.putText(frame, "REC", (30, HEADER_H // 2 + 5),
                FONT, 0.38, th["text_dim"], 1, cv2.LINE_AA)

    # App name
    cv2.putText(frame, "CommSense", (62, 34),
                FONT_BOLD, 0.88, th["text_primary"], 1, cv2.LINE_AA)
    cv2.putText(frame, "AI COMMUNICATION COACH", (62, 48),
                FONT, 0.30, th["accent1"], 1, cv2.LINE_AA)

    # Scenario badge (centered)
    badge = f"  {CONTEXT.upper()}  "
    btw   = cv2.getTextSize(badge, FONT, 0.38, 1)[0][0]
    bx    = fw // 2 - btw // 2
    cv2.rectangle(frame, (bx - 4, 14), (bx + btw + 4, 42),
                  tuple(c // 3 for c in th["accent1"]), -1)
    cv2.rectangle(frame, (bx - 4, 14), (bx + btw + 4, 42), th["accent1"], 1)
    cv2.putText(frame, badge, (bx, 33), FONT, 0.38, th["accent1"], 1, cv2.LINE_AA)

    # Timer (right)
    mins, secs = divmod(int(elapsed), 60)
    timer = f"{mins:02d}:{secs:02d}"
    ttw   = cv2.getTextSize(timer, FONT_BOLD, 0.82, 1)[0][0]
    cv2.putText(frame, timer, (fw - ttw - 14, 38),
                FONT_BOLD, 0.82, th["text_primary"], 1, cv2.LINE_AA)

    # Theme indicator (top-right small)
    cv2.putText(frame, f"[T] {th['name']} MODE", (fw - 120, 52),
                FONT, 0.28, th["text_dim"], 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════
    # 3. CAMERA REGION CORNER BRACKETS
    # ══════════════════════════════════════════════════════════════
    m = 6
    corner_brackets(frame, m, HEADER_H + m, CAM_W - m, h - FOOTER_H - m,
                    th["accent1"], L=26, t=2)

    # ══════════════════════════════════════════════════════════════
    # 4. RIGHT PANEL CONTENT
    # ══════════════════════════════════════════════════════════════
    px  = CAM_W + 12
    pw  = PANEL_W - 24
    pcx = CAM_W + PANEL_W // 2  # panel center x

    cy  = HEADER_H + 20  # current y cursor

    # ── 4a. METRICS section header ────────────────────────────────
    draw_section_header(frame, px, cy, pw, "PERFORMANCE METRICS", th["accent1"])
    cy += 16

    # ── 4b. Arc meters in 2×2 grid ────────────────────────────────
    ARC_R   = 40
    col_gap = PANEL_W // 2
    # Row spacing: label(above) + arc diameter + bottom padding
    row_h   = ARC_R * 2 + 30   # room for label above + arc + gap

    if face_result and face_result.face_detected:
        vals = [face_result.confidence_score, face_result.trust_score,
                face_result.nervousness_score, face_result.eye_contact_score]
    else:
        vals = [0, 0, 0, 0]

    metrics = [
        ("CONFIDENCE",  vals[0], False),
        ("TRUST",       vals[1], False),
        ("NERVOUSNESS", vals[2], True),
        ("EYE CONTACT", vals[3], False),
    ]

    for i, (lbl, val, inv) in enumerate(metrics):
        row = i // 2
        col = i % 2
        cx_arc = CAM_W + col_gap // 2 + col * col_gap
        cy_arc = cy + ARC_R + 18 + row * (row_h + 8)
        draw_arc_meter(frame, cx_arc, cy_arc, ARC_R, val, lbl, invert=inv)

    cy += 2 * (row_h + 8) + 10

    # ── 4c. Emotion badge ─────────────────────────────────────────
    emo_map = {
        "happy":   th["accent_green"],  "neutral": th["text_dim"],
        "sad":     th["accent2"],       "angry":   th["accent_red"],
        "fear":    th["accent_amber"],  "surprise":(60, 220, 220),
        "disgust": th["accent1"],       "unknown": th["text_dim"],
    }
    emo     = (face_result.emotion if face_result and face_result.face_detected else "unknown")
    ec      = emo_map.get(emo, th["text_dim"])
    emo_lbl = f"  EMOTION :  {emo.upper()}  "
    etw     = cv2.getTextSize(emo_lbl, FONT, 0.44, 1)[0][0]
    ex      = pcx - etw // 2
    cv2.rectangle(frame, (ex - 4, cy - 2), (ex + etw + 4, cy + 22),
                  tuple(c // 4 for c in ec), -1)
    cv2.rectangle(frame, (ex - 4, cy - 2), (ex + etw + 4, cy + 22), ec, 1)
    cv2.putText(frame, emo_lbl, (ex, cy + 16), FONT, 0.44, ec, 1, cv2.LINE_AA)
    cy += 34


    # ── 4d. VERBAL-FACIAL ALERTS section ─────────────────────────
    glow_line(frame, px, cy, px + pw, cy, th["border"])
    cy += 10
    draw_section_header(frame, px, cy, pw,
                        "VERBAL-FACIAL CONTRADICTIONS", th["accent2"])
    cy += 8

    # Explanation line (so user knows what this section is)
    cv2.putText(frame,
                "Detected when words contradict face signals",
                (px, cy + 10), FONT, 0.31, th["text_dim"], 1, cv2.LINE_AA)
    cy += 20

    # Alert count badge
    cnt     = len(contradictions)
    cnt_col = th["accent_red"] if cnt > 0 else th["text_dim"]
    cv2.circle(frame, (px + pw - 10, cy - 12), 13, cnt_col, 1, cv2.LINE_AA)
    put_text_centered(frame, str(cnt), px + pw - 10, cy - 7,
                      FONT, 0.42, cnt_col, 1)

    sev_cols = {"high": th["accent_red"], "medium": th["accent_amber"], "low": th["accent2"]}
    shown    = contradictions[-3:] if contradictions else []

    if not shown:
        cv2.putText(frame, "None detected — communication aligned",
                    (px, cy + 14), FONT, 0.35, th["text_dim"], 1, cv2.LINE_AA)
        cy += 24
    else:
        for ev in reversed(shown):
            col    = sev_cols.get(ev.severity, th["text_dim"])
            card_h = 44
            # Card bg
            grad_rect(frame, px, cy, px + pw, cy + card_h,
                      tuple(c // 5 for c in col), th["bg_card"])
            cv2.rectangle(frame, (px, cy), (px + pw, cy + card_h), th["border"], 1)
            # Severity stripe
            cv2.rectangle(frame, (px, cy), (px + 5, cy + card_h), col, -1)
            # Type label
            type_lbl = ev.contradiction_type.replace("_", " ")
            cv2.putText(frame, type_lbl, (px + 10, cy + 16),
                        FONT, 0.40, th["text_primary"], 1, cv2.LINE_AA)
            # Severity + short description
            desc_short = ev.description[:45] + "..." if len(ev.description) > 45 else ev.description
            cv2.putText(frame, f"{ev.severity.upper()}  |  {desc_short}",
                        (px + 10, cy + 32), FONT, 0.30, th["text_dim"], 1, cv2.LINE_AA)
            cy += card_h + 4

    # ── 4e. AI COACHING TIP ───────────────────────────────────────
    glow_line(frame, px, cy + 4, px + pw, cy + 4, th["border"])
    cy += 14
    draw_section_header(frame, px, cy, pw, "AI COACHING TIP", th["accent_green"])
    cy += 12

    tip_box_h = max(h - FOOTER_H - cy - 8, 50)
    grad_rect(frame, px, cy, px + pw, cy + tip_box_h,
              tuple(c // 5 for c in th["accent_green"]), th["bg_card"])
    cv2.rectangle(frame, (px, cy), (px + pw, cy + tip_box_h), th["accent_green"], 1)

    if tip:
        tip_lines = wrap_text(tip, FONT, 0.37, 1, pw - 14)
        for i, ln in enumerate(tip_lines[:5]):
            cv2.putText(frame, ln, (px + 7, cy + 16 + i * 20),
                        FONT, 0.37, th["text_primary"], 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "AI tip generates every 30 sec...",
                    (px + 7, cy + 20), FONT, 0.35, th["text_dim"], 1, cv2.LINE_AA)
        cv2.putText(frame, "Speak naturally to trigger analysis.",
                    (px + 7, cy + 38), FONT, 0.33, th["text_dim"], 1, cv2.LINE_AA)

    # ══════════════════════════════════════════════════════════════
    # 5. FOOTER
    # ══════════════════════════════════════════════════════════════
    grad_rect(frame, 0, h - FOOTER_H, fw, h,
              th["footer_bg"], th["bg_panel"])
    glow_line(frame, 0, h - FOOTER_H, fw, h - FOOTER_H, th["border"])

    cv2.putText(frame, "TRANSCRIPT", (12, h - FOOTER_H + 16),
                FONT, 0.32, th["text_dim"], 1, cv2.LINE_AA)
    short = (transcript[-110:] if transcript else "Listening for speech  —  speak naturally")
    cv2.putText(frame, short, (12, h - FOOTER_H + 36),
                FONT, 0.40, th["text_primary"], 1, cv2.LINE_AA)

    cv2.putText(frame, "[Q] Stop & Report    [T] Toggle Theme",
                (fw - 295, h - FOOTER_H + 28), FONT, 0.33, th["text_dim"], 1, cv2.LINE_AA)

    return frame


# ── Report window ─────────────────────────────────────────────────────────────

def generate_report_window(report: str):
    th      = T()
    lines   = report.split("\n")
    line_h  = 24
    win_w   = 960
    win_h   = min(880, len(lines) * line_h + 90)

    img = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    grad_rect(img, 0, 0, win_w, win_h, th["bg_panel"], th["bg_card"])

    # Header
    grad_rect(img, 0, 0, win_w, 58, th["bg_header"], th["bg_panel"])
    glow_line(img, 0, 58, win_w, 58, th["header_line"])
    cv2.putText(img, "CommSense  —  Session Coaching Report",
                (20, 38), FONT_BOLD, 0.85, th["text_primary"], 1, cv2.LINE_AA)

    corner_brackets(img, 8, 66, win_w - 8, win_h - 8, th["accent1"], 20)

    y = 78
    for line in lines:
        if y > win_h - 24:
            break
        heading = any(line.startswith(p) for p in
                      ("1.","2.","3.","4.","5.","OVERALL","TOP","KEY","DRILL","SESSION"))
        color = th["accent_green"] if heading else th["text_primary"]
        cv2.putText(img, line[:118], (20, y), FONT, 0.42, color, 1, cv2.LINE_AA)
        y += line_h

    cv2.putText(img, "Press any key to close", (20, win_h - 10),
                FONT, 0.40, th["text_dim"], 1, cv2.LINE_AA)

    cv2.imshow("CommSense — Report", img)
    cv2.waitKey(0)
    cv2.destroyWindow("CommSense — Report")

    path = Path("data/reports") / f"report_{int(time.time())}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(f"\n  Report saved: {path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    global current_theme

    print("=" * 60)
    print("  CommSense — AI Communication Coach")
    print("  DRDO Internship Project")
    print("=" * 60)
    print(f"\n  Scenario : {CONTEXT.title()}")
    print(f"  Press Q to stop  |  Press T to toggle theme\n")

    for d in ["data/session_logs", "data/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print("  Loading face analyzer...")
    analyzer = FaceAnalyzer()
    ce       = ContradictionEngine()
    fb       = FeedbackGenerator()
    sm       = SessionManager()
    sp       = SpeechProcessor(use_whisper=False)

    print("  Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("  ERROR: Cannot open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    sm.start_session(CONTEXT)
    sp.start()
    print("  Session started. Camera window opening...\n")

    face_results   = []
    contradictions = []
    transcript     = ""
    latest_tip     = ""
    last_tip_time  = time.time()
    session_start  = time.time()

    cv2.namedWindow("CommSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CommSense", FRAME_W, FRAME_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        elapsed = time.time() - session_start

        face_result = analyzer.analyze_frame(frame)
        sm.log_face_result(face_result)
        ce.update_face(face_result)
        if face_result.face_detected:
            face_results.append(face_result)

        seg = sp.get_latest_transcript()
        if seg:
            transcript += " " + seg.text
            sm.log_transcript(seg.text, seg.timestamp)
            new_c = ce.analyze_transcript(seg.text, seg.timestamp)
            for c in new_c:
                sm.log_contradiction(c)
            contradictions.extend(new_c)
            if new_c:
                print(f"  [ALERT] {new_c[0].contradiction_type} ({new_c[0].severity})")

        now = time.time()
        if now - last_tip_time > TIP_INTERVAL:
            last_tip_time = now
            _f, _t = face_result, transcript[-200:]
            def fetch_tip(f=_f, t=_t):
                nonlocal latest_tip
                latest_tip = fb.generate_realtime_tip(f, t, None)
            threading.Thread(target=fetch_tip, daemon=True).start()

        frame = draw_dashboard(frame, face_result, contradictions,
                               latest_tip, elapsed, transcript)
        cv2.imshow("CommSense", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("\n  Stopping session...")
            break
        if key in (ord('t'), ord('T')):
            current_theme = "light" if current_theme == "dark" else "dark"
            print(f"  Theme: {current_theme}")
        if cv2.getWindowProperty("CommSense", cv2.WND_PROP_VISIBLE) < 1:
            break

    sp.stop()
    cap.release()
    cv2.destroyAllWindows()

    print("  Generating AI coaching report...")
    report = fb.generate_session_report(
        face_timeline=face_results,
        full_transcript=transcript,
        contradiction_report=ce.get_report(),
        session_duration=time.time() - session_start,
        context=CONTEXT
    )
    sm.end_session(llm_report=report)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    analyzer.release()
    generate_report_window(report)


if __name__ == "__main__":
    main()

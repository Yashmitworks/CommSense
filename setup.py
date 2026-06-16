"""
Quick setup script — installs dependencies and verifies the environment.
Run: python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd, desc=""):
    print(f"\n{'-'*50}")
    if desc:
        print(f"  {desc}")
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  CommSense Setup")
    print("=" * 60)

    # Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    # Install requirements
    req_path = Path(__file__).parent / "requirements.txt"
    success = run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
        "Installing requirements"
    )

    if not success:
        print("\n⚠  Some packages failed. Try installing manually:")
        print("   pip install opencv-python mediapipe deepface streamlit plotly")
        print("   pip install SpeechRecognition pyaudio google-generativeai openai fpdf2")

    # Create .env if missing
    env_example = Path(".env.example")
    env_file = Path(".env")
    if env_example.exists() and not env_file.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("\n✓  Created .env from .env.example")
        print("   Edit .env and add your GROQ_API_KEY (free at console.groq.com)")

    # Create data dirs
    for d in ["data/session_logs", "data/reports", "models"]:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("\n✓  Data directories created")

    # Verify key imports
    print("\n" + "-" * 50)
    print("  Verifying imports...")
    checks = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("deepface", "DeepFace"),
        ("streamlit", "Streamlit"),
        ("plotly", "Plotly"),
        ("speech_recognition", "SpeechRecognition"),
    ]
    all_ok = True
    for module, name in checks:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} — NOT INSTALLED")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  Setup complete! Run: python main.py")
        print("  Don't forget to add your GROK_API_KEY to .env")
        print("  Get your key at: https://console.x.ai")
    else:
        print("  Some packages missing. Check errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""Settings, read from the environment (and from .env during local development)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SR = 16000                     # sample rate used everywhere on the server
VAD_FRAME = 512                # Silero VAD window: 32 ms at 16 kHz
FRAME_S = VAD_FRAME / SR

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_MODEL", "").strip()   # empty = pick from the account's model list
GROQ_URL = "https://api.groq.com/openai/v1"

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "af_heart")
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.0"))
KOKORO_BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
KOKORO_MODEL_FILE = os.environ.get("KOKORO_MODEL_FILE", "kokoro-v1.0.onnx")
KOKORO_VOICES_FILE = "voices-v1.0.bin"

# A fine-tuned model from Kaggle goes in models/emotion; until then a public model is used.
EMOTION_LOCAL_DIR = MODELS_DIR / "emotion"
EMOTION_FALLBACK = os.environ.get("EMOTION_MODEL", "superb/wav2vec2-base-superb-er")
EMOTION_WINDOW_S = 3.0

# Turn-taking thresholds (conversation with the coach).
SPEECH_PROB = 0.5              # VAD probability counted as speech
BARGE_IN_MS = 300              # speech needed to interrupt the coach
TURN_START_MS = 160            # speech needed to open a user turn when the coach is silent
TURN_END_MS = 700              # silence that closes a user turn
# The talk itself ends on the code word, not on silence. These only cut it into phrases for live STT.
PHRASE_END_MS = 500
PHRASE_MAX_S = 12.0

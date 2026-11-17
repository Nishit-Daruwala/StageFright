"""Text-to-speech with Kokoro (runs locally through ONNX)."""

import io
import wave

import httpx
import numpy as np

from . import config


def wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    """Mono float samples in [-1, 1] as a 16-bit WAV file in memory."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _fetch(name: str) -> str:
    """Download a Kokoro model file on first use."""
    path = config.MODELS_DIR / name
    if not path.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", f"{config.KOKORO_BASE_URL}/{name}", follow_redirects=True, timeout=None) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(path)
    return str(path)


class Speaker:
    def __init__(self):
        from kokoro_onnx import Kokoro
        self.kokoro = Kokoro(_fetch(config.KOKORO_MODEL_FILE), _fetch(config.KOKORO_VOICES_FILE))

    def synth(self, text: str) -> bytes:
        """One sentence of text as WAV bytes."""
        samples, sr = self.kokoro.create(text, voice=config.KOKORO_VOICE, speed=config.KOKORO_SPEED, lang="en-us")
        return wav_bytes(np.asarray(samples, dtype=np.float32), sr)

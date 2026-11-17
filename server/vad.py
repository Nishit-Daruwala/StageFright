"""Streaming voice activity detection (Silero) and turn-taking on top of it."""

import numpy as np

from . import config


class StreamingVAD:
    """Speech probability for each 32 ms frame of live audio. Keeps state between frames."""

    def __init__(self):
        import torch
        from silero_vad import load_silero_vad
        self._torch = torch
        self.model = load_silero_vad(onnx=True)

    def reset(self) -> None:
        self.model.reset_states()

    def prob(self, frame: np.ndarray) -> float:
        return float(self.model(self._torch.from_numpy(frame), config.SR).item())


class TurnDetector:
    """Decides when the user starts and stops speaking, from per-frame speech probabilities.

    start_ms of continuous speech opens a turn; end_ms of continuous silence closes it.
    A short pre-roll is kept so the first syllable is not cut off.
    """

    PRE_ROLL_FRAMES = 20          # about 0.6 s

    def __init__(self, start_ms: int, end_ms: int):
        self.start_frames = max(1, round(start_ms / 1000 / config.FRAME_S))
        self.end_frames = max(1, round(end_ms / 1000 / config.FRAME_S))
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.run = 0
        self.silence = 0
        self.frames: list[np.ndarray] = []
        self.pre: list[np.ndarray] = []

    def feed(self, frame: np.ndarray, prob: float) -> str | None:
        """Returns "start" when a turn opens, "end" when it closes, else None."""
        speech = prob >= config.SPEECH_PROB
        if not self.active:
            self.pre.append(frame)
            del self.pre[:-self.PRE_ROLL_FRAMES]
            self.run = self.run + 1 if speech else 0
            if self.run >= self.start_frames:
                self.active, self.silence = True, 0
                self.frames = list(self.pre)
                return "start"
            return None
        self.frames.append(frame)
        self.silence = 0 if speech else self.silence + 1
        if self.silence >= self.end_frames:
            return "end"
        return None

    def take(self) -> np.ndarray:
        """Audio of the finished turn; resets the detector."""
        audio = np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)
        self.reset()
        return audio

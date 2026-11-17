"""Loads the speech models once, in the background, and reports their status."""

import asyncio
import logging

log = logging.getLogger("stagefright")


class Models:
    def __init__(self):
        self.stt = None
        self.tts = None
        self.emotion = None
        self.status = {"vad": "loading", "stt": "loading", "tts": "loading", "emotion": "loading"}

    @property
    def ready(self) -> bool:
        """The talk needs VAD and STT. Voice and emotion are reported but do not block."""
        return self.status["vad"] == "ready" and self.status["stt"] == "ready"

    @property
    def settled(self) -> bool:
        return all(v != "loading" for v in self.status.values())

    def _load(self, name: str, factory) -> None:
        try:
            obj = factory()
            if name != "vad":
                setattr(self, name, obj)
            self.status[name] = "ready"
            log.info("%s ready", name)
        except Exception as exc:                      # a missing optional model must not stop the app
            self.status[name] = f"error: {exc}"
            log.exception("%s failed to load", name)

    async def load_all(self) -> None:
        def vad():
            from .vad import StreamingVAD
            return StreamingVAD()

        def stt():
            from .stt import Transcriber
            return Transcriber()

        def tts():
            from .tts import Speaker
            return Speaker()

        def emotion():
            from .emotion import EmotionModel
            return EmotionModel()

        await asyncio.gather(*(asyncio.to_thread(self._load, n, f)
                               for n, f in [("vad", vad), ("stt", stt), ("tts", tts), ("emotion", emotion)]))

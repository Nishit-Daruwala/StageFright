"""One live coaching session: the state machine that ties audio, analysis and the coach together.

States
  idle       waiting for the speaker to press Start
  talk       recording the talk; live transcript; ends on the code word (or the Finish button)
  analysing  computing the report
  speaking   the coach is talking; the mic stays open so the speaker can interrupt
  listening  waiting for, or recording, the speaker's question
  thinking   transcribing the question and starting the reply
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket

from . import agent, config, dsp, metrics
from .models import Models
from .stt import find_code_word
from .tts import wav_bytes
from .vad import StreamingVAD, TurnDetector

log = logging.getLogger("stagefright")
SR = config.SR


def _plain(value):
    """Lets json.dumps handle NumPy scalars."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


class AudioBuffer:
    """Growing mono float32 recording."""

    def __init__(self):
        self.data = np.zeros(SR * 60, dtype=np.float32)
        self.n = 0

    def append(self, x: np.ndarray) -> None:
        if self.n + len(x) > len(self.data):
            self.data = np.concatenate([self.data, np.zeros(len(self.data), dtype=np.float32)])
        self.data[self.n:self.n + len(x)] = x
        self.n += len(x)

    def slice(self, a: int, b: int) -> np.ndarray:
        return self.data[max(0, a):min(b, self.n)].copy()


@dataclass
class Turn:
    """One reply from the coach, as an ordered list of items sent to the browser."""
    id: int
    items: list[dict] = field(default_factory=list)   # {id, kind, text, status: queued|playing|played}
    generated: bool = False
    task: asyncio.Task | None = None


class Session:
    def __init__(self, ws: WebSocket, models: Models):
        self.ws = ws
        self.m = models
        self.vad: StreamingVAD | None = None
        self.send_lock = asyncio.Lock()
        self.state = "idle"
        self.pending = np.zeros(0, dtype=np.float32)

        self.talk = AudioBuffer()
        self.talk_gen = 0                 # increases with every new talk; stale workers check it
        self.talk_done = False
        self.talk_end = 0
        self.words: list[dict] = []
        self.emotions: list[dict] = []
        self.stt_queue: asyncio.Queue = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self.in_phrase = False
        self.phrase_start = 0
        self.last_phrase_end = 0
        self.silence = 0
        self.frame_count = 0

        self.report: dict | None = None
        self.previous: dict | None = None
        self.messages: list[dict] = []
        self.turn: Turn | None = None
        self.turn_seq = 0
        self.unsaid = ""
        self.listen = TurnDetector(config.TURN_START_MS, config.TURN_END_MS)
        self.background: set[asyncio.Task] = set()

    # ------------------------------------------------------------ transport

    async def send(self, obj: dict) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj, default=_plain))

    async def send_with_audio(self, header: dict, audio: bytes) -> None:
        """Header and its audio are sent back to back so the browser can pair them."""
        async def pair():
            async with self.send_lock:
                await self.ws.send_text(json.dumps(header))
                await self.ws.send_bytes(audio)
        await asyncio.shield(pair())

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.send({"type": "state", "state": state})

    def spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self.background.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self.background.discard(task)
        if not task.cancelled() and task.exception():
            log.error("session task failed", exc_info=task.exception())

    async def close(self) -> None:
        for task in list(self.background):
            task.cancel()

    # ------------------------------------------------------------ incoming

    async def on_event(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "start":
            await self.start_talk()
        elif kind == "stop" and self.state == "talk":
            self._close_phrase()
            await self.stt_queue.put(None)
        elif kind in ("playing", "played") and self.turn and msg.get("turn") == self.turn.id:
            idx = msg.get("id", -1)
            if 0 <= idx < len(self.turn.items):
                self.turn.items[idx]["status"] = kind
        elif kind == "turn_done" and self.turn and msg.get("turn") == self.turn.id:
            await self._finish_turn()

    async def on_audio(self, chunk: bytes) -> None:
        if self.vad is None:
            return
        samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= config.VAD_FRAME:
            frame, self.pending = self.pending[:config.VAD_FRAME], self.pending[config.VAD_FRAME:]
            if self.state in ("talk", "speaking", "listening"):
                await self._on_frame(frame, self.vad.prob(frame))

    async def _on_frame(self, frame: np.ndarray, prob: float) -> None:
        if self.state == "talk":
            await self._talk_frame(frame, prob)
        elif self.state == "speaking":
            if self.listen.feed(frame, prob) == "start":
                await self._barge_in()
        elif self.state == "listening":
            event = self.listen.feed(frame, prob)
            if event == "start":
                await self.send({"type": "user_speaking"})
            elif event == "end":
                audio = self.listen.take()
                await self.set_state("thinking")
                self.spawn(self._answer(audio))

    # ------------------------------------------------------------ the talk

    async def start_talk(self) -> None:
        if not self.m.ready:
            await self.send({"type": "notice", "text": "Models are still loading."})
            return
        await self._cancel_turn()
        for task in self.workers:
            task.cancel()
        if self.vad is None:
            self.vad = await asyncio.to_thread(StreamingVAD)
        self.vad.reset()
        self.talk_gen += 1
        self.talk = AudioBuffer()
        self.talk_done, self.talk_end = False, 0
        self.words, self.emotions = [], []
        self.stt_queue = asyncio.Queue()
        self.in_phrase, self.phrase_start, self.last_phrase_end = False, 0, 0
        self.silence = self.frame_count = 0
        self.unsaid = ""
        self.workers = [self.spawn(self._stt_worker(self.talk_gen))]
        if self.m.emotion:
            self.workers.append(self.spawn(self._emotion_worker(self.talk_gen)))
        await self.set_state("talk")

    async def _talk_frame(self, frame: np.ndarray, prob: float) -> None:
        self.talk.append(frame)
        self.frame_count += 1
        speech = prob >= config.SPEECH_PROB
        n = self.talk.n
        if not self.in_phrase:
            if speech:
                self.in_phrase, self.silence = True, 0
                self.phrase_start = max(self.last_phrase_end, n - 10 * config.VAD_FRAME)
        else:
            self.silence = 0 if speech else self.silence + 1
            too_long = (n - self.phrase_start) / SR >= config.PHRASE_MAX_S
            if self.silence * config.FRAME_S * 1000 >= config.PHRASE_END_MS or too_long:
                self._close_phrase()
        if self.frame_count % 16 == 0:
            await self._send_meter()

    def _close_phrase(self) -> None:
        """Hand the phrase just spoken to the transcription queue."""
        if self.in_phrase:
            # Keep only about 130 ms of the trailing silence; Whisper invents words on longer silence.
            end = self.talk.n - max(0, self.silence - 4) * config.VAD_FRAME
            self.stt_queue.put_nowait((self.phrase_start, end))
            self.last_phrase_end = end
            self.in_phrase = False

    async def _send_meter(self) -> None:
        n = self.talk.n
        recent = self.talk.slice(n - SR // 2, n).astype(np.float64)
        level = float(dsp.rms_db(recent[None, :])[0])
        f0 = dsp.autocorr_f0(recent, SR)
        voiced = f0[f0 > 0]
        now = n / SR
        span = min(now, 15.0)
        wpm = round(sum(w["start"] > now - 15.0 for w in self.words) / (span / 60)) if span > 3 else None
        await self.send({"type": "meter", "t": round(now, 2), "level": round(level, 1),
                         "f0": round(float(np.median(voiced))) if len(voiced) >= 5 else None, "wpm": wpm})

    async def _stt_worker(self, gen: int) -> None:
        """Transcribes each phrase as it ends and watches for the code word."""
        while True:
            item = await self.stt_queue.get()
            if gen != self.talk_gen or self.state != "talk":
                return
            if item is None:                                   # Finish button
                await self._finish_talk(None)
                return
            start, end = item
            words = await asyncio.to_thread(self.m.stt.transcribe, self.talk.slice(start, end), start / SR)
            if gen != self.talk_gen or self.state != "talk":
                return
            self.words += words
            code = find_code_word(self.words)
            shown = self.words[:code] if code is not None else self.words
            await self.send({"type": "transcript", "words": shown})
            if code is not None:
                await self._finish_talk(code)
                return

    async def _emotion_worker(self, gen: int) -> None:
        """Classifies emotion window by window while the talk is still going on."""
        win = int(config.EMOTION_WINDOW_S * SR)
        k = 0
        while gen == self.talk_gen:
            limit = self.talk_end if self.talk_done else self.talk.n
            if (k + 1) * win <= limit:
                res = await asyncio.to_thread(self.m.emotion.predict, self.talk.slice(k * win, (k + 1) * win))
                if gen != self.talk_gen:
                    return
                self.emotions.append({"t0": k * win / SR, "t1": (k + 1) * win / SR, **res})
                k += 1
            elif self.talk_done:
                return
            else:
                await asyncio.sleep(0.2)

    async def _finish_talk(self, code_index: int | None) -> None:
        """The talk is over: cut off the code word, analyse, and let the coach speak."""
        if self.state != "talk":
            return
        await self.set_state("analysing")
        if code_index is not None:
            # Cut just after the last real word, so the code word and the pause before it are excluded.
            code_start = self.words[code_index]["start"]
            self.words = self.words[:code_index]
            cut = min(code_start, self.words[-1]["end"] + 0.15) if self.words else code_start
            self.talk_end = int(cut * SR)
        else:
            self.talk_end = self.talk.n
        self.talk_done = True
        gen = self.talk_gen
        for task in self.workers[1:]:                           # let the emotion worker finish its windows
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=60)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                task.cancel()

        audio = self.talk.slice(0, self.talk_end)
        report = await asyncio.to_thread(metrics.analyse, audio, self.words, SR, self.emotions)
        if gen != self.talk_gen:
            return
        if report.get("empty"):
            await self.send({"type": "notice", "text": "I could not hear a talk. Press Start and try again."})
            await self.set_state("idle")
            return

        self.previous, self.report = self.report, report
        prev = {"overall": self.previous["overall"], "scores": self.previous["scores"],
                "summary": self.previous["summary"]} if self.previous else None
        await self.send({"type": "report", "report": report, "previous": prev})

        brief = metrics.coach_brief(report, self.previous)
        self.messages = [
            {"role": "system", "content": agent.SYSTEM_PROMPT},
            agent.data_message(brief, metrics.transcript_with_times(self.words)),
            {"role": "user", "content": "(The speaker has just finished the talk. Give your first feedback.)"},
        ]
        await self._start_turn(fallback=agent.offline_feedback(report))

    # ------------------------------------------------------------ the coach speaks

    async def _start_turn(self, fallback: str, scripted: str | None = None) -> None:
        self.turn_seq += 1
        self.turn = Turn(id=self.turn_seq)
        self.listen = TurnDetector(config.BARGE_IN_MS, config.TURN_END_MS)
        self.vad.reset()
        await self.set_state("speaking")
        self.turn.task = self.spawn(self._run_turn(self.turn, fallback, scripted))

    async def _run_turn(self, turn: Turn, fallback: str, scripted: str | None) -> None:
        queue: asyncio.Queue = asyncio.Queue()

        def add(items: list[tuple]) -> None:
            for it in items:
                entry = {"id": len(turn.items), "kind": it[0], "status": "queued",
                         "text": it[1] if it[0] == "say" else "", "args": it[1:]}
                turn.items.append(entry)
                queue.put_nowait(entry)

        async def produce() -> None:
            stream = agent.ItemStream()
            try:
                if scripted is not None:
                    add(stream.feed(scripted))
                else:
                    async for delta in agent.stream_reply(self.messages):
                        add(stream.feed(delta))
            except agent.AgentError as exc:
                log.warning("language model unavailable: %s", exc)
                await self.send({"type": "notice", "text": f"Language model unavailable ({exc}). Using built-in replies."})
                if not turn.items:
                    stream = agent.ItemStream()
                    add(stream.feed(fallback))
            add(stream.finish())
            queue.put_nowait(None)

        producer = asyncio.create_task(produce())
        try:
            while (entry := await queue.get()) is not None:
                head = {"turn": turn.id, "id": entry["id"]}
                if entry["kind"] == "say":
                    if self.m.tts:
                        wav = await asyncio.to_thread(self.m.tts.synth, entry["text"])
                        await self.send_with_audio({"type": "tts", "text": entry["text"], **head}, wav)
                    else:
                        await self.send({"type": "tts", "text": entry["text"], "audio": False, **head})
                elif entry["kind"] == "show":
                    await self.send({"type": "cue", "action": "show", "name": entry["args"][0], **head})
                elif entry["kind"] == "play":
                    a, b = entry["args"]
                    clip = self.talk.slice(int(a * SR), min(int(b * SR), self.talk_end))
                    if len(clip) > SR // 4:
                        await self.send_with_audio({"type": "clip", "start": a, "end": b, **head}, wav_bytes(clip, SR))
                    else:
                        entry["status"] = "played"
            turn.generated = True
            await self.send({"type": "turn_end", "turn": turn.id})
        finally:
            producer.cancel()

    async def _finish_turn(self) -> None:
        """The browser has played the whole reply."""
        turn = self.turn
        if self.state != "speaking" or not turn or not turn.generated:
            return
        said = " ".join(it["text"] for it in turn.items if it["kind"] == "say")
        if said:
            self.messages.append({"role": "assistant", "content": said})
        self.turn = None
        self.listen = TurnDetector(config.TURN_START_MS, config.TURN_END_MS)
        self.vad.reset()
        await self.set_state("listening")

    async def _cancel_turn(self) -> None:
        if self.turn:
            if self.turn.task:
                self.turn.task.cancel()
            await self.send({"type": "stop_audio", "turn": self.turn.id})
            self.turn = None

    async def _barge_in(self) -> None:
        """The speaker cut in. Stop talking at once and remember what was and was not said."""
        turn = self.turn
        if not turn:
            return
        says = [it for it in turn.items if it["kind"] == "say"]
        spoken = " ".join(it["text"] for it in says if it["status"] != "queued")
        self.unsaid = " ".join(it["text"] for it in says if it["status"] == "queued")
        await self._cancel_turn()
        if spoken:
            self.messages.append({"role": "assistant", "content": f"{spoken} [interrupted here by the speaker]"})
        await self.send({"type": "barge_in", "turn": turn.id})
        await self.set_state("listening")           # self.listen is already recording the question

    async def _answer(self, audio: np.ndarray) -> None:
        """Transcribe the speaker's question and reply in context."""
        words = await asyncio.to_thread(self.m.stt.transcribe, audio, 0.0)
        if self.state != "thinking":                 # a new talk was started meanwhile
            return
        confident = [w for w in words if w["prob"] >= 0.35]
        text = " ".join(w["word"] for w in words) if confident else ""
        await self.send({"type": "user", "text": text})
        if not text:
            # Nothing intelligible (a cough, background noise): carry on where the coach left off.
            if self.unsaid:
                rest, self.unsaid = self.unsaid, ""
                await self._start_turn(fallback=rest, scripted=rest)
            else:
                self.listen = TurnDetector(config.TURN_START_MS, config.TURN_END_MS)
                await self.set_state("listening")
            return
        note = ""
        if self.unsaid:
            note = f'\n\n(Coach note: you were interrupted. You had not yet said: "{self.unsaid}")'
            self.unsaid = ""
        self.messages.append({"role": "user", "content": text + note})
        await self._start_turn(fallback=agent.offline_answer(text, self.report))

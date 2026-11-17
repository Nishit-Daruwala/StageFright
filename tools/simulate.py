"""End-to-end test without a browser or microphone.

Acts as the browser: synthesises a practice talk with the TTS voice, streams it to the
server as live microphone audio, "plays" the coach's replies, interrupts one of them
with a spoken question, then asks a second question normally.

    .venv/Scripts/python -m tools.simulate            (server must be running on port 8000)
"""

import asyncio
import io
import json
import sys
import time
import wave

import httpx
import numpy as np
import websockets
from scipy.signal import resample_poly

from server import config
from server.tts import Speaker

URL = "localhost:8000"
FRAME = 512
TALK = [
    "Good morning everyone. Today I want to talk about why sleep matters for students.",
    "Um, most of us think that staying up late helps us, like, get more work done.",
    "But research shows that memory is actually built while we sleep.",
    "So basically, if you cut sleep, you know, you forget more of what you studied.",
    "My advice is simple. Protect seven hours of sleep before every exam.",
    "Thank you for listening.",
]
CODE_WORD = "Hey StageFright, I have finished my speech."
QUESTION_1 = "Wait, where exactly did I speak too fast?"
QUESTION_2 = "How many filler words did I use?"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Voice:
    def __init__(self):
        self.speaker = Speaker()

    def say(self, text: str, voice: str = "am_michael") -> np.ndarray:
        samples, sr = self.speaker.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
        return resample_poly(np.asarray(samples, dtype=np.float64), config.SR, sr).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * config.SR), dtype=np.float32)


class Client:
    def __init__(self, ws):
        self.ws = ws
        self.mic = np.zeros(0, dtype=np.float32)      # audio waiting to be "spoken"
        self.rng = np.random.default_rng(0)
        self.header = None
        self.queue: asyncio.Queue = asyncio.Queue()
        self.turn = None
        self.turn_ended = False
        self.playing = False
        self.player: asyncio.Task | None = None
        self.state = "idle"
        self.events: dict[str, asyncio.Event] = {}
        self.log: list[dict] = []
        self.report = None
        self.spoken_items = 0

    def event(self, name: str) -> asyncio.Event:
        return self.events.setdefault(name, asyncio.Event())

    async def wait(self, name: str, timeout: float = 180) -> None:
        await asyncio.wait_for(self.event(name).wait(), timeout)
        self.event(name).clear()

    def forget(self, *names: str) -> None:
        """Clear events that fired earlier, so the next wait only sees new ones."""
        for name in names:
            self.event(name).clear()

    async def send(self, obj: dict) -> None:
        await self.ws.send(json.dumps(obj))

    def speak(self, audio: np.ndarray) -> None:
        self.mic = np.concatenate([self.mic, audio])

    async def mic_loop(self) -> None:
        """Sends one 32 ms frame every 32 ms, like a live microphone."""
        nxt = time.perf_counter()
        while True:
            if len(self.mic) >= FRAME:
                frame, self.mic = self.mic[:FRAME], self.mic[FRAME:]
            else:
                frame = np.zeros(FRAME, dtype=np.float32)
            frame = frame + 0.001 * self.rng.standard_normal(FRAME).astype(np.float32)
            await self.ws.send((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())
            nxt += FRAME / config.SR
            await asyncio.sleep(max(0.0, nxt - time.perf_counter()))

    async def play_loop(self) -> None:
        """Pretends to play each queued item for its real duration, reporting progress like the browser."""
        while True:
            item = await self.queue.get()
            self.playing = True
            await self.send({"type": "playing", "turn": item["turn"], "id": item["id"]})
            if item["type"] in ("tts", "clip"):
                if item["type"] == "tts":
                    log(f"  coach> {item['text']}")
                else:
                    log(f"  coach replays the talk from {item['start']} s to {item['end']} s")
                self.spoken_items += 1
                self.event("coach_audio").set()
                await asyncio.sleep(item["dur"])
            else:
                log(f"  screen switches to the {item['name']} chart")
            await self.send({"type": "played", "turn": item["turn"], "id": item["id"]})
            self.playing = False
            if self.turn_ended and self.queue.empty():
                self.turn_ended = False
                await self.send({"type": "turn_done", "turn": self.turn})

    def stop_playback(self) -> None:
        if self.player:
            self.player.cancel()
        self.queue = asyncio.Queue()
        self.playing = False
        self.turn_ended = False
        self.player = asyncio.create_task(self.play_loop())

    async def receive_loop(self) -> None:
        async for raw in self.ws:
            if isinstance(raw, bytes):
                if self.header:
                    with wave.open(io.BytesIO(raw)) as w:
                        dur = w.getnframes() / w.getframerate()
                    self.queue.put_nowait({**self.header, "dur": dur})
                    self.header = None
                continue
            msg = json.loads(raw)
            kind = msg["type"]
            self.log.append(msg if kind not in ("report", "meter") else {"type": kind})
            if kind == "state":
                self.state = msg["state"]
                log(f"state: {msg['state']}")
                self.event(f"state:{msg['state']}").set()
            elif kind == "models":
                if msg["ready"]:
                    self.event("ready").set()
            elif kind == "transcript":
                log(f"  transcript ({len(msg['words'])} words): ... {' '.join(w['word'] for w in msg['words'][-8:])}")
            elif kind == "report":
                self.report = msg["report"]
                self.event("report").set()
            elif kind in ("tts", "clip"):
                self.turn = msg["turn"]
                self.header = msg
            elif kind == "cue":
                self.turn = msg["turn"]
                self.queue.put_nowait(msg)
            elif kind == "turn_end":
                self.turn = msg["turn"]
                if self.queue.empty() and not self.playing:
                    await self.send({"type": "turn_done", "turn": self.turn})
                else:
                    self.turn_ended = True
            elif kind == "stop_audio":
                self.stop_playback()
                log("  coach audio stopped")
            elif kind == "barge_in":
                self.event("barge_in").set()
            elif kind == "user":
                log(f"  heard> {msg['text']!r}")
                self.event("user").set()
            elif kind == "notice":
                log(f"  notice: {msg['text']}")


async def main() -> int:
    log("waiting for the server's models")
    async with httpx.AsyncClient() as http:
        while True:
            status = (await http.get(f"http://{URL}/health")).json()["models"]
            if all(v != "loading" for v in status.values()):
                break
            await asyncio.sleep(2)
    log(f"models: {status}")

    log("synthesising the practice talk")
    voice = Voice()
    parts = []
    for sentence in TALK:
        parts += [voice.say(sentence), silence(0.7)]
    parts += [silence(1.2), voice.say(CODE_WORD), silence(1.0)]
    talk = np.concatenate([silence(0.5)] + parts)
    q1, q2 = voice.say(QUESTION_1), voice.say(QUESTION_2)
    log(f"talk is {len(talk) / config.SR:.1f} s long")

    async with websockets.connect(f"ws://{URL}/ws", max_size=None) as ws:
        c = Client(ws)
        tasks = [asyncio.create_task(c.receive_loop()), asyncio.create_task(c.mic_loop())]
        c.player = asyncio.create_task(c.play_loop())
        failures = []
        try:
            await c.wait("ready")
            await c.send({"type": "start"})
            await c.wait("state:talk")
            c.speak(talk)
            t0 = time.perf_counter()

            await c.wait("state:analysing", timeout=len(talk) / config.SR + 60)
            log(f"code word recognised {time.perf_counter() - t0 - len(talk) / config.SR + 1.0:+.1f} s after it ended")
            await c.wait("report")
            t_report = time.perf_counter()
            r = c.report
            log(f"report: overall {r['overall']}, scores {r['scores']}")
            log(f"summary: {json.dumps(r['summary'])}")
            log(f"fillers: {[(f['text'], f['start']) for f in r['fillers']]}")
            log(f"emotion windows: {[e['label'] for e in r['emotions']]}")
            log(f"transcript: {' '.join(w['word'] for w in r['words'])}")
            log(f"low-confidence words: {[(w['word'], w['prob']) for w in r['words'] if w['prob'] < 0.5]}")
            text = " ".join(w["word"] for w in r["words"]).lower()
            if "finished my speech" in text or "stagefright" in text.replace(" ", ""):
                failures.append("code word leaked into the analysed transcript")
            if r["summary"]["end_s"] > len(talk) / config.SR - 3.0:
                failures.append("analysed audio was not cut before the code word")

            await c.wait("coach_audio")
            log(f"coach started speaking {time.perf_counter() - t_report:.1f} s after the report")

            # Interrupt in the middle of the feedback.
            while c.spoken_items < 2:
                await c.wait("coach_audio")
            await asyncio.sleep(0.8)
            log(f"interrupting with: {QUESTION_1}")
            t_q = time.perf_counter()
            c.speak(q1)
            await c.wait("barge_in", timeout=20)
            log(f"coach stopped {time.perf_counter() - t_q:.2f} s after the question began")
            c.forget("coach_audio", "state:speaking")
            await c.wait("user", timeout=60)
            t_asked = time.perf_counter()
            await c.wait("coach_audio", timeout=60)
            log(f"answer began {time.perf_counter() - t_asked:.1f} s after the question was transcribed")
            c.forget("state:listening")
            await c.wait("state:listening", timeout=180)          # the coach finished its answer

            # A normal question once the coach is quiet.
            await asyncio.sleep(1.0)
            log(f"asking: {QUESTION_2}")
            c.forget("barge_in", "user")
            c.speak(q2)
            await c.wait("user", timeout=60)
            await c.wait("state:speaking", timeout=60)
            c.forget("state:listening")
            await c.wait("state:listening", timeout=180)
            if c.event("barge_in").is_set():
                failures.append("the second question was treated as an interruption")

            roles = [m for m in c.log if m["type"] == "user"]
            if len(roles) != 2 or not all(m["text"] for m in roles):
                failures.append(f"expected two transcribed questions, got {roles}")
        except asyncio.TimeoutError:
            failures.append(f"timed out waiting in state {c.state!r}")
        finally:
            for t in tasks + [c.player]:
                t.cancel()

    if failures:
        log("FAILED: " + "; ".join(failures))
        return 1
    log("PASSED: talk, code word, report, spoken feedback, barge-in and follow-up question all worked")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

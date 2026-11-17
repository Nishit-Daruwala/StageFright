"""The coach: prompt, language-model streaming (Groq), and splitting its reply into speakable items."""

import json
import re
from collections.abc import AsyncIterator

import httpx

from . import config

PLOTS = ["timeline", "pitch", "pace", "energy", "spectrogram", "scores"]

SYSTEM_PROMPT = f"""You are StageFright, a warm and direct public-speaking coach. You are talking out loud \
with a speaker who has just finished a practice talk. Everything you write is turned into speech.

How to talk:
- Spoken English only. Short sentences. No lists, markdown, headings, emoji or symbols.
- Say times the way a person would, for example "around thirty-five seconds in".
- Use only the measurements and transcript you are given. If something was not measured, say so plainly.
- Emotion labels come from a model trained on acted speech, and hesitation sounds are acoustic guesses. \
Treat both as rough hints.
- If a previous take is included, say what improved and what did not.

Your first feedback after a talk: 90 to 130 words. Give the overall impression with the score, one or two \
real strengths, the two or three fixes that matter most with where they happened, and one drill to practise. \
End by inviting questions.

Answers to questions: one to four sentences, straight to the point, with the number or moment that backs it up.

Interruptions: the speaker can cut in while you talk. Your earlier message is then marked as interrupted and \
you are told what you had not said yet. Answer the question first. Then offer in one short sentence to carry on. \
If the speaker agrees, continue with what you had not said, without repeating yourself.

Screen and playback tags. These are not spoken. Put them between sentences:
- [[show:NAME]] switches the screen to a chart while you discuss it. NAME is one of: {", ".join(PLOTS)}.
- [[play:START-END]] replays the speaker's own recording between two times in seconds, at most 8 seconds long. \
Use it at most once per reply, when hearing the moment helps."""

_TAG = re.compile(r"\[\[\s*(show|play)\s*:\s*([^\]]*?)\s*\]\]")
_SENT_END = re.compile(r"[.!?]+[\"')\]]*\s")


class AgentError(Exception):
    pass


def data_message(brief: dict, transcript: str) -> dict:
    return {"role": "system",
            "content": f"MEASUREMENTS (JSON):\n{json.dumps(brief, separators=(',', ':'))}\n\nTRANSCRIPT:\n{transcript}"}


# ---------------------------------------------------------------- reply -> speakable items

class ItemStream:
    """Turns streamed text into ordered items: ("say", sentence), ("show", name), ("play", start, end)."""

    def __init__(self):
        self.buf = ""

    def feed(self, delta: str) -> list[tuple]:
        self.buf += delta
        return self._drain(final=False)

    def finish(self) -> list[tuple]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[tuple]:
        items: list[tuple] = []
        while True:
            buf = self.buf
            open_at = buf.find("[[")
            sent = _SENT_END.search(buf)
            if open_at != -1 and (sent is None or open_at < sent.end()):
                tag = _TAG.match(buf, open_at)
                if tag:
                    items += _say(buf[:open_at])
                    items += _tag_item(tag)
                    self.buf = buf[tag.end():]
                elif "]]" in buf[open_at:] or final:
                    self.buf = buf[:open_at] + buf[open_at + 2:]      # malformed tag: drop the brackets
                else:
                    break                                             # tag still arriving
            elif sent:
                items += _say(buf[:sent.end()])
                self.buf = buf[sent.end():]
            else:
                break
        if final:
            items += _say(self.buf)
            self.buf = ""
        return items


def _say(text: str) -> list[tuple]:
    text = re.sub(r"[*_#`]|\]\]", "", text).strip()
    return [("say", text)] if re.search(r"\w", text) else []


def _tag_item(tag: re.Match) -> list[tuple]:
    kind, arg = tag.group(1), tag.group(2)
    if kind == "show":
        return [("show", arg.lower())] if arg.lower() in PLOTS else []
    nums = re.findall(r"\d+(?:\.\d+)?", arg)
    if len(nums) == 2:
        a, b = float(nums[0]), float(nums[1])
        if b > a:
            return [("play", a, min(b, a + 8.0))]
    return []


# ---------------------------------------------------------------- language model

_model_cache: str | None = None


async def _pick_model(client: httpx.AsyncClient) -> str:
    """Use GROQ_MODEL if set; otherwise choose a chat model from the account's current list."""
    global _model_cache
    if config.GROQ_MODEL:
        return config.GROQ_MODEL
    if _model_cache:
        return _model_cache
    r = await client.get(f"{config.GROQ_URL}/models")
    if r.status_code != 200:
        raise AgentError(f"Groq model list failed ({r.status_code}): {r.text[:200]}")
    ids = [m["id"] for m in r.json().get("data", [])]
    chat = [i for i in ids if not re.search(r"whisper|tts|guard|embed|speech|orpheus", i, re.I)]
    for want in ("gpt-oss-120b", "llama-3.3-70b", "qwen", "gpt-oss-20b", "llama"):
        for i in chat:
            if want in i:
                _model_cache = i
                return i
    if not chat:
        raise AgentError("Groq returned no chat models")
    _model_cache = chat[0]
    return _model_cache


async def stream_reply(messages: list[dict]) -> AsyncIterator[str]:
    """Stream the coach's reply text from Groq."""
    if not config.GROQ_API_KEY:
        raise AgentError("GROQ_API_KEY is not set")
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        model = await _pick_model(client)
        body = {"model": model, "messages": messages, "stream": True, "temperature": 0.6, "max_tokens": 700}
        async with client.stream("POST", f"{config.GROQ_URL}/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise AgentError(f"Groq error {r.status_code}: {(await r.aread()).decode(errors='replace')[:300]}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content")
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
                if delta:
                    yield delta


# ---------------------------------------------------------------- replies without a language model

def offline_feedback(report: dict) -> str:
    """Spoken feedback built from the numbers alone. Used when the language model is unavailable."""
    s, sc = report["summary"], report["scores"]
    names = {"pace": "pace", "fillers": "filler words", "pauses": "pausing", "pitch": "pitch variety",
             "energy": "loudness variety"}
    order = sorted(sc, key=sc.get)
    best, worst = order[-1], order[0]
    tips = {
        "pace": f"Your pace was {s['wpm']} words a minute. Aim for between one hundred thirty and one hundred sixty five.",
        "fillers": f"I counted {s['filler_count']} filler words, about {s['fillers_per_min']} a minute. Try a silent pause instead.",
        "pauses": f"Pauses took {round(s['pause_ratio'] * 100)} percent of your time, and the longest was {s['longest_pause_s']} seconds.",
        "pitch": f"Your pitch moved by {s['pitch_std_st']} semitones on average. Three or more sounds lively.",
        "energy": f"Your loudness range was {s['level_range_db']} decibels. Stress the key words to widen it.",
    }
    return (f"Thanks, I have your talk. [[show:scores]] Your overall delivery score is {report['overall']} out of one hundred. "
            f"Your strongest area was {names[best]}. The area to work on first is {names[worst]}. {tips[worst]} "
            f"{tips[order[1]] if len(order) > 1 else ''} Ask me about any part of your talk.")


def offline_answer(question: str, report: dict) -> str:
    q, s = question.lower(), report["summary"]
    if re.search(r"fast|slow|pace|speed|rate", q):
        fastest = max(report["windows"], key=lambda w: w["wpm"], default=None)
        extra = f" Your fastest stretch was around {round(fastest['t0'])} seconds in, at {fastest['wpm']}." if fastest else ""
        return f"[[show:pace]] You averaged {s['wpm']} words a minute.{extra}"
    if re.search(r"filler|um|uh|like", q):
        return f"[[show:timeline]] I counted {s['filler_count']} filler words, about {s['fillers_per_min']} a minute."
    if re.search(r"pause|silence|stop", q):
        return f"[[show:timeline]] You paused {s['pause_count']} times. The longest pause was {s['longest_pause_s']} seconds."
    if re.search(r"pitch|monotone|flat|tone", q):
        return f"[[show:pitch]] Your pitch varied by {s['pitch_std_st']} semitones around {s['pitch_median_hz']} hertz."
    if re.search(r"loud|volume|energy|quiet", q):
        return f"[[show:energy]] Your loudness range was {s['level_range_db']} decibels."
    return (f"Your overall score was {report['overall']}. I can tell you about pace, fillers, pauses, pitch or loudness. "
            "For open questions I need the language model connected.")

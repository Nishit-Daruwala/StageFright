"""Speech-to-text (Whisper) plus the text-side detectors: code word and filler words."""

import re

import numpy as np

from . import config

# Whisper tends to tidy hesitations away. Starting it with a disfluent prompt makes it keep them.
# Keep it short and closed: a longer, trailing-off prompt made Whisper append invented words.
VERBATIM_PROMPT = "Um, uh, like, you know."

SOUND_FILLERS = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm", "eh"}
WORD_FILLERS = {"like", "basically", "actually", "literally", "honestly", "obviously"}
_LIKE_AS_VERB = {"i", "you", "we", "they", "would", "dont", "didnt", "to", "really", "looks", "look",
                 "looked", "feels", "feel", "felt", "sounds", "sound", "something", "just", "more"}
PHRASE_FILLERS = [("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of")]

_STAGE_ONE = {"stagefright", "stagefried", "stageflight", "stagefrite", "stagefreight"}
_STAGE_TWO = {"fright", "fried", "flight", "freight", "frite", "fight", "bright", "right"}
_GREET = {"hey", "hi", "hay", "okay", "ok", "hello"}
_SELF = {"i", "im", "ive", "we", "weve", "were"}
_DONE = {"done", "completed", "complete", "ended", "over"}


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


class Transcriber:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio: np.ndarray, offset: float = 0.0) -> list[dict]:
        """Return words as {word, start, end, prob}, with times shifted by offset seconds."""
        segments, _ = self.model.transcribe(
            audio.astype(np.float32), language="en", beam_size=1, temperature=0.0,
            word_timestamps=True, condition_on_previous_text=False,
            initial_prompt=VERBATIM_PROMPT, vad_filter=False,
        )
        words = []
        for seg in segments:
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue                          # likely a hallucination on noise
            for w in seg.words or []:
                text = w.word.strip()
                if norm(text):                    # skip tokens that are only punctuation
                    words.append({"word": text, "start": round(w.start + offset, 3),
                                  "end": round(w.end + offset, 3), "prob": round(w.probability, 3)})
        # A very unsure word tacked on after the speech has ended is an invention, not speech.
        while words and words[-1]["prob"] < 0.15:
            words.pop()
        return words


def find_code_word(words: list[dict]) -> int | None:
    """Index of the first word of "Hey StageFright, I have finished my speech", or None.

    Matching is loose because Whisper may write the name as one word or two, or mishear
    it. It needs the name, then "I/we" and "finish..." or "done" within the next few words,
    so a talk that merely mentions stage fright does not end early.
    """
    toks = [norm(w["word"]) for w in words]
    for i, tok in enumerate(toks):
        if tok in _STAGE_ONE:
            after = i + 1
        elif tok == "stage" and i + 1 < len(toks) and toks[i + 1] in _STAGE_TWO:
            after = i + 2
        else:
            continue
        tail = toks[after:after + 7]
        finished = any(t.startswith("finish") or t in _DONE for t in tail)
        greeted = i > 0 and toks[i - 1] in _GREET
        if finished and (greeted or any(t in _SELF for t in tail)):
            return i - 1 if greeted else i
    return None


def find_fillers(words: list[dict]) -> list[dict]:
    """Filler words in the transcript: hesitation sounds, filler words and filler phrases."""
    toks = [norm(w["word"]) for w in words]
    out, skip = [], -1
    for i, tok in enumerate(toks):
        if i <= skip:
            continue
        if tok in SOUND_FILLERS:
            out.append({"text": tok, "kind": "sound", "start": words[i]["start"], "end": words[i]["end"], "index": i})
        elif tok in WORD_FILLERS:
            if tok == "like" and i > 0 and toks[i - 1] in _LIKE_AS_VERB:
                continue                          # "I like", "looks like": a real word, not a filler
            out.append({"text": tok, "kind": "word", "start": words[i]["start"], "end": words[i]["end"], "index": i})
        elif i + 1 < len(toks) and (tok, toks[i + 1]) in PHRASE_FILLERS:
            out.append({"text": f"{tok} {toks[i + 1]}", "kind": "word",
                        "start": words[i]["start"], "end": words[i + 1]["end"], "index": i})
            skip = i + 1
    return out

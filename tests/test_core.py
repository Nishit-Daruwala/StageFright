"""Checks for the hand-written DSP, the text detectors and the coach's reply parser."""

import numpy as np
import pytest

from server import dsp, metrics
from server.agent import ItemStream
from server.stt import find_code_word, find_fillers

SR = 16000


def tone(freq: float, dur: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def voiced(freq: float, dur: float, amp: float = 0.4) -> np.ndarray:
    """Harmonic-rich signal, closer to a vowel than a pure sine."""
    t = np.arange(int(dur * SR)) / SR
    return amp * sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, 6))


def words_from(text: str, start: float = 0.0, step: float = 0.4) -> list[dict]:
    return [{"word": w, "start": start + i * step, "end": start + i * step + 0.3, "prob": 0.9}
            for i, w in enumerate(text.split())]


@pytest.mark.parametrize("freq", [90, 120, 220, 330])
def test_pitch_of_pure_tone(freq):
    f0 = dsp.autocorr_f0(tone(freq, 1.0), SR)
    mid = f0[20:-20]
    assert (mid > 0).all()
    assert abs(np.median(mid) - freq) < 0.01 * freq


def test_pitch_of_harmonic_signal_has_no_octave_error():
    f0 = dsp.autocorr_f0(voiced(150, 1.0), SR)
    assert abs(np.median(f0[20:-20]) - 150) < 2


def test_noise_is_unvoiced():
    rng = np.random.default_rng(0)
    f0 = dsp.autocorr_f0(0.3 * rng.standard_normal(SR), SR)
    assert (f0 > 0).mean() < 0.05


def test_vad_finds_speech_and_pause():
    rng = np.random.default_rng(1)
    noise = lambda d: 0.002 * rng.standard_normal(int(d * SR))
    x = np.concatenate([noise(0.5), voiced(140, 1.0), noise(0.8), voiced(160, 1.2), noise(0.5)])
    x = x + 0.002 * rng.standard_normal(len(x))
    segs = dsp.mask_to_segments(dsp.energy_zcr_vad(x, SR))
    assert len(segs) == 2
    assert abs(segs[0][0] - 0.5) < 0.05 and abs(segs[0][1] - 1.5) < 0.05
    assert abs((segs[1][0] - segs[0][1]) - 0.8) < 0.06


def test_energy_and_zcr():
    frames = dsp.frame_signal(tone(100, 0.5, amp=1.0), 400, 160)
    assert abs(dsp.rms_db(frames)[10] - (-3.01)) < 0.2            # sine RMS is amplitude / sqrt(2)
    assert abs(dsp.zero_crossing_rate(frames)[10] - 2 * 100 / SR) < 0.005


def test_mfcc_and_spectrogram_shapes():
    x = voiced(120, 0.5)
    assert dsp.mfcc(x, SR).shape == (1 + len(x) // 160, 13)
    spec = dsp.stft_db(tone(1000, 0.5), SR)
    assert spec.shape[1] == 257
    assert abs(spec[20].argmax() * SR / 512 - 1000) < 40


def test_code_word_variants():
    talk = "so that is my idea thank you"
    for ending in ["Hey StageFright, I have finished my speech.", "hey stage fright I have finished my speech",
                   "Hey, Stage Fried. I'm finished with my speech.", "Stagefright, I am done."]:
        words = words_from(f"{talk} {ending}")
        assert find_code_word(words) == 7, ending


def test_code_word_not_triggered_by_topic():
    for text in ["my stage fright was terrible before the show", "hey everyone today I talk about stage fright",
                 "I finished the race and I was done"]:
        assert find_code_word(words_from(text)) is None, text


def test_fillers():
    found = find_fillers(words_from("Um, so I like pizza and, like, basically you know it is uh good"))
    assert [f["text"] for f in found] == ["um", "like", "basically", "you know", "uh"]
    assert [f["kind"] for f in found] == ["sound", "word", "word", "word", "sound"]


def test_item_stream_splits_sentences_and_tags():
    s = ItemStream()
    items = []
    for chunk in ["Your pace was 3.5 words a second. [[sh", "ow:pace]] You sped up at 34 seconds! [[play:34-50]]", " Try pausing"]:
        items += s.feed(chunk)
    items += s.finish()
    assert items == [("say", "Your pace was 3.5 words a second."), ("show", "pace"),
                     ("say", "You sped up at 34 seconds!"), ("play", 34.0, 42.0), ("say", "Try pausing")]


def test_item_stream_drops_bad_tags():
    s = ItemStream()
    items = s.feed("Hello there. [[show:unknown]] [[oops]] Done. ") + s.finish()
    assert [i for i in items if i[0] != "say"] == []
    assert " ".join(i[1] for i in items) == "Hello there. oops Done."


def test_analyse_synthetic_talk():
    rng = np.random.default_rng(2)
    quiet = lambda d: 0.002 * rng.standard_normal(int(d * SR))
    x = np.concatenate([quiet(0.5), voiced(130, 4.0), quiet(1.0), voiced(170, 4.0), quiet(0.5)])
    x = (x + 0.002 * rng.standard_normal(len(x))).astype(np.float32)
    words = words_from("one two three four five six seven eight", 0.6, 0.45) + \
        words_from("um nine ten eleven twelve thirteen fourteen fifteen", 5.6, 0.45)
    r = metrics.analyse(x, words, SR)
    s = r["summary"]
    assert not r["empty"]
    assert abs(s["duration_s"] - 9.0) < 0.2
    assert s["pause_count"] == 1 and abs(s["longest_pause_s"] - 1.0) < 0.1
    assert s["words"] == 15 and s["filler_count"] == 1
    assert abs(s["wpm"] - 100) <= 3
    assert 140 < s["pitch_median_hz"] < 160
    assert 0 <= r["overall"] <= 100
    import json
    json.dumps(r)                                                   # the report must be plain JSON

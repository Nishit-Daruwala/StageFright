"""Turns the talk's audio and word timestamps into delivery metrics, scores and plot data."""

import json

import numpy as np

from . import dsp
from .stt import SOUND_FILLERS, find_fillers, norm

MIN_PAUSE_S = 0.25
LONG_PAUSE_S = 2.0
WINDOW_S, WINDOW_STEP_S = 10.0, 5.0


def _band(x: float, lo_bad: float, lo_good: float, hi_good: float, hi_bad: float) -> float:
    """100 inside [lo_good, hi_good], falling linearly to 0 at lo_bad and hi_bad."""
    if lo_good <= x <= hi_good:
        return 100.0
    if x < lo_good:
        return float(np.clip(100 * (x - lo_bad) / (lo_good - lo_bad), 0, 100))
    return float(np.clip(100 * (hi_bad - x) / (hi_bad - hi_good), 0, 100))


def _ramp(x: float, zero_at: float, full_at: float) -> float:
    return float(np.clip(100 * (x - zero_at) / (full_at - zero_at), 0, 100))


def _pool(a: np.ndarray, max_cols: int) -> np.ndarray:
    """Average neighbouring time frames so a plot has at most max_cols columns."""
    step = int(np.ceil(len(a) / max_cols))
    if step <= 1:
        return a
    n = (len(a) // step) * step
    return a[:n].reshape(-1, step, *a.shape[1:]).mean(axis=1)


def _fmt_time(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def analyse(audio: np.ndarray, words: list[dict], sr: int, emotions: list[dict] | None = None) -> dict:
    """Full analysis of one talk. Times are seconds from the start of the recording."""
    x = dsp.remove_dc(audio.astype(np.float64))
    hop, flen = int(dsp.HOP_S * sr), int(dsp.FRAME_S * sr)
    frames = dsp.frame_signal(x, flen, hop)
    energy = dsp.rms_db(frames)
    mask = dsp.energy_zcr_vad(x, sr)
    segments = dsp.mask_to_segments(mask)
    if not segments or not words:
        return {"empty": True}

    f0 = dsp.autocorr_f0(x, sr, speech_mask=mask)
    t_first, t_last = segments[0][0], segments[-1][1]
    duration = max(t_last - t_first, 1e-3)
    speech_time = sum(e - s for s, e in segments)

    # ---- pauses: silent gaps between speech segments
    pauses = [{"start": round(a[1], 2), "end": round(b[0], 2), "dur": round(b[0] - a[1], 2)}
              for a, b in zip(segments, segments[1:]) if b[0] - a[1] >= MIN_PAUSE_S]
    pause_time = sum(p["dur"] for p in pauses)
    long_pauses = [p for p in pauses if p["dur"] >= LONG_PAUSE_S]

    # ---- pace
    spoken = [w for w in words if norm(w["word"]) not in SOUND_FILLERS]
    n_words = len(spoken)
    wpm = n_words / (duration / 60)
    articulation = n_words / (max(speech_time, 1e-3) / 60)

    # ---- fillers: from the transcript, plus an acoustic detector for sounds Whisper dropped
    fillers = find_fillers(words)
    hesitations = _acoustic_hesitations(mask, f0, words)
    fillers_per_min = len(fillers) / (duration / 60)

    # ---- pitch
    voiced = f0 > 0
    if voiced.sum() >= 20:
        f0_med = float(np.median(f0[voiced]))
        st = np.where(voiced, dsp.hz_to_semitones(f0, f0_med), np.nan)
        st_std = float(np.nanstd(st))
        f0_lo, f0_hi = (float(v) for v in np.percentile(f0[voiced], [5, 95]))
    else:
        f0_med, st, st_std, f0_lo, f0_hi = 0.0, np.full(len(f0), np.nan), 0.0, 0.0, 0.0
    flat = _flat_stretches(st)

    # ---- loudness
    e_speech = energy[mask]
    e_p10, e_p90 = (float(v) for v in np.percentile(e_speech, [10, 90]))
    e_range = e_p90 - e_p10

    # ---- emotion: a window that is mostly silence carries no emotion
    emotions = [dict(e) for e in (emotions or []) if e["t0"] < t_last]
    for e in emotions:
        a, b = int(e["t0"] / dsp.HOP_S), int(e["t1"] / dsp.HOP_S)
        if mask[a:b].mean() < 0.3:
            e["label"] = "silence"
    emo_counts: dict[str, int] = {}
    for e in emotions:
        if e["label"] != "silence":
            emo_counts[e["label"]] = emo_counts.get(e["label"], 0) + 1

    # ---- 10-second windows, so the coach can say where something happened
    windows = []
    t = t_first
    while t < t_last - 2.0:
        t1 = min(t + WINDOW_S, t_last)
        a, b = int(t / dsp.HOP_S), int(t1 / dsp.HOP_S)
        st_w = st[a:b][~np.isnan(st[a:b])]
        sp = mask[a:b]
        emo = [e["label"] for e in emotions if e["t0"] < t1 and e["t1"] > t and e["label"] != "silence"]
        windows.append({
            "t0": round(t, 1), "t1": round(t1, 1),
            "wpm": round(sum(t <= w["start"] < t1 for w in spoken) / ((t1 - t) / 60)),
            "pitch_std_st": round(float(np.std(st_w)), 2) if len(st_w) > 20 else None,
            "level_db": round(float(energy[a:b][sp].mean()), 1) if sp.any() else None,
            "fillers": sum(t <= f["start"] < t1 for f in fillers),
            "pause_s": round(sum(max(0.0, min(p["end"], t1) - max(p["start"], t)) for p in pauses), 1),
            "emotion": max(set(emo), key=emo.count) if emo else None,
        })
        t += WINDOW_STEP_S

    # ---- scores (heuristic target ranges; tune on real good and bad takes)
    pause_ratio = pause_time / duration
    scores = {
        "pace": _band(wpm, 90, 130, 165, 210),
        "fillers": 100 - _ramp(fillers_per_min, 1, 10),
        "pauses": max(0.0, _band(pause_ratio, 0.03, 0.10, 0.30, 0.50) - min(30, 8 * len(long_pauses))),
        "pitch": _ramp(st_std, 0.5, 3.0),
        "energy": _ramp(e_range, 4, 14),
    }
    weights = {"pace": 0.25, "fillers": 0.25, "pitch": 0.20, "pauses": 0.15, "energy": 0.15}
    scores = {k: round(v) for k, v in scores.items()}
    overall = round(sum(scores[k] * w for k, w in weights.items()))

    # ---- plot data
    env = np.max(np.abs(frames), axis=1)
    spec = dsp.stft_db(x, sr)[:, :256]
    spec = np.maximum(spec, spec.max() - 80)
    spec = _pool(spec, 500).reshape(-1, 128, 2).mean(axis=2)
    mf = _pool(dsp.mfcc(x, sr), 500)

    return _plain({
        "empty": False,
        "summary": {
            "duration_s": round(duration, 1), "start_s": round(t_first, 2), "end_s": round(t_last, 2),
            "speech_s": round(speech_time, 1), "words": n_words,
            "wpm": round(wpm), "articulation_wpm": round(articulation),
            "pause_count": len(pauses), "pause_ratio": round(pause_ratio, 3),
            "longest_pause_s": max((p["dur"] for p in pauses), default=0.0),
            "filler_count": len(fillers), "fillers_per_min": round(fillers_per_min, 1),
            "hesitation_sounds": len(hesitations),
            "pitch_median_hz": round(f0_med), "pitch_std_st": round(st_std, 2),
            "pitch_range_hz": [round(f0_lo), round(f0_hi)],
            "level_range_db": round(e_range, 1),
            "emotions": emo_counts,
        },
        "scores": scores, "overall": overall,
        "pauses": pauses, "fillers": fillers, "hesitations": hesitations,
        "flat_stretches": flat, "windows": windows, "emotions": emotions, "words": words,
        "series": {
            "hop": dsp.HOP_S,
            "env": np.round(env, 3).tolist(),
            "energy": np.round(energy, 1).tolist(),
            "f0": [round(float(v), 1) if v > 0 else None for v in f0],
            "speech": mask.astype(int).tolist(),
        },
        "spectrogram": {"z": np.round(spec.T, 1).tolist(), "dt": duration_of(len(x), sr) / max(len(spec), 1),
                        "df": (sr / 2) / 128 * (256 / 257)},
        "mfcc": {"z": np.round(mf[:, 1:].T, 1).tolist(), "dt": duration_of(len(x), sr) / max(len(mf), 1)},
    })


def _plain(obj):
    """Convert NumPy scalars to plain Python numbers so the report is valid JSON."""
    return json.loads(json.dumps(obj, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


def duration_of(n_samples: int, sr: int) -> float:
    return n_samples / sr


def _acoustic_hesitations(mask: np.ndarray, f0: np.ndarray, words: list[dict]) -> list[dict]:
    """Voiced, flat-pitched sound where the transcript has no word: a likely "um" or "uh".

    Whisper often leaves hesitation sounds out. They still show up in the signal as
    speech (VAD on) with steady pitch, in a stretch no transcribed word covers.
    """
    covered = np.zeros(len(mask), dtype=bool)
    for w in words:
        a = max(0, int((w["start"] - 0.06) / dsp.HOP_S))
        b = int((w["end"] + 0.06) / dsp.HOP_S)
        covered[a:b] = True
    out = []
    for s, e in dsp._runs(mask & ~covered):
        dur = (e - s) * dsp.HOP_S
        f = f0[s:e]
        v = f[f > 0]
        if 0.2 <= dur <= 1.5 and len(v) >= 0.6 * (e - s):
            if np.std(dsp.hz_to_semitones(v, np.median(v))) < 1.0:
                out.append({"start": round(s * dsp.HOP_S, 2), "end": round(e * dsp.HOP_S, 2)})
    return out


def _flat_stretches(st: np.ndarray, win_s: float = 5.0, thr_st: float = 1.0) -> list[dict]:
    """Stretches of at least win_s where pitch varies by less than thr_st semitones (monotone)."""
    win, step = int(win_s / dsp.HOP_S), int(1.0 / dsp.HOP_S)
    flags = []
    for a in range(0, max(len(st) - win, 0) + 1, step):
        w = st[a:a + win]
        w = w[~np.isnan(w)]
        flags.append(len(w) >= 150 and float(np.std(w)) < thr_st)
    out = []
    for s, e in dsp._runs(np.array(flags, dtype=bool)):
        out.append({"start": round(s * step * dsp.HOP_S, 1), "end": round(((e - 1) * step + win) * dsp.HOP_S, 1)})
    return out


def transcript_with_times(words: list[dict], every_s: float = 10.0) -> str:
    """Transcript as lines that each start with a timestamp, for the coach's context."""
    lines, cur, mark = [], [], None
    for w in words:
        if mark is None or w["start"] - mark >= every_s:
            if cur:
                lines.append(" ".join(cur))
            cur, mark = [f"[{_fmt_time(w['start'])}]"], w["start"]
        cur.append(w["word"])
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def coach_brief(report: dict, previous: dict | None = None) -> dict:
    """The part of the report the language model sees: numbers and times, no plot arrays."""
    brief = {
        "overall_score": report["overall"], "scores": report["scores"], "summary": report["summary"],
        "ten_second_windows": report["windows"],
        "pauses_over_1s": [p for p in report["pauses"] if p["dur"] >= 1.0],
        "fillers": [{"text": f["text"], "at": f["start"]} for f in report["fillers"]],
        "possible_hesitation_sounds_at": [h["start"] for h in report["hesitations"]],
        "monotone_stretches": report["flat_stretches"],
        "targets": {"wpm": "130-165", "fillers_per_min": "under 1", "pause_ratio": "0.10-0.30",
                    "pitch_std_st": "3 or more", "level_range_db": "14 or more"},
    }
    if previous and not previous.get("empty"):
        brief["previous_take"] = {"overall_score": previous["overall"], "scores": previous["scores"],
                                  "summary": previous["summary"]}
    return brief

"""Hand-written speech signal processing: framing, energy, ZCR, VAD, pitch, STFT, MFCC.

Everything works on a mono float signal in [-1, 1]. All frame-level features use the
same 10 ms hop and centred frames, so index i always means time i * HOP_S.
"""

import numpy as np

HOP_S = 0.010
FRAME_S = 0.025
F0_FRAME_S = 0.040
EPS = 1e-10


# ---------------------------------------------------------------- basics

def remove_dc(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x) if len(x) else x


def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - alpha * x[n-1]. Boosts high frequencies, which speech has less of."""
    if len(x) == 0:
        return x
    return np.append(x[0], x[1:] - alpha * x[:-1])


def hamming(n: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Cut x into overlapping frames. Frame i is centred on sample i * hop."""
    half = frame_len // 2
    padded = np.pad(x, (half, frame_len - half))
    n_frames = 1 + len(x) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[idx]


def frame_times(n_frames: int) -> np.ndarray:
    return np.arange(n_frames) * HOP_S


# ---------------------------------------------------------------- energy, ZCR

def rms_db(frames: np.ndarray) -> np.ndarray:
    """Short-time energy as RMS level in dB relative to full scale."""
    return 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + EPS)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs that change sign. High for noise-like sounds (s, f)."""
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=1)


# ---------------------------------------------------------------- voice activity

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/end (exclusive) of every run of True values."""
    if len(mask) == 0:
        return []
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def smooth_mask(mask: np.ndarray, fill_gap: int, min_run: int) -> np.ndarray:
    """Fill silent gaps shorter than fill_gap frames, then drop speech runs shorter than min_run."""
    out = mask.copy()
    for s, e in _runs(~out):
        if e - s < fill_gap and s > 0 and e < len(out):
            out[s:e] = True
    for s, e in _runs(out):
        if e - s < min_run:
            out[s:e] = False
    return out


def energy_zcr_vad(x: np.ndarray, sr: int) -> np.ndarray:
    """Speech/non-speech decision per 10 ms frame.

    The threshold adapts to the recording: it sits between the noise floor (10th
    percentile of frame energy) and the speech level (95th percentile). Frames just
    under the threshold still count when their ZCR is high, which keeps weak
    fricatives at word edges. Short gaps are filled and short blips removed.
    """
    hop, flen = int(HOP_S * sr), int(FRAME_S * sr)
    frames = frame_signal(x, flen, hop)
    e = rms_db(frames)
    z = zero_crossing_rate(frames)
    floor, peak = np.percentile(e, 10), np.percentile(e, 95)
    if peak - floor < 6:                       # no contrast: all speech or all silence
        return np.full(len(e), peak > -45.0)
    thr = floor + max(6.0, 0.3 * (peak - floor))
    mask = (e > thr) | ((e > thr - 6.0) & (z > 0.25))
    return smooth_mask(mask, fill_gap=int(0.15 / HOP_S), min_run=int(0.08 / HOP_S))


def mask_to_segments(mask: np.ndarray) -> list[tuple[float, float]]:
    return [(float(s * HOP_S), float(e * HOP_S)) for s, e in _runs(mask)]


# ---------------------------------------------------------------- pitch

def median_filter(x: np.ndarray, k: int = 5) -> np.ndarray:
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, k), axis=1)


def autocorr_f0(x: np.ndarray, sr: int, fmin: float = 75.0, fmax: float = 400.0,
                speech_mask: np.ndarray | None = None, voicing_thr: float = 0.5) -> np.ndarray:
    """Fundamental frequency per 10 ms frame by autocorrelation. 0 means unvoiced.

    A voiced frame repeats every pitch period T, so its autocorrelation peaks at lag T.
    Steps per 40 ms frame: remove the mean, centre-clip (removes formant ripple),
    autocorrelate, normalise so lag 0 is 1, then take the first strong peak between
    the lags for fmax and fmin. Taking the first strong peak rather than the highest
    avoids picking 2T, which would halve the pitch (an octave error).
    """
    hop, flen = int(HOP_S * sr), int(F0_FRAME_S * sr)
    frames = frame_signal(x, flen, hop).astype(np.float64)
    n = len(frames)
    f0 = np.zeros(n)
    if n == 0:
        return f0
    frames = frames - frames.mean(axis=1, keepdims=True)

    clip = 0.3 * np.max(np.abs(frames), axis=1, keepdims=True)
    clipped = np.where(frames > clip, frames - clip, np.where(frames < -clip, frames + clip, 0.0))

    # Autocorrelation through the power spectrum (Wiener-Khinchin).
    nfft = 1 << int(np.ceil(np.log2(2 * flen)))
    spec = np.fft.rfft(clipped, nfft, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, nfft, axis=1)[:, :flen]
    r0 = ac[:, :1]
    ok = r0[:, 0] > EPS
    # Fewer sample pairs overlap at longer lags; divide that taper out.
    ac = ac / np.maximum(r0, EPS) / (1.0 - np.arange(flen) / flen)

    lo, hi = int(sr / fmax), min(int(sr / fmin), flen - 2)
    seg = ac[:, lo:hi + 1]
    best = seg.max(axis=1)
    inner = ac[:, lo:hi + 1]
    is_peak = (inner >= ac[:, lo - 1:hi]) & (inner >= ac[:, lo + 1:hi + 2])
    strong = is_peak & (seg >= 0.9 * best[:, None])
    has = strong.any(axis=1)
    lag = lo + np.where(has, strong.argmax(axis=1), seg.argmax(axis=1))

    # Parabolic interpolation around the peak for sub-sample precision.
    rows = np.arange(n)
    a, b, c = ac[rows, lag - 1], ac[rows, lag], ac[rows, lag + 1]
    denom = a - 2 * b + c
    shift = np.where(np.abs(denom) > EPS, 0.5 * (a - c) / np.where(np.abs(denom) > EPS, denom, 1.0), 0.0)
    period = lag + np.clip(shift, -1, 1)

    voiced = ok & (best > voicing_thr)
    if speech_mask is not None:
        voiced &= speech_mask[:n]
    f0[voiced] = sr / period[voiced]

    # Median smoothing inside voiced runs removes isolated octave jumps.
    for s, e in _runs(f0 > 0):
        if e - s >= 5:
            f0[s:e] = median_filter(f0[s:e], 5)
        elif e - s < 3:
            f0[s:e] = 0.0
    return f0


def hz_to_semitones(f0: np.ndarray, ref: float) -> np.ndarray:
    """Pitch distance from ref in semitones. Hearing is logarithmic, so this compares speakers fairly."""
    return 12.0 * np.log2(np.maximum(f0, EPS) / ref)


# ---------------------------------------------------------------- spectral

def stft_db(x: np.ndarray, sr: int, n_fft: int = 512) -> np.ndarray:
    """Magnitude spectrogram in dB, shape (frames, n_fft // 2 + 1)."""
    hop, flen = int(HOP_S * sr), int(FRAME_S * sr)
    frames = frame_signal(pre_emphasis(x), flen, hop) * hamming(flen)
    return 20 * np.log10(np.abs(np.fft.rfft(frames, n_fft, axis=1)) + EPS)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    """Triangular filters equally spaced on the mel scale, shape (n_mels, n_fft // 2 + 1)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, centre):
            fb[m - 1, k] = (k - left) / max(centre - left, 1)
        for k in range(centre, right):
            fb[m - 1, k] = (right - k) / max(right - centre, 1)
    return fb


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13, n_mels: int = 26, n_fft: int = 512) -> np.ndarray:
    """Mel-frequency cepstral coefficients, shape (frames, n_mfcc).

    Pre-emphasis -> frames -> Hamming -> power spectrum -> mel filterbank -> log -> DCT-II.
    """
    hop, flen = int(HOP_S * sr), int(FRAME_S * sr)
    frames = frame_signal(pre_emphasis(x), flen, hop) * hamming(flen)
    power = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2 / n_fft
    log_mel = np.log(power @ mel_filterbank(sr, n_fft, n_mels).T + EPS)
    k = np.arange(n_mfcc)[:, None]
    m = np.arange(n_mels)[None, :]
    dct = np.cos(np.pi * k * (2 * m + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    dct[0] /= np.sqrt(2.0)
    return log_mel @ dct.T

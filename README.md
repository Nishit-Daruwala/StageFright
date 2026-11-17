# StageFright AI

A real-time voice coach for public speaking. You give a short talk into the microphone. StageFright transcribes it live, measures your delivery, and tells you the result out loud. You can interrupt it with a question at any point and it answers from your own talk.

Speech Processing Lab mini-project, Christ University.

## How a session goes

1. Press **Start talk** and speak. Your words, loudness, pitch and pace appear live.
2. Say **"Hey StageFright, I have finished my speech."** The code word and the pause before it are cut out of the analysis. (The **Finish talk** button does the same.)
3. The coach speaks its feedback and the report appears: scores, an annotated timeline and charts.
4. Speak at any time to interrupt. The coach stops, answers, and offers to carry on.
5. Press **New take** to try again. The report then shows the change from your previous take.

## Run it

Use Chrome or Edge, and a headset if possible so the coach's voice stays out of the microphone.

**With Docker (macOS, Windows, Linux)**

```
cp .env.example .env        # then put your Groq key in .env
docker compose up --build
```

Open http://localhost:8000. The first start downloads about 800 MB of models into a Docker volume.

**Without Docker (development)**

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn server.main:app --port 8000
```

Without a Groq key the coach still works, using short built-in replies made from the measurements.

## How it works

```
Browser: mic capture (echo cancellation), coach audio playback, live meters, charts
   ⇅ WebSocket: 16 kHz PCM frames up; events and coach audio down
Server (FastAPI)
   ├─ Silero VAD          speech probability per 32 ms frame
   ├─ Whisper             each phrase is transcribed when it ends; word timestamps
   ├─ DSP (hand-written)  energy, zero-crossing rate, VAD, autocorrelation pitch, STFT, MFCC
   ├─ Emotion model       wav2vec2 classifier on 3-second windows
   ├─ Session             idle → talk → analysing → speaking ⇄ listening → thinking
   ├─ Coach               Groq-hosted language model with the measurements and transcript
   └─ Kokoro TTS          the coach's reply, one sentence at a time
```

Turn-taking:

- **During the talk** only the code word ends it, so you can pause as long as you like.
- **While the coach speaks**, 300 ms of your speech counts as an interruption. The server stops the audio, records which sentences were actually played, and tells the coach what it had not said yet.
- **When the coach is silent**, 160 ms of speech opens your turn and 700 ms of silence closes it.

## What is measured

| Measure | Method |
|---|---|
| Speech and pauses | Short-time energy and zero-crossing rate with a threshold that adapts to the recording; pauses are gaps of 250 ms or more |
| Pitch | Autocorrelation of 40 ms frames (75–400 Hz), centre clipping, median smoothing; variation in semitones |
| Loudness | RMS level in dB per frame; range between the 10th and 90th percentile of speech frames |
| Pace | Words per minute from Whisper word timestamps, overall and in 10-second windows |
| Filler words | Matched in the transcript; plus an acoustic detector for hesitation sounds the transcript missed |
| Emotion | wav2vec2 classifier per 3-second window (a rough indication on natural speech) |

Scores use fixed target ranges (for example 130–165 words per minute). They are a guide, not a grade.

## Files

| Path | Purpose |
|---|---|
| `server/dsp.py` | Hand-written signal processing |
| `server/metrics.py` | Delivery metrics, scores and chart data |
| `server/stt.py` | Whisper wrapper, code-word and filler detection |
| `server/vad.py` | Streaming VAD and turn detection |
| `server/session.py` | Session state machine, interruptions |
| `server/agent.py` | Coach prompt, Groq streaming, reply parsing |
| `server/tts.py`, `server/emotion.py` | Voice and emotion models |
| `web/` | Front end (plain HTML, CSS and JavaScript) |
| `kaggle/` | Emotion-model training script for Kaggle |
| `tests/`, `tools/simulate.py` | Unit tests and an end-to-end test that plays the part of the browser |

## Tests

```
.venv\Scripts\python -m pytest tests
.venv\Scripts\python -m tools.simulate      # needs the server running
```

`tools/simulate.py` synthesises a practice talk, streams it as live microphone audio, interrupts the coach with a spoken question and checks each step.

## Training the emotion model on Kaggle

1. Put your Kaggle API token in `~/.kaggle/kaggle.json`.
2. Replace `YOUR_KAGGLE_USERNAME` in `kaggle/kernel-metadata.json`.
3. Run:

```
kaggle kernels push -p kaggle
kaggle kernels status <username>/stagefright-emotion
kaggle kernels output <username>/stagefright-emotion -p kaggle/output
```

4. Copy `kaggle/output/emotion` to `models/emotion`. The server uses it on the next start; until then it uses the public model `superb/wav2vec2-base-superb-er`.

The script fine-tunes `wav2vec2-base` on RAVDESS, CREMA-D, TESS and SAVEE, holds out whole speakers for testing, and saves the accuracy, macro F1 and a confusion matrix.

## Settings

Set in `.env` or the environment:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | Key for the coach's language model |
| `GROQ_MODEL` | chosen from the account's model list | Model ID to use |
| `WHISPER_MODEL` | `base.en` | Use `small.en` on a faster machine |
| `KOKORO_VOICE` | `af_heart` | Coach voice |

Only practise with your own voice or with the speaker's consent; the talk is processed on your machine and only the transcript and measurements are sent to the language model.

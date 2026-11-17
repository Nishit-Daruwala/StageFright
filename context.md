# StageFright AI — project context

Last updated: 2026-11-17

## What this is

Mini-project for the Speech Processing Lab (Christ University, Trimester 5), demonstrated live at the Speech Processing Project Showcase / Innovation Expo.

StageFright AI is a real-time voice coach for public speaking and interviews. You speak into the microphone, it listens and transcribes live, and when you stop it analyses your delivery and tells you the result out loud. You can interrupt it mid-sentence with a question and it answers with full knowledge of your talk.

## Marking (25)

| Criterion | Marks |
|---|---|
| Execution of speech processing tasks | 10 |
| Correctness of code / algorithm | 4 |
| Concept clarity (viva) | 5 |
| Presentation and neatness | 3 |
| Innovation and real-world application | 3 |

## Fixed decisions

- Input is the live microphone only. No file upload.
- Live transcript while speaking (streaming STT).
- The first talk ends when the speaker says the code word, "Hey StageFright, I have finished my speech" (not a silence timeout). Matching is fuzzy on the live transcript ("stage fright" as one or two words, plus "finished"). The code word and the silence before it are cut out of the analysis.
- During the conversation with the agent, turn-taking uses normal VAD thresholds (about 300 ms of speech to interrupt, about 700 ms of silence to end a turn).
- Feedback is spoken by a TTS voice agent.
- The speaker can interrupt the agent (barge-in); the agent answers in context and can resume.
- Training / fine-tuning runs on Kaggle through the Kaggle CLI.
- Development happens on the Windows laptop, in `.venv`.
- The demo runs on a MacBook Pro M1 Max with 32 GB RAM, with internet available, from a Docker container (`docker compose up --build`).
- Models must be free. OpenAI is not used unless a model there costs nothing.
- The project uses its own virtual environment at `.venv`.
- This file is kept up to date as the project changes.

## Machines

- Demo: MacBook Pro M1 Max, 32 GB RAM.
- This Windows laptop: Windows 11, Intel Iris Xe, 16 GB RAM, Python 3.13 only. `.venv` here is built on 3.13.
- Docker 28.3 and Compose 2.38 are installed here.
- Not installed yet: Kaggle CLI (install inside `.venv`).

## Docker notes

- The browser captures the mic and plays the audio, so the container needs no audio devices. Open `http://localhost:8000`; the browser allows mic access on localhost.
- The image builds natively for the host (arm64 on the Mac). Every package added to `requirements.txt` must have a Linux arm64 build.
- Docker on macOS cannot use the Mac's GPU, so Whisper, Kokoro and the emotion model run on CPU inside the container.
- Model weights are kept in the `models` volume so they download once.
- `GROQ_API_KEY` goes in `.env` (see `.env.example`).

## Architecture

```
Browser (mic capture with echo cancellation, audio playback, live plots)
   ⇅ WebSocket: 16 kHz PCM frames up, events + TTS audio down
FastAPI server
   ├─ Streaming VAD (Silero)         speech start / end, barge-in trigger
   ├─ STT (faster-whisper)           per-utterance transcription, word timestamps
   ├─ DSP (hand-written)             energy, ZCR, F0, pauses, pace — incremental
   ├─ Emotion (wav2vec2, Kaggle)     3 s sliding windows
   ├─ Session state machine          idle → talk → analysing → speaking ⇄ listening → thinking
   ├─ Coach agent (LLM, streaming)   metrics + transcript + history + inline tags
   └─ TTS (local, sentence-by-sentence)
```

## Repo layout

```
context.md, README.md
server/
  main.py          FastAPI app, WebSocket endpoint
  config.py        settings and thresholds (read from .env)
  models.py        background model loading and status
  session.py       state machine, turn-taking, barge-in
  vad.py           streaming Silero VAD, turn detector
  stt.py           Whisper wrapper, code word, filler detection
  dsp.py           framing, energy, ZCR, VAD, autocorrelation F0, STFT, MFCC (hand-written)
  metrics.py       pauses, pace, pitch and loudness statistics, scoring, plot data
  emotion.py       windowed wav2vec2 inference
  agent.py         coach prompt, Groq streaming, reply parser, built-in replies
  tts.py           Kokoro sentence synthesis
web/               index.html, style.css, app.js, worklet.js
kaggle/            train_emotion.py + kernel-metadata.json
tests/             unit tests (pytest)
tools/simulate.py  end-to-end test that plays the browser's part
```

## How things work (implementation notes)

- Live STT: the Silero VAD cuts the talk into phrases at 500 ms pauses (max 12 s); each phrase is transcribed when it ends. The code word is searched in the accumulated words.
- Whisper prompt is the short "Um, uh, like, you know." A longer trailing prompt made Whisper append invented words at phrase ends; trailing words under 0.15 confidence are also dropped.
- At the code word the audio is cut 0.15 s after the last real word.
- The coach's "tools" are inline tags in its reply, `[[show:NAME]]` and `[[play:START-END]]`, stripped before speech. This works with any provider and with streaming.
- The browser reports `playing` / `played` per sentence, so on an interruption the server knows what was actually said. The unsaid remainder is passed to the coach as a note with the question.
- An unintelligible interruption (cough, noise) makes the coach resume the unsaid remainder without calling the LLM.
- Without `GROQ_API_KEY`, or if Groq fails, built-in template replies are used and a notice is shown.
- If `GROQ_MODEL` is unset, the model is picked from the account's `/models` list.
- Emotion model: `models/emotion` if it holds a fine-tuned model, otherwise `superb/wav2vec2-base-superb-er`.
- Windows: stop the dev server by killing the python process whose command line contains `uvicorn`; killing by port did not work.

## Status

- [x] Idea chosen and plan agreed (real-time voice-agent version)
- [x] `.venv` created (Python 3.13); all packages install on Windows and resolve for Linux arm64
- [x] Server skeleton, Dockerfile and compose file
- [x] Audio loop: browser mic → WebSocket → VAD → live transcript
- [x] DSP core with unit tests (15 passing)
- [x] Analysis, report, spoken feedback
- [x] Barge-in, contextual questions, resume
- [x] Report charts and live meters
- [x] End-to-end simulation passes (`tools/simulate.py`) with built-in replies
- [ ] Try it in a real browser with a real microphone (not done yet: only simulated)
- [ ] Add `GROQ_API_KEY` and test the LLM coach (never run against Groq yet)
- [ ] Fine-tune the emotion model on Kaggle (needs Kaggle credentials and the username in `kaggle/kernel-metadata.json`; script is untested)
- [ ] Run the container on the Mac
- [ ] Validation table (DSP against Praat / manual counts on real recordings), tune score ranges and VAD thresholds, rehearsal

Measured on the Windows laptop (CPU): coach stops 0.4 s after an interruption begins; first spoken sentence about 1.2 s after the report; code word recognised about 1–2 s after it is said.

## Models (chosen 2026-11-17)

| Part | Choice |
|---|---|
| STT | Whisper run locally (`faster-whisper`) |
| TTS | Kokoro run locally |
| LLM | Groq free tier through an OpenAI-compatible client |
| Emotion | wav2vec2 fine-tuned on Kaggle, run locally |

Notes:
- Groq's free daily token cap (about 200K tokens/day on its larger models, per third-party reports) covers development but may run out on expo day. Free backups: Gemini Flash-Lite (reported 500 requests/day) or a second Groq model with its own cap. Switching provider should be a config change.
- Gemini's free Flash quota was reported cut to 20 requests/day in October 2026.
- Do not use a speech-to-speech API (OpenAI Realtime, Gemini Live): it would replace the VAD, STT and TTS pipeline that the marks are for.
- All quota and price figures came from third-party pages and must be checked on the providers' own pages.

## Open choices

- Which Groq model to use (check the current list and limits in the Groq console; set `GROQ_MODEL`).
- `WHISPER_MODEL`: `base.en` here; consider `small.en` on the Mac.

## Change log

- 2026-11-17: Chose StageFright AI over VoiceShield (deepfake detector).
- 2026-11-17: Dropped file upload. Redesigned as a real-time voice agent with live mic, spoken feedback and barge-in.
- 2026-11-17: End of the first talk is now a spoken code word; VAD thresholds apply only in the conversation. Demo machine is a MacBook Pro M1 Max. Model options analysed.
- 2026-11-17: Chose local Whisper, local Kokoro and Groq free tier; no paid OpenAI. Code word set. Development on Windows, demo on the Mac through Docker. Server skeleton and Docker setup added.
- 2026-11-17: Built the whole application: DSP, metrics, STT, VAD, emotion, TTS, coach agent, session state machine, front end, tests, simulator, Kaggle training script, README.

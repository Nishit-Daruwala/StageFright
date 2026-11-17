/* StageFright AI front end: mic capture, WebSocket, coach audio playback and the report charts. */

const $ = (id) => document.getElementById(id);

const C = {
  surface: '#1a1a19', page: '#0d0d0d', ink: '#f0efec', ink2: '#c3c2b7', muted: '#898781',
  grid: '#2c2c2a', axis: '#383835',
  blue: '#3987e5', orange: '#d95926', aqua: '#199e70', yellow: '#c98500',
  magenta: '#d55181', violet: '#9085e9', red: '#e66767',
};
const EMOTION_COLORS = {
  neutral: C.muted, happy: C.yellow, angry: C.red, sad: C.blue,
  calm: C.aqua, fearful: C.violet, surprised: C.magenta, disgust: C.orange,
};
const STATE_LABELS = {
  idle: 'Ready', talk: 'Listening to your talk', analysing: 'Analysing', speaking: 'Coach speaking',
  listening: 'Your turn', thinking: 'Thinking',
};
const TABS = [['timeline', 'Timeline'], ['pitch', 'Pitch'], ['pace', 'Pace'], ['energy', 'Loudness'],
  ['spectrogram', 'Spectrogram'], ['mfcc', 'MFCC'], ['scores', 'Scores']];
const SCORE_NAMES = { pace: 'Pace', fillers: 'Filler words', pauses: 'Pausing', pitch: 'Pitch variety', energy: 'Loudness variety' };

let ws = null;
let state = 'idle';
let modelsReady = false;
let micCtx = null;
let playCtx = null;
let pendingHeader = null;

// Coach playback: items arrive in order and are played one at a time.
let queue = [];
let current = null;
let turnId = null;
let turnEnded = false;
let coachBubble = null;

let report = null;
let previous = null;
let activeTab = 'timeline';
let clipRange = null;
const levels = [];

/* ------------------------------------------------------------------ connection */

function connect() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = onMessage;
  ws.onclose = () => {
    setState('idle', 'Reconnecting');
    $('start').disabled = true;
    stopPlayback();
    setTimeout(connect, 1500);
  };
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
}

function onMessage(ev) {
  if (typeof ev.data !== 'string') {
    if (pendingHeader) {
      const item = { ...pendingHeader, audio: ev.data };
      pendingHeader = null;
      enqueue(item);
    }
    return;
  }
  const msg = JSON.parse(ev.data);
  switch (msg.type) {
    case 'models': showModels(msg); break;
    case 'state': setState(msg.state); break;
    case 'transcript': showTranscript(msg.words); break;
    case 'meter': showMeter(msg); break;
    case 'report': report = msg.report; previous = msg.previous; showReport(); break;
    case 'notice': showNotice(msg.text); break;
    case 'tts':
      if (msg.audio === false) enqueue({ ...msg, audio: null }); else pendingHeader = msg;
      break;
    case 'clip': pendingHeader = msg; break;
    case 'cue': enqueue(msg); break;
    case 'turn_end':
      if (msg.turn !== turnId) beginTurn(msg.turn);
      turnEnded = true;
      maybeDone();
      break;
    case 'stop_audio': stopPlayback(); break;
    case 'barge_in':
      if (coachBubble) addTag(coachBubble, 'interrupted');
      coachBubble = null;
      break;
    case 'user': if (msg.text) addBubble('user', 'You', msg.text); break;
    default: break;
  }
}

/* ------------------------------------------------------------------ microphone */

async function startMic() {
  if (micCtx) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    // Echo cancellation keeps the coach's voice from triggering an interruption.
    // Gain control stays off so loudness measurements reflect how the speaker really spoke.
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false, channelCount: 1 },
  });
  micCtx = new AudioContext({ sampleRate: 16000 });
  await micCtx.audioWorklet.addModule('worklet.js');
  const node = new AudioWorkletNode(micCtx, 'pcm-sender');
  micCtx.createMediaStreamSource(stream).connect(node);
  node.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
    pushLevel(new Int16Array(e.data));
  };
  playCtx = new AudioContext();
}

function pushLevel(block) {
  let peak = 0;
  for (let i = 0; i < block.length; i += 4) peak = Math.max(peak, Math.abs(block[i]));
  levels.push(peak / 32768);
  if (levels.length > 300) levels.shift();
}

function drawWave() {
  const cv = $('wave');
  const g = cv.getContext('2d');
  const w = cv.width, h = cv.height, mid = h / 2;
  g.clearRect(0, 0, w, h);
  const live = state === 'talk' || state === 'listening' || state === 'speaking';
  g.fillStyle = state === 'talk' ? C.blue : (live ? C.aqua : C.axis);
  const bw = w / 300;
  for (let i = 0; i < levels.length; i++) {
    const a = Math.max(1, Math.sqrt(levels[i]) * (h - 8) / 2);
    g.fillRect((300 - levels.length + i) * bw, mid - a, Math.max(1, bw - 1), a * 2);
  }
  requestAnimationFrame(drawWave);
}

/* ------------------------------------------------------------------ status */

function setState(s, label) {
  state = s;
  const pill = $('state');
  pill.dataset.state = s;
  pill.textContent = label || STATE_LABELS[s] || s;
  $('finish').hidden = s !== 'talk';
  $('start').textContent = s === 'idle' && !report ? 'Start talk' : (s === 'talk' ? 'Restart' : 'New take');
  if (s === 'talk') {
    $('transcript').textContent = '';
    $('chat').textContent = '';
    for (const id of ['m-level', 'm-pitch', 'm-pace']) $(id).textContent = '–';
    $('m-time').textContent = '0:00';
  }
}

function showModels(msg) {
  modelsReady = msg.ready;
  const names = { vad: 'Voice detection', stt: 'Speech-to-text', tts: 'Coach voice', emotion: 'Emotion model' };
  const box = $('models');
  box.textContent = '';
  for (const [key, label] of Object.entries(names)) {
    const st = msg.status[key] || 'loading';
    const el = document.createElement('span');
    el.dataset.s = st.startsWith('error') ? 'error' : st;
    el.textContent = `${label}: ${st.startsWith('error') ? 'unavailable' : st}`;
    if (st.startsWith('error')) el.title = st;
    box.appendChild(el);
  }
  const llm = document.createElement('span');
  llm.dataset.s = msg.llm ? 'ready' : 'error';
  llm.textContent = `Language model: ${msg.llm ? 'Groq key set' : 'no key, built-in replies'}`;
  box.appendChild(llm);
  $('start').disabled = !modelsReady;
  if (state === 'idle') setState('idle', modelsReady ? 'Ready' : 'Loading models');
}

function showNotice(text) {
  const el = $('notice');
  el.textContent = text;
  el.hidden = false;
  clearTimeout(showNotice.timer);
  showNotice.timer = setTimeout(() => { el.hidden = true; }, 9000);
}

const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

function showMeter(m) {
  $('m-time').textContent = fmtTime(m.t);
  $('m-level').textContent = m.level > -90 ? Math.round(m.level) : '–';
  $('m-pitch').textContent = m.f0 ?? '–';
  $('m-pace').textContent = m.wpm ?? '–';
}

const FILLER_RE = /^(um+|uh+|uhm|er|erm|ah|hmm|mm|like|basically|actually|literally)$/;

function showTranscript(words) {
  const box = $('transcript');
  box.textContent = '';
  for (const w of words) {
    const bare = w.word.toLowerCase().replace(/[^a-z]/g, '');
    const el = FILLER_RE.test(bare) ? document.createElement('mark') : document.createElement('span');
    el.textContent = w.word;
    box.append(el, ' ');
  }
  box.scrollTop = box.scrollHeight;
}

/* ------------------------------------------------------------------ coach playback */

function addBubble(kind, who, text) {
  const el = document.createElement('div');
  el.className = `bubble ${kind}`;
  const label = document.createElement('span');
  label.className = 'who';
  label.textContent = who;
  const body = document.createElement('span');
  body.className = 'body';
  body.textContent = text;
  el.append(label, body);
  $('chat').appendChild(el);
  $('chat').scrollTop = $('chat').scrollHeight;
  return el;
}

function addTag(bubble, text) {
  const tag = document.createElement('span');
  tag.className = 'tag';
  tag.textContent = text;
  bubble.append(document.createElement('br'), tag);
}

function beginTurn(id) {
  turnId = id;
  turnEnded = false;
  queue = [];
  coachBubble = null;
}

function enqueue(item) {
  if (item.turn !== turnId) beginTurn(item.turn);
  queue.push(item);
  pump();
}

async function pump() {
  if (current) return;
  if (!queue.length) { maybeDone(); return; }
  const item = queue.shift();
  current = item;
  send({ type: 'playing', turn: item.turn, id: item.id });

  if (item.type === 'cue') {
    if (item.action === 'show') selectTab(item.name);
    finishItem(item);
    return;
  }
  if (item.type === 'tts') {
    if (!coachBubble) coachBubble = addBubble('coach', 'Coach', '');
    coachBubble.querySelector('.body').textContent += `${item.text} `;
    $('chat').scrollTop = $('chat').scrollHeight;
  } else if (item.type === 'clip') {
    clipRange = [item.start, item.end];
    selectTab('timeline');
    if (coachBubble) addTag(coachBubble, `replaying your talk, ${fmtTime(item.start)} to ${fmtTime(item.end)}`);
  }
  if (!item.audio || !playCtx) {
    setTimeout(() => { if (current === item) finishItem(item); }, 1200);
    return;
  }
  try {
    const buffer = await playCtx.decodeAudioData(item.audio.slice(0));
    if (current !== item) return;                 // stopped while decoding
    const src = playCtx.createBufferSource();
    src.buffer = buffer;
    src.connect(playCtx.destination);
    src.onended = () => { if (current === item) finishItem(item); };
    item.src = src;
    src.start();
  } catch (err) {
    console.error('playback failed', err);
    if (current === item) finishItem(item);
  }
}

function finishItem(item) {
  send({ type: 'played', turn: item.turn, id: item.id });
  if (item.type === 'clip') { clipRange = null; if (activeTab === 'timeline') drawPlot(); }
  current = null;
  pump();
}

function maybeDone() {
  if (turnEnded && !current && !queue.length && turnId !== null) {
    turnEnded = false;
    coachBubble = null;
    send({ type: 'turn_done', turn: turnId });
  }
}

function stopPlayback() {
  queue = [];
  pendingHeader = null;
  turnEnded = false;
  const item = current;
  current = null;
  if (item && item.src) { try { item.src.stop(); } catch (e) { /* already stopped */ } }
  if (clipRange) { clipRange = null; if (report && activeTab === 'timeline') drawPlot(); }
}

/* ------------------------------------------------------------------ report */

function tile(label, value, unit, sub) {
  const el = document.createElement('div');
  el.className = 'tile';
  el.innerHTML = '<span class="label"></span><span class="value"></span><span class="unit"></span><span class="sub"></span>';
  el.children[0].textContent = label;
  el.children[1].textContent = value;
  el.children[2].textContent = unit || '';
  el.children[3].textContent = sub || '';
  return el;
}

function showReport() {
  $('report').hidden = false;
  const s = report.summary;
  $('overall').textContent = report.overall;
  const d = previous ? report.overall - previous.overall : null;
  $('overall-delta').textContent = d === null ? '' : `${d > 0 ? '▲' : d < 0 ? '▼' : '■'} ${Math.abs(d)} vs previous take`;

  const bars = $('score-bars');
  bars.textContent = '';
  for (const [key, name] of Object.entries(SCORE_NAMES)) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const delta = previous ? report.scores[key] - previous.scores[key] : null;
    row.innerHTML = '<span class="name"></span><div class="track"><div class="fill"></div></div><span class="num"></span>';
    row.children[0].textContent = name;
    row.querySelector('.fill').style.width = `${report.scores[key]}%`;
    row.children[2].textContent = report.scores[key];
    if (delta !== null) {
      const small = document.createElement('small');
      small.textContent = ` ${delta > 0 ? '▲' : delta < 0 ? '▼' : '■'}${Math.abs(delta)}`;
      row.children[2].appendChild(small);
    }
    bars.appendChild(row);
  }

  const stats = $('stats');
  stats.textContent = '';
  stats.append(
    tile('Duration', fmtTime(s.duration_s), '', `${s.words} words`),
    tile('Pace', s.wpm, 'wpm', 'target 130–165'),
    tile('Filler words', s.filler_count, '', `${s.fillers_per_min} per minute`),
    tile('Pauses', `${Math.round(s.pause_ratio * 100)}%`, 'of time', `longest ${s.longest_pause_s} s`),
    tile('Pitch variety', s.pitch_std_st, 'semitones', `median ${s.pitch_median_hz} Hz`),
    tile('Loudness range', s.level_range_db, 'dB', 'target 14 or more'),
  );

  const tabs = $('tabs');
  tabs.textContent = '';
  for (const [key, label] of TABS) {
    const b = document.createElement('button');
    b.textContent = label;
    b.dataset.tab = key;
    b.setAttribute('role', 'tab');
    b.onclick = () => selectTab(key);
    tabs.appendChild(b);
  }
  selectTab('timeline');
  $('report').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function selectTab(name) {
  if (!report || !TABS.some(([k]) => k === name)) return;
  activeTab = name;
  for (const b of $('tabs').children) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  drawPlot();
}

function baseLayout(xTitle, yTitle) {
  const axis = { gridcolor: C.grid, linecolor: C.axis, zerolinecolor: C.axis, tickfont: { color: C.muted }, title: { font: { color: C.ink2, size: 12 } } };
  return {
    paper_bgcolor: C.surface, plot_bgcolor: C.surface,
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: C.ink2, size: 12 },
    margin: { l: 56, r: 16, t: 34, b: 44 },
    xaxis: { ...axis, title: { ...axis.title, text: xTitle } },
    yaxis: { ...axis, title: { ...axis.title, text: yTitle } },
    hovermode: 'x unified',
    hoverlabel: { bgcolor: C.page, bordercolor: C.axis, font: { color: C.ink } },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { color: C.ink2 } },
    shapes: [], annotations: [],
  };
}

const band = (x0, x1, color, opacity) => ({
  type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, fillcolor: color, opacity, line: { width: 0 }, layer: 'below',
});

function drawPlot() {
  const r = report;
  const hop = r.series.hop;
  const t = r.series.env.map((_, i) => +(i * hop).toFixed(2));
  let traces = [];
  let layout = baseLayout('Time (s)', '');
  let note = '';

  if (activeTab === 'timeline') {
    const top = Math.max(...r.series.env, 0.01);
    traces.push({ x: t, y: r.series.env, type: 'scatter', mode: 'lines', name: 'Waveform envelope',
      line: { color: C.blue, width: 1 }, fill: 'tozeroy', fillcolor: 'rgba(57,135,229,0.35)', hoverinfo: 'skip' });
    const seen = new Set();
    for (const e of r.emotions) {
      if (e.label === 'silence') continue;
      const color = EMOTION_COLORS[e.label] || C.muted;
      layout.shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: e.t0, x1: e.t1, y0: 0, y1: 0.07,
        fillcolor: color, line: { width: 1, color: C.surface } });
      if (!seen.has(e.label)) {
        seen.add(e.label);
        traces.push({ x: [null], y: [null], type: 'scatter', mode: 'markers', name: `Emotion: ${e.label}`,
          marker: { color, symbol: 'square', size: 10 } });
      }
    }
    for (const p of r.pauses) if (p.dur >= 0.5) layout.shapes.push(band(p.start, p.end, C.muted, 0.16));
    if (r.fillers.length) {
      traces.push({ x: r.fillers.map((f) => f.start), y: r.fillers.map(() => top * 1.08), type: 'scatter',
        mode: 'markers', name: 'Filler word', text: r.fillers.map((f) => `"${f.text}"`),
        hovertemplate: '%{text} at %{x:.1f} s<extra></extra>',
        marker: { color: C.orange, size: 10, symbol: 'diamond', line: { color: C.surface, width: 2 } } });
    }
    if (r.hesitations.length) {
      traces.push({ x: r.hesitations.map((f) => f.start), y: r.hesitations.map(() => top * 1.2), type: 'scatter',
        mode: 'markers', name: 'Possible hesitation sound', hovertemplate: 'at %{x:.1f} s<extra></extra>',
        marker: { color: C.violet, size: 9, symbol: 'circle', line: { color: C.surface, width: 2 } } });
    }
    if (clipRange) layout.shapes.push(band(clipRange[0], clipRange[1], C.ink, 0.22));
    layout.yaxis.title.text = 'Amplitude';
    layout.hovermode = 'closest';
    note = 'Grey bands are pauses of half a second or more. The strip along the bottom is the emotion model\'s label for each 3-second window.';
  } else if (activeTab === 'pitch') {
    traces.push({ x: t, y: r.series.f0, type: 'scatter', mode: 'lines', name: 'Pitch', connectgaps: false,
      line: { color: C.blue, width: 2 }, hovertemplate: '%{y:.0f} Hz<extra></extra>' });
    for (const f of r.flat_stretches) {
      layout.shapes.push(band(f.start, f.end, C.orange, 0.14));
      layout.annotations.push({ x: (f.start + f.end) / 2, y: 1, yref: 'paper', text: 'monotone', showarrow: false, font: { color: C.ink2, size: 11 } });
    }
    layout.yaxis.title.text = 'Fundamental frequency (Hz)';
    note = `Pitch from autocorrelation, voiced frames only. Variation: ${r.summary.pitch_std_st} semitones around a median of ${r.summary.pitch_median_hz} Hz.`;
  } else if (activeTab === 'pace') {
    traces.push({ x: r.windows.map((w) => (w.t0 + w.t1) / 2), y: r.windows.map((w) => w.wpm), type: 'scatter',
      mode: 'lines+markers', name: 'Pace', line: { color: C.blue, width: 2 }, marker: { size: 8, color: C.blue },
      hovertemplate: '%{y} wpm<extra></extra>' });
    layout.shapes.push({ type: 'rect', xref: 'paper', yref: 'y', x0: 0, x1: 1, y0: 130, y1: 165, fillcolor: C.aqua, opacity: 0.14, line: { width: 0 }, layer: 'below' });
    layout.annotations.push({ x: 1, xref: 'paper', xanchor: 'right', y: 165, yanchor: 'bottom', text: 'target 130–165', showarrow: false, font: { color: C.ink2, size: 11 } });
    layout.yaxis.title.text = 'Words per minute';
    layout.yaxis.rangemode = 'tozero';
    note = `Pace in 10-second windows. Average ${r.summary.wpm} words per minute; ${r.summary.articulation_wpm} when pauses are left out.`;
  } else if (activeTab === 'energy') {
    traces.push({ x: t, y: r.series.energy.map((v, i) => (r.series.speech[i] ? v : null)), type: 'scatter', mode: 'lines',
      name: 'Level', connectgaps: false, line: { color: C.blue, width: 2 }, hovertemplate: '%{y:.0f} dB<extra></extra>' });
    layout.yaxis.title.text = 'Short-time energy (dB full scale)';
    note = `Loudness while speaking. The range between quiet and loud speech was ${r.summary.level_range_db} dB.`;
  } else if (activeTab === 'spectrogram') {
    traces.push({ z: r.spectrogram.z, x0: 0, dx: r.spectrogram.dt, y0: 0, dy: r.spectrogram.df, type: 'heatmap',
      colorscale: [[0, C.surface], [0.35, '#104281'], [0.7, '#3987e5'], [1, '#cde2fb']],
      colorbar: { title: { text: 'dB', font: { color: C.ink2 } }, tickfont: { color: C.muted }, thickness: 12 },
      hovertemplate: '%{x:.1f} s, %{y:.0f} Hz: %{z:.0f} dB<extra></extra>' });
    layout.yaxis.title.text = 'Frequency (Hz)';
    layout.hovermode = 'closest';
    note = 'Short-time Fourier transform: 25 ms Hamming windows, 10 ms hop, after pre-emphasis.';
  } else if (activeTab === 'mfcc') {
    const flat = r.mfcc.z.flat();
    const lim = Math.max(...flat.map(Math.abs), 1);
    traces.push({ z: r.mfcc.z, x0: 0, dx: r.mfcc.dt, y0: 1, dy: 1, type: 'heatmap', zmin: -lim, zmax: lim,
      colorscale: [[0, '#3987e5'], [0.5, '#383835'], [1, '#e66767']],
      colorbar: { tickfont: { color: C.muted }, thickness: 12 },
      hovertemplate: '%{x:.1f} s, coefficient %{y}: %{z:.1f}<extra></extra>' });
    layout.yaxis.title.text = 'MFCC coefficient';
    layout.hovermode = 'closest';
    note = 'Mel-frequency cepstral coefficients 1 to 12: mel filterbank, log, then discrete cosine transform.';
  } else if (activeTab === 'scores') {
    const keys = Object.keys(SCORE_NAMES);
    const names = keys.map((k) => SCORE_NAMES[k]);
    traces.push({ y: names, x: keys.map((k) => r.scores[k]), type: 'bar', orientation: 'h', name: 'This take',
      marker: { color: C.blue }, text: keys.map((k) => r.scores[k]), textposition: 'outside',
      textfont: { color: C.ink }, hovertemplate: '%{y}: %{x}<extra>This take</extra>' });
    if (previous) {
      traces.push({ y: names, x: keys.map((k) => previous.scores[k]), type: 'bar', orientation: 'h', name: 'Previous take',
        marker: { color: C.orange }, text: keys.map((k) => previous.scores[k]), textposition: 'outside',
        textfont: { color: C.ink }, hovertemplate: '%{y}: %{x}<extra>Previous take</extra>' });
    }
    layout.barmode = 'group';
    layout.bargap = 0.35;
    layout.xaxis.range = [0, 110];
    layout.xaxis.title.text = 'Score out of 100';
    layout.yaxis.autorange = 'reversed';
    layout.margin.l = 120;
    layout.hovermode = 'closest';
    layout.showlegend = Boolean(previous);
    note = 'Scores use fixed target ranges for each measure. They are a guide, not a grade.';
  }

  $('plot-note').textContent = note;
  Plotly.react('plot', traces, layout, { displayModeBar: false, responsive: true });
}

/* ------------------------------------------------------------------ start-up */

$('start').onclick = async () => {
  try {
    await startMic();
    await playCtx.resume();
    await micCtx.resume();
  } catch (err) {
    showNotice(`Microphone unavailable: ${err.message}`);
    return;
  }
  stopPlayback();
  send({ type: 'start' });
};
$('finish').onclick = () => send({ type: 'stop' });

connect();
requestAnimationFrame(drawWave);

// Runs on the audio thread. Converts the mic stream to 16 kHz, 16-bit mono and posts it
// to the page in 512-sample blocks (32 ms), the frame size the server's VAD expects.
class PcmSender extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Int16Array(512);
    this.fill = 0;
    this.ratio = sampleRate / 16000;   // 1 when the context already runs at 16 kHz
    this.pos = 0;
    this.prev = 0;
  }

  push(sample) {
    const s = Math.max(-1, Math.min(1, sample));
    this.block[this.fill++] = s < 0 ? s * 32768 : s * 32767;
    if (this.fill === this.block.length) {
      this.port.postMessage(this.block.buffer.slice(0));
      this.fill = 0;
    }
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < input.length; i++) this.push(input[i]);
      return true;
    }
    // Linear-interpolation resampler for browsers that ignore the requested sample rate.
    for (let i = 0; i < input.length; i++) {
      const cur = input[i];
      while (this.pos < 1) {
        this.push(this.prev + (cur - this.prev) * this.pos);
        this.pos += this.ratio;
      }
      this.pos -= 1;
      this.prev = cur;
    }
    return true;
  }
}

registerProcessor('pcm-sender', PcmSender);

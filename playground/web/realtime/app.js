/**
 * WIRE SERVICE — sglang-omni /v1/realtime
 *
 * Captures mic → 16 kHz mono PCM16 via AudioWorklet → base64-encodes →
 * sends `input_audio_buffer.append`. Renders incoming `transcription.delta`
 * events as italic Newsreader paragraphs with serif drop-caps; the
 * server-VAD pill pulses while a turn is in flight; an oscilloscope
 * canvas shows the live mic waveform.
 *
 * Vanilla — no framework, no build step, no error handling. Per house
 * style: if something fails, the browser console gets the exception.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // ─────────────────────  DOM refs  ─────────────────────
  const wsUrlEl       = $("ws-url");
  const modeEl        = $("mode");
  const instructionsEl = $("instructions");
  const connectBtn    = $("connect");
  const disconnectBtn = $("disconnect");
  const statusEl      = $("status");
  const statusDotEl   = $("status-dot");
  const livePillEl    = $("live-pill");
  const liveTextEl    = livePillEl.querySelector(".live-text");

  const micStartBtn   = $("mic-start");
  const micStopBtn    = $("mic-stop");
  const commitBtn     = $("commit");
  const clearBufferBtn = $("clear-buffer");
  const micStatusEl   = $("mic-status");
  const oscilloCanvas = $("oscilloscope");
  const oscilloCtx    = oscilloCanvas.getContext("2d");

  const transcriptsEl = $("transcripts");
  const logEl         = $("log");
  const logDeltasEl   = $("log-deltas");
  const clearLogBtn   = $("clear-log");

  // ─────────────────────  State  ─────────────────────
  let ws = null;
  let audioCtx = null;
  let micStream = null;
  let workletNode = null;
  let analyserNode = null;
  let drawRaf = 0;
  let utteranceCounter = 0;
  const utteranceNodes = new Map();   // item_id → DOM node
  const utteranceSerials = new Map(); // item_id → serial string
  const TARGET_SR = 16000;

  // ─────────────────────  Status helpers  ─────────────────────

  function setStatus(text, mode = "") {
    statusEl.textContent = text;
    statusDotEl.className = "status-dot" + (mode ? " " + mode : "");
  }

  function setLive(on) {
    if (on) {
      livePillEl.classList.add("on");
      liveTextEl.textContent = "ON THE WIRE";
    } else {
      livePillEl.classList.remove("on");
      liveTextEl.textContent = "OFFLINE";
    }
  }

  function setMicStatus(text) {
    micStatusEl.textContent = text;
  }

  // ─────────────────────  Wire feed log  ─────────────────────

  function logEntry(direction, payload) {
    const t = payload && payload.type;
    if (
      t === "conversation.item.input_audio_transcription.delta" &&
      !logDeltasEl.checked
    ) {
      return;
    }
    const arrow = direction === "in" ? "←" : "→";
    const cls = direction === "in" ? "arrow-down" : "arrow-up";
    const ts = new Date().toLocaleTimeString("en-GB", { hour12: false });
    const summary = JSON.stringify(payload).slice(0, 280);
    const line = document.createElement("div");
    line.className = "row";
    line.innerHTML =
      `<span class="ts">${ts}</span>` +
      `<span class="${cls}">${arrow}</span> ` +
      escapeHtml(summary);
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ─────────────────────  WebSocket  ─────────────────────

  function wsSend(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
    logEntry("out", payload);
  }

  connectBtn.addEventListener("click", () => {
    const url = wsUrlEl.value.trim();
    ws = new WebSocket(url);
    setStatus("Opening line…");

    ws.onopen = () => {
      setStatus("Wire open", "connected");
      setLive(true);
      connectBtn.disabled = true;
      disconnectBtn.disabled = false;
      micStartBtn.disabled = false;

      const sessionConfig = {
        modalities: ["text"],
        input_audio_format: "pcm16",
        instructions: instructionsEl.value,
      };
      if (modeEl.value === "server_vad") {
        sessionConfig.turn_detection = {
          type: "server_vad",
          threshold: 0.5,
          silence_duration_ms: 600,
          prefix_padding_ms: 200,
        };
      }
      wsSend({ type: "session.update", session: sessionConfig });
    };

    ws.onmessage = (ev) => {
      const evt = JSON.parse(ev.data);
      logEntry("in", evt);
      handleServerEvent(evt);
    };

    ws.onclose = () => {
      setStatus("Standing by");
      setLive(false);
      connectBtn.disabled = false;
      disconnectBtn.disabled = true;
      micStartBtn.disabled = true;
      micStopBtn.disabled = true;
      commitBtn.disabled = true;
      clearBufferBtn.disabled = true;
      stopMic();
      ws = null;
    };

    ws.onerror = () => {
      setStatus("Wire error", "error");
    };
  });

  disconnectBtn.addEventListener("click", () => {
    if (ws) ws.close();
  });

  // ─────────────────────  Microphone  ─────────────────────

  micStartBtn.addEventListener("click", () => startMic());
  micStopBtn.addEventListener("click", () => stopMic());
  commitBtn.addEventListener("click", () =>
    wsSend({ type: "input_audio_buffer.commit" }),
  );
  clearBufferBtn.addEventListener("click", () =>
    wsSend({ type: "input_audio_buffer.clear" }),
  );

  async function startMic() {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: TARGET_SR,
    });

    const source = audioCtx.createMediaStreamSource(micStream);

    const workletCode = `
      class Forwarder extends AudioWorkletProcessor {
        process(inputs) {
          const input = inputs[0];
          if (input && input[0]) {
            this.port.postMessage(input[0]);
          }
          return true;
        }
      }
      registerProcessor('forwarder', Forwarder);
    `;
    const blob = new Blob([workletCode], { type: "application/javascript" });
    await audioCtx.audioWorklet.addModule(URL.createObjectURL(blob));

    workletNode = new AudioWorkletNode(audioCtx, "forwarder");
    workletNode.port.onmessage = (e) => onAudioFrame(e.data);

    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 1024;
    analyserNode.smoothingTimeConstant = 0.6;

    source.connect(workletNode);
    source.connect(analyserNode);

    setMicStatus("Channel hot");
    micStartBtn.disabled = true;
    micStopBtn.disabled = false;
    commitBtn.disabled = modeEl.value === "server_vad";
    clearBufferBtn.disabled = false;
    drawScope();
  }

  function stopMic() {
    if (workletNode) { workletNode.disconnect(); workletNode = null; }
    if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (drawRaf) {
      cancelAnimationFrame(drawRaf);
      drawRaf = 0;
    }
    clearScope();
    setMicStatus("Channel cold");
    micStopBtn.disabled = true;
    commitBtn.disabled = true;
    clearBufferBtn.disabled = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      micStartBtn.disabled = false;
    }
  }

  function onAudioFrame(float32) {
    const pcm16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const b64 = bytesToBase64(new Uint8Array(pcm16.buffer));
    wsSend({ type: "input_audio_buffer.append", audio: b64 });
  }

  function bytesToBase64(bytes) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode.apply(
        null,
        bytes.subarray(i, i + chunkSize),
      );
    }
    return btoa(binary);
  }

  // ─────────────────────  Oscilloscope  ─────────────────────

  function clearScope() {
    const w = oscilloCanvas.width;
    const h = oscilloCanvas.height;
    oscilloCtx.clearRect(0, 0, w, h);
  }

  function drawScope() {
    if (!analyserNode) return;
    const w = oscilloCanvas.width;
    const h = oscilloCanvas.height;
    const buf = new Uint8Array(analyserNode.fftSize);
    analyserNode.getByteTimeDomainData(buf);

    oscilloCtx.clearRect(0, 0, w, h);

    // Faint baseline.
    oscilloCtx.strokeStyle = "rgba(26, 24, 20, 0.18)";
    oscilloCtx.lineWidth = 1;
    oscilloCtx.beginPath();
    oscilloCtx.moveTo(0, h / 2);
    oscilloCtx.lineTo(w, h / 2);
    oscilloCtx.stroke();

    // Waveform — vermilion ink.
    oscilloCtx.strokeStyle = "#c5392b";
    oscilloCtx.lineWidth = 1.5;
    oscilloCtx.lineJoin = "round";
    oscilloCtx.beginPath();
    const slice = w / buf.length;
    let x = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128.0;
      const y = h / 2 + v * (h / 2) * 0.95;
      if (i === 0) oscilloCtx.moveTo(x, y);
      else oscilloCtx.lineTo(x, y);
      x += slice;
    }
    oscilloCtx.stroke();

    drawRaf = requestAnimationFrame(drawScope);
  }

  // ─────────────────────  Server events  ─────────────────────

  function handleServerEvent(evt) {
    switch (evt.type) {
      case "session.created":
      case "session.updated":
        if (evt.session && evt.session.id) {
          setStatus(`session ${evt.session.id.slice(0, 12)}…`, "connected");
        }
        return;

      case "input_audio_buffer.speech_started":
        ensureUtterance(evt.item_id, "in-progress");
        setItemMeta(evt.item_id, `started ${ms(evt.audio_start_ms)}`);
        return;

      case "input_audio_buffer.speech_stopped":
        setItemMeta(evt.item_id, `stopped ${ms(evt.audio_end_ms)}`);
        return;

      case "input_audio_buffer.committed":
        ensureUtterance(evt.item_id, "in-progress");
        setItemMeta(evt.item_id, "committed · transcribing");
        return;

      case "conversation.item.input_audio_transcription.delta": {
        const node = ensureUtterance(evt.item_id, "in-progress");
        const body = node.querySelector(".utterance-body");
        body.textContent += evt.delta || "";
        return;
      }

      case "conversation.item.input_audio_transcription.completed": {
        const node = ensureUtterance(evt.item_id, "completed");
        node.dataset.state = "completed";
        const body = node.querySelector(".utterance-body");
        body.textContent = evt.transcript || body.textContent;
        setItemMeta(evt.item_id, "completed");
        return;
      }

      case "conversation.item.input_audio_transcription.failed": {
        const node = ensureUtterance(evt.item_id, "failed");
        node.dataset.state = "failed";
        setItemMeta(
          evt.item_id,
          "failed: " + ((evt.error && evt.error.message) || "unknown"),
        );
        return;
      }

      case "error":
        setStatus("error: " + (evt.error && evt.error.code), "error");
        return;

      default:
        return;
    }
  }

  function ensureUtterance(itemId, state) {
    let node = utteranceNodes.get(itemId);
    if (node) return node;

    const empty = transcriptsEl.querySelector(".empty-state");
    if (empty) empty.remove();

    utteranceCounter += 1;
    const serial = "№ " + String(utteranceCounter).padStart(3, "0");
    utteranceSerials.set(itemId, serial);

    node = document.createElement("article");
    node.className = "utterance";
    node.dataset.state = state;
    node.innerHTML =
      `<div class="utterance-meta">` +
      `<span class="serial">${serial}</span>` +
      `<span class="ts">${nowTime()}</span>` +
      `<span class="state">opening</span>` +
      `</div>` +
      `<p class="utterance-body"></p>`;
    transcriptsEl.appendChild(node);
    transcriptsEl.scrollTop = transcriptsEl.scrollHeight;
    utteranceNodes.set(itemId, node);
    return node;
  }

  function setItemMeta(itemId, stateText) {
    const node = utteranceNodes.get(itemId);
    if (!node) return;
    const stateEl = node.querySelector(".utterance-meta .state");
    if (stateEl) stateEl.textContent = stateText;
  }

  function ms(n) { return typeof n === "number" ? n + "ms" : ""; }

  function nowTime() {
    return new Date().toLocaleTimeString("en-GB", { hour12: false });
  }

  // ─────────────────────  Misc UI  ─────────────────────

  modeEl.addEventListener("change", () => {
    if (audioCtx) {
      commitBtn.disabled = modeEl.value === "server_vad";
    }
  });

  clearLogBtn.addEventListener("click", () => {
    logEl.innerHTML = "";
  });
})();

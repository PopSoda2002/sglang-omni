/**
 * sglang-omni /v1/realtime demo
 *
 * - Captures mic audio via getUserMedia + AudioContext.
 * - Downsamples to 16 kHz mono PCM16, base64-encodes, and sends
 *   `input_audio_buffer.append` frames over WebSocket.
 * - Renders incoming transcription deltas live.
 *
 * No build step, no framework. Targets modern browsers (Chrome / Safari /
 * Firefox with AudioWorklet). Per project style: zero error handling —
 * if something fails, the browser console gets the stack and the UI
 * shows the last known status.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  // -----------------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------------
  const wsUrlEl = $("ws-url");
  const modeEl = $("mode");
  const instructionsEl = $("instructions");
  const connectBtn = $("connect");
  const disconnectBtn = $("disconnect");
  const statusEl = $("status");
  const micStartBtn = $("mic-start");
  const micStopBtn = $("mic-stop");
  const commitBtn = $("commit");
  const clearBufferBtn = $("clear-buffer");
  const micStatusEl = $("mic-status");
  const vuEl = $("vu");
  const transcriptsEl = $("transcripts");
  const logEl = $("log");
  const logDeltasEl = $("log-deltas");
  const clearLogBtn = $("clear-log");

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  let ws = null;
  let audioCtx = null;
  let micStream = null;
  let workletNode = null;
  let analyserNode = null;
  let vuRaf = 0;
  // Map item_id -> DOM node showing the in-progress utterance.
  const utteranceNodes = new Map();

  const TARGET_SR = 16000;

  // -----------------------------------------------------------------------
  // WebSocket
  // -----------------------------------------------------------------------

  function setStatus(text, cls = "") {
    statusEl.textContent = text;
    statusEl.className = "status " + cls;
  }

  function setMicStatus(text, cls = "") {
    micStatusEl.textContent = text;
    micStatusEl.className = "status " + cls;
  }

  function logEntry(direction, payload) {
    const t = payload && payload.type;
    if (t === "conversation.item.input_audio_transcription.delta" && !logDeltasEl.checked) {
      return;
    }
    const arrow = direction === "in" ? "←" : "→";
    const cls = direction === "in" ? "arrow-down" : "arrow-up";
    const summary = JSON.stringify(payload).slice(0, 240);
    const line = document.createElement("div");
    line.innerHTML = `<span class="${cls}">${arrow}</span> ${escapeHtml(summary)}`;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;",
      '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function wsSend(payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
    logEntry("out", payload);
  }

  connectBtn.addEventListener("click", () => {
    const url = wsUrlEl.value.trim();
    ws = new WebSocket(url);
    setStatus("connecting…");

    ws.onopen = () => {
      setStatus("connected", "connected");
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
      setStatus("disconnected");
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
      setStatus("error", "error");
    };
  });

  disconnectBtn.addEventListener("click", () => {
    if (ws) ws.close();
  });

  // -----------------------------------------------------------------------
  // Microphone capture + PCM16 streaming
  // -----------------------------------------------------------------------

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

    // Define an inline AudioWorklet that emits Float32 frames; we
    // resample (if needed) and convert to PCM16 in the main thread.
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
    analyserNode.fftSize = 256;

    source.connect(workletNode);
    source.connect(analyserNode);
    // workletNode does not connect to destination — we don't want to
    // hear ourselves.

    setMicStatus("recording", "recording");
    micStartBtn.disabled = true;
    micStopBtn.disabled = false;
    commitBtn.disabled = modeEl.value === "server_vad";
    clearBufferBtn.disabled = false;
    pumpVU();
  }

  function stopMic() {
    if (workletNode) {
      workletNode.disconnect();
      workletNode = null;
    }
    if (analyserNode) {
      analyserNode.disconnect();
      analyserNode = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    if (vuRaf) {
      cancelAnimationFrame(vuRaf);
      vuRaf = 0;
    }
    vuEl.value = 0;
    setMicStatus("idle");
    micStopBtn.disabled = true;
    commitBtn.disabled = true;
    clearBufferBtn.disabled = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      micStartBtn.disabled = false;
    }
  }

  function onAudioFrame(float32) {
    // The AudioContext is constructed at TARGET_SR so the worklet
    // already delivers frames at 16 kHz. Convert Float32 [-1,1] →
    // PCM16 little-endian.
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

  // VU meter
  function pumpVU() {
    if (!analyserNode) return;
    const buf = new Uint8Array(analyserNode.fftSize);
    analyserNode.getByteTimeDomainData(buf);
    let peak = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = Math.abs(buf[i] - 128) / 128;
      if (v > peak) peak = v;
    }
    vuEl.value = peak;
    vuRaf = requestAnimationFrame(pumpVU);
  }

  // -----------------------------------------------------------------------
  // Server event handlers
  // -----------------------------------------------------------------------

  function handleServerEvent(evt) {
    switch (evt.type) {
      case "session.created":
      case "session.updated":
        // Surface model / config in the status line.
        if (evt.session && evt.session.id) {
          setStatus(`session ${evt.session.id}`, "connected");
        }
        return;

      case "input_audio_buffer.speech_started":
        ensureUtterance(evt.item_id, "in-progress");
        setItemMeta(evt.item_id,
          `speech_started @ ${evt.audio_start_ms}ms`);
        return;

      case "input_audio_buffer.speech_stopped":
        setItemMeta(evt.item_id,
          `speech_stopped @ ${evt.audio_end_ms}ms`);
        return;

      case "input_audio_buffer.committed":
        ensureUtterance(evt.item_id, "in-progress");
        setItemMeta(evt.item_id, "committed — transcribing…");
        return;

      case "conversation.item.input_audio_transcription.delta": {
        const node = ensureUtterance(evt.item_id, "in-progress");
        node.querySelector(".text").textContent += evt.delta || "";
        return;
      }

      case "conversation.item.input_audio_transcription.completed": {
        const node = ensureUtterance(evt.item_id, "completed");
        node.classList.remove("in-progress");
        node.classList.add("completed");
        node.querySelector(".text").textContent = evt.transcript || "";
        setItemMeta(evt.item_id, "completed");
        return;
      }

      case "conversation.item.input_audio_transcription.failed": {
        const node = ensureUtterance(evt.item_id, "completed");
        setItemMeta(evt.item_id,
          `failed: ${evt.error && evt.error.message}`);
        node.querySelector(".text").textContent +=
          "  [transcription failed]";
        return;
      }

      case "error":
        setStatus(`error: ${evt.error && evt.error.code}`, "error");
        return;

      default:
        // Other events (response.*, conversation.item.created without
        // tracked id, etc.) are visible in the event log.
        return;
    }
  }

  function ensureUtterance(itemId, state) {
    let node = utteranceNodes.get(itemId);
    if (node) return node;
    node = document.createElement("div");
    node.className = "utterance " + state;
    node.innerHTML =
      `<div class="meta">${escapeHtml(itemId || "")}</div>` +
      `<div class="text"></div>`;
    transcriptsEl.appendChild(node);
    transcriptsEl.scrollTop = transcriptsEl.scrollHeight;
    utteranceNodes.set(itemId, node);
    return node;
  }

  function setItemMeta(itemId, text) {
    const node = utteranceNodes.get(itemId);
    if (!node) return;
    const meta = node.querySelector(".meta");
    meta.textContent = `${itemId} · ${text}`;
  }

  // -----------------------------------------------------------------------
  // Misc UI
  // -----------------------------------------------------------------------

  modeEl.addEventListener("change", () => {
    if (audioCtx) {
      // re-evaluate manual-commit button state when mode toggles mid-recording
      commitBtn.disabled = modeEl.value === "server_vad";
    }
  });

  clearLogBtn.addEventListener("click", () => {
    logEl.innerHTML = "";
  });
})();

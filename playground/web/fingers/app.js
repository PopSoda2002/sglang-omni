/**
 * COUNTING HOUSE — sglang-omni vision demo
 *
 * Webcam → snapshot a 16:10 frame → send to /v1/chat/completions with a
 * "reply with a single digit" prompt → stream the response → render the
 * digit. Inspired by thinkingmachines.ai/blog/interaction-models — a
 * stacked-row timeline of input frames and model outputs.
 *
 * Vanilla JS — no framework, no build step. House style: errors land in
 * the console; the readout stays put.
 */

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* ─────────────────────  DOM  ───────────────────── */
  const camEl        = $("cam");
  const snapEl       = $("snap");
  const tapeEl       = $("tape");

  const digitEl      = $("digit");
  const stateEl      = $("state");
  const statFramesEl = $("stat-frames");
  const statP50El    = $("stat-p50");
  const statLastEl   = $("stat-last");

  const serverEl     = $("server");
  const modelEl      = $("model");
  const modeEl       = $("mode");
  const intervalEl   = $("interval");

  const camOnBtn     = $("cam-on");
  const camOffBtn    = $("cam-off");
  const countBtn     = $("count-once");
  const loopBtn      = $("loop-toggle");

  const trackFrames  = $("track-frames");
  const trackCounts  = $("track-counts");

  /* ─────────────────────  State  ───────────────────── */
  let micStream  = null;
  let videoReady = false;
  let loopTimer  = null;
  let inflight   = false;
  let frameIdx   = 0;
  const latencies = [];   // last N completion latencies, ms

  const FRAME_W = 480;    // snapshot width — small to keep upload light
  const PROMPT  = (
    "Look at the image. Count the total number of fingers extended toward " +
    "the camera (across both hands). Respond with ONLY a single digit " +
    "from 0 to 10. No words, no punctuation — just the digit."
  );

  /* ─────────────────────  Helpers  ───────────────────── */

  function setState(text) {
    stateEl.textContent = text;
  }

  function setTape(text, on) {
    tapeEl.textContent = text;
    tapeEl.classList.toggle("on", !!on);
  }

  function extractDigit(s) {
    const m = String(s || "").match(/\d{1,2}/);
    if (!m) return null;
    const n = parseInt(m[0], 10);
    if (Number.isNaN(n) || n < 0 || n > 10) return null;
    return n;
  }

  function updateLatencyStats(ms) {
    latencies.push(ms);
    while (latencies.length > 20) latencies.shift();
    statLastEl.textContent = `${Math.round(ms)} ms`;
    const sorted = [...latencies].sort((a, b) => a - b);
    const p50 = sorted[Math.floor(sorted.length / 2)];
    statP50El.textContent = `${Math.round(p50)} ms`;
  }

  /* ─────────────────────  Camera  ───────────────────── */

  async function startCam() {
    micStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 800 } },
      audio: false,
    });
    camEl.srcObject = micStream;
    await new Promise((res) => {
      if (camEl.readyState >= 2) res();
      else camEl.onloadedmetadata = () => res();
    });
    videoReady = true;

    camOnBtn.disabled  = true;
    camOffBtn.disabled = false;
    countBtn.disabled  = false;
    loopBtn.disabled   = false;
    setTape("live · camera on", true);
    setState("ready");
  }

  function stopCam() {
    stopLoop();
    if (micStream) {
      micStream.getTracks().forEach((t) => t.stop());
      micStream = null;
    }
    camEl.srcObject = null;
    videoReady = false;

    camOnBtn.disabled  = false;
    camOffBtn.disabled = true;
    countBtn.disabled  = true;
    loopBtn.disabled   = true;
    setTape("camera off", false);
    setState("idle");
    digitEl.textContent = "·";
    digitEl.classList.remove("streaming", "done");
  }

  camOnBtn.addEventListener("click", () => { startCam(); });
  camOffBtn.addEventListener("click", () => { stopCam(); });

  /* ─────────────────────  Frame capture  ───────────────────── */

  function snapshot() {
    if (!videoReady) return null;
    const vw = camEl.videoWidth  || 1280;
    const vh = camEl.videoHeight || 800;
    const w  = FRAME_W;
    const h  = Math.round(FRAME_W * vh / vw);
    snapEl.width  = w;
    snapEl.height = h;
    const ctx = snapEl.getContext("2d");
    ctx.drawImage(camEl, 0, 0, w, h);
    return snapEl.toDataURL("image/jpeg", 0.85);
  }

  /* ─────────────────────  Timeline rows  ───────────────────── */

  function clearTimelineEmpties() {
    trackFrames.querySelectorAll(".tl-empty").forEach((n) => n.remove());
    trackCounts.querySelectorAll(".tl-empty").forEach((n) => n.remove());
  }

  function pushFrameTile(dataUrl, idx) {
    clearTimelineEmpties();
    const thumb = document.createElement("div");
    thumb.className = "tl-thumb";
    thumb.style.backgroundImage = `url("${dataUrl}")`;
    thumb.dataset.idx = String(idx).padStart(3, "0");
    trackFrames.appendChild(thumb);
    trackFrames.scrollLeft = trackFrames.scrollWidth;

    const tile = document.createElement("div");
    tile.className = "tl-tile streaming";
    tile.textContent = "…";
    const meta = document.createElement("span");
    meta.className = "tl-tile-meta";
    meta.textContent = String(idx).padStart(3, "0");
    tile.appendChild(meta);
    trackCounts.appendChild(tile);
    trackCounts.scrollLeft = trackCounts.scrollWidth;
    return tile;
  }

  /* ─────────────────────  Inference  ───────────────────── */

  async function countOnce() {
    if (!videoReady || inflight) return;
    const dataUrl = snapshot();
    if (!dataUrl) return;

    inflight = true;
    frameIdx += 1;
    statFramesEl.textContent = String(frameIdx);

    const tile = pushFrameTile(dataUrl, frameIdx);
    digitEl.textContent = "…";
    digitEl.classList.add("streaming");
    digitEl.classList.remove("done");
    setState("looking");

    const t0 = performance.now();

    const body = {
      model: modelEl.value.trim() || "qwen3-omni",
      messages: [{ role: "user", content: PROMPT }],
      images: [dataUrl],
      modalities: ["text"],
      max_tokens: 8,
      temperature: 0.0,
      stream: true,
    };

    let response;
    try {
      response = await fetch(serverEl.value.trim(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      tile.classList.remove("streaming");
      tile.textContent = "✕";
      digitEl.classList.remove("streaming");
      setState("network error");
      inflight = false;
      console.error("fetch failed", err);
      return;
    }

    if (!response.ok || !response.body) {
      const text = await response.text().catch(() => "");
      tile.classList.remove("streaming");
      tile.textContent = "✕";
      digitEl.classList.remove("streaming");
      setState(`http ${response.status}`);
      inflight = false;
      console.error("bad response", response.status, text);
      return;
    }

    /* Read SSE stream, harvest first digit, keep reading until done so
       the connection isn't left half-open. */
    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let acc    = "";
    let finalDigit = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      /* Parse SSE: each event is a line starting with "data: ". */
      let nl;
      while ((nl = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const evt = JSON.parse(payload);
          const delta =
            evt.choices && evt.choices[0] && evt.choices[0].delta &&
            evt.choices[0].delta.content;
          if (typeof delta === "string" && delta.length > 0) {
            acc += delta;
            if (finalDigit === null) {
              const d = extractDigit(acc);
              if (d !== null) {
                finalDigit = d;
                digitEl.textContent = String(d);
                tile.firstChild.textContent = String(d); // text node before meta span
              }
            }
          }
        } catch {
          /* skip malformed SSE chunk */
        }
      }
    }

    const latency = performance.now() - t0;
    updateLatencyStats(latency);

    if (finalDigit === null) {
      const d = extractDigit(acc);
      if (d !== null) {
        finalDigit = d;
        digitEl.textContent = String(d);
        tile.firstChild.textContent = String(d);
      }
    }

    digitEl.classList.remove("streaming");
    tile.classList.remove("streaming");

    if (finalDigit !== null) {
      digitEl.classList.add("done");
      tile.classList.add("done");
      setState(`counted · ${Math.round(latency)} ms`);
    } else {
      tile.textContent = "?";
      const meta = document.createElement("span");
      meta.className = "tl-tile-meta";
      meta.textContent = String(frameIdx).padStart(3, "0");
      tile.appendChild(meta);
      digitEl.textContent = "?";
      setState("no digit in reply");
    }

    inflight = false;
  }

  countBtn.addEventListener("click", () => countOnce());

  /* ─────────────────────  Live loop  ───────────────────── */

  function startLoop() {
    if (loopTimer) return;
    const interval = parseInt(intervalEl.value, 10) || 1000;
    const tick = async () => {
      if (!inflight) await countOnce();
      loopTimer = setTimeout(tick, interval);
    };
    loopTimer = setTimeout(tick, 0);
    loopBtn.textContent = "Stop live loop";
    loopBtn.classList.remove("btn-ghost");
    loopBtn.classList.add("btn-secondary");
    modeEl.value = "live";
    setState("live loop");
  }

  function stopLoop() {
    if (loopTimer) {
      clearTimeout(loopTimer);
      loopTimer = null;
    }
    loopBtn.textContent = "Begin live loop";
    loopBtn.classList.remove("btn-secondary");
    loopBtn.classList.add("btn-ghost");
  }

  loopBtn.addEventListener("click", () => {
    if (loopTimer) stopLoop();
    else            startLoop();
  });

  modeEl.addEventListener("change", () => {
    if (modeEl.value === "live") {
      if (videoReady && !loopTimer) startLoop();
    } else {
      stopLoop();
    }
  });

})();

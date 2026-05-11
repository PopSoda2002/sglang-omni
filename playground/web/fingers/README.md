# Counting House — `/v1/chat/completions` vision demo

A single-page demo that **counts the fingers you hold up to the webcam**.
Frame in, digit out — streamed through `/v1/chat/completions` with the
`images` extension. Inspired by Thinking Machines' [interaction
models](https://thinkingmachines.ai/blog/interaction-models/) writeup:
stacked rows showing the input frame and the model's reply side by
side, like a tiny benchmark you can shake your hand at.

## Run

1. **Start the server** (any sglang-omni server that exposes
   `/v1/chat/completions` works — Qwen3-Omni is the one this UI defaults
   to):

   ```bash
   sgl-omni serve \
     --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
     --port 8765
   ```

   The realtime branch's `--enable-realtime` flag is optional here — the
   demo only uses HTTP, not the websocket.

2. **Serve this directory** over HTTP (browsers refuse `getUserMedia`
   to `file://`):

   ```bash
   cd playground/web/fingers
   python -m http.server 8080
   ```

3. **Open** <http://127.0.0.1:8080> in a modern browser. Click
   **Start camera**, then either:
   - **Snap** mode → click **Count now** for a one-shot prediction;
   - **Live** mode → click **Begin live loop** and the page will
     snapshot a frame every N seconds and stream answers continuously.

## What you'll see

| Panel | Meaning |
|---|---|
| **Camera frame** | Mirrored live preview at 16:10. |
| **Predicted count** | The latest digit the model returned, in display serif. Vermilion while a reply is streaming; green once it lands. |
| **Stats** | Total frames sent, p50 round-trip latency over the last 20 calls, and the most recent single-call latency. |
| **Interaction timeline** | Two stacked rows — the *frame* row holds thumbnails of every snapshot you've sent; the *sglang-omni* row holds the matching digit tiles. New entries scroll in on the right, oldest on the left. |

## How it works

- Each call is a stateless `POST /v1/chat/completions` carrying:
  - a fixed prompt (`"Look at the image. Count the total number of
    fingers extended toward the camera (across both hands). Respond with
    ONLY a single digit from 0 to 10."`)
  - the snapshot as a `data:image/jpeg;base64,…` entry in the
    sglang-omni `images` field
  - `max_tokens: 8`, `temperature: 0`, `stream: true`
- The page parses the SSE stream and updates the readout as soon as the
  first digit shows up in `choices[0].delta.content` — usually after one
  token.
- No frameworks, no build step. `index.html` + `styles.css` + `app.js`.

## Notes

- Snapshots are downsampled to 480 px wide JPEG-85 before upload, so
  one round-trip is ~30 KB on the wire even at 1 fps.
- The preview is mirrored (selfie-cam style); the timeline thumbnails
  are mirrored to match. The bytes sent to the model are **not**
  mirrored — counting fingers is left/right-symmetric so it doesn't
  matter, but worth knowing if you adapt this for OCR or gesture work.
- If the model replies with text instead of a digit (rare, with
  `max_tokens: 8` and a strict prompt), the readout shows `?` and the
  tile is marked with a question mark. The console keeps the raw reply
  for inspection.

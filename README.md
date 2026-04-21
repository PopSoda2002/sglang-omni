# Boson AI Fork test

This README records the smoke test used to verify the `boson-ai/sglang-omni` fork against
the [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) reference TTS model.
Run it whenever the docker image, `sglang`, or `sglang-omni` dependencies change to
confirm the baseline TTS pipeline still works end-to-end.

Upstream: [sgl-project/sglang-omni](https://github.com/sgl-project/sglang-omni).

## Verified environment

- Commit: `boson-ai/sglang-omni@main` (`5eca6a6`, 2026-04-21)
- Docker image: `frankleeeee/sglang-omni:dev` (38.4 GB)
- GPU: 1 × NVIDIA A100-SXM4-40GB
- CUDA driver: sufficient for PyTorch 2.9.1 / CUDA inside the image
- Host: Linux x86_64

## Known tweak for 40 GB GPUs

The stage factory
[`sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py`](./sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py)
hardcodes `mem_fraction_static=0.85` when constructing the sglang `ServerArgs`. On an
A100 40 GB, this leaves too little headroom for the DAC codec load and the run OOMs
during vocoder init. Drop it to `0.7` (inside the installed venv copy — the source tree
stays clean) before launching the server:

```bash
sed -i 's/mem_fraction_static=0.85/mem_fraction_static=0.7/' \
  /workspace/sglang-omni/.venv/lib/python3.12/site-packages/sglang_omni/models/fishaudio_s2_pro/pipeline/stages.py
```

On an 80 GB GPU this patch is unnecessary.

## Commands

### 1. Pull image and start the container

```bash
docker pull frankleeeee/sglang-omni:dev

docker run -d --name sglang-omni-smoke \
  --gpus '"device=<FREE_GPU_INDEX>"' \
  --shm-size 32g \
  -v $(pwd):/workspace/sglang-omni \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 18000:8000 \
  frankleeeee/sglang-omni:dev \
  sleep infinity
```

Port `18000` is mapped on the host because `8000` is often in use. Mount your
huggingface cache to avoid re-downloading S2-Pro on every run.

### 2. Install and download the model inside the container

```bash
docker exec sglang-omni-smoke bash -lc '
  cd /workspace/sglang-omni &&
  uv venv .venv -p 3.12 --seed &&
  source .venv/bin/activate &&
  uv pip install . &&
  hf download fishaudio/s2-pro
'
```

### 3. Apply the 40 GB GPU tweak (skip on 80 GB)

See "Known tweak" above.

### 4. Launch the server

```bash
docker exec -d sglang-omni-smoke bash -lc '
  cd /workspace/sglang-omni && source .venv/bin/activate &&
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  sgl-omni serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 8000 > /tmp/sgl-server.log 2>&1
'
```

Wait for `/health` from the host:

```bash
until curl -sf -o /dev/null http://localhost:18000/health; do sleep 5; done
```

Model + codec load takes roughly 7–9 minutes on first launch (subsequent launches with a
warm HF cache are faster).

### 5. Smoke request

```bash
curl -sS -X POST http://localhost:18000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello from the Boson AI sglang-omni fork."}' \
  --output /tmp/s2pro-smoke.wav \
  -w "HTTP %{http_code}, %{size_download} bytes, %{time_total}s\n"
```

### 6. Verify the WAV

```bash
file /tmp/s2pro-smoke.wav
python3 -c "import wave; w=wave.open('/tmp/s2pro-smoke.wav'); print(f'channels={w.getnchannels()}, rate={w.getframerate()}Hz, frames={w.getnframes()}, duration={w.getnframes()/w.getframerate():.2f}s')"
```

## Expected output

```
HTTP 200, 348204 bytes, 3.93s
/tmp/s2pro-smoke.wav: RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 44100 Hz
channels=1, rate=44100Hz, frames=174080, duration=3.95s
```

File size, exact byte count, and generation latency will vary; the checks that matter
are: `HTTP 200`, a non-trivial WAV (≥ ~200 KB for a short phrase), mono 16-bit PCM at
44100 Hz, and a duration in the several-seconds range.

## Teardown

```bash
docker rm -f sglang-omni-smoke
```

## Related

- Full S2-Pro usage guide: [docs/basic_usage/tts_s2pro.md](./docs/basic_usage/tts_s2pro.md)
- Developer reference: [docs/developer_reference/architecture.md](./docs/developer_reference/architecture.md)
- Examples: [examples/README.md](./examples/README.md)

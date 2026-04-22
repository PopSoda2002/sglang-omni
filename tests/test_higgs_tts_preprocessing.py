# SPDX-License-Identifier: Apache-2.0
"""Tests for the Higgs TTS preprocessing stage (PR3a)."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.higgs_tts.io import HiggsTtsState
from sglang_omni.models.higgs_tts.pipeline.stages import build_preprocess_fn
from sglang_omni.models.higgs_tts.pipeline.state_io import load_state
from sglang_omni.models.higgs_tts.tokenizer import (
    AUDIO_PLACEHOLDER_ID,
    HiggsTokenizerAdapter,
)
from sglang_omni.proto import StagePayload
from sglang_omni.proto.request import OmniRequest

_SPECIAL_IDS = {
    "<|ref_audio|>": 151679,
    "<|text|>": 151672,
    "<|audio|>": 151670,
}


class _StubTokenizer:
    def __init__(self, added_vocab: dict[str, int]):
        self._added = dict(added_vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(1, ord(c) - 32) for c in text]


def _make_fn(num_codebooks: int = 8):
    adapter = HiggsTokenizerAdapter(_StubTokenizer(_SPECIAL_IDS))
    return build_preprocess_fn(adapter, num_codebooks=num_codebooks, codebook_size=1026)


def _payload(inputs, params: dict | None = None) -> StagePayload:
    return StagePayload(
        request_id="r0",
        request=OmniRequest(inputs=inputs, params=params or {}),
        data=None,
    )


def test_zero_shot_string_input():
    out = _make_fn()(_payload("hi"))
    state = load_state(out)
    assert state.reference_codes_delayed is None
    assert AUDIO_PLACEHOLDER_ID not in state.prompt_token_ids
    assert state.prompt_token_ids[0] == _SPECIAL_IDS["<|text|>"]
    assert state.prompt_token_ids[-1] == _SPECIAL_IDS["<|audio|>"]


def test_zero_shot_dict_input_uses_input_key():
    state = load_state(_make_fn()(_payload({"input": "abc"})))
    assert state.prompt_token_ids[0] == _SPECIAL_IDS["<|text|>"]


def test_voice_cloning_applies_delay_and_placeholders():
    N, T = 4, 6
    codes = torch.randint(0, 1024, (T, N), dtype=torch.long).tolist()
    out = _make_fn(num_codebooks=N)(_payload({"input": "hi", "reference_codes": codes}))
    state = load_state(out)

    expected_len = T + N - 1
    assert state.reference_codes_delayed is not None
    assert len(state.reference_codes_delayed) == expected_len
    assert all(len(row) == N for row in state.reference_codes_delayed)

    ids = state.prompt_token_ids
    assert ids[0] == _SPECIAL_IDS["<|ref_audio|>"]
    assert ids[1 : 1 + expected_len] == [AUDIO_PLACEHOLDER_ID] * expected_len
    assert ids[1 + expected_len] == _SPECIAL_IDS["<|text|>"]
    assert ids[-1] == _SPECIAL_IDS["<|audio|>"]


def test_voice_cloning_rejects_wrong_codebook_dim():
    fn = _make_fn(num_codebooks=8)
    bad = torch.zeros(3, 5, dtype=torch.long).tolist()
    with pytest.raises(ValueError, match=r"\[T, 8\]"):
        fn(_payload({"input": "x", "reference_codes": bad}))


def test_empty_reference_codes_fall_back_to_zero_shot():
    state = load_state(_make_fn()(_payload({"input": "x", "reference_codes": []})))
    assert state.reference_codes_delayed is None


def test_generation_params_propagate():
    out = _make_fn()(
        _payload(
            {"input": "x"},
            params={
                "max_new_tokens": 512,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 42,
            },
        )
    )
    state = load_state(out)
    assert state.max_new_tokens == 512
    assert state.temperature == 0.8
    assert state.top_p == 0.9
    assert state.top_k == 40
    assert state.seed == 42


def test_state_roundtrips_through_payload_data():
    codes = torch.randint(0, 1024, (3, 4), dtype=torch.long).tolist()
    out = _make_fn(num_codebooks=4)(
        _payload({"input": "hello", "reference_codes": codes})
    )
    assert isinstance(out.data, dict)
    restored = HiggsTtsState.from_dict(out.data)
    assert restored.prompt_token_ids == load_state(out).prompt_token_ids


# ---------------------------------------------------------------------------
# PR3b: server-side reference_audio → codes via an injected codec.
# ---------------------------------------------------------------------------


class _StubCodec:
    """Deterministic codec stand-in: maps waveform length to a fixed
    number of frames and returns zeros. Exercises the plumbing without
    needing the real audio tokenizer checkpoint."""

    SAMPLE_RATE = 24_000

    def __init__(self, frames_per_sec: int = 25, num_codebooks: int = 8):
        self.frames_per_sec = frames_per_sec
        self.num_codebooks = num_codebooks
        self.calls: list[tuple[int, int]] = []

    def encode_reference(self, waveform, *, sample_rate):
        # Match the real codec's pad-to-1s behaviour so the frame count
        # below never rounds down to 0.
        length = waveform.shape[-1]
        duration = max(length / sample_rate, 1.0)
        num_frames = max(int(round(duration * self.frames_per_sec)), 1)
        self.calls.append((length, sample_rate))
        return torch.zeros(num_frames, self.num_codebooks, dtype=torch.long)


def _make_fn_with_codec(codec, num_codebooks: int = 8):
    adapter = HiggsTokenizerAdapter(_StubTokenizer(_SPECIAL_IDS))
    return build_preprocess_fn(
        adapter,
        num_codebooks=num_codebooks,
        codebook_size=1026,
        audio_codec=codec,
    )


def _write_wav(tmp_path, *, duration: float = 1.0, sr: int = 24_000):
    """Write a tiny sine wave to ``tmp_path/ref.wav`` and return the path."""
    import math as _math

    import soundfile as sf

    n = int(duration * sr)
    samples = [0.3 * _math.sin(2 * _math.pi * 440 * (i / sr)) for i in range(n)]
    path = tmp_path / "ref.wav"
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return path


def test_reference_audio_path_triggers_codec(tmp_path):
    codec = _StubCodec()
    wav = _write_wav(tmp_path, duration=0.5, sr=24_000)

    fn = _make_fn_with_codec(codec)
    state = load_state(fn(_payload({"input": "hello", "reference_audio": str(wav)})))

    assert len(codec.calls) == 1
    assert codec.calls[0][1] == codec.SAMPLE_RATE
    # Encoded → delay pattern applied → placeholders in the prompt.
    assert state.reference_codes_delayed is not None
    assert AUDIO_PLACEHOLDER_ID in state.prompt_token_ids


def test_reference_audio_dict_with_audio_path(tmp_path):
    codec = _StubCodec()
    wav = _write_wav(tmp_path, duration=0.8, sr=24_000)

    fn = _make_fn_with_codec(codec)
    state = load_state(
        fn(_payload({"input": "x", "reference_audio": {"audio_path": str(wav)}}))
    )
    assert len(codec.calls) == 1
    assert state.reference_codes_delayed is not None


def test_reference_audio_dict_with_bytes(tmp_path):
    codec = _StubCodec()
    wav = _write_wav(tmp_path, duration=0.6, sr=24_000)
    raw = wav.read_bytes()

    fn = _make_fn_with_codec(codec)
    state = load_state(fn(_payload({"input": "x", "reference_audio": {"bytes": raw}})))
    assert len(codec.calls) == 1
    assert state.reference_codes_delayed is not None


def test_reference_codes_beats_reference_audio(tmp_path):
    """When both are provided, pre-encoded codes win (cheaper, deterministic)."""
    codec = _StubCodec()
    wav = _write_wav(tmp_path, duration=1.0, sr=24_000)
    precoded = torch.randint(0, 1024, (5, 8), dtype=torch.long).tolist()

    fn = _make_fn_with_codec(codec)
    state = load_state(
        fn(
            _payload(
                {
                    "input": "x",
                    "reference_codes": precoded,
                    "reference_audio": str(wav),
                }
            )
        )
    )
    assert codec.calls == []  # codec never invoked
    assert len(state.reference_codes_delayed or []) == 5 + 8 - 1


def test_reference_audio_without_codec_raises(tmp_path):
    wav = _write_wav(tmp_path, duration=0.5, sr=24_000)

    fn = build_preprocess_fn(
        HiggsTokenizerAdapter(_StubTokenizer(_SPECIAL_IDS)),
        num_codebooks=8,
        codebook_size=1026,
        audio_codec=None,
    )
    with pytest.raises(RuntimeError, match="audio_codec"):
        fn(_payload({"input": "x", "reference_audio": str(wav)}))


def test_reference_audio_rejects_unknown_shape(tmp_path):
    codec = _StubCodec()
    fn = _make_fn_with_codec(codec)
    # Neither path nor a known dict schema.
    with pytest.raises(TypeError, match="audio_path"):
        fn(_payload({"input": "x", "reference_audio": 12345}))


def test_reference_audio_rejects_empty_path_dict(tmp_path):
    codec = _StubCodec()
    fn = _make_fn_with_codec(codec)
    # The key is present but its value is empty — guard against the
    # ``get(...) or ...[other]`` KeyError pattern.
    with pytest.raises(ValueError, match="empty audio_path/path"):
        fn(_payload({"input": "x", "reference_audio": {"audio_path": ""}}))


def test_reference_audio_rejects_empty_base64_dict(tmp_path):
    codec = _StubCodec()
    fn = _make_fn_with_codec(codec)
    with pytest.raises(ValueError, match="empty base64/data"):
        fn(_payload({"input": "x", "reference_audio": {"base64": ""}}))

# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared codec cache in ``pipeline/stages.py``.

The preprocessing stage (encodes reference audio) and the vocoder stage
(decodes output codes) both need a :class:`HiggsAudioCodec`. Loading the
checkpoint twice would double load time and VRAM — the module-level
``_CODEC_CACHE`` ensures they share when configured identically.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from sglang_omni.models.higgs_tts.pipeline import stages


def _reset_cache():
    stages._CODEC_CACHE.clear()


def _patch_both_loaders(*, standalone_side_effect=None, tts_side_effect=None):
    """Patch both codec load paths — ``_is_tts_ckpt`` auto-detects which
    one to hit, so tests need to mock both regardless."""
    sa = patch(
        "sglang_omni.models.higgs_tts.audio_codec.HiggsAudioCodec.from_pretrained",
        side_effect=standalone_side_effect,
    )
    tc = patch(
        "sglang_omni.models.higgs_tts.audio_codec.HiggsAudioCodec.from_tts_ckpt",
        side_effect=tts_side_effect,
    )
    # Always default to "not a TTS ckpt" so the standalone path is taken.
    det = patch(
        "sglang_omni.models.higgs_tts.pipeline.stages._is_tts_ckpt",
        return_value=False,
    )
    return sa, tc, det


def test_identical_calls_share_instance():
    _reset_cache()
    fake_codec = MagicMock(name="codec")
    sa, tc, det = _patch_both_loaders(standalone_side_effect=[fake_codec])
    with sa as mocked_sa, tc, det:
        a = stages._get_or_load_codec("/some/path", "cpu", "float32")
        b = stages._get_or_load_codec("/some/path", "cpu", "float32")
    assert a is b is fake_codec
    assert mocked_sa.call_count == 1


def test_different_device_loads_separately():
    _reset_cache()
    cpu_codec, cuda_codec = MagicMock(), MagicMock()
    sa, tc, det = _patch_both_loaders(standalone_side_effect=[cpu_codec, cuda_codec])
    with sa as mocked_sa, tc, det:
        cpu = stages._get_or_load_codec("/p", "cpu", "float32")
        cuda = stages._get_or_load_codec("/p", "cuda:0", "float32")
    assert cpu is cpu_codec and cuda is cuda_codec
    assert mocked_sa.call_count == 2


def test_different_path_loads_separately():
    _reset_cache()
    a_codec, b_codec = MagicMock(), MagicMock()
    sa, tc, det = _patch_both_loaders(standalone_side_effect=[a_codec, b_codec])
    with sa, tc, det:
        a = stages._get_or_load_codec("/path/a", "cpu", "float32")
        b = stages._get_or_load_codec("/path/b", "cpu", "float32")
    assert a is a_codec and b is b_codec


def test_different_dtype_loads_separately():
    _reset_cache()
    f32, bf16 = MagicMock(), MagicMock()
    sa, tc, det = _patch_both_loaders(standalone_side_effect=[f32, bf16])
    with sa, tc, det:
        a = stages._get_or_load_codec("/p", "cpu", "float32")
        b = stages._get_or_load_codec("/p", "cpu", "bfloat16")
    assert a is f32 and b is bf16


def test_tts_ckpt_routes_to_from_tts_ckpt():
    """When ``_is_tts_ckpt`` returns True, the factory routes to the
    TTS-ckpt path, not the standalone ``from_pretrained``."""
    _reset_cache()
    tts_codec = MagicMock(name="tts_codec")
    sa, tc, _ = _patch_both_loaders(tts_side_effect=[tts_codec])
    det = patch(
        "sglang_omni.models.higgs_tts.pipeline.stages._is_tts_ckpt",
        return_value=True,
    )
    with sa as mocked_sa, tc as mocked_tc, det:
        out = stages._get_or_load_codec("/tts/ckpt", "cpu", "float32")
    assert out is tts_codec
    assert mocked_tc.call_count == 1
    assert mocked_sa.call_count == 0


def test_tts_vs_standalone_cached_separately():
    """Same path keys — but one resolves as TTS ckpt, the other as
    standalone. They must not collide in the cache."""
    _reset_cache()
    tts_codec, sa_codec = MagicMock(), MagicMock()

    # First call: _is_tts_ckpt returns True for /p
    with (
        patch(
            "sglang_omni.models.higgs_tts.pipeline.stages._is_tts_ckpt",
            return_value=True,
        ),
        patch(
            "sglang_omni.models.higgs_tts.audio_codec.HiggsAudioCodec.from_tts_ckpt",
            return_value=tts_codec,
        ),
    ):
        a = stages._get_or_load_codec("/p", "cpu", "float32")

    # Second call: same path but _is_tts_ckpt returns False (e.g. test
    # toggled it, or two different paths happen to share the string —
    # we just want the cache key to distinguish).
    with (
        patch(
            "sglang_omni.models.higgs_tts.pipeline.stages._is_tts_ckpt",
            return_value=False,
        ),
        patch(
            "sglang_omni.models.higgs_tts.audio_codec.HiggsAudioCodec.from_pretrained",
            return_value=sa_codec,
        ),
    ):
        b = stages._get_or_load_codec("/p", "cpu", "float32")

    assert a is tts_codec and b is sa_codec

# SPDX-License-Identifier: Apache-2.0
"""Vendored third-party modules that must track upstream closely.

Kept isolated under ``_vendored`` so that adapting to the transformers
version pinned by sglang-omni (< 5) stays a contained set of diffs, and
so future upstream bumps are straightforward (drop-in replace + re-apply
the documented patches).
"""

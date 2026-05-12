"""Kokoro-82M model converter (full: text_encoder + bert + predictor + decoder + bert_encoder).

Plan 2 Task 5 final. Converts ALL 548 .pth tensors to cactus .weights files.

Loads kokoro-v0_19.pth, applies:
  1. Strip `net/<submodule>/` top-level prefix and `module.` DataParallel prefix
  2. Fold weight_norm: weight = weight_g * weight_v / ||weight_v||_along_dim
  3. Repack bidirectional LSTM into cactus_lstm_cell-compatible layout
  4. Per-tensor groupwise INT8 (or auto-FP16) via tensor_io.save_tensor_with_header

After folding the 86 weight_norm pairs in the .pth, ~459 .weights files are emitted.

Output: per-tensor .weights files in cactus's binary format under assets/kokoro/weights/.
"""

import json
import os
import re
import struct
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .tensor_io import (
    save_tensor_with_header,
    GROUP_SIZE,
    INTERLEAVE_BLOCK,
    CACTUS_MAGIC,
    FLAG_INTERLEAVED,
)

# ------------------------------------------------------------------------
# Helpers - testable in isolation
# ------------------------------------------------------------------------


def strip_kokoro_prefix(name: str) -> str:
    """Strip 'net/<submodule>/module.' prefix.

    Example:
      'net/text_encoder/module.cnn.0.0.weight_v' -> 'text_encoder/cnn.0.0.weight_v'
    """
    # The walk produces names like 'net/text_encoder/module.cnn.0.0.weight_v'
    # We want 'text_encoder/cnn.0.0.weight_v' (drop 'net/', drop 'module.', keep submodule prefix
    # so downstream consumers can route by submodule).
    name = re.sub(r"^net/", "", name)            # drop top-level
    name = re.sub(r"/module\.", "/", name)       # drop DataParallel wrap
    return name


def fold_weight_norm(weight_v: torch.Tensor, weight_g: torch.Tensor, norm_dim: int = 0) -> torch.Tensor:
    """Fold PyTorch weight_norm parameterization back to a single weight tensor.

    weight = g * v / ||v||_along_dims(except norm_dim)

    PyTorch convention for Conv1d/Linear weight_norm:
      weight_v shape: same as the underlying weight (e.g. (out, in, k) for Conv1d)
      weight_g shape: (out,) reshaped via the norm_dim - typically (out, 1, 1) for Conv1d
      norm_dim defaults to 0 (per-output-channel normalization)
    """
    # Compute v's norm over all dims EXCEPT norm_dim
    dims_to_norm = tuple(d for d in range(weight_v.dim()) if d != norm_dim)
    v_norm = weight_v.norm(p=2, dim=dims_to_norm, keepdim=True)
    # Avoid div-by-zero
    v_norm = torch.clamp(v_norm, min=1e-12)
    return weight_g.view(v_norm.shape) * weight_v / v_norm


def find_weight_norm_pairs(state_dict: dict) -> dict:
    """Identify (base_name, weight_v_name, weight_g_name) triples in a state dict.

    Returns dict mapping base_name -> (weight_v, weight_g) tensors.
    Other tensors are left untouched by the caller.
    """
    pairs = {}
    seen = set(state_dict.keys())
    for k in list(state_dict.keys()):
        if k.endswith(".weight_v"):
            base = k[:-len(".weight_v")]
            g_key = base + ".weight_g"
            if g_key in seen:
                pairs[base] = (state_dict[k], state_dict[g_key])
    return pairs


def apply_weight_norm_folding(state_dict: dict) -> dict:
    """Return a NEW state dict with all weight_v/weight_g pairs folded into single weights."""
    pairs = find_weight_norm_pairs(state_dict)
    folded_keys = set()
    for base, (v, g) in pairs.items():
        folded_keys.add(base + ".weight_v")
        folded_keys.add(base + ".weight_g")
    out = {}
    for k, t in state_dict.items():
        if k in folded_keys:
            continue
        out[k] = t
    for base, (v, g) in pairs.items():
        out[base + ".weight"] = fold_weight_norm(v, g, norm_dim=0)
    return out


# ------------------------------------------------------------------------
# Sanity tests for the helpers - run BEFORE attempting any real conversion
# ------------------------------------------------------------------------

def _test_strip_prefix():
    cases = [
        ("net/text_encoder/module.cnn.0.0.weight_v", "text_encoder/cnn.0.0.weight_v"),
        ("net/bert/module.embeddings.word_embeddings.weight", "bert/embeddings.word_embeddings.weight"),
        ("net/decoder/module.decode.0.conv1.weight_v", "decoder/decode.0.conv1.weight_v"),
    ]
    for inp, want in cases:
        got = strip_kokoro_prefix(inp)
        assert got == want, f"strip_kokoro_prefix({inp!r}) -> {got!r}, want {want!r}"
    print("  strip_kokoro_prefix: ok")


def _test_fold_weight_norm():
    # Build a known weight, decompose into weight_v + weight_g manually, then refold.
    torch.manual_seed(0)
    w_orig = torch.randn(8, 4, 3)  # (out=8, in=4, k=3) - Conv1d weight
    # Decompose: g = ||w||_along_(in,k); v = w (any vector parallel to w works as v)
    g = w_orig.norm(p=2, dim=(1, 2), keepdim=True)  # shape (8, 1, 1)
    v = w_orig  # use w_orig itself as v; folded result should be w_orig back
    refolded = fold_weight_norm(v, g, norm_dim=0)
    err = (refolded - w_orig).abs().max().item()
    assert err < 1e-5, f"refold error {err} too large"
    print(f"  fold_weight_norm: ok (max abs err {err:.2e})")


def _test_backward_compat_5c():
    """Verify the full naming pipeline (literal-precedence + function fallback)
    reproduces all 46 filenames shipped in 5c.

    Note: `_kokoro_canonical_to_filename` is deterministic but does not match
    the 5c literals for entries that were hand-shortened (e.g. `cnn.0.0` ->
    `cnn_0_conv`, `bert/embeddings.word_embeddings.weight` ->
    `bert_word_embeddings`, lstm prefix collapsed). LITERAL_NAME_MAP_5C is the
    source of truth for those - this test makes sure `_ensure_name_map()`
    correctly composes the two so 5c output stays byte-identical."""
    # Function-side determinism check on a NEW tensor name (not in literal-46)
    sample = "predictor/text_encoder.lstms.0.weight_ih_l0"
    assert _kokoro_canonical_to_filename(sample) == \
        "predictor_text_encoder_lstms_0_lstm_fwd_0_weight_ih", \
        f"function determinism failed for {sample}"
    print(f"  function-side determinism: ok")
    if not Path("assets/kokoro/kokoro-v0_19.pth").exists():
        print("  (literal-precedence check skipped: checkpoint not present)")
        return
    name_map = _ensure_name_map()
    n_compat = 0
    for canonical, expected in LITERAL_NAME_MAP_5C.items():
        if canonical not in name_map:
            raise AssertionError(f"5c entry {canonical!r} missing from full name map")
        got = name_map[canonical]
        if got != expected:
            raise AssertionError(
                f"5c regression {canonical!r}: name_map gave {got!r}, want {expected!r}"
            )
        n_compat += 1
    print(f"  backward compat with literal-46 (via _ensure_name_map): ok ({n_compat}/{len(LITERAL_NAME_MAP_5C)})")


def _test_apply_folding_dict():
    # Build a tiny state dict with one weight-norm pair + one normal tensor
    sd = {
        "encoder.conv.weight_v": torch.randn(4, 2, 3),
        "encoder.conv.weight_g": torch.randn(4, 1, 1),
        "encoder.conv.bias": torch.zeros(4),
        "encoder.norm.weight": torch.ones(4),
    }
    folded = apply_weight_norm_folding(sd)
    assert "encoder.conv.weight" in folded
    assert "encoder.conv.weight_v" not in folded
    assert "encoder.conv.weight_g" not in folded
    assert "encoder.conv.bias" in folded
    assert "encoder.norm.weight" in folded
    print("  apply_weight_norm_folding: ok")


# ------------------------------------------------------------------------
# Subset loader + filename mapping
# ------------------------------------------------------------------------


# --- Backward-compat literal map (frozen) ---
#
# These 46 entries shipped in 5c with hand-picked filename shortenings (e.g.
# `cnn.0.0` -> `cnn_0_conv` instead of `cnn_0_0`, BERT prefix collapses).
# They MUST stay byte-identical so the existing .weights files keep round-tripping.
# All NEW tensors get filenames from `_kokoro_canonical_to_filename` instead.
LITERAL_NAME_MAP_5C = {
    # ---- text_encoder ----
    "text_encoder/embedding.weight":              "text_encoder_embedding",
    # cnn block 0
    "text_encoder/cnn.0.0.weight":  "text_encoder_cnn_0_conv_weight",   # 3D -> FP16 by shape fallback
    "text_encoder/cnn.0.0.bias":    "text_encoder_cnn_0_conv_bias",      # FP16 via 'bias'
    "text_encoder/cnn.0.1.gamma":   "text_encoder_cnn_0_norm_gamma",     # FP16 via 'norm'
    "text_encoder/cnn.0.1.beta":    "text_encoder_cnn_0_norm_beta",      # FP16 via 'norm'
    # cnn block 1
    "text_encoder/cnn.1.0.weight":  "text_encoder_cnn_1_conv_weight",
    "text_encoder/cnn.1.0.bias":    "text_encoder_cnn_1_conv_bias",
    "text_encoder/cnn.1.1.gamma":   "text_encoder_cnn_1_norm_gamma",
    "text_encoder/cnn.1.1.beta":    "text_encoder_cnn_1_norm_beta",
    # cnn block 2
    "text_encoder/cnn.2.0.weight":  "text_encoder_cnn_2_conv_weight",
    "text_encoder/cnn.2.0.bias":    "text_encoder_cnn_2_conv_bias",
    "text_encoder/cnn.2.1.gamma":   "text_encoder_cnn_2_norm_gamma",
    "text_encoder/cnn.2.1.beta":    "text_encoder_cnn_2_norm_beta",
    # bidirectional LSTM (1 layer, fwd + bwd)
    "text_encoder/lstm.weight_ih_l0":          "text_encoder_lstm_fwd_0_weight_ih",
    "text_encoder/lstm.weight_hh_l0":          "text_encoder_lstm_fwd_0_weight_hh",
    "text_encoder/lstm.bias_ih_l0":            "text_encoder_lstm_fwd_0_bias_ih",
    "text_encoder/lstm.bias_hh_l0":            "text_encoder_lstm_fwd_0_bias_hh",
    "text_encoder/lstm.weight_ih_l0_reverse":  "text_encoder_lstm_bwd_0_weight_ih",
    "text_encoder/lstm.weight_hh_l0_reverse":  "text_encoder_lstm_bwd_0_weight_hh",
    "text_encoder/lstm.bias_ih_l0_reverse":    "text_encoder_lstm_bwd_0_bias_ih",
    "text_encoder/lstm.bias_hh_l0_reverse":    "text_encoder_lstm_bwd_0_bias_hh",

    # ---- bert (ALBERT-tiny, 1 layer group, 1 layer) ----
    "bert/embeddings.word_embeddings.weight":        "bert_word_embeddings",
    "bert/embeddings.position_embeddings.weight":    "bert_position_embeddings",  # auto FP16
    "bert/embeddings.token_type_embeddings.weight":  "bert_token_type_embeddings",
    "bert/embeddings.LayerNorm.weight":              "bert_embeddings_norm_weight",  # auto FP16 via 'norm'
    "bert/embeddings.LayerNorm.bias":                "bert_embeddings_norm_bias",
    "bert/encoder.embedding_hidden_mapping_in.weight": "bert_emb_hidden_mapping_weight",
    "bert/encoder.embedding_hidden_mapping_in.bias":   "bert_emb_hidden_mapping_bias",
    # ALBERT layer 0
    "bert/encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.weight": "bert_layer_0_full_norm_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.full_layer_layer_norm.bias":   "bert_layer_0_full_norm_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight":  "bert_layer_0_attn_q_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.query.bias":    "bert_layer_0_attn_q_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight":    "bert_layer_0_attn_k_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.key.bias":      "bert_layer_0_attn_k_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight":  "bert_layer_0_attn_v_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.value.bias":    "bert_layer_0_attn_v_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight":  "bert_layer_0_attn_dense_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.dense.bias":    "bert_layer_0_attn_dense_bias",
    # NB: actual checkpoint key is `attention.LayerNorm.{weight,bias}` (HuggingFace
    # ALBERT convention), NOT `attention_layer_norm.*` as the spec sketch suggested.
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.weight": "bert_layer_0_attn_norm_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.attention.LayerNorm.bias":   "bert_layer_0_attn_norm_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.ffn.weight":         "bert_layer_0_ffn_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.ffn.bias":           "bert_layer_0_ffn_bias",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight":  "bert_layer_0_ffn_output_weight",
    "bert/encoder.albert_layer_groups.0.albert_layers.0.ffn_output.bias":    "bert_layer_0_ffn_output_bias",
    "bert/pooler.weight":  "bert_pooler_weight",
    "bert/pooler.bias":    "bert_pooler_bias",
}


def _kokoro_canonical_to_filename(name: str) -> str:
    """Map a canonical (post-strip + post-fold) tensor name to a cactus filename basename.

    Convention used for predictor/decoder/bert_encoder tensors:
      - submodule prefix preserved at the start
      - dots become underscores
      - LSTM gates `weight_ih_l<N>[_reverse]` -> `lstm_{fwd,bwd}_<N>_weight_ih`
        (so the filename matches the text_encoder LSTM convention from 5c)
      - Trailing `LayerNorm.weight/bias` -> `norm_weight/norm_bias`
        (substring 'norm' in filename triggers FP16 auto-detect in tensor_io)
      - Trailing `_<N>_gamma` / `_<N>_beta` -> `_<N>_norm_gamma` / `_<N>_norm_beta`
        (covers AdaIN-style scale params; safety transform - not exercised by the
        post-fold predictor/decoder names but harmless if they appear)

    The filename is checked for filesystem safety + uniqueness inside
    `_ensure_name_map`. Backward compat with LITERAL_NAME_MAP_5C is enforced
    in the same function (literal entries take precedence).
    """
    if "/" in name:
        submodule, rest = name.split("/", 1)
    else:
        submodule, rest = "", name

    # LSTM gates: weight_ih_l<N>[_reverse], weight_hh_l<N>[_reverse], bias_ih/hh
    m = re.match(r"^(.*?)\.?(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)(_reverse)?$", rest)
    if m:
        path, gate, layer_idx, rev = m.groups()
        direction = "bwd" if rev else "fwd"
        prefix_part = path.rstrip(".")
        if prefix_part:
            prefix_part = prefix_part.replace(".", "_") + "_"
        out_rest = f"{prefix_part}lstm_{direction}_{layer_idx}_{gate}"
    else:
        out_rest = rest.replace(".", "_")
        out_rest = out_rest.replace("_LayerNorm_weight", "_norm_weight")
        out_rest = out_rest.replace("_LayerNorm_bias", "_norm_bias")
        out_rest = re.sub(r"_(\d+)_gamma$", r"_\1_norm_gamma", out_rest)
        out_rest = re.sub(r"_(\d+)_beta$", r"_\1_norm_beta", out_rest)

    return f"{submodule}_{out_rest}" if submodule else out_rest


# Module-level cache; populated by _ensure_name_map() on first call.
TENSOR_NAME_MAP = None


def _ensure_name_map(ckpt_path: str = "assets/kokoro/kokoro-v0_19.pth") -> dict:
    """Load the .pth, walk + strip + fold to discover all canonical names, then
    build {canonical -> filename basename} for the entire model.

    LITERAL_NAME_MAP_5C entries take precedence (frozen for backward compat with
    the .weights files shipped in 5c). All others are produced by the
    `_kokoro_canonical_to_filename` generator. Asserts uniqueness of filenames
    and filesystem-safety, then caches the result on the module so subsequent
    calls are O(1).
    """
    global TENSOR_NAME_MAP
    if TENSOR_NAME_MAP is not None:
        return TENSOR_NAME_MAP

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flat: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, prefix + "/" + str(k) if prefix else str(k))
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            flat[prefix] = node
    _walk(ckpt)

    renamed = OrderedDict((strip_kokoro_prefix(k), v) for k, v in flat.items())
    folded = apply_weight_norm_folding(renamed)

    name_map: "OrderedDict[str, str]" = OrderedDict()
    for canonical in folded:
        if canonical in LITERAL_NAME_MAP_5C:
            name_map[canonical] = LITERAL_NAME_MAP_5C[canonical]
        else:
            name_map[canonical] = _kokoro_canonical_to_filename(canonical)

    # Uniqueness + filesystem-safety check
    seen: dict[str, str] = {}
    bad_chars = []
    safe_re = re.compile(r"^[a-zA-Z0-9_\-]+$")
    for canonical, fn in name_map.items():
        if not safe_re.match(fn):
            bad_chars.append((canonical, fn))
        if fn in seen:
            raise AssertionError(
                f"Duplicate filename {fn!r}: from {seen[fn]!r} and {canonical!r}"
            )
        seen[fn] = canonical
    if bad_chars:
        raise AssertionError(
            f"Filesystem-unsafe filenames: {bad_chars[:5]}"
        )

    TENSOR_NAME_MAP = name_map
    return TENSOR_NAME_MAP


def _flatten_kokoro_checkpoint(ckpt_path: str, scope: tuple[str, ...] | None = None) -> "OrderedDict":
    """Load .pth, walk the nested dict, return flat {dotted_name: tensor} after
    stripping kokoro prefixes and folding weight_norm pairs.

    If `scope` is provided, only tensors whose canonical name starts with one
    of the given prefixes are returned (kept for compatibility with existing
    callers; default of None returns ALL tensors).
    """
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    flat: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    def _walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, prefix + "/" + str(k) if prefix else str(k))
        elif hasattr(node, "shape") and hasattr(node, "dtype"):
            flat[prefix] = node
    _walk(ckpt)

    renamed = OrderedDict((strip_kokoro_prefix(k), v) for k, v in flat.items())
    if scope is not None:
        renamed = OrderedDict((k, v) for k, v in renamed.items() if k.startswith(scope))
    return apply_weight_norm_folding(renamed)


# ------------------------------------------------------------------------
# Loader for round-trip verification
# ------------------------------------------------------------------------
#
# tensor_io.save_tensor_with_header has no public inverse. We implement the
# minimum needed to dequantize and verify: reverses INT8 group-quant +
# 4-row interleave for 2D, INT8 1D group-quant for 1D, and FP16 passthrough.
# (INT4 not used in this subset.)

def _deinterleave_weights(interleaved: np.ndarray, N_padded: int, K: int,
                          block_size: int = INTERLEAVE_BLOCK) -> np.ndarray:
    """Inverse of tensor_io.interleave_weights. Returns (N_padded, K) int8 array."""
    # Layout produced by interleave_weights:
    #   reshape(N//block, block, K//4, 4) -> transpose(0, 2, 1, 3) -> flatten
    # So inverse: reshape(N//block, K//4, block, 4) -> transpose(0, 2, 1, 3) -> reshape(N, K)
    arr = interleaved.reshape(N_padded // block_size, K // 4, block_size, 4)
    arr = arr.transpose(0, 2, 1, 3)
    return arr.reshape(N_padded, K)


def _deinterleave_scales(scales_inter: np.ndarray, N_padded: int, num_groups: int,
                         block_size: int = INTERLEAVE_BLOCK) -> np.ndarray:
    """Inverse of tensor_io.interleave_scales. Returns (N_padded, num_groups)."""
    arr = scales_inter.reshape(N_padded // block_size, num_groups, block_size)
    arr = arr.transpose(0, 2, 1)
    return arr.reshape(N_padded, num_groups)


def load_tensor_with_header(path: str) -> tuple[np.ndarray, list[int]]:
    """Read a cactus .weights file and return (dequantized_fp32, original_shape).

    Handles INT8 (1D + 2D), FP16 (any shape). INT4 not implemented (not used here).
    Original shape is the LOGICAL shape; the returned array may be padded along
    K (for INT8) which the caller must crop.
    """
    with open(path, "rb") as f:
        data = f.read()

    assert data[:4] == CACTUS_MAGIC, f"bad magic in {path}"
    flags = struct.unpack_from("<I", data, 4)[0]
    # alignment at offset 8 (uint32), ndim at offset 12 (uint32)
    ndim = struct.unpack_from("<I", data, 12)[0]
    # 4 dims at offsets 16, 24, 32, 40 (uint64 each)
    dims = list(struct.unpack_from("<QQQQ", data, 16))
    precision_code = struct.unpack_from("<I", data, 48)[0]
    data_bytes = struct.unpack_from("<Q", data, 52)[0]
    scales_bytes = struct.unpack_from("<Q", data, 60)[0]
    group_size = struct.unpack_from("<I", data, 68)[0]
    num_groups = struct.unpack_from("<I", data, 72)[0]
    original_N = struct.unpack_from("<Q", data, 76)[0]
    # header total: 84 bytes

    # alignment padding after header to CACTUS_ALIGNMENT (32) - so 96
    HEADER = 84
    ALIGN = 32
    def _align(off):
        rem = off % ALIGN
        return off if rem == 0 else off + (ALIGN - rem)

    if precision_code == 1:  # FP16
        # No scales, data starts after aligned header
        body_off = _align(HEADER)
        n_elems = data_bytes // 2
        arr = np.frombuffer(data, dtype=np.float16, count=n_elems, offset=body_off).astype(np.float32)
        shape = [d for d in dims[:ndim]]
        return arr.reshape(shape), shape

    if precision_code == 0:  # INT8
        # Layout: header + pad to 32 + scales (FP16) + pad to 32 + quantized data (INT8)
        scales_off = _align(HEADER)
        n_scales = scales_bytes // 2
        scales_fp16 = np.frombuffer(data, dtype=np.float16, count=n_scales, offset=scales_off).astype(np.float32)
        data_off = _align(scales_off + scales_bytes)
        quantized = np.frombuffer(data, dtype=np.int8, count=data_bytes, offset=data_off)

        if ndim == 2:
            # dims[0] = N_padded, dims[1] = K
            N_padded = dims[0]
            K = dims[1]
            # de-interleave
            quantized_2d = _deinterleave_weights(quantized, N_padded, K).astype(np.float32)
            scales_2d = _deinterleave_scales(scales_fp16, N_padded, num_groups)
            # dequantize: (N_padded, num_groups, group_size) * scales (N_padded, num_groups, 1)
            grouped = quantized_2d.reshape(N_padded, num_groups, group_size)
            dequant = grouped * scales_2d[:, :, np.newaxis]
            dequant = dequant.reshape(N_padded, K)
            # Crop padded N down to original_N. (K padding can't be undone here -
            # caller knows the original K from the .pth.)
            return dequant[:original_N, :], [original_N, K]

        if ndim == 1:
            # dims[0] = K (possibly padded), original_N stores logical K
            K = dims[0]
            quantized_1d = quantized.astype(np.float32).reshape(1, num_groups, group_size)
            scales_1d = scales_fp16.reshape(1, num_groups, 1)
            dequant = (quantized_1d * scales_1d).reshape(K)
            return dequant, [K]

    raise NotImplementedError(f"precision_code {precision_code} ndim {ndim} not supported")


# ------------------------------------------------------------------------
# Conversion + round-trip verification
# ------------------------------------------------------------------------


def _git_head_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _write_manifest(out_dir: Path, name_map: dict, total_bytes: int,
                    n_int8: int, n_fp16: int) -> None:
    """Write MANIFEST.json describing the converted weights directory."""
    submodule_prefixes = ["text_encoder/", "bert/", "bert_encoder/", "predictor/", "decoder/"]
    submodule_counts = {
        prefix: sum(1 for k in name_map if k.startswith(prefix))
        for prefix in submodule_prefixes
    }
    manifest = {
        "model": "kokoro-v0_19",
        "source": "hexgrad/kLegacy/v0.19/kokoro-v0_19.pth",
        "converter_commit": _git_head_sha(),
        "n_tensors": len(name_map),
        "submodule_counts": submodule_counts,
        "total_size_bytes": total_bytes,
        "precision_split": {"int8": n_int8, "fp16": n_fp16},
    }
    with open(out_dir / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Wrote {out_dir / 'MANIFEST.json'}")


def convert_kokoro_subset_weights(ckpt_path: str = "assets/kokoro/kokoro-v0_19.pth",
                                  out_dir: str = "assets/kokoro/weights",
                                  scope: tuple[str, ...] | None = None,
                                  verbose_per_tensor: bool = False):
    """Convert (all of, or a scoped subset of) Kokoro to cactus .weights files.

    Default behavior (scope=None) converts ALL submodules: text_encoder, bert,
    predictor, decoder, bert_encoder. Pass `scope=("text_encoder/",)` etc. to
    restrict.
    """
    name_map = _ensure_name_map(ckpt_path)
    folded = _flatten_kokoro_checkpoint(ckpt_path, scope=scope)

    missing_in_map = [k for k in folded if k not in name_map]
    if missing_in_map:
        print(f"ERROR: {len(missing_in_map)} tensors have no filename mapping:")
        for k in missing_in_map[:20]:
            print(f"    {k}")
        raise SystemExit(1)

    if scope is None:
        # Sanity: all mapped names must be present in folded data
        missing_in_data = [k for k in name_map if k not in folded]
        if missing_in_data:
            print(f"ERROR: {len(missing_in_data)} mapped names not present in checkpoint:")
            for k in missing_in_data[:20]:
                print(f"    {k}")
            raise SystemExit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_int8 = 0
    n_fp16 = 0
    total_bytes = 0
    for canonical_name, tensor in folded.items():
        out_basename = name_map[canonical_name]
        target = out_path / (out_basename + ".weights")
        # Pass tensor as-is (torch tensor) - save_tensor_with_header handles it.
        # We pass precision='INT8' so the auto-detect runs against the filename;
        # it will demote to FP16 for 'norm'/'bias'/'position_embeddings' substrings,
        # and 3D tensors fall through to FP16 by shape (no INT8 3D path in tensor_io).
        save_tensor_with_header(tensor, target, precision='INT8')

        size = target.stat().st_size
        total_bytes += size
        with open(target, "rb") as f:
            f.seek(48)
            prec_code = struct.unpack("<I", f.read(4))[0]
        is_fp16 = (prec_code == 1)
        n_fp16 += int(is_fp16)
        n_int8 += int(not is_fp16)
        if verbose_per_tensor:
            shape_str = "x".join(str(d) for d in tensor.shape)
            print(f"  wrote {out_basename}.weights  shape={shape_str:<14s} "
                  f"prec={'FP16' if is_fp16 else 'INT8'}  size={size:>9,d}")

    print(f"\nTotal: {len(folded)} tensors, {total_bytes:,} bytes "
          f"({total_bytes / 1024 / 1024:.2f} MB) in {out_dir}/")
    print(f"  INT8: {n_int8}   FP16: {n_fp16}")
    if scope is None:
        _write_manifest(out_path, name_map, total_bytes, n_int8, n_fp16)


def round_trip_verify(ckpt_path: str = "assets/kokoro/kokoro-v0_19.pth",
                      weights_dir: str = "assets/kokoro/weights",
                      scope: tuple[str, ...] | None = None):
    """Load each .weights file back, dequantize, compare to original tensor.

    Acceptance:
      - FP16 tensors: max abs relative error <= 1e-2
      - INT8 tensors: max abs relative error <= 5e-2 (group quant headroom)
    Fails loudly if anything diverges or is missing.
    """
    print(f"\nRound-trip verification...")
    name_map = _ensure_name_map(ckpt_path)
    folded = _flatten_kokoro_checkpoint(ckpt_path, scope=scope)

    errors = 0
    failures: list[tuple[str, str, float]] = []
    worst_rel = 0.0
    worst_name = ""
    n_int8 = 0
    n_fp16 = 0

    for canonical_name, original in folded.items():
        out_basename = name_map[canonical_name]
        path = Path(weights_dir) / (out_basename + ".weights")
        if not path.exists():
            print(f"  MISSING: {path}")
            errors += 1
            continue

        try:
            recovered_arr, recovered_shape = load_tensor_with_header(str(path))
        except Exception as e:
            print(f"  LOAD_ERR  {out_basename}: {e}")
            errors += 1
            continue

        # Identify precision from header
        with open(path, "rb") as f:
            f.seek(48)
            prec_code = struct.unpack("<I", f.read(4))[0]
        is_fp16 = (prec_code == 1)
        if is_fp16:
            n_fp16 += 1
        else:
            n_int8 += 1

        original_np = original.detach().cpu().float().numpy()

        if is_fp16:
            # Recovered already same shape; reshape just in case
            recovered_t = recovered_arr.reshape(original_np.shape)
        else:
            # INT8 path. recovered is 2D (N, K_padded) or 1D (K_padded). Crop K to original.
            if original_np.ndim == 2:
                N_orig, K_orig = original_np.shape
                recovered_t = recovered_arr[:N_orig, :K_orig]
            elif original_np.ndim == 1:
                K_orig = original_np.shape[0]
                recovered_t = recovered_arr[:K_orig]
            else:
                # 3D should always be FP16 path; unreachable
                raise AssertionError(f"INT8 with ndim {original_np.ndim} for {out_basename}")

        diff = recovered_t - original_np
        max_abs_err = float(np.abs(diff).max())
        denom = max(float(np.abs(original_np).max()), 1e-9)
        rel_err = max_abs_err / denom
        if rel_err > worst_rel:
            worst_rel = rel_err
            worst_name = out_basename

        bar = 1e-2 if is_fp16 else 5e-2
        if rel_err > bar:
            print(f"  REL_ERR  {out_basename}  ({'FP16' if is_fp16 else 'INT8'})  "
                  f"rel={rel_err:.4e} > {bar:.0e}  abs={max_abs_err:.4e}")
            errors += 1
            failures.append((out_basename, "FP16" if is_fp16 else "INT8", rel_err))

    print(f"\n  Checked {len(folded)} tensors  ({n_int8} INT8, {n_fp16} FP16)")
    print(f"  Worst rel-err: {worst_rel:.4e}  ({worst_name})")
    if errors == 0:
        print(f"  PASS - all tensors round-tripped within tolerance.")
    else:
        print(f"  FAIL - {errors} / {len(folded)} tensors exceeded tolerance.")
        print("  Failure list:")
        for name, prec, rel in sorted(failures, key=lambda x: -x[2]):
            print(f"    {name}  ({prec})  rel-err={rel:.4e}")
        raise SystemExit(1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _test_strip_prefix()
        _test_fold_weight_norm()
        _test_apply_folding_dict()
        _test_backward_compat_5c()
        print("All helper self-tests passed.")
        sys.exit(0)

    if "--convert" in sys.argv:
        convert_kokoro_subset_weights()
        sys.exit(0)

    if "--verify" in sys.argv:
        round_trip_verify()
        sys.exit(0)

    if "--inspect-subset" in sys.argv:
        ckpt_path = "assets/kokoro/kokoro-v0_19.pth"
        print(f"Loading {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # The inspection showed top-level is {"net": {...}}; flatten.
        from collections import OrderedDict
        flat = OrderedDict()

        def _walk(node, prefix=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(v, prefix + "/" + str(k) if prefix else str(k))
            elif hasattr(node, "shape") and hasattr(node, "dtype"):
                flat[prefix] = node
        _walk(ckpt)
        print(f"  total tensors: {len(flat)}")

        # Strip prefixes
        renamed = OrderedDict((strip_kokoro_prefix(k), v) for k, v in flat.items())

        # Subset: only text_encoder + bert (skip predictor, decoder, bert_encoder for now)
        SCOPE = ("text_encoder/", "bert/")
        scoped = OrderedDict((k, v) for k, v in renamed.items() if k.startswith(SCOPE))
        print(f"  subset (text_encoder + bert): {len(scoped)} tensors")
        for prefix in SCOPE:
            n = sum(1 for k in scoped if k.startswith(prefix))
            params = sum(t.numel() for k, t in scoped.items() if k.startswith(prefix))
            print(f"    {prefix}  {n} tensors  {params:,} params")

        # Identify weight-norm pairs per submodule (so we can flag if BERT has any).
        for prefix in SCOPE:
            sub_sd = {k: v for k, v in scoped.items() if k.startswith(prefix)}
            pairs = find_weight_norm_pairs(sub_sd)
            print(f"  weight-norm pairs in {prefix}: {len(pairs)}")

        # Fold weight-norm pairs (text_encoder has them in cnn.*; bert has plain Linears, no weight_norm)
        folded = apply_weight_norm_folding(scoped)
        print(f"  after weight-norm folding: {len(folded)} tensors")

        # Sanity: confirm no NaNs in folded weights
        nan_count = 0
        for k, t in folded.items():
            if torch.isnan(t).any():
                nan_count += 1
                print(f"    WARNING: NaN in {k}")
        if nan_count == 0:
            print("  NaN check: clean (0 tensors with NaN)")

        # List all final tensor names + shapes (for human review)
        print("\n  Final tensor list (post-prefix-strip + post-fold):")
        for k, t in folded.items():
            shape = "x".join(str(d) for d in t.shape) if t.shape else "scalar"
            print(f"    {k:<70s} {shape:<24s} {str(t.dtype):<10s} {t.numel():>10,d}")

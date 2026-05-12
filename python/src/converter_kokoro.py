"""Kokoro-82M model converter (subset: text_encoder + bert only for now).

Plan 2 Task 5b. Decoder and predictor deferred.

Loads kokoro-v0_19.pth, applies:
  1. Strip `net/<submodule>/` top-level prefix and `module.` DataParallel prefix
  2. Fold weight_norm: weight = weight_g * weight_v / ||weight_v||_along_dim
  3. Repack bidirectional LSTM into cactus_lstm_cell-compatible layout
  4. (Eventually) per-tensor groupwise INT8 quantization for weights >= 1024 elements

Output: .bin file in cactus's existing binary format (see converter.py).
"""

import os
import re
import sys

import numpy as np
import torch

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


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _test_strip_prefix()
        _test_fold_weight_norm()
        _test_apply_folding_dict()
        print("All helper self-tests passed.")
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

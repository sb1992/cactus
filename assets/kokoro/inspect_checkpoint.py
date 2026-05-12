"""Dump the structure of kokoro-v0_19.pth for tensor mapping decisions.

Output: assets/kokoro/checkpoint_inspection.txt (gitignored)

Run: /opt/homebrew/bin/python3.11 assets/kokoro/inspect_checkpoint.py
"""
import os
import sys
from datetime import datetime, timezone

import torch

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kokoro-v0_19.pth")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_inspection.txt")

print(f"Loading {CKPT} ...")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
print(f"  loaded; top-level type: {type(ckpt).__name__}")

# Walk the checkpoint, collecting every leaf tensor
def walk(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, prefix + "/" + str(k) if prefix else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from walk(v, prefix + f"[{i}]")
    elif hasattr(node, "shape") and hasattr(node, "dtype"):
        yield prefix, node

tensors = list(walk(ckpt))

with open(OUT, "w") as f:
    f.write(f"# Kokoro-82M PyTorch checkpoint inspection\n")
    f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
    f.write(f"# Source: {CKPT}\n")
    f.write(f"# File size: {os.path.getsize(CKPT) / 1024 / 1024:.2f} MB\n")
    f.write(f"# Tensor count: {len(tensors)}\n")
    total_params = sum(t.numel() for _, t in tensors)
    f.write(f"# Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB as fp32)\n")
    f.write(f"\n## Top-level structure\n")
    if isinstance(ckpt, dict):
        for k, v in ckpt.items():
            if hasattr(v, "shape"):
                f.write(f"  {k}  <Tensor {tuple(v.shape)} {v.dtype}>\n")
            elif isinstance(v, dict):
                f.write(f"  {k}  <dict, {len(v)} keys>\n")
            elif isinstance(v, (list, tuple)):
                f.write(f"  {k}  <{type(v).__name__}, len {len(v)}>\n")
            else:
                f.write(f"  {k}  <{type(v).__name__}>\n")
    else:
        f.write(f"  (top-level is {type(ckpt).__name__}, not a dict)\n")

    f.write(f"\n## All tensors (name, shape, dtype, params)\n")
    f.write(f"  {'name':<70s} {'shape':<24s} {'dtype':<10s} {'params':>12s}\n")
    f.write(f"  {'-'*70} {'-'*24} {'-'*10} {'-'*12}\n")
    for name, t in tensors:
        shape_str = "x".join(str(d) for d in t.shape) if t.shape else "scalar"
        f.write(f"  {name:<70s} {shape_str:<24s} {str(t.dtype):<10s} {t.numel():>12,d}\n")

    f.write(f"\n## Top 20 largest tensors\n")
    sorted_t = sorted(tensors, key=lambda x: -x[1].numel())[:20]
    for name, t in sorted_t:
        shape_str = "x".join(str(d) for d in t.shape)
        f.write(f"  {t.numel():>12,d} params  {shape_str:<24s}  {name}\n")

print(f"Wrote {OUT}")
print(f"  {len(tensors)} tensors, {total_params:,} total params")

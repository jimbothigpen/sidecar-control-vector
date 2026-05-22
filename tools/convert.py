#!/usr/bin/env python3
"""Convert an old-format control-vector GGUF (`direction.<idx>` tensors,
emitted by llama-cvector-generator) into the new sidecar `control_vector`
schema readable by `--sidecar-vectors`.

Old schema (one tensor per layer):
    direction.<idx>     f32[n_embd]      idx >= 1; layer 0 not stored

New schema (single packed tensor):
    sidecar.type        str    "control_vector"
    cv.arch             str    target arch (e.g. qwen35)
    cv.n_embd           u32
    cv.layer_start      i32    inclusive (clamped to >=1)
    cv.layer_end        i32    inclusive
    cv.vectors          f32  ggml shape [n_embd, n_layer]; layer 0 row zero

Layer count must be passed explicitly (--n-layer): the old format doesn't
record it. Extra direction tensors above n_layer-1 are an error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input",   required=True, type=Path,
                    help="path to existing *.cvec.gguf produced by llama-cvector-generator")
    ap.add_argument("--output",  required=True, type=Path,
                    help="path to write the new *.cv.gguf sidecar")
    ap.add_argument("--arch",    required=True, type=str,
                    help="target model architecture string (e.g. qwen35)")
    ap.add_argument("--n-layer", required=True, type=int,
                    help="target model n_layer (cv.vectors gets shape [n_embd, n_layer])")
    ap.add_argument("--layer-start", type=int, default=None,
                    help="layer_start (inclusive); defaults to min direction idx")
    ap.add_argument("--layer-end",   type=int, default=None,
                    help="layer_end (inclusive); defaults to max direction idx")
    args = ap.parse_args()

    try:
        from gguf import GGUFReader, GGUFWriter
    except ImportError:
        import os
        engine = os.environ.get("FRANKENTURBO2_DIR", "/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/frankenturbo2/src/jimbothigpen/frankenturbo2")
        sys.path.insert(0, f"{engine}/gguf-py")
        from gguf import GGUFReader, GGUFWriter

    reader = GGUFReader(str(args.input))
    n_embd: int | None = None
    per_layer: dict[int, np.ndarray] = {}

    for t in reader.tensors:
        name = t.name if isinstance(t.name, str) else t.name.decode("utf-8")
        if not name.startswith("direction."):
            continue
        try:
            idx = int(name.split(".", 1)[1])
        except ValueError:
            print(f"WARN: skipping unparsable tensor name {name!r}", file=sys.stderr)
            continue
        if idx <= 0:
            print(f"WARN: skipping invalid layer index {idx} in {name}", file=sys.stderr)
            continue
        data = np.asarray(t.data, dtype=np.float32).reshape(-1)
        if n_embd is None:
            n_embd = int(data.shape[0])
        elif int(data.shape[0]) != n_embd:
            print(f"ERROR: tensor {name} has n_embd={data.shape[0]}, expected {n_embd}",
                  file=sys.stderr)
            return 2
        per_layer[idx] = data

    if not per_layer or n_embd is None:
        print("ERROR: input has no direction.<idx> tensors", file=sys.stderr)
        return 2
    if max(per_layer) >= args.n_layer:
        print(f"ERROR: input has direction for layer {max(per_layer)} but --n-layer={args.n_layer}",
              file=sys.stderr)
        return 2

    layer_start = args.layer_start if args.layer_start is not None else min(per_layer)
    layer_end   = args.layer_end   if args.layer_end   is not None else max(per_layer)
    if layer_start < 1:
        layer_start = 1
    if layer_end < layer_start or layer_end >= args.n_layer:
        print(f"ERROR: invalid effective range [{layer_start}, {layer_end}] for n_layer={args.n_layer}",
              file=sys.stderr)
        return 2

    # Pack into [n_layer, n_embd] (numpy) → ggml [n_embd, n_layer].
    packed = np.zeros((args.n_layer, n_embd), dtype=np.float32)
    used = 0
    for idx in range(layer_start, layer_end + 1):
        if idx in per_layer:
            packed[idx, :] = per_layer[idx]
            used += 1
    if used == 0:
        print(f"ERROR: no direction tensors in [{layer_start}, {layer_end}]", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    w = GGUFWriter(str(args.output), "control_vector")
    w.add_string("sidecar.type",      "control_vector")
    w.add_string("cv.arch",           args.arch)
    w.add_uint32("cv.n_embd",         n_embd)
    w.add_int32 ("cv.layer_start",    int(layer_start))
    w.add_int32 ("cv.layer_end",      int(layer_end))
    # numpy (n_layer, n_embd) is contiguous → ggml [n_embd, n_layer].
    w.add_tensor("cv.vectors",        np.ascontiguousarray(packed))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"Wrote control_vector sidecar with {used} layer(s) "
          f"in [{layer_start}, {layer_end}] to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

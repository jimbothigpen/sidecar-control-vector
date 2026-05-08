# sidecar-control-vector

Out-of-tree sidecar handler plugin for [frankenturbo2](https://github.com/jimbothigpen/frankenturbo2):
per-layer additive steering vector applied at the residual-stream output
of each transformer layer in a configured range.

Equivalent semantics to the legacy `--control-vector` primitive, exposed
via the registry so users can do `--sidecar-vectors my_steering.cv.gguf`
and chain with other sidecars (abliteration, logit_bias, etc.).

## Build

Requires an installed frankenturbo2 engine.

```bash
cmake -S . -B build -DLLAMA_INSTALL_PREFIX=/opt/llama-frankenturbo2-vulkan
cmake --build build
```

Output: `build/libsidecar_control_vector.so`.

## Use

```bash
LD_LIBRARY_PATH=/opt/llama-frankenturbo2-vulkan/lib \
/opt/llama-frankenturbo2-vulkan/bin/llama-cli \
  --sidecar-load-plugin /path/to/libsidecar_control_vector.so \
  --sidecar-vectors /path/to/your.cv.gguf \
  -m model.gguf -p "..."
```

## Producer

`tools/convert.py` — converts a legacy `*.cvec.gguf` (with
`direction.<idx>` tensors emitted by the engine's
`tools/cvector-generator`) into the new `cv.*` schema this plugin reads:

```bash
python tools/convert.py \
  --input legacy.cvec.gguf \
  --arch <arch>            # e.g. gemma4, qwen3, llama
  --n-layer <N>            # target model's n_layer (legacy format doesn't store it)
  --output new.cv.gguf
```

`gguf-py` is needed; either `pip install gguf` or set
`FRANKENTURBO2_DIR=/path/to/frankenturbo2` to point at the engine clone.

## On-disk schema

```
sidecar.type            str    "control_vector"
cv.arch                 str    target arch (matches model's general.architecture)
cv.n_embd               u32    must equal model n_embd
cv.layer_start          i32    inclusive (clamped to >= 1; layer 0 never gets a bias)
cv.layer_end            i32    inclusive
cv.vectors              f32  ggml shape [n_embd, n_layer]; row 0 unused
```

## Coexistence with `--control-vector`

The engine's legacy `--control-vector` path remains untouched — both can be
used concurrently against the same model with no conflict. Eventual
deprecation of the legacy path is a frankenturbo2-side decision.

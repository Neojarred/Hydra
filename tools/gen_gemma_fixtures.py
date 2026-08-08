#!/usr/bin/env python3
"""Generates the reference vectors for Gemma 4's operators.

An **independent** transcription of `transformers/models/gemma4/modeling_gemma4.py`, in pure
Python — no dependency, and the test tensors fit in a few kilobytes. The Swift implementation
is then compared against these vectors.

Writing it twice is not redundancy. Every detail below was read in the official source and
recorded in D-022, and each one fails **silently**: the model would load and produce plausible
degraded text. The ones that cost the most to rediscover:

  - RMSNorm scales by `w`, **not** `1 + w`. Gemma 3 used `1 + w`; carrying that habit over is a
    2x error on every normalized activation.
  - Attention scaling is **1.0**, not `1/sqrt(head_dim)`. The query norm absorbs it.
  - `v_norm` exists, has **no weight tensor** (`with_scale=False`), and therefore nothing in
    the checkpoint reveals that the operation is there.
  - Embeddings are multiplied by `sqrt(hidden_size)` on lookup.
  - The router softmaxes over **all** experts, then takes top-k, then **renormalizes**. GPT-OSS
    softmaxes over the top-k only — same logits, different weights.
  - The MoE branch reads the **residual**, not the dense MLP's output: the two are parallel
    branches over the same input, summed.
  - Full-attention layers rotate only a quarter of the head dimension, expressed as **zero
    inverse frequencies** for the remainder.

Usage: python3 tools/gen_gemma_fixtures.py Tests/HydraReferenceTests/Fixtures
"""
import json
import math
import os
import sys

# --------------------------------------------------------------------- operators


def rms_norm(x, weight, eps=1e-6):
    """`x * (mean(x^2) + eps)^-0.5 * w`, in float32, weight applied as w and not 1 + w.

    The source uses `torch.pow(mean_squared, -0.5)` rather than `rsqrt` deliberately, to keep
    Torch and JAX agreeing; the value is the same.
    """
    mean_squared = sum(v * v for v in x) / len(x) + eps
    scale = mean_squared ** -0.5
    if weight is None:  # with_scale=False — v_norm and the router's norm
        return [v * scale for v in x]
    return [v * scale * w for v, w in zip(x, weight)]


def gelu_tanh(x):
    """`gelu_pytorch_tanh`, the activation of both the dense MLP and the experts."""
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + math.tanh(inner))


def inv_freq(head_dim, theta, rotating_pairs):
    """Inverse frequencies, with the unrotated tail set to **zero**.

    `rope_type: "proportional"` keeps `int(factor * head_dim // 2)` real frequencies and pads
    the rest with zeros. A zero frequency gives cos = 1 and sin = 0 — the identity — which is
    why partial rotation costs the kernel nothing.
    """
    pairs = head_dim // 2
    out = []
    for i in range(rotating_pairs):
        out.append(1.0 / (theta ** (2 * i / head_dim)))
    out += [0.0] * (pairs - rotating_pairs)
    return out


def rope(x, position, freqs):
    """RoPE over **halves**, as `emb = cat((freqs, freqs))` implies — not interleaved pairs."""
    half = len(x) // 2
    out = list(x)
    for i in range(half):
        angle = position * freqs[i]
        c, s = math.cos(angle), math.sin(angle)
        out[i] = x[i] * c - x[i + half] * s
        out[i + half] = x[i + half] * c + x[i] * s
    return out


def softmax(values):
    peak = max(values)
    exp = [math.exp(v - peak) for v in values]
    total = sum(exp)
    return [e / total for e in exp]


def router(hidden, proj, scale, per_expert_scale, top_k, eps=1e-6):
    """norm (no weight) -> * scale * hidden^-0.5 -> softmax over ALL -> top-k -> renormalize."""
    h = rms_norm(hidden, None, eps)
    root = len(hidden) ** -0.5
    h = [v * s * root for v, s in zip(h, scale)]

    logits = [sum(h[i] * row[i] for i in range(len(h))) for row in proj]
    probabilities = softmax(logits)

    order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], i))
    index = order[:top_k]
    weights = [probabilities[i] for i in index]
    total = sum(weights)
    weights = [w / total for w in weights]
    weights = [w * per_expert_scale[i] for w, i in zip(weights, index)]
    return index, weights


def attention(q, keys, values, sinkless_scaling=1.0, window=0):
    """Scaling is **1.0**. No sinks — that is a GPT-OSS mechanism, absent here."""
    scores = []
    first = 0 if window == 0 else max(0, len(keys) - window)
    for k in keys[first:]:
        scores.append(sum(a * b for a, b in zip(q, k)) * sinkless_scaling)
    weights = softmax(scores)
    out = [0.0] * len(values[0])
    for w, v in zip(weights, values[first:]):
        for i in range(len(out)):
            out[i] += w * v[i]
    return out


def softcap(logits, cap):
    return [cap * math.tanh(v / cap) for v in logits]


# -------------------------------------------------------------------- generation


def deterministic(count, seed):
    state = seed
    out = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        out.append((state >> 8) / (1 << 24) * 2 - 1)
    return out


def build():
    fixtures = {}

    # --- RMSNorm: with a weight, and without one (v_norm, router.norm) ---
    x = deterministic(64, 0x1234)
    w = [0.5 + 0.01 * i for i in range(64)]
    fixtures["rms_norm"] = {"input": x, "weight": w, "eps": 1e-6,
                            "expected": rms_norm(x, w, 1e-6)}
    fixtures["rms_norm_unscaled"] = {"input": x, "eps": 1e-6,
                                     "expected": rms_norm(x, None, 1e-6)}

    # --- The activation both MLP paths use ---
    grid = [-8.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 8.0] + deterministic(24, 0x55AA)
    fixtures["gelu_tanh"] = {"input": grid, "expected": [gelu_tanh(v) for v in grid]}

    # --- RoPE, both layer types. The full layer is the one with zero-padded frequencies ---
    for label, head_dim, theta, pairs in [
        ("rope_sliding", 32, 10_000.0, 16),
        ("rope_full_partial", 32, 1_000_000.0, 4),
    ]:
        freqs = inv_freq(head_dim, theta, pairs)
        vector = deterministic(head_dim, 0x9E37)
        fixtures[label] = {
            "headDim": head_dim, "theta": theta, "rotatingPairs": pairs,
            "inverseFrequencies": freqs,
            "position": 7, "input": vector,
            "expected": rope(vector, 7, freqs),
        }

    # --- Router: the whole chain, where GPT-OSS's differs ---
    hidden = deterministic(32, 0xC0FFEE)
    proj = [deterministic(32, 0x1000 + i) for i in range(8)]
    scale = [0.9 + 0.01 * i for i in range(32)]
    per_expert = [1.0 + 0.05 * i for i in range(8)]
    index, weights = router(hidden, proj, scale, per_expert, top_k=3)
    fixtures["router"] = {
        "hidden": hidden, "projection": proj, "scale": scale,
        "perExpertScale": per_expert, "topK": 3,
        "expectedIndices": index, "expectedWeights": weights,
    }

    # --- Attention with scaling 1.0, full and windowed ---
    head_dim, keys_count = 16, 12
    q = deterministic(head_dim, 0xABCD)
    keys = [deterministic(head_dim, 0x2000 + i) for i in range(keys_count)]
    values = [deterministic(head_dim, 0x3000 + i) for i in range(keys_count)]
    fixtures["attention_full"] = {
        "query": q, "keys": keys, "values": values, "window": 0,
        "expected": attention(q, keys, values),
    }
    fixtures["attention_windowed"] = {
        "query": q, "keys": keys, "values": values, "window": 5,
        "expected": attention(q, keys, values, window=5),
    }

    # --- Logit softcapping ---
    logits = [-90.0, -30.0, -5.0, 0.0, 5.0, 30.0, 90.0] + deterministic(16, 0x77)
    fixtures["softcap"] = {"input": logits, "cap": 30.0,
                           "expected": softcap(logits, 30.0)}

    # --- The embedding scale, which nothing in the checkpoint reveals ---
    fixtures["embedding_scale"] = {"hiddenSize": 2816, "expected": math.sqrt(2816)}

    return fixtures


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Tests/HydraReferenceTests/Fixtures"
    os.makedirs(out, exist_ok=True)
    fixtures = build()
    path = os.path.join(out, "gemma4_operators.json")
    with open(path, "w") as f:
        json.dump(fixtures, f, indent=1)
    print(f"{path}  {len(fixtures)} operators, {os.path.getsize(path):,} B")

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

    fixtures["layer"] = layer_fixture()
    return fixtures


def layer_fixture():
    """A whole decoder layer, where the topology matters more than any single operator.

    Two things here are not obvious from the config and are the reason this exists:

      - the MoE branch reads the **residual** — the state before the dense MLP — so the two
        are parallel branches over the same input, summed;
      - `post_attention_layernorm` is applied **before** the residual add, which is post-norm
        where GPT-OSS is pre-norm.

    Small dimensions, real structure: two heads of eight over one key/value head, a dense MLP
    beside four experts of which two are active.
    """
    H, I, M, E, K = 16, 12, 8, 4, 2
    heads, head_dim, kv_heads = 2, 8, 1
    eps = 1e-6

    def mat(rows, cols, seed):
        return [deterministic(cols, seed + r) for r in range(rows)]

    def matvec(w, x):
        return [sum(row[i] * x[i] for i in range(len(x))) for row in w]

    w = {
        "input_layernorm": [0.8 + 0.01 * i for i in range(H)],
        "q_proj": mat(heads * head_dim, H, 0x100),
        "k_proj": mat(kv_heads * head_dim, H, 0x200),
        "v_proj": mat(kv_heads * head_dim, H, 0x300),
        "o_proj": mat(H, heads * head_dim, 0x400),
        "q_norm": [0.9 + 0.02 * i for i in range(head_dim)],
        "k_norm": [1.1 - 0.02 * i for i in range(head_dim)],
        "post_attention_layernorm": [0.7 + 0.02 * i for i in range(H)],
        "pre_feedforward_layernorm": [1.0 + 0.01 * i for i in range(H)],
        "gate_proj": mat(I, H, 0x500),
        "up_proj": mat(I, H, 0x600),
        "down_proj": mat(H, I, 0x700),
        "post_feedforward_layernorm_1": [0.95 + 0.01 * i for i in range(H)],
        "pre_feedforward_layernorm_2": [1.05 - 0.01 * i for i in range(H)],
        "router_proj": mat(E, H, 0x800),
        "router_scale": [0.9 + 0.02 * i for i in range(H)],
        "router_per_expert_scale": [1.0 + 0.1 * e for e in range(E)],
        "post_feedforward_layernorm_2": [0.85 + 0.01 * i for i in range(H)],
        "post_feedforward_layernorm": [1.15 - 0.01 * i for i in range(H)],
        "layer_scalar": 1.25,
    }
    experts = [
        {"gate": mat(M, H, 0x900 + 64 * e),
         "up": mat(M, H, 0xA00 + 64 * e),
         "down": mat(H, M, 0xB00 + 64 * e)}
        for e in range(E)
    ]

    hidden = deterministic(H, 0xF00D)
    position = 3
    freqs = inv_freq(head_dim, 10_000.0, head_dim // 2)

    # --- Attention ---
    residual = list(hidden)
    normed = rms_norm(hidden, w["input_layernorm"], eps)

    q_flat = matvec(w["q_proj"], normed)
    k_flat = matvec(w["k_proj"], normed)
    v_flat = matvec(w["v_proj"], normed)

    q_heads = []
    for h in range(heads):
        head = q_flat[h * head_dim:(h + 1) * head_dim]
        q_heads.append(rope(rms_norm(head, w["q_norm"], eps), position, freqs))
    key = rope(rms_norm(k_flat, w["k_norm"], eps), position, freqs)
    # v_norm has no weight, and V never goes through RoPE.
    value = rms_norm(v_flat, None, eps)

    attn_flat = []
    for h in range(heads):
        attn_flat += attention(q_heads[h], [key], [value])
    attended = matvec(w["o_proj"], attn_flat)

    # Post-norm: applied to the attention output, before the residual add.
    attended = rms_norm(attended, w["post_attention_layernorm"], eps)
    hidden = [a + b for a, b in zip(residual, attended)]

    # --- Feed-forward: two parallel branches over the same residual ---
    residual = list(hidden)

    dense_in = rms_norm(hidden, w["pre_feedforward_layernorm"], eps)
    dense = matvec(w["down_proj"],
                   [gelu_tanh(g) * u for g, u in
                    zip(matvec(w["gate_proj"], dense_in), matvec(w["up_proj"], dense_in))])
    branch_1 = rms_norm(dense, w["post_feedforward_layernorm_1"], eps)

    index, weights = router(residual, w["router_proj"], w["router_scale"],
                            w["router_per_expert_scale"], K, eps)
    expert_in = rms_norm(residual, w["pre_feedforward_layernorm_2"], eps)
    mixed = [0.0] * H
    for e, weight in zip(index, weights):
        g = matvec(experts[e]["gate"], expert_in)
        u = matvec(experts[e]["up"], expert_in)
        out = matvec(experts[e]["down"], [gelu_tanh(a) * b for a, b in zip(g, u)])
        for i in range(H):
            mixed[i] += weight * out[i]
    branch_2 = rms_norm(mixed, w["post_feedforward_layernorm_2"], eps)

    combined = rms_norm([a + b for a, b in zip(branch_1, branch_2)],
                        w["post_feedforward_layernorm"], eps)
    hidden = [a + b for a, b in zip(residual, combined)]
    hidden = [v * w["layer_scalar"] for v in hidden]

    return {
        "hiddenSize": H, "intermediateSize": I, "moeIntermediateSize": M,
        "expertCount": E, "topK": K, "heads": heads, "headDim": head_dim,
        "keyValueHeads": kv_heads, "position": position, "eps": eps,
        "inverseFrequencies": freqs,
        "weights": w, "experts": experts,
        "input": deterministic(H, 0xF00D),
        "expectedIndices": index, "expectedWeights": weights,
        "expected": hidden,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Tests/HydraReferenceTests/Fixtures"
    os.makedirs(out, exist_ok=True)
    fixtures = build()
    path = os.path.join(out, "gemma4_operators.json")
    with open(path, "w") as f:
        json.dump(fixtures, f, indent=1)
    print(f"{path}  {len(fixtures)} operators, {os.path.getsize(path):,} B")

#!/usr/bin/env python3
"""Generates the reference vectors for Qwen3.5/3.6 MoE's linear attention.

An **independent** transcription of `torch_recurrent_gated_delta_rule` and its surroundings in
`transformers/models/qwen3_5/modeling_qwen3_5.py`, in pure Python, no dependency. The Swift
implementation in `QwenReferenceOps` is compared against these vectors.

Writing it twice is not redundancy. Every detail below was read in the official source and
recorded in D-027, and each one fails **silently**: the model would load and produce plausible
degraded text. The ones easiest to get wrong:

  - q and k are **l2 normalized**, `rsqrt(sum(x*x) + eps) * x`, not RMS normalized. There is no
    division by the length and no learned weight, so confusing the two is a factor of
    sqrt(head_dim) on both.
  - The scale is `1/sqrt(key_head_dim)`, applied to the query **after** that norm.
  - The decay multiplies the state **before** `kv_mem` reads it. Those two lines cannot be
    swapped.
  - `g = exp(-exp(A_log) * softplus(a + dt_bias))`, built from two learned per-head parameters,
    not a projection output used directly.
  - The output norm is gated: the learned weight scales the normalized value and `silu(z)`
    multiplies **afterwards**, outside the variance.
  - The convolution is depthwise and **causal**, kernel 4, left-padded, with SiLU after it.
  - The state and the whole recurrence are float32 in the reference whatever the checkpoint
    stores. Here everything is Python float, which is float64, so the Swift side compares in
    double for the same reason.

Usage: python3 tools/gen_qwen_fixtures.py Tests/HydraReferenceTests/Fixtures
"""
import json
import math
import os
import sys


def deterministic(count, seed):
    """A small reproducible vector, centred on zero and modest in magnitude."""
    state = seed & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(((state >> 33) % 2000 - 1000) / 1000.0)
    return out


def l2norm(x, eps=1e-6):
    scale = 1.0 / math.sqrt(sum(v * v for v in x) + eps)
    return [v * scale for v in x]


def softplus(x):
    return x if x > 20 else math.log1p(math.exp(x))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def silu(x):
    return x * sigmoid(x)


def delta_rule_step(query, key, value, g, beta, state, eps=1e-6):
    """One token, one value head. `state` is [key_dim][value_dim] and is mutated."""
    key_dim = len(key)
    value_dim = len(value)

    k = l2norm(key, eps)
    q = [v / math.sqrt(key_dim) for v in l2norm(query, eps)]

    for i in range(key_dim):
        for j in range(value_dim):
            state[i][j] *= g

    memory = [0.0] * value_dim
    for i in range(key_dim):
        for j in range(value_dim):
            memory[j] += state[i][j] * k[i]

    delta = [(value[j] - memory[j]) * beta for j in range(value_dim)]

    for i in range(key_dim):
        for j in range(value_dim):
            state[i][j] += k[i] * delta[j]

    out = [0.0] * value_dim
    for i in range(key_dim):
        for j in range(value_dim):
            out[j] += state[i][j] * q[i]
    return out


def gated_rms_norm(x, weight, gate, eps):
    inverse = 1.0 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [weight[i] * (x[i] * inverse) * silu(gate[i]) for i in range(len(x))]


def causal_depthwise_conv(sequence, weight, bias, kernel):
    """`sequence` is [tokens][channels]. Returns the same shape, SiLU applied."""
    tokens = len(sequence)
    channels = len(sequence[0])
    out = []
    for t in range(tokens):
        row = []
        for c in range(channels):
            total = bias[c] if bias else 0.0
            for tap in range(kernel):
                age = kernel - 1 - tap
                source = t - age
                value = sequence[source][c] if source >= 0 else 0.0
                total += value * weight[c][tap]
            row.append(silu(total))
        out.append(row)
    return out


def build():
    key_dim, value_dim, tokens = 8, 6, 5
    eps = 1e-6

    query_seq = [deterministic(key_dim, 0x51 + t) for t in range(tokens)]
    key_seq = [deterministic(key_dim, 0xA1 + t) for t in range(tokens)]
    value_seq = [deterministic(value_dim, 0xF1 + t) for t in range(tokens)]
    a_seq = deterministic(tokens, 0x2B)
    b_seq = deterministic(tokens, 0x3C)
    log_a = -0.5
    dt_bias = 0.25

    # The recurrence over the whole sequence, so the fixture pins the carried state and not
    # only one step. A kernel that decays after reading, or that drops the state between
    # tokens, agrees on token zero and diverges from token one.
    state = [[0.0] * value_dim for _ in range(key_dim)]
    outputs = []
    decays = []
    betas = []
    for t in range(tokens):
        g = math.exp(-math.exp(log_a) * softplus(a_seq[t] + dt_bias))
        beta = sigmoid(b_seq[t])
        decays.append(g)
        betas.append(beta)
        outputs.append(
            delta_rule_step(query_seq[t], key_seq[t], value_seq[t], g, beta, state, eps))

    norm_weight = deterministic(value_dim, 0x7E)
    gate = deterministic(value_dim, 0x8F)
    normed = gated_rms_norm(outputs[-1], norm_weight, gate, 1e-6)

    channels, kernel = 4, 4
    conv_sequence = [deterministic(channels, 0xC0 + t) for t in range(tokens)]
    conv_weight = [deterministic(kernel, 0xD0 + c) for c in range(channels)]
    conv_bias = deterministic(channels, 0xE0)
    convolved = causal_depthwise_conv(conv_sequence, conv_weight, conv_bias, kernel)

    return {
        "keyDim": key_dim, "valueDim": value_dim, "tokens": tokens, "eps": eps,
        "logA": log_a, "dtBias": dt_bias,
        "query": query_seq, "key": key_seq, "value": value_seq,
        "a": a_seq, "b": b_seq,
        "expectedDecays": decays, "expectedBetas": betas,
        "expectedOutputs": outputs,
        "expectedFinalState": state,
        "normWeight": norm_weight, "gate": gate, "expectedGatedNorm": normed,
        "convChannels": channels, "convKernel": kernel,
        "convInput": conv_sequence, "convWeight": conv_weight, "convBias": conv_bias,
        "expectedConv": convolved,
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Tests/HydraReferenceTests/Fixtures"
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "qwen_linear_attention.json")
    with open(path, "w") as f:
        json.dump(build(), f, indent=1)
    print(f"{path}  {os.path.getsize(path):,} B")

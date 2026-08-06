#!/usr/bin/env python3
"""Génère les vecteurs de référence des opérateurs de GPT-OSS.

Transcription **indépendante** de `gpt_oss/torch/model.py` (OpenAI, Apache 2.0), en
Python pur — aucune dépendance, les tenseurs de test tiennent en quelques kilo-octets.
L'implémentation Swift est ensuite comparée à ces vecteurs.

Cette double écriture n'est pas de la redondance : les détails qui suivent ne se
devinent pas, et se tromper sur l'un d'eux produit un modèle qui génère du texte
plausible mais dégradé, sans jamais lever d'erreur.

  - **SwiGLU découpe en indices pairs/impairs** (`x[..., ::2]`, `x[..., 1::2]`),
    pas en deux moitiés. Les lignes de `gate_up_proj` sont donc entrelacées.
  - Sa branche gate est écrêtée **seulement par le haut**, la branche linéaire des
    deux côtés, et la linéaire reçoit **+1**.
  - Le swish utilise `sigmoid(1.702 * x)`, pas `sigmoid(x)`.
  - **RoPE découpe en deux moitiés** (`torch.chunk`), à l'inverse du SwiGLU.
  - Les cos/sin sont multipliés par la **concentration** YaRN.
  - Les **sinks** sont une colonne de logits supplémentaire dans le softmax, retirée
    ensuite : ils n'apportent aucune valeur, ils modifient le dénominateur.

Usage : python3 tools/gen_reference_fixtures.py Tests/HydraReferenceTests/Fixtures
"""
import json
import math
import os
import random
import sys

# ---------------------------------------------------------------- opérateurs


def rms_norm(row, scale, eps=1e-5):
    mean_square = sum(v * v for v in row) / len(row)
    inv = 1.0 / math.sqrt(mean_square + eps)
    return [v * inv * s for v, s in zip(row, scale)]


def yarn_concentration_and_inv_freq(
    head_dim, base, initial_context_length, scaling_factor, ntk_alpha, ntk_beta
):
    freq = [base ** (i / head_dim) for i in range(0, head_dim, 2)]
    if scaling_factor > 1.0:
        concentration = 0.1 * math.log(scaling_factor) + 1.0
        d_half = head_dim / 2
        low = d_half * math.log(initial_context_length / (ntk_beta * 2 * math.pi)) / math.log(base)
        high = d_half * math.log(initial_context_length / (ntk_alpha * 2 * math.pi)) / math.log(base)
        assert 0 < low < high < d_half - 1, (low, high, d_half)

        inv_freq = []
        for i, f in enumerate(freq):
            interpolation = 1.0 / (scaling_factor * f)
            extrapolation = 1.0 / f
            ramp = (i - low) / (high - low)
            mask = 1.0 - min(max(ramp, 0.0), 1.0)
            inv_freq.append(interpolation * (1.0 - mask) + extrapolation * mask)
    else:
        concentration = 1.0
        inv_freq = [1.0 / f for f in freq]
    return concentration, inv_freq


def cos_sin(positions, head_dim, base, initial_context_length, scaling_factor, ntk_alpha, ntk_beta):
    concentration, inv_freq = yarn_concentration_and_inv_freq(
        head_dim, base, initial_context_length, scaling_factor, ntk_alpha, ntk_beta
    )
    cos, sin = [], []
    for position in positions:
        cos.append([math.cos(position * f) * concentration for f in inv_freq])
        sin.append([math.sin(position * f) * concentration for f in inv_freq])
    return cos, sin


def apply_rope(x, cos, sin):
    """x : [tokens][heads][head_dim]. Découpage en DEUX MOITIÉS."""
    out = []
    for t, token in enumerate(x):
        heads = []
        for head in token:
            half = len(head) // 2
            x1, x2 = head[:half], head[half:]
            o1 = [x1[i] * cos[t][i] - x2[i] * sin[t][i] for i in range(half)]
            o2 = [x2[i] * cos[t][i] + x1[i] * sin[t][i] for i in range(half)]
            heads.append(o1 + o2)
        out.append(heads)
    return out


def swiglu(row, alpha=1.702, limit=7.0):
    """Découpage en indices PAIRS/IMPAIRS, écrêtage asymétrique, +1 sur la linéaire."""
    out = []
    for i in range(0, len(row), 2):
        x_glu = min(row[i], limit)                      # écrêtage par le haut seulement
        x_linear = min(max(row[i + 1], -limit), limit)  # écrêtage des deux côtés
        out_glu = x_glu * (1.0 / (1.0 + math.exp(-alpha * x_glu)))
        out.append(out_glu * (x_linear + 1.0))          # le +1 est bien là
    return out


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """Q : [tokens][kv][mult][d]. K, V : [tokens][kv][d]. S : [kv*mult]."""
    n_tokens = len(Q)
    n_kv = len(Q[0])
    q_mult = len(Q[0][0])
    d = len(Q[0][0][0])

    out = [[[[0.0] * d for _ in range(q_mult)] for _ in range(n_kv)] for _ in range(n_tokens)]
    for h in range(n_kv):
        for m in range(q_mult):
            sink = S[h * q_mult + m]
            for q in range(n_tokens):
                logits = []
                for k in range(n_tokens):
                    if k > q:
                        continue                                   # causal
                    if sliding_window > 0 and k <= q - sliding_window:
                        continue                                   # fenêtre glissante
                    logits.append((k, sum(Q[q][h][m][i] * K[k][h][i] for i in range(d)) * sm_scale))
                # Le sink est une colonne de plus dans le softmax, retirée après.
                peak = max(max(v for _, v in logits), sink)
                denominator = sum(math.exp(v - peak) for _, v in logits) + math.exp(sink - peak)
                for k, v in logits:
                    w = math.exp(v - peak) / denominator
                    for i in range(d):
                        out[q][h][m][i] += w * V[k][h][i]

    flat = []
    for q in range(n_tokens):
        row = []
        for h in range(n_kv):
            for m in range(q_mult):
                row.extend(out[q][h][m])
        flat.append(row)
    return flat


def router(logits, top_k):
    """Softmax sur les seuls top-k logits, après sélection."""
    indices, weights = [], []
    for row in logits:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:top_k]
        values = [row[i] for i in order]
        peak = max(values)
        exponentials = [math.exp(v - peak) for v in values]
        total = sum(exponentials)
        indices.append(order)
        weights.append([e / total for e in exponentials])
    return indices, weights


# ---------------------------------------------------------------- génération


def normals(count, seed):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(count)]


def reshape(flat, *dims):
    if len(dims) == 1:
        return flat[: dims[0]]
    size = 1
    for d in dims[1:]:
        size *= d
    return [reshape(flat[i * size : (i + 1) * size], *dims[1:]) for i in range(dims[0])]


def build():
    out = {}

    # --- RMSNorm ---
    x = reshape(normals(3 * 64, 1), 3, 64)
    scale = [v * 0.5 + 1.0 for v in normals(64, 2)]
    out["rms_norm"] = {
        "input": x, "scale": scale, "eps": 1e-5,
        "expected": [rms_norm(row, scale) for row in x],
    }

    # --- YaRN, avec les constantes réelles de GPT-OSS ---
    head_dim, base = 64, 150000.0
    concentration, inv_freq = yarn_concentration_and_inv_freq(head_dim, base, 4096, 32.0, 1.0, 32.0)
    positions = [0, 1, 2, 17, 128, 4095, 4096, 100000]
    cos, sin = cos_sin(positions, head_dim, base, 4096, 32.0, 1.0, 32.0)
    out["yarn"] = {
        "head_dim": head_dim, "base": base, "initial_context_length": 4096,
        "scaling_factor": 32.0, "ntk_alpha": 1.0, "ntk_beta": 32.0,
        "concentration": concentration, "inv_freq": inv_freq,
        "positions": positions, "cos": cos, "sin": sin,
    }

    # --- RoPE ---
    q = reshape(normals(len(positions) * 2 * head_dim, 3), len(positions), 2, head_dim)
    out["rope"] = {"positions": positions, "input": q, "expected": apply_rope(q, cos, sin)}

    # --- SwiGLU, avec des valeurs qui franchissent les seuils d'écrêtage ---
    g = reshape([v * 6.0 for v in normals(4 * 32, 4)], 4, 32)
    g[0][0] = 20.0    # gate au-dessus de la limite : écrêté
    g[0][1] = -20.0   # linéaire sous la limite : écrêté
    g[1][0] = -20.0   # gate sous la limite : PAS écrêté, l'écrêtage est asymétrique
    g[1][1] = 20.0    # linéaire au-dessus de la limite : écrêté
    out["swiglu"] = {
        "input": g, "alpha": 1.702, "limit": 7.0,
        "expected": [swiglu(row) for row in g],
    }

    # --- SDPA avec sinks, en attention pleine puis à fenêtre glissante ---
    tokens, kv_heads, q_mult, d = 6, 2, 2, 8
    Q = reshape(normals(tokens * kv_heads * q_mult * d, 5), tokens, kv_heads, q_mult, d)
    K = reshape(normals(tokens * kv_heads * d, 6), tokens, kv_heads, d)
    V = reshape(normals(tokens * kv_heads * d, 7), tokens, kv_heads, d)
    S = normals(kv_heads * q_mult, 8)
    sm_scale = 1.0 / math.sqrt(d)
    common = {
        "tokens": tokens, "kv_heads": kv_heads, "q_mult": q_mult, "head_dim": d,
        "sm_scale": sm_scale, "q": Q, "k": K, "v": V, "sinks": S,
    }
    out["sdpa_full"] = dict(common, sliding_window=0,
                            expected=sdpa(Q, K, V, S, sm_scale, 0))
    out["sdpa_sliding"] = dict(common, sliding_window=3,
                               expected=sdpa(Q, K, V, S, sm_scale, 3))

    # --- Routeur ---
    logits = reshape(normals(5 * 32, 9), 5, 32)
    indices, weights = router(logits, 4)
    out["router"] = {"top_k": 4, "logits": logits, "indices": indices, "weights": weights}

    return out


if __name__ == "__main__":
    destination = sys.argv[1] if len(sys.argv) > 1 else "Tests/HydraReferenceTests/Fixtures"
    os.makedirs(destination, exist_ok=True)
    data = build()
    path = os.path.join(destination, "reference_ops.json")
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"{path}  ({os.path.getsize(path):,} o)")
    y = data["yarn"]
    print(f"  concentration YaRN : {y['concentration']:.9f}")
    print(f"  inv_freq[0]        : {y['inv_freq'][0]:.9e}")
    print(f"  inv_freq[31]       : {y['inv_freq'][-1]:.9e}")
    print(f"  sdpa pleine, [0][0]: {data['sdpa_full']['expected'][0][0]:.9f}")

// Concaténé après common.metal, qui fournit bf16_to_float.
//
// Opérateurs de GPT-OSS hors MoE : normalisation, RoPE, attention avec puits, SwiGLU,
// routeur. Chacun est validé contre l'implémentation CPU de HydraReference, elle-même
// validée contre une transcription indépendante de gpt_oss/torch/model.py.

/// RMSNorm : x / sqrt(moyenne(x²) + eps) * échelle.
/// Un threadgroup par ligne, réduction SIMD.
kernel void rms_norm(
    device const float  *x       [[buffer(0)]],
    device const ushort *scale   [[buffer(1)]],  // BF16
    device float        *out     [[buffer(2)]],
    constant uint       &size    [[buffer(3)]],
    constant float      &eps     [[buffer(4)]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) {
        const float v = x[i];
        partial += v * v;
    }

    threadgroup float shared[32];
    partial = simd_sum(partial);
    if (lane % 32u == 0) { shared[lane / 32u] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float inverse;
    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint i = 0; i < simdCount; ++i) { total += shared[i]; }
        inverse = rsqrt(total / float(size) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < size; i += laneCount) {
        out[i] = x[i] * inverse * bf16_to_float(scale[i]);
    }
}

/// Applique RoPE à un vecteur de têtes.
///
/// **Découpage en deux moitiés**, pas en paires entrelacées : la composante `i` se marie
/// avec la composante `i + headDim/2`. C'est l'inverse du SwiGLU du même modèle. Les
/// tables cos/sin portent déjà la concentration YaRN.
kernel void rope_apply(
    device float       *x        [[buffer(0)]],  // [heads * headDim], modifié sur place
    device const float *cosTable [[buffer(1)]],  // [headDim/2]
    device const float *sinTable [[buffer(2)]],
    constant uint2     &dims     [[buffer(3)]],  // (heads, headDim)
    uint gid [[thread_position_in_grid]])
{
    const uint heads   = dims.x;
    const uint headDim = dims.y;
    // `half` est un type réservé en Metal : le nom de cette variable ne peut pas l'être.
    const uint halfDim = headDim / 2u;
    if (gid >= heads * halfDim) { return; }

    const uint head  = gid / halfDim;
    const uint index = gid % halfDim;
    const uint base  = head * headDim;

    const float c = cosTable[index];
    const float s = sinTable[index];
    const float x1 = x[base + index];
    const float x2 = x[base + halfDim + index];

    x[base + index]           = x1 * c - x2 * s;
    x[base + halfDim + index] = x2 * c + x1 * s;
}

/// SwiGLU de GPT-OSS.
///
/// Trois écarts par rapport à la formulation habituelle, tous vérifiés sur
/// l'implémentation de référence :
///   1. découpage en **indices pairs et impairs**, donc `gate_up` est entrelacé ;
///   2. écrêtage **asymétrique** — la gate seulement par le haut, la linéaire des deux côtés ;
///   3. la branche linéaire reçoit **+1**, et le swish utilise `sigmoid(1,702·x)`.
kernel void swiglu(
    device const float *x     [[buffer(0)]],  // [2 * size] entrelacé
    device float       *out   [[buffer(1)]],  // [size]
    constant uint      &size  [[buffer(2)]],
    constant float2    &params [[buffer(3)]], // (alpha, limit)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) { return; }
    const float alpha = params.x;
    const float limit = params.y;

    const float gate   = min(x[2u * gid], limit);
    const float linear = clamp(x[2u * gid + 1u], -limit, limit);
    const float activated = gate * (1.0f / (1.0f + exp(-alpha * gate)));
    out[gid] = activated * (linear + 1.0f);
}

/// Attention de décodage : une requête contre le cache KV, avec puits.
///
/// Le **puits** est un logit appris par tête, présent dans le softmax mais absent du
/// numérateur : il grossit le dénominateur, ce qui permet à une tête de ne rien regarder.
/// On l'exploite ici pour initialiser proprement le softmax en ligne — le maximum courant
/// démarre au puits, et le dénominateur à 1.
///
/// Le cache KV est indexé de façon circulaire quand `ringSize > 0`, ce qui couvre les
/// couches à fenêtre glissante sans copier quoi que ce soit : la fenêtre de 128 tokens de
/// GPT-OSS tient dans un anneau borné, quelle que soit la longueur du contexte.
///
/// Un threadgroup par tête de requête, réduction SIMD sur la dimension des clés.
kernel void attention_decode(
    device const float  *q         [[buffer(0)]],  // [qHeads * headDim]
    device const half   *kCache    [[buffer(1)]],  // [capacity][kvHeads][headDim]
    device const half   *vCache    [[buffer(2)]],
    device const ushort *sinks     [[buffer(3)]],  // BF16, [qHeads]
    device float        *out       [[buffer(4)]],  // [qHeads * headDim]
    constant uint4      &dims      [[buffer(5)]],  // (qHeads, kvHeads, headDim, keyCount)
    constant uint2      &ring      [[buffer(6)]],  // (ringSize, startPosition)
    constant float      &smScale   [[buffer(7)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]])
{
    const uint qHeads   = dims.x;
    const uint kvHeads  = dims.y;
    const uint headDim  = dims.z;
    const uint keyCount = dims.w;
    if (head >= qHeads) { return; }

    const uint qMult  = qHeads / kvHeads;
    const uint kvHead = head / qMult;
    const uint ringSize = ring.x;
    const uint startPosition = ring.y;

    // Softmax en ligne, amorcé sur le puits : maximum = puits, dénominateur = 1.
    float runningMax = bf16_to_float(sinks[head]);
    float denominator = 1.0f;
    // Accumulateur partiel de chaque voie sur sa part des composantes.
    float accumulator[8];
    for (uint i = 0; i < 8u; ++i) { accumulator[i] = 0.0f; }
    const uint slice = (headDim + 31u) / 32u;  // composantes par voie

    for (uint key = 0; key < keyCount; ++key) {
        const uint position = startPosition + key;
        const uint physical = ringSize > 0u ? (position % ringSize) : position;
        const uint kBase = (physical * kvHeads + kvHead) * headDim;

        // Produit scalaire réparti sur les voies du groupe SIMD.
        float partial = 0.0f;
        for (uint i = lane; i < headDim; i += 32u) {
            partial += q[head * headDim + i] * float(kCache[kBase + i]);
        }
        const float logit = simd_sum(partial) * smScale;

        const float newMax = max(runningMax, logit);
        const float correction = exp(runningMax - newMax);
        const float weight = exp(logit - newMax);
        denominator = denominator * correction + weight;
        runningMax = newMax;

        for (uint s = 0; s < slice; ++s) {
            const uint i = lane + s * 32u;
            if (i < headDim) {
                accumulator[s] = accumulator[s] * correction
                    + weight * float(vCache[kBase + i]);
            }
        }
    }

    const float inverse = 1.0f / denominator;
    for (uint s = 0; s < slice; ++s) {
        const uint i = lane + s * 32u;
        if (i < headDim) { out[head * headDim + i] = accumulator[s] * inverse; }
    }
}

/// Sélection des `topK` meilleurs experts puis softmax **sur ces seuls logits**.
///
/// Une seule voie fait la sélection : avec 32 ou 128 experts, la parallélisation coûterait
/// plus que le calcul. En cas d'égalité, l'indice le plus petit gagne — sans cette
/// convention, le décodage glouton ne serait pas reproductible.
kernel void router_topk(
    device const float *logits  [[buffer(0)]],  // [expertCount]
    device uint        *indices [[buffer(1)]],  // [topK]
    device float       *weights [[buffer(2)]],  // [topK]
    constant uint2     &dims    [[buffer(3)]],  // (expertCount, topK)
    uint lane [[thread_position_in_threadgroup]])
{
    if (lane != 0) { return; }
    const uint expertCount = dims.x;
    const uint topK = min(dims.y, 8u);

    float chosenValues[8];
    uint  chosenIndices[8];

    for (uint k = 0; k < topK; ++k) {
        float best = -INFINITY;
        uint bestIndex = 0;
        for (uint e = 0; e < expertCount; ++e) {
            bool taken = false;
            for (uint j = 0; j < k; ++j) { taken = taken || (chosenIndices[j] == e); }
            if (taken) { continue; }
            // Strictement supérieur : à égalité, le plus petit indice est conservé.
            if (logits[e] > best) { best = logits[e]; bestIndex = e; }
        }
        chosenValues[k] = best;
        chosenIndices[k] = bestIndex;
    }

    float peak = chosenValues[0];
    for (uint k = 1; k < topK; ++k) { peak = max(peak, chosenValues[k]); }
    float total = 0.0f;
    for (uint k = 0; k < topK; ++k) {
        chosenValues[k] = exp(chosenValues[k] - peak);
        total += chosenValues[k];
    }
    for (uint k = 0; k < topK; ++k) {
        indices[k] = chosenIndices[k];
        weights[k] = chosenValues[k] / total;
    }
}

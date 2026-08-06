// Concatenated after common.metal, which provides bf16_to_float.
//
// GPT-OSS's non-MoE operators: normalization, RoPE, attention with sinks, SwiGLU, router.
// Each is validated against HydraReference's CPU implementation, itself validated against an
// independent transcription of gpt_oss/torch/model.py.

/// RMSNorm: x / sqrt(mean(x²) + eps) * scale.
/// One threadgroup per row, SIMD reduction.
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

/// Applies RoPE to a vector of heads.
///
/// **Split into two halves**, not into interleaved pairs: component `i` pairs with component
/// `i + headDim/2`. This is the opposite of the SwiGLU in the same model. The cos/sin tables
/// already carry the YaRN concentration.
kernel void rope_apply(
    device float       *x        [[buffer(0)]],  // [heads * headDim], modifié sur place
    device const float *cosTable [[buffer(1)]],  // [headDim/2]
    device const float *sinTable [[buffer(2)]],
    constant uint2     &dims     [[buffer(3)]],  // (heads, headDim)
    uint gid [[thread_position_in_grid]])
{
    const uint heads   = dims.x;
    const uint headDim = dims.y;
    // `half` is a reserved type in Metal: this variable cannot carry that name.
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

/// GPT-OSS's SwiGLU.
///
/// Three departures from the usual formulation, all verified against the reference
/// implementation:
///   1. split on **even and odd indices**, so `gate_up` is interleaved;
///   2. **asymmetric** clamping — the gate from above only, the linear branch on both sides;
///   3. the linear branch gets **+1**, and the swish uses `sigmoid(1.702·x)`.
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

/// Decoding attention: one query against the KV cache, with sinks.
///
/// The **sink** is a learned per-head logit, present in the softmax but absent from the
/// numerator: it enlarges the denominator, which lets a head look at nothing. We exploit it
/// here to initialize the online softmax cleanly — the running maximum starts at the sink,
/// and the denominator at 1.
///
/// The KV cache is indexed circularly when `ringSize > 0`, which covers the sliding-window
/// layers without copying anything: GPT-OSS's 128-token window fits in a bounded ring
/// whatever the context length.
///
/// One threadgroup per query head, SIMD reduction over the key dimension.
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

    // Online softmax, seeded on the sink: maximum = sink, denominator = 1.
    float runningMax = bf16_to_float(sinks[head]);
    float denominator = 1.0f;
    // Each lane's partial accumulator over its share of the components.
    float accumulator[8];
    for (uint i = 0; i < 8u; ++i) { accumulator[i] = 0.0f; }
    const uint slice = (headDim + 31u) / 32u;  // components per lane

    for (uint key = 0; key < keyCount; ++key) {
        const uint position = startPosition + key;
        const uint physical = ringSize > 0u ? (position % ringSize) : position;
        const uint kBase = (physical * kvHeads + kvHead) * headDim;

        // Dot product spread across the lanes of the SIMD group.
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

/// Selects the `topK` best experts, then softmax **over those logits alone**.
///
/// A single lane does the selection: with 32 or 128 experts, parallelizing would cost more
/// than the computation. On a tie the smallest index wins — without that convention, greedy
/// decoding would not be reproducible.
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
            // Strictly greater: on a tie, the smaller index is kept.
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

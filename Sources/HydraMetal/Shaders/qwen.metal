#include <metal_stdlib>
using namespace metal;

// Qwen3.5/3.6 MoE's linear attention.
//
// Every constant and every ordering here is transcribed in D-027 and checked against
// `QwenReferenceOps`, which is checked in turn against an independent Python transcription of
// `modeling_qwen3_5.py`. Five of them cannot be guessed and each is finite when wrong.

/// One decode step of the gated delta rule, for every value head of one layer.
///
/// **One threadgroup a value head, one thread a column of that head's state.** The state is
/// `[keyDim][valueDim]` and the recurrence contracts over the key dimension, so a thread that
/// owns column `j` can compute `kv_mem[j]`, `delta[j]` and `out[j]` entirely on its own. The
/// only value shared across threads is the l2 norm of q and k, which is a reduction over the
/// key dimension and is done once at the top.
///
/// The state does not fit in threadgroup memory: 128 by 128 floats is 64 KiB against a 32 KiB
/// budget. So it stays in device memory and the cost of a step is how many times it is
/// traversed. Written to be traversed **three** times rather than four:
///
///   - pass one reads it, decays each value and accumulates `kv_mem`, and writes nothing;
///   - `delta` is then a per-thread scalar;
///   - pass two reads it again, recomputes the decay, adds the outer product, writes once, and
///     accumulates the output from the value it has in hand.
///
/// Recomputing the decay in the second pass is a multiply, against a 64 KiB write to avoid it.
kernel void qwen_delta_rule_step(
    device float        *state    [[buffer(0)]],  // [valueHeads][keyDim][valueDim]
    device const float  *query    [[buffer(1)]],  // [keyHeads][keyDim], before the l2 norm
    device const float  *key      [[buffer(2)]],  // [keyHeads][keyDim], before the l2 norm
    device const float  *value    [[buffer(3)]],  // [valueHeads][valueDim]
    device const float  *a        [[buffer(4)]],  // [valueHeads], the decay projection
    device const float  *b        [[buffer(5)]],  // [valueHeads], the gate projection
    device const float  *logA     [[buffer(6)]],  // [valueHeads], learned
    device const float  *dtBias   [[buffer(7)]],  // [valueHeads], learned
    device float        *out      [[buffer(8)]],  // [valueHeads][valueDim]
    constant uint4      &dims     [[buffer(9)]],  // (valueHeads, keyHeads, keyDim, valueDim)
    constant float      &eps      [[buffer(10)]],
    uint head      [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    const uint valueHeads = dims.x;
    const uint keyHeads   = dims.y;
    const uint keyDim     = dims.z;
    const uint valueDim   = dims.w;
    if (head >= valueHeads) { return; }

    // Query and key heads are shared across value heads, the same grouped arrangement the
    // attention layers use, applied to a recurrence (D-027).
    const uint keyHead = head / (valueHeads / keyHeads);
    device const float *q = query + (ulong)keyHead * keyDim;
    device const float *k = key + (ulong)keyHead * keyDim;
    device float *S = state + (ulong)head * keyDim * valueDim;

    // --- The l2 norms, shared by every thread in this head ---
    //
    // Not an RMS norm: no division by the length and no learned weight. Confusing the two is a
    // factor of sqrt(keyDim) on both q and k, and leaves everything finite.
    threadgroup float partials[32];
    threadgroup float qScale;
    threadgroup float kScale;
    threadgroup float decay;
    threadgroup float beta;

    float qSum = 0.0f;
    float kSum = 0.0f;
    for (uint i = lane; i < keyDim; i += laneCount) {
        qSum += q[i] * q[i];
        kSum += k[i] * k[i];
    }
    // Two reductions, run one after the other through the same scratch.
    float reduced = simd_sum(qSum);
    if (lane % 32u == 0) { partials[lane / 32u] = reduced; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint s = 0; s < simdCount; ++s) { total += partials[s]; }
        // The scale uses the **key** dimension and multiplies the query **after** its norm.
        qScale = rsqrt(total + eps) / sqrt((float)keyDim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    reduced = simd_sum(kSum);
    if (lane % 32u == 0) { partials[lane / 32u] = reduced; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint s = 0; s < simdCount; ++s) { total += partials[s]; }
        kScale = rsqrt(total + eps);

        // `g` is not a projection output: it is built from two learned per-head parameters and
        // lies in (0, 1], which is what makes the state decay rather than grow.
        const float x = a[head] + dtBias[head];
        const float softplus = x > 20.0f ? x : log(1.0f + exp(x));
        decay = exp(-exp(logA[head]) * softplus);
        beta = 1.0f / (1.0f + exp(-b[head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane >= valueDim) { return; }
    const uint j = lane;
    const float g = decay;

    // --- Pass one: what the decayed state already holds for this key ---
    //
    // The decay multiplies the state **before** this read. Reading first and decaying after is
    // a different answer, and a finite one.
    float memory = 0.0f;
    for (uint i = 0; i < keyDim; ++i) {
        memory += (S[(ulong)i * valueDim + j] * g) * (k[i] * kScale);
    }

    const float delta = (value[(ulong)head * valueDim + j] - memory) * beta;

    // --- Pass two: accumulate the outer product, and read with the query ---
    float result = 0.0f;
    for (uint i = 0; i < keyDim; ++i) {
        const ulong at = (ulong)i * valueDim + j;
        const float updated = S[at] * g + (k[i] * kScale) * delta;
        S[at] = updated;
        result += updated * (q[i] * qScale);
    }
    out[(ulong)head * valueDim + j] = result;
}

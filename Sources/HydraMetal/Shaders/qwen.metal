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

/// One decode step of the depthwise causal convolution, and the shift of its window.
///
/// Depthwise: one kernel a channel, no mixing between channels, so a thread owns a channel
/// outright. Causal: the taps reach backwards only, and anything before the start of the
/// sequence is zero.
///
/// The window is a fixed `[kernel - 1][convDim]` shift register, oldest row first, zeroed when
/// a sequence starts. That zeroing **is** the left padding: a tap reaching past the beginning
/// reads a row that has never been written, which is what the reference computes as padding,
/// so the two agree without the kernel tracking how full the window is.
///
/// The shift happens here rather than in a second dispatch because a thread owns its channel's
/// whole column and nothing else reads it. This is per-layer state exactly as the recurrent
/// state is, and forgetting to advance it is silent: the first three tokens of every turn would
/// be computed against stale neighbours.
kernel void qwen_causal_conv_step(
    device float       *window [[buffer(0)]],  // [kernel - 1][convDim], oldest row first
    device const float *input  [[buffer(1)]],  // [convDim]
    device const float *weight [[buffer(2)]],  // [convDim][kernel]
    device const float *bias   [[buffer(3)]],  // [convDim], read only when dims.z is set
    device float       *out    [[buffer(4)]],  // [convDim]
    constant uint3     &dims   [[buffer(5)]],  // (convDim, kernel, hasBias)
    uint channel [[thread_position_in_grid]])
{
    const uint convDim = dims.x;
    const uint kernelSize = dims.y;
    const bool hasBias = dims.z != 0u;
    if (channel >= convDim) { return; }

    const float current = input[channel];
    float sum = hasBias ? bias[channel] : 0.0f;

    // Taps run oldest to newest, and the newest is the token being decoded.
    for (uint tap = 0; tap < kernelSize; ++tap) {
        const uint age = kernelSize - 1u - tap;
        const float value = age == 0u
            ? current
            : window[(ulong)((kernelSize - 1u) - age) * convDim + channel];
        sum += value * weight[(ulong)channel * kernelSize + tap];
    }

    out[channel] = sum / (1.0f + exp(-sum));  // SiLU: x · sigmoid(x)

    // Advance the window: drop the oldest, append the token just consumed.
    for (uint i = 0; i + 2u < kernelSize; ++i) {
        window[(ulong)i * convDim + channel] = window[(ulong)(i + 1u) * convDim + channel];
    }
    if (kernelSize > 1u) {
        window[(ulong)(kernelSize - 2u) * convDim + channel] = current;
    }
}

/// Splits `q_proj`'s output into the query and its gate, per head.
///
/// The projection emits `heads · headDim · 2` values, viewed as `[heads][headDim · 2]` and
/// chunked, so each head's slice holds its own query followed by its own gate. Splitting the
/// tensor down the middle instead hands head `h` the gate of a different head, which is finite
/// and wrong for every head but the first (D-027).
kernel void qwen_split_query_gate(
    device const float *combined [[buffer(0)]],  // [heads][headDim * 2]
    device float       *query    [[buffer(1)]],  // [heads][headDim]
    device float       *gate     [[buffer(2)]],  // [heads][headDim]
    constant uint2     &dims     [[buffer(3)]],  // (heads, headDim)
    uint gid [[thread_position_in_grid]])
{
    const uint heads = dims.x;
    const uint headDim = dims.y;
    if (gid >= heads * headDim) { return; }

    const uint head = gid / headDim;
    const uint i = gid % headDim;
    const ulong source = (ulong)head * headDim * 2u;
    query[gid] = combined[source + i];
    gate[gid] = combined[source + headDim + i];
}

/// `output · sigmoid(gate)`, in place.
///
/// Applied to the attention output, not to the query: the gate decides how much of what
/// attention returned survives, which is a different operation from scaling what it attends
/// with.
kernel void qwen_apply_output_gate(
    device float       *output [[buffer(0)]],
    device const float *gate   [[buffer(1)]],
    constant uint      &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) { return; }
    output[gid] = output[gid] / (1.0f + exp(-gate[gid]));
}

/// The gated RMS norm on a linear layer's output, one threadgroup a value head.
///
/// Per head, over the value dimension: the variance is a head's own, the learned weight is
/// shared by every head, and the gate is **this head's slice** of the `z` projection. A kernel
/// that gates every head with the first slice produces plausible values and is wrong for every
/// head but one, which no property of the block detects (D-027).
///
/// The order is load-bearing and matches the reference: normalize, apply the learned weight,
/// then multiply by `silu(gate)`. The gate is outside the variance and does not participate in
/// it.
kernel void qwen_gated_rms_norm_heads(
    device const float  *input  [[buffer(0)]],  // [heads][dim]
    device const ushort *weight [[buffer(1)]],  // BF16, [dim], shared by every head
    device const float  *gate   [[buffer(2)]],  // [heads][dim]
    device float        *out    [[buffer(3)]],  // [heads][dim]
    constant uint2      &dims   [[buffer(4)]],  // (heads, dim)
    constant float      &eps    [[buffer(5)]],
    uint head      [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    const uint heads = dims.x;
    const uint dim = dims.y;
    if (head >= heads) { return; }

    device const float *x = input + (ulong)head * dim;
    device const float *g = gate + (ulong)head * dim;
    device float *o = out + (ulong)head * dim;

    float partial = 0.0f;
    for (uint i = lane; i < dim; i += laneCount) { partial += x[i] * x[i]; }

    threadgroup float shared[32];
    partial = simd_sum(partial);
    if (lane % 32u == 0) { shared[lane / 32u] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float inverse;
    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint s = 0; s < simdCount; ++s) { total += shared[s]; }
        inverse = rsqrt(total / (float)dim + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < dim; i += laneCount) {
        const float gated = g[i] / (1.0f + exp(-g[i]));   // silu
        o[i] = bf16_to_float(weight[i]) * (x[i] * inverse) * gated;
    }
}

/// `silu(gate) · up`, the SwiGLU Qwen's `hidden_act` asks for.
///
/// Neither of the two already here. Gemma's `gelu_mul` uses `gelu_pytorch_tanh`, and GPT-OSS's
/// `swiglu` clamps its branches and adds one to the linear side (D-014). Both are finite and
/// both are wrong here, differing from this by a few percent per element, which compounds over
/// forty layers into a model that is merely worse.
kernel void qwen_silu_multiply(
    device const float *gate  [[buffer(0)]],
    device const float *up    [[buffer(1)]],
    device float       *out   [[buffer(2)]],
    constant uint      &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) { return; }
    const float g = gate[gid];
    out[gid] = (g / (1.0f + exp(-g))) * up[gid];
}

/// Scales a vector by `sigmoid` of a single logit, in place.
///
/// The shared expert's gate is one row, so one number a token, and it scales that branch's
/// **output**. Applying it to the input instead would put it inside the projections and change
/// what they see rather than how much of them survives.
kernel void qwen_scale_by_sigmoid(
    device float       *target [[buffer(0)]],
    device const float *logit  [[buffer(1)]],
    constant uint      &count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) { return; }
    target[gid] = target[gid] / (1.0f + exp(-logit[0]));
}

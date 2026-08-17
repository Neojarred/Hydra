#include <metal_stdlib>
using namespace metal;

// The vision tower's kernels.
//
// Everything linear in the tower is a BF16 matrix against a batch of patches, which `bf16_gemm`
// already does. What is missing is the four operations that are not: a LayerNorm rather than the
// RMSNorm the text models use, the tanh GELU, a two-dimensional rotary, and an attention with no
// causal mask.
//
// The last one is the reason these cannot be borrowed from `batch.metal`. Every attention in
// this codebase is causal because every model in it is a language model; a vision tower's
// patches all see each other, and running the causal kernel here would give the first patch no
// context and the last one everything, which is finite, plausible and wrong.

float bf16_to_float_v(ushort bits) {
    const uint widened = uint(bits) << 16;
    return as_type<float>(widened);
}

/// LayerNorm over a batch of tokens: subtract the mean, divide by the deviation, scale, shift.
///
/// **The mean is subtracted**, which is what separates this from `rms_norm` beside it. The
/// tower carries a `.bias` next to every `.weight`, which is the tell that it is not an RMSNorm.
///
/// One threadgroup a token, reduced through threadgroup memory. Two passes over the row: the
/// mean, then the variance. A single-pass sum of squares would be one pass and is the classic
/// way to lose precision here, since it subtracts two large nearly equal numbers.
kernel void vision_layer_norm(
    device const float  *x      [[buffer(0)]],  // [tokens][width]
    device const ushort *weight [[buffer(1)]],  // BF16 [width]
    device const ushort *bias   [[buffer(2)]],  // BF16 [width]
    device float        *y      [[buffer(3)]],  // [tokens][width]
    constant uint2      &dims   [[buffer(4)]],  // (width, tokens)
    constant float      &eps    [[buffer(5)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane  [[thread_position_in_threadgroup]],
    uint width [[threads_per_threadgroup]])
{
    const uint size = dims.x;
    if (token >= dims.y) { return; }
    device const float *row = x + (ulong)token * size;
    device float *out = y + (ulong)token * size;

    // Every thread reduces the per-simdgroup partials itself rather than electing one to do it
    // and publish the answer. The elected form needs a threadgroup variable written on one
    // branch and read on all of them, which the Metal compiler rejects outright as possibly
    // uninitialized however many barriers stand between the write and the read.
    threadgroup float shared[32];
    const uint simdCount = (width + 31u) / 32u;

    float sum = 0.0f;
    for (uint i = lane; i < size; i += width) { sum += row[i]; }
    sum = simd_sum(sum);
    if ((lane % 32u) == 0u) { shared[lane / 32u] = sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gathered = 0.0f;
    for (uint s = 0; s < simdCount; ++s) { gathered += shared[s]; }
    const float mean = gathered / float(size);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float variance = 0.0f;
    for (uint i = lane; i < size; i += width) {
        const float centred = row[i] - mean;
        variance += centred * centred;
    }
    variance = simd_sum(variance);
    if ((lane % 32u) == 0u) { shared[lane / 32u] = variance; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    gathered = 0.0f;
    for (uint s = 0; s < simdCount; ++s) { gathered += shared[s]; }
    const float scale = rsqrt(gathered / float(size) + eps);
    for (uint i = lane; i < size; i += width) {
        out[i] = (row[i] - mean) * scale * bf16_to_float_v(weight[i]) + bf16_to_float_v(bias[i]);
    }
}

/// `gelu_pytorch_tanh`, in place.
///
/// The approximation the config names, not the exact erf form. They differ by about 1e-3 at the
/// knee, which is above the tolerance this project checks kernels to, so they are not
/// interchangeable here even though they are usually described as if they were.
kernel void vision_gelu(
    device float   *x     [[buffer(0)]],
    constant uint  &count [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= count) { return; }
    const float value = x[index];
    const float inner = 0.7978845608028654f * (value + 0.044715f * value * value * value);
    x[index] = 0.5f * value * (1.0f + precise::tanh(inner));
}

/// The tower's two-dimensional rotary, applied to the query and key halves of `qkv` in place.
///
/// **`rotate_half`, not the interleaved form.** Component `i` turns against `i + headDim/2`, so
/// the first quarter of a head turns with the patch's row and the second with its column. The
/// text model beside this one is interleaved, by its own `mrope_interleaved` flag, and the two
/// are indistinguishable by any check on shape, finiteness or norm.
///
/// `qkv` is `[patches][3 * heads * headDim]`, laid out query block, key block, value block, as
/// the reference's `reshape(seq, 3, heads, -1)` implies. The value block is left alone.
kernel void vision_rotary(
    device float       *qkv    [[buffer(0)]],  // [patches][3 * hidden]
    device const float *angles [[buffer(1)]],  // [patches][headDim / 2]
    constant uint4     &dims   [[buffer(2)]],  // (patches, heads, headDim, _)
    uint2 grid [[thread_position_in_grid]])    // (component, patch)
{
    const uint patches = dims.x;
    const uint heads = dims.y;
    const uint headDim = dims.z;
    const uint halfDim = headDim / 2u;  // `half` is a Metal type name


    const uint patch = grid.y;
    const uint slot = grid.x;                  // head * half + componentWithinHalf
    if (patch >= patches || slot >= heads * halfDim) { return; }

    const uint head = slot / halfDim;
    const uint component = slot % halfDim;
    const uint hidden = heads * headDim;

    const float angle = angles[(ulong)patch * halfDim + component];
    const float c = cos(angle), s = sin(angle);

    // The query block, then the key block. The value block is not rotated.
    for (uint block = 0; block < 2u; ++block) {
        const ulong base = (ulong)patch * 3u * hidden + (ulong)block * hidden
            + (ulong)head * headDim;
        const float low = qkv[base + component];
        const float high = qkv[base + component + halfDim];
        qkv[base + component] = low * c - high * s;
        qkv[base + component + halfDim] = high * c + low * s;
    }
}

/// Bidirectional attention over one image's patches.
///
/// Every patch attends to every patch: there is no mask and no position ordering. The online
/// softmax is the same shape as `attention_decode`'s, and for the same reason: the score matrix
/// for a full-resolution image is 16384 by 16384 a head, which is a gigabyte that must never be
/// written down.
///
/// One threadgroup a (head, query), with the simdgroups splitting the keys. That is the same
/// launch shape M-068 found too narrow for decode, and here it is not: the grid is
/// `heads * patches`, which is thousands of threadgroups even for a small image.
kernel void vision_attention(
    device const float *qkv     [[buffer(0)]],  // [patches][3 * hidden]
    device float       *out     [[buffer(1)]],  // [patches][hidden]
    constant uint4     &dims    [[buffer(2)]],  // (patches, heads, headDim, _)
    constant float     &scale   [[buffer(3)]],
    uint2 group     [[threadgroup_position_in_grid]],   // (head, query)
    uint  lane      [[thread_index_in_simdgroup]],
    uint  simd      [[simdgroup_index_in_threadgroup]],
    uint  simdCount [[simdgroups_per_threadgroup]])
{
    const uint patches = dims.x;
    const uint heads = dims.y;
    const uint headDim = dims.z;
    const uint head = group.x;
    const uint query = group.y;
    if (head >= heads || query >= patches) { return; }

    const uint hidden = heads * headDim;
    const ulong queryBase = (ulong)query * 3u * hidden + (ulong)head * headDim;

    // Seeded from the first key rather than from -inf: a running maximum that starts at
    // -infinity gives `exp(-inf - max)` for a simdgroup that saw no keys, and that NaN reaches
    // every patch. The same trap the decode kernel carries a comment about.
    float runningMax = -INFINITY;
    float denominator = 0.0f;

    constexpr uint maxSlice = 512u / 32u;
    float accumulator[maxSlice];
    for (uint i = 0; i < maxSlice; ++i) { accumulator[i] = 0.0f; }
    const uint slice = (headDim + 31u) / 32u;

    for (uint key = simd; key < patches; key += simdCount) {
        const ulong keyBase = (ulong)key * 3u * hidden + (ulong)hidden + (ulong)head * headDim;

        float partial = 0.0f;
        for (uint i = lane; i < headDim; i += 32u) {
            partial += qkv[queryBase + i] * qkv[keyBase + i];
        }
        const float logit = simd_sum(partial) * scale;

        const float newMax = max(runningMax, logit);
        const float correction = isinf(runningMax) ? 0.0f : exp(runningMax - newMax);
        const float weight = exp(logit - newMax);
        denominator = denominator * correction + weight;
        runningMax = newMax;

        const ulong valueBase = keyBase + (ulong)hidden;
        for (uint s = 0; s < slice; ++s) {
            const uint i = lane + s * 32u;
            if (i < headDim) {
                accumulator[s] = accumulator[s] * correction + weight * qkv[valueBase + i];
            }
        }
    }

    threadgroup float sharedMax[32];
    threadgroup float sharedDen[32];
    threadgroup float sharedOut[512];

    if (lane == 0u) { sharedMax[simd] = runningMax; }
    for (uint i = simd * 32u + lane; i < headDim; i += simdCount * 32u) { sharedOut[i] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float groupMax = sharedMax[0];
    for (uint s = 1; s < simdCount; ++s) { groupMax = max(groupMax, sharedMax[s]); }

    // A simdgroup that saw no keys at all keeps -inf and must contribute nothing rather than a
    // NaN, which is what the guard below is for: `exp(-inf - finite)` is 0, but
    // `exp(-inf - -inf)` is NaN, and both arise when `patches < simdCount`.
    const float rescale = isinf(runningMax) ? 0.0f : exp(runningMax - groupMax);
    if (lane == 0u) { sharedDen[simd] = denominator * rescale; }

    for (uint s = 0; s < simdCount; ++s) {
        if (s == simd) {
            for (uint k = 0; k < slice; ++k) {
                const uint i = lane + k * 32u;
                if (i < headDim) { sharedOut[i] += accumulator[k] * rescale; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float total = 0.0f;
    for (uint s = 0; s < simdCount; ++s) { total += sharedDen[s]; }
    const float inverse = 1.0f / total;
    for (uint i = simd * 32u + lane; i < headDim; i += simdCount * 32u) {
        out[(ulong)query * hidden + (ulong)head * headDim + i] = sharedOut[i] * inverse;
    }
}

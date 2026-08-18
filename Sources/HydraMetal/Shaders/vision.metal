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

/// Bidirectional attention over one image's patches, with the keys staged in threadgroup memory.
///
/// **Templated on the head width**, which is what makes it fast rather than any of the three
/// things that looked like the problem. M-070 measured this kernel at 1.8 GMAC/s where the
/// projections in the same tower reach 800, and isolation by removal settled what costs:
///
///     without simd_sum          22.6 s     no effect
///     without the exponentials  22.5 s     no effect
///     without the V accumulation 14.8 s    a third of the time
///
/// So it is neither the cross-lane reduction nor the transcendentals. It is the per-element
/// loops, which could not unroll because `slice` was a runtime value, and which carried a bounds
/// branch on every element for the same reason. With the width a compile-time constant the trip
/// count is known, the loops unroll, and the branch disappears; `mlx_affine_gemv_t` is templated
/// on its bit width for the same reason.
///
/// A threadgroup owns eight queries, one to a simdgroup, and the keys are pulled into
/// threadgroup memory a tile at a time so the same bytes serve all eight. Because a simdgroup
/// owns a whole query rather than a slice of its keys, each runs its own online softmax from
/// start to finish and there is no cross-simdgroup merge.
template <uint HEAD_DIM>
kernel void vision_attention_t(
    device const float *qkv     [[buffer(0)]],  // [patches][3 * hidden]
    device float       *out     [[buffer(1)]],  // [patches][hidden]
    constant uint4     &dims    [[buffer(2)]],  // (patches, heads, headDim, _)
    constant float     &scale   [[buffer(3)]],
    uint2 group     [[threadgroup_position_in_grid]],   // (head, query tile)
    uint2 threadId  [[thread_position_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  simd      [[simdgroup_index_in_threadgroup]],
    uint  simdCount [[simdgroups_per_threadgroup]])
{
    constexpr uint kKeyTile = 16u;
    constexpr uint kSlice = (HEAD_DIM + 31u) / 32u;

    const uint patches = dims.x;
    const uint heads = dims.y;
    const uint head = group.x;
    if (head >= heads) { return; }

    const uint hidden = heads * HEAD_DIM;
    const uint query = group.y * simdCount + simd;

    threadgroup float keyTile[kKeyTile * HEAD_DIM];
    threadgroup float valueTile[kKeyTile * HEAD_DIM];

    // A simdgroup past the end still takes the tour, because a barrier some threads never reach
    // is undefined behaviour rather than a hang.
    const bool active = query < patches;
    const ulong queryBase = active ? (ulong)query * 3u * hidden + (ulong)head * HEAD_DIM : 0;

    float runningMax = -INFINITY;
    float denominator = 0.0f;
    float accumulator[kSlice];
    float queryValues[kSlice];
    #pragma clang loop unroll(full)
    for (uint s = 0; s < kSlice; ++s) {
        accumulator[s] = 0.0f;
        const uint i = lane + s * 32u;
        queryValues[s] = (active && i < HEAD_DIM) ? qkv[queryBase + i] : 0.0f;
    }

    constexpr uint tileValues = kKeyTile * HEAD_DIM;
    const uint width = simdCount * 32u;

    for (uint tileStart = 0; tileStart < patches; tileStart += kKeyTile) {
        const uint tileCount = min(kKeyTile, patches - tileStart);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = threadId.x; i < tileValues; i += width) {
            const uint key = i / HEAD_DIM;
            const uint component = i % HEAD_DIM;
            if (key < tileCount) {
                const ulong base = (ulong)(tileStart + key) * 3u * hidden
                    + (ulong)head * HEAD_DIM + component;
                keyTile[i] = qkv[base + hidden];
                valueTile[i] = qkv[base + 2u * hidden];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) { continue; }

        for (uint key = 0; key < tileCount; ++key) {
            const uint keyBase = key * HEAD_DIM;
            float partial = 0.0f;
            #pragma clang loop unroll(full)
            for (uint s = 0; s < kSlice; ++s) {
                const uint i = lane + s * 32u;
                if (i < HEAD_DIM) { partial += queryValues[s] * keyTile[keyBase + i]; }
            }
            const float logit = simd_sum(partial) * scale;

            const float newMax = max(runningMax, logit);
            const float correction = isinf(runningMax) ? 0.0f : exp(runningMax - newMax);
            const float weight = exp(logit - newMax);
            denominator = denominator * correction + weight;
            runningMax = newMax;

            #pragma clang loop unroll(full)
            for (uint s = 0; s < kSlice; ++s) {
                const uint i = lane + s * 32u;
                if (i < HEAD_DIM) {
                    accumulator[s] = accumulator[s] * correction
                        + weight * valueTile[keyBase + i];
                }
            }
        }
    }

    if (!active) { return; }
    const float inverse = 1.0f / denominator;
    #pragma clang loop unroll(full)
    for (uint s = 0; s < kSlice; ++s) {
        const uint i = lane + s * 32u;
        if (i < HEAD_DIM) {
            out[(ulong)query * hidden + (ulong)head * HEAD_DIM + i] = accumulator[s] * inverse;
        }
    }
}

// Qwen's tower is 1152 over 16 heads. The others exist so a different geometry does not fall off
// a cliff, and `ForwardEncoder` picks by width with a generic fallback.
template [[host_name("vision_attention_72")]] kernel void
vision_attention_t<72>(
    device const float *, device float *, constant uint4 &, constant float &,
    uint2, uint2, uint, uint, uint);

template [[host_name("vision_attention_64")]] kernel void
vision_attention_t<64>(
    device const float *, device float *, constant uint4 &, constant float &,
    uint2, uint2, uint, uint, uint);

template [[host_name("vision_attention_128")]] kernel void
vision_attention_t<128>(
    device const float *, device float *, constant uint4 &, constant float &,
    uint2, uint2, uint, uint, uint);

/// Bidirectional attention over one image's patches, with the keys staged in threadgroup memory.
///
/// M-070 measured the naive form at 2.1 GFLOP/s on a machine that does thousands, and it is 99 %
/// of the tower. It gave one threadgroup to each (head, query) and split the keys across its
/// simdgroups, so every one of the 65,536 threadgroups read its head's entire key and value set
/// from device memory: 154 GB for a 4096-patch image.
///
/// Here a threadgroup owns **eight queries, one to a simdgroup**, and the keys are pulled into
/// threadgroup memory a tile at a time and read by all eight. The same bytes now serve eight
/// queries instead of one, which is 19 GB rather than 154.
///
/// It also gets simpler rather than more complicated: because a simdgroup owns a whole query
/// rather than a slice of its keys, each one runs its own online softmax from start to finish
/// and there is no cross-simdgroup merge at the end. The `-infinity` seeding trap that the merge
/// carried goes with it.
kernel void vision_attention(
    device const float *qkv     [[buffer(0)]],  // [patches][3 * hidden]
    device float       *out     [[buffer(1)]],  // [patches][hidden]
    constant uint4     &dims    [[buffer(2)]],  // (patches, heads, headDim, _)
    constant float     &scale   [[buffer(3)]],
    uint2 group     [[threadgroup_position_in_grid]],   // (head, query tile)
    // A uint2 because Metal refuses to mix scalar and vector position attributes in one
    // signature, which is the same constraint `bf16_gemm` carries a comment about.
    uint2 threadId  [[thread_position_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  simd      [[simdgroup_index_in_threadgroup]],
    uint  simdCount [[simdgroups_per_threadgroup]])
{
    constexpr uint kKeyTile = 16u;
    constexpr uint kMaxHeadDim = 128u;

    const uint patches = dims.x;
    const uint heads = dims.y;
    const uint headDim = dims.z;
    const uint head = group.x;
    if (head >= heads || headDim > kMaxHeadDim) { return; }

    const uint hidden = heads * headDim;
    const uint query = group.y * simdCount + simd;

    threadgroup float keyTile[kKeyTile * kMaxHeadDim];
    threadgroup float valueTile[kKeyTile * kMaxHeadDim];

    // A simdgroup past the end still has to reach every barrier below, so it takes the tour
    // with no query rather than returning. A `threadgroup_barrier` that some threads of the
    // threadgroup never reach is undefined behaviour, not a hang.
    const bool active = query < patches;
    const ulong queryBase = active
        ? (ulong)query * 3u * hidden + (ulong)head * headDim : 0;

    float runningMax = -INFINITY;
    float denominator = 0.0f;
    constexpr uint maxSlice = kMaxHeadDim / 32u;
    float accumulator[maxSlice];
    for (uint i = 0; i < maxSlice; ++i) { accumulator[i] = 0.0f; }
    const uint slice = (headDim + 31u) / 32u;

    const uint tileValues = kKeyTile * headDim;
    const uint width = simdCount * 32u;

    for (uint tileStart = 0; tileStart < patches; tileStart += kKeyTile) {
        const uint tileCount = min(kKeyTile, patches - tileStart);

        // The whole threadgroup fills the tile, then everyone reads it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = threadId.x; i < tileValues; i += width) {
            const uint key = i / headDim;
            const uint component = i % headDim;
            if (key < tileCount) {
                const ulong base = (ulong)(tileStart + key) * 3u * hidden
                    + (ulong)head * headDim + component;
                keyTile[i] = qkv[base + hidden];
                valueTile[i] = qkv[base + 2u * hidden];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) { continue; }

        for (uint key = 0; key < tileCount; ++key) {
            float partial = 0.0f;
            for (uint i = lane; i < headDim; i += 32u) {
                partial += qkv[queryBase + i] * keyTile[key * headDim + i];
            }
            const float logit = simd_sum(partial) * scale;

            const float newMax = max(runningMax, logit);
            const float correction = isinf(runningMax) ? 0.0f : exp(runningMax - newMax);
            const float weight = exp(logit - newMax);
            denominator = denominator * correction + weight;
            runningMax = newMax;

            for (uint s = 0; s < slice; ++s) {
                const uint i = lane + s * 32u;
                if (i < headDim) {
                    accumulator[s] = accumulator[s] * correction
                        + weight * valueTile[key * headDim + i];
                }
            }
        }
    }

    if (!active) { return; }
    const float inverse = 1.0f / denominator;
    for (uint s = 0; s < slice; ++s) {
        const uint i = lane + s * 32u;
        if (i < headDim) {
            out[(ulong)query * hidden + (ulong)head * headDim + i] = accumulator[s] * inverse;
        }
    }
}

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

// ---------------------------------------------------------------------------------------------
// Gemma 4's vision tower.
//
// Most of what it needs already exists: `rms_norm_batch` scales by the weight directly, which is
// Gemma 4's convention (Gemma 3's `1 + w` is the other family member); `rms_norm_heads` normalizes
// each head and already handles the unscaled case its own comment describes as v_norm;
// `gelu_mul` is the GeGLU; `vision_attention` takes its scale as a parameter and Gemma's is 1.
// These three are what is left.

/// RMSNorm with no learned weight, over a batch of tokens.
///
/// Used twice, in the two places Gemma has a norm with no parameter at all: before the projector
/// into the text model, and on the attention values. `rms_norm_unscaled` beside it does one row.
kernel void rms_norm_unscaled_batch(
    device const float *x    [[buffer(0)]],
    device float       *out  [[buffer(1)]],
    constant uint2     &dims [[buffer(2)]],  // (size, tokens)
    constant float     &eps  [[buffer(3)]],
    uint  token     [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint size = dims.x;
    if (token >= dims.y) { return; }
    device const float *row = x + (ulong)token * size;
    device float *dst = out + (ulong)token * size;

    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) { partial += row[i] * row[i]; }
    partial = simd_sum(partial);

    threadgroup float shared[32];
    const uint simdCount = (laneCount + 31u) / 32u;
    if (lane % 32u == 0u) { shared[lane / 32u] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint s = 0; s < simdCount; ++s) { total += shared[s]; }
    const float inverse = rsqrt(total / float(size) + eps);
    for (uint i = lane; i < size; i += laneCount) { dst[i] = row[i] * inverse; }
}

/// Gemma's two-dimensional rotary, in place over a `[tokens][heads * headDim]` buffer.
///
/// **The head is split into two halves and each is turned on its own axis**, x for the first 36
/// channels and y for the second. Inside a half the pairing is the ordinary `rotate_half`, so
/// channel `i` turns against `i + 18` **within that half**, never across the boundary.
///
/// That last point is the trap. A single 72-wide `rotate_half`, which is what Qwen's tower does
/// and what this file's other rotary kernel implements, pairs channel `i` with `i + 36`: one
/// channel from the x half against one from the y half. It preserves norms, produces finite
/// output, and mixes the two spatial axes into each other.
///
/// The base is 100, not 10000, because the positions are patch coordinates.
kernel void gemma_vision_rotary(
    device float       *x      [[buffer(0)]],  // [tokens][heads * headDim]
    device const float *angles [[buffer(1)]],  // [tokens][2 * pairs], x angles then y angles
    constant uint4     &dims   [[buffer(2)]],  // (tokens, heads, headDim, perAxis)
    uint2 grid [[thread_position_in_grid]])    // (slot, token)
{
    const uint tokens = dims.x;
    const uint heads = dims.y;
    const uint headDim = dims.z;
    const uint perAxis = dims.w;               // 36
    const uint pairs = perAxis / 2u;           // 18

    const uint token = grid.y;
    const uint slot = grid.x;                  // head * (2 * pairs) + axis * pairs + component
    if (token >= tokens || slot >= heads * 2u * pairs) { return; }

    const uint head = slot / (2u * pairs);
    const uint within = slot % (2u * pairs);
    const uint axis = within / pairs;
    const uint component = within % pairs;

    const float angle = angles[(ulong)token * 2u * pairs + axis * pairs + component];
    const float c = cos(angle), s = sin(angle);

    const ulong base = (ulong)token * heads * headDim + (ulong)head * headDim
        + (ulong)axis * perAxis;
    const float low = x[base + component];
    const float high = x[base + pairs + component];
    x[base + component] = low * c - high * s;
    x[base + pairs + component] = high * c + low * s;
}

/// The 3x3 average pool, the `sqrt(hiddenSize)` scale and the learned standardization, in one.
///
/// Fused because they are three elementwise passes over the same data with nothing between them,
/// and because the scale is what makes the intermediate large: the reference keeps this stretch
/// in float32 for exactly that reason and notes the magnitude can leave float16's range.
///
/// One threadgroup a pooled token. Patches arrive in reading order, so the nine that make a token
/// are three runs of three, `poolingKernel` rows apart.
kernel void gemma_vision_pool(
    device const float  *patches [[buffer(0)]],  // [gridHeight * gridWidth][hidden]
    device const ushort *bias    [[buffer(1)]],  // BF16 [hidden]
    device const ushort *scale   [[buffer(2)]],  // BF16 [hidden]
    device float        *out     [[buffer(3)]],  // [pooledHeight * pooledWidth][hidden]
    constant uint4      &dims    [[buffer(4)]],  // (hidden, gridWidth, kernel, pooledWidth)
    uint  token [[threadgroup_position_in_grid]],
    uint  lane  [[thread_position_in_threadgroup]],
    uint  width [[threads_per_threadgroup]])
{
    const uint hidden = dims.x;
    const uint gridWidth = dims.y;
    const uint k = dims.z;
    const uint pooledWidth = dims.w;

    const uint blockY = token / pooledWidth;
    const uint blockX = token % pooledWidth;
    const float divisor = 1.0f / float(k * k);
    const float root = sqrt(float(hidden));

    for (uint i = lane; i < hidden; i += width) {
        float sum = 0.0f;
        for (uint dy = 0; dy < k; ++dy) {
            const uint row = blockY * k + dy;
            for (uint dx = 0; dx < k; ++dx) {
                const uint patch = row * gridWidth + blockX * k + dx;
                sum += patches[(ulong)patch * hidden + i];
            }
        }
        const float pooled = sum * divisor * root;
        out[(ulong)token * hidden + i] =
            (pooled - bf16_to_float_v(bias[i])) * bf16_to_float_v(scale[i]);
    }
}

/// Gathers separately projected q, k and v into the packed layout `vision_attention` reads.
///
/// Qwen's tower produces one fused `qkv` projection, so the attention kernel was written to read
/// `[tokens][3 * hidden]`. Gemma projects the three separately. Rather than write a second
/// attention kernel for identical arithmetic, the three are gathered here.
kernel void gemma_vision_pack_qkv(
    device const float *query [[buffer(0)]],
    device const float *key   [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device float       *qkv   [[buffer(3)]],
    constant uint2     &dims  [[buffer(4)]],  // (hidden, tokens)
    uint2 grid [[thread_position_in_grid]])   // (component, token)
{
    const uint hidden = dims.x;
    const uint token = grid.y;
    const uint i = grid.x;
    if (token >= dims.y || i >= hidden) { return; }
    const ulong source = (ulong)token * hidden + i;
    const ulong destination = (ulong)token * 3u * hidden + i;
    qkv[destination] = query[source];
    qkv[destination + hidden] = key[source];
    qkv[destination + 2u * hidden] = value[source];
}

/// Bidirectional patch attention using simdgroup matrix instructions.
///
/// The tiled kernel above is tuned as far as its shape allows: its key tile, its queries a
/// threadgroup and its unrolling were each swept and each is already at its optimum. It still
/// runs at roughly 4 GMAC/s where the projections in the same tower reach 800, and the reason is
/// not the cross-lane reduction, which M-073 measured as free, nor the exponentials. It is that
/// every lane re-reads K and V from threadgroup memory for each key, doing about four useful
/// multiply-accumulates per key against six loads and the address arithmetic around them.
///
/// `simdgroup_float8x8` fixes exactly that: a matrix is loaded once into registers and reused
/// across an entire 8x8 tile, so one load feeds eight rows of work.
///
/// The shape: a simdgroup owns eight queries and walks the keys eight at a time, holding the
/// output as `HEAD_DIM / 8` accumulator matrices. Each step is three matrix products, `Q·Kᵀ`
/// for the scores, a diagonal rescale of the accumulator when the running maximum moves, and
/// `P·V` for the values.
///
/// The softmax cannot stay in the matrices: it needs a row maximum and a row sum, which are
/// reductions across a matrix's columns and have no simdgroup-matrix form. So the score tile is
/// written to threadgroup memory, normalized there by eight of the lanes, and read back as `P`.
/// That round trip is 64 floats a step against the 1152 multiply-accumulates it enables.
template <uint HEAD_DIM>
kernel void vision_attention_mma_t(
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
    constexpr uint kTiles = HEAD_DIM / 8u;      // 9 for a 72-wide head
    constexpr uint kKeyStep = 8u;
    constexpr uint kStaged = 32u;               // keys staged in threadgroup memory at once

    const uint patches = dims.x;
    const uint heads = dims.y;
    const uint head = group.x;
    if (head >= heads) { return; }

    const uint hidden = heads * HEAD_DIM;
    const uint queryBase = (group.y * simdCount + simd) * 8u;

    threadgroup float keyStage[kStaged * HEAD_DIM];
    threadgroup float valueStage[kStaged * HEAD_DIM];
    // One score tile and one set of softmax state a simdgroup.
    // **Sized for both halves.** The scores occupy the first 64 floats a simdgroup and the
    // diagonal scratch the next 64, and `rowDen` likewise carries the denominators then the
    // corrections. Declared at one simdgroup's worth each, the indices below run off the end
    // into the next simdgroup's state: no fault, no warning, just another group's numbers.
    threadgroup float scoreTile[8u * 64u * 2u];
    threadgroup float rowMax[8u * 8u];
    threadgroup float rowDen[8u * 8u * 2u];

    threadgroup float *scores = scoreTile + simd * 64u;
    threadgroup float *runningMax = rowMax + simd * 8u;
    threadgroup float *denominator = rowDen + simd * 8u;

    // Queries past the end still take the tour: every barrier below must be reached by the
    // whole threadgroup, and a barrier some threads skip is undefined behaviour, not a hang.
    const bool active = queryBase < patches;

    if (lane < 8u) {
        runningMax[lane] = -INFINITY;
        denominator[lane] = 0.0f;
    }

    // Q, loaded once and held for the whole pass.
    simdgroup_float8x8 queryTiles[kTiles];
    for (uint t = 0; t < kTiles; ++t) {
        if (active) {
            simdgroup_load(
                queryTiles[t],
                qkv + (ulong)queryBase * 3u * hidden + (ulong)head * HEAD_DIM + t * 8u,
                3u * hidden);
        } else {
            queryTiles[t] = simdgroup_float8x8(0.0f);
        }
    }

    simdgroup_float8x8 accumulator[kTiles];
    for (uint t = 0; t < kTiles; ++t) { accumulator[t] = simdgroup_float8x8(0.0f); }

    const uint width = simdCount * 32u;
    for (uint staged = 0; staged < patches; staged += kStaged) {
        const uint stagedCount = min(kStaged, patches - staged);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = threadId.x; i < kStaged * HEAD_DIM; i += width) {
            const uint key = i / HEAD_DIM;
            const uint component = i % HEAD_DIM;
            if (key < stagedCount) {
                const ulong base = (ulong)(staged + key) * 3u * hidden
                    + (ulong)head * HEAD_DIM + component;
                keyStage[i] = qkv[base + hidden];
                valueStage[i] = qkv[base + 2u * hidden];
            } else {
                // Padding keys score -inf below, so their contents never matter; zeroed anyway
                // so a stale tile cannot leak into a partial last step.
                keyStage[i] = 0.0f;
                valueStage[i] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint sub = 0; sub < kStaged; sub += kKeyStep) {
            if (sub >= stagedCount) { break; }

            // --- Scores: Q times K transposed, accumulated over the head's tiles ---
            simdgroup_float8x8 score = simdgroup_float8x8(0.0f);
            for (uint t = 0; t < kTiles; ++t) {
                simdgroup_float8x8 keyTile;
                // Transposed, so the product contracts over the head dimension.
                simdgroup_load(
                    keyTile, keyStage + sub * HEAD_DIM + t * 8u, HEAD_DIM, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(score, queryTiles[t], keyTile, score);
            }
            simdgroup_store(score, scores, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // --- The softmax, which the matrices cannot express ---
            float correction = 1.0f;
            if (lane < 8u) {
                const uint valid = min(kKeyStep, stagedCount - sub);
                float peak = runningMax[lane];
                for (uint k = 0; k < valid; ++k) {
                    peak = max(peak, scores[lane * 8u + k] * scale);
                }
                const float shift = isinf(runningMax[lane])
                    ? 0.0f : exp(runningMax[lane] - peak);
                float total = denominator[lane] * shift;
                for (uint k = 0; k < 8u; ++k) {
                    const float weight = k < valid
                        ? exp(scores[lane * 8u + k] * scale - peak) : 0.0f;
                    scores[lane * 8u + k] = weight;
                    total += weight;
                }
                runningMax[lane] = peak;
                denominator[lane] = total;
                rowDen[64u + simd * 8u + lane] = shift;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            correction = rowDen[64u + simd * 8u + (lane % 8u)];

            // --- Rescale the accumulator, then add this tile's contribution ---
            //
            // A per-row scale is a diagonal matrix on the left, which is the only form a
            // simdgroup matrix multiply can express. Building it costs one store.
            if (lane < 8u) {
                for (uint k = 0; k < 8u; ++k) {
                    scoreTile[512u + simd * 64u + lane * 8u + k] =
                        (k == lane) ? rowDen[64u + simd * 8u + lane] : 0.0f;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 diagonal;
            simdgroup_load(diagonal, scoreTile + 512u + simd * 64u, 8u);
            simdgroup_float8x8 weights;
            simdgroup_load(weights, scores, 8u);

            for (uint t = 0; t < kTiles; ++t) {
                simdgroup_float8x8 scaled = simdgroup_float8x8(0.0f);
                simdgroup_multiply_accumulate(scaled, diagonal, accumulator[t], scaled);
                simdgroup_float8x8 valueTile;
                simdgroup_load(valueTile, valueStage + sub * HEAD_DIM + t * 8u, HEAD_DIM);
                simdgroup_multiply_accumulate(scaled, weights, valueTile, scaled);
                accumulator[t] = scaled;
            }
        }
    }

    if (!active) { return; }
    // One divide a row, applied the same way: a diagonal on the left.
    if (lane < 8u) {
        for (uint k = 0; k < 8u; ++k) {
            scoreTile[512u + simd * 64u + lane * 8u + k] =
                (k == lane) ? 1.0f / denominator[lane] : 0.0f;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 inverse;
    simdgroup_load(inverse, scoreTile + 512u + simd * 64u, 8u);

    for (uint t = 0; t < kTiles; ++t) {
        simdgroup_float8x8 normalized = simdgroup_float8x8(0.0f);
        simdgroup_multiply_accumulate(normalized, inverse, accumulator[t], normalized);
        const uint rows = min(8u, patches - queryBase);
        if (rows == 8u) {
            simdgroup_store(
                normalized,
                out + (ulong)queryBase * hidden + (ulong)head * HEAD_DIM + t * 8u, hidden);
        } else {
            // A partial last tile cannot be stored directly without writing past the end.
            simdgroup_store(normalized, scoreTile + 512u + simd * 64u, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (lane < rows * 8u) {
                const uint r = lane / 8u, c = lane % 8u;
                out[(ulong)(queryBase + r) * hidden + (ulong)head * HEAD_DIM + t * 8u + c] =
                    scoreTile[512u + simd * 64u + r * 8u + c];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template [[host_name("vision_attention_mma_72")]] kernel void
vision_attention_mma_t<72>(
    device const float *, device float *, constant uint4 &, constant float &,
    uint2, uint2, uint, uint, uint);

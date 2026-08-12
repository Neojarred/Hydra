// Concatenated after common.metal, which provides bf16_to_float.
//
// Operations on unquantized weights. In GPT-OSS everything that is not an expert stays in
// BF16: attention projections, routers, norms, sinks, LM head. Those tensors are 2.27 GiB
// for the 20B and 2.88 GiB for the 120B, and are re-read on every token, so they weigh
// more in the bandwidth budget than the experts do.

/// y = W·x + bias, with W in BF16 laid out as [rows, cols].
///
/// Weights are read as `uint4`, i.e. eight BF16 values per load. Alignment is guaranteed by
/// the format: every resident tensor starts at an offset that is a multiple of 256 bytes,
/// and a row's stride is `cols × 2` bytes, a multiple of 16 for all of GPT-OSS's
/// dimensions.
kernel void bf16_gemv(
    device const ushort *w        [[buffer(0)]],
    device const ushort *bias     [[buffer(1)]],
    device const float  *x        [[buffer(2)]],
    device float        *y        [[buffer(3)]],
    constant uint2      &dims     [[buffer(4)]],  // (rows, cols)
    constant uint       &hasBias  [[buffer(5)]],
    uint  row       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) { return; }

    const uint groups = cols / 8u;  // eight BF16 values per uint4
    device const uint4  *wv = reinterpret_cast<device const uint4 *>(w + row * cols);
    device const float4 *xv = reinterpret_cast<device const float4 *>(x);

    float4 acc = 0.0f;
    for (uint g = lane; g < groups; g += laneCount) {
        const uint4 packed = wv[g];
        // A uint carries two BF16s: the low 16 bits first, little-endian.
        acc += float4(bf16_to_float(ushort(packed.x & 0xFFFFu)),
                      bf16_to_float(ushort(packed.x >> 16)),
                      bf16_to_float(ushort(packed.y & 0xFFFFu)),
                      bf16_to_float(ushort(packed.y >> 16))) * xv[g * 2u];
        acc += float4(bf16_to_float(ushort(packed.z & 0xFFFFu)),
                      bf16_to_float(ushort(packed.z >> 16)),
                      bf16_to_float(ushort(packed.w & 0xFFFFu)),
                      bf16_to_float(ushort(packed.w >> 16))) * xv[g * 2u + 1u];
    }

    float partial = acc.x + acc.y + acc.z + acc.w;
    threadgroup float shared[32];
    partial = simd_sum(partial);
    if (lane % 32u == 0) { shared[lane / 32u] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint i = 0; i < simdCount; ++i) { total += shared[i]; }
        if (hasBias != 0u) { total += bf16_to_float(bias[row]); }
        y[row] = total;
    }
}

/// Writes K and V into the FP16 cache at a given position.
///
/// Circular indexing covers the sliding-window layers: the 128-token window fits in a
/// bounded ring whatever the context length, and nothing is ever copied.
kernel void kv_cache_write(
    device const float *k        [[buffer(0)]],  // [kvHeads * headDim]
    device const float *v        [[buffer(1)]],
    device half        *kCache   [[buffer(2)]],
    device half        *vCache   [[buffer(3)]],
    constant uint2     &dims     [[buffer(4)]],  // (kvHeads, headDim)
    constant uint2     &slot     [[buffer(5)]],  // (position, ringSize)
    uint gid [[thread_position_in_grid]])
{
    const uint kvHeads = dims.x;
    const uint headDim = dims.y;
    if (gid >= kvHeads * headDim) { return; }

    const uint position = slot.x;
    const uint ringSize = slot.y;
    const uint physical = ringSize > 0u ? (position % ringSize) : position;
    const uint destination = physical * kvHeads * headDim + gid;

    kCache[destination] = half(k[gid]);
    vCache[destination] = half(v[gid]);
}

/// out += in, the transformer residual.
kernel void add_inplace(
    device float       *out  [[buffer(0)]],
    device const float *in   [[buffer(1)]],
    constant uint      &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) { out[gid] += in[gid]; }
}

kernel void copy_buffer(
    device float       *out  [[buffer(0)]],
    device const float *in   [[buffer(1)]],
    constant uint      &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) { out[gid] = in[gid]; }
}

/// Accumulates an expert's weighted contribution: `out += weight · contribution`.
///
/// The router weight is read from a GPU buffer rather than passed as a constant: expert
/// identifiers are produced by the GPU, and bringing the weights back to the CPU to feed
/// them in again would cost a synchronization round trip, 45 µs, more than the compute
/// itself.
kernel void accumulate_expert(
    device float       *out         [[buffer(0)]],
    device const float *contribution [[buffer(1)]],
    device const float *weights     [[buffer(2)]],
    constant uint2     &dims        [[buffer(3)]],  // (size, weightIndex)
    uint gid [[thread_position_in_grid]])
{
    const uint size = dims.x;
    if (gid >= size) { return; }
    out[gid] += weights[dims.y] * contribution[gid];
}

/// Writes an expert's weighted contribution into **its own slot**.
///
/// Distinct from `accumulate_expert`: since each expert has its own slot, the order in which
/// they are computed no longer affects the result. That is what allows processing the
/// already-resident experts first while the missing ones are read, without making the output
/// depend on the state of the cache.
kernel void write_expert_scaled(
    device float       *out          [[buffer(0)]],
    device const float *contribution [[buffer(1)]],
    device const float *weights      [[buffer(2)]],
    constant uint2     &dims         [[buffer(3)]],  // (size, weightIndex)
    uint gid [[thread_position_in_grid]])
{
    if (gid >= dims.x) { return; }
    out[gid] = weights[dims.y] * contribution[gid];
}

/// Sums the slots in the fixed order of the slots.
///
/// Since floating-point addition is not associative, this order, and it alone, determines
/// the result. It depends on no data, so the output stays identical from one run to the
/// next.
kernel void sum_expert_slices(
    device float       *out    [[buffer(0)]],
    device const float *slices [[buffer(1)]],
    constant uint2     &dims   [[buffer(2)]],  // (size, count)
    uint gid [[thread_position_in_grid]])
{
    const uint size = dims.x;
    if (gid >= size) { return; }
    float total = 0.0f;
    for (uint i = 0; i < dims.y; ++i) { total += slices[i * size + gid]; }
    out[gid] = total;
}

kernel void fill_zero(
    device float  *out  [[buffer(0)]],
    constant uint &size [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) { out[gid] = 0.0f; }
}

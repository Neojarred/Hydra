// This file is concatenated after common.metal, which provides bf16_to_float.

// MXFP4 decoding on the GPU.
//
// The layout is the one verified against the real checkpoints (docs/01-DECISIONS.md, D-011):
// a block covers 32 consecutive values of the last dimension, stored in 16 bytes — two E2M1
// FP4 values per uchar, low nibble first — plus one E8M0 scale byte.
//
// This file is compiled at runtime by MTLDevice.makeLibrary(source:). Dimensions arrive as
// `constant` for now; moving them to function_constants, which makes them compile-time
// constants, comes with the model contract (phase 3).

constant float FP4_TABLE[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// The E8M0 scale is a biased exponent: value = fp4 * 2^(byte - 127).
inline float mxfp4_scale(uchar s) {
    return exp2(float(int(s) - 127));
}

/// Decodes `blockCount` blocks. One thread per block.
/// Serves as a correctness reference against the CPU decoder, not as a production path.
kernel void mxfp4_dequantize(
    device const uchar *blocks   [[buffer(0)]],
    device const uchar *scales   [[buffer(1)]],
    device float       *out      [[buffer(2)]],
    constant uint      &blockCount [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= blockCount) { return; }

    const float factor = mxfp4_scale(scales[gid]);
    const uint src = gid * 16u;
    const uint dst = gid * 32u;

    for (uint i = 0; i < 16u; ++i) {
        const uchar packed = blocks[src + i];
        out[dst + 2u * i]      = FP4_TABLE[packed & 0x0Fu] * factor;
        out[dst + 2u * i + 1u] = FP4_TABLE[packed >> 4]    * factor;
    }
}

/// Matrix-vector product over an MXFP4 matrix: y = W·x + bias.
///
/// W is stored as [rows, cols] with blocks running along the columns, exactly as in the
/// checkpoint: `blocks` is rows × (cols/32) × 16 bytes, `scales` is rows × (cols/32) bytes.
///
/// One threadgroup per row, SIMD reduction. This split gives the GPU a great deal of
/// independent work — TurboFieldfare's most expensive lesson was that a "more elegant"
/// cooperative kernel starved the GPU of parallelism and doubled the time.
kernel void mxfp4_gemv(
    device const uchar  *blocks    [[buffer(0)]],
    device const uchar  *scales    [[buffer(1)]],
    device const ushort *bias      [[buffer(2)]],  // BF16, pas half — voir common.metal
    device const float  *x         [[buffer(3)]],
    device float        *y         [[buffer(4)]],
    constant uint2      &dims      [[buffer(5)]],  // (rows, cols)
    constant uint       &hasBias   [[buffer(6)]],
    uint  row          [[threadgroup_position_in_grid]],
    uint  lane         [[thread_position_in_threadgroup]],
    uint  laneCount    [[threads_per_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) { return; }

    const uint blocksPerRow = cols / 32u;
    const uint blockBase    = row * blocksPerRow;

    float partial = 0.0f;

    // Each lane handles whole blocks: a block's 32 values share one scale, so it is applied
    // once per block rather than once per value.
    for (uint b = lane; b < blocksPerRow; b += laneCount) {
        const uint packedBase = (blockBase + b) * 16u;
        const uint valueBase  = b * 32u;
        const float factor    = mxfp4_scale(scales[blockBase + b]);

        float acc = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            const uchar packed = blocks[packedBase + i];
            acc += FP4_TABLE[packed & 0x0Fu] * x[valueBase + 2u * i];
            acc += FP4_TABLE[packed >> 4]    * x[valueBase + 2u * i + 1u];
        }
        partial += acc * factor;
    }

    // Reduction across the threadgroup.
    threadgroup float shared[32];
    const uint simdLane = lane % 32u;
    const uint simdIndex = lane / 32u;

    partial = simd_sum(partial);
    if (simdLane == 0) { shared[simdIndex] = partial; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint i = 0; i < simdCount; ++i) { total += shared[i]; }
        if (hasBias != 0u) { total += bf16_to_float(bias[row]); }
        y[row] = total;
    }
}

/// Vectorized GEMV variant.
///
/// The reference kernel loads weights byte by byte: 16 `uchar` loads per block. Here we read
/// a block's 16 bytes as a single `uint4`, then extract the nibbles by shifting. That is
/// legitimate because the format guarantees the alignment: a blob's stride is page-aligned
/// and every sub-tensor is 256-byte aligned, so a block's address is always a multiple of 16.
///
/// TurboFieldfare documented the opposite trap — a 32-bit path that passed tests at offset
/// zero and then produced noise in real decoding, because the live offsets were only 2-byte
/// aligned. Here the alignment is a property of the format, not an assumption, and the tests
/// exercise it at realistic offsets.
kernel void mxfp4_gemv_vectorized(
    device const uchar  *blocks    [[buffer(0)]],
    device const uchar  *scales    [[buffer(1)]],
    device const ushort *bias      [[buffer(2)]],
    device const float  *x         [[buffer(3)]],
    device float        *y         [[buffer(4)]],
    constant uint2      &dims      [[buffer(5)]],
    constant uint       &hasBias   [[buffer(6)]],
    uint  row          [[threadgroup_position_in_grid]],
    uint  lane         [[thread_position_in_threadgroup]],
    uint  laneCount    [[threads_per_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) { return; }

    const uint blocksPerRow = cols / 32u;
    const uint blockBase    = row * blocksPerRow;

    float partial = 0.0f;

    for (uint b = lane; b < blocksPerRow; b += laneCount) {
        const uint4 packed = *reinterpret_cast<device const uint4 *>(
            blocks + (blockBase + b) * 16u);
        device const float4 *xv = reinterpret_cast<device const float4 *>(x + b * 32u);

        float4 acc = 0.0f;
        for (uint word = 0; word < 4u; ++word) {
            const uint bits = packed[word];
            // A 32-bit word carries 8 values, i.e. two float4s of activations.
            const float4 a0 = xv[word * 2u];
            const float4 a1 = xv[word * 2u + 1u];

            const uchar b0 = uchar(bits & 0xFFu);
            const uchar b1 = uchar((bits >> 8) & 0xFFu);
            const uchar b2 = uchar((bits >> 16) & 0xFFu);
            const uchar b3 = uchar((bits >> 24) & 0xFFu);

            acc += float4(FP4_TABLE[b0 & 0x0Fu], FP4_TABLE[b0 >> 4],
                          FP4_TABLE[b1 & 0x0Fu], FP4_TABLE[b1 >> 4]) * a0;
            acc += float4(FP4_TABLE[b2 & 0x0Fu], FP4_TABLE[b2 >> 4],
                          FP4_TABLE[b3 & 0x0Fu], FP4_TABLE[b3 >> 4]) * a1;
        }
        partial += (acc.x + acc.y + acc.z + acc.w) * mxfp4_scale(scales[blockBase + b]);
    }

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

/// One-SIMD-group-per-row variant.
///
/// The two previous kernels use a wide threadgroup (96 lanes for 90 blocks), which forces a
/// two-stage reduction: `simd_sum`, then threadgroup memory, then a barrier, then a final
/// sum by a single lane. Over 5,760 rows, that machinery costs more than the compute itself.
///
/// Here the threadgroup is exactly 32 lanes — one SIMD group. `simd_sum` suffices: no shared
/// memory, no barrier. Each lane handles about three blocks, which gives it enough work to
/// amortize the reduction.
kernel void mxfp4_gemv_simd(
    device const uchar  *blocks    [[buffer(0)]],
    device const uchar  *scales    [[buffer(1)]],
    device const ushort *bias      [[buffer(2)]],
    device const float  *x         [[buffer(3)]],
    device float        *y         [[buffer(4)]],
    constant uint2      &dims      [[buffer(5)]],
    constant uint       &hasBias   [[buffer(6)]],
    uint  row  [[threadgroup_position_in_grid]],
    uint  lane [[thread_position_in_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    if (row >= rows) { return; }

    const uint blocksPerRow = cols / 32u;
    const uint blockBase    = row * blocksPerRow;

    float partial = 0.0f;

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        const uint4 packed = *reinterpret_cast<device const uint4 *>(
            blocks + (blockBase + b) * 16u);
        device const float4 *xv = reinterpret_cast<device const float4 *>(x + b * 32u);

        float4 acc = 0.0f;
        for (uint word = 0; word < 4u; ++word) {
            const uint bits = packed[word];
            const uchar b0 = uchar(bits & 0xFFu);
            const uchar b1 = uchar((bits >> 8) & 0xFFu);
            const uchar b2 = uchar((bits >> 16) & 0xFFu);
            const uchar b3 = uchar((bits >> 24) & 0xFFu);

            acc += float4(FP4_TABLE[b0 & 0x0Fu], FP4_TABLE[b0 >> 4],
                          FP4_TABLE[b1 & 0x0Fu], FP4_TABLE[b1 >> 4]) * xv[word * 2u];
            acc += float4(FP4_TABLE[b2 & 0x0Fu], FP4_TABLE[b2 >> 4],
                          FP4_TABLE[b3 & 0x0Fu], FP4_TABLE[b3 >> 4]) * xv[word * 2u + 1u];
        }
        partial += (acc.x + acc.y + acc.z + acc.w) * mxfp4_scale(scales[blockBase + b]);
    }

    partial = simd_sum(partial);
    if (lane == 0) {
        if (hasBias != 0u) { partial += bf16_to_float(bias[row]); }
        y[row] = partial;
    }
}

/// Tiled GEMV: activations are placed in threadgroup memory once per threadgroup.
///
/// **This was meant to fix the real bottleneck.** The previous variants assign one
/// threadgroup per row. Each then re-reads the entire activation vector: for gate_up at
/// [5760 × 2880], that is 5,760 re-reads of 11.5 KiB, i.e. 66 MB of traffic — against 8.3 MB
/// for the weights themselves. The kernel spent eight times longer re-reading `x` than
/// reading the weights, which would explain it topping out at a tenth of memory bandwidth.
///
/// Here one threadgroup covers several rows: it copies `x` into threadgroup memory once,
/// then each SIMD group handles one row by drawing from it. Activation traffic is divided by
/// the number of rows per threadgroup.
kernel void mxfp4_gemv_tiled(
    device const uchar  *blocks    [[buffer(0)]],
    device const uchar  *scales    [[buffer(1)]],
    device const ushort *bias      [[buffer(2)]],
    device const float  *x         [[buffer(3)]],
    device float        *y         [[buffer(4)]],
    constant uint2      &dims      [[buffer(5)]],
    constant uint       &hasBias   [[buffer(6)]],
    threadgroup float   *xs        [[threadgroup(0)]],
    uint  group     [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;

    // Cooperative load of the activations, once for the whole threadgroup.
    for (uint i = lane; i < cols; i += laneCount) { xs[i] = x[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simdIndex   = lane / 32u;
    const uint simdLane    = lane % 32u;
    const uint rowsPerGroup = laneCount / 32u;
    const uint row = group * rowsPerGroup + simdIndex;
    if (row >= rows) { return; }

    const uint blocksPerRow = cols / 32u;
    const uint blockBase    = row * blocksPerRow;

    float partial = 0.0f;
    for (uint b = simdLane; b < blocksPerRow; b += 32u) {
        const uint4 packed = *reinterpret_cast<device const uint4 *>(
            blocks + (blockBase + b) * 16u);
        threadgroup const float4 *xv =
            reinterpret_cast<threadgroup const float4 *>(xs + b * 32u);

        float4 acc = 0.0f;
        for (uint word = 0; word < 4u; ++word) {
            const uint bits = packed[word];
            const uchar b0 = uchar(bits & 0xFFu);
            const uchar b1 = uchar((bits >> 8) & 0xFFu);
            const uchar b2 = uchar((bits >> 16) & 0xFFu);
            const uchar b3 = uchar((bits >> 24) & 0xFFu);

            acc += float4(FP4_TABLE[b0 & 0x0Fu], FP4_TABLE[b0 >> 4],
                          FP4_TABLE[b1 & 0x0Fu], FP4_TABLE[b1 >> 4]) * xv[word * 2u];
            acc += float4(FP4_TABLE[b2 & 0x0Fu], FP4_TABLE[b2 >> 4],
                          FP4_TABLE[b3 & 0x0Fu], FP4_TABLE[b3 >> 4]) * xv[word * 2u + 1u];
        }
        partial += (acc.x + acc.y + acc.z + acc.w) * mxfp4_scale(scales[blockBase + b]);
    }

    partial = simd_sum(partial);
    if (simdLane == 0) {
        if (hasBias != 0u) { partial += bf16_to_float(bias[row]); }
        y[row] = partial;
    }
}

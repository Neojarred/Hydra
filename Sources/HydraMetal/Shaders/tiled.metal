// Concatenated after common.metal and mxfp4.metal.
//
// The prefill's tiled GEMMs, with **both operands placed in threadgroup memory**.
//
// The traffic model that governs these kernels:
//
//     bytes read = cols × rows × tokens × (2/BT + 4/BR)
//
// where BR is the number of rows and BT the number of tokens one threadgroup handles.
// An earlier version took BR=4 and BT=16, a factor of 1.125 — that is, it re-read the
// activations once per output row, which cancelled the whole benefit of chunked processing.
// For `q_proj` at [4096 × 2880] over 68 tokens, that meant 902 MB of traffic for 23.6 MB of
// weights.
//
// Here BR=64 and BT=32 give a factor of 0.125, **nine times less**. The price is threadgroup
// memory: the weight and activation tiles are copied into it at each step along the
// contraction dimension, and each thread reuses those values for 8 outputs.

// Two quantities govern this kernel, and they have to be tuned together.
//
// **Traffic** is `cols × rows × tokens × (2/TILE_TOKENS + 4/TILE_ROWS)`. Enlarging either
// tile reduces it directly.
//
// **The compute / threadgroup-memory ratio** is `(THREAD_ROWS + THREAD_TOKENS)` reads for
// `THREAD_ROWS × THREAD_TOKENS` multiplications. At 4 × 2 that is 6 reads for 8 operations —
// threadgroup memory becomes the bottleneck. At 8 × 4 it is 12 for 32, twice as good.
//
// The ceiling is the available threadgroup memory, 32 KiB:
//   (TILE_ROWS + TILE_TOKENS) × (TILE_COLS + 1) × 4 bytes ≤ 32768
#define TILE_ROWS    128u
#define TILE_TOKENS   64u
#define TILE_COLS     32u
#define THREAD_ROWS    8u   // outputs per thread, row dimension
#define THREAD_TOKENS  4u   // outputs per thread, token dimension
// 16 × 16 threads, each with 8 × 4 outputs, covers exactly 128 × 64.
#define TILE_THREADS 256u

/// The stride of a row in threadgroup memory: the tile width **plus one**.
///
/// Without that offset, a row is 64 floats and threadgroup memory has 32 banks of 4 bytes:
/// every lane of a SIMD group reading the same column of different rows lands on the same
/// bank, and the accesses serialize. Measured, the kernel topped out at 7 GB/s against
/// 99 GB/s for the GEMV — fourteen times below bandwidth. One float of padding per row makes
/// the stride coprime with 32 and removes the conflict.
#define TILE_PITCH (TILE_COLS + 1u)

/// Tiled dense BF16 projection: `y[t] = W·x[t] + bias`.
kernel void bf16_gemm_tiled(
    device const ushort *w      [[buffer(0)]],
    device const ushort *bias   [[buffer(1)]],
    device const float  *x      [[buffer(2)]],  // [tokens][cols]
    device float        *y      [[buffer(3)]],  // [tokens][rows]
    constant uint4      &dims   [[buffer(4)]],  // (rows, cols, tokens, hasBias)
    uint2 group [[threadgroup_position_in_grid]],   // (bloc de lignes, bloc de jetons)
    uint2 local [[thread_position_in_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint tokens = dims.z;

    // `thread` is a reserved address-space qualifier in Metal.
    const uint tid = local.x;
    const uint rowBase = (tid / 16u) * THREAD_ROWS;
    const uint tokenBase = (tid % 16u) * THREAD_TOKENS;

    const uint firstRow = group.x * TILE_ROWS;
    const uint firstToken = group.y * TILE_TOKENS;
    if (firstRow >= rows || firstToken >= tokens) { return; }

    threadgroup float weightTile[TILE_ROWS * TILE_PITCH];
    threadgroup float inputTile[TILE_TOKENS * TILE_PITCH];

    float acc[THREAD_ROWS][THREAD_TOKENS];
    for (uint i = 0; i < THREAD_ROWS; ++i) {
        for (uint j = 0; j < THREAD_TOKENS; ++j) { acc[i][j] = 0.0f; }
    }

    for (uint colBase = 0; colBase < cols; colBase += TILE_COLS) {
        // Cooperative load: 4096 weights and 2048 activations for 256 threads.
        for (uint slot = tid; slot < TILE_ROWS * TILE_COLS; slot += TILE_THREADS) {
            const uint r = slot / TILE_COLS;
            const uint c = slot % TILE_COLS;
            const uint row = firstRow + r;
            const uint col = colBase + c;
            weightTile[r * TILE_PITCH + c] = (row < rows && col < cols)
                ? bf16_to_float(w[row * cols + col]) : 0.0f;
        }
        for (uint slot = tid; slot < TILE_TOKENS * TILE_COLS; slot += TILE_THREADS) {
            const uint t = slot / TILE_COLS;
            const uint c = slot % TILE_COLS;
            const uint token = firstToken + t;
            const uint col = colBase + c;
            inputTile[t * TILE_PITCH + c] =
                (token < tokens && col < cols) ? x[token * cols + col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_COLS; ++k) {
            float wv[THREAD_ROWS];
            for (uint i = 0; i < THREAD_ROWS; ++i) {
                wv[i] = weightTile[(rowBase + i) * TILE_PITCH + k];
            }
            for (uint j = 0; j < THREAD_TOKENS; ++j) {
                const float xv = inputTile[(tokenBase + j) * TILE_PITCH + k];
                for (uint i = 0; i < THREAD_ROWS; ++i) { acc[i][j] += wv[i] * xv; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_ROWS; ++i) {
        const uint row = firstRow + rowBase + i;
        if (row >= rows) { continue; }
        const float biasValue = dims.w != 0u ? bf16_to_float(bias[row]) : 0.0f;
        for (uint j = 0; j < THREAD_TOKENS; ++j) {
            const uint token = firstToken + tokenBase + j;
            if (token < tokens) { y[token * rows + row] = acc[i][j] + biasValue; }
        }
    }
}

/// Tiled MXFP4 expert projection, over a selection of tokens.
///
/// Same structure, except that the weight tile is **dequantized during the load**: MXFP4
/// values become floats in threadgroup memory, and decoding happens once per tile instead of
/// once per token.
kernel void mxfp4_gemm_tiled(
    device const uchar  *blocks     [[buffer(0)]],
    device const uchar  *scales     [[buffer(1)]],
    device const ushort *bias       [[buffer(2)]],
    device const float  *x          [[buffer(3)]],  // [tokens][cols]
    device float        *y          [[buffer(4)]],  // [count][rows], compacté
    device const uint   *rowIndices [[buffer(5)]],
    constant uint4      &dims       [[buffer(6)]],  // (rows, cols, count, hasBias)
    uint2 group [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]])
{
    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint count = dims.z;

    // `thread` is a reserved address-space qualifier in Metal.
    const uint tid = local.x;
    const uint rowBase = (tid / 16u) * THREAD_ROWS;
    const uint tokenBase = (tid % 16u) * THREAD_TOKENS;

    const uint firstRow = group.x * TILE_ROWS;
    const uint firstToken = group.y * TILE_TOKENS;
    if (firstRow >= rows || firstToken >= count) { return; }

    const uint blocksPerRow = cols / 32u;

    threadgroup float weightTile[TILE_ROWS * TILE_PITCH];
    threadgroup float inputTile[TILE_TOKENS * TILE_PITCH];

    float acc[THREAD_ROWS][THREAD_TOKENS];
    for (uint i = 0; i < THREAD_ROWS; ++i) {
        for (uint j = 0; j < THREAD_TOKENS; ++j) { acc[i][j] = 0.0f; }
    }

    for (uint colBase = 0; colBase < cols; colBase += TILE_COLS) {
        for (uint slot = tid; slot < TILE_ROWS * TILE_COLS; slot += TILE_THREADS) {
            const uint r = slot / TILE_COLS;
            const uint c = slot % TILE_COLS;
            const uint row = firstRow + r;
            const uint col = colBase + c;
            if (row >= rows || col >= cols) {
                weightTile[r * TILE_PITCH + c] = 0.0f;
                continue;
            }
            // An MXFP4 block covers 32 columns: locate the block, then the nibble.
            const uint blockIndex = row * blocksPerRow + col / 32u;
            const uint inBlock = col % 32u;
            const uchar packed = blocks[blockIndex * 16u + inBlock / 2u];
            const uchar nibble = (inBlock % 2u == 0u) ? (packed & 0x0Fu) : (packed >> 4);
            weightTile[r * TILE_PITCH + c] =
                FP4_TABLE[nibble] * mxfp4_scale(scales[blockIndex]);
        }
        for (uint slot = tid; slot < TILE_TOKENS * TILE_COLS; slot += TILE_THREADS) {
            const uint t = slot / TILE_COLS;
            const uint c = slot % TILE_COLS;
            const uint token = firstToken + t;
            const uint col = colBase + c;
            inputTile[t * TILE_PITCH + c] = (token < count && col < cols)
                ? x[rowIndices[token] * cols + col] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_COLS; ++k) {
            float wv[THREAD_ROWS];
            for (uint i = 0; i < THREAD_ROWS; ++i) {
                wv[i] = weightTile[(rowBase + i) * TILE_PITCH + k];
            }
            for (uint j = 0; j < THREAD_TOKENS; ++j) {
                const float xv = inputTile[(tokenBase + j) * TILE_PITCH + k];
                for (uint i = 0; i < THREAD_ROWS; ++i) { acc[i][j] += wv[i] * xv; }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_ROWS; ++i) {
        const uint row = firstRow + rowBase + i;
        if (row >= rows) { continue; }
        const float biasValue = dims.w != 0u ? bf16_to_float(bias[row]) : 0.0f;
        for (uint j = 0; j < THREAD_TOKENS; ++j) {
            const uint token = firstToken + tokenBase + j;
            if (token < count) { y[token * rows + row] = acc[i][j] + biasValue; }
        }
    }
}

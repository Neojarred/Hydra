// Concatenated after common.metal and mxfp4.metal.
//
// The chunked-prefill kernels. The principle fits in one sentence: **read the weights once
// for several tokens** instead of once per token.
//
// Measured on a 78-token prompt of the 20B: processing tokens one at a time re-reads
// 92.9 GiB of dense weights; in chunks, 1.2 GiB. The computation is identical, only the
// scheduling changes — no value is altered, so there is no quality risk.
//
// The memory counterpart is 8.7 MiB of activations for a 128-token chunk, against the
// 1.18 GiB of the expert cache. The number of expert slots does not move: we iterate expert
// by expert, gathering the tokens routed to each.

/// Tokens a threadgroup handles at once. Each lane keeps one accumulator per token of the
/// tile: raising this value reduces weight re-reads but consumes registers. 16 re-reads the
/// weights 5 times for a 78-token prompt, against 78.
#define TOKEN_TILE 16u

/// Rows handled by a single threadgroup.
///
/// **This is the parameter that matters, and it is not the one you would expect.** With one
/// row per threadgroup, every threadgroup re-reads the entire activation vector: for
/// `q_proj` at [4096 × 2880] over 68 tokens, that is 3.8 GB of traffic on `x` against 23.6 MB
/// of weights. The GEMM was then pointless — it divided weight re-reads by 5 without
/// touching the dominant term.
///
/// Grouping several rows shares the activations between them. Each row is given to one SIMD
/// group, which also removes every threadgroup barrier: the reduction fits entirely in
/// `simd_sum`.
#define ROW_TILE 4u

/// Dense BF16 projection over a chunk of tokens: `y[t] = W·x[t] + bias`.
///
/// One threadgroup per (row, token tile). The row's weights are read once and serve all
/// `TOKEN_TILE` tokens of the tile.
kernel void bf16_gemm(
    device const ushort *w      [[buffer(0)]],
    device const ushort *bias   [[buffer(1)]],
    device const float  *x      [[buffer(2)]],  // [tokens][cols]
    device float        *y      [[buffer(3)]],  // [tokens][rows]
    constant uint4      &dims   [[buffer(4)]],  // (rows, cols, tokens, hasBias)
    // Metal requires every position attribute to have the same dimensionality: mixing uint2
    // and uint in one signature does not compile.
    uint2 group      [[threadgroup_position_in_grid]],  // (row block, token tile)
    uint2 laneVector [[thread_position_in_threadgroup]])
{
    const uint lane = laneVector.x;
    const uint rowInTile = lane / 32u;   // un groupe SIMD par ligne
    const uint simdLane  = lane % 32u;

    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint tokens = dims.z;
    const uint row = group.x * ROW_TILE + rowInTile;
    const uint tileStart = group.y * TOKEN_TILE;
    if (row >= rows || tileStart >= tokens) { return; }

    const uint tileCount = min(TOKEN_TILE, tokens - tileStart);
    const uint groups = cols / 8u;  // eight BF16 per uint4

    float acc[TOKEN_TILE];
    for (uint t = 0; t < TOKEN_TILE; ++t) { acc[t] = 0.0f; }

    device const uint4 *wv = reinterpret_cast<device const uint4 *>(w + row * cols);

    for (uint g = simdLane; g < groups; g += 32u) {
        const uint4 packed = wv[g];
        const float4 w0 = float4(
            bf16_to_float(ushort(packed.x & 0xFFFFu)), bf16_to_float(ushort(packed.x >> 16)),
            bf16_to_float(ushort(packed.y & 0xFFFFu)), bf16_to_float(ushort(packed.y >> 16)));
        const float4 w1 = float4(
            bf16_to_float(ushort(packed.z & 0xFFFFu)), bf16_to_float(ushort(packed.z >> 16)),
            bf16_to_float(ushort(packed.w & 0xFFFFu)), bf16_to_float(ushort(packed.w >> 16)));

        for (uint t = 0; t < tileCount; ++t) {
            device const float4 *xv =
                reinterpret_cast<device const float4 *>(x + (tileStart + t) * cols);
            acc[t] += dot(w0, xv[g * 2u]) + dot(w1, xv[g * 2u + 1u]);
        }
    }

    // The row fits in one SIMD group: `simd_sum` suffices, no barrier.
    const float biasValue = dims.w != 0u ? bf16_to_float(bias[row]) : 0.0f;
    for (uint t = 0; t < tileCount; ++t) {
        const float total = simd_sum(acc[t]);
        if (simdLane == 0) { y[(tileStart + t) * rows + row] = total + biasValue; }
    }
}

/// MXFP4 expert projection over a **selection** of tokens.
///
/// This is the kernel that lets chunked prefill cost no memory: rather than loading all of
/// the chunk's experts at once, we iterate expert by expert. For each, `rowIndices` names the
/// tokens the router assigned to it, and the output is compacted — only one expert slot is
/// needed at any moment, exactly as in decoding.
kernel void mxfp4_gemm_gathered(
    device const uchar  *blocks     [[buffer(0)]],
    device const uchar  *scales     [[buffer(1)]],
    device const ushort *bias       [[buffer(2)]],
    device const float  *x          [[buffer(3)]],  // [tokens][cols]
    device float        *y          [[buffer(4)]],  // [count][rows], compacted
    device const uint   *rowIndices [[buffer(5)]],  // [count], indices dans x
    constant uint4      &dims       [[buffer(6)]],  // (rows, cols, count, hasBias)
    uint2 group      [[threadgroup_position_in_grid]],
    uint2 laneVector [[thread_position_in_threadgroup]])
{
    const uint lane = laneVector.x;
    const uint rowInTile = lane / 32u;
    const uint simdLane  = lane % 32u;

    const uint rows = dims.x;
    const uint cols = dims.y;
    const uint count = dims.z;
    const uint row = group.x * ROW_TILE + rowInTile;
    const uint tileStart = group.y * TOKEN_TILE;
    if (row >= rows || tileStart >= count) { return; }

    const uint tileCount = min(TOKEN_TILE, count - tileStart);
    const uint blocksPerRow = cols / 32u;
    const uint blockBase = row * blocksPerRow;

    float acc[TOKEN_TILE];
    for (uint t = 0; t < TOKEN_TILE; ++t) { acc[t] = 0.0f; }

    for (uint b = simdLane; b < blocksPerRow; b += 32u) {
        const uint4 packed = *reinterpret_cast<device const uint4 *>(
            blocks + (blockBase + b) * 16u);
        const float factor = mxfp4_scale(scales[blockBase + b]);

        // The weights are decoded into eight float4s kept in registers.
        //
        // An earlier version stored them in a `float weights[32]` reused across all tokens of
        // the tile: 32 more registers, which spilled to memory and cost far more than the
        // re-decoding they avoided.
        float4 wv[8];
        for (uint word = 0; word < 4u; ++word) {
            const uint bits = packed[word];
            const uchar b0 = uchar(bits & 0xFFu);
            const uchar b1 = uchar((bits >> 8) & 0xFFu);
            const uchar b2 = uchar((bits >> 16) & 0xFFu);
            const uchar b3 = uchar((bits >> 24) & 0xFFu);
            wv[word * 2u] = float4(
                FP4_TABLE[b0 & 0x0Fu], FP4_TABLE[b0 >> 4],
                FP4_TABLE[b1 & 0x0Fu], FP4_TABLE[b1 >> 4]) * factor;
            wv[word * 2u + 1u] = float4(
                FP4_TABLE[b2 & 0x0Fu], FP4_TABLE[b2 >> 4],
                FP4_TABLE[b3 & 0x0Fu], FP4_TABLE[b3 >> 4]) * factor;
        }

        for (uint t = 0; t < tileCount; ++t) {
            device const float4 *xr = reinterpret_cast<device const float4 *>(
                x + rowIndices[tileStart + t] * cols + b * 32u);
            float sum = 0.0f;
            for (uint i = 0; i < 8u; ++i) { sum += dot(wv[i], xr[i]); }
            acc[t] += sum;
        }
    }

    const float biasValue = dims.w != 0u ? bf16_to_float(bias[row]) : 0.0f;
    for (uint t = 0; t < tileCount; ++t) {
        const float total = simd_sum(acc[t]);
        if (simdLane == 0) { y[(tileStart + t) * rows + row] = total + biasValue; }
    }
}

/// Adds an expert's weighted contribution to the tokens it served.
kernel void scatter_expert(
    device float        *mixture    [[buffer(0)]],  // [tokens][size]
    device const float  *outputs    [[buffer(1)]],  // [count][size], compacted
    device const uint   *rowIndices [[buffer(2)]],
    device const float  *weights    [[buffer(3)]],  // [count], poids du routeur
    constant uint2      &dims       [[buffer(4)]],  // (size, count)
    uint gid [[thread_position_in_grid]])
{
    const uint size = dims.x;
    const uint count = dims.y;
    if (gid >= size * count) { return; }
    const uint slot = gid / size;
    const uint component = gid % size;
    mixture[rowIndices[slot] * size + component] += weights[slot] * outputs[gid];
}

/// RMSNorm over a chunk of tokens. One threadgroup per token.
kernel void rms_norm_batch(
    device const float  *x     [[buffer(0)]],
    device const ushort *scale [[buffer(1)]],
    device float        *out   [[buffer(2)]],
    constant uint2      &dims  [[buffer(3)]],  // (size, tokens)
    constant float      &eps   [[buffer(4)]],
    uint  token     [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint size = dims.x;
    if (token >= dims.y) { return; }
    const uint base = token * size;

    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) {
        const float v = x[base + i];
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
        out[base + i] = x[base + i] * inverse * bf16_to_float(scale[i]);
    }
}

/// RoPE over a chunk: each token has **its own** position, hence its own tables.
kernel void rope_apply_batch(
    device float       *x        [[buffer(0)]],  // [tokens][heads * headDim]
    device const float *cosTable [[buffer(1)]],  // [tokens][headDim/2]
    device const float *sinTable [[buffer(2)]],
    constant uint4     &dims     [[buffer(3)]],  // (heads, headDim, tokens, _)
    uint gid [[thread_position_in_grid]])
{
    const uint heads = dims.x;
    const uint headDim = dims.y;
    const uint tokens = dims.z;
    const uint halfDim = headDim / 2u;
    if (gid >= tokens * heads * halfDim) { return; }

    const uint perToken = heads * halfDim;
    const uint token = gid / perToken;
    const uint rest = gid % perToken;
    const uint head = rest / halfDim;
    const uint index = rest % halfDim;

    const uint base = token * heads * headDim + head * headDim;
    const float c = cosTable[token * halfDim + index];
    const float s = sinTable[token * halfDim + index];
    const float x1 = x[base + index];
    const float x2 = x[base + halfDim + index];

    x[base + index]           = x1 * c - x2 * s;
    x[base + halfDim + index] = x2 * c + x1 * s;
}

/// Writes a whole chunk's K and V into the cache.
kernel void kv_cache_write_batch(
    device const float *k      [[buffer(0)]],  // [tokens][kvHeads * headDim]
    device const float *v      [[buffer(1)]],
    device half        *kCache [[buffer(2)]],
    device half        *vCache [[buffer(3)]],
    constant uint4     &dims   [[buffer(4)]],  // (kvHeads, headDim, tokens, ringSize)
    constant uint      &first  [[buffer(5)]],  // position absolue du premier jeton
    uint gid [[thread_position_in_grid]])
{
    const uint kvHeads = dims.x;
    const uint headDim = dims.y;
    const uint tokens = dims.z;
    const uint ringSize = dims.w;
    const uint perToken = kvHeads * headDim;
    if (gid >= tokens * perToken) { return; }

    const uint token = gid / perToken;
    const uint offset = gid % perToken;
    const uint position = first + token;
    const uint physical = ringSize > 0u ? (position % ringSize) : position;

    kCache[physical * perToken + offset] = half(k[gid]);
    vCache[physical * perToken + offset] = half(v[gid]);
}

/// SwiGLU over a chunk. Same rules as the single-token version: even/odd indices,
/// asymmetric clamping, +1 on the linear branch.
kernel void swiglu_batch(
    device const float *x      [[buffer(0)]],  // [tokens][2 * size]
    device float       *out    [[buffer(1)]],  // [tokens][size]
    constant uint2     &dims   [[buffer(2)]],  // (size, tokens)
    constant float2    &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint size = dims.x;
    if (gid >= size * dims.y) { return; }
    const uint token = gid / size;
    const uint index = gid % size;

    const float alpha = params.x;
    const float limit = params.y;
    device const float *row = x + token * 2u * size;

    const float gate   = min(row[2u * index], limit);
    const float linear = clamp(row[2u * index + 1u], -limit, limit);
    out[gid] = gate * (1.0f / (1.0f + exp(-alpha * gate))) * (linear + 1.0f);
}

/// Causal attention over a chunk. One threadgroup per (token, query head).
///
/// Each query sees its own past: the chunk's tokens already written into the cache **and**
/// everything preceding the chunk. The mask is therefore not explicit, it follows from the
/// loop bound.
kernel void attention_prefill(
    device const float  *q       [[buffer(0)]],  // [tokens][qHeads * headDim]
    device const half   *kCache  [[buffer(1)]],
    device const half   *vCache  [[buffer(2)]],
    device const ushort *sinks   [[buffer(3)]],
    device float        *out     [[buffer(4)]],  // [tokens][qHeads * headDim]
    constant uint4      &dims    [[buffer(5)]],  // (qHeads, kvHeads, headDim, tokens)
    constant uint4      &window  [[buffer(6)]],  // (ringSize, firstPosition, slidingWindow, _)
    constant float      &smScale [[buffer(7)]],
    uint2 group      [[threadgroup_position_in_grid]],  // (head, token)
    uint2 laneVector [[thread_position_in_threadgroup]])
{
    const uint lane = laneVector.x;
    const uint qHeads = dims.x;
    const uint kvHeads = dims.y;
    const uint headDim = dims.z;
    const uint tokens = dims.w;
    const uint head = group.x;
    const uint token = group.y;
    if (head >= qHeads || token >= tokens) { return; }

    const uint ringSize = window.x;
    const uint firstPosition = window.y;
    const uint slidingWindow = window.z;

    const uint position = firstPosition + token;
    const uint start = (slidingWindow > 0u && position + 1u > slidingWindow)
        ? (position + 1u - slidingWindow) : 0u;

    const uint qMult = qHeads / kvHeads;
    const uint kvHead = head / qMult;
    const uint perEntry = kvHeads * headDim;
    device const float *query = q + token * qHeads * headDim + head * headDim;

    float runningMax = bf16_to_float(sinks[head]);
    float denominator = 1.0f;
    float accumulator[8];
    for (uint i = 0; i < 8u; ++i) { accumulator[i] = 0.0f; }
    const uint slice = (headDim + 31u) / 32u;

    for (uint key = start; key <= position; ++key) {
        const uint physical = ringSize > 0u ? (key % ringSize) : key;
        const uint base = physical * perEntry + kvHead * headDim;

        float partial = 0.0f;
        for (uint i = lane; i < headDim; i += 32u) {
            partial += query[i] * float(kCache[base + i]);
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
                accumulator[s] = accumulator[s] * correction + weight * float(vCache[base + i]);
            }
        }
    }

    const float inverse = 1.0f / denominator;
    device float *destination = out + token * qHeads * headDim + head * headDim;
    for (uint s = 0; s < slice; ++s) {
        const uint i = lane + s * 32u;
        if (i < headDim) { destination[i] = accumulator[s] * inverse; }
    }
}

/// Router top-k and softmax for a chunk. One threadgroup per token.
kernel void router_topk_batch(
    device const float *logits  [[buffer(0)]],  // [tokens][expertCount]
    device uint        *indices [[buffer(1)]],  // [tokens][topK]
    device float       *weights [[buffer(2)]],
    constant uint4     &dims    [[buffer(3)]],  // (expertCount, topK, tokens, _)
    uint token [[threadgroup_position_in_grid]],
    uint lane  [[thread_position_in_threadgroup]])
{
    if (lane != 0 || token >= dims.z) { return; }
    const uint expertCount = dims.x;
    const uint topK = min(dims.y, 8u);
    device const float *row = logits + token * expertCount;

    float chosenValues[8];
    uint chosenIndices[8];

    for (uint k = 0; k < topK; ++k) {
        float best = -INFINITY;
        uint bestIndex = 0;
        for (uint e = 0; e < expertCount; ++e) {
            bool taken = false;
            for (uint j = 0; j < k; ++j) { taken = taken || (chosenIndices[j] == e); }
            if (taken) { continue; }
            if (row[e] > best) { best = row[e]; bestIndex = e; }
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
        indices[token * topK + k] = chosenIndices[k];
        weights[token * topK + k] = chosenValues[k] / total;
    }
}

kernel void add_inplace_batch(
    device float       *out  [[buffer(0)]],
    device const float *in   [[buffer(1)]],
    constant uint      &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < size) { out[gid] += in[gid]; }
}

// Concatenated after common.metal.
//
// Gemma 4's operators that GPT-OSS has no equivalent for. Deliberately few: most of the
// runtime already serves both models unchanged, and the ones that do not are here rather than
// as branches inside the shared kernels (D-023).
//
// What is **not** here, and why:
//
//   - the expert GEMV. Gemma's experts are plain BF16 matrices, so `bf16_gemv` already is it.
//     MXFP4's decoder is what made GPT-OSS need its own.
//   - RoPE. Full-attention layers rotate only a quarter of the head dimension, and that is
//     expressed as **zero inverse frequencies** in the table, `cos = 1, sin = 0`, the
//     identity. `rope_apply` needs no knowledge of it.
//   - attention. `attention_decode` seeds its online softmax on the sink, and a sink of −1e30
//     contributes `exp(−1e30 − max) = 0` while leaving the denominator at one. Gemma has no
//     sinks, and that is exactly what passing an unreachable one produces. The equivalence is
//     asserted numerically in the tests rather than assumed here.

/// RMSNorm **without a learned scale**.
///
/// `v_norm` and the router's normalization are built `with_scale: false` and therefore have no
/// tensor in the checkpoint, nothing in the weight index reveals that the operation exists at
/// all. Reusing `rms_norm` with a buffer of ones would work and would also mean allocating and
/// reading a vector of ones on every token, per layer.
kernel void rms_norm_unscaled(
    device const float *x    [[buffer(0)]],
    device float       *out  [[buffer(1)]],
    constant uint      &size [[buffer(2)]],
    constant float     &eps  [[buffer(3)]],
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
        // `pow(m, -0.5)` rather than `rsqrt`, matching the reference: the source chose it to
        // keep Torch and JAX agreeing, and the oracle is compared bit for bit.
        inverse = pow(total / float(size) + eps, -0.5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < size; i += laneCount) { out[i] = x[i] * inverse; }
}

/// `gelu_pytorch_tanh(gate) · up`, over two **separate** vectors.
///
/// Not `swiglu`: GPT-OSS clamps its gate from above, clamps the linear branch on both sides,
/// adds one to it, and uses `sigmoid(1.702·x)`. Gemma does none of that. The activation curves
/// happen to agree to about 0.2 %, which is why 1.702 was chosen, so substituting one for the
/// other would pass a casual eye and fail on the clamps.
///
/// The inputs are separate rather than interleaved because Gemma's `gate_up_proj` is stored as
/// two halves, where GPT-OSS interleaves `[gate0, up0, gate1, up1, …]`.
///
/// **The argument to `tanh` must be clamped.** Metal compiles with fast math by default, where
/// `tanh` is evaluated through `exp(2·inner)`. That overflows to infinity once `2·inner`
/// passes ~88, and `inf / inf` is NaN, so a gate value above about 10.1 poisons the element,
/// then the layer, then every logit. Real Gemma weights produce gates of 11 at layer 26 of 30;
/// the 64-wide test configuration never exceeded 2, which is why every test passed and the
/// first real run returned 262,144 NaNs.
///
/// Clamping costs nothing in accuracy: `tanh(15)` differs from 1 by 1e-13, far below what
/// `float` can represent, so the saturated region is exact where it matters.
kernel void gelu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float       *out  [[buffer(2)]],
    constant uint      &size [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) { return; }
    const float x = gate[gid];
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[gid] = 0.5f * x * (1.0f + tanh(clamp(inner, -15.0f, 15.0f))) * up[gid];
}

/// `c · tanh(logits / c)`, applied to the logits before sampling.
///
/// Bounds the distribution's dynamic range. Omitting it leaves the extremes untouched, which
/// changes which token wins wherever the model is confident, the opposite of harmless.
kernel void logit_softcap(
    device float  *logits [[buffer(0)]],
    constant uint &size   [[buffer(1)]],
    constant float &cap   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) { return; }
    // Clamped for the same reason as `gelu_mul`: the softcap exists precisely to tame
    // logits that ran away, so it is the one place guaranteed to be handed large arguments.
    logits[gid] = cap * tanh(clamp(logits[gid] / cap, -15.0f, 15.0f));
}

/// `x · w · factor`, with `w` in BF16, the router's learned scale and its `hidden^-0.5`.
///
/// A separate kernel rather than folding the factor into the projection: the scale is applied
/// to the **normalized** vector before the projection sees it, and the two cannot be swapped
/// because normalization is not linear in the scale.
kernel void scale_by_bf16(
    device float        *x      [[buffer(0)]],
    device const ushort *scale  [[buffer(1)]],
    constant uint       &size   [[buffer(2)]],
    constant float      &factor [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) { return; }
    x[gid] = x[gid] * bf16_to_float(scale[gid]) * factor;
}

/// `x · s`, where `s` is a **single** BF16 value broadcast over the whole vector.
///
/// Distinct from `scale_by_bf16`, which reads one weight per element. `layer_scalar` is stored
/// as a tensor of one and multiplies the entire hidden state; passing it to the per-element
/// kernel with `size = 1` would scale only the first component and silently leave the rest,
/// a layer that is almost right, which is the worst kind.
kernel void scale_by_bf16_scalar(
    device float        *x     [[buffer(0)]],
    device const ushort *scale [[buffer(1)]],
    constant uint       &size  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) { return; }
    x[gid] = x[gid] * bf16_to_float(scale[0]);
}

/// Gemma's router selection, which is **not** GPT-OSS's.
///
/// The chain, from `Gemma4TextRouter.forward`:
///
///   1. softmax over **all** experts, GPT-OSS softmaxes over the top-k only, and the two give
///      different weights for identical logits;
///   2. take the top-k of those probabilities;
///   3. **renormalize** them to sum to one;
///   4. multiply each by its expert's learned scale, after which they no longer sum to one.
///
/// Step four is why a test asserting the weights sum to one would be wrong here, and step one
/// is why `router_topk` cannot simply be reused with different constants.
///
/// On a tie the smaller index wins, so decoding stays reproducible.
/// Experts the router kernel can select for one token.
///
/// Qwen3.6-35B-A3B routes to exactly eight, so this bound is met with no margin. It is a
/// register-file limit, not a modelling one: raising it costs the two arrays below.
constant constexpr uint kMaxRouterTopK = 8u;

kernel void gemma_router_topk(
    device const float  *logits         [[buffer(0)]],  // [expertCount]
    device const ushort *perExpertScale [[buffer(1)]],  // BF16, [expertCount]
    device uint         *indices        [[buffer(2)]],  // [topK]
    device float        *weights        [[buffer(3)]],  // [topK]
    constant uint2      &dims           [[buffer(4)]],  // (expertCount, topK)
    uint lane [[thread_position_in_threadgroup]])
{
    if (lane != 0) { return; }
    const uint expertCount = dims.x;
    // Clamped, not chosen: the accumulators below are fixed at `kMaxRouterTopK`, and a model
    // wanting more would have routed on the first eight and produced plausible wrong output
    // with nothing to say so. The caller is required to refuse instead, so reaching this with
    // a larger `dims.y` is a bug on that side; the clamp stays only so the kernel cannot read
    // past its own arrays.
    const uint topK = min(dims.y, kMaxRouterTopK);

    // Softmax over every expert, in float, before any selection happens.
    float peak = -INFINITY;
    for (uint e = 0; e < expertCount; ++e) { peak = max(peak, logits[e]); }
    float total = 0.0f;
    for (uint e = 0; e < expertCount; ++e) { total += exp(logits[e] - peak); }

    float chosen[kMaxRouterTopK];
    uint  chosenIndices[kMaxRouterTopK];
    for (uint k = 0; k < topK; ++k) {
        float best = -INFINITY;
        uint bestIndex = 0;
        for (uint e = 0; e < expertCount; ++e) {
            bool taken = false;
            for (uint j = 0; j < k; ++j) { taken = taken || (chosenIndices[j] == e); }
            if (taken) { continue; }
            if (logits[e] > best) { best = logits[e]; bestIndex = e; }
        }
        chosen[k] = exp(best - peak) / total;
        chosenIndices[k] = bestIndex;
    }

    // Renormalize the selected probabilities, then apply the per-expert scale.
    float selected = 0.0f;
    for (uint k = 0; k < topK; ++k) { selected += chosen[k]; }
    for (uint k = 0; k < topK; ++k) {
        indices[k] = chosenIndices[k];
        weights[k] = chosen[k] / selected * bf16_to_float(perExpertScale[chosenIndices[k]]);
    }
}

/// `y = W · x` for an MLX affine-quantized matrix.
///
/// Three inputs make one matrix: values packed into 32-bit words, a BF16 scale per group, and a
/// BF16 **bias** per group. A weight is `q · scale + bias`.
///
/// Two conventions are baked in here and neither is guessable, both were settled by decoding a
/// real tensor from the published checkpoint against Google's QAT weights (D-024):
///
/// - the **first** value in a word occupies the least significant bits;
/// - the bias is applied. Dropping it shifts every weight by a per-group constant, which costs
///   the model quality and raises nothing.
///
/// A group never straddles a word, `group_size` is 64 and a word holds 8 values at 4 bits or 4
/// at 8, so the scale and bias are fetched once per word rather than once per value.
/// `y = W · x` for an MLX affine-quantized matrix, **one simdgroup to a row**.
///
/// The shape is the point, and it is what the previous version got wrong.
///
/// That one gave each *threadgroup* a single output row: 256 threads over 352 words, so about
/// eleven values a lane, followed by a `simd_sum`, a `threadgroup_barrier` and a walk over
/// shared memory, a full reduction to produce one scalar. The arithmetic was small beside the
/// synchronization, and because the threadgroup count equals the row count however the work is
/// grouped, batching eight experts into one dispatch changed nothing (M-034). That null result
/// is what identified this.
///
/// Here a simdgroup owns a row and a threadgroup holds eight. Each lane covers `cols / 32`
/// words, 88 values at 4 bits rather than 11, the reduction is one `simd_sum` with no barrier
/// and no shared memory, and there are eight times fewer threadgroups.
///
/// **The bias is hoisted out of the inner loop.** `Σ (q·scale + bias)·x` factors into
/// `scale·Σ(q·x) + bias·Σx`: one multiply-add per weight instead of two, and the scale and bias
/// read once per group rather than once per weight. Distributivity, not an approximation.
/// `y = W · x` for an MLX affine matrix, specialized on the bit width.
///
/// `bits` used to arrive in `dims`, at runtime, which cost more than it looked like: it made
/// `perWord` a dynamic loop bound and every shift amount a computed value, so the innermost
/// loop, the one that runs once per weight in the model, could not be unrolled and the
/// shifts could not be folded. The kernel is bound by values unpacked rather than bytes moved
/// (M-039), so that loop is the whole cost.
///
/// Instantiated per width below rather than made a function constant, so that the pipeline
/// cache stays keyed by name and needs no notion of specialization.
template <uint BITS>
[[kernel]] void mlx_affine_gemv_t(
    device const uint   *words   [[buffer(0)]],
    device const ushort *scales  [[buffer(1)]],
    device const ushort *biases  [[buffer(2)]],
    device const float  *x       [[buffer(3)]],
    device float        *y       [[buffer(4)]],
    constant uint4      &dims    [[buffer(5)]],  // (rows, cols, bits, groupSize)
    uint  group     [[threadgroup_position_in_grid]],
    uint  simd      [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  simdCount [[simdgroups_per_threadgroup]])
{
    const uint rows = dims.x, cols = dims.y, groupSize = dims.w;  // dims.z is BITS
    const uint row = group * simdCount + simd;
    if (row >= rows) { return; }

    constexpr uint perWord  = 32u / BITS;
    const uint wordsPerRow  = cols / perWord;
    const uint groupsPerRow = cols / groupSize;
    constexpr uint mask     = (1u << BITS) - 1u;

    device const uint   *w = words  + (ulong)row * wordsPerRow;
    device const ushort *s = scales + (ulong)row * groupsPerRow;
    device const ushort *b = biases + (ulong)row * groupsPerRow;

    device const uint4 *wv = reinterpret_cast<device const uint4 *>(w);
    const uint chunks = wordsPerRow / 4u;
    const uint valuesPerChunk = 4u * perWord;

    float acc = 0.0f;
    for (uint c = lane; c < chunks; c += 32u) {
        const uint4 packed = wv[c];
        const uint base = c * valuesPerChunk;
        // A chunk never straddles a group: 32 values at 4 bits, 16 at 8, against a group of 64.
        const uint g = base / groupSize;
        const float scale = bf16_to_float(s[g]);
        const float bias  = bf16_to_float(b[g]);
        float dotQX = 0.0f;
        float sumX  = 0.0f;
        for (uint j = 0; j < 4u; ++j) {
            const uint word = packed[j];
            const uint wbase = base + j * perWord;
            for (uint slot = 0; slot < perWord; ++slot) {
                const float q  = (float)((word >> (slot * BITS)) & mask);
                const float xv = x[wbase + slot];
                dotQX = fma(q, xv, dotQX);
                sumX += xv;
            }
        }
        acc = fma(scale, dotQX, fma(bias, sumX, acc));
    }

    // The tail, for rows whose word count is not a multiple of four. Real Gemma never takes it
    //, 2816 and 704 both divide, but the tiny test configuration does, and dropping the
    // remainder would pass every large-shape test.
    for (uint i = chunks * 4u + lane; i < wordsPerRow; i += 32u) {
        const uint packed = w[i];
        const uint base = i * perWord;
        const uint g = base / groupSize;
        const float scale = bf16_to_float(s[g]);
        const float bias  = bf16_to_float(b[g]);
        float dotQX = 0.0f;
        float sumX  = 0.0f;
        for (uint slot = 0; slot < perWord; ++slot) {
            const float q  = (float)((packed >> (slot * BITS)) & mask);
            const float xv = x[base + slot];
            dotQX = fma(q, xv, dotQX);
            sumX += xv;
        }
        acc = fma(scale, dotQX, fma(bias, sumX, acc));
    }

    const float total = simd_sum(acc);
    if (lane == 0) { y[row] = total; }
}

/// Per-chunk sums of the activations, `[chunks][paddedTokens]`.
///
/// The affine form is `y = scale · Σ(q·x) + bias · Σx`, so the inner loop carried two
/// instructions a value a token: one `fma` for the dot product and one `add` for `Σx`. But
/// `Σx` does not depend on the row, it is the same for all 4096 of them, and the batched
/// kernel was recomputing it in every one, which is half of the only loop that matters.
///
/// Computed once per input, in the chunk's column order, so the sum is bit-identical to the
/// one the GEMV accumulates inline and the batched result stays bit-identical to the
/// per-token loop.
///
/// Laid out `[chunk][token]` rather than `[token][chunk]` because the projection reads one
/// chunk across a tile of consecutive tokens, which is then a vector load.
/// Not specialized on the bit width, unlike the projections: the width only sets a loop bound
/// here, and this kernel runs once per input rather than once per weight.
kernel void chunk_sums(
    device const float *x      [[buffer(0)]],  // [cols][paddedTokens], transposed
    device float       *sums   [[buffer(1)]],  // [chunks][paddedTokens]
    constant uint3     &dims   [[buffer(2)]],  // (chunks, paddedTokens, valuesPerChunk)
    uint2 gid [[thread_position_in_grid]])
{
    const uint chunks = dims.x, padded = dims.y, valuesPerChunk = dims.z;
    const uint chunk = gid.x, token = gid.y;
    if (chunk >= chunks || token >= padded) { return; }

    const uint base = chunk * valuesPerChunk;
    // Column order, matching the GEMV's `for j { for slot { sumX += xv } }`.
    float total = 0.0f;
    for (uint i = 0; i < valuesPerChunk; ++i) {
        total += x[(ulong)(base + i) * padded + token];
    }
    sums[(ulong)chunk * padded + token] = total;
}

/// Rows a simdgroup carries together. Must equal `ForwardEncoder.rowBlock`.
///
/// Measured, at a layer's seven projections over a 128-token chunk. What matters is the
/// *area* `RB × TB`, which is the register budget, every pair whose product is 32 lands in
/// the same place, and both larger and smaller products are worse:
///
///     RB × TB    2×4   2×8  2×16   4×4   4×8  4×16   8×4   8×8  8×16
///     ms          82    42    28    47    26    49    28    59    78
///
/// Below the budget the traffic is not being amortized; above it, the block spills.
constant constexpr uint kRowBlock = 4u;

/// Tokens the batched projection carries together. Must equal `ForwardEncoder.batchTile`.
///
/// Measured, at a layer's seven projections over a 128-token chunk, the tile sets how many
/// times the weight row is read, and the curve is not monotonic:
///
///     tile   4     8    16    32    64
///     ms   144    74    57    81   149
///
/// Below sixteen the weight passes dominate; above it the per-token accumulators do, though
/// Metal still reports the full 1024 threads a threadgroup, so it is not a reported spill.
constant constexpr uint kBatchTile = 8u;

/// Transposes a chunk's activations from `[tokens][cols]` to `[cols][paddedTokens]`.
///
/// The batched projection reads the same column of eight consecutive tokens on every unpacked
/// weight. Row-major, those eight floats are `cols` apart, eight cache lines, eight loads,
/// for one weight value, which made the first batched kernel three times slower than the
/// per-token loop it replaced. Transposed they are contiguous and the tile is two vector
/// loads.
///
/// `paddedTokens` rounds up to the tile so the kernel needs no bounds test in its innermost
/// loop. The padding is zeroed, and zero contributes nothing to either accumulator.
kernel void transpose_activations(
    device const float *src        [[buffer(0)]],  // [tokens][cols]
    device float       *dst        [[buffer(1)]],  // [cols][paddedTokens]
    constant uint3     &dims       [[buffer(2)]],  // (tokens, cols, paddedTokens)
    uint2 gid [[thread_position_in_grid]])
{
    const uint tokens = dims.x, cols = dims.y, padded = dims.z;
    const uint col = gid.x, token = gid.y;
    if (col >= cols || token >= padded) { return; }
    dst[(ulong)col * padded + token] = token < tokens ? src[(ulong)token * cols + col] : 0.0f;
}

/// `Y = W · X` for an MLX affine matrix, blocked in both rows and tokens.
///
/// Prefill's whole cost. `Gemma4PrefillRunner` batches the *expert* reads but runs the
/// attention and dense projections through the GEMV once per prompt token, so the 1.15 GiB of
/// resident weights is re-read for every token of the prompt.
///
/// The traffic that decides this kernel is not the weights. For `q_proj` over a 128-token
/// chunk:
///
///     weights      (tokens / TB) × 6.2 MiB      =  50 MiB at TB = 16
///     activations  rows × cols × tokens × 4 / RB =  5.9 GB at RB = 1
///
/// **The activations are read 120× more than the weights.** A lane loads `TB` floats of `x`
/// per weight value and does `TB` fused multiply-adds against them, four bytes a flop, which
/// no cache sustains. Blocking the rows fixes it: one `x` tile serves `RB` rows, so that
/// traffic divides by `RB` while the weight traffic divides by `TB`.
///
/// Both cost registers, and their product is the budget. The tile pair is chosen by
/// measurement, not by principle.
///
/// **Each token's arithmetic is left exactly as the GEMV does it**, same chunk order, same
/// `fma` structure, same `simd_sum`, so the result is bit-identical to the per-token loop and
/// the test asserts equality rather than a tolerance. That is also why the scale is not folded
/// into the quantized value, which would save a register file but change the rounding.
template <uint BITS>
[[kernel]] void mlx_affine_gemm_t(
    device const uint   *words   [[buffer(0)]],
    device const ushort *scales  [[buffer(1)]],
    device const ushort *biases  [[buffer(2)]],
    device const float  *x       [[buffer(3)]],  // [cols][paddedTokens], transposed
    device float        *y       [[buffer(4)]],  // [tokens][rows]
    constant uint4      &dims    [[buffer(5)]],  // (rows, cols, tokens, groupSize)
    constant uint       &padded  [[buffer(6)]],  // tokens rounded up to the tile
    device const float  *sums    [[buffer(7)]],  // [chunks][paddedTokens], from chunk_sums
    uint  group     [[threadgroup_position_in_grid]],
    uint  simd      [[simdgroup_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  simdCount [[simdgroups_per_threadgroup]])
{
    const uint rows = dims.x, cols = dims.y, tokens = dims.z, groupSize = dims.w;

    constexpr uint TB = kBatchTile;
    constexpr uint RB = kRowBlock;
    constexpr uint VEC = TB / 4u;

    const uint row0 = (group * simdCount + simd) * RB;
    if (row0 >= rows) { return; }
    const uint rowSpan = min(RB, rows - row0);

    constexpr uint perWord  = 32u / BITS;
    constexpr uint mask     = (1u << BITS) - 1u;
    const uint wordsPerRow  = cols / perWord;
    const uint groupsPerRow = cols / groupSize;
    const uint chunks = wordsPerRow / 4u;
    const uint valuesPerChunk = 4u * perWord;

    for (uint t0 = 0; t0 < tokens; t0 += TB) {
        const uint span = min(TB, tokens - t0);

        float acc[RB][TB];
        for (uint r = 0; r < RB; ++r) {
            for (uint i = 0; i < TB; ++i) { acc[r][i] = 0.0f; }
        }

        for (uint c = lane; c < chunks; c += 32u) {
            const uint base = c * valuesPerChunk;
            // A chunk never straddles a group: 32 values at 4 bits, 16 at 8, against 64.
            const uint g = base / groupSize;

            float dotQX[RB][TB];
            for (uint r = 0; r < RB; ++r) {
                for (uint i = 0; i < TB; ++i) { dotQX[r][i] = 0.0f; }
            }

            for (uint j = 0; j < 4u; ++j) {
                uint packed[RB];
                for (uint r = 0; r < RB; ++r) {
                    const ulong at = (ulong)(row0 + r) * wordsPerRow + c * 4u + j;
                    packed[r] = r < rowSpan ? words[at] : 0u;
                }
                for (uint slot = 0; slot < perWord; ++slot) {
                    const uint column = base + j * perWord + slot;
                    // Loaded once for the whole row block, this is the reuse.
                    device const float4 *xt = reinterpret_cast<device const float4 *>(
                        x + (ulong)column * padded + t0);
                    float4 xv[VEC];
                    for (uint v = 0; v < VEC; ++v) { xv[v] = xt[v]; }

                    for (uint r = 0; r < RB; ++r) {
                        const float q = (float)((packed[r] >> (slot * BITS)) & mask);
                        for (uint v = 0; v < VEC; ++v) {
                            dotQX[r][v * 4 + 0] = fma(q, xv[v].x, dotQX[r][v * 4 + 0]);
                            dotQX[r][v * 4 + 1] = fma(q, xv[v].y, dotQX[r][v * 4 + 1]);
                            dotQX[r][v * 4 + 2] = fma(q, xv[v].z, dotQX[r][v * 4 + 2]);
                            dotQX[r][v * 4 + 3] = fma(q, xv[v].w, dotQX[r][v * 4 + 3]);
                        }
                    }
                }
            }

            // `Σx` for this chunk, read across the token tile rather than recomputed per row.
            device const float4 *cs =
                reinterpret_cast<device const float4 *>(sums + (ulong)c * padded + t0);
            float4 sx[VEC];
            for (uint v = 0; v < VEC; ++v) { sx[v] = cs[v]; }

            for (uint r = 0; r < RB; ++r) {
                const float scale = bf16_to_float(scales[(ulong)(row0 + r) * groupsPerRow + g]);
                const float bias  = bf16_to_float(biases[(ulong)(row0 + r) * groupsPerRow + g]);
                for (uint v = 0; v < VEC; ++v) {
                    acc[r][v * 4 + 0] =
                        fma(scale, dotQX[r][v * 4 + 0], fma(bias, sx[v].x, acc[r][v * 4 + 0]));
                    acc[r][v * 4 + 1] =
                        fma(scale, dotQX[r][v * 4 + 1], fma(bias, sx[v].y, acc[r][v * 4 + 1]));
                    acc[r][v * 4 + 2] =
                        fma(scale, dotQX[r][v * 4 + 2], fma(bias, sx[v].z, acc[r][v * 4 + 2]));
                    acc[r][v * 4 + 3] =
                        fma(scale, dotQX[r][v * 4 + 3], fma(bias, sx[v].w, acc[r][v * 4 + 3]));
                }
            }
        }

        // The tail, for rows whose word count is not a multiple of four. Real Gemma never
        // takes it; the tiny test configuration does. It keeps its inline `Σx`, since
        // `chunk_sums` covers only whole chunks.
        for (uint idx = chunks * 4u + lane; idx < wordsPerRow; idx += 32u) {
            const uint tailBase = idx * perWord;
            const uint g = tailBase / groupSize;
            for (uint r = 0; r < rowSpan; ++r) {
                const uint packed = words[(ulong)(row0 + r) * wordsPerRow + idx];
                const float scale = bf16_to_float(scales[(ulong)(row0 + r) * groupsPerRow + g]);
                const float bias  = bf16_to_float(biases[(ulong)(row0 + r) * groupsPerRow + g]);
                for (uint i = 0; i < span; ++i) {
                    float dotQX = 0.0f, sumX = 0.0f;
                    for (uint slot = 0; slot < perWord; ++slot) {
                        const float q = (float)((packed >> (slot * BITS)) & mask);
                        const float xv = x[(ulong)(tailBase + slot) * padded + t0 + i];
                        dotQX = fma(q, xv, dotQX);
                        sumX += xv;
                    }
                    acc[r][i] = fma(scale, dotQX, fma(bias, sumX, acc[r][i]));
                }
            }
        }

        for (uint r = 0; r < RB; ++r) {
            for (uint i = 0; i < TB; ++i) {
                const float total = simd_sum(acc[r][i]);
                if (lane == 0u && i < span && r < rowSpan) {
                    y[(ulong)(t0 + i) * rows + row0 + r] = total;
                }
            }
        }
    }
}

template [[host_name("mlx_affine_gemm_4")]] kernel void
mlx_affine_gemm_t<4>(
    device const uint *, device const ushort *, device const ushort *,
    device const float *, device float *, constant uint4 &, constant uint &,
    device const float *, uint, uint, uint, uint);

template [[host_name("mlx_affine_gemm_8")]] kernel void
mlx_affine_gemm_t<8>(
    device const uint *, device const ushort *, device const ushort *,
    device const float *, device float *, constant uint4 &, constant uint &,
    device const float *, uint, uint, uint, uint);

template [[host_name("mlx_affine_gemv_4")]] kernel void
mlx_affine_gemv_t<4>(
    device const uint *, device const ushort *, device const ushort *,
    device const float *, device float *, constant uint4 &,
    uint, uint, uint, uint);

template [[host_name("mlx_affine_gemv_8")]] kernel void
mlx_affine_gemv_t<8>(
    device const uint *, device const ushort *, device const ushort *,
    device const float *, device float *, constant uint4 &,
    uint, uint, uint, uint);


/// `hidden += rmsNorm(x, w)`, and `residual = hidden`, in one dispatch.
///
/// Three kernels became one. Gemma closes its attention with a post-norm, a residual add and a
/// copy of the result for the two feed-forward branches to share, three dispatches over 2,816
/// floats, about 11 KB each, which is nothing in bytes and a full launch-and-drain in latency.
///
/// **Latency, not bandwidth, is what this buys.** `cb1` spends 1.97 ms a layer moving 38.8 MB;
/// at the 95 GB/s the machine gives (M-037) that traffic is 0.41 ms, so roughly 62 µs of every
/// dispatch is the dependency chain rather than the work. Reducing the *count* of independent
/// dispatches does nothing, batching the experts proved that, but shortening the chain
/// removes the latency outright.
kernel void fused_norm_add_copy(
    device const float  *x        [[buffer(0)]],
    device const ushort *scale    [[buffer(1)]],  // BF16
    device float        *hidden   [[buffer(2)]],
    device float        *residual [[buffer(3)]],
    constant uint       &size     [[buffer(4)]],
    constant float      &eps      [[buffer(5)]],
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
        const float value = hidden[i] + x[i] * inverse * bf16_to_float(scale[i]);
        hidden[i] = value;
        residual[i] = value;
    }
}

/// `out = rmsNorm(x) · scale · factor`, with no learned weight on the norm.
///
/// The router's input: an unscaled normalization followed by a per-channel BF16 scale and a
/// constant. Two dispatches over 2,816 floats, fused for the same reason as above.
kernel void fused_unscaled_norm_scale(
    device const float  *x      [[buffer(0)]],
    device const ushort *scale  [[buffer(1)]],  // BF16, per channel
    device float        *out    [[buffer(2)]],
    constant uint       &size   [[buffer(3)]],
    constant float      &eps    [[buffer(4)]],
    constant float      &factor [[buffer(5)]],
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
        out[i] = x[i] * inverse * bf16_to_float(scale[i]) * factor;
    }
}

/// RMS-normalizes **every head at once**, one threadgroup to a head.
///
/// The largest single source of chain latency in the layer. Q, K and V are each normalized per
/// head, RMS normalization is not separable, so a head's slice must be normalized alone
/// (D-022), and that was written as a Swift loop issuing one dispatch a head: sixteen for the
/// queries, eight for the keys, eight for the values. **Thirty-two dispatches a layer, 960 a
/// token**, each over 256 floats.
///
/// Each is a kilobyte of work behind a full launch and drain. Measured, a dispatch removed from
/// the chain is worth about 28 µs (M-038), so this loop alone was costing on the order of 27 ms
/// a token, a third of all GPU time, to move almost no bytes.
///
/// The heads are independent, which is exactly what a grid is for.
///
/// `scale` is optional: `v_norm` is built `with_scale=False` and has no tensor in the
/// checkpoint at all, so a zero flag means "normalize only".
kernel void rms_norm_heads(
    device float        *x        [[buffer(0)]],
    device const ushort *scale    [[buffer(1)]],  // BF16, one head's worth, shared by all heads
    constant uint2      &dims     [[buffer(2)]],  // (headDim, hasScale)
    constant float      &eps      [[buffer(3)]],
    uint  head      [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint headDim = dims.x;
    const bool hasScale = dims.y != 0u;
    device float *v = x + (ulong)head * headDim;

    float partial = 0.0f;
    for (uint i = lane; i < headDim; i += laneCount) {
        const float value = v[i];
        partial += value * value;
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
        inverse = rsqrt(total / float(headDim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = lane; i < headDim; i += laneCount) {
        v[i] = hasScale ? v[i] * inverse * bf16_to_float(scale[i]) : v[i] * inverse;
    }
}

// MARK: - Batched forms, for prefill
//
// Prefill runs a chunk of tokens through the same layer, and ran every one of these once per
// token: about 334,000 dispatches for a 557-token prompt, which is 13.7 s of the 21.8 s that
// phase measures. Each kernel below is the per-token one with a token index on the grid rather
// than in a Swift loop.
//
// The arithmetic is deliberately untouched, same thread count, same reduction order, same
// expressions, because prefill's contract is that it agrees with token-by-token decoding
// *exactly*, and a second implementation that merely came close is the failure this project
// keeps meeting.

/// `rms_norm` for a chunk: one threadgroup a token.
kernel void rms_norm_batched(
    device const float  *x       [[buffer(0)]],  // [tokens][size]
    device const ushort *scale   [[buffer(1)]],  // BF16, [size], shared by every token
    device float        *out     [[buffer(2)]],  // [tokens][size]
    constant uint       &size    [[buffer(3)]],
    constant float      &eps     [[buffer(4)]],
    uint token     [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    device const float *xi = x + (ulong)token * size;
    device float *oi = out + (ulong)token * size;

    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) {
        const float v = xi[i];
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
        oi[i] = xi[i] * inverse * bf16_to_float(scale[i]);
    }
}

/// `fused_norm_add_copy` for a chunk: one threadgroup a token.
kernel void fused_norm_add_copy_batched(
    device const float  *input    [[buffer(0)]],  // [tokens][size]
    device const ushort *scale    [[buffer(1)]],
    device float        *hidden   [[buffer(2)]],  // [tokens][size], read and written
    device float        *residual [[buffer(3)]],  // [tokens][size], written
    constant uint       &size     [[buffer(4)]],
    constant float      &eps      [[buffer(5)]],
    uint token     [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    device const float *xi = input + (ulong)token * size;
    device float *hi = hidden + (ulong)token * size;
    device float *ri = residual + (ulong)token * size;

    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) {
        const float v = xi[i];
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
        const float value = hi[i] + xi[i] * inverse * bf16_to_float(scale[i]);
        hi[i] = value;
        ri[i] = value;
    }
}

/// `fused_unscaled_norm_scale` for a chunk: one threadgroup a token.
kernel void fused_unscaled_norm_scale_batched(
    device const float  *input   [[buffer(0)]],  // [tokens][size]
    device const ushort *scale   [[buffer(1)]],  // BF16, [size]
    device float        *out     [[buffer(2)]],  // [tokens][size]
    constant uint       &size    [[buffer(3)]],
    constant float      &eps     [[buffer(4)]],
    constant float      &factor  [[buffer(5)]],
    uint token     [[threadgroup_position_in_grid]],
    uint lane      [[thread_position_in_threadgroup]],
    uint laneCount [[threads_per_threadgroup]])
{
    device const float *xi = input + (ulong)token * size;
    device float *oi = out + (ulong)token * size;

    float partial = 0.0f;
    for (uint i = lane; i < size; i += laneCount) {
        const float v = xi[i];
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
        oi[i] = xi[i] * inverse * bf16_to_float(scale[i]) * factor;
    }
}

/// `rope_apply` for a chunk, each token against its own table.
///
/// Prefill already builds one rotary table per token of the chunk; this indexes them by the
/// token on the grid instead of rebinding the buffer 128 times.
kernel void rope_apply_batched(
    device float       *vector [[buffer(0)]],  // [tokens][heads * headDim]
    device const float *cosT   [[buffer(1)]],  // [tokens][pairs]
    device const float *sinT   [[buffer(2)]],
    constant uint4     &dims   [[buffer(3)]],  // (heads, headDim, pairs, tokensStride)
    uint2 gid [[thread_position_in_grid]])
{
    const uint heads = dims.x, headDim = dims.y, pairs = dims.z, stride = dims.w;
    const uint index = gid.x, token = gid.y;
    // `half` is a reserved type in Metal: this variable cannot carry that name.
    const uint halfDim = headDim / 2u;
    if (index >= heads * halfDim) { return; }

    const uint head = index / halfDim;
    const uint i = index % halfDim;
    device float *v = vector + (ulong)token * heads * headDim + (ulong)head * headDim;
    const float c = cosT[(ulong)token * stride + i];
    const float s = sinT[(ulong)token * stride + i];
    const float a = v[i];
    const float b = v[i + halfDim];
    v[i] = a * c - b * s;
    v[i + halfDim] = b * c + a * s;
    (void)pairs;
}

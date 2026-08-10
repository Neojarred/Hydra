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
//     expressed as **zero inverse frequencies** in the table — `cos = 1, sin = 0`, the
//     identity. `rope_apply` needs no knowledge of it.
//   - attention. `attention_decode` seeds its online softmax on the sink, and a sink of −1e30
//     contributes `exp(−1e30 − max) = 0` while leaving the denominator at one. Gemma has no
//     sinks, and that is exactly what passing an unreachable one produces. The equivalence is
//     asserted numerically in the tests rather than assumed here.

/// RMSNorm **without a learned scale**.
///
/// `v_norm` and the router's normalization are built `with_scale: false` and therefore have no
/// tensor in the checkpoint — nothing in the weight index reveals that the operation exists at
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
/// happen to agree to about 0.2 % — which is why 1.702 was chosen — so substituting one for the
/// other would pass a casual eye and fail on the clamps.
///
/// The inputs are separate rather than interleaved because Gemma's `gate_up_proj` is stored as
/// two halves, where GPT-OSS interleaves `[gate0, up0, gate1, up1, …]`.
///
/// **The argument to `tanh` must be clamped.** Metal compiles with fast math by default, where
/// `tanh` is evaluated through `exp(2·inner)`. That overflows to infinity once `2·inner`
/// passes ~88, and `inf / inf` is NaN — so a gate value above about 10.1 poisons the element,
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
/// changes which token wins wherever the model is confident — the opposite of harmless.
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

/// `x · w · factor`, with `w` in BF16 — the router's learned scale and its `hidden^-0.5`.
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
/// kernel with `size = 1` would scale only the first component and silently leave the rest —
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
///   1. softmax over **all** experts — GPT-OSS softmaxes over the top-k only, and the two give
///      different weights for identical logits;
///   2. take the top-k of those probabilities;
///   3. **renormalize** them to sum to one;
///   4. multiply each by its expert's learned scale, after which they no longer sum to one.
///
/// Step four is why a test asserting the weights sum to one would be wrong here, and step one
/// is why `router_topk` cannot simply be reused with different constants.
///
/// On a tie the smaller index wins, so decoding stays reproducible.
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
    const uint topK = min(dims.y, 8u);

    // Softmax over every expert, in float, before any selection happens.
    float peak = -INFINITY;
    for (uint e = 0; e < expertCount; ++e) { peak = max(peak, logits[e]); }
    float total = 0.0f;
    for (uint e = 0; e < expertCount; ++e) { total += exp(logits[e] - peak); }

    float chosen[8];
    uint  chosenIndices[8];
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
/// Two conventions are baked in here and neither is guessable — both were settled by decoding a
/// real tensor from the published checkpoint against Google's QAT weights (D-024):
///
/// - the **first** value in a word occupies the least significant bits;
/// - the bias is applied. Dropping it shifts every weight by a per-group constant, which costs
///   the model quality and raises nothing.
///
/// A group never straddles a word — `group_size` is 64 and a word holds 8 values at 4 bits or 4
/// at 8 — so the scale and bias are fetched once per word rather than once per value.
kernel void mlx_affine_gemv(
    device const uint   *words   [[buffer(0)]],
    device const ushort *scales  [[buffer(1)]],
    device const ushort *biases  [[buffer(2)]],
    device const float  *x       [[buffer(3)]],
    device float        *y       [[buffer(4)]],
    constant uint4      &dims    [[buffer(5)]],  // (rows, cols, bits, groupSize)
    uint  row       [[threadgroup_position_in_grid]],
    uint  lane      [[thread_position_in_threadgroup]],
    uint  laneCount [[threads_per_threadgroup]])
{
    const uint rows      = dims.x;
    const uint cols      = dims.y;
    const uint bits      = dims.z;
    const uint groupSize = dims.w;
    if (row >= rows) { return; }

    const uint perWord      = 32u / bits;
    const uint wordsPerRow  = cols / perWord;
    const uint groupsPerRow = cols / groupSize;
    const uint wordsPerGroup = groupSize / perWord;
    const uint mask = (1u << bits) - 1u;

    device const uint   *w = words  + (ulong)row * wordsPerRow;
    device const ushort *s = scales + (ulong)row * groupsPerRow;
    device const ushort *b = biases + (ulong)row * groupsPerRow;

    float acc = 0.0f;
    for (uint i = lane; i < wordsPerRow; i += laneCount) {
        const uint  packed = w[i];
        const uint  group  = i / wordsPerGroup;
        const float scale  = bf16_to_float(s[group]);
        const float bias   = bf16_to_float(b[group]);
        const uint  base   = i * perWord;

        for (uint slot = 0; slot < perWord; ++slot) {
            const float q = (float)((packed >> (slot * bits)) & mask);
            acc += (q * scale + bias) * x[base + slot];
        }
    }

    threadgroup float shared[32];
    acc = simd_sum(acc);
    if (lane % 32u == 0) { shared[lane / 32u] = acc; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const uint simdCount = (laneCount + 31u) / 32u;
        float total = 0.0f;
        for (uint i = 0; i < simdCount; ++i) { total += shared[i]; }
        y[row] = total;
    }
}

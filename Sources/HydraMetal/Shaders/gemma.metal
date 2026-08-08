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
    out[gid] = 0.5f * x * (1.0f + tanh(inner)) * up[gid];
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
    logits[gid] = cap * tanh(logits[gid] / cap);
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

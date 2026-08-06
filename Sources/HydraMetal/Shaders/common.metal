#include <metal_stdlib>
using namespace metal;

// bfloat16 -> float conversion.
//
// **A major trap, and one a dedicated test guards against.** BF16 is not `half`. Metal's
// `half` is IEEE binary16 (1 sign bit, 5 exponent, 10 mantissa); bfloat16 is the top half of
// a float32 (1, 8, 7). Reading one as the other raises no error and produces no NaN — just
// plausible, wrong values: 1.0 in BF16 reads as 1.875 as a half. Since everything in GPT-OSS
// that is not an expert is BF16 — attention, norms, routers, sinks, LM head — the confusion
// would corrupt the entire model in silence.
//
// Conversion to float is exact and lossless: it is enough to move the 16 bits into the high
// position. No rounding, no special case — NaNs and infinities carry over as they are.
inline float bf16_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

/// The reverse, rounding to nearest even. Used when BF16 has to be written back.
inline ushort float_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    // Round to nearest, ties to even.
    uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return ushort((bits + rounding) >> 16);
}

/// Reads one element of a BF16 vector as a float.
inline float bf16_load(device const ushort *source, uint index) {
    return bf16_to_float(source[index]);
}

// Concatenated after common.metal.

/// Streams a large buffer, to measure real memory bandwidth.
///
/// The accumulator is written under an impossible condition: without that, the compiler
/// sees the result is unused and removes the whole loop, which yields an absurdly high
/// measured bandwidth.
kernel void bandwidth_probe(
    device const float4 *input  [[buffer(0)]],
    device float4       *sink   [[buffer(1)]],
    constant uint       &count  [[buffer(2)]],
    uint gid    [[thread_position_in_grid]],
    uint stride [[threads_per_grid]])
{
    float4 acc = 0.0f;
    for (uint i = gid; i < count; i += stride) {
        acc += input[i];
    }
    if (acc.x == 1234567.0f) { sink[gid % 256] = acc; }
}

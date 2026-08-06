// Concaténé après common.metal.

/// Lecture en flux d'un grand tampon, pour mesurer la bande passante mémoire réelle.
///
/// L'accumulateur est écrit sous une condition impossible : sans cela, le compilateur
/// constate que le résultat est inutilisé et supprime toute la boucle, ce qui donne une
/// bande passante mesurée absurdement élevée.
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

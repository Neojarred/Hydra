// Concaténé après common.metal, qui fournit bf16_to_float.
//
// Opérations sur les poids non quantifiés. Dans GPT-OSS, tout ce qui n'est pas expert
// reste en BF16 : projections d'attention, routeurs, normes, sinks, tête LM. Ces tenseurs
// représentent 2,27 Gio pour le 20B et 2,88 Gio pour le 120B, et sont relus à chaque
// token — ils pèsent donc plus lourd dans le budget de bande passante que les experts.

/// y = W·x + biais, avec W en BF16 disposée en [rows, cols].
///
/// Les poids sont lus par `uint4`, soit huit valeurs BF16 par chargement. L'alignement
/// est garanti par le format : chaque tenseur résident commence à un décalage multiple de
/// 256 octets, et le pas d'une ligne vaut `cols × 2` octets, multiple de 16 pour toutes
/// les dimensions de GPT-OSS.
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

    const uint groups = cols / 8u;  // huit valeurs BF16 par uint4
    device const uint4  *wv = reinterpret_cast<device const uint4 *>(w + row * cols);
    device const float4 *xv = reinterpret_cast<device const float4 *>(x);

    float4 acc = 0.0f;
    for (uint g = lane; g < groups; g += laneCount) {
        const uint4 packed = wv[g];
        // Un uint porte deux BF16 : les 16 bits bas d'abord, en petit-boutiste.
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

/// Écrit K et V dans le cache FP16 à une position donnée.
///
/// L'indexation circulaire couvre les couches à fenêtre glissante : la fenêtre de 128
/// tokens tient dans un anneau borné, quelle que soit la longueur du contexte, et rien
/// n'est jamais recopié.
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

/// out += in — le résidu du transformeur.
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

/// Accumule la contribution pondérée d'un expert : `out += poids · contribution`.
///
/// Le poids du routeur est lu depuis un tampon GPU plutôt que passé en constante : les
/// identifiants d'experts sont produits par le GPU, et faire redescendre les poids côté
/// CPU pour les réinjecter coûterait un aller-retour de synchronisation — 45 µs, soit
/// plus cher que le calcul lui-même.
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

/// Écrit la contribution pondérée d'un expert dans **sa propre case**.
///
/// Distinct de `accumulate_expert` : chaque expert ayant sa case, l'ordre dans lequel on
/// les calcule n'influe plus sur le résultat. C'est ce qui autorise à traiter d'abord les
/// experts déjà en mémoire pendant que les manquants se lisent, sans rendre la sortie
/// dépendante de l'état du cache.
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

/// Somme les cases dans l'ordre fixe des slots.
///
/// L'addition flottante n'étant pas associative, c'est cet ordre — et lui seul — qui
/// détermine le résultat. Il ne dépend d'aucune donnée, donc la sortie reste identique
/// d'une exécution à l'autre.
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

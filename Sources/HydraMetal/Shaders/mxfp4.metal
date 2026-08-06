// Ce fichier est concaténé après common.metal, qui fournit bf16_to_float.

// Décodage MXFP4 sur GPU.
//
// Le layout est celui vérifié sur les checkpoints réels (docs/01-DECISIONS.md, D-011) :
// un bloc couvre 32 valeurs consécutives de la dernière dimension, stockées sur 16 octets
// — deux valeurs FP4 E2M1 par uchar, nibble bas d'abord — plus un octet d'échelle E8M0.
//
// Ce fichier est compilé à l'exécution par MTLDevice.makeLibrary(source:). Les dimensions
// arrivent en `constant` pour l'instant ; leur passage en function_constant, qui les rend
// constantes de compilation, viendra avec le contrat de modèle (phase 3).

constant float FP4_TABLE[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

// L'échelle E8M0 est un exposant biaisé : valeur = fp4 * 2^(octet - 127).
inline float mxfp4_scale(uchar s) {
    return exp2(float(int(s) - 127));
}

/// Décode `blockCount` blocs. Un thread par bloc.
/// Sert de référence de correction face au décodeur CPU, pas de chemin de production.
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

/// Produit matrice-vecteur sur une matrice MXFP4 : y = W·x + biais.
///
/// W est stockée en [rows, cols] avec les blocs le long des colonnes, exactement comme
/// dans le checkpoint : `blocks` fait rows × (cols/32) × 16 octets, `scales`
/// rows × (cols/32) octets.
///
/// Un threadgroup par ligne, réduction SIMD. Ce découpage donne au GPU beaucoup de
/// travail indépendant — la leçon la plus coûteuse de TurboFieldfare était qu'un noyau
/// coopératif « plus élégant » privait le GPU de parallélisme et doublait le temps.
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

    // Chaque voie traite des blocs entiers : les 32 valeurs d'un bloc partagent une
    // échelle, autant l'appliquer une fois par bloc plutôt qu'une fois par valeur.
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

    // Réduction sur le threadgroup.
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

/// Variante vectorisée du GEMV.
///
/// Le noyau de référence charge les poids octet par octet : 16 chargements `uchar` par
/// bloc. Ici on lit les 16 octets d'un bloc en un seul `uint4`, puis on extrait les
/// nibbles par décalage. C'est légitime parce que le format garantit l'alignement : le
/// pas d'un blob est aligné sur la page et chaque sous-tenseur sur 256 octets, donc
/// l'adresse d'un bloc est toujours multiple de 16.
///
/// TurboFieldfare a documenté le piège inverse — un chemin 32 bits qui passait les tests
/// à l'offset zéro puis produisait du bruit en décodage réel, parce que les décalages
/// vivants n'étaient alignés que sur 2 octets. Ici l'alignement est une propriété du
/// format, pas une supposition, et les tests l'exercent à des décalages réels.
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
            // Un mot de 32 bits porte 8 valeurs, soit deux float4 d'activations.
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

/// Variante à un seul groupe SIMD par ligne.
///
/// Les deux noyaux précédents utilisent un threadgroup large (96 voies pour 90 blocs), ce
/// qui oblige à une réduction en deux temps : `simd_sum`, puis mémoire partagée, puis
/// barrière, puis somme finale par une seule voie. Avec 5 760 lignes, cette mécanique
/// coûte plus cher que le calcul lui-même.
///
/// Ici le threadgroup fait exactement 32 voies — un groupe SIMD. `simd_sum` suffit :
/// ni mémoire partagée, ni barrière. Chaque voie traite environ trois blocs, ce qui lui
/// donne assez de travail pour amortir la réduction.
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

/// GEMV tuilé : les activations sont mises en mémoire partagée une fois par threadgroup.
///
/// **C'est la correction du vrai goulot.** Les variantes précédentes affectent un
/// threadgroup à chaque ligne. Chacun relit alors l'intégralité du vecteur d'activations :
/// pour gate_up en [5760 × 2880], cela fait 5 760 relectures de 11,5 Kio, soit 66 Mo de
/// trafic — contre 8,3 Mo pour les poids eux-mêmes. Le noyau passait huit fois plus de
/// temps à relire `x` qu'à lire les poids, ce qui explique qu'il plafonnait à un dixième
/// de la bande passante mémoire.
///
/// Ici un threadgroup couvre plusieurs lignes : il copie `x` en mémoire partagée une
/// seule fois, puis chaque groupe SIMD traite une ligne en y puisant. Le trafic
/// d'activations est divisé par le nombre de lignes par threadgroup.
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

    // Chargement coopératif des activations, une fois pour tout le threadgroup.
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

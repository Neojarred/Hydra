// Concaténé après common.metal et mxfp4.metal.
//
// GEMM tuilés du prefill, avec **les deux opérandes mis en mémoire partagée**.
//
// Le modèle de trafic qui gouverne ces noyaux :
//
//     octets lus = cols × rows × tokens × (2/BT + 4/BR)
//
// où BR est le nombre de lignes et BT le nombre de jetons traités par un threadgroup.
// Une version antérieure prenait BR=4 et BT=16, soit un facteur 1,125 — autrement dit
// elle relisait les activations une fois par ligne de sortie, ce qui annulait tout le
// bénéfice du traitement par blocs. Pour `q_proj` en [4096 × 2880] sur 68 jetons, cela
// représentait 902 Mo de trafic pour 23,6 Mo de poids.
//
// Ici BR=64 et BT=32 donnent un facteur 0,125, soit **9 fois moins**. Le prix à payer est
// la mémoire partagée : les tuiles de poids et d'activations y sont copiées à chaque pas
// sur la dimension de contraction, et chaque fil réutilise ces valeurs pour 8 sorties.

// Deux grandeurs gouvernent ce noyau, et il faut les régler ensemble.
//
// **Le trafic** vaut `cols × rows × tokens × (2/TILE_TOKENS + 4/TILE_ROWS)`. Agrandir
// l'une ou l'autre tuile le réduit directement.
//
// **Le ratio calcul / mémoire partagée** vaut `(THREAD_ROWS + THREAD_TOKENS)` lectures
// pour `THREAD_ROWS × THREAD_TOKENS` multiplications. Avec 4 × 2, cela fait 6 lectures
// pour 8 opérations — la mémoire partagée devient le goulot. Avec 8 × 4, c'est 12 pour
// 32, soit deux fois mieux.
//
// Le plafond est la mémoire partagée disponible, 32 Kio :
//   (TILE_ROWS + TILE_TOKENS) × (TILE_COLS + 1) × 4 octets ≤ 32768
#define TILE_ROWS    128u
#define TILE_TOKENS   64u
#define TILE_COLS     32u
#define THREAD_ROWS    8u   // sorties par fil, dimension lignes
#define THREAD_TOKENS  4u   // sorties par fil, dimension jetons
// 16 × 16 fils, chacun 8 × 4 sorties, couvre exactement 128 × 64.
#define TILE_THREADS 256u

/// Pas d'une ligne en mémoire partagée : la largeur de tuile **plus un**.
///
/// Sans ce décalage, une ligne fait 64 flottants et la mémoire partagée compte 32 bancs
/// de 4 octets : toutes les voies d'un groupe SIMD qui lisent la même colonne de lignes
/// différentes tombent sur le même banc, et les accès se sérialisent. Mesuré, le noyau
/// plafonnait à 7 Go/s contre 99 Go/s pour le GEMV — quatorze fois sous la bande passante.
/// Un flottant de padding par ligne rend le pas premier avec 32 et supprime le conflit.
#define TILE_PITCH (TILE_COLS + 1u)

/// Projection dense BF16 tuilée : `y[t] = W·x[t] + biais`.
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

    // `thread` est un qualificatif d'espace d'adressage réservé en Metal.
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
        // Chargement coopératif : 4096 poids et 2048 activations pour 256 fils.
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

/// Projection d'expert MXFP4 tuilée, sur une sélection de jetons.
///
/// Même structure, à ceci près que la tuile de poids est **déquantifiée pendant le
/// chargement** : les valeurs MXFP4 deviennent des flottants en mémoire partagée, et le
/// décodage n'a lieu qu'une fois par tuile au lieu d'une fois par jeton.
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

    // `thread` est un qualificatif d'espace d'adressage réservé en Metal.
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
            // Un bloc MXFP4 couvre 32 colonnes : on localise le bloc, puis le demi-octet.
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

// Concaténé après common.metal et mxfp4.metal.
//
// Noyaux du prefill par blocs. Le principe tient en une phrase : **lire les poids une
// fois pour plusieurs jetons** au lieu d'une fois par jeton.
//
// Mesuré sur une invite de 78 jetons du 20B : traiter les jetons un par un fait relire
// 92,9 Gio de poids denses ; par blocs, 1,2 Gio. Le calcul est identique, seul
// l'ordonnancement change — aucune valeur n'est modifiée, donc aucun risque de qualité.
//
// La contrepartie mémoire est de 8,7 Mio d'activations pour un bloc de 128, à comparer
// aux 1,18 Gio du cache d'experts. Le nombre de slots d'experts, lui, ne bouge pas :
// on itère expert par expert en rassemblant les jetons qui lui sont routés.

/// Jetons traités simultanément par un threadgroup. Chaque voie garde un accumulateur par
/// jeton de la tuile : monter cette valeur réduit les relectures de poids mais consomme
/// des registres. 16 relit les poids 5 fois pour une invite de 78 jetons, contre 78.
#define TOKEN_TILE 16u

/// Lignes traitées par un même threadgroup.
///
/// **C'est le paramètre qui compte, et ce n'est pas celui qu'on croit.** Avec une ligne
/// par threadgroup, chaque threadgroup relit l'intégralité des activations : pour
/// `q_proj` en [4096 × 2880] sur 68 jetons, cela fait 3,8 Go de trafic sur `x` contre
/// 23,6 Mo de poids. Le GEMM ne servait alors à rien — il divisait les relectures de
/// poids par 5 sans toucher au terme dominant.
///
/// En regroupant plusieurs lignes, les activations sont partagées entre elles. Chaque
/// ligne est confiée à un groupe SIMD, ce qui supprime au passage toute barrière de
/// threadgroup : la réduction tient entièrement dans `simd_sum`.
#define ROW_TILE 4u

/// Projection dense BF16 sur un bloc de jetons : `y[t] = W·x[t] + biais`.
///
/// Un threadgroup par (ligne, tuile de jetons). Les poids de la ligne sont lus une seule
/// fois et servent aux `TOKEN_TILE` jetons de la tuile.
kernel void bf16_gemm(
    device const ushort *w      [[buffer(0)]],
    device const ushort *bias   [[buffer(1)]],
    device const float  *x      [[buffer(2)]],  // [tokens][cols]
    device float        *y      [[buffer(3)]],  // [tokens][rows]
    constant uint4      &dims   [[buffer(4)]],  // (rows, cols, tokens, hasBias)
    // Metal exige que tous les attributs de position aient la même dimension :
    // mélanger uint2 et uint dans une même signature ne compile pas.
    uint2 group      [[threadgroup_position_in_grid]],  // (bloc de lignes, tuile de jetons)
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
    const uint groups = cols / 8u;  // huit BF16 par uint4

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

    // La ligne tient dans un groupe SIMD : `simd_sum` suffit, aucune barrière.
    const float biasValue = dims.w != 0u ? bf16_to_float(bias[row]) : 0.0f;
    for (uint t = 0; t < tileCount; ++t) {
        const float total = simd_sum(acc[t]);
        if (simdLane == 0) { y[(tileStart + t) * rows + row] = total + biasValue; }
    }
}

/// Projection d'expert MXFP4 sur une **sélection** de jetons.
///
/// C'est le noyau qui permet au prefill par blocs de ne pas coûter de mémoire : plutôt
/// que de charger tous les experts du bloc en même temps, on itère expert par expert.
/// Pour chacun, `rowIndices` désigne les jetons que le routeur lui a attribués, et la
/// sortie est compactée — un seul slot d'expert est nécessaire à un instant donné,
/// exactement comme en décodage.
kernel void mxfp4_gemm_gathered(
    device const uchar  *blocks     [[buffer(0)]],
    device const uchar  *scales     [[buffer(1)]],
    device const ushort *bias       [[buffer(2)]],
    device const float  *x          [[buffer(3)]],  // [tokens][cols]
    device float        *y          [[buffer(4)]],  // [count][rows], compacté
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

        // Les poids sont décodés en huit float4 gardés en registres.
        //
        // Une version antérieure les rangeait dans un `float weights[32]` réutilisé pour
        // tous les jetons de la tuile : 32 registres de plus, qui débordaient en mémoire
        // et coûtaient bien davantage que le redécodage évité.
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

/// Ajoute la contribution pondérée d'un expert aux jetons qu'il a servis.
kernel void scatter_expert(
    device float        *mixture    [[buffer(0)]],  // [tokens][size]
    device const float  *outputs    [[buffer(1)]],  // [count][size], compacté
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

/// RMSNorm sur un bloc de jetons. Un threadgroup par jeton.
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

/// RoPE sur un bloc : chaque jeton a **sa** position, donc ses propres tables.
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

/// Écrit K et V d'un bloc entier dans le cache.
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

/// SwiGLU sur un bloc. Mêmes règles que la version unitaire : indices pairs/impairs,
/// écrêtage asymétrique, +1 sur la branche linéaire.
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

/// Attention causale sur un bloc. Un threadgroup par (jeton, tête de requête).
///
/// Chaque requête voit son propre passé : les jetons du bloc déjà écrits dans le cache
/// **et** tout ce qui précède le bloc. Le masque n'est donc pas explicite, il découle de
/// la borne de la boucle.
kernel void attention_prefill(
    device const float  *q       [[buffer(0)]],  // [tokens][qHeads * headDim]
    device const half   *kCache  [[buffer(1)]],
    device const half   *vCache  [[buffer(2)]],
    device const ushort *sinks   [[buffer(3)]],
    device float        *out     [[buffer(4)]],  // [tokens][qHeads * headDim]
    constant uint4      &dims    [[buffer(5)]],  // (qHeads, kvHeads, headDim, tokens)
    constant uint4      &window  [[buffer(6)]],  // (ringSize, firstPosition, slidingWindow, _)
    constant float      &smScale [[buffer(7)]],
    uint2 group      [[threadgroup_position_in_grid]],  // (tête, jeton)
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

/// Top-k et softmax du routeur pour un bloc. Un threadgroup par jeton.
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

#include <metal_stdlib>
using namespace metal;

// Conversion bfloat16 -> float.
//
// **Piège majeur, et vérifié par un test dédié.** BF16 n'est pas `half`. Le `half` de
// Metal est l'IEEE binary16 (1 bit de signe, 5 d'exposant, 10 de mantisse) ; le bfloat16
// est la moitié haute d'un float32 (1, 8, 7). Interpréter l'un pour l'autre ne produit ni
// erreur ni NaN — juste des valeurs plausibles et fausses : 1,0 en BF16 se lit 1,875 en
// half. Comme tout ce qui n'est pas expert dans GPT-OSS est en BF16 — attention, normes,
// routeurs, sinks, tête LM — la confusion corromprait le modèle entier en silence.
//
// La conversion vers float est exacte et sans perte : il suffit de replacer les 16 bits
// en position haute. Aucun arrondi, aucun cas particulier — les NaN et infinis se
// transposent tels quels.
inline float bf16_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

/// Sens inverse, arrondi au plus proche pair. Utilisé quand on doit réécrire du BF16.
inline ushort float_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    // Arrondi au plus proche, égalité vers le pair.
    uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
    return ushort((bits + rounding) >> 16);
}

/// Lecture d'un vecteur BF16 comme float.
inline float bf16_load(device const ushort *source, uint index) {
    return bf16_to_float(source[index]);
}

#!/usr/bin/env python3
"""Génère le vecteur de référence MXFP4 des tests de HydraFormat.

Implémentation **indépendante** de celle de Swift, écrite directement d'après la
sémantique de `gpt_oss/torch/weights.py` :

  - table E2M1 indexée par le nibble brut ;
  - nibble bas -> index pair, nibble haut -> index impair ;
  - valeur = fp4 * 2^(octet_échelle - 127).

Le calcul est fait en float64 puis arrondi en float32 par `struct.pack`, ce qui est
exactement ce que produit Swift : toutes les valeurs de la table et toutes les
puissances de deux en jeu sont exactement représentables en float32, donc l'égalité
attendue est **bit à bit**, pas approchée.

Usage : python3 tools/gen_mxfp4_fixture.py Tests/HydraFormatTests/Fixtures
"""
import os
import struct
import sys

FP4 = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

BLOCK_VALUES = 32
PACKED_BYTES = 16

# Octets d'échelle couvrant les extrêmes utiles sans provoquer d'inf en float32 :
#   0   -> 2^-127 (sous-normal côté produit, exactement représentable)
#   200 -> 2^73
# On évite volontairement 255, qui encode un NaN dans la spécification OCP et que la
# référence transforme en facteur infini : ce cas a son propre test.
SCALE_POOL = [0, 1, 60, 100, 120, 126, 127, 128, 130, 150, 200]


def build():
    packed = bytearray()
    scales = bytearray()

    # Bloc A — exhaustif : les 256 valeurs d'octet possibles, donc les 16 nibbles
    # dans les deux positions. 256 octets = 16 blocs.
    for b in range(256):
        packed.append(b)
    for i in range(256 // PACKED_BYTES):
        scales.append(SCALE_POOL[i % len(SCALE_POOL)])

    # Bloc B — pseudo-aléatoire déterministe, 48 blocs.
    state = 0x2545F491
    for i in range(48 * PACKED_BYTES):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        packed.append((state >> 16) & 0xFF)
    for i in range(48):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        scales.append(SCALE_POOL[(state >> 16) % len(SCALE_POOL)])

    assert len(packed) == len(scales) * PACKED_BYTES

    expected = bytearray()
    for blk in range(len(scales)):
        factor = 2.0 ** (scales[blk] - 127)
        base = blk * PACKED_BYTES
        for i in range(PACKED_BYTES):
            byte = packed[base + i]
            expected += struct.pack("<f", FP4[byte & 0x0F] * factor)
            expected += struct.pack("<f", FP4[byte >> 4] * factor)

    return bytes(packed), bytes(scales), bytes(expected)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "Tests/HydraFormatTests/Fixtures"
    os.makedirs(out, exist_ok=True)
    packed, scales, expected = build()
    for name, blob in (("mxfp4_packed.bin", packed),
                       ("mxfp4_scales.bin", scales),
                       ("mxfp4_expected.f32", expected)):
        with open(os.path.join(out, name), "wb") as f:
            f.write(blob)
        print(f"{name:24s} {len(blob):>8,d} o")
    print(f"\n{len(scales)} blocs, {len(scales) * BLOCK_VALUES} valeurs décodées")

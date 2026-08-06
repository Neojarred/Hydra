#!/usr/bin/env python3
"""Generates the MXFP4 reference vector for HydraFormat's tests.

An implementation **independent** of Swift's, written straight from the semantics of
`gpt_oss/torch/weights.py`:

  - an E2M1 table indexed by the raw nibble;
  - low nibble -> even index, high nibble -> odd index;
  - value = fp4 * 2^(scale_byte - 127).

The computation runs in float64 then rounds to float32 through `struct.pack`, which is
exactly what Swift produces: every value in the table and every power of two involved is
exactly representable in float32, so the expected equality is **bit-exact**, not
approximate.

Usage: python3 tools/gen_mxfp4_fixture.py Tests/HydraFormatTests/Fixtures
"""
import os
import struct
import sys

FP4 = [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

BLOCK_VALUES = 32
PACKED_BYTES = 16

# Scale bytes covering the useful extremes without producing an inf in float32:
#   0   -> 2^-127 (subnormal on the product side, exactly representable)
#   200 -> 2^73
# We deliberately avoid 255, which encodes a NaN in the OCP specification and which the
# reference turns into an infinite factor: that case has its own test.
SCALE_POOL = [0, 1, 60, 100, 120, 126, 127, 128, 130, 150, 200]


def build():
    packed = bytearray()
    scales = bytearray()

    # Block A — exhaustive: all 256 possible byte values, hence the 16 nibbles in both
    # positions. 256 bytes = 16 blocks.
    for b in range(256):
        packed.append(b)
    for i in range(256 // PACKED_BYTES):
        scales.append(SCALE_POOL[i % len(SCALE_POOL)])

    # Block B — deterministic pseudo-random, 48 blocks.
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
        print(f"{name:24s} {len(blob):>8,d} B")
    print(f"\n{len(scales)} blocks, {len(scales) * BLOCK_VALUES} decoded values")

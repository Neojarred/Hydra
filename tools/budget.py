#!/usr/bin/env python3
"""Hydra — memory budget and throughput calculator.

Every tensor size is derived from the real safetensors headers
(see docs/00-AUDIT-CHECKPOINTS.md), not from estimates.

Hardware constants measured on the target machine (MacBook M4, 24 GiB):
  - MTLDevice.recommendedMaxWorkingSetSize = 19,069,665,280 B (17.76 GiB)
  - MTLDevice.maxBufferLength              = 14,302,248,960 B (13.32 GiB)
  - GPU memory bandwidth (streaming read)   ~94 GB/s
  - SSD bandwidth, parallel 13.24 MB preads, F_NOCACHE  ~5.5 GB/s
"""

GIB = 1024 ** 3
MIB = 1024 ** 2

# --- measured hardware -------------------------------------------------------
METAL_DEFAULT_CEILING = 19_069_665_280      # recommendedMaxWorkingSetSize
RAM_TOTAL = 25_769_803_776                  # 24 GiB
SSD_BW = 5.5e9                              # B/s, cold, 4-8 parallel reads
MEM_BW = 94e9                               # B/s, measured GPU streaming read


def bf16(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 2


class GptOss:
    """GPT-OSS: MXFP4 MoE, no shared expert, alternating SWA/full attention."""

    def __init__(self, name, layers, experts, ctx_default=8192):
        self.name = name
        self.L = layers
        self.E = experts
        self.top_k = 4
        self.hidden = 2880
        self.inter = 2880
        self.heads = 64
        self.kv_heads = 8
        self.head_dim = 64
        self.vocab = 201_088
        self.sliding_window = 128
        self.ctx = ctx_default
        # half the layers sliding, half full (layer_types alternates)
        self.full_layers = layers // 2
        self.swa_layers = layers - self.full_layers

    # --- weights ---
    def expert_blob(self):
        """One MXFP4 expert blob: blocks of 32 values, 16 packed B + 1 scale B."""
        nb_in = self.hidden // 32                      # 90 blocks
        gate_up = (2 * self.inter) * nb_in * 16        # blocks
        gate_up += (2 * self.inter) * nb_in            # scales (u8)
        gate_up += bf16(2 * self.inter)                # bias
        nb_in2 = self.inter // 32
        down = self.hidden * nb_in2 * 16
        down += self.hidden * nb_in2
        down += bf16(self.hidden)
        return gate_up + down

    def attn_per_layer(self):
        q = bf16(self.heads * self.head_dim, self.hidden) + bf16(self.heads * self.head_dim)
        k = bf16(self.kv_heads * self.head_dim, self.hidden) + bf16(self.kv_heads * self.head_dim)
        v = k
        o = bf16(self.hidden, self.heads * self.head_dim) + bf16(self.hidden)
        sinks = bf16(self.heads)
        return q + k + v + o + sinks

    def resident_per_layer(self):
        router = bf16(self.E, self.hidden) + bf16(self.E)
        norms = 2 * bf16(self.hidden)
        return self.attn_per_layer() + router + norms

    def resident(self):
        embed = bf16(self.vocab, self.hidden)
        head = bf16(self.vocab, self.hidden)          # tie_word_embeddings = false
        return embed + head + self.L * self.resident_per_layer() + bf16(self.hidden)

    def expert_pool(self):
        return self.L * self.E * self.expert_blob()

    # --- KV cache FP16 ---
    def kv_per_token_full(self):
        return 2 * self.kv_heads * self.head_dim * 2   # K + V, FP16

    def kv(self, ctx=None):
        ctx = ctx or self.ctx
        full = self.full_layers * self.kv_per_token_full() * ctx
        ring_rows = self.sliding_window + 128          # margin for the chunked prefill
        swa = self.swa_layers * self.kv_per_token_full() * ring_rows
        return full + swa

    # --- throughput ---
    def bytes_read_per_token_gpu(self):
        """Bytes the GPU must read from memory for one decoding token."""
        attn = self.L * self.resident_per_layer()
        head = bf16(self.vocab, self.hidden)
        experts = self.L * self.top_k * self.expert_blob()
        return attn + head + experts

    def io_per_token(self, hit_rate):
        return self.L * self.top_k * self.expert_blob() * (1.0 - hit_rate)


class QwenA3B:
    """Qwen3.6-35B-A3B: MoE + shared expert, 30 Gated DeltaNet layers / 10 full attn."""

    def __init__(self, bits=4.25):
        self.name = "Qwen3.6-35B-A3B"
        self.L = 40
        self.E = 256
        self.top_k = 8
        self.hidden = 2048
        self.moe_inter = 512
        self.shared_inter = 512
        self.vocab = 248_320
        self.full_layers = 10
        self.linear_layers = 30
        self.kv_heads = 2
        self.head_dim = 256
        self.bits = bits

    def expert_params(self):
        return self.hidden * (2 * self.moe_inter) + self.moe_inter * self.hidden

    def expert_blob(self):
        return int(self.expert_params() * self.bits / 8)

    def expert_pool(self):
        return self.L * self.E * self.expert_blob()

    def kv_per_token_full(self):
        return 2 * self.kv_heads * self.head_dim * 2

    def kv(self, ctx):
        return self.full_layers * self.kv_per_token_full() * ctx


def gib(x):
    return x / GIB


def row(label, value_bytes, width=46):
    print(f"  {label:<{width}} {value_bytes:>16,d} B   {gib(value_bytes):>7.3f} GiB")


def report_gptoss(m, ctxs, ceilings, scratch=512 * MIB):
    print("=" * 88)
    print(f"  {m.name}  —  {m.L} layers, {m.E} experts/layer, top-{m.top_k}")
    print("=" * 88)
    blob = m.expert_blob()
    print("\n  WEIGHTS")
    row("One expert's blob (MXFP4)", blob)
    row("Complete expert pool (on disk)", m.expert_pool())
    row("Resident weights (BF16, required in RAM)", m.resident())
    row("  of which embed_tokens", bf16(m.vocab, m.hidden))
    row("  of which lm_head", bf16(m.vocab, m.hidden))
    row("  of which attention+router+norms", m.L * m.resident_per_layer())
    row("Total installation on disk", m.expert_pool() + m.resident())

    print(f"\n  FP16 KV CACHE ({m.full_layers} full layers + {m.swa_layers} SWA rings of "
          f"{m.sliding_window + 128} rows)")
    for c in ctxs:
        row(f"context {c//1024}k", m.kv(c))

    print("\n  BUDGET — expert cache slots available")
    print(f"  {'Metal ceiling':<20} {'ctx':>6} {'free for experts':>18} {'slots tot.':>11} {'slots/layer':>13}")
    for cname, ceiling in ceilings:
        for c in ctxs:
            free = ceiling - m.resident() - m.kv(c) - scratch
            slots = int(free // blob)
            per_layer = slots // m.L
            flag = "" if free > 0 else "  <-- DOES NOT FIT"
            print(f"  {cname:<20} {c//1024:>5}k {gib(free):>15.2f} GiB {slots:>11,d} {per_layer:>13,d}{flag}")

    print("\n  MODELLED THROUGHPUT")
    gpu_bytes = m.bytes_read_per_token_gpu()
    t_gpu = gpu_bytes / MEM_BW
    print(f"  Bytes read by the GPU / token: {gpu_bytes:,} B")
    print(f"  Compute floor (BW {MEM_BW/1e9:.0f} GB/s): {t_gpu*1000:.1f} ms/token"
          f"  ->  ceiling {1/t_gpu:.1f} tok/s")
    print(f"\n  {'cache hit':>10} {'I/O/token':>14} {'t_io':>9} {'t_tok (serial)':>14} {'tok/s':>8}")
    for h in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        io = m.io_per_token(h)
        t_io = io / SSD_BW
        t = t_gpu + t_io          # pessimistic assumption: no overlap
        print(f"  {h*100:>9.0f}% {io/1e6:>12.0f} MB {t_io*1000:>7.0f} ms {t*1000:>12.0f} ms {1/t:>8.2f}")
    print()


if __name__ == "__main__":
    ceilings = [
        ("default 17.76 GiB", METAL_DEFAULT_CEILING),
        ("wired_limit 20 GiB", 20 * GIB),
        ("wired_limit 21 GiB", 21 * GIB),
    ]
    for m in (GptOss("GPT-OSS 20B", 24, 32), GptOss("GPT-OSS 120B", 36, 128)):
        report_gptoss(m, [8192, 32768, 131072], ceilings)

    q = QwenA3B()
    print("=" * 88)
    print(f"  {q.name}  —  {q.L} layers ({q.linear_layers} Gated DeltaNet / {q.full_layers} full attn),")
    print(f"  {q.E} experts/layer, top-{q.top_k}, + 1 shared expert")
    print("=" * 88)
    row("One expert's blob (~4.25 bits)", q.expert_blob())
    row("Complete expert pool", q.expert_pool())
    print(f"\n  Worst-case I/O per token (top-{q.top_k} x {q.L} layers):"
          f" {q.L*q.top_k*q.expert_blob()/1e6:.0f} MB"
          f"  ->  {q.L*q.top_k*q.expert_blob()/SSD_BW*1000:.0f} ms")
    for c in (32768, 262144):
        row(f"FP16 KV cache, context {c//1024}k", q.kv(c))
    print("  (the 30 Gated DeltaNet layers have no KV cache but a bounded recurrent state)")
    print()

#!/usr/bin/env python3
"""Hydra — calculateur de budget mémoire et de débit.

Toutes les tailles de tenseurs sont dérivées des en-têtes safetensors réels
(voir docs/00-AUDIT-CHECKPOINTS.md), pas d'estimations.

Constantes matérielles mesurées sur la machine cible (MacBook M4, 24 Gio) :
  - MTLDevice.recommendedMaxWorkingSetSize = 19 069 665 280 o (17,76 Gio)
  - MTLDevice.maxBufferLength              = 14 302 248 960 o (13,32 Gio)
  - Bande passante mémoire GPU (lecture streaming)   ~94 Go/s
  - Bande passante SSD, preads 13,24 Mo parallèles, F_NOCACHE  ~5,5 Go/s
"""

GIB = 1024 ** 3
MIB = 1024 ** 2

# --- matériel mesuré ---------------------------------------------------------
METAL_DEFAULT_CEILING = 19_069_665_280      # recommendedMaxWorkingSetSize
RAM_TOTAL = 25_769_803_776                  # 24 Gio
SSD_BW = 5.5e9                              # o/s, à froid, 4-8 lectures parallèles
MEM_BW = 94e9                               # o/s, lecture streaming GPU mesurée


def bf16(*dims):
    n = 1
    for d in dims:
        n *= d
    return n * 2


class GptOss:
    """GPT-OSS : MoE MXFP4, pas d'expert partagé, attention alternée SWA/full."""

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
        # moitié des couches en sliding, moitié en full (layer_types alterne)
        self.full_layers = layers // 2
        self.swa_layers = layers - self.full_layers

    # --- poids ---
    def expert_blob(self):
        """Un blob d'expert MXFP4 : blocs de 32 valeurs, 16 o packés + 1 o d'échelle."""
        nb_in = self.hidden // 32                      # 90 blocs
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
        ring_rows = self.sliding_window + 128          # marge pour le prefill chunké
        swa = self.swa_layers * self.kv_per_token_full() * ring_rows
        return full + swa

    # --- débit ---
    def bytes_read_per_token_gpu(self):
        """Octets que le GPU doit lire en mémoire pour un token de décodage."""
        attn = self.L * self.resident_per_layer()
        head = bf16(self.vocab, self.hidden)
        experts = self.L * self.top_k * self.expert_blob()
        return attn + head + experts

    def io_per_token(self, hit_rate):
        return self.L * self.top_k * self.expert_blob() * (1.0 - hit_rate)


class QwenA3B:
    """Qwen3.6-35B-A3B : MoE + expert partagé, 30 couches Gated DeltaNet / 10 full attn."""

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
    print(f"  {label:<{width}} {value_bytes:>16,d} o   {gib(value_bytes):>7.3f} Gio")


def report_gptoss(m, ctxs, ceilings, scratch=512 * MIB):
    print("=" * 88)
    print(f"  {m.name}  —  {m.L} couches, {m.E} experts/couche, top-{m.top_k}")
    print("=" * 88)
    blob = m.expert_blob()
    print("\n  POIDS")
    row("Blob d'un expert (MXFP4)", blob)
    row("Pool d'experts complet (sur disque)", m.expert_pool())
    row("Poids résidents (BF16, obligatoires en RAM)", m.resident())
    row("  dont embed_tokens", bf16(m.vocab, m.hidden))
    row("  dont lm_head", bf16(m.vocab, m.hidden))
    row("  dont attention+routeur+normes", m.L * m.resident_per_layer())
    row("Installation totale sur disque", m.expert_pool() + m.resident())

    print(f"\n  KV CACHE FP16 ({m.full_layers} couches full + {m.swa_layers} anneaux SWA de "
          f"{m.sliding_window + 128} lignes)")
    for c in ctxs:
        row(f"contexte {c//1024}k", m.kv(c))

    print("\n  BUDGET — slots de cache d'experts disponibles")
    print(f"  {'plafond Metal':<20} {'ctx':>6} {'libre p. experts':>18} {'slots tot.':>11} {'slots/couche':>13}")
    for cname, ceiling in ceilings:
        for c in ctxs:
            free = ceiling - m.resident() - m.kv(c) - scratch
            slots = int(free // blob)
            per_layer = slots // m.L
            flag = "" if free > 0 else "  <-- NE TIENT PAS"
            print(f"  {cname:<20} {c//1024:>5}k {gib(free):>15.2f} Gio {slots:>11,d} {per_layer:>13,d}{flag}")

    print("\n  DEBIT MODELISE")
    gpu_bytes = m.bytes_read_per_token_gpu()
    t_gpu = gpu_bytes / MEM_BW
    print(f"  Octets lus par le GPU / token : {gpu_bytes:,} o")
    print(f"  Plancher calcul (BP {MEM_BW/1e9:.0f} Go/s) : {t_gpu*1000:.1f} ms/token"
          f"  ->  plafond {1/t_gpu:.1f} tok/s")
    print(f"\n  {'hit cache':>10} {'I/O/token':>14} {'t_io':>9} {'t_tok (série)':>14} {'tok/s':>8}")
    for h in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0):
        io = m.io_per_token(h)
        t_io = io / SSD_BW
        t = t_gpu + t_io          # hypothèse pessimiste : aucun recouvrement
        print(f"  {h*100:>9.0f}% {io/1e6:>12.0f} Mo {t_io*1000:>7.0f} ms {t*1000:>12.0f} ms {1/t:>8.2f}")
    print()


if __name__ == "__main__":
    ceilings = [
        ("défaut 17,76 Gio", METAL_DEFAULT_CEILING),
        ("wired_limit 20 Gio", 20 * GIB),
        ("wired_limit 21 Gio", 21 * GIB),
    ]
    for m in (GptOss("GPT-OSS 20B", 24, 32), GptOss("GPT-OSS 120B", 36, 128)):
        report_gptoss(m, [8192, 32768, 131072], ceilings)

    q = QwenA3B()
    print("=" * 88)
    print(f"  {q.name}  —  {q.L} couches ({q.linear_layers} Gated DeltaNet / {q.full_layers} full attn),")
    print(f"  {q.E} experts/couche, top-{q.top_k}, + 1 expert partagé")
    print("=" * 88)
    row("Blob d'un expert (~4,25 bits)", q.expert_blob())
    row("Pool d'experts complet", q.expert_pool())
    print(f"\n  I/O par token au pire (top-{q.top_k} x {q.L} couches) :"
          f" {q.L*q.top_k*q.expert_blob()/1e6:.0f} Mo"
          f"  ->  {q.L*q.top_k*q.expert_blob()/SSD_BW*1000:.0f} ms")
    for c in (32768, 262144):
        row(f"KV cache FP16 contexte {c//1024}k", q.kv(c))
    print("  (les 30 couches Gated DeltaNet n'ont pas de KV cache mais un état récurrent borné)")
    print()

# Hydra, Feasibility study (Phase 0)

> **A historical document, kept as written.** This is the phase-0 study, made before any
> runtime code existed. It is not updated to agree with what was later measured, because its
> value is in showing what could be predicted from the hardware and the checkpoints alone, and
> where that prediction went wrong. `docs/02-MEASUREMENTS.md` holds what actually happened.
>
> The estimate that aged worst is the throughput model: it reasoned from memory bandwidth, and
> the 4-bit kernels turned out to be bound by the number of values unpacked rather than the
> bytes moved (M-040), which no bandwidth argument would have found.

Status: **to be approved**. No runtime code written at this stage.
Reference machine: MacBook **Apple M4** (10 GPU cores), **24 GiB** of unified memory,
macOS **26.5.2**, Swift **6.3.2**, **126 GiB** free on the internal SSD.

Every figure below is **measured or derived from real safetensors headers**. Where a value
remains an assumption, it says so explicitly. The calculator is reproducible:
`python3 tools/budget.py`.

---

## Verdict in ten lines

1. **GPT-OSS 120B fits in 24 GiB.** The budget closes with room to spare: 3.96 GiB of
   resident weights, 1.13 GiB of KV at 32k, leaving **12.2 GiB for the expert cache** under
   the default Metal ceiling. The project is not blocked by a physical constraint.
2. **But it will be slow**: **3 to 5 tok/s** expected. The absolute ceiling, with a perfect
   cache and zero I/O, is **18.8 tok/s**, set by memory bandwidth, not by the SSD.
3. **GPT-OSS 20B *can* fit entirely in memory (12.82 GiB under a 17.76 GiB ceiling), but
   that is not the intended mode**, see D-012. It runs at a minimum of **3.77 GiB**, i.e.
   28 % of its installed size. The fully resident mode serves as a **correctness
   reference**: under greedy decoding the two must produce exactly the same tokens.
4. **GPT-OSS has no shared expert.** Confirmed against the safetensors index. The mechanism
   that hides I/O latency in TurboFieldfare does not exist here.
5. Good news: that mechanism was worth only **+7.5 %** in TurboFieldfare (published
   measurement). Losing it is real but not disqualifying.
6. **The SSD is twice as fast as expected**: 5.5 GB/s cold on the real access pattern.
7. **This machine's real Metal ceiling is 17.76 GiB**, not 24. Measured, not assumed.
8. **GPT-OSS's sliding window is 128 tokens**, 18 layers out of 36 keep almost nothing. The
   KV cache is surprisingly cheap: 4.51 GiB at 128k.
9. **Qwen3.6-35B-A3B is indeed the right MoE candidate**, but 30 of its 40 layers are
   **Gated DeltaNet**, an entirely new family of kernels. It is the project's most
   expensive item, and it needs discussing.
10. **Storage is the tightest constraint**, not memory: 126 GiB free for ~91 GiB of models.
    Streaming repack is not an elegance, it is a necessity.

11. **The memory floor is not the expert cache, it is the BF16 resident weights.** 2.27 GiB
    for the 20B, 2.88 GiB for the 120B, of which 1.08 GiB for the LM head alone. GPT-OSS
    quantizes only its experts; TurboFieldfare reached ~2 GB total because Gemma 4 had those
    tensors in 4-bit. That is the main remaining lever, and it costs a numerical validation.

---

## 1. The machine, measured

| Quantity | Measured value | How |
| --- | ---: | --- |
| `MTLDevice.recommendedMaxWorkingSetSize` | **19,069,665,280 B (17.76 GiB)** | Swift/Metal probe |
| `MTLDevice.maxBufferLength` | 14,302,248,960 B (13.32 GiB) | Swift/Metal probe |
| Physical memory | 25,769,803,776 B (24.00 GiB) | `ProcessInfo` |
| GPU family | **apple9** (not apple10) | `supportsFamily` |
| GPU memory bandwidth, streaming read | **~94 GB/s** | Metal kernel, 2 GB, 3 passes |
| SSD, 13.24 MB random parallel preads, `F_NOCACHE` | **5.3 to 5.7 GB/s** (4 to 8 threads) | dedicated C bench |
| SSD, same pattern, 1 thread | 3.0 GB/s | idem |
| SSD, same pattern, page cache allowed | up to 18.0 GB/s | idem |
| Free disk space | 126 GiB | `df` |

Three immediate consequences.

**The 17.76 GiB ceiling confirms your warning**, that is 74 % of 24 GiB. The workaround
`sudo sysctl iogpu.wired_limit_mb=<MB>` still works under macOS 26. Raising it to 20,480
(20 GiB) gains only **+5 expert slots per layer** on the 120B (27 → 32). A real but small
gain, at the price of a `sudo` command and a risk of system memory pressure. **My
recommendation: detect and display the ceiling, offer the increase as an explicit and
documented option, but never require it nor apply it without user action.** The application
must be correct within the default budget.

**We are on apple9, not apple10.** TurboFieldfare's "TensorOps" path, which makes attention
11× faster at 64k, is apple10-only (M5). On M4 we inherit the tiled path, which is slower.
An important corollary: **the "31-35 tok/s on M5 Pro 24 GB" benchmark is not a reachable
target for us**, the M5 Pro combines substantially higher memory bandwidth with a newer GPU
family. The right mental reference is the M2 8 GB (5-6 tok/s), with our RAM advantage on top.

**The 13.32 GiB `maxBufferLength`** rules out any single `MTLBuffer` covering the expert
pool. Irrelevant here, the architecture allocates one buffer per slot, but it definitively
closes the "map the whole pool as one buffer" option.

---

## 2. Checkpoint audit

### 2.1 GPT-OSS, exact structure

Extracted from real safetensors headers (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`):

| Key (layer 0) | dtype | shape | bytes |
| --- | --- | --- | ---: |
| `self_attn.q_proj.weight` | BF16 | [4096, 2880] | 23,592,960 |
| `self_attn.k_proj.weight` | BF16 | [512, 2880] | 2,949,120 |
| `self_attn.v_proj.weight` | BF16 | [512, 2880] | 2,949,120 |
| `self_attn.o_proj.weight` | BF16 | [2880, 4096] | 23,592,960 |
| `self_attn.sinks` | BF16 | [64] | 128 |
| `mlp.router.weight` | BF16 | [E, 2880] | E × 5,760 |
| `mlp.experts.gate_up_proj_blocks` | **U8** | [E, 5760, **90, 16**] | E × 8,294,400 |
| `mlp.experts.gate_up_proj_scales` | **U8** | [E, 5760, **90**] | E × 518,400 |
| `mlp.experts.gate_up_proj_bias` | BF16 | [E, 5760] | E × 11,520 |
| `mlp.experts.down_proj_blocks` | **U8** | [E, 2880, **90, 16**] | E × 4,147,200 |
| `mlp.experts.down_proj_scales` | **U8** | [E, 2880, **90**] | E × 259,200 |
| `mlp.experts.down_proj_bias` | BF16 | [E, 2880] | E × 5,760 |

**The MXFP4 layout is fully determined by these shapes.** `[…, 90, 16]` with an input
dimension of 2880 gives 2880 / 90 = **32 values per block**, stored in **16 bytes** (two FP4
E2M1 per `uint8`), plus **1 scale byte per block** (`scales` in U8, i.e. E8M0). Total
**4.25 bits per weight**. That is the OCP Microscaling standard, with a block of 32, not 64
as in Gemma 4's affine MLX format.

Your estimate of "~13.2 MB per expert blob" was right: the exact value is **13,236,480
bytes**, and it is **identical for the 20B and the 120B** (same `hidden_size` 2880, same
`intermediate_size` 2880). Your figure of 4.2 GB of resident weights for the 120B was right
too: **4,255,115,904 B = 3.963 GiB**.

### 2.2 The four findings that change the design

**(a) There is no shared expert.** A complete inventory of the 120B's keys contains only
`mlp.router.*` and `mlp.experts.*`, no `shared_expert`. The question you flagged as
structural is settled: **TurboFieldfare's CPU/GPU overlap does not transfer.**

What it actually costs: TurboFieldfare measured that overlap at **4.404 → 4.736 tok/s, i.e.
+7.5 %**. Not the collapse one might fear. But our ratio is unfavourable, their I/O is
88 ms/token, ours 173 to 347 ms, so the maskable share is larger, and the real loss will
exceed 7 %.

**There is no clean substitute**, and it should be said plainly:
- the router depends on the same layer's attention output → impossible to launch reads
  before `cb1` finishes;
- layer L+1 depends on layer L → no inter-layer pipeline in decoding;
- the LM head of token *t−1* precedes the sampling that determines token *t* → no overlap
  there either;
- speculative inter-layer prefetching is dead: TurboFieldfare measured that one layer's
  choices predict only **7 %** of the next layer's experts.

**Accepted conclusion: for the 120B we do not fight for overlap, we fight for the hit rate.**
It is the only lever that matters, and it drives the whole cache design.

**(b) `sliding_window = 128`.** Every other layer looks back only 128 tokens. Only the 18
full-attention layers carry long context. The KV cache collapses to **4.51 GiB at 128k**
instead of the ~9 GiB it would be without a window. Long context is nearly free, excellent
news, and worth exploiting.

**(c) `embed_tokens` does not need to be resident.** 1.079 GiB, but we read only **one row
per token** (128 in chunked prefill). It can stay `mmap`'d and paged on demand, outside the
Metal working set. **1.079 GiB recovered for free**, i.e. +87 expert slots. Conversely
`lm_head` is read in full on every token and must stay resident.

**(d) The LM head is a major compute item.** 1.079 GiB in BF16 read per token = **11.5 ms of
the 53 ms** compute floor of the 120B, i.e. 22 %. Quantizing it to MXFP4 would bring it to
0.29 GiB: ~8 ms/token saved **and** 0.8 GiB freed. It is the best identified gain/effort
ratio, but it changes outputs, so it is an experiment to validate against a reference, not
a design decision.

### 2.3 Which Hugging Face source for the repacker

The `openai/gpt-oss-120b` repository offers three forms. Measured:

| Form | Contents | Size |
| --- | --- | ---: |
| root | 14 safetensors, HF standard | **65,248,815,744 B** (60.77 GiB) |
| `original/` | 7 safetensors, PyTorch reference | ~65.25 GB |
| `metal/model.bin` | single file for OpenAI's Metal implementation | 65,238,253,568 B |

**Recommendation: the root.** MXFP4 values there are already split into `blocks` / `scales`
/ `bias`, exactly the division the repacker needs. Every expert sub-tensor is a contiguous,
addressable byte range, so the repack proceeds by bounded HTTP `Range` requests without ever
materializing a shard. The other two forms would impose an extra layout transformation for
no benefit.

### 2.4 Qwen3.6-35B-A3B, verification

**Your correction was right, and so was your candidate**: `Qwen/Qwen3.6-35B-A3B` is indeed
MoE. Real config verified:

| Parameter | Value |
| --- | ---: |
| `num_experts` | **256** |
| `num_experts_per_tok` | **8** |
| `shared_expert_intermediate_size` | **512** → **yes, there is a shared expert** |
| `num_hidden_layers` | 40 |
| `hidden_size` | 2048 |
| `moe_intermediate_size` | 512 |
| `full_attention_interval` | 4 → **10 full-attn layers, 30 `linear_attention`** |
| `head_dim` / `num_key_value_heads` | 256 / 2 |
| `max_position_embeddings` | 262,144 |
| `vision_config` | **present, the model is multimodal** |

Quantified consequences: expert blob ≈ **1.67 MB** at 4.25 bits, full pool **15.94 GiB**,
worst-case I/O **535 MB/token → 97 ms**. Streaming would work very well there, and the
shared expert restores the overlap lost on GPT-OSS.

**But there is a problem I want to flag early.** The 30 `linear_attention` layers are **Gated
DeltaNet**: causal convolution, recurrent delta rule, gating, a family of operators with
**nothing in common** with GPT-OSS attention. This is not "one more MoE model": it is a
second sequence engine to write and validate in full. Roughly, that represents **as much
kernel work as both GPT-OSS models combined**. Add a vision tower to explicitly ignore.

I am not saying it is infeasible. I am saying that placing it at priority 3 probably
underestimates its cost by a factor of 2 to 3, and that **this trade-off is yours**, I put
it in the open questions.

---

## 3. Memory budget

Assumptions: reusable scratch 512 MiB, KV in FP16, SWA rings of 256 rows (128 of window +
128 of margin for chunked prefill).

### GPT-OSS 20B, fits entirely, but does not use that by default

| Item | Size |
| --- | ---: |
| Resident weights | 3.349 GiB |
| **Full** expert pool | 9.467 GiB |
| KV cache at 32k | 0.756 GiB |
| Scratch | 0.500 GiB |
| **Total** | **14.07 GiB** |
| Default Metal ceiling | 17.76 GiB |

**The 20B fits entirely in memory, experts included.** The calculator reports 44 slots
available per layer while the model has only 32: the cache is saturated by construction, the
hit rate is 100 %, I/O is zero after loading. It is an ideal test bench, it validates the
whole chain (download, repack, MXFP4 kernels, attention, sinks, Harmony, UI) **without
streaming masking a kernel bug**.

### GPT-OSS 120B, the target

| Item | Size |
| --- | ---: |
| Resident weights | **3.963 GiB** |
| KV cache 8k / 32k / 128k | 0.290 / 1.134 / 4.509 GiB |
| Scratch | 0.500 GiB |
| Expert pool on disk | 56.805 GiB |

Expert cache slots available (out of 128 per layer):

| Metal ceiling | ctx 8k | ctx 32k | ctx 128k |
| --- | ---: | ---: | ---: |
| **17.76 GiB (default)** | **29** | **27** | 19 |
| 20 GiB (`wired_limit`) | 34 | 32 | 24 |
| 21 GiB (`wired_limit`) | 36 | 34 | 27 |

Taking `embed_tokens` out of the working set (§2.2c) adds **+2 slots per layer**.

**Reading: we can keep 21 to 28 % of the experts cached.** That is the figure that decides
the project.

### Qwen3.6-35B-A3B

Pool 15.94 GiB + residents ≈ 2.3 GiB ≈ **18.2 GiB at 4 bits**. Just above the default
ceiling, so light streaming, or slightly more aggressive quantization, or a raised
`wired_limit`. The 30 DeltaNet layers have no KV cache but a bounded recurrent state, which
makes 256k context realistic on the memory side.

---

## 4. Throughput, the estimate you asked for before investing weeks

Model: `t_token = t_compute + t_io`, **with no overlap at all**. Deliberately pessimistic,
but it is the honest regime for GPT-OSS since there is no shared expert (§2.2a).

### Compute floor (independent of the SSD)

| Model | Bytes read by the GPU / token | at 94 GB/s | ceiling |
| --- | ---: | ---: | ---: |
| GPT-OSS 20B | 3,708,077,568 | 39.4 ms | **25.4 tok/s** |
| GPT-OSS 120B | 5,002,896,384 | 53.2 ms | **18.8 tok/s** |

**This ceiling is structural.** Even with a perfect cache and an infinitely fast SSD, the
120B will not exceed ~19 tok/s on this M4, because 5 GB per token must pass through 94 GB/s
of bandwidth. No I/O optimization crosses that wall.

### GPT-OSS 120B as a function of hit rate

| Hit rate | I/O / token | t_io | t_token | **tok/s** |
| ---: | ---: | ---: | ---: | ---: |
| 0 % | 1,906 MB | 347 ms | 400 ms | 2.50 |
| 20 % | 1,525 MB | 277 ms | 330 ms | 3.03 |
| **30 %** | 1,334 MB | 243 ms | 296 ms | **3.38** |
| **40 %** | 1,144 MB | 208 ms | 261 ms | **3.83** |
| **50 %** | 953 MB | 173 ms | 226 ms | **4.42** |
| 60 % | 762 MB | 139 ms | 192 ms | 5.21 |
| 80 % | 381 MB | 69 ms | 123 ms | 8.16 |

With 27 slots out of 128 (21 %), uniform routing would give a 21 % hit rate. Real MoE
routing is appreciably skewed, some experts are called far more often, so **the realistic
range is 30 to 50 %, i.e. 3.4 to 4.4 tok/s.** Partial overlap and the macOS page cache may
add 10 to 20 %.

**Direct answer to your question "1 tok/s or 15 tok/s?": about 4 tok/s.**

That is usable for document analysis, code review, asynchronous work. It is uncomfortable
for real-time chat, roughly three words per second. I think it is a result worth the
project, but you should know it now, not in phase 4.

### The uncertainty to remove first

**Everything rests on the hit rate, and I cannot compute it, it has to be measured.**

It happens to come **free in phase 1**: the 20B is fully resident, so we can instrument its
router and record the real expert distribution over thousands of tokens, then simulate the
"LFU hit rate versus slot count" curve offline. Same family, same top-4, same training. It
is not a proof for the 120B, but it is a solid predictor obtained without writing an extra
line of code.

**Proposed decision criterion: if the extrapolated curve gives less than 25 % hit at 27
slots, we stop and reassess before writing the 120B repacker.**

---

## 5. Proposed architecture

### 5.1 Modules

```
HydraCore        model contract, types, budget, telemetry
HydraFormat      .hydra format: manifest, layout, integrity, range reads
HydraInstall     streaming HTTP Range -> .hydra repacker, resume, verification
HydraMetal       Metal context, kernel library, pipeline specialization
HydraRuntime     execution graph, expert cache, KV cache, prefill/decoding
HydraTokenize    tokenizers + Harmony rendering/parsing
HydraCLI         command-line tool (install, bench, gen, verify)
HydraApp         SwiftUI + AppKit
```

Separating `HydraInstall` from `HydraRuntime` is deliberate: the installer is the only module
allowed to talk to the network, which makes the memory invariant checkable by code review.

### 5.2 The `.hydra` format

Taken from `.gturbo`, with two differences motivated by MXFP4:

```
gpt-oss-120b.hydra/
  manifest.json          architecture, sizes, SHA-256, format version
  verified-install.json  verification receipt
  resident.bin           attention, routers, norms, lm_head, sinks
  embed.bin              embed_tokens, separate because mapped and not resident (§2.2c)
  experts/
    layout.json          sub-tensor offsets within a blob
    layer_00.bin ... layer_35.bin
  tokenizer/
```

Each `layer_XX.bin` holds E blobs at a **fixed stride, aligned on 16 KiB**. A blob groups
`gate_up_blocks`, `gate_up_scales`, `gate_up_bias`, `down_blocks`, `down_scales`,
`down_bias` in that order, each sub-tensor aligned on 64 bytes.

Two lessons from TurboFieldfare, built in from the start:
- **sub-tensor alignment governs load width in the shaders.** TF had a bug where a 32-bit
  path passed tests at offset zero and then produced garbage in real decoding, because live
  offsets were only 2-byte aligned. We align on 64 bytes **and** kernel tests use realistic
  offsets, never zero.
- **the repack never dequantizes.** It copies the MXFP4 bytes as they are. The manifest
  records the format; any unknown quantization metadata is rejected.

### 5.3 Expert cache

Since the hit rate is the only lever (§2.2a), the cache deserves more than a uniform LFU:

- **LFU with recency as a tiebreaker**, like TurboFieldfare (measured better than LRU: 72.6 →
  64.8 ms/token).
- **Non-uniform slot allocation across layers.** Not all layers have the same routing
  entropy. Distributing the ~1000 slots in proportion to measured routing concentration,
  rather than equally, is a cheap experiment and potentially the best gain available. To
  validate on the 20B trace.
- **Explicit `pread`, not `mmap`.** TF measured 0.50 tok/s with `mmap` against 3.97 with
  parallel `pread`. The question is settled; we do not replay it.
- **4 to 8 parallel reads**, the measured optimum on this machine (§1).
- **`F_RDADVISE` off by default**, TF showed the gain is not reproducible.
- **`F_NOCACHE` on expert files: to be measured, not decided now.** With 12 GiB of
  application cache, letting macOS keep one more copy is a waste… or a useful second-level
  cache. Our measurements show 5.5 GB/s without cache against 18 GB/s with. A priority
  experiment.

### 5.4 Execution pipeline (decoding, GPT-OSS)

```
cb1 : input norm, QKV, RoPE+YaRN, KV write, attention (with sinks),
      O projection, residual, post-attention norm, router -> top-4
CPU : LFU plan, bounded parallel preads for the missing slots
      (experts already cached start immediately)
cb2 : top-4 MoE in persistent workgroups, weighted reduction, residual
```

**Persistent workgroups** are taken from TF without argument: it is their largest measured
kernel gain (MoE phase 239 → 60 ms, throughput +51 %). The cooperative SIMD kernel is a
documented trap (230 → 527 ms), we do not attempt it.

GPT-OSS-specific details to implement, absent from Gemma 4:
- **attention sinks**: one learned logit per head, added to the softmax denominator;
- **RoPE + YaRN** (`factor` 32, base 4096 → 131072, `beta_fast` 32, `beta_slow` 1);
- **GQA group 8** (64 Q heads, 8 KV heads);
- **SwiGLU with `swiglu_limit = 7.0`**, the exact variant is to be copied from OpenAI's
  reference implementation, not guessed;
- **biases everywhere**: `attention_bias = true`, and the experts have BF16 biases;
- **SWA(128)/full alternation** starting at layer 0.

### 5.5 Harmony

`openai/harmony` is a **Rust** library with PyO3 bindings. Three options:

1. **Reimplement in Swift.** Full control, zero dependency, but the format is rich (roles,
   `analysis`/`commentary`/`final` channels, tool calls, effort levels) and a silent
   divergence degrades outputs with no visible error.
2. **Statically link the Rust crate** through a C facade. Guaranteed fidelity, but it
   introduces Rust into the build chain and the application signature.
3. **Reimplement in Swift, but with a conformance harness**: a corpus of reference
   conversations rendered by the official library, frozen as fixtures, and a test requiring
   byte-for-byte equality of the rendering and equivalence of the parsing.

**I recommend option 3.** It gives the fidelity of option 2 without the dependency, and it
turns "is our Harmony correct" into a testable question. That is consistent with the
"correctness before performance" requirement.

---

## 6. The genericity / performance trade-off

You asked for explicit options before deciding. Here they are.

**Option A, a runtime specialized per model (TurboFieldfare's choice).**
One engine per family, invariants hard-coded. Maximum performance. Adding a model means
writing an engine. That is exactly what you refuse.

**Option B, generic kernels parameterized at runtime.**
Dimensions arrive through `constant` buffers. A single kernel set covers every model. Cost:
loop bounds are no longer known to the compiler, unrolling and register allocation degrade.
On memory-bound kernels like ours I estimate the loss at **10-25 %**, but that is an
estimate, not a measurement.

**Option C, declarative contract + pipeline specialization at build time. ⟵ recommended**

The model is described by a `ModelContract` value (layers, attention pattern, MoE topology,
quantization format, conversation template). That description feeds **Metal
`function_constant`s**, resolved **at pipeline-build time**, once when the model is loaded.

The key point: `hidden_size`, `head_dim`, the quantization block size and the top-k become
**compile-time constants** in the shader. The compiler unrolls and allocates exactly as if
they had been written literally. **The GPU code is as specialized as in option A; only the
Swift orchestration is generic.**

What it costs honestly:
- building the pipelines at load time (a few hundred ms, once);
- genericity stops at **operator families**. A new quantization format or a new sequence
  operator (Qwen's Gated DeltaNet, precisely) requires genuinely new code. No abstraction
  avoids that, and claiming otherwise would be dishonest.

So: `ModelContract` absorbs **shape variation** for free, and makes **operator variation**
explicit, which remains work.

**And I agree with your prioritization: we write `ModelContract` only in phase 3**, once the
20B and the 120B work. Two real models give the right axes of variation; zero models give a
speculative abstraction.

### On the dual execution path (resident dense / streaming MoE)

You ask for a reasoned opinion. **I recommend against it.**

A dense path has no expert cache, no router, no streaming, and none of the `cb1`/io/`cb2`
pipeline. It is a second runtime, not a variant. And its added value is small: running a
dense Qwen3.6-27B in Q4_K_M on 24 GiB is precisely what llama.cpp, MLX and LM Studio already
do very well. Hydra would add nothing there while doubling its maintenance surface and
diluting the one thing that sets it apart.

**My proposal: Hydra owns being a streaming MoE engine, and documents that as a deliberate
limit.** If you want the dense 27B, MLX does it well today.

---

## 7. Phase plan

Every milestone has an objective criterion. A missed milestone triggers a decision, not a
workaround.

### Phase 1, Full chain on GPT-OSS 20B, everything resident

No streaming. We validate the repack, the MXFP4 kernels, attention, Harmony, the minimal UI.

| Milestone | Validation criterion |
| --- | --- |
| 1.1 `.hydra` repacker | **✔ MET.** 12.82 GiB installed in 778 s (44 MB/s peak). Process footprint **50.9 MiB**, largest network block 3.6 MiB, 0.39 % of the checkpoint. 200 windows re-compared against the upstream source: all match. Resume and atomicity tested. |
| 1.2 MXFP4 dequantization | **✔ MET.** **Bit-for-bit** agreement with an independent reference implementation, on a vector covering all 16 E2M1 values and the extreme exponents. |
| 1.3 Attention + sinks + YaRN | **✔ MET for the isolated operators.** RMSNorm, RoPE+YaRN, SwiGLU, attention with sinks and the router agree to **1e-12** between `HydraReference` and an independent transcription of OpenAI's code, and to **< 1e-4** between the Metal kernels and `HydraReference`. The full layer remains to be assembled. |
| 1.4 Full forward | Greedy decoding, 64 tokens, **identical sequence** to the reference on 3 prompts. |
| 1.5 Harmony | Rendering **identical byte for byte** to the official library's fixtures. |
| 1.6 Throughput | **≥ 10 tok/s** decoding, 20B, 4k context. |
| **1.7 Routing trace** | **Expert distribution recorded over ≥ 20,000 tokens; hit/slots curve produced.** |

> **GO/NO-GO decision point.** If 1.7 extrapolates to less than 25 % hit at 27 slots for the
> 120B, we stop and reassess before committing to phase 2.

### Phase 2, Streaming and GPT-OSS 120B

| Milestone | Criterion |
| --- | --- |
| 2.1 Expert streamer | Slots + LFU + parallel preads. Sustained throughput ≥ 5.0 GB/s, measured under real conditions. |
| 2.2 120B repack | 60.77 GiB installed, heap peak < 8 MiB, resume after interruption verified. |
| 2.3 120B correctness | Logits agreement against the reference on a short prompt. |
| 2.4 Budget held | RSS + Metal working set **< measured ceiling**, over a 512-token generation at 8k. |
| 2.5 120B throughput | **≥ 3.0 tok/s** at 8k. Below that we open the "asynchronous mode" discussion. |
| 2.6 Cache tuning | Non-uniform slots, `F_NOCACHE`, overlap granularity, measured, not assumed. |

### Phase 3, `ModelContract` and generalization

Extract the abstraction **from** the two engines that work. Criterion: the 20B and the 120B
both run through the contract, **with no throughput regression above 3 %** (measured
alternating control/candidate).

### Phase 4, Third model

Target to be decided (see questions). Criterion: correct installation and generation, budget
held.

### Phase 5, Finishing

CLI, optional local server, UI telemetry (tok/s, memory, hit rate), distribution.

---

## 8. What I recommend not doing

Drawn from TurboFieldfare's 102 experiments, no point paying for those mistakes twice:

- `mmap` for the expert pool (0.50 vs 3.97 tok/s);
- cooperative SIMD MoE kernel (230 → 527 ms);
- speculative inter-layer prefetching (7 % predictability);
- `F_RDADVISE` by default (non-reproducible gains);
- 4-bit quantized KV cache (loses its memory advantage beyond 4k **and** fails on quality);
- monolithic post-attention/pre-FFN fusion (2.756 → 1.811 tok/s);
- reusing Metal argument buffers (−9 % on long prefill);
- optimizing a kernel that accounts for < 1 % of the token step.

And one methodological rule I propose adopting as is: **a candidate becomes the default if it
shows a reproducible end-to-end gain; if it is within the noise, the default does not
change.**

---

## 9. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Real hit rate < 25 % | 120B at ~2.5 tok/s | Measured in phase 1.7, before any commitment |
| No shared expert costs more than expected | −10 to 20 % | Accepted and quantified; we optimize hit rate, not overlap |
| **Storage: 126 GiB for ~91 GiB of models** | Blocking in phase 4 | Storage policy to decide **now** (question 2) |
| SwiGLU/YaRN/sinks subtly wrong | Degraded outputs with no error | Per-layer numerical comparison from 1.3 |
| Divergent Harmony | Model unusable, silently | Official fixtures, byte-for-byte equality |
| Gated DeltaNet (Qwen) | Phase 4 × 2-3 in duration | Trade-off to make now (question 1) |
| apple9 without TensorOps | Slow attention at long context | Accepted; priority context to decide (question 3) |

---

## 10. Open questions

Addressed in the accompanying message. The six blocking ones:

1. **The third model**, Qwen3.6-35B-A3B with its Gated DeltaNet, or a cheaper target?
2. **Storage**, 126 GiB free, ~91 GiB of models. What policy?
3. **The priority context**, 8k, 32k or 128k? It drives the expert cache.
4. **The Metal ceiling**, offer `iogpu.wired_limit_mb` in the application, or stay at the
   default?
5. **If the 120B tops out around 4 tok/s**, accept an openly asynchronous mode?
6. **Minimum macOS and distribution**, macOS 26 and build from source, or wider?

And the non-blocking ones, which can wait for phase 5: CLI, OpenAI-compatible server, tool
calling.

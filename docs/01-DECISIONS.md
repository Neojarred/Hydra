# Hydra — Decisions on record

Every entry is dated, argued, and states what would have to be observed to overturn it.

---

## D-001 — Hydra is a proof of concept; throughput is not an acceptance criterion
**2026-08-05 — agreed**

The goal is to **show that a model which does not fit in memory can run there anyway**. The
3.4–4.4 tok/s projected for GPT-OSS 120B on an M4 with 24 GiB is accepted as a result, not endured
as a failure.

**Direct consequences for the phase plan:**
- Throughput thresholds disappear from the GO/NO-GO criteria. Milestone 2.5 becomes *"produces
  correct output while staying inside the memory budget"*, and throughput is a **published
  measurement**, not a gate.
- The decision point after the routing trace (milestone 1.7) is **removed as a blocker**. The
  hit-rate/slots curve is still produced — it sizes the cache and predicts target machines — but a
  low hit rate no longer suspends the project.
- The priority order becomes unambiguous: **memory invariant > correctness > throughput.**

---

## D-022 — Gemma 4 operator semantics that cannot be guessed
**2026-08-06 — verified against `transformers/models/gemma4/modeling_gemma4.py`**

Same exercise as D-014 and D-011, and it found more traps. Every item below was read in the
official source, not inferred. Getting any of them wrong yields a model that produces plausible
degraded text with no error raised.

**RMSNorm — the weight is `w`, not `1 + w`.**
```python
normed = hidden.float() * torch.pow(hidden.pow(2).mean(-1, keepdim=True) + eps, -0.5)
normed = normed * self.weight.float()
```
Gemma 3 used `(1 + weight)`. Gemma 4 does not. Carrying the old habit over is a silent 2× on every
normalized activation. `eps` goes **inside** before the power, as in GPT-OSS.

**Some norms carry no weight at all.** `v_norm` and `router.norm` are built with
`with_scale=False`, so they have no tensor in the checkpoint. `v_norm` in particular is an
operation with **nothing in the weight index to hint that it exists**.

**Embeddings are scaled by `sqrt(hidden_size)`** — `embed_scale=config.hidden_size**0.5`, i.e.
≈ 53.07 for 2816. Omitting it mis-scales the entire forward pass.

**Attention scaling is `1.0`, not `1/sqrt(head_dim)`.** `self.scaling = 1.0`, flatly. The query
norm absorbs it. This is the single easiest thing to get wrong by habit.

**Order around RoPE.** `q_norm` and `k_norm` are applied **before** RoPE; `v_norm` after the
projection, and **V never goes through RoPE**.

**`attention_k_eq_v` applies to full-attention layers only** (`and not self.is_sliding`), and it
does not mean K and V are equal downstream:
```python
value_states = key_states            # the raw k_proj output, before k_norm
key_states   = rope(k_norm(key_states))
value_states = v_norm(value_states)
```
Same projection, two different post-processings.

**Two head geometries in one model.** Sliding layers: 16 heads × `head_dim` 256, 8 KV heads. Full
layers: 16 heads × `global_head_dim` 512, 2 KV heads. **`q_proj` therefore has a different shape
per layer type** — 4096 rows on sliding layers, 8192 on full ones. `HydraLayout` sizes tensors from
a single config today.

**RoPE splits into halves**, as GPT-OSS does — `emb = cat((freqs, freqs))`. That convention carries
over unchanged.

**Partial rotation is expressible as zero frequencies.** Full layers use `rope_type: "proportional"`
with `partial_rotary_factor: 0.25`:
```python
rope_angles = int(0.25 * 512 // 2)            # 64
inv_freq = cat(inv_freq_rotated,              # 64 real frequencies
               zeros(512 // 2 - 64))          # 192 zeros
```
A zero frequency gives `cos = 1, sin = 0` — the identity. **The kernel needs no change**: only
`RoPETables` must emit the zero-padded frequencies. That is the one piece of good news here.

**The router is not GPT-OSS's.**
```python
h = router.norm(h)                     # no weight
h = h * router.scale * hidden_size**-0.5
probs = softmax(proj(h), dim=-1, dtype=float32)   # over ALL experts
w, idx = topk(probs, k=8)
w = w / w.sum(-1, keepdim=True)                    # renormalized
w = w * per_expert_scale[idx]
```
GPT-OSS softmaxes over the **top-k only**. Gemma softmaxes over all 128, then renormalizes. The
two give different weights for the same logits.

**The layer topology differs, not just its constants.** The MoE branch reads the **residual** — the
state *before* the dense MLP — not the MLP's output:
```python
residual = hidden
hidden   = mlp(pre_feedforward_layernorm(hidden))
h1 = post_feedforward_layernorm_1(hidden)                     # dense branch
h2 = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(residual), …))
hidden = post_feedforward_layernorm(h1 + h2)
hidden = residual + hidden
hidden = hidden * layer_scalar
```
Two parallel branches over the same input, summed. And `post_attention_layernorm` is applied
**before** the residual add — post-norm, where GPT-OSS is pre-norm.

**`final_logit_softcapping`**: `logits = 30 * tanh(logits / 30)`.

**The MLP is plain.** `down(gelu_pytorch_tanh(gate(x)) * up(x))` — no clamping, no `+1`, none of
GPT-OSS's SwiGLU quirks (D-014). Simpler, and the difference must not be papered over by reusing
the existing kernel.

**Disabled for this checkpoint, but present in the code:** `hidden_size_per_layer_input = 0` and
`num_kv_shared_layers = 0`, so the per-layer-input branch and KV sharing are both inert. They must
be **rejected explicitly** if a future checkpoint enables them, not silently ignored.

**Still unverified:** the exact `embed_scale` at runtime for the QAT checkpoints, and the audio
tower's presence in the weight index (the model ships vision *and* audio towers; both must be
excluded from the repack plan).

---

## D-021 — Gemma 4 26B-A4B, in full precision
**2026-08-06 — agreed**

The third model is **Gemma 4 26B-A4B, installed exactly as Google published it: BF16**, not from
the QAT q4_0 checkpoints that also exist.

**Why full precision.** It is D-015 stated on a model that was not chosen to flatter it. Everyone
ships Gemma 4 at Q4; running it undegraded on 24 GiB is the project's thesis, and it makes the one
comparison that cannot be argued with — **the same model, against TurboFieldfare's published Q4
numbers**, with the quality difference on our side and the cost visible.

It also fills the gap in the catalogue: 20B, **26B**, 120B.

**Accepted cost.** The experts are 45.7 GB and 2.86 GB are read per token — 2.25× GPT-OSS 20B.
Expect roughly **2–2.5 tok/s** on the M4, in the 120B's range. D-001 already settled that this is a
result, not a failure.

**Blocked on storage, not on code.** ~50 GB installed against **28 GB free** with both GPT-OSS
models present. Dropping the vision tower saves only ~1.5 GB. Something has to give: remove the
120B (it is re-downloadable), install to an external volume via `hydra install <model> <dir>`, or
defer.

### What the audit found, and what D-018 got wrong

D-018 called Gemma "the cheapest, no new kernel family". **That was wrong.** The fused per-layer
expert tensors do match what `ScatterCopy` already splits, but the rest does not:

- **two attention geometries in one model** — sliding layers 16 heads × 256, full layers
  `global_head_dim 512` with 2 KV heads;
- **`attention_k_eq_v: true`** — K and V are one tensor;
- **a 5:1 sliding/full pattern**, where `KVCache` keys off `layer % 2`;
- **two RoPE configurations** — sliding θ=10 000; full θ=1e6 with `partial_rotary_factor 0.25`,
  so RoPE covers a quarter of the head dimension;
- **q_norm / k_norm** before attention;
- **`final_logit_softcapping` 30.0**;
- **tied embeddings** — no `lm_head`; the 1.48 GB embedding *is* the head and must be resident,
  where GPT-OSS maps it outside the working set;
- **five feed-forward norms per layer**, plus `layer_scalar`, `router.scale`,
  `router.per_expert_scale`;
- **gelu_pytorch_tanh**, not GPT-OSS's clamped SwiGLU;
- a **27-layer vision tower** to exclude from the plan.

Five of those touch kernels. Still less than Qwen's Gated DeltaNet, so the D-018 order stands — but
the estimate that went with it did not.

**One structural gain:** the always-active dense MLP beside the experts (1.07 GB resident) is the
I/O-overlap mechanism `00-FEASIBILITY.md` records GPT-OSS as lacking.

---

## D-020 — Variable precision by role: measured, built, abandoned
**2026-08-06 — closed. Implemented, measured (M-027), reverted.**

> **Outcome.** Q8 on the dense weights was built end to end and removed. The quality
> gate passed (M-026), and then the point collapsed: the dense weights are 66 % of the
> **bytes** per token but **13 % of the time**, so halving them bought **3.5 %**. The
> memory freed could not be spent either — more expert slots measure *slower*, not
> faster (M-027). D-015 stands, now for two reasons rather than one.
>
> The reasoning below is kept because it was sound and the conclusion was not: it
> reasoned in bytes and concluded in seconds. That step is the mistake to remember.

Not one precision for the whole model, but **one precision per role**. What is at stake is
bandwidth, not disk: decoding reads the dense weights on **every** token.

Per-token budget of the 20B (`tools/budget.py`, 3.708 GB total):

| role | bytes | share | format today |
|---|---|---|---|
| attention + routers + norms | 1.279 GB | 35 % | BF16 |
| LM head | 1.158 GB | 31 % | BF16 |
| 4 experts × 24 layers | 1.271 GB | 34 % | MXFP4 |

**The dense part is 66 % of the traffic.** The experts — the point of the project — are 34 %.

**Frozen, not up for measurement.** Routers and norms stay BF16. A router is 184 KB per layer on
the 20B: there is nothing to win. And a routing error is not a small perturbation — it is
**discrete**: the token goes to the wrong expert entirely, and nothing downstream recovers it.

**Worth measuring.** BF16 → Q8 on the LM head and on attention. Removes 1.219 GB per token, 33 %
of the traffic. Weight-only 8-bit is the case where the literature is least ambiguous, and D-015
explicitly leaves that door open.

**Still refused without proof: 4-bit on the dense.** The symmetry *"the experts are already 4-bit,
so the rest can be"* is **false**. GPT-OSS's experts are MXFP4 because OpenAI trained them that
way — quantization-aware. Post-training 4-bit applied to weights trained in BF16 is a different
and worse operation. Confusing the two would degrade the model precisely as D-015 forbids.

**What decides.** Not an opinion, a measurement, on a fixed prompt corpus:
- KL divergence of the logit distribution against the BF16 reference;
- top-1 agreement rate under greedy decoding, which must stay at 100 % or explain each departure.

The machinery already exists: `HydraReference` in double precision, and the exact-equivalence
harness written for speculative decoding.

**Adopt if** the two criteria hold and no reasoning regression shows up on long chains of thought —
the case where errors compound, and the one GPT-OSS exercises most. **Abandon otherwise**, and the
36 % of throughput stays unclaimed. That is D-015's arbitration, unchanged.

---

## D-019 — Precision variants are produced by the repacker, not downloaded
**2026-08-06 — moot, kept for the next model that offers variants (see D-020)**

Offering Q8 and Q4 variants of a model means **producing them ourselves during installation**, from
the same upstream safetensors — not ingesting community GGUFs.

For GPT-OSS the question does not even arise: OpenAI publishes MXFP4 experts and BF16 dense, and
nothing else. Any other precision is **our artifact**, and must be labelled as such.

**Why the repacker rather than GGUF.** It already performs a streaming transformation under a plan
verified before a single byte is downloaded (`RepackPlan.validate`). Quantizing inside a
`ScatterCopy` extends a mechanism that exists. Reading GGUF would mean a second format reader, with
its own quantization conventions to verify from scratch — the kind of undocumented detail D-014 and
D-011 exist to guard against. One download path, N precision outputs.

**Consequence to absorb.** `InstallationVerifier` compares installed bytes against the upstream
checkpoint. That check is **void for any requantized group**: the bytes are legitimately different.
Those groups need their own verification — dequantize, and compare against the source within a
stated tolerance. The manifest must therefore record the precision of every tensor group, and
verification must branch on it. Shipping requantization without that branch would silently lose
the guarantee that makes installation trustworthy.

---

## D-018 — Two models before any platform decision
**2026-08-06 — agreed**

Order of work: **Gemma 4 26B-A4B, then Qwen3.6-35B-A3B.** Once both run, four models work
(GPT-OSS 20B and 120B, plus those two), and only then do we choose between Qwen3.5 122B-A10B,
DeepSeek V4 Flash, and the Linux/Windows port — on evidence, not in advance.

**Why models before the port.** The project rests on having an oracle: *a slow, obviously correct
implementation against which a fast and subtle one is measured*. Porting first means rewriting
HydraMetal against **one** model, where every divergence is ambiguous — wrong kernel, or wrong model
wiring? Adding models first turns each one into a **test vector for the port**: a disagreement
localized on a single model says where to look.

`HydraReference` depends on no platform (`Package.swift`). The oracle travels with the port without
a line of change. It is the repository's most valuable asset for that work, and it is already
written.

**Why Gemma first — it is the cheapest, not the hardest.** Its alternating local-sliding-window /
global attention is the pattern `KVCache` already implements. **No new kernel family.** The real
work is the SentencePiece tokenizer (262 k, where `HydraTokenize` is o200k BPE), the chat template,
Gemma's norm placement, and a **shared expert** path — one pinned slot per layer, which incidentally
gives Hydra the I/O-overlap mechanism `00-FEASIBILITY.md` notes GPT-OSS lacks.

**Why Qwen second.** Gated DeltaNet covers 30 of its 40 layers and is, per the feasibility study,
*the project's most expensive item*. It is worth attempting once a second model has proven the
`.hydra` format and the repacker are genuinely generic — which one model cannot show.

**Reopen if** the Gemma work uncovers that the format is not generic: that would make the port the
better next step, since it would no longer depend on an assumption the models were meant to confirm.

---

## D-017 — Sampling defaults: departing from OpenAI's recommendation
**2026-08-05 — decided on observation**

OpenAI recommends `temperature = 1.0` and `top_p = 1.0` for GPT-OSS — the raw, untruncated
distribution. **We depart from it**: 0.7 and 0.9 by default.

**What prompted it.** On the prompt "hi", the 20B reasoned as follows: *"User says hi. Need to
respond in Bengali because user asked earlier to translate."* — and answered in Bengali. No such
instruction existed; the model hallucinated an antecedent.

**The template is not at fault.** The rendered Harmony prompt is identical to the published
`chat_template.jinja`, and tests lock that down. The problem is sampling: on a two-token prompt,
the raw distribution of a 20 B model leaves appreciable mass on aberrant continuations, and nothing
in the context rules them out.

**What it remains.** A setting, not a constraint: the slider goes to 1.5, and OpenAI's
recommendation stays reachable. A developer-instructions field is also exposed, which can pin a
reply language — that is the mechanism the format provides for it.

---

## D-016 — Scope of the first application
**2026-08-05 — agreed**

**The memory gauge is the centrepiece.** It shows the gigabytes actually resident against the
model's full weight. Three distinct quantities, never conflated: memory engaged by the process,
mapped weights (reclaimable by the system under pressure), installed size on disk. Announcing a
single flattering number would run against the project's honesty.

**Rejected:** the expert-slot visualiser (spectacular but costly in space for little information)
and the display of hit rate and SSD reads (useful for tuning, not for demonstrating).

**Kept:** persistent multi-session history, reasoning in a disclosure panel, and exposed settings —
context length and expert slot count at load time, temperature and reasoning effort per
conversation. The slot count is exposed deliberately: it is what makes the memory/speed trade-off
tangible.

**Catalogue frozen** to OpenAI's two official models. Accepting an arbitrary Hugging Face
repository would open the door to unsupported architectures, which would have to be detected and
refused cleanly — work unrelated to the goal.

**Installing in the background during a conversation: yes.** It is both the easy path *and* the
correct one: installing engages no runtime, only I/O bounded to ~50 MiB. **One model is loaded for
inference at a time**, however — two concurrent runtimes would double the footprint, against the
whole point.

**Distribution:** build from source, published on GitHub. No notarization for now. The CLI is kept:
it is the project's measurement instrument.

---

## D-015 — Do not degrade the model: the goal is intelligence at a reduced footprint, not a minimal footprint
**2026-08-05 — agreed, refines D-012**

**Refused: quantizing the dense weights.** Moving attention, routers and the LM head from BF16 to
MXFP4 would save 1.67 GiB, or 73 % of the resident floor. It is the largest memory gain available,
and it is **ruled out** if it costs quality.

**What the project demonstrates, exactly.** Not "fit an LLM into little memory" — any aggressive
quantization achieves that, at the price of a dulled and uninteresting model. But: **run a large
*intelligent* model on a small machine without degrading it.**

Expert streaming is precisely what makes the two compatible. The footprint shrinks by limiting how
many experts are **resident**, not by damaging weights. Hydra runs GPT-OSS **in the exact format
OpenAI published**: experts in native MXFP4, attention and LM head in BF16. No requantization, no
loss introduced by us.

**Practical consequence.** The resident floor stays at 2.27 GiB for the 20B and 2.88 GiB for the
120B, and that is accepted. The target of a 3.68 GiB footprint for a 12.82 GiB model is already
met; shaving it at the cost of quality would defeat the purpose.

**Reopen if:** a measurement shows a given quantization has **no measurable effect** on quality.
The burden of proof sits on that side, and it is demanding: not "the numerical difference is
small", but "outputs stay equivalent on a serious evaluation".

**Still allowed:** any optimization that does not touch the weights. Chunked prefill and I/O
overlap qualify — they change scheduling, not values.

---

## D-014 — GPT-OSS operator semantics that cannot be guessed
**2026-08-05 — verified against `gpt_oss/torch/model.py`**

Each of these was read off the reference implementation, not inferred. Getting one wrong raises
**no error**: the model loads, generates plausible text, and comes out degraded. They are pinned by
reference vectors in `tools/gen_reference_fixtures.py`.

**SwiGLU splits on even and odd indices**, not into two halves:
`x_glu, x_linear = x[..., ::2], x[..., 1::2]`. The rows of `gate_up_proj` are therefore
**interleaved** `[gate₀, up₀, gate₁, up₁, …]`. Splitting into halves would give a model that works
but mixes channels.

**RoPE, on the other hand, does split into halves** (`torch.chunk`). Two opposite conventions in
the same architecture — which is exactly what makes the mistake easy.

**SwiGLU clamping is asymmetric**: the gate branch is bounded **from above only** (`min(x, 7)`),
the linear branch on both sides. And the linear branch gets **+1** before the product. The swish
uses `sigmoid(1.702·x)`, not `sigmoid(x)`.

**YaRN applies a concentration** on top of frequency rescaling: `0.1·ln(factor) + 1`, i.e.
**1.3466** for GPT-OSS. It multiplies cos and sin. Omitting it breaks nothing visible but shifts
all attention.

**Attention sinks are one extra logit column** in the softmax, dropped afterwards. They contribute
nothing to the result: they enlarge the denominator, which lets a head look at nothing. In the
kernel this falls out elegantly — the online softmax starts with `max = sink` and `denominator = 1`.

**The router applies its softmax to the top-k logits only**, after selection, not to the full
distribution.

**The sliding window is 128 and applies to even-indexed layers.** The mask `tril(diagonal=-128)`
admits exactly 128 positions.

---

## D-013 — The repacker's memory bound comes from streaming, not from splitting requests
**2026-08-05 — decided on measurement, corrects an initial choice**

**What I had chosen.** Split every source range into 4 MiB sub-requests, so that the memory bound
would be a property of the splitting — trivially checkable — rather than a dependency on
`URLSession`'s buffering behaviour.

**What the measurement said.** On the real repository:

| Pattern | Throughput |
| --- | ---: |
| one 64 MiB `Range` request | **33.5 MB/s** |
| eight 4 MiB `Range` requests in series | **5.2 MB/s** |

A factor of **6.4**. The cause: Hugging Face answers with a **302 to a signed CDN**, and every
request pays the redirect again plus a TLS handshake to another host. Reusing the resolved URL is
impossible — its policy carries a `ByteRange` condition tied to the exact range requested.

**What we do instead.** A **single request per contiguous region** of the source checkpoint, whose
response is **consumed as it streams**: every block the network stack delivers is routed to its
destination and released before the next arrives. Since the plan covers the checkpoint exactly with
no gaps, neighbouring tensors form long regions — the source file is read almost end to end,
sequentially.

**Measured on the real 20B install: 44 MB/s**, or **8.5×** the initial approach.

**What it costs.** The bound is no longer a property of the splitting: it depends on the size of
the blocks `URLSession` delivers (measured up to 3.5 MiB). It is therefore now **checked by tests
and instrumented in production** — the repacker tracks the largest block received and exposes it in
its progress — rather than guaranteed by construction. An accepted trade-off: a bound measured on
every run beats a theoretical bound that divides throughput by six.

---

## D-012 — Minimizing memory is the goal, not filling the available ceiling
**2026-08-05 — agreed, corrects D-001**

The expert cache is **never** sized by "what the hardware allows". It is an **explicit policy**
(`ExpertCachePolicy`), and the default is `.minimal`: one slot per selected expert, i.e.
`top_k = 4` per layer.

**What this corrects.** The feasibility study presented "GPT-OSS 20B fits entirely in memory, so no
streaming needed" as good news. That misreads the project's goal: the 20B must also run at a
reduced footprint, otherwise it demonstrates nothing. It remains usable fully resident, but **as a
correctness reference**, not as a mode of operation.

**What it gives, measured by `hydra budget`:**

| Model | Policy | Footprint | Share of installed model |
| --- | --- | ---: | ---: |
| 20B (12.82 GiB installed) | `.minimal` | **3.77 GiB** | 28 % |
| 20B | `.maximize` (reference) | 12.06 GiB | 93 % |
| 120B (60.77 GiB installed) | `.minimal` | **5.07 GiB** | **8 %** |

**The correctness test that follows**, and the best one in the project: on an identical prompt with
greedy decoding, `.minimal` and `.maximize` must produce **exactly the same token sequence** on the
20B. Cache size is a performance characteristic; it must have no observable effect on outputs. Any
divergence signals an eviction or slot-ownership bug.

**Portability corollary:** nothing in the sizing is specific to the development machine.
`HardwareProfile` is injected, and a test checks that the 20B fits at minimum under a 5 GiB ceiling
— that is, an 8 GiB machine.

**Related finding: the floor is no longer the experts, it is the resident weights.** GPT-OSS keeps
attention, routers and LM head in **unquantized BF16** — 2.27 GiB for the 20B, 2.88 GiB for the
120B, of which 1.08 GiB for the LM head alone. That is why TurboFieldfare reaches ~2 GB total on
Gemma 4 while we bottom out near 3.8 GiB: on their side, those same tensors are 4-bit. **Going
lower would require quantizing the LM head and attention**, which changes outputs — an experiment
to validate against a reference, not a design decision. It is the main remaining lever.

---

## D-002 — Portability is a stated goal, but no abstraction is written in advance
**2026-08-05 — agreed**

Eventual targets mentioned: M3 Ultra / M5 Max short on unified memory, and an **x86_64 + CUDA**
port letting an RTX 5090 (32 GiB of VRAM) run models sized for an RTX Pro 6000.

**What we do now:** nothing speculative. No `ComputeBackend` protocol, no GPU abstraction layer. Per
the brief, we generalize only from code that works.

**What we do forbid ourselves right away**, because it is free and the reverse is expensive to
undo: **the `HydraCore`, `HydraFormat`, `HydraInstall` and `HydraTokenize` modules do not import
Metal.** Format, streaming, caching and tokenization logic stays pure Swift, portable as-is to
Linux and Windows. Only `HydraMetal`, `HydraRuntime` and `HydraApp` are platform-bound. That is a
layering discipline, not an abstraction.

**Technical note for a future CUDA port** — not to be implemented, only kept in mind: a machine
with a discrete GPU offers a **three-level** hierarchy (VRAM / system RAM / SSD) where the Mac has
only two. That is structurally **more favourable**: a 5090 has 32 GiB of VRAM, a hundred-odd GiB of
system RAM usable as a second-level cache, and memory bandwidth an order of magnitude above the
M4's. There the bottleneck returns to the SSD and the PCIe bus, not compute.

---

## D-003 — No third model for now
**2026-08-05 — agreed**

Qwen3.6-35B-A3B is out of the initial scope: its 30 Gated DeltaNet layers represent as much kernel
work as both GPT-OSS models combined.

Scope becomes **GPT-OSS 20B then GPT-OSS 120B**. Phase 4 (third model) is dropped from the plan;
phase 3 (generalization) will be carried out on two real models, and a third target will be chosen
later, with hindsight.

**Reopen if:** phase 3 shows that two models from the same family are not enough to bring out the
right axes of variation for the model contract.

---

## D-004 — Storage managed the way LM Studio does it
**2026-08-05 — agreed**

The user installs and uninstalls models from the interface, knowingly. Zero, one or several models
may coexist.

**The only automatic rule:** Hydra computes the space remaining **after** the contemplated install
and **warns** if it would fall below **10 GB**. A warning, not a block — the user decides.

For each model the interface shows: size on disk, state (installed / partial / absent), current
free space.

---

## D-005 — Context length chosen when the model is loaded
**2026-08-05 — agreed**

Like LM Studio: when loading a model, a dialog offers the context length.

An important technical consequence: **the expert cache slot count is computed at load time**, not
fixed at compile time. The budget is derived at runtime from `recommendedMaxWorkingSetSize`, the
chosen context and the size of the resident weights. The interface shows the resulting slot count
and expected throughput **before** confirming.

Offered values: 4k, 8k, 16k, 32k, 64k, 128k. Default: 32k.

---

## D-006 — Metal ceiling: detect and suggest, never impose
**2026-08-05 — recommendation, uncontested**

Hydra reads `recommendedMaxWorkingSetSize` and computes everything from it. It **detects** whether
`iogpu.wired_limit_mb` has been raised and accounts for it. It **suggests** the command to the
user, documented, with its quantified effect (+5 slots/layer on the 120B) and its risk. It never
runs it itself, and works correctly at the default budget.

---

## D-007 — macOS 26 minimum, Apple Silicon only
**2026-08-05 — recommendation, contestable**

Reasons: it is the version of the development and validation machine; it gives access to Metal 4
and the Metal Performance Primitives for prefill; and supporting earlier versions would mean
validating on hardware we do not have.

The code detects the GPU family at runtime. **This machine is apple9, not apple10**: the TensorOps
path that accelerates long-context attention for TurboFieldfare is not available to us.

---

## D-008 — Harmony reimplemented in Swift, under a conformance harness
**2026-08-05 — recommendation**

The official library is in Rust. Rather than introduce Rust into the build chain, we reimplement in
Swift and freeze a corpus of conversations rendered by the official library as **fixtures**. The
test requires **byte-for-byte** equality of the rendering.

A Harmony divergence degrades outputs **without raising an error**: exactly the kind of bug that
must be made impossible by construction.

---

## D-009 — No dense execution path
**2026-08-05 — recommendation**

Hydra is a streaming MoE engine and owns that as a deliberate limit. A resident dense path would be
a second runtime for a use case already well covered by MLX and llama.cpp, with nothing
differentiating to add.

---

## D-010 — Hugging Face source: the repository root
**2026-08-05 — decided on the audit**

For GPT-OSS, the repacker reads the **root** of the repository (14 shards for the 120B), not
`original/` nor `metal/model.bin`. MXFP4 values there are already split into `blocks` / `scales` /
`bias`, each sub-tensor being a contiguous byte range directly addressable by an HTTP `Range`
request.

---

## D-011 — MXFP4 format details verified against the reference implementation
**2026-08-05 — verified**

Locked against `openai/gpt-oss` (`gpt_oss/torch/weights.py`):

- E2M1 table: `[+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]`;
- **low nibble → even index, high nibble → odd index** within each `uint8`;
- E8M0 scale: `value = fp4 * 2^(scale_byte - 127)`, applied with `ldexp`;
- block of **32 values** along the last dimension: 16 packed bytes + 1 scale byte.

Getting the nibble order wrong produces a model that generates plausible but degraded text, with no
error — hence verifying up front rather than while debugging.

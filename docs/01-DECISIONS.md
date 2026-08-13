# Hydra, Decisions on record

Every entry is dated, argued, and states what would have to be observed to overturn it. A
decision that was later reversed keeps its original entry and gains the reversal underneath,
so the reasoning that led to the wrong call stays readable.

Newest first. The one that defines the project is D-015: never quantize the dense weights, even
when it would be faster.

---

## D-001, Hydra is a proof of concept; throughput is not an acceptance criterion
**2026-08-05, agreed**

The goal is to **show that a model which does not fit in memory can run there anyway**. The
3.4–4.4 tok/s projected for GPT-OSS 120B on an M4 with 24 GiB is accepted as a result, not endured
as a failure.

**Direct consequences for the phase plan:**
- Throughput thresholds disappear from the GO/NO-GO criteria. Milestone 2.5 becomes *"produces
  correct output while staying inside the memory budget"*, and throughput is a **published
  measurement**, not a gate.
- The decision point after the routing trace (milestone 1.7) is **removed as a blocker**. The
  hit-rate/slots curve is still produced, it sizes the cache and predicts target machines, but a
  low hit rate no longer suspends the project.
- The priority order becomes unambiguous: **memory invariant > correctness > throughput.**

---

## D-027, Gated DeltaNet semantics, transcribed rather than inferred
**2026-08-12, from `transformers/models/qwen3_5/modeling_qwen3_5.py`**

D-022 exists because Gemma's operator semantics could not be guessed and two of them were got
wrong by guessing. This is the same document for Qwen's linear attention layers, written before
any kernel, from the reference implementation and not from the paper.

### The recurrence

Per token, per **value** head. The state `S` is `[linear_key_head_dim, linear_value_head_dim]`,
128 by 128, held in float32 (`mamba_ssm_dtype`).

```
q = l2norm(q, eps = 1e-6) * (1 / sqrt(linear_key_head_dim))
k = l2norm(k, eps = 1e-6)
g = exp( -exp(A_log) * softplus(a + dt_bias) )     # scalar per head
beta = sigmoid(b)                                   # scalar per head

S      = S * g                                      # decay first
kv_mem = kᵀ · S                                     # [v_dim], contract over the key dimension
delta  = (v - kv_mem) * beta
S      = S + k ⊗ delta                              # outer product
out    = qᵀ · S                                     # [v_dim]
```

Five things here cannot be guessed and each is a plausible way to be wrong:

1. **`l2norm` is applied inside the kernel, to q and k only**, with `eps = 1e-6`, as
   `rsqrt(sum(x²) + eps) · x`. Not RMS norm: there is no mean and no learned weight.
2. **The scale is `1/sqrt(key_head_dim)` and it multiplies the query *after* the l2 norm.**
   Applying it before, or using the value dimension, changes every output.
3. **The decay multiplies the state before the read.** `kv_mem` is taken from the decayed
   state, not the previous one, so the order of those two lines is load-bearing.
4. **`g` is `exp` of a negative quantity built through `softplus`**, so it lies in (0, 1]. The
   parameters are `A_log` and `dt_bias`, one of each per value head; `g` is not a projection
   output on its own.
5. **The whole recurrence runs in float32**, including the state, whatever the checkpoint's
   storage dtype.

### Around it

A depthwise **causal** convolution of kernel 4 with a SiLU activation runs over the
concatenated q, k and v before the split, so `conv_dim` is
`2 · (16 · 128) + 32 · 128 = 8192`. Padding is `kernel - 1` on the left, then sliced back to
the sequence length. Its rolling window is per-layer state exactly as the recurrent state is,
and it must be carried between tokens for the same reason.

The output is normalized by a **gated RMS norm** whose gate is a separate projection `z`:

```
out = weight · (out · rsqrt(mean(out²) + eps))
out = out · silu(z)
```

Note the order: the learned weight is applied to the normalized value, and the gate multiplies
afterwards. The gate is not inside the norm.

### The attending layers, which are not Gemma's

Three things, from the same source.

**The output gate is packed per head.** `q_proj` emits `heads · headDim · 2` values, viewed as
`[heads][headDim · 2]` and chunked, so each head's slice holds its own query followed by its
own gate. Splitting the tensor down the middle instead hands head `h` the gate of a different
head: finite, plausible, and wrong for every head but the first.

**The gate multiplies the attention output, not the query.** `attn_output · sigmoid(gate)`,
after attention rather than before it. It decides how much of what attention returned survives,
which is a different operation from scaling what it attends with.

**q_norm and k_norm come before the rotary**, as Gemma's do.

### Interleaved mRoPE is ordinary RoPE, for text

`apply_interleaved_mrope` starts from the temporal frequencies and overwrites them with the
height component at indices 1, 4, 7 … and the width component at 2, 5, 8 …, each bounded by its
`mrope_section` times three. The sections are `[11, 11, 10]`, summing to 32 pairs, which is 64
components: exactly the quarter of a 256-wide head that `partial_rotary_factor 0.25` rotates.

A text token's three position components are the same number, so every one of those writes
copies a value onto itself and the result is the rotary this project already has. **No new
rotary kernel is needed until images arrive**, and the existing partial-rotary handling covers
the rest. That is asserted in the tests rather than believed, because it is the assumption that
would quietly stop being true the day the vision path is switched on.

### Shapes, and the head asymmetry

`linear_num_key_heads` is 16 and `linear_num_value_heads` is 32, both of dimension 128. The
state and the recurrence are per **value** head, so the key and query heads are shared two ways,
the same grouped-query arrangement the full-attention layers use, applied to a recurrence.

Per linear layer, resident: `in_proj_qkv` 2048 to 8192, `in_proj_z` 2048 to 4096, `in_proj_b`
and `in_proj_a` 2048 to 32 each, `out_proj` 4096 to 2048, a depthwise conv weight of 8192 by 4,
`A_log` and `dt_bias` of 32, and a norm weight of 128.

This is more than D-026 assumed when it estimated resident bytes: that estimate treated the
linear layers as four square projections and they are not. The figure there is low, and the
correction belongs with the first real measurement rather than with another guess.

### The tokenizer, read from its own file

`tokenizer.json` is stored in LFS and a raw fetch returns the pointer, so this was inferred
until the file was actually downloaded. Byte-level BPE, as expected from the family, and
**three differences from GPT-OSS's**, any one of which produces a token sequence the model has
never seen from a vocabulary that looks familiar enough to borrow:

| | GPT-OSS | Qwen |
| --- | --- | --- |
| `ignore_merges` | true | **false** |
| pre-tokenizer | o200k's pattern | the GPT-2 lineage pattern |
| normalizer | none | **NFC** |

The pre-tokenizers differ in ways that matter on ordinary text: GPT-OSS splits digit runs at
three and treats case boundaries specially, where Qwen takes digits one at a time and letters in
a single run.

The NFC normalizer needed a new field on `Conventions`, which had no notion of one because
neither existing model declares any. It is a no-op on ASCII, which is precisely why it would
have gone unnoticed: the text that distinguishes them is accented or CJK, where a decomposed
sequence and a composed one are different strings and therefore different tokens.

### The prompt format

From `chat_template.jinja`. Turns are `<|im_start|>role\n … <|im_end|>\n`, reasoning is a
`<think>` block inside the assistant turn, and history is replayed **without** it.

Two details worth the same care as the operator semantics. **The generation prompt opens the
thinking block itself**, `<|im_start|>assistant\n<think>\n`, exactly as Gemma's opens its
thought channel, so the opening tag never reaches the parser and one that waits to see it files
the whole answer as reasoning. And **thinking is a switch, not a level**: "off" is expressed by
pre-filling an *empty* block rather than by omitting the tags, so they stay well formed. There
is no equivalent of GPT-OSS's low, medium and high, and offering one would be inventing
behaviour the checkpoint does not have.

### The layer, and the block around the token mixer

Plain pre-norm, and the same shape for both kinds of layer. The token mixer is the only thing
that differs:

```
residual = x
x = input_layernorm(x)
x = linear_attn(x)  or  self_attn(x)
x = residual + x

residual = x
x = post_attention_layernorm(x)
x = moe(x)
x = residual + x
```

Two norms a layer, not Gemma's four, and **the feed-forward branch reads the attention output
rather than the residual**. Gemma runs its dense and expert branches in parallel over the same
input and sums them (D-022); this is the ordinary sequential arrangement, and carrying Gemma's
habit over would feed the expert branch the wrong tensor.

### The mixture, and the shared expert's own gate

```
shared = shared_expert(h)                       # SwiGLU, silu
routed = experts(h, selected, routing_weights)
shared = sigmoid(shared_expert_gate(h)) * shared
out    = routed + shared
```

The shared expert is **always active** and is scaled by a sigmoid of its own projection, which
is a single row: `shared_expert_gate` maps the hidden state to one number a token. That gate is
8-bit in both published builds, alongside the router.

This is the structural gain D-026 hoped for. The shared branch does not wait on the SSD, so it
is work the GPU can do while the routed experts are read, exactly as Gemma's dense MLP is.

### The router, now read rather than guessed

`Qwen3NextTopKRouter.forward`, from the source rather than the summary:

```
router_probs = softmax(router_logits, dim=-1)         # over every expert
top_value, indices = topk(router_probs, top_k)
if norm_topk_prob: top_value /= top_value.sum(-1)     # renormalized
```

`norm_topk_prob` is absent from the published `config.json`, so it takes the class default,
which is `True`. **This is exactly Gemma's convention**, and `gemma_router_topk` implements it
already: softmax over all experts, take the top-k, renormalize. GPT-OSS softmaxes over the
top-k alone and would have been the wrong choice.

It is worth saying that the guess would have been right. That is not the point: the two
conventions differ by a normalization no error surfaces, and reusing a kernel because it looks
right is how a model ends up plausibly worse. The reason to reuse it now is that the source
says so.

### What this means for prefill

There are two reference paths, `torch_recurrent_gated_delta_rule` for a single token and
`torch_chunk_gated_delta_rule` for a sequence, and the second exists because the first is
sequential. The chunked form computes decay masks by `cumsum` and `exp` and updates the state
once per chunk rather than once per token. **That is the form prompt processing needs**, and it
is a second kernel, not a loop around the first.

---

## D-026, Qwen3.6-35B-A3B audit: the checkpoint format is free, the attention is not
**2026-08-12, audited against the published `config.json` of the official release and of both
MLX quantizations**

### The sources

The official model is `Qwen/Qwen3.6-35B-A3B`, Apache 2.0, released 2026-04-16, published in
BF16. Its `architectures` field reads `Qwen3_5MoeForConditionalGeneration` and its `model_type`
is `qwen3_5_moe`: the 3.6 release reuses the 3.5 architecture class, so anything written
against the "3.5 MoE" implementation applies.

**There is no official MLX release.** The two candidates are both community conversions of that
BF16 checkpoint:

| repository | bits | safetensors | note |
| --- | ---: | ---: | --- |
| `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` | 4 | 20.4 GB | LM Studio's curation |
| `lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit` | 8 | 37.7 GB | same |
| `mlx-community/Qwen3.6-35B-A3B-4bit` / `-8bit` | 4 / 8 | , | converted with mlx-vlm 0.4.4 |
| `mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit` | mixed | , | per-layer widths from a KL sensitivity pass |

This is the same provenance question as D-024 and it should be stated the same way: these are
community builds, not the publisher's. Saying otherwise once already cost us a correction.

### The format costs nothing

Both MLX builds are **affine, group 64**, with per-tensor overrides carried in the config, which
is byte for byte the format `MLXAffineLayout` and `mlx_affine_gemv`/`_gemm` already decode. The
overrides differ from Gemma's (here `mlp.gate` and `mlp.shared_expert_gate` are 8-bit and
everything else follows the base width) but `Gemma4MLXWeights.bits(for:)` already reads them
per tensor rather than assuming. The `OptiQ` mixed-precision build should also decode, since
nothing in that path assumes a single width.

So the quantization work is zero. That is the good news and it is most of the good news.

### The memory model

Computed from the config, and checked against the published repository sizes: the 4-bit model
comes to 18.0 GiB of text weights against 19.0 GiB actually published, the difference being the
vision tower.

| | 4-bit | 8-bit |
| --- | ---: | ---: |
| expert pool | 16.88 GiB | 31.88 GiB |
| resident, text only | **1.15 GiB** | 2.17 GiB |
| total, text only | 18.02 GiB | 34.04 GiB |
| experts read a token | 540 MiB | 1020 MiB |
| GPU bytes a token | **1.41 GiB** | 2.66 GiB |

**The expert pool is 94 % of the model.** That is a better fit for this project's thesis than
anything shipped so far: Gemma 4 Q4 is 1.83 GiB resident against 14.6 on disk, and this is
1.15 against 18.0.

At 1.41 GiB a token against Gemma Q4's 2.02, decode should be **faster than Gemma**, in the
region of 11 to 13 tok/s if it stays bandwidth-shaped. The 8-bit build moves nearly twice the
bytes a token and should land near half that.

**A note on the framing.** 8-bit is not the low-memory option: it is 34 GiB against 18 on disk
and 2.17 GiB against 1.15 resident, and it decodes at roughly half the rate. It buys accuracy,
not memory. 4-bit is the smaller *and* faster build; the trade is quality.

### What is genuinely new

**1. Gated DeltaNet on 30 layers of 40.** `layer_types` alternates three `linear_attention` to
one `full_attention`. The linear layers are a recurrence, not attention: a depthwise causal
convolution of kernel 4, then a delta-rule state update carried in a `[32 value heads, 128,
128]` float32 state. Nothing in `HydraMetal` expresses this. It is a new kernel family and it
is the whole cost of this integration.

Two consequences that are easy to miss:

- **It changes what a cache is.** `KVCache` allocates keys and values per layer; a linear layer
  needs a fixed-size recurrent state instead. 62.8 MiB total, and it does not grow with context.
- **It cannot be rewound.** A KV cache can be truncated to a prefix; a running state cannot,
  because it has already absorbed everything after it. Conversation reuse therefore works only
  for a pure append, which is exactly what `canRewind(to:)` was generalized to express
  yesterday. The machinery exists; the answer for a linear layer is "only at the current
  position".

**2. Prefill is the real risk.** Our prompt processing is fast because a chunk of tokens goes
through a layer together. A recurrence is sequential in the token index, so the naive form of
this loses the batching on three quarters of the layers. The literature's chunkwise-parallel
formulation of DeltaNet exists precisely for this and is a second, harder kernel. **Budget the
integration on this, not on the decode path.**

**3. Smaller items**, each real but bounded: `attn_output_gate: true` adds a gate projection on
the attention output; `mrope_interleaved` with sections `[11, 11, 10]` and
`partial_rotary_factor 0.25` is not the rotary we have; a shared expert of intermediate 512 runs
always, which is the same shape as Gemma's dense branch; `tie_word_embeddings` is **false**, so
embedding and head are separate tensors and both stay resident, 0.53 GiB of the 1.15 at 4-bit.

**4. A bound that is exactly met.** `num_experts_per_tok` is 8 and `gemma_router_topk` holds
`float chosen[8]` and computes `min(dims.y, 8u)`. Qwen fits with zero margin, and the kernel
**silently clamps** rather than refusing. A model wanting nine would route on eight and produce
plausible wrong output. That clamp should become a precondition before this model lands, not
after.

**5. The vision tower is installed, as Gemma's is.** 27 layers, hidden 1152, patch 16,
`out_hidden_size` 2048 so it projects straight into the text model's width. Nothing executes it
in a first pass, but the bytes are carried and described in the manifest, because vision is a
planned feature and re-downloading 20 GB to get them back is not a cost worth taking twice.
This is the same call as the Gemma reversal recorded in D-021, and for the same reason.

Worth noting for when that feature arrives: Qwen's tower is a different shape from Gemma's, and
the text side already carries the multimodal plumbing. `image_token_id` 248056 and
`video_token_id` 248057 are real vocabulary entries, `vision_start_token_id` and
`vision_end_token_id` bracket a span, and `mrope_section [11, 11, 10]` exists precisely to give
image patches two spatial position components alongside the temporal one. Implementing mRoPE
correctly for text is therefore not throwaway work: it is the same code the vision path needs.

**6. Not needed at all yet:** `mtp_num_hidden_layers: 1`, a multi-token prediction head that is
ignorable for ordinary decoding, though it is a natural draft model for the speculative decoding
this project already implements.

### Corrected since writing

Two numbers in this entry were wrong, both understated, and both for the same reason: the
linear layers carry more than the recurrence.

- The recurrent state is **62.8 MiB**, not 60. The missing 2.8 MiB is the convolution's rolling
  window, which is state in exactly the same sense and has to be carried between tokens.
- The resident total is low, as D-027 already flagged: this entry modelled a linear layer as
  four square projections and it is a 2048 to 8192 fused qkv projection, a separate gate
  projection, a depthwise convolution and two per-head vectors.

`Qwen35MoeConfigTests` computes both from the transcribed config now, so the next correction
comes from arithmetic rather than from a reader noticing.

### What this revises

D-018 ordered Gemma before Qwen on the grounds that Qwen's Gated DeltaNet was the harder kernel
family. **That ordering was right and the reason was right**, which is worth recording because
the same entry's estimate for Gemma was wrong by five kernels. The estimate here is deliberately
coarser: one new kernel family, one hard question about whether it can be made chunkwise
parallel, and a long tail of small conforming work. Anything more precise would be the same
mistake in the same place.

---

## D-025, Audited against TurboFieldfare: three real differences, and the first one made us slower
**2026-08-10, audited against the published source and system design**

TurboFieldfare runs **the same model and the same checkpoint format** we ship, Gemma 4
26B-A4B, MLX affine 4-bit group 64, 8-bit router, at 14.3 GB installed against our 15.7 GB and
~2 GB resident against our 1.4 GB. It reports **5.1–6.3 tok/s on an 8 GB M2 MacBook Air**; we
measure 6.6–8.0 on a 24 GB M4. On hardware roughly twice as capable we are, at best, level.

### What their decode does differently

From `docs/SYSTEM_DESIGN.md`:

1. **The shared branch covers the reads.** Their `cb1` ends at the router; the dense branch is
   started *after* it, so it runs while the CPU `pread`s the missing experts. Ours computed it
   inside `cb1`, before the readback, where it could overlap nothing, despite D-021 recording
   it as "the I/O-overlap mechanism GPT-OSS lacks" when Gemma arrived.
2. **They do not wait on `cb2`.** *"The command-buffer pipeline can delay waiting for cb2 while
   the CPU encodes and queues the next layer."* We call `waitUntilCompleted` on all sixty
   command buffers a token; only the router readback actually needs one.
3. **Hits start before misses finish.** *"Routed work for cache hits may start while reads for
   missing experts are still running."* We block on all eight.

They also default to 16 slots where we default to 8, and use the same LFU-with-recency policy.

### The first one was implemented, measured, and reverted

Splitting the dense branch out of `cb1` so it runs during the reads is a **~30 % regression**:
6.59–7.99 tok/s before, 5.07–6.02 after, confirmed with the comparison run in both orders.

The reason is that it trades the wrong resource. M-031 established that decode is bound by
submission and dispatch latency, not bandwidth. The split adds one command buffer per layer,
thirty more submissions a token, to hide I/O that the unified memory bus is partly competing
for anyway: measured, `expertIO` rose from ~28–40 ms to ~60–86 ms once the GPU had work to do
during the reads.

**Overlap is worth having only once submissions are cheap.** That inverts the order of the
remaining work: fix the submission count first (2 and 4 below), then revisit 1.

### The order to do it in

1. **Stop waiting on `cb2`**, ~~commit and let the CPU encode the next layer, releasing slots
   from a completion handler~~. **Attempted and reverted (M-033): it is a correctness change,
   not a scheduling one.** The scratch buffers are shared by every layer, and committing to one
   queue does not serialize execution across buffers, the wait was carrying that dependency.
   Do it as **encode-ahead** instead: build the next layer's `cb1` before awaiting `cb2`, then
   wait, then commit. The CPU work overlaps the GPU's and the ordering is unchanged.
2. ~~**Batch the expert dispatches**~~, **done with separate bindings, measured neutral,
   reverted (M-034).** 1,200 launches a token became 150 and throughput did not move, which
   retires the dispatch-count theory along with it. The remaining candidate is the shape of
   the GEMV: one threadgroup per row, each paying a `simd_sum` and a barrier for one scalar.
3. **Start hits before misses complete.**
4. **Then** revisit the shared-branch overlap, which should pay once 1 and 2 have removed the
   submissions it currently competes with.

### What is already settled and should not be reopened

`00-FEASIBILITY.md` records TurboFieldfare's own measurement that one layer's expert choices
predict **7 %** of the next layer's, and that the router depends on the same layer's attention
output. **Speculative inter-layer prefetching is dead**; the shared branch is the only overlap
available. Advice to add prefetching should be declined with that number attached.

**Reopen if** a measurement shows submissions are no longer the binding constraint, at which
point item 4 becomes the next lever rather than the last.

---

## D-024, The Q4 Gemma comes from the MLX build, and what that costs
**2026-08-09, verified against the published checkpoints**

Three Q4 conversions of Gemma 4 26B-A4B exist. The choice is not arbitrary.

| source | size | container | verdict |
|---|---:|---|---|
| `google/…-qat-q4_0-unquantized` | 51.6 GB | safetensors, BF16 | QAT weights, **not quantized**, no I/O win unless we quantize ourselves |
| `google/…-qat-q4_0-gguf` | 15.6 GB | GGUF | would need a container parser and llama.cpp's naming |
| `lmstudio-community/…-QAT-MLX-4bit` | 15.6 GB | **safetensors** | chosen |

**Provenance, stated plainly: the MLX build is not Google's.** It is published by
`lmstudio-community`; there is no first-party MLX repository. It is a conversion of Google's
QAT weights, and the measurement below establishes that chain rather than taking it on trust.

It is chosen because it is **safetensors**, the existing reader, header parsing, streaming
repacker and coverage check all apply unchanged, where GGUF would need a second container
format, and because it is what LM Studio runs on Apple Silicon, which makes a throughput
comparison like-for-like rather than approximate.

### What the checkpoint actually contains, read from its own header

- **Mixed precision.** 4 bits by default, group 64, affine, with **120 tensors at 8 bits**.
  That is `30 layers × 4`: every layer's dense MLP and router, not a special case for layer 0.
  The per-tensor map has to be read; assuming uniform 4-bit halves those matrices' width.
- **Affine, not symmetric.** Every quantized tensor is a triple, `.weight` packed into `U32`,
  `.scales`, `.biases`, and decodes as `q · scale + bias`. MXFP4 and `q4_0` are both
  scale-only. A decoder written from habit reconstructs `q · scale` and shifts every weight by
  a per-group constant, which is a model that still speaks.
- **Experts are unfused.** `experts.switch_glu.{gate,up,down}_proj`, three matrices where the
  BF16 build fuses gate and up into one that the repacker splits. Nine sub-tensors to a blob.
- **Names are inverted**: `language_model.model.layers.N…` against `model.language_model…`.
- The vision tower ships (358 tensors); there is no audio tower.

### The two conventions, settled by measurement

Decoding a real `k_proj` four ways and comparing against Google's QAT weights:

| packing | bias | relative error |
|---|---|---:|
| low-order first | applied | **0.078** |
| low-order first | ignored | 1.854 |
| high-order first | applied | 1.148 |
| high-order first | ignored | 2.138 |

0.078 is what 4-bit at group 64 costs. The same decode against the **non-QAT** weights gives
0.43, which is how we know this build descends from the QAT checkpoint, and how anyone
repeating the exercise can check that it still does.

### What it buys

An expert blob is **3,345,408 B against BF16's 11,894,784**, 3.55×, so the expert pool falls
from 45.7 GB to 12.85 GB. Expert reads are the measured bottleneck in both prefill and decode
(M-028, M-029), so this is aimed at the one number that matters.

**Reopen if** a first-party MLX release appears, or if the 8-bit tensor list changes between
revisions, the second is why the map is read rather than hardcoded.

---

## D-023, Hydra is a hub: one seam per thing that actually differs
**2026-08-06, agreed**

The goal is stated plainly: **choose a model, and that model's path works**, its layout, its
tokenizer, its prompt format, its layer topology, its quirks. Not a runtime that bends to fit
whichever model was loaded last.

This is the abstraction `GptOssConfig` deferred to "phase 3, from two working engines". The
second engine exists, so extracting is now correct. It is extracted **from what was observed to
differ**, not designed from imagination, every seam below was found by a build failure or a
audit finding, never by anticipation.

### The seams, and what sits on each side

| seam | shared | per model |
|---|---|---|
| `ModelDescriptor` | sizes, layer counts, per-layer geometry | the values |
| `ExpertBlob` | payload, stride, source bytes | **what is inside** a blob |
| `HydraLayout` | placement and alignment rules | the tensor list |
| repack plan | `ScatterCopy`, spans, verification | tensor names, exclusions |
| layer runner | command-buffer scheduling, the slot cache | **the forward pass** |
| tokenizer | the BPE algorithm | normalizer, byte handling, merges policy |
| prompt | streaming parse | the chat format and stop tokens |

The rule that decides which side a thing goes on: **if getting it wrong would produce plausible
degraded text rather than an error, it is per-model and it gets a test.** That is what D-011,
D-014 and D-022 are; they are not documentation, they are the specification each engine is
measured against.

### What must not happen

**No `if architecture == .gemma` scattered through the runtime.** Dispatch happens once, where
the model is chosen; everything downstream receives an engine that already knows what it is. A
conditional in a kernel path is how the 20B silently starts using Gemma's attention scaling.

**No shared field that only one model uses.** The `ExpertBlob` contract holds three numbers
rather than MXFP4's six sub-tensors precisely because forcing every architecture to declare the
others' fields is how a BF16 checkpoint acquires an imaginary scales tensor.

**No abstraction ahead of a second implementation.** A protocol with one conformer is a guess.
Each seam above was cut only when a second model pushed against it, `ExpertBlob` when Gemma's
experts turned out to be two plain matrices, `HydraLayout` when its tensor list stopped
matching, the repack coverage check when the towers made the old one unsatisfiable.

### Consequence for the catalogue, implemented 2026-08-09

`CatalogEntry` carried a `GptOssConfig`, which made it a list of one architecture's variants
with display names attached. It now carries `any ModelDescriptor`, and three named factories
turn that into everything else:

| factory | produces | lives in |
|---|---|---|
| `RepackPlanFactory` | `any InstallablePlan` | `HydraInstall` |
| `ModelRuntime` | `any TextModelRunner` | `HydraMetal` |
| `ConversationFormats` | `any ConversationFormat` | `HydraTokenize` |

**Those three switches are the only places `architecture` is read.** Adding a model is a row in
the catalogue plus a case in each, and the app, the CLI and the generation loop change not at
all. Each factory throws rather than falls back when a descriptor's declared architecture does
not match its concrete type, because that mismatch is only reachable by writing a new descriptor
and forgetting a switch.

Two things pushed back while this was cut, and both were kept rather than papered over:

- **`validate` takes a `weightMap`.** GPT-OSS installs the whole checkpoint, so "did we miss a
  tensor" is answered by comparing byte totals. Gemma's plan may leave a tower behind, audio
  today, and the coverage check must hold whether or not one is present, so its total is not
  required to equal the index's and that comparison says nothing. It has to name what it
  skipped. The parameter is the difference, made visible.
- **`ReasoningLevel.off` has no Harmony meaning.** Gemma's template can close the thought
  channel; GPT-OSS always reasons and its levels only say how much. `off` maps to `low` there,
  and the app's picker does not offer a choice that would silently do nothing.

Sampling went the other way. It looked per-model because it lived on `ModelRunner`, but drawing
a token reads a distribution and returns an index, nothing in it depends on how the
distribution was produced. It moved to `TokenSampler`, held by both runners. **The rule runs in
both directions: what does not differ per model does not get a seam.**

**Reopen if** a third architecture needs a seam bent rather than a conformer added, that would
mean one of these lines was drawn in the wrong place, and the audit that revealed it should be
recorded the way D-022 was.

---

## D-022, Gemma 4 operator semantics that cannot be guessed
**2026-08-06, verified against `transformers/models/gemma4/modeling_gemma4.py`**

Same exercise as D-014 and D-011, and it found more traps. Every item below was read in the
official source, not inferred. Getting any of them wrong yields a model that produces plausible
degraded text with no error raised.

**RMSNorm, the weight is `w`, not `1 + w`.**
```python
normed = hidden.float() * torch.pow(hidden.pow(2).mean(-1, keepdim=True) + eps, -0.5)
normed = normed * self.weight.float()
```
Gemma 3 used `(1 + weight)`. Gemma 4 does not. Carrying the old habit over is a silent 2× on every
normalized activation. `eps` goes **inside** before the power, as in GPT-OSS.

**Some norms carry no weight at all.** `v_norm` and `router.norm` are built with
`with_scale=False`, so they have no tensor in the checkpoint. `v_norm` in particular is an
operation with **nothing in the weight index to hint that it exists**.

**Embeddings are scaled by `sqrt(hidden_size)`**, `embed_scale=config.hidden_size**0.5`, i.e.
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
per layer type**, 4096 rows on sliding layers, 8192 on full ones. `HydraLayout` sizes tensors from
a single config today.

**RoPE splits into halves**, as GPT-OSS does, `emb = cat((freqs, freqs))`. That convention carries
over unchanged.

**Partial rotation is expressible as zero frequencies.** Full layers use `rope_type: "proportional"`
with `partial_rotary_factor: 0.25`:
```python
rope_angles = int(0.25 * 512 // 2)            # 64
inv_freq = cat(inv_freq_rotated,              # 64 real frequencies
               zeros(512 // 2 - 64))          # 192 zeros
```
A zero frequency gives `cos = 1, sin = 0`, the identity. **The kernel needs no change**: only
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

**The layer topology differs, not just its constants.** The MoE branch reads the **residual**, the
state *before* the dense MLP, not the MLP's output:
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
**before** the residual add, post-norm, where GPT-OSS is pre-norm.

**`final_logit_softcapping`**: `logits = 30 * tanh(logits / 30)`.

**The MLP is plain.** `down(gelu_pytorch_tanh(gate(x)) * up(x))`, no clamping, no `+1`, none of
GPT-OSS's SwiGLU quirks (D-014). Simpler, and the difference must not be papered over by reusing
the existing kernel.

**Disabled for this checkpoint, but present in the code:** `hidden_size_per_layer_input = 0` and
`num_kv_shared_layers = 0`, so the per-layer-input branch and KV sharing are both inert. They must
be **rejected explicitly** if a future checkpoint enables them, not silently ignored.

**Still unverified:** the exact `embed_scale` at runtime for the QAT checkpoints, and the audio
tower's presence in the weight index (the model ships vision *and* audio towers; both must be
excluded from the repack plan).

---

## D-021, Gemma 4 26B-A4B, in full precision
**2026-08-06, agreed**

The third model is **Gemma 4 26B-A4B, installed exactly as Google published it: BF16**, not from
the QAT q4_0 checkpoints that also exist.

**Why full precision.** It is D-015 stated on a model that was not chosen to flatter it. Everyone
ships Gemma 4 at Q4; running it undegraded on 24 GiB is the project's thesis, and it makes the one
comparison that cannot be argued with, **the same model, against TurboFieldfare's published Q4
numbers**, with the quality difference on our side and the cost visible.

It also fills the gap in the catalogue: 20B, **26B**, 120B.

**Accepted cost.** The experts are 45.7 GB and 2.86 GB are read per token, 2.25× GPT-OSS 20B.
Expect roughly **2–2.5 tok/s** on the M4, in the 120B's range. D-001 already settled that this is a
result, not a failure.

**Blocked on storage, not on code.** ~50 GB installed against **28 GB free** with both GPT-OSS
models present. Dropping the vision tower saves only ~1.5 GB. Something has to give: remove the
120B (it is re-downloadable), install to an external volume via `hydra install <model> <dir>`, or
defer.

### What the audit found, and what D-018 got wrong

D-018 called Gemma "the cheapest, no new kernel family". **That was wrong.** The fused per-layer
expert tensors do match what `ScatterCopy` already splits, but the rest does not:

- **two attention geometries in one model**, sliding layers 16 heads × 256, full layers
  `global_head_dim 512` with 2 KV heads;
- **`attention_k_eq_v: true`**, K and V are one tensor;
- **a 5:1 sliding/full pattern**, where `KVCache` keys off `layer % 2`;
- **two RoPE configurations**, sliding θ=10 000; full θ=1e6 with `partial_rotary_factor 0.25`,
  so RoPE covers a quarter of the head dimension;
- **q_norm / k_norm** before attention;
- **`final_logit_softcapping` 30.0**;
- **tied embeddings**, no `lm_head`; the 1.48 GB embedding *is* the head and must be resident,
  where GPT-OSS maps it outside the working set;
- **five feed-forward norms per layer**, plus `layer_scalar`, `router.scale`,
  `router.per_expert_scale`;
- **gelu_pytorch_tanh**, not GPT-OSS's clamped SwiGLU;
- a **27-layer vision tower** to exclude from the plan.
  **Reversed 2026-08-09, the tower is installed.** The exclusion was reasoned about compute:
  nothing executes it, so why carry it. But the cost is bandwidth, not compute. Measured
  against the real checkpoint the tower is 1.146 GB of 51.6 GB, and leaving it out means
  fetching the whole checkpoint a second time to add image support. It goes to `vision.bin`,
  mapped separately and untouched by a text conversation, described by the manifest with the
  dtype and shape the source declared because no `ModelDescriptor` knows the tower's structure
  and none should pretend to. The encoder remains unwritten; the bytes no longer stand in its
  way. The same measurement settled a second question: this checkpoint carries **no audio
  tower at all**, the entire excluded 1,145,588,832 B was vision.

Five of those touch kernels. Still less than Qwen's Gated DeltaNet, so the D-018 order stands, but
the estimate that went with it did not.

**One structural gain:** the always-active dense MLP beside the experts (1.07 GB resident) is the
I/O-overlap mechanism `00-FEASIBILITY.md` records GPT-OSS as lacking.

---

## D-020, Variable precision by role: measured, built, abandoned
**2026-08-06, closed. Implemented, measured (M-027), reverted.**

> **Outcome.** Q8 on the dense weights was built end to end and removed. The quality
> gate passed (M-026), and then the point collapsed: the dense weights are 66 % of the
> **bytes** per token but **13 % of the time**, so halving them bought **3.5 %**. The
> memory freed could not be spent either, more expert slots measure *slower*, not
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

**The dense part is 66 % of the traffic.** The experts, the point of the project, are 34 %.

**Frozen, not up for measurement.** Routers and norms stay BF16. A router is 184 KB per layer on
the 20B: there is nothing to win. And a routing error is not a small perturbation, it is
**discrete**: the token goes to the wrong expert entirely, and nothing downstream recovers it.

**Worth measuring.** BF16 → Q8 on the LM head and on attention. Removes 1.219 GB per token, 33 %
of the traffic. Weight-only 8-bit is the case where the literature is least ambiguous, and D-015
explicitly leaves that door open.

**Still refused without proof: 4-bit on the dense.** The symmetry *"the experts are already 4-bit,
so the rest can be"* is **false**. GPT-OSS's experts are MXFP4 because OpenAI trained them that
way, quantization-aware. Post-training 4-bit applied to weights trained in BF16 is a different
and worse operation. Confusing the two would degrade the model precisely as D-015 forbids.

**What decides.** Not an opinion, a measurement, on a fixed prompt corpus:
- KL divergence of the logit distribution against the BF16 reference;
- top-1 agreement rate under greedy decoding, which must stay at 100 % or explain each departure.

The machinery already exists: `HydraReference` in double precision, and the exact-equivalence
harness written for speculative decoding.

**Adopt if** the two criteria hold and no reasoning regression shows up on long chains of thought,
the case where errors compound, and the one GPT-OSS exercises most. **Abandon otherwise**, and the
36 % of throughput stays unclaimed. That is D-015's arbitration, unchanged.

---

## D-019, Precision variants are produced by the repacker, not downloaded
**2026-08-06, moot, kept for the next model that offers variants (see D-020)**

Offering Q8 and Q4 variants of a model means **producing them ourselves during installation**, from
the same upstream safetensors, not ingesting community GGUFs.

For GPT-OSS the question does not even arise: OpenAI publishes MXFP4 experts and BF16 dense, and
nothing else. Any other precision is **our artifact**, and must be labelled as such.

**Why the repacker rather than GGUF.** It already performs a streaming transformation under a plan
verified before a single byte is downloaded (`RepackPlan.validate`). Quantizing inside a
`ScatterCopy` extends a mechanism that exists. Reading GGUF would mean a second format reader, with
its own quantization conventions to verify from scratch, the kind of undocumented detail D-014 and
D-011 exist to guard against. One download path, N precision outputs.

**Consequence to absorb.** `InstallationVerifier` compares installed bytes against the upstream
checkpoint. That check is **void for any requantized group**: the bytes are legitimately different.
Those groups need their own verification, dequantize, and compare against the source within a
stated tolerance. The manifest must therefore record the precision of every tensor group, and
verification must branch on it. Shipping requantization without that branch would silently lose
the guarantee that makes installation trustworthy.

---

## D-018, Two models before any platform decision
**2026-08-06, agreed**

Order of work: **Gemma 4 26B-A4B, then Qwen3.6-35B-A3B.** Once both run, four models work
(GPT-OSS 20B and 120B, plus those two), and only then do we choose between Qwen3.5 122B-A10B,
DeepSeek V4 Flash, and the Linux/Windows port, on evidence, not in advance.

**Why models before the port.** The project rests on having an oracle: *a slow, obviously correct
implementation against which a fast and subtle one is measured*. Porting first means rewriting
HydraMetal against **one** model, where every divergence is ambiguous, wrong kernel, or wrong model
wiring? Adding models first turns each one into a **test vector for the port**: a disagreement
localized on a single model says where to look.

`HydraReference` depends on no platform (`Package.swift`). The oracle travels with the port without
a line of change. It is the repository's most valuable asset for that work, and it is already
written.

**Why Gemma first, it is the cheapest, not the hardest.** Its alternating local-sliding-window /
global attention is the pattern `KVCache` already implements. **No new kernel family.** The real
work is the SentencePiece tokenizer (262 k, where `HydraTokenize` is o200k BPE), the chat template,
Gemma's norm placement, and a **shared expert** path, one pinned slot per layer, which incidentally
gives Hydra the I/O-overlap mechanism `00-FEASIBILITY.md` notes GPT-OSS lacks.

**Why Qwen second.** Gated DeltaNet covers 30 of its 40 layers and is, per the feasibility study,
*the project's most expensive item*. It is worth attempting once a second model has proven the
`.hydra` format and the repacker are genuinely generic, which one model cannot show.

**Reopen if** the Gemma work uncovers that the format is not generic: that would make the port the
better next step, since it would no longer depend on an assumption the models were meant to confirm.

---

## D-017, Sampling defaults: departing from OpenAI's recommendation
**2026-08-05, decided on observation**

OpenAI recommends `temperature = 1.0` and `top_p = 1.0` for GPT-OSS, the raw, untruncated
distribution. **We depart from it**: 0.7 and 0.9 by default.

**What prompted it.** On the prompt "hi", the 20B reasoned as follows: *"User says hi. Need to
respond in Bengali because user asked earlier to translate."*, and answered in Bengali. No such
instruction existed; the model hallucinated an antecedent.

**The template is not at fault.** The rendered Harmony prompt is identical to the published
`chat_template.jinja`, and tests lock that down. The problem is sampling: on a two-token prompt,
the raw distribution of a 20 B model leaves appreciable mass on aberrant continuations, and nothing
in the context rules them out.

**What it remains.** A setting, not a constraint: the slider goes to 1.5, and OpenAI's
recommendation stays reachable. A developer-instructions field is also exposed, which can pin a
reply language, that is the mechanism the format provides for it.

---

## D-016, Scope of the first application
**2026-08-05, agreed**

**The memory gauge is the centrepiece.** It shows the gigabytes actually resident against the
model's full weight. Three distinct quantities, never conflated: memory engaged by the process,
mapped weights (reclaimable by the system under pressure), installed size on disk. Announcing a
single flattering number would run against the project's honesty.

**Rejected:** the expert-slot visualiser (spectacular but costly in space for little information)
and the display of hit rate and SSD reads (useful for tuning, not for demonstrating).

**Kept:** persistent multi-session history, reasoning in a disclosure panel, and exposed settings,
context length and expert slot count at load time, temperature and reasoning effort per
conversation. The slot count is exposed deliberately: it is what makes the memory/speed trade-off
tangible.

**Catalogue frozen** to OpenAI's two official models. Accepting an arbitrary Hugging Face
repository would open the door to unsupported architectures, which would have to be detected and
refused cleanly, work unrelated to the goal.

**Installing in the background during a conversation: yes.** It is both the easy path *and* the
correct one: installing engages no runtime, only I/O bounded to ~50 MiB. **One model is loaded for
inference at a time**, however, two concurrent runtimes would double the footprint, against the
whole point.

**Distribution:** build from source, published on GitHub. No notarization for now. The CLI is kept:
it is the project's measurement instrument.

---

## D-015, Do not degrade the model: the goal is intelligence at a reduced footprint, not a minimal footprint
**2026-08-05, agreed, refines D-012**

**Refused: quantizing the dense weights.** Moving attention, routers and the LM head from BF16 to
MXFP4 would save 1.67 GiB, or 73 % of the resident floor. It is the largest memory gain available,
and it is **ruled out** if it costs quality.

**What the project demonstrates, exactly.** Not "fit an LLM into little memory", any aggressive
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
overlap qualify, they change scheduling, not values.

---

## D-014, GPT-OSS operator semantics that cannot be guessed
**2026-08-05, verified against `gpt_oss/torch/model.py`**

Each of these was read off the reference implementation, not inferred. Getting one wrong raises
**no error**: the model loads, generates plausible text, and comes out degraded. They are pinned by
reference vectors in `tools/gen_reference_fixtures.py`.

**SwiGLU splits on even and odd indices**, not into two halves:
`x_glu, x_linear = x[..., ::2], x[..., 1::2]`. The rows of `gate_up_proj` are therefore
**interleaved** `[gate₀, up₀, gate₁, up₁, …]`. Splitting into halves would give a model that works
but mixes channels.

**RoPE, on the other hand, does split into halves** (`torch.chunk`). Two opposite conventions in
the same architecture, which is exactly what makes the mistake easy.

**SwiGLU clamping is asymmetric**: the gate branch is bounded **from above only** (`min(x, 7)`),
the linear branch on both sides. And the linear branch gets **+1** before the product. The swish
uses `sigmoid(1.702·x)`, not `sigmoid(x)`.

**YaRN applies a concentration** on top of frequency rescaling: `0.1·ln(factor) + 1`, i.e.
**1.3466** for GPT-OSS. It multiplies cos and sin. Omitting it breaks nothing visible but shifts
all attention.

**Attention sinks are one extra logit column** in the softmax, dropped afterwards. They contribute
nothing to the result: they enlarge the denominator, which lets a head look at nothing. In the
kernel this falls out elegantly, the online softmax starts with `max = sink` and `denominator = 1`.

**The router applies its softmax to the top-k logits only**, after selection, not to the full
distribution.

**The sliding window is 128 and applies to even-indexed layers.** The mask `tril(diagonal=-128)`
admits exactly 128 positions.

---

## D-013, The repacker's memory bound comes from streaming, not from splitting requests
**2026-08-05, decided on measurement, corrects an initial choice**

**What I had chosen.** Split every source range into 4 MiB sub-requests, so that the memory bound
would be a property of the splitting, trivially checkable, rather than a dependency on
`URLSession`'s buffering behaviour.

**What the measurement said.** On the real repository:

| Pattern | Throughput |
| --- | ---: |
| one 64 MiB `Range` request | **33.5 MB/s** |
| eight 4 MiB `Range` requests in series | **5.2 MB/s** |

A factor of **6.4**. The cause: Hugging Face answers with a **302 to a signed CDN**, and every
request pays the redirect again plus a TLS handshake to another host. Reusing the resolved URL is
impossible, its policy carries a `ByteRange` condition tied to the exact range requested.

**What we do instead.** A **single request per contiguous region** of the source checkpoint, whose
response is **consumed as it streams**: every block the network stack delivers is routed to its
destination and released before the next arrives. Since the plan covers the checkpoint exactly with
no gaps, neighbouring tensors form long regions, the source file is read almost end to end,
sequentially.

**Measured on the real 20B install: 44 MB/s**, or **8.5×** the initial approach.

**What it costs.** The bound is no longer a property of the splitting: it depends on the size of
the blocks `URLSession` delivers (measured up to 3.5 MiB). It is therefore now **checked by tests
and instrumented in production**, the repacker tracks the largest block received and exposes it in
its progress, rather than guaranteed by construction. An accepted trade-off: a bound measured on
every run beats a theoretical bound that divides throughput by six.

---

## D-012, Minimizing memory is the goal, not filling the available ceiling
**2026-08-05, agreed, corrects D-001**

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
that is, an 8 GiB machine.

**Related finding: the floor is no longer the experts, it is the resident weights.** GPT-OSS keeps
attention, routers and LM head in **unquantized BF16**, 2.27 GiB for the 20B, 2.88 GiB for the
120B, of which 1.08 GiB for the LM head alone. That is why TurboFieldfare reaches ~2 GB total on
Gemma 4 while we bottom out near 3.8 GiB: on their side, those same tensors are 4-bit. **Going
lower would require quantizing the LM head and attention**, which changes outputs, an experiment
to validate against a reference, not a design decision. It is the main remaining lever.

---

## D-002, Portability is a stated goal, but no abstraction is written in advance
**2026-08-05, agreed**

Eventual targets mentioned: M3 Ultra / M5 Max short on unified memory, and an **x86_64 + CUDA**
port letting an RTX 5090 (32 GiB of VRAM) run models sized for an RTX Pro 6000.

**What we do now:** nothing speculative. No `ComputeBackend` protocol, no GPU abstraction layer. Per
the brief, we generalize only from code that works.

**What we do forbid ourselves right away**, because it is free and the reverse is expensive to
undo: **the `HydraCore`, `HydraFormat`, `HydraInstall` and `HydraTokenize` modules do not import
Metal.** Format, streaming, caching and tokenization logic stays pure Swift, portable as-is to
Linux and Windows. Only `HydraMetal`, `HydraRuntime` and `HydraApp` are platform-bound. That is a
layering discipline, not an abstraction.

**Technical note for a future CUDA port**, not to be implemented, only kept in mind: a machine
with a discrete GPU offers a **three-level** hierarchy (VRAM / system RAM / SSD) where the Mac has
only two. That is structurally **more favourable**: a 5090 has 32 GiB of VRAM, a hundred-odd GiB of
system RAM usable as a second-level cache, and memory bandwidth an order of magnitude above the
M4's. There the bottleneck returns to the SSD and the PCIe bus, not compute.

---

## D-003, No third model for now
**2026-08-05, agreed**

Qwen3.6-35B-A3B is out of the initial scope: its 30 Gated DeltaNet layers represent as much kernel
work as both GPT-OSS models combined.

Scope becomes **GPT-OSS 20B then GPT-OSS 120B**. Phase 4 (third model) is dropped from the plan;
phase 3 (generalization) will be carried out on two real models, and a third target will be chosen
later, with hindsight.

**Reopen if:** phase 3 shows that two models from the same family are not enough to bring out the
right axes of variation for the model contract.

---

## D-004, Storage managed the way LM Studio does it
**2026-08-05, agreed**

The user installs and uninstalls models from the interface, knowingly. Zero, one or several models
may coexist.

**The only automatic rule:** Hydra computes the space remaining **after** the contemplated install
and **warns** if it would fall below **10 GB**. A warning, not a block, the user decides.

For each model the interface shows: size on disk, state (installed / partial / absent), current
free space.

---

## D-005, Context length chosen when the model is loaded
**2026-08-05, agreed**

Like LM Studio: when loading a model, a dialog offers the context length.

An important technical consequence: **the expert cache slot count is computed at load time**, not
fixed at compile time. The budget is derived at runtime from `recommendedMaxWorkingSetSize`, the
chosen context and the size of the resident weights. The interface shows the resulting slot count
and expected throughput **before** confirming.

Offered values: 4k, 8k, 16k, 32k, 64k, 128k. Default: 32k.

---

## D-006, Metal ceiling: detect and suggest, never impose
**2026-08-05, recommendation, uncontested**

Hydra reads `recommendedMaxWorkingSetSize` and computes everything from it. It **detects** whether
`iogpu.wired_limit_mb` has been raised and accounts for it. It **suggests** the command to the
user, documented, with its quantified effect (+5 slots/layer on the 120B) and its risk. It never
runs it itself, and works correctly at the default budget.

---

## D-007, macOS 26 minimum, Apple Silicon only
**2026-08-05, recommendation, contestable**

Reasons: it is the version of the development and validation machine; it gives access to Metal 4
and the Metal Performance Primitives for prefill; and supporting earlier versions would mean
validating on hardware we do not have.

The code detects the GPU family at runtime. **This machine is apple9, not apple10**: the TensorOps
path that accelerates long-context attention for TurboFieldfare is not available to us.

---

## D-008, Harmony reimplemented in Swift, under a conformance harness
**2026-08-05, recommendation**

The official library is in Rust. Rather than introduce Rust into the build chain, we reimplement in
Swift and freeze a corpus of conversations rendered by the official library as **fixtures**. The
test requires **byte-for-byte** equality of the rendering.

A Harmony divergence degrades outputs **without raising an error**: exactly the kind of bug that
must be made impossible by construction.

---

## D-009, No dense execution path
**2026-08-05, recommendation**

Hydra is a streaming MoE engine and owns that as a deliberate limit. A resident dense path would be
a second runtime for a use case already well covered by MLX and llama.cpp, with nothing
differentiating to add.

---

## D-010, Hugging Face source: the repository root
**2026-08-05, decided on the audit**

For GPT-OSS, the repacker reads the **root** of the repository (14 shards for the 120B), not
`original/` nor `metal/model.bin`. MXFP4 values there are already split into `blocks` / `scales` /
`bias`, each sub-tensor being a contiguous byte range directly addressable by an HTTP `Range`
request.

---

## D-011, MXFP4 format details verified against the reference implementation
**2026-08-05, verified**

Locked against `openai/gpt-oss` (`gpt_oss/torch/weights.py`):

- E2M1 table: `[+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6]`;
- **low nibble → even index, high nibble → odd index** within each `uint8`;
- E8M0 scale: `value = fp4 * 2^(scale_byte - 127)`, applied with `ldexp`;
- block of **32 values** along the last dimension: 16 packed bytes + 1 scale byte.

Getting the nibble order wrong produces a model that generates plausible but degraded text, with no
error, hence verifying up front rather than while debugging.

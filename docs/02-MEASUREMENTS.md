# Hydra, Measurement log

One entry per experiment: what was measured, how, and what we take from it. Negative results
appear on the same footing as positive ones, they are the expensive ones to rediscover, and so
do the times a conclusion was drawn from a benchmark that turned out to be measuring something
else.

Newest first. An entry is never edited to agree with a later one; when a result is overturned
the later entry says which one it retires, so the reasoning that led somewhere wrong stays
readable.

Machine: MacBook Apple M4, 10 GPU cores, 24 GiB, macOS 26.5.2, GPU family **apple9**.
Models installed: GPT-OSS 20B (12.8 GiB), Gemma 4 26B-A4B in BF16 (48.1 GiB) and in the 4-bit
MLX quantization (14.6 GiB), all in `.hydra` format.
Reproduce with: `hydra bench 20b`, `hydra probe 20b`, `hydra bench-gemv`.

## How to read the numbers here

Three things make a figure in this file mean less than it looks like, and all three have
produced a wrong conclusion at least once:

**Thermal drift.** Throughput on this machine moves 20 % between a cold run and a hot one, and
it drifts *within* a run: GPU-busy time was watched wandering from 87 to 109 ms mid-measurement
(M-039). Every comparison here is run against a control built and timed beside it, alternating,
because an A-then-B sweep measures the thermals.

**Warm caches.** A benchmark that reads one matrix repeatedly measures the cache, not the
workload: 6 MiB stays resident and the re-read the change exists to remove never happens
(M-045). A kernel bench has to move the working set production moves.

**Aggregates.** `cb1` at 20 GB/s against a 95 GB/s ceiling looked like a factor of four of
headroom and cost two failed changes, one of them 40 % worse. It is not a kernel: it is the
projections plus the norms, the rotary, the attention and the router, and the last four move
almost no bytes while being averaged in (M-040).

## Index

- [M-067](#m-067-long-context-qwen-is-the-fastest-model-short-and-the-slowest-long) Long context: Qwen is the fastest model short and the slowest long
- [M-066](#m-066-decodes-expert-reads-are-real-io-and-two-attempts-to-remove-them-failed) Decode's expert reads are real I/O, and two attempts to remove them failed
- [M-065](#m-065-decodes-dispatches-are-not-the-cost-and-the-gemv-is-already-at-bandwidth) Decode's dispatches are not the cost, and the GEMV is already at bandwidth
- [M-064](#m-064-prefill-is-at-a-plateau-and-the-two-remaining-costs-trade-against-each-other) Prefill is at a plateau, and the two remaining costs trade against each other
- [M-063](#m-063-the-gemm-tile-does-not-matter-and-three-measurements-said-it-did) The GEMM tile does not matter, and three measurements said it did
- [M-062](#m-062-reading-the-next-batch-while-the-gpu-runs-this-one-another-18-) Reading the next batch while the GPU runs this one: another 18 %
- [M-061](#m-061-one-dispatch-a-token-for-the-router-cost-30--of-qwens-time-to-first-token) One dispatch a token for the router cost 30 % of Qwen's time to first token
- [M-060](#m-060-prefill-issued-118-million-dispatches-removing-950000-of-them-changed-nothing) Prefill issued 1.18 million dispatches; removing 950,000 of them changed nothing
- [M-059](#m-059-2491-dispatches-a-token-and-two-thirds-of-them-are-the-routed-experts) 2491 dispatches a token, and two thirds of them are the routed experts
- [M-058](#m-058-cold-mapping-is-17x-slower-and-the-prefetch-does-not-rescue-it) Cold, mapping is 17x slower, and the prefetch does not rescue it
- [M-057](#m-057-qwens-prefill-chunk-was-gemmas-and-it-cost-41--of-the-bytes) Qwen's prefill chunk was Gemma's, and it cost 41 % of the bytes
- [M-056](#m-056-mapping-the-experts-beats-copying-them-warm-by-31--and-all-the-memory) Mapping the experts beats copying them, warm, by 31 % and all the memory
- [M-055](#m-055-the-expert-cache-is-a-cache-on-top-of-a-cache-and-that-is-why-slots-do-nothing) The expert cache is a cache on top of a cache, and that is why slots do nothing
- [M-054](#m-054-retracted-the-expert-slot-effect-was-thermal-drift) **Retracted**: the expert-slot effect was thermal drift
- [M-053](#m-053-qwen-decodes-at-8-to-14-toks-and-the-round-trips-were-the-first-thing-wrong) Qwen decodes at 8 to 14 tok/s, and the round trips were the first thing wrong
- [M-052](#m-052-the-prefills-expert-slots-never-needed-clearing-and-the-saving-is-45-ms-of-25-s) The prefill's expert slots never needed clearing, and the saving is 45 ms of 25 s
- [M-051](#m-051-a-quantization-kernel-that-was-wrong-at-every-group-size-but-the-one-we-ship) A quantization kernel that was wrong at every group size but the one we ship
- [M-048](#m-048-time-to-first-token-is-not-a-function-of-context-length) Time to first token is not a function of context length
- [M-047](#m-047-prefill-end-to-end-41-s-to-25-s-and-where-the-rest-of-it-is) Prefill, end to end: 41 s to 25 s, and where the rest of it is
- [M-046](#m-046-the-prefill-chunk-has-an-interior-optimum-and-the-app-was-below-it) The prefill chunk has an interior optimum, and the app was below it
- [M-045](#m-045-the-batched-projection-is-bound-by-activations-not-weights) The batched projection is bound by activations, not weights
- [M-044](#m-044-attention-was-the-whole-remaining-budget-and-hid-a-correctness-bug) Attention was the whole remaining budget, and hid a correctness bug
- [M-043](#m-043-overlapping-the-expert-read-with-the-resident-experts-4--reverted) Overlapping the expert read with the resident experts: −4 %, reverted
- [M-042](#m-042-halving-the-round-trips-8--and-gpu-busy-scores-it-zero) Halving the round trips: +8 %, and GPU busy scores it zero
- [M-041](#m-041-one-compute-encoder-per-buffer-instead-of-660) One compute encoder per buffer instead of 660
- [M-040](#m-040-the-gemv-is-bound-by-values-unpacked-not-bytes-moved) The GEMV is bound by values unpacked, not bytes moved
- [M-039](#m-039-per-head-norms-and-an-estimate-wrong-by-45) Per-head norms, and an estimate wrong by 4.5×
- [M-038](#m-038-fusing-the-chain-works-and-gpu-busy-time-makes-it-measurable) Fusing the chain works, and GPU-busy time makes it measurable
- [M-037](#m-037-the-bandwidth-ceiling-and-what-the-seven-failures-actually-meant) The bandwidth ceiling, and what the seven failures actually meant
- [M-036](#m-036-overlapping-cpu-with-gpu-costs-45--on-a-cold-machine-twice-tried) Overlapping CPU with GPU costs 45 %, on a cold machine, twice tried
- [M-035](#m-035-the-gpu-is-idle-46--of-decode-that-is-the-number-and-it-took-five-failures-to-find) The GPU is idle 46 % of decode. That is the number, and it took five failures to find
- [M-034](#m-034-the-cache-default-was-worth-14--batching-the-experts-was-worth-nothing) The cache default was worth 14 %; batching the experts was worth nothing
- [M-033](#m-033-dropping-the-wait-on-cb2-is-not-a-scheduling-change-it-is-a-correctness-change) Dropping the wait on `cb2` is not a scheduling change, it is a correctness change
- [M-032](#m-032-the-shared-branch-overlap-paired-against-its-own-control-30-) The shared-branch overlap, paired against its own control: −30 %
- [M-031](#m-031-the-decode-bottleneck-is-dispatch-latency-and-one-attempt-at-it-failed) The decode bottleneck is dispatch latency, and one attempt at it failed
- [M-030](#m-030-gemma-4-at-4-bits-34x-decode-32x-less-read-same-answer) Gemma 4 at 4 bits: 3.4x decode, 3.2x less read, same answer
- [M-029](#m-029-batched-prefill-for-gemma-225-and-where-the-rest-of-the-time-is) Batched prefill for Gemma: 2.25×, and where the rest of the time is
- [M-028](#m-028-gemma-4-26b-a4b-on-real-weights-correct-and-slow-for-a-known-reason) Gemma 4 26B-A4B on real weights: correct, and slow for a known reason
- [M-001](#m-001-the-repacker-holds-the-memory-invariant) The repacker holds the memory invariant
- [M-002](#m-002-large-requests-beat-splitting-by-a-factor-of-64) Large requests beat splitting, by a factor of 6.4
- [M-003](#m-003-zero-copy-mapping-really-is-free) Zero-copy mapping really is free
- [M-004](#m-004-parallel-expert-reads-are-worth-a-factor-of-18-to-20) Parallel expert reads are worth a factor of 1.8 to 2.0
- [M-005](#m-005-the-gemv-kernel-reaches-52--of-memory-bandwidth) The GEMV kernel reaches 52 % of memory bandwidth
- [M-006](#m-006-the-measurement-error-45-s-of-synchronization-per-command-buffer) The measurement error: 45 µs of synchronization per command buffer
- [M-007](#m-007-two-plausible-optimizations-that-do-not-pay) Two plausible optimizations that do not pay
- [M-008](#m-008-extrapolating-the-moe-alone) Extrapolating the MoE alone
- [M-009](#m-009-the-complete-layer-matches-the-reference-on-first-assembly) The complete layer matches the reference on first assembly
- [M-010](#m-010-the-router-boundary-forces-two-command-buffers-per-layer) The router boundary forces two command buffers per layer
- [M-011](#m-011-prefill-campaign-180-s--56-s-and-four-wrong-hypotheses) Prefill campaign: 18.0 s → 5.6 s, and four wrong hypotheses
- [M-012](#m-012-io-overlap-does-not-pay) I/O overlap does not pay
- [M-013](#m-013-the-decoding-bottleneck-is-not-io) The decoding bottleneck is not I/O
- [M-014](#m-014-fusing-command-buffers-145) Fusing command buffers: ×1.45
- [M-015](#m-015-two-kernel-rewrites-with-no-effect) Two kernel rewrites with no effect
- [M-016](#m-016-top-p-sampling-cost-more-than-the-lm-head) Top-p sampling cost more than the LM head
- [M-017](#m-017-why-the-kv-cache-is-not-reused-across-turns) Why the KV cache is not reused across turns
- [M-018](#m-018-the-eight-slot-lead-was-the-lead-of-the-overheads-it-masked) The eight-slot lead was the lead of the overheads it masked
- [M-019](#m-019-the-prefill-on-the-other-hand-really-is-io-bound) The prefill, on the other hand, really is I/O-bound
- [M-020](#m-020-reusing-the-kv-cache-across-turns-time-to-first-token--6) Reusing the KV cache across turns: time to first token ÷ 6
- [M-021](#m-021-the-120b-is-in-the-inverse-regime-of-the-20b) The 120B is in the inverse regime of the 20B
- [M-022](#m-022-readcompute-overlap-has-nothing-to-overlap) Read/compute overlap has nothing to overlap
- [M-023](#m-023-where-the-headroom-stands-on-this-machine) Where the headroom stands on this machine
- [M-024](#m-024-compute-first-what-is-already-there) Compute first what is already there
- [M-025](#m-025-speculative-decoding-attacking-arithmetic-intensity) Speculative decoding: attacking arithmetic intensity
- [M-026](#m-026-q8-on-the-dense-weights-the-per-position-gate-passes-the-decision-does-not) Q8 on the dense weights: the per-position gate passes, the decision does not
- [M-027](#m-027-q8-on-the-dense-weights-built-measured-removed) Q8 on the dense weights: built, measured, removed

---

## M-067, Long context: Qwen is the fastest model short and the slowest long
**2026-08-15, M4, 24 GiB, 8 slots, 32k context allocated, 4-bit builds**

Never measured before on any model but Gemma. The prediction going in was that Qwen would hold up
best, because three of its four layers are a recurrence with a fixed state and no key/value cache
at all. **It holds up worst.**

| model | ~1.4k | ~5.4k | ~10.6k | ~21k | change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B Q4 | **7.20** | 6.95 | 5.82 | **3.74** | **-48 %** |
| Gemma 4 26B-A4B Q4 | 6.09 | 6.59 | 6.18 | 4.38 | -28 % |
| GPT-OSS 20B | 7.61 | 7.21 | 6.29 | **5.52** | **-27 %** |

Qwen is the fastest of the three at short context and the slowest at long. The second prediction
was wrong too: GPT-OSS, which has the most unbounded layers, decays **least**.

### What decides it is the sliding window, not the recurrence

| model | layers scanning the whole history | the rest |
| --- | --- | --- |
| Qwen | **10 of 40, unbounded** | 30 recurrent, fixed state |
| Gemma | 5 of 30 | 25 bounded at 1024 keys |
| GPT-OSS | 12 of 24 | 12 bounded at **128** keys |

`sliding_window` is **absent from Qwen's config**: every one of its ten attention layers reads
the entire history every token. Gemma bounds 25 layers of 30 and GPT-OSS bounds half of its at a
very tight 128.

So the recurrence buys **memory and nothing else**. The fixed 63 MB state means the model does
not grow with the conversation, and the ten attention layers still scan all of it. Saying Qwen is
"structurally best at long context", which this file nearly did, confuses the two.

In added milliseconds a token, 1.4k to 21k: **Qwen +128 ms**, Gemma +64, GPT-OSS +50.

### The kernel that does it, and it is 12x off

At 21k, Qwen's ten attention layers read 10 x 21000 x 512 x 2 x 2 = **430 MB of key/value a
token**, against 1619 MB of weights. At the ~40 GB/s the weight GEMVs achieve that would be
10.8 ms. The measured cost of the context is **128 ms**.

`attention_decode` is running about **12x off** the rate the rest of the model gets. The
standalone bench says the same thing from the other side: 486 µs for 16 heads over 1024 keys is
about 4 GB/s.

**This is the first identified kernel inefficiency with a large measured cost**, after a session
of finding none. Unlike the expert GEMVs, which are at the machine's limit, this one is nowhere
near it.

### Caveat on the numbers

Each decode figure is 8 generated tokens, 24 to 32 for GPT-OSS, because the prompt asks for one
sentence. Gemma reads 6.09, 6.59, 6.18 across the first three lengths, so there is about
±0.5 tok/s of noise and the small differences below 10k mean nothing. The drops at 21k are far
outside it. Prefill degrades on the same curve and more steeply: Qwen 46 to 23 tokens/s, Gemma 36
to 21, GPT-OSS 65 to 29.

---

## M-066, Decode's expert reads are real I/O, and two attempts to remove them failed
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 300 tokens, 4k context**

M-065 left decode's gap unexplained: 1619 MB of weights a token, which at the machine's 88 GB/s
would be 54 tok/s, against 12 measured, with the kernels already at bandwidth and the dispatches
costing 1.5 ms. The remaining suspect was `expert I/O`, 25 to 33 ms of an 80 ms token.

### Slots raise the hit rate and do not raise throughput

Order alternated, three pairs:

| slots | cache hits | expert I/O | decode |
| ---: | ---: | ---: | ---: |
| 8 | 68 % | 30.5 ms | 12.35 tok/s |
| 64 | **94 %** | 25.6 ms | 12.50 tok/s |

Hits go from 68 % to 94 %, the I/O bucket falls 5 ms, and throughput moves **0.15 tok/s**. This
is M-055 again from a different angle, and this time with the hit rate visible: the misses are
genuinely cheaper than they look, because most of them are served by the OS file cache rather
than the SSD.

### `concurrentPerform` was not the overhead, and assuming it was cost 10 %

25 ms a token at a 94 % hit rate looked impossible for reads that mostly hit, so the suspicion
fell on `load`, which sends every expert through `DispatchQueue.concurrentPerform`: eight
thread-pool handoffs to do eight dictionary lookups, forty times a token.

Filtering to the actually-missing experts first, and pinning the rest serially, is **9.6 % slower
and lost all six pairs**: 12.09 tok/s against 10.93.

The filter added a residency check per expert, each taking the cache's lock, and then a second
serial pass to pin, taking it again. **The lock traffic doubled.** `concurrentPerform` was not
the cost; the change that was supposed to remove overhead added more of it than it saved.

Reverted.

### What the two results leave

The I/O bucket is real reading, not bookkeeping: at 68 % hits a token misses about 100 experts
and moves 177 MB. But raising the hit rate to 94 % does not pay, which means the bucket is
substantially overlapped with GPU work already, or the misses it removes were the cheap ones.

**Nothing further should be attempted here without a measurement that separates a hit's cost
from a miss's.** Three sessions have now guessed at this bucket and been wrong each time.

---

## M-065, Decode's dispatches are not the cost, and the GEMV is already at bandwidth
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 8 slots, 4k context**

M-059 counted 2491 dispatches a token and named the routed experts, about 1600 of them, as the
target. This is what happened when that was tested.

### The dispatch widths, which M-061 says are the number to watch

| kernel | a token | threadgroups |
| --- | ---: | ---: |
| `mlx_affine_gemv_4` | 1236 | 187 |
| `qwen_silu_multiply` | 350 | **1** |
| `write_expert_scaled` | 311 | **2** |
| `add_inplace` | 117 | **2** |
| `rms_norm` | 79 | **1** |

About **900 dispatches a token at one or two threadgroups**, exactly the shape that cost prefill
30 %. Apple's guidance agrees: batch small work into one kernel, because a launch stalls the GPU.

### Batching them changed nothing

The expert path was restructured to run phase by phase rather than expert by expert: all eight
gates, all eight ups, **one** activation over the whole set, all eight downs, **one** scaling.
`qwen_silu_multiply` went from 350 dispatches a token to 78 and `write_expert_scaled` from 311 to
39, a total of 2491 to 1946.

Six pairs, warm, **with the order alternated so position could not bias it**: 11.69 tok/s against
11.87. Nothing. And the binary that ran *second* in a pair usually won, whichever it was.

### Nor does consolidating the expert GEMVs

Eight separate `[512 x 2048]` dispatches against one `[4096 x 2048]`, the same arithmetic and the
same bytes: **0.257 ms against 0.262 ms**. Consolidating the eight is worth **minus two percent**.

### Why, and the number that was already in the repository

`hydra bench-gemv` has measured this all along and was not consulted before the session spent a
day on dispatch counts:

**A dispatch costs 0.6 microseconds on this machine.** A command buffer costs 0.9. At 2491
dispatches a token that is 1.5 ms of an 83 ms token, under two percent. Apple's published figure
of "as much as 10 microseconds" is an upper bound for a different situation, and taking it as the
local cost is what made this look promising.

So M-061's router win was **not** launch overhead. It was 62,400 one-threadgroup dispatches in a
serial chain with nothing to overlap them, which is underutilization. In decode the narrow
kernels sit between wide GEMVs that fill the machine, and the GPU covers them.

### And the same bench says the GEMV is already at the ceiling

| shape | GB/s | share of the machine's 88 GB/s |
| --- | ---: | ---: |
| `q_proj` 4096 x 2816 | 99 | 113 % |
| head 262144 x 2816 | 94 | 107 % |
| `expert.gate` 704 x 2816 | 87 | 100 % |
| **`expert.down` 2816 x 704** | **45** | **52 %** |

Over 100 % is cache. These kernels are not slow; they are streaming at the machine's limit.

### The one real inefficiency, and it is narrow matrices

The kernel strides `uint4` chunks across a 32-lane simdgroup, so a row needs 32 chunks, meaning
1024 columns at 4 bits, to occupy the simdgroup:

| matrix | columns | chunks a row | simdgroup busy |
| --- | ---: | ---: | ---: |
| Qwen `expert.gate` / `up` | 2048 | 64 | 100 % |
| Gemma `expert.down` | 704 | 22 | 69 % |
| **Qwen `expert.down`** | **512** | **16** | **50 %** |

Qwen's `down` projection leaves **half the simdgroup idle**, and it is a third of the expert
bytes. That is the first identified inefficiency in decode that is neither dispatch count nor
bandwidth, and it is not yet fixed.

### The byte budget, for whatever comes next

| | MB a token | share |
| --- | ---: | ---: |
| linear-attention projections, 30 layers | 568 | 35 % |
| routed experts, 8 of 256, 40 layers | 566 | 35 % |
| LM head | 286 | 18 % |
| full-attention projections, 10 layers | 106 | 7 % |
| shared expert, 40 layers | 71 | 4 % |
| **total** | **1619** | |

At the machine's 88 GB/s that is 18.4 ms a token, or **54 tok/s**. Decoding runs at 12. The gap
is not the kernels and not the launches, so it is the expert reads and the time the GPU spends
idle waiting for them, which is where the next measurement belongs.

---

## M-064, Prefill is at a plateau, and the two remaining costs trade against each other
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 1560-token prompt**

After M-061 and M-062, prefill is **34.5 s with GPU busy 24.6 s**. Four candidates were measured
and none of them moves it.

| candidate | result |
| --- | --- |
| 24 expert slots instead of 8 | 34.2 s to 32.9 s, for 3x the cache memory (553 MiB to 1.66 GiB) |
| chunk 1024 instead of 512 | 34.6 s to 34.9 s |
| tile `RB 2, TB 16` instead of `4 x 8` | 34.6 s to 34.1 s, **GPU busy identical** |
| deeper prefetch | not built: needs a third of the slots a batch, so batches of 2 |

### The two costs trade

Chunk 1024 halves the expert I/O wait, 7.17 s to 3.52, and **raises GPU busy by 5 s**, 24.6 to
29.6. The activation tile is re-read `rows / RB` times by the batched projection, and at 512
tokens it is 4 MiB and stays in cache while at 1024 it is 8 MiB and does not. Buying back the
I/O wait costs more GPU than it saves.

512 is not a tuned optimum, it is where the two curves cross.

### An isolated kernel time is not a production budget

`bench-delta` times one linear layer's kernels alone and reports **107 ms of projections against
20 ms of recurrence**, which over 30 layers and 3 chunks would be about 9.8 s of the 24.6 s the
GPU is busy. A tile measured **1.9x faster in that bench**, 60.8 ms against 113.3, changed GPU
busy by nothing at all: 24.7 s against 25.0.

So the bench overstates. A kernel run alone pays its own ramp-up and drain; the same kernel in a
stream of hundreds has both hidden by its neighbours. **Per-kernel timings do not sum to a
production profile**, and this file has now been wrong about that in both directions: M-040
warned that an aggregate hides the kernels, and this is the converse.

What survived from the bench is the *ordering*, which is worth something: the recurrence is not
the cost, the projections are. What did not survive is the size.

### Where that leaves prefill

The floor for this architecture is around 34 s, at 71 % GPU busy. Below it needs less GPU work
rather than better scheduling, which means the projections themselves, and the tile is not the
lever. Nothing further was attempted, because every remaining idea is in the class this session
has been wrong about five times.

---

## M-063, The GEMM tile does not matter, and three measurements said it did
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4 and Gemma 4 26B-A4B Q4**

After M-062, prefill's remaining GPU cost is the projections. Benchmarked alone at Qwen's shape
and chunk, one linear layer is **107 ms of projections against 20 ms of recurrence**:

| | ms a layer, 512 tokens |
| --- | ---: |
| `in_proj_qkv`, 8192 x 2048 | 55.1 |
| `out_proj`, 2048 x 4096 | 28.2 |
| `in_proj_z`, 4096 x 2048 | 21.9 |
| the delta rule | 19.1 |
| `in_proj_a` and `_b`, 32 x 2048 | 2.2 |
| staging, conv, gated norm | 1.8 |

M-045 tuned the tile to `RB = 4, TB = 8` by sweeping at **128 tokens on Gemma's narrower
matrices**. Qwen runs 512 tokens against an 8192-row projection, so the sweep was worth redoing.

**It was worth redoing three times, and only the third was measuring anything.**

### Attempt one: 3.6x, from a kernel reading out of bounds

Changing `kBatchTile` in the shader reported `RB = 2, TB = 16` at 30.5 ms against 110.4. The tile
is **two constants in two languages**: `ForwardEncoder.batchTile` sizes the transposed activation
buffer, and it was still 8. The kernel read 16 tokens at a time from a buffer padded for 8, past
the end of every column. The 3.6x was undefined behaviour.

`MLXAffineKernelTests`, which compares the batched projection against the per-token one **bit for
bit**, failed immediately once both constants moved. That test is the only reason this was caught
before it reached a measurement anyone believed.

### Attempt two: 2x slower, from interleaving across two models

With both constants moved, alternating 4 x 8 and 2 x 16 gave Qwen 13.9 s against 29.0 and Gemma
23.7 against 42.6. Clear, consistent over three pairs, and wrong.

The loop ran Qwen, Qwen, Gemma, Gemma per iteration. **The two models evict each other's pages**,
so the arm that ran first in each iteration met a different cache than the arm that ran second,
and the position in the loop, not the tile, decided the result. The same 4 x 8 binary read
**13.8 s in that loop and 35.4 s in a sweep**, a factor of 2.5 on identical code.

### Attempt three: no difference

Alternating the two tiles on **one model only**, four pairs:

| | 1 | 2 | 3 | 4 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| RB 4, TB 8 | 34.6 | 34.4 | 33.8 | 34.1 | **34.2 s** |
| RB 2, TB 16 | 34.4 | 34.0 | 33.7 | 33.9 | **34.0 s** |

**0.2 s.** The tile does not matter end to end at these shapes, and 4 x 8 stays because it is the
tested, committed value.

### What this costs and what it teaches

Interleaving is not sufficient. It removes drift that is *independent* of the arms, and does
nothing about **arms that interact through shared state**. Two models sharing a page cache
interact; so would two configurations sharing anything else warmed by use.

The protocol that survived all three attempts: **one model, one variable, alternating, and read
the column down**. A byte count beside the time is worth more than another repetition, and here
it was flat at 39.92 GiB throughout, which by itself said the tile was not changing the work.

---

## M-062, Reading the next batch while the GPU runs this one: another 18 %
**2026-08-14, M4, 24 GiB, 1560/1579-token prompts, 8 slots, 8k context**

After M-061 the buckets moved again. Qwen's prefill was 41.9 s with **GPU busy 59 %**, and
`expert I/O` was 14.7 s of it: the CPU blocked in `pread` with the GPU holding nothing.

The loop was: load a batch, encode it, commit, **wait**, release, load the next. The read and the
execution never overlapped although they need nothing from each other.

Now the read for batch N+1 is issued between committing N and waiting on it. Two things had to
change to allow it:

- **Half the slots a batch**, so two batches are resident at once. `load`'s parallel `pread`
  issues four at a time instead of eight, and M-058 measured that width to make no difference
  on this machine.
- **A per-expert release.** `release(layer:)` unpins everything, which would unpin the batch the
  prefetch had just filled and let the batch after it evict a slot the GPU had not yet read.
  `release(layer:experts:)` unpins the finished batch alone.

| | pair 1 | 2 | 3 | mean |
| --- | ---: | ---: | ---: | ---: |
| Qwen, blocking | 42.0 | 41.4 | 41.7 | **41.7 s** |
| Qwen, overlapped | 33.9 | 34.0 | 34.1 | **34.0 s** |
| Gemma, blocking | 54.0 | 53.4 | 53.5 | **53.6 s** |
| Gemma, overlapped | 43.5 | 43.4 | 43.4 | **43.4 s** |

**Qwen -18.5 %, Gemma -19.0 %**, every pair, spread under 0.7 s. `expert I/O` fell from 14.7 s to
7.2 and GPU busy rose from 59 % to 71 %.

### The session, end to end

| | before | after | |
| --- | ---: | ---: | ---: |
| Qwen 3.6 35B-A3B Q4 | 59.7 s | **34.0 s** | **-43 %** |
| Gemma 4 26B-A4B Q4 | 61.5 s | **43.4 s** | **-29 %** |

Two changes, both structural, neither touching a kernel's arithmetic: put the token on the grid
where a dispatch had one threadgroup, and stop waiting for a read that nothing depends on yet.

### What is left in prefill

Qwen at 34 s: GPU busy 24.4 s, so 10 s of CPU remains, of which `expert I/O` is 7.2. The reads
are now hidden behind GPU work only to the extent the GPU has work; when the batch is small the
read is longer than the execution and the wait reappears. Prefetching **two** batches ahead needs
a third of the slots each and is the obvious next step, with the same caveat M-055 raises: the
slots cost memory and this is the first thing that has made them buy anything.

---

## M-061, One dispatch a token for the router cost 30 % of Qwen's time to first token
**2026-08-14, M4, 24 GiB, 1560-token prompt, 8 slots, 8k context**

M-060 attributed prefill's 60 s to three buckets and found the token mixer, not the experts,
carried 36 s of it. Splitting that bucket further, with the phase committed separately so the GPU
time could be read off each half:

| | GPU seconds |
| --- | ---: |
| the token mixer | 16.9 s |
| **the router and the shared branch** | **19.4 s** |

The recurrence, benchmarked alone at prefill's shape (`hydra bench-delta`), is 17.7 ms a layer a
chunk, 94 % of a recurrent layer's kernels and **2.3 s over the whole prompt**. So the mixer's 17 s
is projections, and the router's 19.4 s was the surprise.

`router_topk` dispatches **one threadgroup**. Prefill called it once a token a layer: **62,400
dispatches**, each a point where the GPU has a single threadgroup of work in flight. The batched
kernel, one threadgroup a token, already existed for GPT-OSS and this path had never used it.

**The router bucket went from 19.4 s to 1.2 s.**

### Paired

| | pair 1 | 2 | 3 | mean |
| --- | ---: | ---: | ---: | ---: |
| Qwen, per token | 60.2 | 59.7 | 59.2 | **59.7 s** |
| Qwen, batched | 41.4 | 41.4 | 41.7 | **41.5 s** |
| Gemma, per token | 61.5 | 61.6 | 61.4 | **61.5 s** |
| Gemma, batched | 53.1 | 53.3 | 53.6 | **53.3 s** |

**Qwen -30.5 %, Gemma -13.3 %**, every pair, spread under 1 s. Gemma gains less: 128 experts
against 256 makes each top-k cheaper, and it has 30 layers to Qwen's 40.

Gemma's result is bit-identical, which its own test asserts: "Staged prefill matches
token-by-token decoding exactly" compares the staged path against sequential decoding without a
tolerance, and the batched kernel is the single-token one with the token on the grid.

### Why this one worked when the other four did not

M-060 removed 950,000 dispatches and changed nothing. This removed 62,400 and took a third off the
prompt. The difference is not the count, it is **what the GPU can do while the dispatch runs**.
The 950,000 were `copy_buffer` over 2048 elements, wide enough to fill the machine and to
pipeline behind each other. A one-threadgroup dispatch cannot be filled by anything, and 62,400
of them in sequence is 62,400 opportunities for the GPU to be almost idle.

**The number to watch is not dispatches, it is threadgroups a dispatch.** A grep for dispatches
that issue one threadgroup is a better optimization list than a profile sorted by count, and
nothing in this project was looking at it.

---

## M-060, Prefill issued 1.18 million dispatches; removing 950,000 of them changed nothing
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 1560-token prompt, 8 slots, 8k context**

Prefill had no instrumentation at all, which made it the one part of the system where the wait a
user actually notices was unaccounted for. Counting:

**1,176,851 dispatches for 1560 tokens, 754 a token.** Two kernels were 85 % of them:

| kernel | count |
| --- | ---: |
| `copy_buffer` | 499,361 |
| `write_expert_scaled` | 499,200 |
| `mlx_affine_gemm_4` | 73,906 |
| `router_topk` | 62,400 |

Both are the grouped expert path gathering its members' rows and scattering the results, **one
dispatch a member**, 8 KiB of work behind each launch. `gather_rows` and `scatter_expert_scaled`
do each in one dispatch for the whole group.

**1,176,851 to 226,895, an 81 % cut. Prefill went from 61.3 s to 60.6 s.**

That is inside the run-to-run spread, and GPU busy did not move either, 72 % to 71 %. Nine
hundred and fifty thousand dispatches were worth about one percent.

The dispatches were nearly free because they are encoded into shared command buffers and the GPU
pipelines them; the cost of a launch is not the launch. This is the fourth idea in a row that
looked structural and measured empty, after the expert slots (M-055), the mapped reads (M-058),
and the merged command buffers whose throughput claim did not survive pairing (M-053).

**The change is kept anyway**, on the same grounds M-052 used: a million dispatches to move 8 KiB
each is work that does not need doing, and the batched form is simpler to read. It is not
presented as a speed-up.

### Where prefill's time actually goes

The buckets were in the output the whole time and say something the dispatch count does not:

| phase | seconds | share |
| --- | ---: | ---: |
| `cb1`, the token mixer and the router | **36.3 s** | 60 % |
| expert I/O, the CPU's `pread` | 14.6 s | 24 % |
| the experts themselves | 10.1 s | 17 % |

**The experts are not the problem. The token mixer is**, and three layers in four of this model
are the recurrence. `qwen_delta_rule_chunk` dispatches 32 threadgroups of 128 threads and loops
512 tokens inside the kernel, which is 4096 threads on a 10-core GPU carrying a sequential loop.
That is the next thing to measure, and this time in isolation before anything is changed.

---

## M-059, 2491 dispatches a token, and two thirds of them are the routed experts
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 8 slots, 4k context**

M-053 measured the GPU idle for a quarter to a half of every token and named dispatch count as
the next target without ever counting them. `ForwardEncoder` now counts, at the one point every
dispatch passes through.

**2491 a token**, 89,670 over 36 tokens:

| kernel | a token | what it is |
| --- | ---: | --- |
| `mlx_affine_gemv_4` | 1236 | every 4-bit projection |
| `qwen_silu_multiply` | 350 | 8 routed experts + 1 shared, 40 layers |
| `write_expert_scaled` | 311 | one a routed expert a layer |
| `add_inplace` | 117 | residuals |
| `rms_norm` | 79 | two a layer |
| `mlx_affine_gemv_8` | 78 | the router and the shared gate, 8-bit |

The 1236 quantized projections decompose as **960 routed experts** (8 experts x 3 matrices x 40
layers), 120 shared expert, 150 linear-attention projections, 40 attention. So the routed
experts alone are 960 + 350 + 311, near enough **1600 of 2491, about two thirds**.

### What follows

Every one of those 1600 is a separate launch that computes one expert's one matrix against one
token. They are independent, identically shaped, and issued back to back. **One dispatch per
matrix over all eight experts** would take that 1600 to roughly 200, a 56 % cut in total
dispatches, and would raise the threadgroup count per launch eightfold on kernels that are
currently narrow.

The ceiling on it: GPU busy is 69 % here, so the whole CPU share is about 20 ms of a 63 ms
token, and not all of that is encoding. This is worth single-digit milliseconds a token, not a
factor. It is still the largest identified item, and unlike the expert cache and the mapped
reads it has not yet been measured and found empty.

---

## M-058, Cold, mapping is 17x slower, and the prefetch does not rescue it
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q8, `hydra bench-cold`**

M-056 measured warm and found mapping 31 % faster with no copy, and proposed a hybrid: map for
the read, keep an explicit `madvise(MADV_WILLNEED)` so the faulting is batched the way
`load(layer:experts:)` batches its reads. This is that experiment. **The hybrid does not work
and the rewrite it was pointing at should not happen.**

32 blobs of 3.2 MiB, 102 MiB a sample, each sample on its own layer file:

| scheme | cold | GB/s |
| --- | ---: | ---: |
| `pread`, serial | 30 ms | **3.54** |
| `pread`, 8 at once | 32 ms | 3.38 |
| mapped, faulting | 529 ms | 0.20 |
| mapped + `MADV_WILLNEED` | 512 ms | 0.21 |

**17x.** And `MADV_WILLNEED` moves it by 3 %, which is nothing: the advice is advisory, and the
GPU still faults synchronously when it touches a page. This reproduces the inherited ratio,
0.50 tok/s against 3.97, from the other end and confirms it.

### Coldness was verified, not hoped for

`mincore` reports which pages of a mapping are resident, and a layer was used only if under 10 %
of the span it would touch already was. **20 of 32 candidate layers were rejected as warm.**
Without that check this would have quietly become a second warm benchmark, which is the confound
that has produced four retracted results in this file.

### What it means for the memory question

Mapping is only viable where residency is guaranteed, and guaranteed residency is pinning. So
the shape that survives is not "map instead of copy" but **map the part the user chooses to pin,
`pread` the rest**, which is a much larger change for a warm-path saving of 0.12 ms an expert.

The arithmetic that decides it: 320 expert fetches a token, so the warm saving is worth about
38 ms a token if every fetch hits. One cold fetch costs 15 ms extra. **At 5 % cold the mapped
path is already 6x behind.** A design whose advantage evaporates at a 5 % miss rate is not one to
build on a machine whose file cache holds 7 GiB of a 53 GiB working set.

`ExpertSlotCache` keeps `pread`. The decision at the top of that file was inherited and is now
measured here, at both temperatures.

### A second inherited claim that did not reproduce

The same comment credits parallelism: 3.0 GB/s for a single read against 5.5 for four or more.
Here **serial and eight-at-once are the same**, 3.54 against 3.38 GB/s, with serial marginally
ahead. At 3.2 MiB a read one stream already saturates this SSD.

That does not make the parallel path wrong, and it is not being removed on one measurement at
one blob size. It does mean the reason written beside it no longer describes this machine, and
the 8-wide batching in `load(layer:experts:)` is buying nothing measurable here.

---

## M-057, Qwen's prefill chunk was Gemma's, and it cost 41 % of the bytes
**2026-08-14, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, 1560-token prompt, 8 slots, 8k context**

`QwenPrefillRunner.chunk` was 256, taken from Gemma's measured optimum (M-046) with a comment
saying it had not been measured here. It had no reason to transfer: Gemma has 128 experts a
layer and Qwen has **256**, so a chunk of 256 tokens does not saturate the union and the pool is
re-read on every chunk.

| chunk | prefill | read from SSD | prefill arena |
| ---: | ---: | ---: | ---: |
| 256 | 67.8 s | 67.62 GiB | 92 MiB |
| **512** | **59.9 s** | **39.92 GiB** | 184 MiB |
| 1024 | 60.5 s | 25.57 GiB | 368 MiB |

Interleaved, three pairs of 256 against 512: 68.4/59.8, 67.4/59.9, 67.6/60.1. Every pair favours
512 and the spread is under 0.5 s. Prefill resolves what decode cannot, because the run is long
and dominated by bulk work rather than by per-token overhead.

**The bytes and the clock stop agreeing after 512.** 1024 halves the reads again and buys no
time, because the reads it saves come from the OS file cache and not from disk (M-055, M-056).
So the byte count keeps falling past the point where it matters, and the arena doubles with it.
512 takes the whole time saving at half the memory of 1024.

### The bug underneath, which the byte count found

The first attempt measured **67.62 GiB at all three chunk sizes, identical to the last digit**.
That is the signature of a number that does not depend on the run, and it was:
`Qwen35MoeRunner` accepted a `prefillChunk` parameter and constructed `QwenPrefillRunner`
without passing it. Ignored since the day it was added.

**The boundary test passed throughout.** "Chunked prefill matches token-by-token decoding across
chunk boundaries" asked for a chunk of 4 against 10 tokens and got 256, so it never crossed a
boundary, and it agreed anyway, because one chunk and three chunks produce the same answer by
construction. A test whose whole subject is a boundary, never reaching one, and green.

It now asserts `prefillChunkTokens == 4` before running, which is the cheapest possible check
that the thing under test is the thing configured.

### The constant that could not stay a constant

`InferenceEngine` slices prefill into 256-token pieces so the stop button works during a long
prompt, with a comment: *it must not be below the prefill runner's own chunk, or the batching
never sees a full one and the expert reads are paid twice*. True, and unenforceable while both
were constants that happened to match. Qwen at 512 would have been sliced back to 256 by the
application and none of this would have reached a user.

Runners now report `prefillChunkTokens` and the engine takes the larger of the two.

---

## M-056, Mapping the experts beats copying them, warm, by 31 % and all the memory
**2026-08-13, M4, 24 GiB, Qwen 3.6 35B-A3B Q4, `hydra bench-map`**

`ExpertSlotCache` copies: `pread` from the layer file into a preallocated slot, then the GPU
reads the slot. The decision is recorded at the top of that file and was **inherited**, not
measured here: `mmap` at 0.50 tok/s against `pread` at 3.97, because demand paging gives no
control over when reads happen or how many run at once.

That is a claim about **cold** reads, and M-055 found this machine no longer reads cold.

32 experts of one layer, read both ways, interleaved, each forced through a real projection so
the GPU actually touches the bytes:

| pair | pread + GPU | of which pread | mapped GPU |
| ---: | ---: | ---: | ---: |
| 1 | 16.5 ms | 5.0 | 10.0 |
| 2 | 12.7 | 4.0 | 8.9 |
| 3 | 11.4 | 3.4 | 8.1 |
| 4 | 10.7 | 3.1 | 7.9 |
| 5 | 10.7 | 3.1 | 7.3 |
| 6 | 10.4 | 3.0 | 7.6 |
| **mean** | **12.1 ms** | 3.6 | **8.3 ms** |

**Mapped is 31 % faster**, 0.26 ms an expert against 0.38. Both columns fall together as the
pairs go on, which is the page cache warming, and interleaving is what keeps that from being
mistaken for the result.

Two things beside the time:

**The `pread` runs at 18 GB/s.** 55 MiB in 3.0 ms is RAM bandwidth, not storage. There is no
disk in this measurement at all, which is the direct confirmation of M-055's mechanism.

**The process footprint is 18 MiB.** The mapped form holds no copy: the pages it reads are the
OS file cache's own, shared rather than duplicated. The copying form needs a slot per expert it
wants to keep.

### What this does not say

It is **warm only**. Cold is exactly where the inherited measurement applies, a fault being
serial and one page wide against eight `pread`s issued at once, and nothing here tests it. A
design that took this result and dropped `pread` would be fast on a second run and slow on the
first, which is the run a user notices.

The shape that follows from both results is a hybrid: map for the read, and keep the explicit
parallel prefetch as `madvise(MADV_WILLNEED)` on the selected blobs, so the faulting is batched
the way `load(layer:experts:)` batches its reads. That is not built and not measured.

---

## M-055, The expert cache is a cache on top of a cache, and that is why slots do nothing
**2026-08-13, M4, 24 GiB, Qwen 3.6 35B-A3B Q4 and Gemma 4 26B-A4B Q4, 4k context**

The question that prompted this: if the whole expert pool fits in memory, there is no SSD read
left, so decoding should reach the compute ceiling, 48 tok/s for Qwen and 39 for Gemma. It does
not come close. Why?

### First, what unpaired runs said

| model | slots | cache hits | decode | GPU busy |
| --- | ---: | ---: | ---: | ---: |
| Qwen | 8 | 69 % | 9.50 tok/s | 76 % |
| Qwen | 64 | 92 % | 14.48 tok/s | 62 % |
| Qwen | 200 | 94 % | 11.16 tok/s | 56 % |
| Gemma | 8 | 69 % | 7.44 tok/s | 24 % |
| Gemma | 128 (whole pool) | 97 % | 11.60 tok/s | 54 % |

Which reads as a clean story: more slots, more hits, faster, with an interior optimum near 64.

**It is an artifact for the third time today.** Interleaved, 8 against 64 on Qwen:

| pair | 8 slots | 64 slots |
| ---: | ---: | ---: |
| 1 | 9.55 | 10.52 |
| 2 | 9.52 | 12.34 |
| 3 | **14.32** | 12.45 |
| 4 | **13.67** | 12.63 |
| mean | 11.77 | 11.99 |

**+0.22 tok/s, two pairs each way.** And look down the 8-slot column: 9.55, 9.52, 14.32, 13.67.
It got faster as the session went on, which no slot count explains and page-cache warming does.
The first table compared a cold configuration against a warm one, run in that order, twice.

### The mechanism, confirmed directly

`vm_stat` during these runs: **8.2 GiB of file-backed pages resident**, against 4.8 GiB free.
macOS is holding most of the expert files in its own page cache.

So the expert slot cache is **a cache in front of a cache**. A miss does not cost a disk seek; it
costs a `pread` served from the page cache, which is a copy through RAM. Buying more slots buys
skipping a memcpy, and pays for it in GiB of duplicated data, which is why the curve is flat from
8 to 200 and why the footprint triples for nothing.

This retires the intuition behind the whole slot setting, which is worth stating plainly: **on a
machine whose RAM is comparable to the model, the OS is already doing the caching.** D-012's
"minimize memory, do not fill the ceiling" turns out to be right for a reason nobody wrote down.

### Why full residency still does not reach the ceiling

Gemma with all 128 experts resident and 97 % hits: **GPU busy 54 %**. Nearly half of every token
is spent with the GPU idle, and no amount of expert caching touches that half. The remaining
share is CPU: encoding the dispatches, and the one synchronization a layer where the router's
choice has to come back before the experts can be fetched.

So the ceiling figure of 39 or 48 tok/s is a bandwidth bound on the GPU alone, and this
implementation is nowhere near being bandwidth-bound. **The next thing to attack is dispatch
count and the per-layer round trip, not the storage path.**

### The rule, restated for the third time

Cold-to-warm is as strong a confound here as thermal drift, and it runs the same direction as
the hypothesis whenever the cheap configuration is tested first. Interleave, and read the column
down as well as across: a monotone trend inside one configuration is the tell.

---

## M-054, **Retracted**: the expert-slot effect was thermal drift
**2026-08-13, retracted the same day it was written**

This entry claimed 16 slots a layer beat the app's 8 by **+40 %** on Qwen, from 891 MiB to
1.44 GiB, and the README quoted it. It was wrong, and wrong in the way this file exists to
prevent: the two figures came from batches of runs taken minutes apart.

Interleaved in one thermal window, three runs each:

| slots | run 1 | 2 | 3 | mean |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 11.45 | 12.13 | 8.21 | 10.60 |
| 16 | 12.90 | 8.33 | 8.81 | 10.01 |
| 24 | 10.49 | 9.23 | 10.13 | 9.95 |

**There is no effect.** The spread inside one setting, 8.21 to 12.13 at eight slots, is larger
than any gap between settings. Cache hits do rise with slots, 66 % to 75 % to 78 %, and it buys
nothing measurable: the reads it saves are served from page cache, which M-018 and M-027 both
found on other models and this entry managed to forget.

The README's own preamble says throughput here moves 20 % with thermal state alone and that
every comparison in this file is run against a control timed beside it. The claim was published
anyway, against unpaired data, within an hour of writing that sentence down.

**The correction that generalizes**: on this machine, an unpaired throughput comparison of two
configurations is not evidence, whatever the gap. Interleave or do not publish.

The default stays at eight slots, now for a better reason than before: nothing measurable is
lost, and it is 550 MiB cheaper.

---

## M-053, Qwen decodes at 8 to 14 tok/s, and the round trips were the first thing wrong
**2026-08-13, M4, 24 GiB, Qwen 3.6 35B-A3B MLX 4-bit, 16 slots, short prompt**

The first working numbers from the real checkpoint, and the first structural fix, which was
found by measuring rather than by reasoning about the architecture.

### What the model does

19.00 GiB installed in 352 s at 58 MB/s, peak process memory **55.5 MiB**: the checkpoint never
exists in memory. Running, the footprint is **1.44 GiB** at 16 slots a layer, against 18.2 GiB on
disk. That is the thesis stated as a ratio: **8 % of the model resident**, and it is a better one
than Gemma 4 Q4's 1.83 of 14.6.

### The prediction was wrong in an instructive way

D-026 predicted 10 to 12 tok/s from bytes a token alone. The first measurement read **6.8 tok/s**
(one run, and single runs here are worth little, see below), with

```
GPU busy 83.8 ms of 190.0 ms wall (44 %), the rest is CPU
```

**The GPU was idle more than half the time.** No amount of thinking about 1.55 GiB a token
predicts that, and the bandwidth model cannot even express it. The instrumentation that says so
is four lines and should have been there before the first number was quoted, which is the same
conclusion M-031 to M-034 reached on the 20B and it did not carry over.

### The fix, which Gemma had already made

`QwenLayerRunner` committed and waited **twice a layer**: once for the mixer and the router,
once for the experts. Eighty round trips a token. Only the first wait earns its place, because
the router picks the experts and the CPU cannot `pread` them until it has read that choice back.
The second synchronizes nothing the GPU needs: layer N's experts and layer N+1's mixer are
sequentially dependent, and nothing between them touches the CPU, so they belong in one buffer
where the shared serial encoder already orders them.

Merging them is M-042 applied to a second architecture, and the layer runner had to give the
command buffers back to the caller to allow it.

### The throughput claim, corrected

This entry first reported **+31 %**, from four runs of one build followed by four of the other.
That is the unpaired comparison M-054 was retracted for, and it does not survive interleaving.

Twelve pairs, alternating the two binaries, same prompt, same slots, same thermal window:

| | mean | pairs won |
| --- | ---: | ---: |
| two buffers a layer | 10.67 tok/s | 4 of 12 |
| one buffer a layer | **11.75 tok/s** | 8 of 12 |

**+1.08 tok/s, about 10 %, and it does not clear the noise**: the paired differences have a
standard deviation of 2.6, giving t = 1.4 on 11 degrees of freedom, p ≈ 0.19. Four pairs go the
other way, one of them by 3.5 tok/s.

So the throughput number is **not a result**, and the change is not kept on it.

### What the change is kept on

Two things that are not throughput measurements.

**The round trips are a fact about the code**, not an inference: eighty commit-and-wait pairs a
token became forty-one. Only one wait a layer can matter, because only the router's choice has
to reach the CPU.

**GPU busy went from 44 % to 76 % of wall.** That is a ratio measured inside a token rather than
across a run, so it is far less exposed to thermal drift than tok/s, and a 32-point move is not
drift. It says the GPU was idle waiting for the CPU and now is much less so.

Both binaries built from a clean `swift build -c release` and alternated run by run.

### A build failure read as a result, again

The first attempt at the control stashed the runner but not the CLI that printed its timings, so
the release build **failed and the stale binary ran**. It produced 9.4, 14.7 and 15.5 tok/s,
which is the change measured against itself and would have been reported as "the merge does
nothing". Caught only because the command grepped for `error:` and printed the count.

M-050 recorded this exact failure four days ago and the harness built then covers the test suite,
not benchmark runs. **The rule that generalizes: never read a number from a binary you did not
watch build.**

### What is left

At 76 % busy the remaining CPU share is the expert `pread` (23 ms a token) and the encoding. The
next question is whether the delta rule's own kernel is efficient: it dispatches 32 threadgroups
of 128 threads, which is thin for a 10-core GPU, and nothing has measured it in isolation yet.
That is a guess, and it is exactly the kind this entry exists to stop being acted on.

---

## M-052, The prefill's expert slots never needed clearing, and the saving is 45 ms of 25 s
**2026-08-13, M4, 24 GiB, Gemma 4 26B-A4B MLX 4-bit, 740-token prompt, 8 slots, 8k context**

Phase B of `Gemma4PrefillRunner` zeroed `expertSlices` before the expert groups ran, in a
command buffer of its own that was committed and waited on, once per layer per chunk. The
groups built just above it enumerate every `(token, rank)` pair exactly once, every group is
processed, and `write_expert_scaled` **assigns** the whole slot rather than accumulating into
it, so every slot `sum_expert_slices` reads is written first. The fill was dead.

Removed, and replaced by the invariant checked in Swift: each slot claimed once, and all of
them claimed. Reproduce with
`hydra chat gemma-q4 "<740 tokens>" --tokens 1 --slots 8 --context 8192`.

### End to end, alternating control and change

| pair | 1 | 2 | 3 | 4 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| with the fill | 25.6 s | 25.4 s | 26.0 s | 26.2 s | 25.80 s |
| without | 25.4 s | 25.5 s | 25.7 s | 25.6 s | 25.55 s |

0.25 s, or 1 %, in the expected direction in three pairs of four. **That is not a result**: the
run-to-run spread of the control alone is 0.8 s, and prefill is reported to a tenth of a
second. Taken by itself this table says the change is not measurable end to end.

### What the removed work actually cost

So it was timed where it was, with an instrumented build that kept the fill and accumulated
`commit` + `wait` around it. Over the same prompt, three runs:

| | total | sites | per site |
| --- | ---: | ---: | ---: |
| the zero fill | **0.044 to 0.048 s** | 90 | ~0.5 ms |
| the precondition that replaces it | **0.001 s** | 90 | ~11 µs |

90 is 30 layers × 3 chunks. The fill moved 22 MiB a layer, 660 MiB a chunk, **1.90 GiB** over
the prompt. Net saving **~45 ms of a 25.7 s prefill, 0.18 %**, which is a quarter of the
control's own drift. The paired table above cannot resolve it, and now does not have to.

**The saving was not observable, and the removal does not rest on it.** It rests on the fill
being dead: 1.9 GiB written and overwritten before anything read it, plus 90 round trips to do
it. The number's job here was to stop the change being sold as a prefill optimization.

### Reasoning in bytes, again

650 MiB a chunk sounds like it must be worth something. It is a blit fill on unified memory,
and the round trip costs more than the bytes. M-027 recorded the same error with the opposite
sign, 66 % of the bytes read turning out to be 13 % of the time; this is the small version of
it, and the correction is the same one: **a byte count is a hypothesis, not a measurement.**

### What the same-looking zero fill elsewhere is doing

`PrefillRunner.encodeMixtureStart`, the GPT-OSS path, clears `mixture` and **must**:
`scatter_expert` there does `mixture[...] += ...`, accumulating every expert into one row per
token. There are no per-slot writes to make the clear redundant. The two paths differ in the
one property this entry turns on, so the argument does not carry across, and nobody should
carry it.

**Written against a base where Qwen had no prefill runner**, which it now has. `QwenPrefillRunner`
reached the same conclusion independently, from a falsification pass whose one survivor was that
same fill, and dropped it for the same reason. The two arrived at it from opposite directions,
one from a byte count and one from a deliberate error nothing caught, and both had to be checked
against the same property: the groups partition the slots. Qwen's precondition now matches this
one, which is the stronger of the two, because counting the members can be satisfied by a slot
claimed twice beside a slot claimed never.

Full suite, 346 tests, passes, including "Staged prefill matches token-by-token decoding
exactly", which is the test a slot left stale would fail: it compares the staged path against
sequential decoding bit for bit, and a stale slot is a finite, plausible number in the sum.

---

## M-051, A quantization kernel that was wrong at every group size but the one we ship
**2026-08-13, from putting the Qwen model together**

`mlx_affine_gemv` hoisted the per-group scale and bias out of its inner loop, once per `uint4`
chunk, with a comment giving the reason: *a chunk never straddles a group, 32 values at 4 bits
against a group of 64*. True for the published checkpoints. False for a group of **16**, where
one chunk spans two groups and the first group's scale is applied to all 32 values.

The error is about **ten percent** on a projection. Finite, plausible, and it survives every
test that asks whether a model behaves like a model.

### Why nothing caught it

Both tiny fixtures, Gemma's and Qwen's, use a group of 16, because a group of 64 does not
divide a 16-wide `down_proj`. So every small-scale GPU test ran through the broken path, and
none of them compared against a CPU decoding:

- `MLXAffineKernelTests` compares the **batched kernel against the per-token one**. Both hoist
  per chunk, so both were wrong together and agreed to the bit.
- `GemmaMLXModelTests` compares **staged prefill against sequential decoding**. Same two
  kernels, same agreement.
- The end-to-end tests assert the distribution is finite and spread. It was.

Every one of those is a self-consistency check. The suite had no answer computed by anything
other than itself, at any shape where the bug was live.

### What found it

The Qwen oracle: the whole model on the CPU in double precision, decoded from the very bytes
the installation contains. It disagreed at layer 0 by 6.0 in a logit, which narrowed to one
projection, which narrowed to one row.

### The fix, and its shape

The wide-group path is kept **instruction for instruction**, so the published models neither
slow down nor change a bit; the narrow path looks the group up per word. The batched kernel
gets the same treatment, structured in the same order, because `GemmaMLXModelTests` asserts
staged and sequential are identical and a correct path that summed in a different order would
fail it for no fault. `ForwardEncoder` now refuses a group that is not a power-of-two multiple
of the values in a word, which rejects nothing MLX produces.

### What to take from it

An agreement between two things you wrote is worth less than it feels, and the feeling scales
with how much machinery is involved. Both kernels came from the same reasoning, so they
inherited the same false premise, and comparing them measured the reasoning against itself.

The Gemma 4-bit build shipped through three releases with this in it. It was never wrong in
production, because production uses group 64. That is luck, not design, and the tests could not
have told the difference.

---

## M-050, Two ways a falsification can lie to you
**2026-08-12, from building the Qwen mixture**

Deliberate errors are injected and the failing tests counted. Zero failures means the suite is
blind. That inference is wrong twice over, and both showed up in one afternoon.

**A build failure counts as zero failures.** The harness was `swift test | grep -c "✘ Test "`.
A test that does not compile produces no failing tests, so an injected error that happens to
break the build reads exactly like an error the suite could not see. Three results were reported
that way before the pattern was noticed, all of them a Swift string concatenation that
`#expect` will not take as a comment. The harness now checks for `error:` first and says
**BUILD FAILED (not a result)**.

**A test can be blind to the thing it was written for.** The zeroing of the expert slots was
checked by running the mixture twice with the same subset of ranks and comparing. Both runs
share the same stale slots, so they agree whether or not the zeroing happens. The comparison has
to be against a run with *more* ranks: without the zeroing those two are equal, because the
unused slots still hold the earlier contributions.

Both failures have the same shape as the thing this project keeps meeting in the models
themselves: a result that is finite, plausible, and not measuring what it claims to.

---

## M-049, Property tests do not catch bookkeeping, in either block
**2026-08-12, from falsifying the Qwen reference layers**

Both Qwen blocks were first tested with properties: statements about what the block *means*,
written without a fixture. Each block then had deliberate errors injected. The pattern repeated
exactly.

| block | injected | caught by properties |
| --- | ---: | ---: |
| linear | 4 | 2 |
| attention | 4 | 1 |

The five that escaped were all bookkeeping or a numeric constant:

- every head gated by the first slice of `z`
- value heads mapped to key heads by `%` instead of `/`
- Gemma's attention scale of 1.0 in place of `1/sqrt(headDim)`
- the head norms moved after the rotary instead of before

**None of them changes what the block means.** A head reading another head's gate still
produces a gated output; a differently grouped head still attends. And the scale is invisible
under a single token, because a softmax over one key is 1 however it is scaled, which is exactly
the shape a minimal property test takes.

What caught all five, in both blocks, was **composing the block a second time inside the test
and comparing**. It is duplication, and it is the same discipline as the Python transcription
that guards the operators: written twice, compared, on the theory that the two transcriptions
will not fail identically.

The conclusion for the blocks still to be written, the mixture and the model runner: **the
independent composition is the test, and the properties are the supplement.** Writing the
properties first was not wasted, they are cheap and they read as documentation, but they are not
evidence that a block is wired correctly.

---

## M-048, Time to first token is not a function of context length
**2026-08-12, M4, 24 GiB, reported from the app in use**

Four turns, in order:

| context | time to first token | |
| ---: | ---: | --- |
| 800 | 31 s | fresh paste, all 800 new |
| 1500 | 19 s | partly continuing |
| 2400 | 30 s | continuing, large new part |
| ~2500 | **6.5 s** | continuing, only the question new |

A 2400-token turn beats a fresh 800-token one. **The wait is proportional to the tokens that
are not already in the cache**, not to the conversation's length, because the rest is reused.
The scatter on top of that is expert paging: a 3.3 MiB blob read cold against one the OS still
has in page cache, ~128 of them a layer.

This makes the headline figures in M-047 the worst case rather than the typical one, and it is
worth stating that way, a benchmark that always pastes a fresh prompt measures the case a
conversation only pays once.

The app now shows the new-token count beside the time, because the behaviour is legible only
when that number is visible: without it the wait looks arbitrary, and the reasonable
conclusion from the outside is that the measurements are noise.

**A flaky failure, investigated.** "The quantized model produces a usable distribution"
failed once with flat logits, the same symptom, and the same test, that caught the
out-of-bounds attention accumulator in M-044.

**Second occurrence, 2026-08-13**, on a full-suite run during the Qwen work, and again not
reproduced: 13 further runs, 5 of the suite alone and 8 full-suite. So it is not a once-off, and
the twice-in-two-days rate is not consistent with the 41 clean runs either side of it being the
whole story.

The diagnosis the test prints, which says whether the logits are all zero or merely flat, **was
not captured**, because the run was filtered with a `head` that cut the message. That is the one
thing this occurrence could have contributed and it was thrown away by how the output was read.
The next occurrence should be captured whole before anything else is done with it.

Not reproduced: **28 runs** (10 full-suite, 12 of that suite alone, 6 more with the GPU under
load from a bench loop, which is the condition it appeared under). So the cause is not
established. What the investigation did establish is worth more than the reproduction would
have been.

The failure printed a spread of **exactly 0.0**, and the finite and softcap assertions beside
it passed. Identical *and* finite points at zero, and a Metal buffer starts zeroed, which
raised the question of whether some work had simply not run. It could have:

**No command buffer's completion was ever checked.** `.error` and `.status` appeared nowhere in
`HydraMetal`, across 22 `waitUntilCompleted()` sites. A command buffer that fails, timeout,
resource limit, memory pressure, leaves the buffers it was going to write untouched, and the
forward pass then runs to completion on zeros and produces a finite, plausible, wrong answer.
Every wait in the inference paths is checked now, and a GPU failure raises instead of
returning zeros.

Two more silent paths came out of the same sweep. `readEmbedding` on the MLX build returned
early when the embedding tensors were missing, leaving the destination holding the previous
token's row; it stops loudly now. And `ModelRunner`'s asynchronous expert prefetch discards its
error with `try?`, left alone, because the synchronous `expert()` that follows retries the
read and does throw.

The assertion itself could not distinguish an all-zero head output (work that never ran) from a
flat non-zero one (work that ran on a degenerate input). It reports which now. That is the
point of this entry: the next occurrence will say something.

---

## M-047, Prefill, end to end: 41 s to 25 s, and where the rest of it is
**2026-08-11, M4, 24 GiB, 800 prompt tokens, the app's settings (8 slots, 8k)**

| phase | before | after |
| --- | ---: | ---: |
| cb1 (attention, dense, router) | 21.8 s | 13.3 s |
| experts | 12.4 s | 6.8 s |
| expert I/O | 6.5 s | 3.5 s |
| **total** | **40.8 s** | **~25 s** |
| prompt tokens/s | 14 | 34 |

Measured through the app's own engine rather than the CLI, which is the number a user sees:
**25.4 s to first token, and 4.2 s on a follow-up turn**, cache reuse works, so the cost is
paid by a fresh long paste and not by continuing a conversation.

What remains is cb1, and it is quadratic in context because attention is. The expert side is
near its floor: sweeping the chunk bottoms the I/O out around 3 s.

**A reporting lesson.** The user reported 27 s and 40 s; I read a UI label showing total time
and generation rate, back-computed that generation must account for most of it, and told them
prefill was fine. It was not, they had watched it. Separately I quoted 16 s at them, measured
at 16 slots on a warm page cache, which is not what the app runs. **Reproduce the reported
configuration before answering a report with a number.**

---

## M-046, The prefill chunk has an interior optimum, and the app was below it
**2026-08-11, M4, 24 GiB, 800 prompt tokens**

| chunk | 128 | **256** | 384 | 512 |
| --- | ---: | ---: | ---: | ---: |
| total | 27.7 s | **25.4 s** | 25.7 s | 27.8 s |
| expert I/O | 4.5 | 3.4 | 3.0 | 3.9 |
| cb1 | 15.3 | 15.2 | 16.2 | 17.5 |

Two costs pull against each other. A larger chunk shares each expert's read across more
tokens, the I/O column keeps improving. But the batched projections re-read the *activations*
far more than the weights (M-045), and past 256 the transposed activation buffer stops fitting
in cache: 16.8 MiB at 512 against 4.2 at 128. That is cb1 climbing.

The app also capped prefill at 128 tokens a call for its stop button, **below** the runner's
own chunk, so the batching never saw a full one.

Raising the expert cache from 8 slots to `balanced`'s 16 was tried and **reverted**: 758 MiB
for 3 % of decoding. That policy was measured before prefill was batched and the gap has since
closed; the trade is now the wrong side for an app whose point is a small footprint.

---

## M-045, The batched projection is bound by activations, not weights
**2026-08-11, M4, 24 GiB, a layer's seven projections over 128 tokens**

The first batched GEMM was 1.2× against the per-token loop it replaced, after an hour of
sweeping the token tile. Counting the traffic explains why, for `q_proj` over a chunk:

    weights      (tokens / TB) × 6.2 MiB       =  50 MiB at TB = 16
    activations  rows × cols × tokens × 4 / RB = 5.9 GB at RB = 1

**The activations are read 120× more than the weights**, and the tile only divides the weight
traffic. Blocking the rows so one activation tile serves `RB` of them, plus hoisting `Σx`,
row-independent, and being recomputed in all 4096 rows, took it to 2.5×.

The register budget is the constraint and it is the *area* that matters. Every pair with
`RB × TB = 32` lands in the same place:

    RB × TB    2×4   2×8  2×16   4×4   4×8  4×16   8×4   8×8  8×16
    ms          82    42    28    47    26    49    28    59    78

**Two benches were wrong before this one was right.** Benching one matrix 128 times measures
cache, not prefill: 6 MiB stays resident, so the per-token loop never paid the re-read the
batched kernel exists to remove. The bench must run a whole layer, 36.6 MiB of distinct
weights, past this machine's cache, which is what production does.

And the projections were not even the main cost: measured by phase, they were ~8 s of 41. The
real cost was ~334,000 tiny per-token dispatches, and moving the token loop from Swift onto
the grid is what took cb1 from 21.8 s to 9.2.

---

## M-044, Attention was the whole remaining budget, and hid a correctness bug
**2026-08-11, M4, 24 GiB, `hydra bench-gemv`, paired at two context lengths**

The projections account for ~20 ms of a token's 60 on the GPU. The rest was in kernels nobody
had timed. Timed:

| kernel | shape | µs |
| --- | --- | ---: |
| **attention_decode** | 16h × 1024 keys | **4030** |
| rms_norm_heads | 16 × 256 | 2 |
| rms_norm | 2816 | 4 |
| fused_norm_add_copy | 2816 | 4 |
| apply_rope | 16 × 256 | 1 |
| gelu_mul | 2112 | 1 |
| add_in_place / copy | 2816 | 1 |

Three orders of magnitude between attention and everything it shares a layer with. **The
elementwise kernels M-038 and M-039 spent two commits fusing are 1 µs each**, those results
were real and they were rounding error, and this table would have said so in one run for the
cost of an afternoon's instrumentation.

Split-K flash-decoding, each simdgroup takes a stride of the window, partials merged by
rescaling to the common maximum, takes it to 520 µs.

| | short (~60 keys) | long (~1200 keys) |
| --- | ---: | ---: |
| GPU busy, control | 60.1 ms | 192 ms |
| GPU busy, split | 48.2 ms | 65.3 ms |
| tok/s, control | 9.15 | 4.43 |
| tok/s, split | 9.36 | **8.45** |

Throughput used to halve as context grew. It now barely moves. Everything this session
reported before this entry is a short-context number.

**The correctness finding matters more than the speed.** The per-lane accumulator was sized
`[8]`, correct for a 256-wide head, out of bounds for a 512-wide one. Gemma's full-attention
layers are 512 wide, one in six. It produced visibly corrupted tokens in real generations and
had been there all along.

It survived because the bound was asserted in the *test harness*, which rejected
`headDim > 256`, while the production encoder had no check and passed 512 through. The one
geometry that needed testing was the one the tests refused to run. Compounding it, the harness
dispatched 32 threads where production dispatches 256, so every attention test ever run
validated a configuration the model does not use.

Fixed: the bound is a shared constant, asserted where production dispatches; the harness
dispatches production's shape; and the suite now covers both head widths over a 600-key window
and windows shorter than the split. Reverting the accumulator size fails the 512 case at
deviation 1.8 against a 1e-4 tolerance and leaves 256 passing, the check was confirmed to
fail before it was trusted to pass.

**Method note.** Two diagnoses were wrong before this one. First that the NaN came from
seeding idle simdgroups with -INFINITY: plausible, and the suite passes with it reintroduced,
so it was not the cause. Second, and worse, `git checkout` on the uncommitted kernel to undo a
temporary edit destroyed the draft that had actually failed, so it could no longer be diffed.
What resolved it was reading the generated text, which was corrupted in a way no aggregate
metric showed.

---

## M-043, Overlapping the expert read with the resident experts: −4 %, reverted
**2026-08-11, M4, 24 GiB, seven interleaved pairs**

The `pread` for a layer's missing experts is the largest block of CPU time left in a token
(~20 ms against ~60 ms of GPU), and the GPU is idle throughout it. At a 74 % hit rate three
experts in four are already resident, so they were committed as their own buffer to give the
GPU something to do while the fourth was read, the shape `ModelRunner` already uses for
GPT-OSS, minus its wait on the warm buffer, which queue ordering makes unnecessary.

| | control | treatment |
| --- | ---: | ---: |
| tok/s, seven pairs | 8.60 9.41 9.19 7.88 9.37 8.80 9.20 | 7.95 8.91 9.54 7.29 8.37 8.41 8.31 |
| mean | **9.06** | 8.40 |

Control ahead in six pairs of seven. Reverted.

Output was byte-identical to the control across three repeats, so this is a performance
result and not a correctness one, the pinning discipline `ModelRunner` documents (encode
before launching the reads, or the reads evict the slots about to be used) does hold.

The likely cause, untested: splitting the layer into two command buffers reinstates a
boundary the GPU has to drain at. The buffer holding the resident experts must complete
before the one holding the sum and the next layer's attention begins, where a single buffer
lets the whole layer pipeline. That is the same lesson as M-041 one level up, the ordering
was already free inside a buffer, and paying a buffer boundary to express it costs more than
the overlap returns.

**Three attempts at CPU/GPU overlap have now failed** (M-036 twice, this once). The pattern
is consistent: on this machine, work removed from the critical path is worth less than the
structure needed to remove it. Not a proof it cannot pay, but the next attempt should be
expected to lose, and needs a mechanism argument before it is worth the measurement.

A reporting note: the "cache hits %" line counts `expert()` calls, so it moves when the
number of calls changes, not only when residency does. It read 73 % against 67 % here for
structural reasons. It is not comparable across changes that alter the call pattern; bytes
read is.

---

## M-042, Halving the round trips: +8 %, and GPU busy scores it zero
**2026-08-11, M4, 24 GiB, five interleaved pairs**

The decode loop committed and waited twice a layer, 60 round trips a token. Only the first
wait is real: the CPU must read the router's choice before it can `pread` the experts. Layer
N's experts and layer N+1's attention need no CPU between them, so they now share a buffer.

| | control | treatment |
| --- | ---: | ---: |
| GPU busy | 60.4 ms | 60.0 ms |
| tok/s | 8.55 | **9.27** |

GPU busy is flat because no GPU work was removed. The whole gain is round-trip latency, and
that is the warning: **GPU busy, the low-variance instrument M-038 established, is blind to
this entire class of change.** It would have scored the best result of the session as zero.
Use it for kernel work; use paired tok/s for anything that touches the CPU/GPU boundary.

Generation is byte-identical to the control across the sample. The test suite does not cover
this: no test runs thirty real layers on real weights, and a buffer merged one layer early
still produces plausible text.

---

## M-041, One compute encoder per buffer instead of 660
**2026-08-11, M4, 24 GiB, five interleaved pairs**

Every dispatch created and ended its own encoder. A compute encoder defaults to serial
dispatch, which already orders the commands inside it, so the boundaries synchronized nothing.

| | control | treatment |
| --- | ---: | ---: |
| GPU busy | 64.0 ms | 60.4 ms |
| tok/s | 8.36 | 8.79 |

660 boundaries for 3.6 ms, about 5 µs each, matching the per-dispatch cost measured in M-040.
Committing with an encoder open is a Metal abort rather than a wrong answer, which is how the
four test helpers that commit their own buffers were found.

---

## M-040, The GEMV is bound by values unpacked, not bytes moved
**2026-08-11, M4, 24 GiB, `hydra bench-gemv`**

Two changes aimed at bandwidth failed first, one of them 40 % *worse*, both reasoning from
cb1's aggregate 20 GB/s against a 95 GB/s ceiling. cb1 is not one kernel: it is the
projections plus the norms, the rotary, the attention and the router, and the last four move
almost no bytes while being charged into the same average.

Measured alone, at the real shapes:

| shape | bits | MiB | µs | GB/s |
| --- | ---: | ---: | ---: | ---: |
| q_proj 4096 × 2816 | 4 | 6.2 | 144 | 45 |
| mlp.gate 2112 × 2816 | 8 | 6.0 | 81 | 78 |

Same kernel, same bytes, twice the time. Per *value* they are identical, 12.5 ps at 4 bits,
14 at 8. **The kernel is bound by values unpacked.** 4-bit runs at half the bandwidth of 8-bit
for exactly the reason it is worth using.

This retires M-031's "4-bit unpack is free", which compared these kernels to a ceiling they
cannot reach.

The fix follows from it: `bits` arrived at runtime in the dims buffer, making `perWord` a
dynamic loop bound and `slot * bits` a computed shift, so the innermost loop in the model
could not be unrolled. Instantiated as a template per width:

| shape | before | after |
| --- | ---: | ---: |
| q_proj | 144 µs | 62 µs |
| o_proj | 142 µs | 67 µs |
| mlp.gate | 81 µs | 47 µs |
| head 262144 × 2816 | 8272 µs | 4315 µs |

End to end, four interleaved pairs: GPU busy 82.1 → 62.3 ms (−24 %), tok/s 7.30 → 8.26.

The lesson is the bench, not the fix. Two failures came from optimizing against an aggregate;
the isolated bench took twenty minutes and answered it in one run.

---

## M-039, Per-head norms, and an estimate wrong by 4.5×
**2026-08-11, M4, 24 GiB, five interleaved pairs**

Q, K and V are normalized per head, the norm is not separable (D-022), and that was a Swift
loop issuing one dispatch a head: 32 a layer, 960 a token, each over 256 floats.

Predicted 27 ms saved, reasoning from M-038's 28 µs a dispatch. Actual: 6 ms. So that 28 µs
does not generalize, consecutive small dispatches inside one encoder cost about 6 µs, and
what made the M-038 ops expensive was not their place in the chain.

| | control | treatment |
| --- | ---: | ---: |
| GPU busy (median) | 88.3 ms | 82.3 ms |
| tok/s (median) | 7.05 | 7.43 |

**Method note.** The first two attempts ran A-then-B, and the machine drifted from 87 to 109 ms
mid-run as it heated, read as a result, it flattered the change. Interleaving the two builds
and alternating is now the only design used here. M-038's claim that GPU-busy variance is
0.5 % holds only within a thermally steady run; across one it is 20 %.

---

## M-038, Fusing the chain works, and GPU-busy time makes it measurable
**2026-08-11, M4, 24 GiB, Low Power Mode off, paired**

Three dispatches a layer removed, post-attention norm, residual add and the shared copy fused
into one kernel; the router's unscaled norm and per-channel scale into another.

| | GPU busy, four runs | median |
| --- | --- | ---: |
| unfused | 85.8 86.7 87.1 126.7 | 86.9 ms |
| **fused** | 84.2 84.2 84.3 84.6 | **84.3 ms** |

**2.6 ms, about 3 %**, for 90 fewer dispatches a token, roughly 28 µs of chain latency each.

### The measurement is the real result

Look at the spreads. GPU busy varies **0.5 %** across runs; end-to-end `tok/s` varies **20 %**
on this same machine (M-037). Every "neutral" verdict in M-031 to M-036 was taken in tok/s,
where a 3 % effect is invisible, **several of them were unresolvable rather than genuinely
zero**, and at least the batched expert dispatch deserves re-measuring this way.

### What it implies about the ceiling

The chain is about 22 dispatches a layer. Fusing everything fusable might reach 10, saving
~12 × 30 × 28 µs ≈ 10 ms, so **GPU work bottoms out near 75 ms**, against 84 now.

With the CPU's ~60 ms of `pread` and encoding still serialized behind it, that is ~135 ms a
token, or **7.4 tok/s**. The target range needs 88 to 119 ms.

**So fusion alone cannot reach it.** The arithmetic has said the same thing three times now:
the CPU and GPU halves have to overlap, and the two attempts at that cost 30 % and 45 %
(M-032, M-036). That remains the one open problem, and it is now the *only* one, every other
candidate has a measured ceiling that rules it out.

---

## M-037, The bandwidth ceiling, and what the seven failures actually meant
**2026-08-11, M4, 24 GiB, from a Metal System Trace and a microbenchmark**

### Low Power Mode was capping the GPU clock. It was not the bottleneck.

The trace settles it. Aggregating `gpu-performance-state-intervals` by duration:

| | Minimum | Medium | Maximum |
| --- | ---: | ---: | ---: |
| Low Power Mode **on** | 79.9 % | 16.2 % | 3.9 % |
| Low Power Mode **off** | 6.5 % | 14.1 % | **79.4 %** |

The clock state inverts completely, and throughput moves **7.2 to 7.8 → 7.98 tok/s**. About 5 %.

More telling: **GPU busy time barely changed**, 81 ms against 83 to 96 ms, while the clock went
from minimum to maximum. Work that does not speed up when the clock triples is not
compute-bound.

**Every measurement in M-031 through M-036 was taken with the clock capped.** The cap turned
out not to matter, so those results are unlikely to change, but they were taken under a
confound nobody had noticed and should be re-run before anyone leans on them.

### The ceiling, measured

A microbenchmark shaped exactly like our GEMV, one simdgroup a row, eight rows a threadgroup,
`uint4` loads, over 512 MiB:

| row length | stream only | with 4-bit unpack and `fma` |
| --- | ---: | ---: |
| 65,536 B | 92.9 GB/s | 95.2 GB/s |
| 8,192 B | 95.4 GB/s | 97.0 GB/s |
| **1,408 B** (a real Gemma row) | 97.0 GB/s | **94.2 GB/s** |
| 352 B (`down_proj`) | 98.9 GB/s | 37.0 GB/s |

Two things fall out. **The machine gives ~95 GB/s for our exact access pattern**, and **the
unpacking is free**, 4-bit decode plus the affine arithmetic costs nothing against a plain
streaming read. Neither the kernel shape nor the quantization maths was ever the problem.

### What we actually achieve, per bucket

| bucket | bytes/token | time | achieved |
| --- | ---: | ---: | ---: |
| head, **one** dispatch of 32,768 threadgroups | 415 MB | 8.7 ms | **48 GB/s** |
| `cb1`, ~660 small dependent dispatches | 1.16 GB | 59 ms | 20 GB/s |
| experts, ~240 small dispatches | 803 MB | 42 ms | 19 GB/s |

The one bucket that is a single large dispatch runs 2.4× faster per byte than the two made of
many small dependent ones. **That is the shape of the remaining problem**, and it is the first
statement in this whole effort with a measured ceiling behind it rather than an analogy.

It also explains the seven results at last: none of them changed bytes moved or the length of
the dependency chain. Reducing dispatch *count* while keeping the chain serial does not help,
which is why batching the experts was neutral.

### And the other half is the CPU

GPU busy is 81 ms of a 157 ms token. The remaining 76 ms is `pread` (47 ms) and encoding
(~29 ms). Even a perfect GPU, 50 ms at the head's rate, would leave 126 ms and 7.9 tok/s.
**Both halves have to move**, which is why every single-sided attempt has been marginal.

### A methodology finding, which matters more than any of the above

Re-running the cache-size sweep with the clock unlocked gives 16 slots at both **6.57 and
7.99 tok/s**. Run-to-run variance is around 20 %, larger than most of the effects being
chased. End-to-end `tok/s` cannot resolve a 10 % change without many repetitions.

`lastGPUSeconds` is far quieter, because it excludes CPU scheduling noise. **GPU work should be
optimized against GPU busy time, and only the final result reported in tok/s.**

---

## M-036, Overlapping CPU with GPU costs 45 %, on a cold machine, twice tried
**2026-08-11, M4, 24 GiB, rebooted and idle overnight, page cache warmed**

The premise from M-035 is sound and unchanged: at 16 slots the GPU is busy **82 to 96 ms of a
140 to 156 ms token, 57 to 62 %**. Fully hiding the CPU's ~60 ms would give ~10.7 tok/s, which is
inside the competitive range. Everything about the arithmetic says overlap is the whole
remaining game.

Two ways of reaching it have now been measured, and both are worse:

| attempt | result |
| --- | ---: |
| shared branch in its own buffer, run during the `pread` (M-032) | −30 % |
| **encode layer L+1's `cb1` while layer L's mixture executes** | **−45 %** |

The second is the one M-033 recommended as the safe form, and its correctness is not in
question, 254 tests pass, the oracle included. It is simply slower: **4.3 tok/s against 7.6**,
paired on a cold machine.

Both share a shape. Doing CPU work *while a command buffer is in flight on the same queue* is
apparently not free on this driver, and costs more than the idle time it fills. Building a
command buffer while another executes is the specific operation both attempts have in common,
and it is the thing to measure directly before a third attempt.

### Two costs measured directly, so they stop being suspects

- **Command buffer round trips**: 60 empty commit-and-wait pairs cost **1.25 ms**, 0.021 ms
  each. Submission latency is not a factor.
- **`concurrentPerform` when everything is resident**: skipping it entirely for the ~78 % of
  layers that need no read changed nothing. GCD dispatch is not a factor either.

### The tally

Seven structural changes attempted against decode; one helped.

| change | result |
| --- | ---: |
| expert cache 8 → 16 slots | **+14 %, shipped** |
| shared-branch overlap | −30 % |
| dropping the `cb2` wait | wrong logits |
| batched expert dispatch | none |
| simdgroup-per-row GEMV | none |
| resident fast path in `load` | none |
| encode-ahead | −45 % |

Every one after the first was reasoned from a model of the bottleneck, and the models were
wrong in a way that reading the code could not reveal. **What is needed next is not another
hypothesis but a GPU trace**, Metal System Trace or a frame capture, which shows kernel
occupancy and where the gaps actually are, and which cannot be run from this environment.
Anyone continuing should start there rather than from this list.

---

## M-035, The GPU is idle 46 % of decode. That is the number, and it took five failures to find
**2026-08-10, M4, 24 GiB, Gemma 4 MLX 4-bit, 16 slots**

```
GPU busy 147.1 ms of 270.2 ms wall (54 %), the rest is CPU
cb1 104.7 ms · expert I/O 74.0 ms · experts 71.9 ms · head 19.6 ms
```

Measured with `MTLCommandBuffer.gpuStartTime/gpuEndTime` summed across a token's buffers, which
separates GPU execution from wall time. **This should have been the first measurement taken.**

### What it retires

Five structural changes were tried before it, and the reason each was expected to help was a
guess about the bottleneck. In order:

| change | expected | measured |
| --- | --- | --- |
| shared-branch overlap (M-032) | hide I/O | **−30 %** |
| stop waiting on `cb2` (M-033) | halve syncs | **wrong logits** |
| bigger expert cache (M-034) | fewer reads | **+14 %, shipped** |
| batch expert dispatches (M-034) | 8× fewer launches | **nothing** |
| simdgroup-per-row GEMV | amortize the reduction | **nothing** |

And two costs measured directly rather than assumed:

- **Command buffer round trips are free.** 60 empty commit-and-wait pairs cost **1.25 ms**,
  0.021 ms each. Submission latency was never the problem, which retires the theory behind two
  of the failures above.
- **Dispatch count is not the problem either**, 1,200 launches became 150 with no effect.

### What it points at

Forty-six per cent of decode is CPU, and the largest identified block is the 74 ms of `pread`
during which the GPU has nothing to do. **The shared-branch overlap is therefore the right
idea**, it is what TurboFieldfare uses that window for, and M-032's 30 % loss is evidence
against *that implementation*, not against the approach. Splitting `cb1` in two was the wrong
way to reach it.

The other CPU work worth accounting for before the next attempt: encoding ~22 dispatches a
layer, and the embedding row, which for the MLX build is 2,816 four-bit values unpacked and
dequantized **on the CPU** every token.

**The rule this establishes:** measure whether the GPU is busy before changing anything that
assumes it is. Four of the five changes above were aimed at making GPU work cheaper, and the
GPU was not the constraint.

---

## M-034, The cache default was worth 14 %; batching the experts was worth nothing
**2026-08-10, M4, 24 GiB, Gemma 4 MLX 4-bit, interleaved**

### The cache size, interleaved across three rounds

| slots | runs | median |
| ---: | --- | ---: |
| 8 (the old default) | 6.62 6.94 6.90 | 6.90 |
| **16** | 7.23 7.90 8.00 | **7.90** |
| 32 | 7.32 6.97 7.85 | 7.32 |

**Shipped**, as `ExpertCachePolicy.balanced`, twice the minimum, and it is now what the app
and the CLI use. About 800 MB for 14 %, and the curve turns over after that. TurboFieldfare
settled on sixteen independently.

`minimal` is what demonstrates the thesis and stays available; it is no longer what people run
by default, because running the configuration that proves a point is not the same as running
the one that works best.

### Batching the expert dispatches: neutral

Done correctly this time, eight **separate** blob bindings, so none of M-031's residency
collapse, the mixture went from 40 dispatches a layer to 5, or 1,200 a token to 150.

| | @8 slots, median |
| --- | ---: |
| per-expert | 7.23 |
| batched | 7.25 |

Nothing. **Reverted**, because a second implementation of the mixture that has to agree with
the first forever is not worth zero (D-023).

### What that costs the model of the bottleneck

M-031 concluded decode was bound by dispatch count. **Cutting the mixture's dispatches by 8×
changed nothing, so that conclusion was wrong**, or at least dispatch count is not the
dominant term.

What the three experiments now rule out, at 16 slots:

- **not I/O**, the cache curve turns over at 16;
- **not the inner loop**, vectorizing and hoisting the bias changed nothing (M-031);
- **not dispatch count**, 8× fewer changed nothing.

What is left is the **shape of the GEMV itself**: one threadgroup per output row, each doing a
`simd_sum` and a threadgroup barrier to produce a single scalar. At 704 rows × 352 words a lane
handles about eleven values, so the reduction plausibly costs as much as the arithmetic, and
the total threadgroup count is identical whether the work arrives in one dispatch or eight,
which is exactly why batching changed nothing.

The next experiment is a GEMV where one threadgroup covers **several rows** and each thread
accumulates across them, so the per-row reduction is amortized. That is a kernel rewrite with a
clear hypothesis, which is more than the last three had.

---

## M-033, Dropping the wait on `cb2` is not a scheduling change, it is a correctness change
**2026-08-10, reverted before measuring**

D-025 listed "stop waiting on `cb2`" as the cheapest remaining win: only the router readback
needs a CPU synchronization, so the other thirty waits a token look like pure serialization
between CPU encode and GPU execution.

Committing `cb2` without awaiting it, and releasing the expert slots from a completion handler,
**produces wrong logits**. The batched-prefill-versus-sequential tests caught it at once, and
the failure is not float noise: the sequential path's first logit moved from −0.4784 to −0.4594
and the divergence grows from there.

The reason is that the scratch buffers are reused by every layer. `cb2` of layer *L* writes
`scratch.hidden`; `cb1` of layer *L+1* reads it and overwrites `normed`, `query` and the rest.
Metal tracks hazards **within** a command buffer, and the assumption that committing to one
queue also serializes execution **across** buffers is what this disproves. Nothing in the
runtime expressed the dependency, so the wait was carrying it silently.

**So the wait is not overhead, it is the synchronization.** Removing it needs the dependency
stated some other way, and there are two honest options:

- an `MTLEvent` signalled at the end of `cb2` and waited on at the start of the next `cb1`,
  which keeps the ordering on the GPU and off the CPU; or
- **encode-ahead**: build the next layer's `cb1` *before* awaiting `cb2`, then wait, then
  commit. Encoding only records commands and reads no results, so the CPU work overlaps the
  GPU's while the ordering stays exactly as it is today. This is the smaller change and needs
  no new synchronization primitive.

Neither was attempted here. Reverted at the point the tests failed, because a throughput
measurement of an incorrect decode is worth nothing.

**Two of D-025's four items have now been tried and neither shipped**, the shared-branch
overlap cost 30 % (M-032), and this one is a correctness change wearing a scheduling change's
clothes. Both failures were found by tests or paired measurement rather than by reading, which
is the argument for keeping both.

---

## M-032, The shared-branch overlap, paired against its own control: −30 %
**2026-08-10, M4, 24 GiB, Gemma 4 MLX 4-bit, 8 slots**

| order | dense inside `cb1` | dense overlapping the reads |
| --- | ---: | ---: |
| control first | 6.59 6.67 7.01 7.95 | 5.07 5.25 5.30 5.69 |
| candidate first | 7.30 7.73 7.99 | 5.23 5.71 6.02 |

Run in both orders because the first result was the opposite. Measured against a baseline taken
an hour earlier, 4.47 to 4.72, the change looked like a **+16 % improvement**, with the two
distributions cleanly separated and no overlap. It was the machine cooling down.

**That is the whole lesson.** M-031 warned that throughput drifts 3.5 to 8 tok/s with thermal
state, and the warning was not enough: two non-overlapping distributions taken an hour apart
still read as a clean win. Only the paired control, built and run back to back, showed the
change was a 30 % loss. Every comparison from here needs its control rebuilt and rerun beside
it, which is the discipline `Bench.swift` already enforces for kernels and which nothing was
enforcing for end-to-end throughput.

**Why it loses.** Decode is bound by submission latency (M-031). The split adds a command
buffer per layer, thirty more submissions a token, to cover reads that the unified memory bus
is already contending for: `expertIO` rose from ~28 to 40 ms to ~60 to 86 ms once the GPU had work
running during them. On Apple Silicon, overlapping I/O with compute is not free; both go
through the same bus.

Reverted. D-025 reorders the remaining work accordingly: submissions first, overlap last.

---

## M-031, The decode bottleneck is dispatch latency, and one attempt at it failed
**2026-08-09, M4, 24 GiB, Gemma 4 MLX 4-bit**

A decode step, at 128 slots: **cb1 65 ms · expert I/O 28 ms · experts 102 ms · head 9 ms.**
That 102 ms moves 802 MB, about 8 GB/s against the M4's ~120, because the mixture is 8
experts × 3 GEMVs × 30 layers = **1,200 kernel launches a token**, each threadgroup reading a
few hundred bytes. Bound by launch latency, not bandwidth.

**Two hypotheses eliminated first, both wrong.**

*Not I/O.* Raising the cache from 8 to 128 slots takes hits from 59 % to 91 % and leaves
throughput flat. The 12.85 GB pool fits in 24 GB, so the obvious answer was available and buys
nothing.

*Not the inner loop.* The affine kernel now reads `uint4` chunks and hoists the bias out of the
per-weight path, `Σ(q·s + b)·x` factors into `s·Σ(q·x) + b·Σx`, distributivity rather than
approximation, held exactly by the oracle. No measurable change. **Kept**, because it is
strictly less work for identical results.

**The attempt that failed, and why it is worth recording.** Batching the mixture into one
dispatch per projection role needs a kernel that can index across experts, which needs a
layer's slots to share one allocation. Both were built. The mixture bucket fell from ~102 ms to
~50 to 75 ms, roughly the predicted 2×.

At 128 slots the same change collapsed to **0.29 tok/s**, a 17× regression. Binding a buffer
makes *all* of it resident, and a layer's pool is 430 MB at 128 slots against 26.9 MB at 8. The
cliff is in the shared allocation alone; it does not need the batched kernel to appear.

Both are reverted. The idea is sound and the implementation is not: batching needs the eight
selected slots bound as **separate** buffers, or residency managed explicitly through a Metal
heap, so that only what is read becomes resident. That belongs to the performance audit, not to
a patch.

**A caution for that audit.** Throughput varies 3.5 to 6.5 tok/s across identical runs after hours
of sustained GPU load. Every single-run comparison in this session was inside that band and
none of them should be trusted; the numbers above are from repeated runs, and the two that
matter, the 17× cliff and the flat cache curve, are far outside it.

---

## M-030, Gemma 4 at 4 bits: 3.4x decode, 3.2x less read, same answer
**2026-08-09, M4, 24 GiB, identical prompt, same session, minimum slots**

| | BF16 | MLX 4-bit | |
| --- | ---: | ---: | ---: |
| Install | 48 GB | **15 GB** | 3.2× |
| Prefill, 25 tokens | 11.4 s | **5.8 s** | 2.0× |
| SSD read for that prefill | 17.31 GiB | **5.42 GiB** | 3.2× |
| Decode | 1.31 tok/s | **4.41 tok/s** | **3.4×** |
| Process footprint | 3,320 MiB | **1,377 MiB** | 2.4× |

Both answered *"The capital of France is Paris."* Asked for the token after
`<|turn>model`, both rank «The» then «Paris», the 4-bit build with more confidence
(25.1 / 18.9 against 18.7 / 11.9), which is a QAT checkpoint behaving as one should.

The install itself: 14.54 GiB in 587 s, **peak process footprint 46.1 MiB**.

**The ratio predicted from the arithmetic was 3.55×, and decode came in at 3.4×.** That is
close enough to be worth stating and not close enough to have been safe to assume, M-027 is
the standing reminder. Prefill gains only 2.0× because it is not purely I/O bound: with the
experts already grouped per chunk (M-029), the attention and dense MLP are a third of its time
and quantization does not make a `GEMV` faster, only smaller.

**A retracted measurement, recorded because the error is instructive.** The first Q4 run
reported 4.55 tok/s on an 83-token prompt, and it was GPT-OSS 20B. `chat gemma-q4 "…"` matched
no case in the argument loop, so the model name fell through into the *prompt*, the command
loaded the default model, and it answered correctly and quickly from the wrong weights. Nothing
in the output said so, the throughput was plausible for either model. The tokenizer dump
showed «gem»«ma»«-q»«4» at the head of the prompt, which is the only reason it was caught.

The accepted names are one set now, beside the resolver, so adding a model to one without the
other stops being possible. The general lesson is the one M-028 already recorded from a
different direction: **a plausible number from an unverified path is worse than no number**,
because it does not invite checking.

---

## M-029, Batched prefill for Gemma: 2.25×, and where the rest of the time is
**2026-08-09, M4, 24 GiB, 25-token prompt**

| | before | after |
| --- | ---: | ---: |
| Prefill | 24.5 s | **10.9 s** |
| SSD read | 46.7 GiB | **17.3 GiB** |
| attention + router |, | 3.28 s |
| expert I/O |, | 6.04 s |
| expert compute |, | 1.55 s |

**The reordering is about I/O, not arithmetic.** Token-major prefill reads a layer's experts,
moves on, and by the time it returns for the next token the slots hold someone else's. Every
token paid for its own eight reads at every layer. Layer-major with the experts grouped reads
each distinct expert once per chunk, for 25 tokens the union is about 50 of 128 per layer
instead of 200 reads.

Three findings, in the order they were measured:

1. **Grouping alone: 24.5 → 14.5 s.** The I/O fell by 2.7×.
2. **Batching the attention phase into one command buffer per layer: 14.5 → 13.7 s.** Almost
   nothing. 750 command buffers were not the cost; the arithmetic was. Worth recording as a
   negative result, the obvious optimization was the wrong one.
3. **Loading experts in slot-sized batches instead of one at a time: 13.7 → 10.9 s.** The first
   version asked for one expert per call, which quietly turned `ExpertSlotCache`'s concurrent
   `pread`s into a serial read. The largest single win, from deleting a mistake rather than
   adding anything.

**What it cost to get right.** The first version encoded a whole chunk's attention into one
command buffer while writing the rotary tables from the CPU per token, so every token used
whichever table was written last. The output stayed finite and plausible and was wrong by about
one part in fifty. The test that caught it asserts batched prefill is **bit-identical** to
feeding the same tokens one at a time, which is the only assertion strong enough: a tolerance
would have passed.

**Where the remaining time is.** Expert I/O at 6.04 s is 17.3 GiB at 2.9 GB/s, and a chunk
cannot avoid reading the experts it uses, though the union saturates at 128 per layer, so the
per-token cost keeps falling as prompts get longer. The 3.28 s of attention and dense MLP is
`GEMV` per token where a batch could use `GEMM`; that is the next lever and it is a real piece
of work, not a patch.

Decode is untouched at 1.2 to 1.4 tok/s. Nothing here was aimed at it.

---

## M-028, Gemma 4 26B-A4B on real weights: correct, and slow for a known reason
**2026-08-09, M4, 24 GiB**

The first correct sentence: *"The capital of France is Paris."* Installed 48.07 GiB in
1500 s at 35 MB/s, **peak process footprint 61.4 MiB**, the streaming repacker held its
invariant on a checkpoint four times the 20B's.

| Quantity | Measurement |
| --- | ---: |
| Install on disk | 51.61 GB (1013 of 1013 tensors) |
| `resident.bin` | 4,790,321,152 B, D-021's 4.46 GiB floor, to the byte |
| `vision.bin` | 1,145,602,048 B, 356 tensors described |
| Expert blob stride | 11,894,784 B |

Throughput, 25 to 28 token prompts, reasoning off:

| Slots / layer | Cache hits | SSD read (prefill) | Decode | Footprint |
| ---: | ---: | ---: | ---: | ---: |
| 8 (minimum) | 65 % | 46.7 GiB | **1.42 tok/s** | 3.08 GiB |
| 32 | 86 % | 15.4 GiB | **1.05 tok/s** | 11.26 GiB |

**More cache bought a better hit rate and less throughput.** Two single runs on different
prompts is not a curve, so the size of the effect is not established, but the direction is
the opposite of the assumption, and it is the second time this project has been wrong about
cache size (M-027 was the first). It goes on the list to measure properly, not to explain
away.

D-021 estimated 2 to 2.5 tok/s; we are at 1.0 to 1.4. The gap is not mysterious. `Gemma4ModelRunner`
was written deliberately without the three optimizations `ModelRunner` earned by measurement,
batched prefill, read/compute overlap, speculative decoding, on the grounds that optimizing
an engine that had never produced a correct token would be optimizing something unproven. It
also commits and waits **twice per layer**, sixty synchronization points per token, where the
GPT-OSS path fuses. That condition has now expired: the engine is correct, so the
optimizations are justified work rather than speculation.

**Two bugs stood in the way, and both produced plausible output rather than an error.**

*Every logit NaN.* Metal compiles with fast math, where `tanh` is evaluated through
`exp(2x)`. Above a gate of ≈10.1 that overflows and `inf / inf` is NaN. Real Gemma reaches
gates of 11 at layer 26 of 30. The kernel test scaled its gate by **nine**, under the cliff
by a hair, so every operator test, the layer test and the whole-model test passed while the
first real run returned 262,144 NaNs. **A test configuration 64 wide cannot reach the range a
2816-wide model produces**; the fixtures have to be chosen against the real dynamic range, not
against a convenient one.

*Wrong tokenizer conventions.* A convenience initializer defaulted to `.gptOss`, so Gemma's
vocabulary loaded with GPT-2 byte-level encoding: every space became `Ġ` instead of `▁`,
`▁capital` never formed, and the model was handed a sequence no Gemma has ever seen. It
answered fluently, apostrophes and hyphens where *Paris* belonged. Removing the default made
the compiler name all five call sites.

**What the diagnosis cost, and why the tools were committed.** Throughput reported 6 tok/s at
99 % cache hits while every logit was NaN: the performance numbers were not merely unhelpful,
they were reassuring. `hydra logits --trace` narrowed it from "the model is broken" to
"`gelu_mul`, layer 26, five elements of 2112" in two runs, and `hydra weights` ruled out
misplacement in one. Both are commits rather than scratch prints for that reason.

---

## M-001, The repacker holds the memory invariant
**Milestone 1.1, ✔**

| Quantity | Measurement |
| --- | ---: |
| Installed | 12.82 GiB in 778 s |
| Peak throughput | 44 MB/s |
| **Process footprint** | **50.9 MiB** |
| Largest network block received | 3.6 MiB |
| Share of checkpoint size | **0.39 %** |

The footprint stayed at 50.9 MiB from 0 % to 100 %: it does not depend on the volume
transferred, which is exactly the property the project has to demonstrate.

Verification: 200 windows drawn at random from the installed files, re-requested from
Hugging Face, compared byte for byte, all match (6.78 MB).

---

## M-002, Large requests beat splitting, by a factor of 6.4
**A result that invalidated a design choice**

| Pattern | Throughput |
| --- | ---: |
| one 64 MiB `Range` request | 33.5 MB/s |
| eight 4 MiB `Range` requests in series | 5.2 MB/s |

Cause: Hugging Face answers with a 302 to a signed CDN; every request pays the redirect and
a TLS handshake again. The resolved URL is not reusable, its policy carries a `ByteRange`
condition tied to the range requested.

**Adopted:** one request per contiguous region, response consumed as it streams. The memory
bound becomes a measured property rather than one guaranteed by construction, trade-off
documented in D-013.

---

## M-003, Zero-copy mapping really is free

| Step | Process footprint |
| --- | ---: |
| before mapping | 557.9 MiB |
| after mapping 3.35 GiB | 558.7 MiB |

`mmap` + `makeBuffer(bytesNoCopy:)` over 2.27 GiB of `resident.bin` and 1.08 GiB of
`embed.bin` cost **0.8 MiB**. Pages only become resident on access.

---

## M-004, Parallel expert reads are worth a factor of 1.8 to 2.0

Four experts (52.9 MB), a fresh cache each round, a layer never read before, median over 5
rounds.

| Configuration | Serial | Parallel | Gain |
| --- | ---: | ---: | ---: |
| `F_NOCACHE` | 7.3 ms, 7.3 GB/s | **4.1 ms, 12.8 GB/s** | **×1.76** |
| page cache allowed | 7.6 ms, 7.0 GB/s | 4.0 ms, 13.3 GB/s | ×1.91 |

**Adopted:** a layer's missing reads are issued in parallel. It is the only I/O lever that
has paid off so far.

**An honesty caveat on the absolute figure.** 13 GB/s exceeds what an Apple SSD is expected
to deliver. `F_NOCACHE` bypasses the system page cache but not the controller's, and those
layers had been read earlier in the session. **The 1.8 to 2.0 ratio is solid; the absolute
bandwidth is not.** A genuinely cold measurement needs `sudo purge`, to be done before
freezing any throughput model.

For the record, a dedicated C benchmark over a 24 GiB file never read before, at random
offsets, gave 3.0 GB/s on one thread and 5.3 to 5.7 GB/s from four upwards.

---

## M-005, The GEMV kernel reaches 52 % of memory bandwidth
**After fixing a measurement error that had made it look five times slower**

`gate_up` of a real expert, [5760 × 2880] MXFP4, median over 7 rounds of 50 passes.

| Variant | ms | GB/s | deviation vs CPU reference |
| --- | ---: | ---: | ---: |
| `mxfp4_gemv` (reference) | 0.20 | 44.6 | 1.15e-07 |
| **`mxfp4_gemv_vectorized`** | **0.19** | **46.5** | 1.14e-07 |
| `mxfp4_gemv_simd` | 0.23 | 38.3 | 1.00e-07 |
| `mxfp4_gemv_tiled` | 0.24 | 37.1 | 1.00e-07 |

The machine's memory bandwidth: 89 to 98 GB/s depending on the measurement. The best kernel
uses **52 %** of it.

All four variants are numerically correct: deviation on the order of 1e-7 against the CPU
decoder, itself validated bit for bit, summed in double precision.

---

## M-006, The measurement error: 45 µs of synchronization per command buffer
**The most instructive negative result of the series**

A first campaign gave 0.95 ms for **every** variant, indistinguishably. A perfectly flat
plateau across four very different kernels is not a result, it is a symptom.

Cause: each measurement encoded **one** pass into a command buffer, committed it and
waited. An empty CPU-GPU round trip costs **45 µs**, and the measured time was dominated by
synchronization, not by the kernel.

Encoding 50 passes per buffer reveals the real time: **0.19 ms**, five times less. The
differences between variants become visible again.

**A design consequence, not only a measurement one.** One 20B token requires
4 experts × 24 layers × 2 GEMVs = 192 dispatches. At 45 µs of synchronization per buffer,
issuing them separately would cost 8.6 ms of pure latency. The execution graph must
therefore encode **a whole layer, or even a whole token, per command buffer**, which
justifies TurboFieldfare's `cb1` / `io` / `cb2` split after the fact.

---

## M-007, Two plausible optimizations that do not pay
**Negative results**

**Tiling activations in threadgroup memory.** Hypothesis: with one threadgroup per row, the
activation vector is re-read 5,760 times, i.e. 66 MB of traffic against 8.3 MB of weights.
Putting `x` in shared memory should divide that traffic.

Measurement: **0.24 ms against 0.19**, 21 % slower. The hypothesis was wrong, the 11.5 KiB
of activations fit comfortably in the GPU cache, which already serves them without going to
main memory. The explicit copy and the threadgroup barrier cost more than they save.

**One SIMD group per row.** Hypothesis: remove the shared memory and the barrier of the
two-stage reduction. Measurement: **0.23 ms against 0.19**, 18 % slower. With 32 lanes for
90 blocks, each lane handles three blocks and parallelism is lacking.

**Adopted:** neither becomes the default. Vectorizing loads (`uint4` instead of 16 `uchar`)
gains ×1.04, a small but reproducible margin with no change in output, so it is kept.

---

## M-008, Extrapolating the MoE alone

Based on the best kernel and the 20B's architecture:

```
0.19 ms × 1.5 (gate_up then down) × 4 experts × 24 layers ≈ 27 ms/token
```

That is **~37 tok/s for the MoE alone**, excluding attention, LM head and I/O. It is an
**optimistic upper bound**: it ignores the rest of the graph and assumes zero cache misses.

Compare with the theoretical floor computed in phase 0, 39.4 ms/token for the 20B across
all components. The two are consistent.

---

## What these measurements changed in the code

1. The repacker downloads by contiguous regions, ×8.5.
2. A layer's expert reads are issued in parallel, ×1.8.
3. The vectorized GEMV becomes the default, ×1.04.
4. Tiling and the SIMD variant are rejected, and documented as such.
5. The future execution graph will have to group dispatches by layer at minimum.

## Method

- Control and candidate **alternate** within a single campaign.
- We report the **median**, not the mean nor the best time.
- Every candidate is compared numerically against the CPU reference **before** being judged
  on time; a gain that changes outputs is not a gain.
- A difference within the noise leaves the default unchanged.

---

## M-009, The complete layer matches the reference on first assembly
**Milestone 1.3 extended, ✔**

Integration test on a miniature model installed by the **real repacker**: 4 layers, 6
experts, top-2, `hidden` 64, GQA group 2, sliding window of 8. Twelve tokens decoded in a
row, KV cache and window in play.

Maximum relative deviation against `ReferenceLayer`: **< 2e-3** at every position.

What this test covers and no unit test did:

- the **order** of operations within the layer;
- the **offsets** of every tensor in `resident.bin`;
- the **KV cache** layout and the circular indexing of the sliding window;
- the **router** wiring, the identifiers the GPU produces do drive the SSD load and then
  the compute;
- the **interleaving** of `gate_up` consumed by SwiGLU, and RoPE's split into halves, in
  the same graph.

It passed without correction, which was not a given: it is the first point where the
repacker, the format, zero-copy mapping, the expert cache and seven Metal kernels meet on
the same bytes.

---

## M-010, The router boundary forces two command buffers per layer

The router produces expert identifiers **on the GPU**, and the CPU must read them to know
which blobs to load from SSD. No reordering escapes this dependency: it cuts the layer in
two.

```
cb1: norm → QKV → RoPE → KV write → attention → O projection → residual
     → post-attention norm → router logits → top-k
I/O: read the identifiers, load the missing experts in parallel
cb2: per expert, gate_up → SwiGLU → down → weighted accumulation → residual
```

At 45 µs per round trip (M-006), that costs **2.2 ms per token on the 20B**, 24 layers × 2
buffers, before any compute at all. It is the structural price of having no shared expert:
in TurboFieldfare, the dense branch runs during the reads; here there is nothing to do
meanwhile.

---

## M-011, Prefill campaign: 18.0 s → 5.6 s, and four wrong hypotheses

78-token prompt, GPT-OSS 20B, 4 expert slots per layer.

| Step | Prefill |
| --- | ---: |
| Token by token (start) | 18.0 s |
| Chunked, naive GEMM | 6.5 s |
| + kernel selection by token count | 4.4 s |
| **Final state** | **5.6 s, 14 tok/s** |

**What I announced: 10 to 30×. Delivered: 3.2×.** The estimate counted only traffic on the
**weights** (92.9 GiB → 1.2 GiB, that factor of 78 is real) while ignoring traffic on the
**activations**, which turned out to be the dominant term.

### Hypotheses tested and rejected

1. **Threadgroup barriers dominate the reduction.** Going from 32 barriers per tile to one:
   *no change*.
2. **Bank conflicts in shared memory.** Offsetting by one float per row to make the stride
   coprime with 32 banks: *no change*.
3. **The `float weights[32]` array spills to registers.** Replaced by eight `float4`: *no
   change*.
4. **Grouping four rows per threadgroup amortizes the activations.** *+10 %.*

Four plausible fixes, one marginally useful. The common thread: all started from an
intuition about the kernel, none from a measurement of the kernel.

### What actually paid off

**An isolated bench, pass by pass.** It should have come first. It showed that a layer's
kernels total **10.9 ms**, 0.26 s for 24 layers, while the end-to-end measurement
attributed 2.09 s to the same portion. The 1.8 s gap was not compute.

**Prefaulting the pages.** The gap came from page faults: the first pass brought in the
2.27 GiB of `resident.bin` one page at a time. A sequential read at load time takes `cb1`
from **2.09 s to 0.88 s**. The cost moves to load time, so it is neutral for a single
prompt and a win across a whole session.

**Choosing the kernel by token count.** The tiled GEMM (`TILE_ROWS = 128`) launches only
`rows/128` threadgroups, 45 for `gate_up`, far short of what ten cores need. And in prefill
an expert serves only about eight tokens out of seventy. Below 32 tokens, the one-row-per-
SIMD-group variant launches thirty times more threadgroups and wins. **The right kernel
depends on the token count, not on the kind of operation.**

### The tiled GEMM, isolated

`q_proj` [4096 × 2880] BF16, median over 5 rounds of 20 passes:

| Tokens | Repeated GEMV | Tiled GEMM | Gain |
| ---: | ---: | ---: | ---: |
| 1 | **0.24 ms, 99 GB/s** | 1.21 ms | ×0.20 |
| 16 | 3.63 ms | 1.67 ms | ×2.18 |
| 68 | 15.66 ms | 3.57 ms | ×4.47 |
| 128 | 31.19 ms | 3.80 ms | **×8.22** |

The GEMV reaches 99 GB/s on a single token, the machine's bandwidth: it is optimal, and it
remains the right choice for decoding. The traffic model governing the tiled GEMM is
`cols × rows × tokens × (2/TILE_TOKENS + 4/TILE_ROWS)`.

---

## M-012, I/O overlap does not pay
**Negative result, reproducible, consistent with TurboFieldfare**

Expert reads launched in the background, each expert's compute submitted as soon as it
arrives. Alternating measurements:

| Round | With overlap | Without |
| --- | ---: | ---: |
| 1 | 5.21 tok/s | **7.27 tok/s** |
| 2 | 5.26 tok/s | **7.28 tok/s** |

**Overlap costs 28 %.** The cause is identified: submitting one command buffer per expert
instead of one per layer adds four synchronizations at 45 µs, and above all serializes the
GPU on shorter buffers. The I/O gain sought is more than cancelled out.

No better is possible without lifting the ordering constraint: the four experts share the
scratch and all accumulate into the same mixture buffer, so their command buffers cannot
overlap without corrupting each other.

TurboFieldfare measured exactly the same effect, 4.799 → 4.648 tok/s. The code is kept
behind `overlapExpertIO`, **disabled by default**, so the measurement stays reproducible.

---

## M-013, The decoding bottleneck is not I/O

Measured on GPT-OSS 20B, 32 tokens, M4 24 GiB. Four cache configurations:

| Slots/layer | Throughput | Hit rate | Cache memory | I/O share |
|---|---|---|---|---|
| 4 (minimum) | 4.58 tok/s | 86.2 % | 1.18 GiB | 19 % |
| 8 | 6.50 tok/s | 93.4 % | 2.37 GiB | 9 % |
| 16 | 5.29 tok/s | 95.2 % | 4.73 GiB | 14 % |
| 32 (the whole pool) | 4.66 tok/s | 95.5 % | 9.47 GiB | 13 % |

Two counter-intuitive conclusions.

**The hit rate was already good at the minimum.** 86 % with four slots for four active
experts: routing has enough temporal locality for the cache to work even at its floor size.
The literature on expert offloading (HOBBIT, arXiv 2411.01433) starts from a regime where
loading takes 94.5 % of the time; here it takes only 9 to 19 %. Predictive prefetching,
transition graphs or mixed precision would therefore optimize a marginal fraction of the
time.

**More memory does not buy speed.** Thirty-two slots, the whole pool resident, eight times
the footprint, are *slower* than eight. The hit gain (93.4 → 95.5 %) does not offset the
memory pressure. That is a direct argument for the project's thesis: the minimal footprint
is not merely sufficient, it is close to optimal.

Eight slots per layer are adopted as the default: better on both axes at once.

## M-014, Fusing command buffers: ×1.45

Decoding made seven CPU-GPU round trips per layer: attention, mixture start, one per
selected expert, end. Over twenty-four layers, **168 waits per token**.

The only genuinely necessary synchronization point is reading the router, the CPU must
know which experts were chosen before reading their weights. Everything else fits in the
same buffer, including the *next* layer's attention, encoded behind the current layer's
mixture. Since encoders within a buffer run in creation order, the dependency on the
residual is respected. That leaves **one wait per layer**.

Overlapping reads with compute was removed in the same move: it imposed one wait per expert
to hide an I/O that accounts for only 9 % of the time.

Paired measurement, 48 tokens, 8 slots, median of three runs:

| | before | after |
|---|---|---|
| throughput | 6.36 tok/s | **9.22 tok/s** |
| ms/token | 132 | 84 |
| attention + router | 44.5 ms (33 %) | 1.2 ms (1 %) |

The 43 ms that vanished from attention were pure waiting.

## M-015, Two kernel rewrites with no effect

Hypothesis: the GEMVs pay a reduction too expensive for the work done. The MXFP4 kernel
gives a row to 96 threads for 90 blocks, one block per thread, then `simd_sum`, shared
memory, a barrier and a serial sum. The BF16 kernel is worse still: 256 threads for 360
groups, i.e. 16 to 32 bytes per thread, and a reduction across eight SIMD groups.

Two variants written, giving a whole row to each SIMD group: reduction limited to
`simd_sum`, no barrier and no shared memory, and three to eleven times more work per thread.

| kernel | reference | one-row-per-SIMD variant |
|---|---|---|
| MXFP4 | 47.2 GB/s | 47.2 GB/s |
| BF16 (end to end) | 9.22 tok/s | 8.82 tok/s |

**No gain, one of the two slightly negative.** Both kernels already run at ~50 % of the
machine's memory bandwidth, and that is the limit, not the structure of the reduction.
Both variants were removed.

The theoretical floor is reachable by calculation: 3.7 GiB read per token, 1.27 for
attention, 1.27 for the experts, 1.16 for the LM head, i.e. 39 ms at 94 GB/s, or
25.6 tok/s. Going below that floor would require reading fewer bytes, that is, quantizing
the dense weights. That is ruled out (D-015).

## M-016, Top-p sampling cost more than the LM head

A factor of two separated the benchmark (9.2 tok/s) from the application (4 to 5). Three
causes, two of them outside the engine.

**The displayed throughput counted the prefill.** It started at the beginning of prompt
processing. On an established conversation the prompt is several thousand tokens and
processing it weighs more than the entire answer: the same engine displayed 6 tok/s on a
fresh conversation and 4 on a loaded one, while decoding at exactly the same speed. The cost
of the prefill remains visible, it is the time to first token, shown right beside it.

**The interface re-rendered the conversation on every token.** `MarkdownView` re-parses the
whole message on every render: on a growing message the cost is quadratic in its length.
That work runs on the main thread but consumes memory bandwidth, the very thing that limits
decoding. Fragments are now batched at 20 Hz, and formatting is computed only once the
answer is complete.

**The sampler sorted the entire vocabulary.** The benchmark decodes greedily; the
application samples with `top_p = 0.9`, and that branch built two arrays of 201,088 entries
per token, 2.4 MiB to allocate, fill and discard, then sorted them entirely to keep about
thirty.

Replaced by a 64-entry min-heap: one pass over the vocabulary, no allocation proportional to
its size, and the set is only widened if the target mass is not reached. Finding the maximum
is fused into the same pass.

| | before | after |
|---|---|---|
| application engine, 8 slots | 4.73 / 4.87 / 4.61 | 6.08 / 5.93 / 7.51 / 7.91 / 6.15 |

Three tests verify that the heap returns exactly the same candidates as a full sort, on
vocabularies up to 201,088 entries, including with equal values, a selection off by a
single rank would change the model's behaviour without signalling anything.

**A caveat on these figures.** Run-to-run variance reaches ±30 % at identical configuration.
The medians point the same way and the removed allocations are a fact independent of the
measurement, but the exact size of the gain is not established better than that order of
magnitude.

## M-017, Why the KV cache is not reused across turns

The prefill is paid in full on every turn, even though one turn's prompt is a prefix of the
next. Reusing the work already done would require rewinding the KV cache to the point of
divergence.

That is not possible as things stand: sliding-window layers keep their keys in a ring of 128
positions. An answer longer than 128 tokens has already overwritten the positions that would
need to be recovered. Rewinding is only safe over fewer than 128 tokens, that is, almost
never.

Making it possible requires giving those layers a full-length cache, hence more memory. That
is a trade-off to be posed explicitly against the project's thesis, not a shortcut to take in
passing.

## M-018, The eight-slot lead was the lead of the overheads it masked

M-013 measured 4.58 tok/s at four slots against 6.50 at eight, a 42 % gap, and the default
had been moved to eight on that basis.

After fixing GPU synchronization (M-014) and sampling (M-016), paired measurement in the
application, **same conversation, same context**:

| slots/layer | throughput | expert cache |
|---|---|---|
| 4 (minimum) | 7.7 tok/s | 1.18 GiB |
| 8 | 8.5 tok/s | 2.37 GiB |

The gap falls from 42 % to 10 %. The 42 % were not the cache's: fixed overheads dominated
decoding so heavily that they amplified any difference in I/O time. Once removed, the minimal
cache turns out to be nearly as fast.

Three runs of the command-line bench at each configuration give 5.61 to 7.68 against 5.53 to 7.33:
the distributions overlap, the bench cannot separate the two. The measurement adopted is the
one made in the application on a single conversation, which is paired where the bench is not.

**The default returns to the minimum.** Ten per cent is not worth 1.19 GiB in an application
whose point is to show what it is enough to hold in memory. The setting stays exposed for
anyone who prefers the opposite trade-off.

Overall result for this session: **4 to 5 tok/s → over 7**, and throughput stays above 7 as
the context fills up.

## M-019, The prefill, on the other hand, really is I/O-bound

Time to first token collapses with context length: 3 s on a fresh conversation, **30 s
beyond a thousand tokens**. The cause is the exact inverse of the decoding case.

A prefill chunk touches nearly all of a layer's experts, 128 tokens routed to 4 experts each
cover almost all 32, while the cache holds only 4. Each chunk therefore re-reads the whole
pool:

```
32 experts × 13.2 MB × 24 layers  =  10.1 GB per 128-token chunk
1000 tokens = 8 chunks            =  81 GB
at 4.3 GB/s                       ≈  19 s
```

That is indeed the observed order of magnitude. **Here, and only here, do the expert-
streaming techniques from the literature apply**: decoding has an 86 % hit rate, prefill has
none.

Since the per-token cost is inversely proportional to the chunk size, that size is the direct
lever. As the sliding-window ring is already sized on `slidingWindow + prefillChunk`,
increasing it stays correct by construction, and the two chunked-prefill equivalence tests
verify it.

Thousand-token prompt, total time to the end of the answer, two runs:

| chunk | total time | footprint |
|---|---|---|
| 128 | 39.2 / 43.6 s | 2683 MiB |
| **512** | **28.5 / 29.1 s** | 2727 MiB |
| 1024 | 25.1 / 27.7 s | 2802 MiB |

512 is adopted: 44 MiB for a third of the total time, where 1024 asks 119 MiB to gain 12 %
more. On prompts longer than a thousand tokens the trade-off would shift in favour of 1024.

## M-020, Reusing the KV cache across turns: time to first token ÷ 6

M-019 had the diagnosis wrong: prefill is not I/O-bound but compute-bound. On a thousand
tokens, measured via `HYDRA_PROFILE`:

```
prefill 1031 tokens:  I/O 3.8 s · compute 8.6 s · attention 3.8 s
```

I/O accounts for only 23 %. The expert pool is 9.47 GiB on a 24 GiB machine: the system page
cache absorbs most of it after the first pass. The gain from 512-token chunks was therefore
real, but because it amortizes the GEMMs better, not because it reads less.

Consequence: speeding up that compute is hard, two attempts on the kernels have already
failed (M-015). But that compute is also **entirely redundant**: every turn re-renders the
whole conversation and recomputes a prompt of which three tokens are new.

### What was blocking

Rewinding the cache to the point of divergence. Impossible with a ring: past its capacity it
has overwritten what would have to be recovered.

Sliding-window layers therefore get **linear** storage up to a context of 8192. The delicate
point is that `ringSize` governed both the storage *and* the reach of attention: linear
storage would have made those layers full-attention, changing the model without signalling
anything. The two notions are now distinct, `ringSize` for indexing, `windowed` for reach.

### Measurement

1031-token prompt, then a follow-up question in the same conversation:

| | time to first token |
|---|---|
| first turn | 18.8 s |
| **follow-up turn** | **3.1 s** |

Footprint: 2727 → 2808 MiB, i.e. **81 MiB measured**.

The follow-up turn is not free because Harmony does not put the analysis channel back into
history: the common prefix stops at the end of the previous prompt, and the answer plus the
new question are recomputed. It is the large prompt that is spared.

Beyond a context of 8192, the ring takes over again and so does the previous behaviour.

### Correctness

Three tests cover what would fail silently: linear storage must window exactly like the ring;
rewinding then resuming must equal a full computation; a modified prompt must restart at the
point of divergence rather than reuse stale keys.

## M-021, The 120B is in the inverse regime of the 20B

The 120B runs: 6.28 GiB resident, 2.32 engaged, 3.96 mapped, for a model of 60.77 GiB
installed, i.e. **10 %**. Without quantizing anything beyond the published MXFP4.

Breakdown of one token, 24 tokens, 4 slots per layer:

```
I/O  expert reads          155.9 ms   47 %
cb2  expert mixture        158.5 ms   48 %
LM head                     16.0 ms    5 %
cb1  attention + router      1.7 ms    1 %
```

**I/O accounts for 47 %, against 9 % for the 20B.** The hit rate falls to 76 %: four slots
for 128 experts instead of 32. This is the regime the expert-offloading literature describes
and therefore the only place in the project where its techniques would genuinely apply.
M-013's conclusion holds for the 20B, not for the 120B.

### Growing the cache degrades throughput

| slots/layer | throughput | hit rate | cache | I/O time |
|---|---|---|---|---|
| **4** | **2.15 tok/s** | 76.0 % | 1.78 GiB | 155.9 ms |
| 8 | 1.99 tok/s | 82.8 % | 3.55 GiB | 157.2 ms |
| 16 | 1.74 tok/s | 86.1 % | 7.10 GiB | 176.0 ms |

The hit rate improves markedly, 76 → 86 %, and **the I/O time does not benefit, it
increases**. The process cache displaces the system page cache, which was serving the misses
more cheaply. We pay twice: the memory, and the loss of what made it unnecessary.

This is the project's sharpest result in favour of its own thesis: on a constrained machine,
**growing the resident cache is counter-productive**, not merely useless.

### Remaining headroom

The bandwidth floor is ~4.9 GiB read per token, i.e. 52 ms at 94 GB/s, or 19 tok/s. At
311 ms we use 17 % of it. Two separate reserves of about 155 ms each: the I/O, which ideal
prefetching would remove, and the compute, at ~31 GB/s effective against 47 measured on the
kernel alone.

## M-022, Read/compute overlap has nothing to overlap

The 120B spends 46 % of its time reading experts. Overlapping those reads with compute
therefore looked worth nearly a factor of two: per layer, 4.3 ms of I/O against 4.4 ms of
compute, two ideally matched quantities.

Paired measurement, 120B, 4 slots, 24 tokens:

| | without overlap | with |
|---|---|---|
| ms/token | 311 | 309 |
| I/O | 152.8 ms | 0.0 ms *(absorbed)* |
| mixture | 160.1 ms | 310.6 ms |

**No gain.** The time changed counters, not value.

The cause is in the cache: `load(layer:experts:)` already reads the `top_k` experts **in
parallel** via `concurrentPerform`. All four therefore arrive together, and expert 0 is not
available before expert 3. There is no staggered availability to exploit. Staggering them to
create one would serialize the reads, 3.0 GB/s instead of 5.7, and cost more than the
overlap returns.

Overlapping layer `L+1` during the compute of `L` would require knowing its routing before
`L` has been computed. That is circular: the input to `L+1`'s router is the output of `L`.
Only a *prediction* would allow it (HOBBIT), at the risk of loading experts for nothing.

## M-023, Where the headroom stands on this machine

State of the 120B after all corrections: 314 ms/token, 4 slots, 6.28 GiB resident.

**The I/O half, 150 ms, is at the hardware ceiling.** A 76 % hit rate leaves on average
0.96 misses per layer: most reads are therefore **isolated**, served at 3.0 GB/s and not at
the 5.7 GB/s of the parallel regime. The 436 MB read per token at 2.9 GB/s effective matches
that regime exactly. Grouping those misses would require knowing several layers ahead, which
the sequential dependency of routing forbids.

**The compute half, 159 ms, retains headroom, but less than the theoretical floor
suggests.** The MXFP4 GEMV reaches 47 GB/s on the bench, but that bench re-reads the same
8.8 MB expert fifty times, and it fits comfortably in the system cache: the figure is
optimistic. In production each expert is read once, cold, and the effective rate is
11.5 GB/s. The real ceiling is probably around 20 to 25 GB/s, i.e. a compute floor near 80 ms
rather than the 19 ms of the pure bandwidth bound.

Realistic total headroom on this machine: **314 → ~235 ms**, i.e. ×1.3. Not ×4.

Going lower requires reading fewer bytes, hence quantizing the dense weights, ruled out
(D-015), or a machine whose memory bandwidth and disk throughput are not an M4's.

## M-024, Compute first what is already there

The overlap of M-022 had failed for a reason the hit-rate measurement made visible all
along: **at 76 % hit, on average only one expert in four is missing.** Three are already
resident and are only waiting for the GPU, but `load(layer:experts:)` blocked on all four
before starting any compute.

Decoding now computes the resident experts first, while the missing ones are read.

### What that required of the structure

Reordering the compute would change the order of the floating-point sum, hence the outputs,
and that order would depend on the state of the cache, making the model non-deterministic.
Each expert therefore writes into **its own slot**, and the sum is then taken in the fixed
order of the slots. The compute order becomes free, the addition order stays pinned.

### An expensive trap

First version: launch the reads, then encode the resident experts. The hit rate fell from
76 to 63.6 %. Encoding is what **pins** a slot; until it has happened, reads launched in the
background pick free slots as victims, precisely the ones about to be used. Encoding must
precede launching the reads.

### Measurement

120B, 4 slots, 24 tokens:

| | ms/token | I/O | mixture | hit |
|---|---|---|---|---|
| before | 311 to 314 | 150 ms (46 %) | 159 ms (48 %) | 76.0 % |
| reads before pinning | 301 to 325 | 96 ms | 220 ms | 63.6 % |
| **after** | **284 to 290** | 105 to 116 ms | 183 to 192 ms | 69.4 % |

**Net gain: 9 %.** The 20B does not regress (median 7.68 tok/s at 4 slots).

Far from the 35 % that arithmetic alone suggested, hiding 3.3 ms of compute behind 4.2 ms
of reading should have done better. The gap probably lies in unified memory: the read ends
in a copy into RAM, which consumes the bandwidth the GPU precisely needs. Two operations
limited by the same resource overlap only partially.

## M-025, Speculative decoding: attacking arithmetic intensity

The previous fixes hid or reorganized work. This one removes some: an ordinary pass re-reads
every weight to produce **one** token; a batched pass re-reads them once to verify `n`. On
the 120B, the dense weights, attention, routers, LM head, are 2.88 GiB re-read on every
token; over a batch of four, that is one read for four.

### Exactness

The output is identical token for token, at equal seed, whether the draft is right or wrong.
Two properties guarantee it:

- **Every emitted token consumes exactly one draw**, as without speculation: the
  pseudo-random sequence is therefore the same. This is the first thing that breaks if the
  algorithm is coded naively.
- **The logits at position `P+i` are used only if tokens `P..P+i-1` were accepted**, that
  is, only if the hypothesis under which they were computed held.

The first token is drawn *before* the batched pass: if the draft is wrong from the start, we
fall back on an ordinary step having spent nothing.

Four tests cover equivalence under both sampling modes, with correct, partially wrong and
nonsensical drafts. A faulty verifier would not crash, it would produce plausible, wrong
text.

### Where drafts come from

Pattern search in what has already been written: we look for the last occurrence of the last
three (then two) tokens and propose what followed. Zero cost, zero memory.

A second model would not do: the 20B is only 4.5 times cheaper than the 120B, and it would
need to be ten.

### Measurement

A copy task on the 20B, the favourable case, where the answer repeats the prompt:

| | tok/s |
|---|---|
| without speculation | 5.79 / 6.21 / 6.13 |
| **with** | **7.55 / 6.89 / 6.52** |

Median 6.13 → 6.89, i.e. **+12 %**. All three runs with exceed all three without, which is a
clear signal despite the usual ±30 % variance.

**On a prompt with no repetition the gain is nil**, the patterns are not found, no draft is
proposed, and decoding is exactly as before. The gain therefore depends on usage: high for
summarizing, rewriting, code, and questions about an attached document; nil in open-ended
conversation.

## M-026, Q8 on the dense weights: the per-position gate passes, the decision does not

D-020's gate, run without writing a kernel, a disk format or a repacker path.
`Q8.simulateInPlace` quantizes then dequantizes the resident weights in memory, so the
existing `bf16_gemv` reads exactly the values a real Q8 path would give it. Both runs use
`--reasoning high` on ten prompts with checkable conclusions
(`Tests/Fixtures/reasoning-prompts.txt`), greedy throughout, the candidate forced onto the
reference's tokens so that every position asks the same question.

| | positions | top-1 agreement | mean KL | worst KL | dense read/token |
|---|---|---|---|---|---|
| 20B, 512 tokens | 5 120 | **98.87 %** | 6.7e-4 nats | 4.0e-2 | 2.27 → 1.20 GiB |
| 120B, 256 tokens | 2 560 | **98.59 %** | 9.2e-4 nats | 3.5e-2 | 2.86 → 1.52 GiB |

**The 120B does not degrade more than the 20B**, despite a dense part that is
proportionally heavier. That was the open question, and the answer is no.

**Not one of the 94 changed positions was held with conviction.** The reference's own
probability for its pick, at the positions that moved, never exceeded **0.527**, it was
already hesitating between the two tokens. A different summation order would flip them just
as well.

### Why this does not settle it

Seven of the changed positions are **digits**, not stylistic choices:

```
20B   prompt  4, token 511   "236" (p=0.333) → "237" (p=0.332)
20B   prompt 10, token 445   "3"   (p=0.527) → "1"   (p=0.469)
20B   prompt 10, token 225   "0"   (p=0.487) → "300" (p=0.484)
120B  prompt  8, token 254   "100" (p=0.514) → "102" (p=0.485)
120B  prompt 10, token 135   "300" (p=0.506) → "0"   (p=0.487)
```

Teacher forcing is what makes the comparison attributable, each position is asked the same
question, and it is also what hides the consequence: the candidate is pushed back onto the
reference's path immediately, so we never observe whether a flipped digit would have
propagated to the conclusion.

**The measurement therefore answers "do the distributions move?" (barely) and not "does the
answer change?"** D-015 asks for the second: *outputs stay equivalent on a serious
evaluation*. That requires letting both models generate **freely** and comparing final
answers against known-correct ones, a different experiment, not a longer version of this
one.

Until then D-015 stays closed, and the 47 % stays unclaimed.

### An unexplained hang

One earlier 20B run wedged at ~33 minutes inside `encodeSingleExpert`, waiting on an expert
read, with zero CPU consumed over three samples two minutes apart. The binary had been
rebuilt underneath the running process, which is enough to invalidate the run and enough to
prevent attributing the hang. Worth watching for: if a read can fail to complete without
raising, it can strand a user mid-generation, and no timeout currently bounds that wait.

## M-027, Q8 on the dense weights: built, measured, removed

M-026 cleared the quality gate. The implementation that followed, split-layout quantizer,
two Metal kernels, install-time conversion, a precision-aware layout, was then measured and
**reverted**. What follows is why, so that the next person tempted by the same 66 % figure
does not spend the same week on it.

### The arithmetic that misled me

D-020 recorded, correctly, that the dense weights are **66 % of the bytes read per token**.
I concluded that halving them would buy 30 to 40 % of throughput. That inference is wrong, and
the per-token breakdown says so plainly:

| pass | ms/token | share |
|---|---|---|
| cb1 attention + router | 1.3 | 1 % |
| I/O reading the experts | 12.0 | 10 % |
| **cb2 mixture of experts** | **90.1** | **76 %** |
| LM head | 14.5 | 12 % |

The dense part is 66 % of the **bytes** and **13 % of the time**. Experts go through the SSD
and an ALU-heavy MXFP4 decode; their cost in seconds far exceeds their cost in bandwidth.
Reasoning in bytes and concluding in seconds is the whole of the error.

### The kernel did not convert bytes into time either

`q8_gemv` reads 47 % fewer bytes and ran **7 %** faster than `bf16_gemv` (13.5 ms against
14.5 on the LM head). Two shapes were wrong before that, both instructive:

- reading `char4` issued eight 4-byte loads per block where BF16 issues four 16-byte ones,
  the byte saving went into instruction issue;
- widening the load to a whole block halved the unit count to `cols/32`, and since the
  threadgroup width derives from it, the kernel launched **96 lanes against BF16's 256**.

Even correct, the GEMV is not weight-bound: for 16 bytes of weights a lane loads **64 bytes
of activations**, identical in both kernels. Those loads dominate, and no weight format
touches them.

**Net measured: +3.5 % end to end**, of which the dense path accounts for about 1 %.

### The memory argument, and why it did not save it

Q8 does free ~1 GiB on the 20B (2.27 → 1.21 GiB resident). The obvious next step is to spend
it on expert slots. Two findings closed that door:

1. **`MemoryBudget` never learned about precision.** It sizes from `config.residentBytes`,
   the BF16 figure, and picked the same 4 slots either way. The freed gigabyte was never
   allocated to anything, so the paired comparison, which held slots equal, could not have
   shown a benefit even if one existed.
2. **More slots is not faster.** Measured on the 120B: 16 slots per layer gives **2 tok/s at
   ~10 GB**, against 3.2 tok/s at 6.28 GiB with 4. Past a point the footprint evicts the
   mapped resident weights and they are re-faulted every token. This is D-012 measured again
  , minimizing memory is the goal, not filling the ceiling, and M-018 already recorded the
   same shape at eight slots.

If more slots do not help, freeing memory to buy more slots cannot help either.

### What stays

The code is gone; the numbers above are the point of the exercise. D-015 stands unchanged,
and now for a second reason: quantizing the dense weights is not only a quality risk, it buys
almost nothing on this architecture.

# Hydra

Run large mixture-of-experts models on a Mac that cannot hold them, **without quantizing them
into stupidity**.

A mixture-of-experts model activates a small fraction of its weights for any given token. The
rest have no reason to occupy memory. Hydra keeps the whole expert pool on SSD, holds a few
experts per layer in a slot cache, and memory-maps everything else. What is loaded is what is
used.

The weights are the publishers' own, decoded bit for bit. Nothing is requantized to make it
fit.

| Model | On disk | Resident | Share | Decode |
|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B (Q4) | 14.6 GiB | 1.83 GiB | 13 % | 8.0 to 8.6 tok/s |
| Gemma 4 26B-A4B (BF16) | 48.1 GiB | 3.69 GiB | 8 % | 1.6 to 2.4 tok/s |
| GPT-OSS 20B | 12.8 GiB | 2.74 GiB | 21 % | ~9 tok/s |
| GPT-OSS 120B | 60.8 GiB | 6.28 GiB | 10 % | 2 to 3 tok/s |

Measured on a 24 GiB M4 MacBook Pro at the application's default settings. Your numbers will
differ: throughput on this hardware moves 20 % with thermal state alone, which is why every
comparison in `docs/02-MEASUREMENTS.md` is run against a control built and timed beside it.

## Reading the speed numbers

Two numbers matter and they behave very differently.

**Decode** is the rate above, the speed at which words appear once they start. It is steady.

**Time to first token** is the wait before that, and it is proportional to **the tokens that
are not already in the cache**, not to the length of the conversation. A turn that continues a
long chat is fast because only your new message is processed. A fresh paste is slow because all
of it is new. Measured on Gemma 4 Q4:

| what | prompt | processed | wait |
|---|---:|---:|---:|
| fresh 800-token paste | 800 | 800 | 25.4 s |
| the next turn of that conversation | 830 | ~30 | 4.2 s |

Both rows are the same model on the same machine, seconds apart. The second is what normal use
looks like. The first is what a benchmark measures if it always starts a new conversation, and
it is the number most easily quoted out of context, including by us: earlier releases of this
file gave it as *the* time to first token, with no mention that continuing a chat costs a
sixth of it.

Hydra shows the processed count next to the timer on every answer, so the wait is never
mysterious.

## How it works

The expert pool lives on SSD. A bounded, LFU-evicted slot cache holds a few experts per layer;
dense weights stay memory-mapped.

One measured result is worth flagging, because it is the opposite of the obvious: **a bigger
cache is not reliably better**. On GPT-OSS 120B, thirty-two slots per layer (14.2 GiB) are
slower than four (1.78 GiB), because the process cache displaces the OS page cache that was
serving the misses cheaply. On Gemma 4 the curve is flatter, and doubling the cache buys about
3 % of decode for 758 MiB. The small footprint is not a compromise here. It is close to
optimal.

Prompt processing runs a chunk of tokens through each layer together, so an expert is read once
for the whole chunk rather than once per token that wanted it, and the projections are
matrix-matrix rather than one matrix-vector per token. Every batched kernel is the single-token
one with the token index moved onto the GPU's grid, at the same thread count and the same
reduction order, so processing a prompt in bulk agrees with feeding it one token at a time
**exactly**. A test asserts equality, not a tolerance.

## Installing

Requires **macOS 26 or later** on Apple Silicon.

1. Download `Hydra-x.y.z.dmg` from the releases page.
2. Open it and drag Hydra into Applications.
3. **macOS will refuse to open it the first time.** See below.

### First launch

Hydra is *ad-hoc signed*: it is not notarized by Apple, which requires a paid developer
account. macOS therefore warns on first launch and you must allow it explicitly once:

1. Double-click Hydra. macOS refuses and offers to move it to the Trash. **Click "Done".**
2. Open **System Settings, Privacy & Security**.
3. Scroll to the message about Hydra and click **"Open Anyway"**.
4. Confirm.

Or, in one command:

```bash
xattr -d com.apple.quarantine /Applications/Hydra.app
```

This warning does not mean the app is unsafe. It means Apple has not vetted it. If you would
rather not run anything you did not build yourself, building from source is one command and the
instructions are below.

### The models

Models are not bundled. Hydra downloads them from Hugging Face and converts them into its own
layout on the fly, never holding more than 51 MiB in memory while doing so. Start an install
from the sidebar.

Currently available: **Gemma 4 26B-A4B**, in Google's BF16 release and in the 4-bit MLX
quantization, and **GPT-OSS 20B and 120B** in OpenAI's published MXFP4. Qwen and DeepSeek are
planned.

### Where models are stored, and how to remove them

Models live under your user account, not inside the app bundle:

```
~/Library/Application Support/Hydra/Models/
```

That is deliberate: upgrading the app must not cost you a 60 GiB re-download.

**It also means dragging Hydra to the Trash leaves the models behind.** macOS runs no code when
an app is deleted, so no application can clean up after itself. This is a limitation of the
platform, not an oversight. Uninstall models from Hydra's sidebar before deleting the app, or
remove the folder yourself:

```bash
rm -rf ~/"Library/Application Support/Hydra"
```

That folder also holds `conversations.json`, your chat history.

## Building from source

```bash
git clone https://github.com/Neojarred/Hydra.git
cd Hydra
./Scripts/build-app.sh
open .build/release/Hydra.app
```

Xcode (or the Command Line Tools) is required to compile the Metal kernels.

Tests:

```bash
./Scripts/test.sh
```

They cover bit-exact MXFP4 decoding against OpenAI's reference, GPU against CPU equivalence for
every operator, byte-exact checkpoint reconstruction, the equivalence of speculative decoding
with ordinary decoding, and the equivalence of batched prompt processing with token-by-token
decoding.

## Documentation

`docs/02-MEASUREMENTS.md` holds every measurement, **including the negative results**. The
optimizations that were tried and did not pay are recorded with the reason why, alongside the
ones that did, and so are the times a conclusion was drawn from a benchmark that turned out to
be measuring something else. If you are working on something adjacent, that file is probably
the most useful thing here.

`docs/01-DECISIONS.md` records the trade-offs, including the one that defines the project:
never quantize the dense weights, even when it would be faster. A model made stupid is not a
model that runs.

`docs/00-FEASIBILITY.md` is the phase-0 study: what the machine actually measures, what the
checkpoints actually contain, and the throughput estimate made before any runtime code existed.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.

Apache 2.0 rather than MIT for a specific reason: inference is a patent-dense area, and Apache
grants an explicit patent license from every contributor, with termination if you sue over
patents on this software. MIT is silent on patents. It also matches the license of the models
Hydra runs.

Model weights are not distributed here and remain under their own licenses.

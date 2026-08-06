# Hydra

Run large mixture-of-experts models on a machine that cannot hold them — **without
quantizing them any further than their authors published**.

GPT-OSS 120B takes 60.77 GiB on disk. Hydra runs it on a 24 GiB M4 MacBook Air while
keeping only **6.28 GiB resident — 10 % of the model**. The weights are OpenAI's own, in
the published MXFP4 format, decoded bit-exactly. Nothing is requantized.

| Model | On disk | Resident | Share | Throughput on M4 |
|---|---|---|---|---|
| GPT-OSS 20B | 12.82 GiB | 3.77 GiB | 29 % | ~7–8 tok/s |
| GPT-OSS 120B | 60.77 GiB | 6.28 GiB | 10 % | ~2–3 tok/s |

## How it works

A mixture-of-experts model activates only a fraction of its weights per token — four
experts out of 128 for the 120B. The rest have no reason to occupy memory.

Hydra keeps the entire expert pool on SSD and holds only a few experts per layer in
memory, in an LFU-evicted slot cache. Dense weights — attention, routers, LM head — stay
memory-mapped. What is loaded is exactly what is used.

One measured result is worth flagging: **growing that cache makes throughput worse**. On
the 120B, thirty-two slots per layer (14.2 GiB) are slower than four (1.78 GiB). The
process cache displaces the OS page cache that was serving the misses cheaply. The minimal
footprint is not a compromise — it is close to optimal.

## Installing

Requires **macOS 26 or later** on Apple Silicon.

1. Download `Hydra-x.y.z.dmg` from the releases page.
2. Open it and drag Hydra into Applications.
3. **macOS will refuse to open it the first time.** See below.

### First launch

Hydra is *ad-hoc signed*: it is not notarized by Apple, which requires a paid developer
account. macOS therefore warns on first launch and you must allow it explicitly — once:

1. Double-click Hydra. macOS refuses and offers to move it to the Trash.
   **Click "Done", not "Move to Trash".**
2. Open **System Settings → Privacy & Security**.
3. Scroll to the message about Hydra and click **"Open Anyway"**.
4. Confirm.

Or, in one command:

```bash
xattr -d com.apple.quarantine /Applications/Hydra.app
```

This warning does not mean the app is unsafe. It means Apple has not vetted it. If you
would rather not run anything you did not build yourself, build from source — the
instructions are below and take one command.

### The models

They are not bundled — they weigh 12.8 and 60.8 GiB. Hydra downloads them from Hugging
Face and converts them on the fly into its own layout, never holding more than 51 MiB in
memory while doing so. Start an install from the sidebar.

Make room first: the 120B needs 60.8 GiB free.

## Building from source

```bash
git clone https://github.com/YOUR-ACCOUNT/hydra.git
cd hydra
./Scripts/build-app.sh
open .build/release/Hydra.app
```

Xcode (or the Command Line Tools) is required to compile the Metal kernels.

Tests:

```bash
./Scripts/test.sh
```

They cover bit-exact MXFP4 decoding against OpenAI's reference, GPU/CPU equivalence of
every operator, byte-exact checkpoint reconstruction, and the equivalence of speculative
decoding with ordinary decoding.

## What this repository documents

`docs/02-MESURES.md` holds every measurement, **including the negative results** — the
optimizations that were tried and did not pay are recorded with the reason why. If you are
working on something adjacent, that file is probably the most useful thing here.

`docs/01-DECISIONS.md` records the trade-offs, including the one that defines the project:
never quantize the dense weights, even when it would be faster. A model made stupid is not
a model that runs.

Both documents are currently written in French; the source comments are too. Translation
is welcome.

## License

MIT — see `LICENSE`.

Model weights are not distributed here and remain under their own license (Apache 2.0 for
GPT-OSS).

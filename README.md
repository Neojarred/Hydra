# Hydra

Run large mixture-of-experts models on a machine that cannot hold them **Without relying on heavy quantization**.

GPT-OSS 120B takes about 60 GiB on disk. Hydra runs it on a 24 GiB M4 MacBook pro while
keeping only **6.28 GiB resident or 10 % of the model**. The weights are OpenAI's own, in
the published MXFP4 format, decoded bit-exactly. Nothing is requantized.

| Model | On disk | Resident | Share | Throughput on M4 |
|---|---|---|---|---|
| GPT-OSS 20B | 12.82 GiB | 3.77 GiB | 29 % | ~7–8 tok/s |
| GPT-OSS 120B | 60.77 GiB | 6.28 GiB | 10 % | ~2–3 tok/s |

## How it works

A mixture-of-experts model activates only a fraction of its weights per token. GPT-OSS uses 4
experts out of 128 for the 120B version. The rest have no reason to occupy memory.

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
account. macOS therefore warns on first launch and you must allow it explicitly once:

1. Double-click Hydra. macOS refuses and offers to move it to the Trash.
   **Click "Done".**
2. Open **System Settings → Privacy & Security**.
3. Scroll to the message about Hydra and click **"Open Anyway"**.
4. Confirm.

Or, in one command:

```bash
xattr -d com.apple.quarantine /Applications/Hydra.app
```

This warning does not mean the app is unsafe. It means Apple has not vetted it. If you
would rather not run anything you did not build yourself, you can build it from the provided source code. The
instructions are below and take one command.

### The models

They are not bundled! At the moment we are only using GPT-OSS models for this project, but we are planning to expand the model pool with Gemma, Qwen and Deepseek options. GPT-OSS weighs 12.8 and 60.8 GiB respectively for the 20B and 120B options. Hydra downloads them from Hugging
Face and converts them on the fly into its own layout, never holding more than 51 MiB in
memory while doing so. You can start an install from the sidebar.

### Where models are stored, and how to remove them

Models live under your user account, not inside the app bundle:

```
~/Library/Application Support/Hydra/Models/
```

That is deliberate — upgrading the app must not cost you a 73 GiB re-download.

**It also means dragging Hydra to the Trash leaves the models behind.** macOS runs no code
when an app is deleted, so no application can clean up after itself; this is a limitation
of the platform, not an oversight. Uninstall models from Hydra's sidebar before deleting
the app, or remove the folder yourself:

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

They cover bit-exact MXFP4 decoding against OpenAI's reference, GPU/CPU equivalence of
every operator, byte-exact checkpoint reconstruction, and the equivalence of speculative
decoding with ordinary decoding.

## Documentation

`docs/02-MEASUREMENTS.md` holds every measurement, **including the negative results** — the
optimizations that were tried and did not pay are recorded with the reason why. If you are
working on something adjacent, that file is probably the most useful thing here.

`docs/01-DECISIONS.md` records the trade-offs, including the one that defines the project:
never quantize the dense weights, even when it would be faster. A model made stupid is not
a model that runs.

`docs/00-FEASIBILITY.md` is the phase-0 study: what the machine actually measures, what the
checkpoints actually contain, and the throughput estimate made before writing any runtime
code.

Source comments are still in French; translating them is in progress.

## License

Apache License 2.0 — see `LICENSE` and `NOTICE`.

Apache 2.0 rather than MIT for a specific reason: inference is a patent-dense area, and
Apache grants an explicit patent license from every contributor, with termination if you
sue over patents on this software. MIT is silent on patents. It also matches the license
of the models Hydra runs.

Model weights are not distributed here and remain under their own license (Apache 2.0 for
GPT-OSS).

#!/usr/bin/env python3
"""How badly a generation degenerated, as a magnitude rather than a rate.

M-072's warning is the design brief: at twelve seeds none of the *rate* differences were
significant, and the only column carrying information was the worst-run magnitude — 88 against 0
is not a statistical claim.  So this reports magnitudes.

M-072's other lesson is that the instrument is the likely fault, not the thing measured.  Six of
its measurements were wrong about their own method.  So `--self-test` runs known-bad and
known-good text through the detector and prints what it says, and the batch refuses to report
numbers until that passes.
"""
import argparse, collections, json, os, re, sys

WORD = re.compile(r"\S+")


def cycle(text, max_period=64):
    """Longest cyclic repeat at **any** period, as (repeats, period, phrase).

    Period-agnostic on purpose.  The first version of this looked only for an n-gram repeating
    exactly n words later, which catches `multilingual support, multilingual support,` and is
    stone blind to `the previous turn had a previous turn with ...` because that cycle is eight
    words long.  The self-test caught it, which is the only reason this docstring exists.
    """
    words = WORD.findall(text)
    best = (0, 0, "")
    for period in range(1, min(max_period, len(words)) + 1):
        run = 0
        for i in range(period, len(words)):
            if words[i] == words[i - period]:
                run += 1
                repeats = run // period
                if repeats > best[0]:
                    best = (repeats, period,
                            " ".join(words[i - period + 1:i + 1])[:60])
            else:
                run = 0
    return best


def top_gram(text, n=4):
    words = WORD.findall(text)
    counts = collections.Counter(
        " ".join(words[i:i + n]) for i in range(len(words) - n + 1))
    if not counts:
        return ("", 0)
    return counts.most_common(1)[0]


# The harness writes a load header before the generation and a kernel dispatch table after it.
# Neither is the model's output, and scoring them was an instrument fault of exactly the kind
# M-072 catalogues: numbers that look like measurement and are partly measuring the harness.
HARNESS_TAIL = re.compile(r"\n\s*\d[\d,]* dispatches over ", re.S)
HARNESS_HEAD = re.compile(r"^.*?(?:\[reasoning\]|prefill…\n)", re.S)


def generated_only(text):
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    cut = HARNESS_TAIL.search(text)
    if cut:
        text = text[:cut.start()]
    text = HARNESS_HEAD.sub("", text, count=1)
    # The per-token prefill statistics block, if it survived the head cut.
    text = re.sub(r"^\s*\d+ tokens in [\d.]+ s.*$", "", text, flags=re.M)
    text = re.sub(r"^\s+(?:mlx_|qwen_|rms_|router_|sum_|kv_|add_|fill_|gather|scatter)\S*"
                  r"\s+[\d\s.]+$", "", text, flags=re.M)
    return text


def score(text):
    repeats, period, phrase = cycle(text)
    return {
        "words": len(WORD.findall(text)),
        # The magnitude M-072 says is the only column that carries information.
        "repeats": repeats,
        "period": period,
        "phrase": phrase,
        "top4_count": top_gram(text, 4)[1],
        "top4": top_gram(text, 4)[0][:60],
    }


def flagged(s):
    """Three back-to-back repeats of anything is not prose."""
    return s["repeats"] >= 3


def self_test():
    """Known-bad first, so a blind detector announces itself before any batch is believed."""
    cases = [
        ("known-bad: verbatim word loop",
         "answer " + "multilingual support, " * 43, True),
        ("known-bad: phrase loop",
         "In the previous turn, " + "the previous turn had a previous turn with " * 10, True),
        ("known-good: ordinary prose",
         "The M4 Max reaches 546 GB/s of memory bandwidth, against 273 for the M4 Pro. "
         "The binned 32-core part is lower again, at 410 GB/s, which is the figure Apple "
         "lists for the fourteen-core configuration of the same chip.", False),
        ("known-good: legitimate repetition",
         "Qwen is a model. Qwen has experts. Qwen runs locally. Each Qwen layer routes "
         "tokens to a few of its experts rather than to all of them.", False),
    ]
    ok = True
    print(f"{'case':38} {'words':>6} {'rep':>5} {'per':>4}  verdict")
    for name, text, should_flag in cases:
        s = score(text)
        hit = flagged(s)
        mark = "FLAG" if hit else "clean"
        agree = hit == should_flag
        ok &= agree
        print(f"{name:38} {s['words']:6d} {s['repeats']:5d} {s['period']:4d}  "
              f"{mark}{'' if agree else '   <-- WRONG'}")
    print("\nself-test", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--conversations", action="store_true",
                    help="score the stored app conversations, the original bad runs")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.conversations:
        p = os.path.expanduser("~/Library/Application Support/Hydra/conversations.json")
        for c in json.load(open(p)):
            for m in c["messages"]:
                for v in m["variants"]:
                    text = (v.get("reasoning") or "") + "\n" + (v.get("text") or "")
                    if len(WORD.findall(text)) < 50:
                        continue
                    s = score(text)
                    print(f"{c['modelID']:24} out={str(v.get('outputTokens')):>5} "
                          f"repeats={s['repeats']:4d} period={s['period']:3d} "
                          f"{'FLAG ' if flagged(s) else '     '}{s['phrase']!r}")
        return

    for path in args.files:
        text = generated_only(open(path, errors="replace").read())
        s = score(text)
        print(f"{os.path.basename(path):26} words={s['words']:5d} "
              f"repeats={s['repeats']:4d} period={s['period']:3d} "
              f"{'FLAG ' if flagged(s) else '     '}{s['phrase']!r}")


if __name__ == "__main__":
    main()

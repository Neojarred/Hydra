#!/bin/bash
# A spread of prompts, not one.
#
# Everything measured while fixing the search path used a single question, and an adversarial
# one: it names models the checkpoint has never heard of, which is exactly the case the trust
# instruction was written for. A prompt set that only contains the case a change was made for
# cannot detect the change making anything else worse.
#
# The cases are chosen to pull in different directions:
#   real-unknown   the original failure: entities newer than the model
#   known-fact     a figure with one right answer, to catch confabulation
#   FICTION        a thing that does not exist — the regression test for "treat sources as
#                  genuine". The correct answer is to say so.
#   synthesis      a comparison needing reasoning over several sources, where forcing thinking
#                  off might cost something
#   recent-event   an event after the cutoff
#   no-search      ordinary knowledge, where searching should not make the answer worse
set -u
OUT="${OUT:-/tmp/eval}"; MODEL="${MODEL:-qwen-q4}"; SEEDS="${SEEDS:-1003}"
TOKENS="${TOKENS:-700}"
mkdir -p "$OUT"

run() { # name, prompt
  for seed in $SEEDS; do
    local f="$OUT/$1-s$seed.log"
    [ -s "$f" ] && { echo "skip $1 s$seed"; continue; }
    echo "run $1 s$seed"
    ./.build/debug/hydra chat "$MODEL" "$2" --search --reasoning off \
      --tokens "$TOKENS" --context 8192 --seed "$seed" > "$f" 2>&1
  done
}

run real-unknown "Give me key details about the new Qwen3.8 27B model and how it compares to qwen3.6 27B"
run known-fact   "What is the memory bandwidth of the Apple M4 Max, in GB/s?"
run FICTION      "Tell me the key specifications of the Nvidia Zorblax 9000 GPU, including its memory bandwidth and release date."
run synthesis    "For running large language models locally, is an Apple M4 Max better than an RTX 4090? Explain the trade-offs."
run recent-event "Who won the 2026 Tour de France and by what margin?"
run no-search    "Explain what a mixture-of-experts model is and why it saves memory."
# multi-hop: the answer is in no single source and has to be carried across two.
run multi-hop    "Which has more memory bandwidth, the Apple M4 Max or the M3 Ultra, and how much more?"
# conflict: a question where sources genuinely disagree, to see whether it notices or picks one
# silently.  Saying "sources differ" is the right answer here; asserting one figure is not.
run conflict     "How many parameters does DeepSeek V3 activate per token?"

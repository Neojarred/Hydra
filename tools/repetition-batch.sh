#!/bin/bash
# Does bounding the presence penalty's window change how badly a long generation degenerates?
#
# Interleaved by construction: each seed runs every configuration back to back, so thermal
# state and page-cache warmth land on all of them equally.  This machine moves 20 % on thermal
# state alone, which is enough to invent a difference that is not there.
#
# Seeds are odd only.  `TokenSampler` seeds itself with `seed | 1`, so consecutive seeds are the
# same stream — M-072 found six seeds were four.
set -u
OUT="${OUT:-/tmp/reps}"
MODEL="${MODEL:-qwen-q4}"
TOKENS="${TOKENS:-1200}"
PROMPT="${PROMPT:-Give me key details about the new Qwen3.8 27B model and how it compares to qwen3.6 27B}"
SEEDS="${SEEDS:-1001 1003 1005 1007}"
# Each arm is window:frequency-penalty, run back to back within a seed.
# window:frequency:presence
ARMS="${ARMS:-256:0:1.5 256:0.05:1.5}"
mkdir -p "$OUT"

# The instrument first.  A batch whose detector is blind reports a clean, confident, entirely
# fictitious table (M-072).
python3 tools/degeneration.py --self-test || { echo "detector self-test failed, refusing"; exit 1; }
echo

for seed in $SEEDS; do
  for arm in $ARMS; do
    w=$(echo "$arm" | cut -d: -f1)
    fp=$(echo "$arm" | cut -d: -f2)
    pp=$(echo "$arm" | cut -d: -f3)
    f="$OUT/w${w}-f${fp}-p${pp}-${REASONING:-medium}-s${seed}.log"
    [ -s "$f" ] && { echo "skip $f"; continue; }
    echo "run window=$w freq=$fp presence=$pp seed=$seed"
    # stderr merged: the reasoning trace lives there, and a harness that drops it measures
    # thinking mode with the thinking thrown away.
    ./.build/debug/hydra chat "$MODEL" "$PROMPT" \
      --analysis --tokens "$TOKENS" --context 8192 ${SEARCH:+--search} \
      --reasoning "${REASONING:-medium}" \
      --seed "$seed" --repeat-window "$w" --frequency-penalty "$fp" \
      --presence-penalty "$pp" > "$f" 2>&1
  done
done
echo
echo "=== results ==="
python3 tools/degeneration.py "$OUT"/*.log

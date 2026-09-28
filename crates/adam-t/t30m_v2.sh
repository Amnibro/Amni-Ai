#!/usr/bin/env bash
CD=$(cd "$(dirname "$0")" && pwd)
HM=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; ML=${MEM_LOG2:-18}; R=${RUN_DIR:-$HM/exports/adam_t/t30m_v2}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$HM RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=${MEM_MAX:-12G} NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
[ "$ML" -ge 15 ] && [ "$ML" -le 18 ] || { echo "MEM_LOG2 $ML outside 15..18: 2^19 and up do not fit the 16 GB card"; exit 5; }
export D=512 L=10 H=8 FF=2048 CTX=${CTX:-1024} TERN=1 MEM=1 NGRAMS=2,3,4,6 MEM_LOG2=$ML MEM_OPT=lazy LPE=1 LPE_PHI=4 LPE_S=64 BPE=$HM/data/clean2/bpe4096.json CORPUS=adam_t/stream SEED=${SEED:-0} BASE= INIT= SRC_W= EPOCHS=0 MODE=train
export BS=${BS:-16} MICRO=${MICRO:-16} STEPS=${STEPS:-35000} WARM=1000 LR=2e-3 LR_MEM=2e-3 WD=0.05 EVAL_EVERY=1000 EVAL_BATCHES=10 CKPT_EVERY=${CKPT_EVERY:-1000} SEG=${SEG:-30} EVAL_SEGS=${EVAL_SEGS:-16} EVAL_MEM=${EVAL_MEM:-$([ "$ML" -ge 18 ] && echo 8G || echo 4G)} PARITY_WINDOWS=2
export TAG=$(basename "$R") RUN_DIR=$R OUT=$R BIN=$R/bin/adam-t
[ "${DRY:-0}" = 1 ] && { echo "=== dry: t30m_v2 -> $R"; env | grep -E "^(D|L|H|FF|CTX|TERN|MEM|NGRAMS|MEM_LOG2|MEM_OPT|LPE|LPE_PHI|LPE_S|BPE|CORPUS|SEED|BASE|INIT|MODE|BS|MICRO|STEPS|WARM|LR|LR_MEM|WD|EVAL_EVERY|EVAL_BATCHES|CKPT_EVERY|SEG|EVAL_SEGS|EVAL_MEM|PARITY_WINDOWS|TAG|RUN_DIR|OUT|BIN|MEM_MAX|CPU_Q|NICE|RAYON_NUM_THREADS)=" | sort; exit 0; }
ME=$(basename "$(cut -d: -f3 /proc/$$/cgroup)")
for u in $(systemctl --user list-units --state=active --no-legend --plain 'adam-t-*' | awk '{print $1}'); do [ "$u" = "$ME" ] || { echo "unit $u active: one GPU job at a time, not starting"; exit 4; }; done
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
[ ! -f "$R/log.jsonl" ] || head -n 1 "$R/log.jsonl" | grep -q "\"mem_log2\": $ML,.*\"lpe\": true.*\"mem_opt\": \"lazy\"" || { echo "$R was started with another MEM_LOG2/LPE/MEM_OPT: $(head -c 400 "$R/log.jsonl")"; exit 5; }
mkdir -p "$R/bin" || exit 5
[ -f "$R/bin/adam-t.sha256" ] || { cp "${BIN_SRC:-$HM/crates/target/release/adam-t}" "$R/bin/adam-t" && cp "$CD/campaign.sh" "$CD/smoke.sh" "$CD/t30m_v2.sh" "$R/bin/" && sha256sum "$R/bin/adam-t" > "$R/bin/adam-t.sha256" && git -C "$HM" rev-parse HEAD > "$R/bin/commit.txt"; } || exit 5
cd "$HM" || exit 5
exec bash "$R/bin/campaign.sh"

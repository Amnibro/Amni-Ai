#!/usr/bin/env bash
CD=$(cd "$(dirname "$0")" && pwd)
H=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; ML=$([ "${CONTROL:-0}" = 1 ] && echo 15 || echo "${MEM_LOG2:-18}"); R=${RUN_DIR:-$H/exports/adam_t/t30m_mem$ML}; BR=${BASE_RUN:-$H/exports/adam_t/t30m_bpe4096}; BS0=${BASE_STEP:-35000}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$H RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=${MEM_MAX:-12G} NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
[ "$ML" -ge 15 ] && [ "$ML" -le 20 ] || { echo "MEM_LOG2 $ML outside 15..20"; exit 5; }
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
for u in ${BUSY_UNITS:-adam-t-30m adam-t-sft adam-t-lorentz}; do systemctl --user is-active --quiet "$u" && { echo "unit $u active: one GPU job at a time, not starting"; exit 4; }; done
mkdir -p "$R/bin" "$R/init/ckpt" || exit 5
[ -f "$R/bin/adam-t.sha256" ] || { cp "${BIN_SRC:-$H/crates/target/release/adam-t}" "$R/bin/adam-t" && cp "$CD/campaign.sh" "$CD/smoke.sh" "$CD/t30m_mem.sh" "$R/bin/" && sha256sum "$R/bin/adam-t" > "$R/bin/adam-t.sha256" && git -C "$H" rev-parse HEAD > "$R/bin/commit.txt"; } || exit 5
[ -f "$R/init/ckpt/mem.bin" ] || { [ "$(cat "$BR/ckpt/state.txt")" = "$BS0" ] || { echo "base $BR is at step $(cat "$BR/ckpt/state.txt"), want $BS0"; exit 5; }; cp "$BR/ckpt/state.txt" "$BR/ckpt/core.bin" "$BR/ckpt/mem.bin" "$R/init/ckpt/" && head -n 1 "$BR/log.jsonl" > "$R/init/log.jsonl" && (cd "$R/init" && sha256sum ckpt/core.bin ckpt/mem.bin > sha256.txt) || exit 5; echo "=== t30m_mem: base snapshot $BR step $BS0 -> $R/init, MEM_LOG2 15 -> $ML" | tee -a "$R/run.log"; }
export INIT=$R/init MEM_LOG2=$ML MEM_OPT=${MEM_OPT:-lazy} CORPUS=adam_t/stream BPE=$H/data/clean2/bpe4096.json TERN=1 MEM=1 SEED=${SEED:-1}
export BS=16 MICRO=2 STEPS=${STEPS:-9000} WARM=${WARM:-200} LR=${LR:-4e-4} LR_MEM=${LR_MEM:-${LR:-4e-4}} WD=0.05 EVAL_EVERY=1000 EVAL_BATCHES=10 CKPT_EVERY=250 SEG=${SEG:-30} EVAL_SEGS=${EVAL_SEGS:-16} EVAL_MEM=${EVAL_MEM:-$([ "$ML" -ge 18 ] && echo 8G || echo 4G)} PARITY_WINDOWS=2
export TAG=$(basename "$R") RUN_DIR=$R OUT=$R BIN=$R/bin/adam-t
cd "$H" || exit 5
exec bash "$R/bin/campaign.sh"

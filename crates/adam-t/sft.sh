#!/usr/bin/env bash
H=/home/anmire/ai/Amni-Ai; R=${RUN_DIR:-$H/exports/adam_t/t30m_sft}; BR=${BASE_RUN:-$H/exports/adam_t/t30m_bpe4096}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$H RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=12G NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
systemctl --user is-active --quiet "${BASE_UNIT:-adam-t-30m}" && { echo "base unit ${BASE_UNIT:-adam-t-30m} still active: wait for it to finish or stop it"; exit 4; }
mkdir -p "$R/bin" "$R/base/ckpt" || exit 5
[ -x "$R/bin/adam-t" ] || cp "$H/crates/target/release/adam-t" "$H/crates/adam-t/campaign.sh" "$H/crates/adam-t/smoke.sh" "$H/crates/adam-t/sft.sh" "$R/bin/" || exit 5
[ -f "$R/base/ckpt/core.bin" ] || { { [ -f "$BR/model.gf17" ] || [ "${ALLOW_PARTIAL_BASE:-0}" = 1 ]; } || { echo "base $BR has no model.gf17 (not finished); ALLOW_PARTIAL_BASE=1 tunes its latest checkpoint"; exit 5; }; cp "$BR/ckpt/state.txt" "$BR/ckpt/core.bin" "$R/base/ckpt/" && { [ ! -f "$BR/ckpt/mem.bin" ] || cp "$BR/ckpt/mem.bin" "$R/base/ckpt/"; } && head -n 1 "$BR/log.jsonl" > "$R/base/log.jsonl" || exit 5; echo "=== sft: base snapshot $BR step $(cat "$R/base/ckpt/state.txt") -> $R/base" | tee -a "$R/run.log"; }
export BASE=$R/base CORPUS=${CORPUS:-adam_t/sft} MODE=sft EPOCHS=${EPOCHS:-$([ -n "${STEPS:-}" ] && echo 0 || echo 2)} LR=${LR:-3e-4} WARM=${WARM:-100} WD=${WD:-0.05} SRC_W=${SRC_W:-} SEED=${SEED:-0}
export LR_MEM=${LR_MEM:-$LR} BS=${BS:-16} MICRO=${MICRO:-2} EVAL_EVERY=${EVAL_EVERY:-250} EVAL_BATCHES=${EVAL_BATCHES:-0} CKPT_EVERY=250 SEG=${SEG:-30} EVAL_SEGS=${EVAL_SEGS:-4} EVAL_ARGS=${EVAL_ARGS:---suites qa,rc,mem,bpb,zs,chat} PARITY_WINDOWS=2
export TAG=$(basename "$R") RUN_DIR=$R OUT=$R BIN=$R/bin/adam-t
exec bash "$R/bin/campaign.sh"

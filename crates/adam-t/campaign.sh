#!/usr/bin/env bash
set -u
export AMNI_HOME=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}
HM=$AMNI_HOME; BIN=${BIN:?set BIN to the adam-t binary}; RUN_DIR=${RUN_DIR:?set RUN_DIR}; SEG=${SEG:-120}; PY=${PY:-$HM/.venv-linux/bin/python}
cd "$HM"
while :; do
  K=$((${K:-0}+1)); MAX_MIN=$SEG nice -n ${NICE:-15} bash "$(dirname "$0")/smoke.sh" "$BIN" train; R=$?
  [ $R -eq 0 ] || { echo "=== campaign: trainer exit $R, stopping" | tee -a "$RUN_DIR/run.log"; exit $R; }
  S=$(cat "$RUN_DIR/ckpt/state.txt")
  [ -f "$RUN_DIR/STOP" ] && { echo "=== campaign: STOP file, paused at $S $(date +%T)" | tee -a "$RUN_DIR/run.log"; exit 0; }
  [ "${EVAL:-1}" = 1 ] && [ $((K % ${EVAL_SEGS:-1})) -eq 0 -o "$S" -ge "${STEPS:-0}" ] && { echo "=== campaign eval step $S $(date +%T)" >> "$RUN_DIR/eval.log"; RAYON_NUM_THREADS=${EVAL_THREADS:-3} nice -n 15 systemd-run --user --scope -q -p MemoryMax=${EVAL_MEM:-4G} -p CPUQuota=${EVAL_CPU:-300%} --setenv=RAYON_NUM_THREADS=${EVAL_THREADS:-3} --setenv=THROTTLE_FILE="$THROTTLE_FILE" "$PY" scripts/adam_t_eval.py run --model proto --proto-cmd "$BIN serve $RUN_DIR" --step "$S" ${EVAL_ARGS:-} >> "$RUN_DIR/eval.log" 2>&1; echo "=== campaign eval exit $? $(date +%T)" >> "$RUN_DIR/eval.log"; }
  [ -f "$RUN_DIR/model.gf17" ] && [ "$S" -ge "${STEPS:-0}" ] && break
done

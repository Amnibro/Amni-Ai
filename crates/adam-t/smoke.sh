#!/usr/bin/env bash
set -u
export AMNI_HOME=${AMNI_HOME:-/home/anmire/ai/Amni-Ai} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-4}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file}
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
pgrep -x adam-train >/dev/null && { echo "adam-train running: GPU busy, not starting"; exit 4; }
pgrep -x adam-t >/dev/null && { echo "adam-t already running: one GPU job at a time"; exit 4; }
D=${RUN_DIR:?set RUN_DIR}; mkdir -p "$D"; B=/sys/class/drm/card1/device/gpu_busy_percent
( while :; do echo "$(date +%s) $(cat $B)"; sleep 1; done ) > "$D/gpu_busy.log" & S=$!
echo "=== $(date +%T) $*" | tee -a "$D/run.log"
systemd-run --user --scope -q -p MemoryMax=6G -p CPUQuota=400% --setenv=RAYON_NUM_THREADS="$RAYON_NUM_THREADS" "$@" 2>&1 | grep --line-buffered -v "radv is not a conformant" | tee -a "$D/run.log"; R=${PIPESTATUS[0]}
kill "$S"
awk -v c=60 '{n++; s+=$2; m=($2>m)?$2:m; o+=($2>c)} END {printf "{\"gpu_busy_samples\": %d, \"gpu_busy_mean\": %.1f, \"gpu_busy_max\": %d, \"frac_over_60\": %.3f}\n", n, s/n, m, o/n}' "$D/gpu_busy.log" | tee -a "$D/run.log"
echo "=== exit $R $(date +%T)" | tee -a "$D/run.log"
exit "$R"

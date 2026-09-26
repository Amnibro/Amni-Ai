#!/usr/bin/env bash
set -u
export AMNI_HOME=${AMNI_HOME:-/home/anmire/ai/Amni-Ai} RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file}
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
pgrep -x adam-train >/dev/null && { echo "adam-train running: GPU busy, not starting"; exit 4; }
for p in $(pgrep -x adam-t); do tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -qE "^[^ ]+ (train|sft)( |$)" && { echo "adam-t $p training: one GPU job at a time"; exit 4; }; done
_RD=${RUN_DIR:?set RUN_DIR}; mkdir -p "$_RD"; _BZ=/sys/class/drm/card1/device/gpu_busy_percent
( while :; do echo "$(date +%s) $(cat $_BZ)"; sleep 1; done ) > "$_RD/gpu_busy.log" & _SP=$!
echo "=== $(date +%T) $*" | tee -a "$_RD/run.log"
systemd-run --user --scope -q -p MemoryMax=${MEM_MAX:-12G} -p CPUQuota=${CPU_Q:-800%} --setenv=RAYON_NUM_THREADS="$RAYON_NUM_THREADS" "$@" 2>&1 | grep --line-buffered -v "radv is not a conformant" | tee -a "$_RD/run.log"; R=${PIPESTATUS[0]}
kill "$_SP"
awk -v c=60 '{n++; s+=$2; m=($2>m)?$2:m; o+=($2>c)} END {printf "{\"gpu_busy_samples\": %d, \"gpu_busy_mean\": %.1f, \"gpu_busy_max\": %d, \"frac_over_60\": %.3f}\n", n, s/n, m, o/n}' "$_RD/gpu_busy.log" | tee -a "$_RD/run.log"
echo "=== exit $R $(date +%T)" | tee -a "$_RD/run.log"
exit "$R"

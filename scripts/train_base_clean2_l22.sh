#!/bin/bash
export AMNI_HOME=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}
BIN=${BIN:-$AMNI_HOME/crates/target/release/adam-train}
export ARCH=${ARCH:-vulkan} VER=18 QT=1 LEVELS=21 HOPS=4 TERN=1 TERN_SCOPE=cell QLEV=5 CTX=256 BS=128 LR=3e-3 LR_END=3e-4 LOG2_ENTRIES=${LOG2_ENTRIES:-22} STEPS=${STEPS:-1600000} CORPUS=clean2
export SYNSETS=$AMNI_HOME/data/clean2/synsets.txt WORDCOUNTS=$AMNI_HOME/data/clean2/wordcounts.txt EVAL_EVERY=${EVAL_EVERY:-2000} CKPT_EVERY=${CKPT_EVERY:-50000}
export TAG=${TAG:-clean2_base_v18_L${LOG2_ENTRIES}_s$((STEPS/1000))k} THROTTLE_FILE=${THROTTLE_FILE:-$AMNI_HOME/throttle.txt}
LOG=$AMNI_HOME/exports/gf17_continuum/rayhash_$TAG.log
[ -f "$LOG" ] && cat "$LOG" >> "$LOG.prev"
exec "$BIN" > "$LOG" 2>&1

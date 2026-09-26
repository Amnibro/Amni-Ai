#!/usr/bin/env bash
R=${RUN_DIR:-/home/anmire/ai/Amni-Ai/exports/adam_t/t30m_bpe4096}
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file} AMNI_HOME=/home/anmire/ai/Amni-Ai RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=12G NICE=15
export D=512 L=10 H=8 FF=2048 CTX=512 TERN=1 MEM=1 NGRAMS=2,3,4,6 MEM_LOG2=15 BPE=/home/anmire/ai/Amni-Ai/data/clean2/bpe4096.json CORPUS=adam_t/stream
export BS=32 MICRO=${MICRO:-4} STEPS=${STEPS:-35000} WARM=1000 LR=2e-3 EVAL_EVERY=1000 EVAL_BATCHES=10 CKPT_EVERY=250 SEG=${SEG:-30} EVAL_SEGS=${EVAL_SEGS:-1} PARITY_WINDOWS=2
export TAG=$(basename $R) RUN_DIR=$R OUT=$R BIN=$R/bin/adam-t
exec bash $R/bin/campaign.sh

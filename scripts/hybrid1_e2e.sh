#!/bin/bash
MODE=$1; NAME=$2; BAKE=$3; OPTS=$4; LIMIT=${5:-0}
cd /home/anmire/ai/Amni-Ai/.claude/worktrees/agent-a5bd8c5f36a3c80c7
export AMNI_COGNITION=1 AMNI_ADAM_T_HARNESS_AGREE=1
if [ "$MODE" = "hybrid" ]; then export AMNI_ADAM_T_HARNESS=1 AMNI_BLENDER="atex:$BAKE" AMNI_BLENDER_OPTS="$OPTS"; fi
if [ "$MODE" = "alone" ]; then export AMNI_ADAM_T_HARNESS=0 AMNI_COGNITION=0 AMNI_BLENDER_NOMAT=1 AMNI_BLENDER="atex:$BAKE" AMNI_BLENDER_OPTS="$OPTS"; fi
exec systemd-run --user --scope -q -p MemoryMax=12G -p CPUQuota=400% nice -n 15 /home/anmire/ai/hybrid1/py scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --limit $LIMIT --name "$NAME" --step 0 --proto-cmd "/home/anmire/ai/Amni-Ai/crates/target/release/adam-t /home/anmire/ai/Amni-Ai/exports/adam_t/t30m_sft3/model.gf17"

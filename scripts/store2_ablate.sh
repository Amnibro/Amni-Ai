#!/bin/bash
cd "$(dirname "$0")/.."
PY=${PY:-/home/anmire/ai/Amni-Ai/.venv-linux/bin/python}
P=${P:-a}
run(){ n=$1;shift;systemd-run --user --scope -q -p MemoryMax=12G -p CPUQuota=100% nice -n 15 $PY scripts/store2_eval.py --split dev --v3 0 --cold 0 --name ${P}_$n "$@" 2>&1|tail -2|head -1|sed "s/^/$n /" >> data/store2/results/ablate_$P.log; }
for spec in "full" "mined_only --abl nohand" "hand_only --abl handonly" "no_wordnet --abl nownet" "templates --abl tmpl" "no_desc --abl nodesc" "store1_absence --abl oldabs" "no_fallback --abl nofb" "fallback_num --abl fbnum" "no_latest --abl latest" "no_prior --abl prior" "no_tgate --abl tgate" "no_cs --abl cs" "no_calc --abl calc" "fd3 --fd 3";do eval run $spec;done

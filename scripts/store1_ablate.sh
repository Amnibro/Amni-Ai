#!/bin/bash
set -u
cd "$(dirname "$0")/.."
P=${PY:-/home/anmire/ai/Amni-Ai/.venv-linux/bin/python}
export CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
for f in 3 2 1 0; do $P scripts/store1_eval.py --split dev --name abl_fd$f --fd $f --v3 0 --warm 1 | tail -1 | sed "s/^/fd$f /"; done
for a in prior tgate hand alias cs calc qual hint case; do $P scripts/store1_eval.py --split dev --name abl_no_$a --abl $a --v3 0 | tail -1 | sed "s/^/no_$a /"; done

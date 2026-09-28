#!/usr/bin/env bash
set -u
HM=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; AB=${AB_DIR:-$HM/exports/adam_t/lorentz_ab}; SRC=${BIN_SRC:-$HM/crates/target/release/adam-t}; PY=${PY:-$HM/.venv-linux/bin/python}; CD=$(cd "$(dirname "$0")" && pwd)
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$HM RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=12G NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
for u in ${BUSY_UNITS:-adam-t-30m adam-t-sft}; do systemctl --user is-active --quiet "$u" && { echo "unit $u active: one GPU job at a time, not starting"; exit 4; }; done
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
mkdir -p "$AB/bin" || exit 5
[ -x "$AB/bin/adam-t" ] || { cp "$SRC" "$CD/campaign.sh" "$CD/smoke.sh" "$CD/lorentz_ab.sh" "$AB/bin/" && sha256sum "$AB/bin/adam-t" > "$AB/bin/adam-t.sha256" && git -C "$HM" rev-parse HEAD > "$AB/bin/commit.txt"; } || exit 5
export D=256 L=8 H=4 CTX=256 FF=1024 TERN=1 MEM=1 NGRAMS=2,3,4,5,6,8 MEM_LOG2=15 BS=64 MICRO=1 STEPS=3000 LR=2e-3 WARM=200 WD=0.05 CORPUS=clean2 BPE= EVAL_EVERY=500 EVAL_BATCHES=20 CKPT_EVERY=250 PARITY_WINDOWS=8 SEG=${SEG:-20} EVAL=0 LPE_PHI=4 LPE_S=64 BIN=$AB/bin/adam-t
cd "$HM" || exit 5
for s in ${SEEDS:-0 1}; do for a in pos lpe; do
  R=$AB/${a}_s$s; [ -f "$R/model.gf17" ] && [ "$(cat "$R/ckpt/state.txt" 2>/dev/null)" = "$STEPS" ] && { echo "=== ab: $a seed $s done"; continue; }
  echo "=== ab: $a seed $s start $(date +%T)" | tee -a "$AB/ab.log"
  RUN_DIR=$R OUT=$R TAG=lorentz_ab_${a}_s$s SEED=$s LPE=$([ $a = lpe ] && echo 1 || echo 0) bash "$AB/bin/campaign.sh"; X=$?
  [ $X -eq 0 ] && [ -f "$R/model.gf17" ] || { echo "=== ab: $a seed $s stopped (exit $X), rerun to resume" | tee -a "$AB/ab.log"; exit $X; }
  [ -f "$R/STOP" ] && { echo "=== ab: STOP in $R, paused" | tee -a "$AB/ab.log"; exit 0; }
done; done
for s in ${SEEDS:-0 1}; do for a in pos lpe; do
  R=$AB/${a}_s$s; F=$HM/data/adam_t/eval/report_lorentz-ab-${a}-s${s}_${STEPS}_dry.json; [ -f "$F" ] && continue
  pgrep -x adam-t >/dev/null && { echo "adam-t running: eval waits, rerun later"; exit 4; }
  echo "=== ab eval $a seed $s $(date +%T)" | tee -a "$AB/ab.log"
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=${EVAL_MEM:-6G} -p CPUQuota=${EVAL_CPU:-300%} --setenv=RAYON_NUM_THREADS=3 --setenv=THROTTLE_FILE="$THROTTLE_FILE" "$PY" scripts/adam_t_eval.py run --model proto --proto-cmd "$BIN serve $R" --suites bpb --name lorentz-ab-$a-s$s --step "$STEPS" >> "$AB/eval.log" 2>&1 || { echo "=== ab eval $a seed $s failed, see $AB/eval.log"; exit 6; }
done; done
: > "$AB/lut_speed.jsonl"
for k in $(seq ${SPEED_REPS:-5}); do for r in pos_s0 lpe_s0 pos_s1 lpe_s1; do
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=300% --setenv=RAYON_NUM_THREADS=3 --setenv=LPE_LUT=gf "$BIN" lut "$AB/$r/model.gf17" ${SPEED_WIN:-16} | sed "s/^{/{\"run\": \"$r\", \"rep\": $k, /" >> "$AB/lut_speed.jsonl"
done; done
"$PY" - "$AB" "$HM/data/adam_t/eval" "$STEPS" <<'EOF' | tee "$AB/ab_result.json"
import json,sys,statistics as st;from pathlib import Path
ab,ev,n=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];A=("pos","lpe");S=(0,1)
m={(a,s):json.loads((ev/f"report_lorentz-ab-{a}-s{s}_{n}_dry.json").read_text())["m"] for a in A for s in S}
lv={(a,s):[json.loads(l) for l in open(ab/f"{a}_s{s}/log.jsonl") if '"val_bits_per_byte"' in l][-1]["val_bits_per_byte"] for a in A for s in S}
sp=[json.loads(l) for l in open(ab/"lut_speed.jsonl")];md={r:st.median(x["bytes_per_s"] for x in sp if x["run"]==r) for r in {x["run"] for x in sp}}
def cmp(k):v={(a,s):m[a,s][k] for a in A for s in S};mu={a:(v[a,0]+v[a,1])/2 for a in A};spr=max(abs(v[a,0]-v[a,1]) for a in A);return {"per_run":{f"{a}_s{s}":v[a,s] for a in A for s in S},"mean":mu,"seed_spread":spr,"lpe_gain":mu["pos"]-mu["lpe"],"lpe_wins":mu["pos"]-mu["lpe"]>spr}
c2,sh=cmp("bpb_clean2"),cmp("bpb_stream");aa=min(abs(md["pos_s0"]-md["pos_s1"])/md["pos_s0"],0.02);ratio=(md["lpe_s0"]+md["lpe_s1"])/(md["pos_s0"]+md["pos_s1"])
spd={"median_bytes_per_s":md,"lpe_over_pos":ratio,"aa_noise":aa,"lpe_not_slower":ratio>=1-aa}
print(json.dumps({"clean2":c2,"stream":sh,"trainer_val_bpb":{f"{a}_s{s}":lv[a,s] for a in A for s in S},"lut_speed":spd,"stream_guard":sh["lpe_gain"]>=-sh["seed_spread"],"adopt_lpe":bool(c2["lpe_wins"] and spd["lpe_not_slower"] and sh["lpe_gain"]>=-sh["seed_spread"])},indent=1))
EOF

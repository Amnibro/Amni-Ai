#!/usr/bin/env bash
set -u
HM=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; AB=${AB_DIR:-$HM/exports/adam_t/mem_ab}; SRC=${BIN_SRC:-$HM/crates/target/release/adam-t}; PY=${PY:-$HM/.venv-linux/bin/python}; CD=$(cd "$(dirname "$0")" && pwd); VR=/sys/class/drm/card1/device/mem_info_vram_used
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$HM RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=12G NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
export D=256 L=8 H=4 CTX=256 FF=1024 TERN=1 MEM=1 NGRAMS=2,3,4,5,6,8 MEM_OPT=lazy LPE=1 LPE_PHI=4 LPE_S=64 BS=64 MICRO=1 STEPS=3000 LR=2e-3 LR_MEM=2e-3 WARM=200 WD=0.05 CORPUS=clean2 BPE= BASE= INIT= SRC_W= EPOCHS=0 MODE=train EVAL_EVERY=500 EVAL_BATCHES=20 CKPT_EVERY=250 PARITY_WINDOWS=8 SEG=${SEG:-20} EVAL=0 BIN=$AB/bin/adam-t
[ "${DRY:-0}" = 1 ] && { for s in ${SEEDS:-0 1}; do for a in 15 18; do echo "=== dry: mem$a seed $s -> $AB/mem${a}_s$s"; done; done; env | grep -E "^(D|L|H|CTX|FF|TERN|MEM|NGRAMS|MEM_OPT|LPE|LPE_PHI|LPE_S|BS|MICRO|STEPS|LR|LR_MEM|WARM|WD|CORPUS|BPE|BASE|INIT|MODE|EVAL_EVERY|EVAL_BATCHES|CKPT_EVERY|PARITY_WINDOWS|SEG|EVAL|BIN|MEM_MAX|CPU_Q|NICE|RAYON_NUM_THREADS)=" | sort; echo "arm var: MEM_LOG2 15|18, SEED 0|1, EVAL_MEM 6G|8G"; exit 0; }
ME=$(basename "$(cut -d: -f3 /proc/$$/cgroup)")
for u in $(systemctl --user list-units --state=active --no-legend --plain 'adam-t-*' | awk '{print $1}'); do [ "$u" = "$ME" ] || { echo "unit $u active: one GPU job at a time, not starting"; exit 4; }; done
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
mkdir -p "$AB/bin" || exit 5
[ -x "$AB/bin/adam-t" ] || { cp "$SRC" "$CD/campaign.sh" "$CD/smoke.sh" "$CD/mem_ab.sh" "$AB/bin/" && sha256sum "$AB/bin/adam-t" > "$AB/bin/adam-t.sha256" && git -C "$HM" rev-parse HEAD > "$AB/bin/commit.txt"; } || exit 5
cd "$HM" || exit 5
for s in ${SEEDS:-0 1}; do for a in 15 18; do
  R=$AB/mem${a}_s$s; [ -f "$R/model.gf17" ] && [ "$(cat "$R/ckpt/state.txt" 2>/dev/null)" = "$STEPS" ] && { echo "=== ab: mem$a seed $s done"; continue; }
  echo "=== ab: mem$a seed $s start $(date +%T)" | tee -a "$AB/ab.log"; mkdir -p "$R"
  ( while :; do cat "$VR"; sleep 5; done ) >> "$R/vram.log" 2>/dev/null & VP=$!
  RUN_DIR=$R OUT=$R TAG=mem_ab_m${a}_s$s SEED=$s MEM_LOG2=$a bash "$AB/bin/campaign.sh"; X=$?; kill "$VP"
  [ $X -eq 0 ] && [ -f "$R/model.gf17" ] || { echo "=== ab: mem$a seed $s stopped (exit $X), rerun to resume" | tee -a "$AB/ab.log"; exit $X; }
  [ -f "$R/STOP" ] && { echo "=== ab: STOP in $R, paused" | tee -a "$AB/ab.log"; exit 0; }
done; done
for s in ${SEEDS:-0 1}; do for a in 15 18; do
  R=$AB/mem${a}_s$s; F=$HM/data/adam_t/eval/report_mem-ab-m${a}-s${s}_${STEPS}_dry.json; [ -f "$F" ] && continue
  pgrep -x adam-t >/dev/null && { echo "adam-t running: eval waits, rerun later"; exit 4; }
  echo "=== ab eval mem$a seed $s $(date +%T)" | tee -a "$AB/ab.log"
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=$([ $a = 18 ] && echo 8G || echo 6G) -p CPUQuota=300% --setenv=RAYON_NUM_THREADS=3 --setenv=THROTTLE_FILE="$THROTTLE_FILE" "$PY" scripts/adam_t_eval.py run --model proto --proto-cmd "$BIN serve $R" --suites bpb --name mem-ab-m$a-s$s --step "$STEPS" >> "$AB/eval.log" 2>&1 || { echo "=== ab eval mem$a seed $s failed, see $AB/eval.log"; exit 6; }
done; done
: > "$AB/lut_speed.jsonl"
for k in $(seq ${SPEED_REPS:-5}); do for r in mem15_s0 mem18_s0 mem15_s1 mem18_s1; do
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=300% --setenv=RAYON_NUM_THREADS=3 --setenv=LPE_LUT=gf --setenv=CORPUS=clean2 "$BIN" lutbench "$AB/$r/model.gf17" 128 128 | sed "s/^{/{\"run\": \"$r\", \"rep\": $k, /" >> "$AB/lut_speed.jsonl"
done; done
"$PY" - "$AB" "$HM/data/adam_t/eval" "$STEPS" <<'EOF' | tee "$AB/ab_result.json"
import json,sys,statistics as st;from pathlib import Path
ab,ev,n=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];A=("15","18");S=(0,1)
m={(a,s):json.loads((ev/f"report_mem-ab-m{a}-s{s}_{n}_dry.json").read_text())["m"] for a in A for s in S}
lg={(a,s):[json.loads(l) for l in open(ab/f"mem{a}_s{s}/log.jsonl") if '"val_bits_per_byte"' in l] for a in A for s in S}
vr={(a,s):max([int(x) for x in open(ab/f"mem{a}_s{s}/vram.log") if x.strip().isdigit()] or [0])/2**30 for a in A for s in S}
sp=[json.loads(l) for l in open(ab/"lut_speed.jsonl")];md={r:st.median(x["step_ms_p50"] for x in sp if x["run"]==r) for r in {x["run"] for x in sp}}
def cmp(k):v={(a,s):m[a,s][k] for a in A for s in S};mu={a:(v[a,0]+v[a,1])/2 for a in A};spr=max(abs(v[a,0]-v[a,1]) for a in A);return {"per_run":{f"mem{a}_s{s}":v[a,s] for a in A for s in S},"mean":mu,"seed_spread":spr,"mem18_gain":mu["15"]-mu["18"],"mem18_wins":mu["15"]-mu["18"]>spr}
c2,sh=cmp("bpb_clean2"),cmp("bpb_stream");ratio=(md["mem18_s0"]+md["mem18_s1"])/(md["mem15_s0"]+md["mem15_s1"])
spd={"median_decode_ms_p50":md,"mem18_over_mem15":ratio,"within_1_2x":ratio<=1.2}
print(json.dumps({"clean2":c2,"stream":sh,"trainer_val_bpb":{f"mem{a}_s{s}":lg[a,s][-1]["val_bits_per_byte"] for a in A for s in S},"gpu_ms_per_step":{f"mem{a}_s{s}":lg[a,s][-1]["gpu_ms_per_step"] for a in A for s in S},"peak_vram_gib_total_card":{f"mem{a}_s{s}":round(vr[a,s],2) for a in A for s in S},"lut_decode":spd,"stream_guard":sh["mem18_gain"]>=-sh["seed_spread"],"adopt_mem18":bool(c2["mem18_wins"] and spd["within_1_2x"] and sh["mem18_gain"]>=-sh["seed_spread"])},indent=1))
EOF

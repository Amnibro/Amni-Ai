#!/usr/bin/env bash
set -u
HM=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; AB=${AB_DIR:-$HM/exports/adam_t/bind_ab}; SRC=${BIN_SRC:-$HM/crates/target/release/adam-t}; PY=${PY:-$HM/.venv-linux/bin/python}; CD=$(cd "$(dirname "$0")" && pwd); VR=/sys/class/drm/card1/device/mem_info_vram_used; RM=0.04; ARMS="both base copy recall"
export THROTTLE_FILE=${THROTTLE_FILE:?set THROTTLE_FILE to the GPU governor file, ~/.local/share/amni-ascend/throttle.txt} AMNI_HOME=$HM RAYON_NUM_THREADS=3 CPU_Q=300% MEM_MAX=12G NICE=15
[ -f "$THROTTLE_FILE" ] || { echo "no throttle file $THROTTLE_FILE"; exit 3; }
export D=256 L=8 H=4 CTX=256 FF=1024 TERN=1 MEM=1 NGRAMS=2,3,4,6 MEM_LOG2=15 MEM_OPT=lazy LPE=1 LPE_PHI=4 LPE_S=64 BPE=$HM/data/clean2/bpe4096.json CORPUS=adam_t/stream BS=64 MICRO=${MICRO:-4} STEPS=3000 LR=2e-3 LR_MEM=2e-3 WARM=200 WD=0.05 BASE= INIT= SRC_W= EPOCHS=0 MODE=train EVAL_EVERY=500 EVAL_BATCHES=20 CKPT_EVERY=250 PARITY_WINDOWS=8 SEG=${SEG:-20} EVAL=0 BIN=$AB/bin/adam-t
av() { case $1 in base) echo "0 0";; recall) echo "0 $RM";; copy) echo "1 0";; both) echo "1 $RM";; esac; }
[ "${DRY:-0}" = 1 ] && { for s in ${SEEDS:-0 1}; do for a in $ARMS; do set -- $(av $a); echo "=== dry: $a seed $s -> $AB/${a}_s$s  COPY=$1 RECALL_MIX=$2"; done; done; env | grep -E "^(D|L|H|CTX|FF|TERN|MEM|NGRAMS|MEM_LOG2|MEM_OPT|LPE|LPE_PHI|LPE_S|BPE|CORPUS|BS|MICRO|STEPS|LR|LR_MEM|WARM|WD|BASE|INIT|MODE|EVAL_EVERY|EVAL_BATCHES|CKPT_EVERY|PARITY_WINDOWS|SEG|EVAL|BIN|MEM_MAX|CPU_Q|NICE|RAYON_NUM_THREADS)=" | sort; echo "arm vars: COPY 0|1, RECALL_MIX 0|$RM, SEED 0|1"; exit 0; }
ME=$(basename "$(cut -d: -f3 /proc/$$/cgroup)")
for u in $(systemctl --user list-units --state=active --no-legend --plain 'adam-t-*' | awk '{print $1}'); do [ "$u" = "$ME" ] || { echo "unit $u active: one GPU job at a time, not starting"; exit 4; }; done
pgrep -x adam-t >/dev/null && { echo "adam-t running: one GPU job at a time, not starting"; exit 4; }
mkdir -p "$AB/bin" "$AB/probe" || exit 5
[ -x "$AB/bin/adam-t" ] || { cp "$SRC" "$CD/campaign.sh" "$CD/smoke.sh" "$CD/bind_ab.sh" "$AB/bin/" && cp "$SRC" "$AB/bin/adamt-probe" && sha256sum "$AB/bin/adam-t" > "$AB/bin/adam-t.sha256" && git -C "$HM" rev-parse HEAD > "$AB/bin/commit.txt"; } || exit 5
cd "$HM" || exit 5
[ -f "$AB/recall_decontam.json" ] || { "$AB/bin/adamt-probe" recall "$AB/recall_sample.txt" 20000 > "$AB/recall_stats.json" && nice -n 15 systemd-run --user --scope -q -p MemoryMax=8G -p CPUQuota=300% "$PY" - "$AB/recall_sample.txt" > "$AB/recall_decontam.json" <<'EOF'
import sys,json;from pathlib import Path;from amni.compute import decontam as DC
u=[x for x in Path(sys.argv[1]).read_text().split('\n\n') if x.strip()];a=Path('data/adam_t');x=sorted(p.resolve() for p in a.glob('eval/**/*.json*') if p.is_file() and not DC.SKIP.search(p.resolve().relative_to(DC.ROOT).as_posix()))+[(a/'probe'/'probes.jsonl').resolve()]
r={nm:sum(1 for t in u if DC.dirty(t,None,ix)) for nm,ix in (('eval_sources_13',DC.index()),('adam_t_eval_probe_8',DC.index(n=8,files=x,cache=None)),('adam_t_eval_probe_13',DC.index(n=13,files=x,cache=None)))}
print(json.dumps({'units':len(u),'dirty':r,'clean':not any(r.values())}))
EOF
} || { echo "recall decontam check failed"; rm -f "$AB/recall_decontam.json"; exit 7; }
grep -q '"clean": true' "$AB/recall_decontam.json" || { echo "recall generator is not clean: $(cat "$AB/recall_decontam.json")"; exit 7; }
for s in ${SEEDS:-0 1}; do for a in $ARMS; do
  R=$AB/${a}_s$s; [ -f "$R/model.gf17" ] && [ "$(cat "$R/ckpt/state.txt" 2>/dev/null)" = "$STEPS" ] && { echo "=== ab: $a seed $s done"; continue; }
  set -- $(av $a); echo "=== ab: $a seed $s start $(date +%T) COPY=$1 RECALL_MIX=$2" | tee -a "$AB/ab.log"; mkdir -p "$R"
  ( while :; do cat "$VR"; sleep 5; done ) >> "$R/vram.log" 2>/dev/null & VP=$!
  RUN_DIR=$R OUT=$R TAG=bind_ab_${a}_s$s SEED=$s COPY=$1 RECALL_MIX=$2 bash "$AB/bin/campaign.sh"; X=$?; kill "$VP"
  [ $X -eq 0 ] && [ -f "$R/model.gf17" ] || { echo "=== ab: $a seed $s stopped (exit $X), rerun to resume" | tee -a "$AB/ab.log"; exit $X; }
  [ -f "$R/STOP" ] && { echo "=== ab: STOP in $R, paused" | tee -a "$AB/ab.log"; exit 0; }
done; done
for s in ${SEEDS:-0 1}; do for a in $ARMS; do
  R=$AB/${a}_s$s; F=$HM/data/adam_t/eval/report_bind-ab-${a}-s${s}_${STEPS}_dry.json; [ -f "$F" ] && continue
  pgrep -x adam-t >/dev/null && { echo "adam-t running: eval waits, rerun later"; exit 4; }
  echo "=== ab eval $a seed $s $(date +%T)" | tee -a "$AB/ab.log"
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=6G -p CPUQuota=300% --setenv=RAYON_NUM_THREADS=3 --setenv=THROTTLE_FILE="$THROTTLE_FILE" "$PY" scripts/adam_t_eval.py run --model proto --proto-cmd "$BIN serve $R" --suites bpb --name bind-ab-$a-s$s --step "$STEPS" >> "$AB/eval.log" 2>&1 || { echo "=== ab eval $a seed $s failed, see $AB/eval.log"; exit 6; }
done; done
for s in ${SEEDS:-0 1}; do for a in $ARMS; do
  R=$AB/${a}_s$s; P=$AB/probe/${a}_s$s
  for k in binding induct; do [ -f "$P.$k.json" ] && continue; echo "=== ab probe $k $a seed $s $(date +%T)" | tee -a "$AB/ab.log"
    nice -n 15 systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=300% --setenv=ADAMT_BIN="$AB/bin/adamt-probe" --setenv=AMNI_HOME="$HM" "$PY" scripts/adam_t_probe.py $k $([ $k = binding ] && echo "--model ${a}_s$s=$R/model.gf17 --room 16 --res $P.binding.jsonl --summary $P.binding.json" || echo "--models ${a}_s$s=$R/model.gf17 --lens 10,50,100 --out $P.induct.json") >> "$AB/probe.log" 2>&1 || { echo "=== ab probe $k $a seed $s failed, see $AB/probe.log"; exit 8; }
  done
done; done
: > "$AB/lut_speed.jsonl"
for k in $(seq ${SPEED_REPS:-5}); do for s in ${SEEDS:-0 1}; do for a in $ARMS; do
  nice -n 15 systemd-run --user --scope -q -p MemoryMax=4G -p CPUQuota=300% --setenv=RAYON_NUM_THREADS=3 --setenv=LPE_LUT=gf --setenv=CORPUS=adam_t/stream "$AB/bin/adamt-probe" lutbench "$AB/${a}_s$s/model.gf17" 128 128 | sed "s/^{/{\"run\": \"${a}_s$s\", \"rep\": $k, /" >> "$AB/lut_speed.jsonl"
done; done; done
"$PY" - "$AB" "$HM/data/adam_t/eval" "$STEPS" <<'EOF' | tee "$AB/ab_result.json"
import json,sys,statistics as st;from pathlib import Path
ab,ev,n=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3];A=("base","recall","copy","both");X=A[1:];S=(0,1);MB=0.05;SP=1.10
m={(a,s):json.loads((ev/f"report_bind-ab-{a}-s{s}_{n}_dry.json").read_text())["m"] for a in A for s in S}
pb={(a,s):json.loads((ab/"probe"/f"{a}_s{s}.binding.json").read_text()) for a in A for s in S};pi={(a,s):json.loads((ab/"probe"/f"{a}_s{s}.induct.json").read_text())[f"{a}_s{s}"] for a in A for s in S}
lg={(a,s):[json.loads(l) for l in open(ab/f"{a}_s{s}/log.jsonl") if '"val_bits_per_byte"' in l] for a in A for s in S}
vr={(a,s):max([int(x) for x in open(ab/f"{a}_s{s}/vram.log") if x.strip().isdigit()] or [0])/2**30 for a in A for s in S}
sp=[json.loads(l) for l in open(ab/"lut_speed.jsonl")];md={r:st.median(x["step_ms_p50"] for x in sp if x["run"]==r) for r in {x["run"] for x in sp}}
v={"binding_mc":{k:pb[k]["mc"] for k in pb},"induct":{k:pi[k]["mean_top1"] for k in pi},"bpb_clean2":{k:m[k]["bpb_clean2"] for k in m},"bpb_stream":{k:m[k]["bpb_stream"] for k in m}}
mu=lambda q,a:(v[q][a,0]+v[q][a,1])/2;spr=lambda q,a:max(abs(v[q][a,0]-v[q][a,1]),abs(v[q]["base",0]-v[q]["base",1]))
def arm(a):
 g={q:mu(q,a)-mu(q,"base") for q in v};sd={q:spr(q,a) for q in v};ratio=(md[f"{a}_s0"]+md[f"{a}_s1"])/(md["base_s0"]+md["base_s1"])
 win={q:g[q]>max(sd[q],MB) for q in ("binding_mc","induct")};lose={q:g[q]<-sd[q] for q in ("binding_mc","induct")};guard={q:g[q]<=sd[q] for q in ("bpb_clean2","bpb_stream")}
 return {"mean":{q:mu(q,a) for q in v},"gain_vs_base":g,"seed_spread":sd,"primary_win":win,"primary_loss":lose,"bpb_guard_ok":guard,"lut_decode_ratio":ratio,"speed_ok":ratio<=SP,"adopt":bool((win["binding_mc"] or win["induct"]) and not any(lose.values()) and all(guard.values()) and ratio<=SP)}
R={a:arm(a) for a in X};C=[a for a in X if R[a]["adopt"]];best=max(C,key=lambda a:mu("binding_mc",a)) if C else None
pick=next((a for a in C if mu("binding_mc",a)>=mu("binding_mc",best)-max(spr("binding_mc",a),abs(v["binding_mc"][best,0]-v["binding_mc"][best,1]))),None) if C else None
env={"base":"COPY=0 RECALL_MIX=0","recall":"COPY=0 RECALL_MIX=0.04","copy":"COPY=1 RECALL_MIX=0","both":"COPY=1 RECALL_MIX=0.04"}
print(json.dumps({"per_run":{q:{f"{a}_s{s}":v[q][a,s] for a in A for s in S} for q in v},"base_mean":{q:mu(q,"base") for q in v},"arms":R,"binding_detail":{f"{a}_s{s}":{k:pb[a,s][k] for k in ("chance","lift","tf_top1","p1_frame","truncated","facts_cut","by_type","by_k")} for a in A for s in S},"induct_detail":{f"{a}_s{s}":pi[a,s] for a in A for s in S},"trainer_val_bpb":{f"{a}_s{s}":lg[a,s][-1]["val_bits_per_byte"] for a in A for s in S},"gpu_ms_per_step":{f"{a}_s{s}":lg[a,s][-1]["gpu_ms_per_step"] for a in A for s in S},"peak_vram_gib_total_card":{f"{a}_s{s}":round(vr[a,s],2) for a in A for s in S},"median_decode_ms_p50":md,"adopted":C,"v2_pick":pick or "base","v2_env":env[pick or "base"]},indent=1))
EOF

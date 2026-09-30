#!/usr/bin/env bash
set -u
H=${AMNI_HOME:-/home/anmire/ai/Amni-Ai}; R=${RUN_DIR:?set RUN_DIR}; PY=${PY:-$H/.venv-linux/bin/python}; TG=${TAG3:-adam-t-t30m-sft3}; B2=${CMP_GF17:-$H/exports/adam_t/t30m_sft2/model.gf17}; S2=${CMP_STEP:-7773}; N2=${CMP_NAME:-adam-t-t30m-sft2}; E2=${CMP_E2E:-$H/data/adam_t/eval/report_adam-t-t30m-sft2-final-h0_7773_e2e_tune_dry.json}; EV=$H/data/adam_t/eval
L(){ echo "=== chain $(date '+%F %T') $*" | tee -a "$R/chain.log"; }
X(){ systemd-run --user --scope -q -p MemoryMax=${CH_MEM:-8G} -p CPUQuota=${CH_CPU:-300%} --setenv=RAYON_NUM_THREADS=3 --setenv=AMNI_HOME="$H" nice -n 15 "$@" >> "$R/chain.log" 2>&1; }
[ -f "$R/report.md" ] && { L "report exists, nothing to do"; exit 0; }
[ -f "$R/model.gf17" ] || { bash "$R/bin/sft.sh"; r=$?; [ $r -eq 0 ] || { L "sft.sh exit $r"; exit $r; }; }
[ -f "$R/STOP" ] && { L "STOP file present: training paused, chain not run (rm STOP and restart the unit to resume)"; exit 0; }
[ -f "$R/model.gf17" ] || { L "no model.gf17: training not finished, chain not run"; exit 0; }
cd "$H" || exit 5; mkdir -p "$R/sel"; cp -f "$R/bin/adam-t" "$R/sel/adamt-sel"; cp -f "$R/bin/adam-t" "$R/sel/adamt-e2e"; P8="$PY $R/bin/adam_t_probe8.py"
for d in "$R"/snap_*; do
 [ -d "$d" ] || continue; s=${d##*_}; o=$R/sel/$s; g=$o/eval/step$s/model.gf17
 [ -f "$g" ] || { mkdir -p "$o"; head -n 1 "$R/log.jsonl" > "$o/log.jsonl"; rm -rf "$o/ckpt"; cp -a "$d" "$o/ckpt"; L "export snapshot $s"; X "$R/sel/adamt-sel" export "$o"; rm -rf "$o/ckpt"; }
 [ -f "$R/sel/probe_$s.json" ] || { L "probe snapshot $s"; X $P8 run --proto-cmd "$R/sel/adamt-sel serve $g" --name "$TG" --step "$s" --out "$R/sel/probe_$s.json"; }
done
BS=$($PY "$R/bin/adam_t_probe8.py" select --run "$R" --name "$TG" 2>>"$R/chain.log" | tail -n 1); [ -n "$BS" ] || { L "selection failed"; exit 6; }; G=$R/sel/$BS/eval/step$BS/model.gf17; ln -sfn "$G" "$R/best.gf17"; L "selected step $BS ($G)"
[ -f "$EV/report_${TG}-best_${BS}_chat.json" ] || { L "A0 chat suite round 3 step $BS"; X $PY scripts/adam_t_eval.py run --model proto --suites chat --proto-cmd "$R/sel/adamt-sel serve $G" --step "$BS" --name "$TG-best"; }
[ -f "$EV/report_${N2}-gf17_${S2}_chat.json" ] || { L "A0 chat suite round 2 step $S2"; X $PY scripts/adam_t_eval.py run --model proto --suites chat --proto-cmd "$R/sel/adamt-sel serve $B2" --step "$S2" --name "$N2-gf17"; }
[ -f "$EV/report_${TG}-final-h0_${BS}_e2e_tune_dry.json" ] || { L "E2E harness off round 3 step $BS"; X env AMNI_ADAM_T_HARNESS=0 AMNI_ADAM_T_HARNESS_AGREE=1 AMNI_COGNITION=0 $PY scripts/adam_t_eval.py run --model proto --suites chat --e2e --tune --proto-cmd "$R/sel/adamt-e2e serve $G" --step "$BS" --name "$TG-final-h0"; }
[ -f "$R/sel/probe_r2.json" ] || { L "probes round 2"; X $P8 run --proto-cmd "$R/sel/adamt-sel serve $B2" --name "$N2" --step "$S2" --out "$R/sel/probe_r2.json"; }
pick(){ ls -t "$EV"/report_$1*.json 2>/dev/null | head -n 1; }
X $P8 report --run "$R" --a0-r3 "$(pick "${TG}-best_${BS}_chat")" --a0-r2 "$(pick "${N2}-gf17_${S2}_chat")" --e2e-r3 "$(pick "${TG}-final-h0_${BS}_e2e")" --e2e-r2 "$E2" --probe-r3 "$R/sel/probe_$BS.json" --probe-r2 "$R/sel/probe_r2.json" --name3 "$TG" --step3 "$BS" --why "$(tr -d '\n' < "$R/sel/select.json" | cut -c1-400)" --out "$R/report.md"
for d in "$R"/sel/[0-9]*; do [ "${d##*/}" = "$BS" ] || rm -f "$d"/eval/step*/model.gf17; done
L "chain done: $R/report.md"

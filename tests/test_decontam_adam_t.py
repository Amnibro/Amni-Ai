import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from amni.compute import decontam as D
def test_eval_sources_include_adam_t_evals():
 fs={D.rel(f) for f in D.eval_sources()};at={f for f in fs if f.startswith("data/adam_t/eval/")};need=[p for p in ("data/adam_t/eval/nq_mc.json","data/adam_t/eval/rc_dev.json","data/adam_t/eval/d3dev.jsonl","data/adam_t/eval/bench/arc_easy_500.jsonl","data/adam_t/eval/bench/piqa_500.jsonl","data/adam_t/eval/chat/chat_src.jsonl","data/adam_t/eval/chat/chat_dev.jsonl","data/adam_t/eval/chat/chat_test.jsonl","data/adam_t/eval/chat/chat_human40.json") if (ROOT/p).exists()]
 assert set(need)<=at and not any("prereg_lock" in f or "report" in f or f.endswith("meta.json") for f in at) and {D.group(ROOT/f) for f in at}<={"adam_t"}

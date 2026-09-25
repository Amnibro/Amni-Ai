import json, math, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
home = os.environ.get("AMNI_HOME", "/home/anmire/ai/Amni-Ai")
hold = np.memmap(f"{home}/data/clean2/hold.bin", dtype=np.uint8, mode="r")
T, NB = 256, int(os.environ.get("REF_BATCHES", "5"))
g = np.random.default_rng(1234)
ix = np.concatenate([g.integers(0, len(hold) - T - 1, 128) for _ in range(NB)])
torch.set_num_threads(int(os.environ.get("REF_THREADS", "8")))
out = []
for name in sys.argv[1:]:
    tok, m = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    nats, nbytes, t0 = 0.0, 0, time.time()
    for i in ix:
        b = bytes(hold[i:i + T + 1])
        s = b.decode("utf-8", errors="ignore")
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        with torch.no_grad():
            lg = m(torch.tensor([ids])).logits[0, :-1].double().log_softmax(-1)
        nats += -lg.gather(1, torch.tensor(ids[1:])[:, None]).sum().item()
        nbytes += len(s.encode("utf-8")) - len(tok.decode(ids[:1]).encode("utf-8"))
    r = {"model": name, "windows": int(len(ix)), "scored_bytes": nbytes, "bits_per_byte": round(nats / nbytes / math.log(2), 4), "seconds": round(time.time() - t0, 1), "note": "eval reference only; 256-byte cold windows of data/clean2/hold.bin (numpy PCG64 seed 1234, bs 128, same sampler as adam-train val); first token unscored"}
    print(json.dumps(r), flush=True)
    out.append(r)

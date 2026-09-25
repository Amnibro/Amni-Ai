import json, os, sys, numpy as np
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers
H, V = os.environ.get("AMNI_HOME", "/home/anmire/ai/Amni-Ai"), int(sys.argv[1]) if len(sys.argv) > 1 else 4096
PAT = r" ?\p{L}+| ?\p{N}{1,3}| ?[^\s\p{L}\p{N}]+|\s+"
tr, g = np.memmap(f"{H}/data/clean2/train.bin", dtype=np.uint8, mode="r"), np.random.default_rng(17)
txt = [bytes(tr[i:i + 65536]).decode("utf-8", errors="ignore") for i in g.integers(0, len(tr) - 65536, int(os.environ.get("BPE_CHUNKS", "3000")))]
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), behavior="isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
t.train_from_iterator(txt, trainers.BpeTrainer(vocab_size=V, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
cs, n = bs[:], 0
for b in range(256):
    if b not in bs: bs.append(b); cs.append(256 + n); n += 1
u2b = {chr(c): b for b, c in zip(bs, cs)}
tb = lambda s: bytes(u2b[ch] for ch in s)
vocab, ids = [bytes([b]) for b in range(256)], {bytes([b]): b for b in range(256)}
merges = []
for m in json.loads(t.to_str())["model"]["merges"]:
    a, b = (m.split(" ") if isinstance(m, str) else m)
    ba, bb = tb(a), tb(b)
    ids[ba + bb] = len(vocab); vocab.append(ba + bb); merges.append([ids[ba], ids[bb]])
out = f"{H}/data/clean2/bpe{len(vocab)}.json"
json.dump({"pattern": PAT, "vocab_size": len(vocab), "merges": merges, "train_sample_bytes": sum(len(s.encode()) for s in txt), "source": "data/clean2/train.bin only, seed 17, HF tokenizers BpeTrainer as code; homegrown merges"}, open(out, "w"))
print(json.dumps({"out": out, "vocab": len(vocab), "merges": len(merges)}))

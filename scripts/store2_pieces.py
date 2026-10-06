import json,time,hashlib,argparse,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store2"/"pieces";NV=83521;NB=512
def corpus(a):
 OUT.mkdir(parents=True,exist_ok=True);p=OUT/"corpus.txt";h=hashlib.sha256();n=0
 lo=np.load(D/"e_lo.npy");lb=np.memmap(D/"e_lab.bin",np.uint8,"r");no=np.load(D/"nm_o.npy");nm=np.memmap(D/"nm.bin",np.uint8,"r")
 with open(p,"w",encoding="utf-8") as f:
  for i in range(0,len(lo)-1,1<<20):
   j=min(len(lo)-1,i+(1<<20));O=(lo[i:j+1]-lo[i]).tolist();B=bytes(lb[lo[i]:lo[j]])
   s="".join(B[O[k]:O[k+1]].decode("utf-8","replace").lower().replace("\n"," ")+"\n" for k in range(j-i) if O[k+1]>O[k]);f.write(s);h.update(s.encode());n+=s.count("\n")
   t=bytes(nm[no[i]:no[j]]).decode("utf-8","replace");t=t if t.endswith("\n") or not t else t+"\n";f.write(t);h.update(t.encode());n+=t.count("\n")
 (OUT/"corpus.json").write_text(json.dumps({"lines":n,"sha256":h.hexdigest(),"sources":["v3 e_lab.bin labels, str.lower()","v3 nm.bin normalized names (amni.compute.wikifact.nrm)"],"license":"CC0 1.0 (Wikidata)"},indent=1));print(n,h.hexdigest())
def train(a):
 from tokenizers import Tokenizer,models,trainers,pre_tokenizers
 cfg={"model":"BPE","vocab_size":NV-NB,"end_of_word_suffix":"</w>","pre_tokenizer":"WhitespaceSplit","limit_alphabet":a.alpha,"min_frequency":2,"special_tokens":[],"byte_fallback_ids":"0-255 mid-word bytes, 256-511 word-final bytes (own encoder)","piece_id_offset":NB,"total_ids":NV,"library":"tokenizers "+__import__("tokenizers").__version__}
 tk=Tokenizer(models.BPE(end_of_word_suffix="</w>"));tk.pre_tokenizer=pre_tokenizers.WhitespaceSplit();tr=trainers.BpeTrainer(vocab_size=NV-NB,min_frequency=2,limit_alphabet=a.alpha,end_of_word_suffix="</w>",show_progress=False)
 t0=time.time();tk.train([str(OUT/"corpus.txt")],trainer=tr);J=json.loads(tk.to_str());V=J["model"]["vocab"];M=J["model"]["merges"]
 (OUT/"vocab.json").write_text(json.dumps(V,ensure_ascii=False));(OUT/"merges.txt").write_text("".join((" ".join(m) if isinstance(m,list) else m)+"\n" for m in M),encoding="utf-8")
 cj=json.loads((OUT/"corpus.json").read_text());(OUT/"trainer.json").write_text(json.dumps({**cfg,"corpus":cj,"vocab":len(V),"merges":len(M),"train_s":round(time.time()-t0,1),"vocab_sha256":hashlib.sha256((OUT/"vocab.json").read_bytes()).hexdigest(),"merges_sha256":hashlib.sha256((OUT/"merges.txt").read_bytes()).hexdigest()},indent=1,ensure_ascii=False));print(len(V),len(M),round(time.time()-t0,1))
def export(a):
 V=json.loads((OUT/"vocab.json").read_text());T=sorted(V,key=V.get);assert [V[t] for t in T]==list(range(len(T))) and all(not any(c.isspace() for c in t) for t in T)
 (OUT/"pieces.txt").write_text("".join(t+"\n" for t in T),encoding="utf-8");print(len(T))
if __name__=="__main__":
 ap=argparse.ArgumentParser();ap.add_argument("cmd");ap.add_argument("--alpha",type=int,default=1000);a=ap.parse_args();{"corpus":corpus,"train":train,"export":export}[a.cmd](a)

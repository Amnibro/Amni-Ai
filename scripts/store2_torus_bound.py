import json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];T=ROOT/"data"/"torus"/"a2b";R=ROOT/"data"/"store2"/"results"
z=np.load(T/"a2b_data.npz",allow_pickle=True);rels=[str(x) for x in z["rels"]];r=z["r"];F=json.loads((R/"fstat_k8_nocyc.json").read_text());pp={p["p"]:p for p in F["per_prop"]}
M=json.loads((ROOT/"data"/"wikifact"/"v3"/"store"/"manifest.json").read_text())["prop_facts"];hit=json.loads((T/"res_test_final.json").read_text())
o={"torus_layer_bytes":(T/"layer2_test_final.gf17l2p").stat().st_size,"relations":{}};tot=0.0
for i,n in enumerate(rels):
 if not n.startswith("wd:"):continue
 p=int(n[4:]);c=int((r==i).sum());b=pp.get(p,{}).get("bits_per_fact");sv=c*max(0.0,(b or 0)-1.0)/8;tot+=sv
 o["relations"][n]={"torus_facts":c,"store_facts":M.get(f"P{p}"),"coverage":round(c/M.get(f"P{p}",1),4),"store2_bits_per_fact":b,"upper_bound_saving_bytes_if_every_covered_fact_cost_1_bit":round(sv)}
o["upper_bound_saving_bytes"]=round(tot);o["kept"]=tot>o["torus_layer_bytes"];o["note"]="Upper bound: every torus-covered fact of the relation coded at 1 bit instead of its measured STORE-2 cost (fstat_k8_nocyc.json per_prop). The torus layer itself would have to ship (torus_layer_bytes). Test hits@10 from res_test_final.json for reference."
o["torus_test_hits10"]={k:v.get("hits10") for k,v in hit.get("per_rel",{}).items()} if isinstance(hit,dict) else {}
(R/"torus_bound.json").write_text(json.dumps(o,indent=1));print(json.dumps(o,indent=1)[:1500])

import os,re,sys,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DR=ROOT/"data"/"store1"/"drive";V3=ROOT/"data"/"wikifact"/"v3"/"store";OUT=ROOT/"data"/"store1"/"results"/"sizes.json"
BYTESECS={"ta","ta_x","tb","tb_x","tc","tc_x"}
def secs():
 b=open(DR/"store1.gf17pg","rb").read(4096);nd,nb,do,dl=struct.unpack("<4Q",b[8:40]);f=open(DR/"store1.gf17pg","rb");f.seek(do);d=f.read(dl).decode()
 return {n:(int(o),int(l)) for n,o,l in (x.split("\t") for x in d.splitlines())},nd,nb
def main(fd:int):
 S,nd,nb=secs();cont=os.path.getsize(DR/"store1.gf17pg");sb={n:(l*17/16 if n in BYTESECS else l*17/32) for n,(o,l) in S.items()};tot_s=sum(sb.values());sb={n:v*(cont/tot_s) for n,v in sb.items()}
 F={p.name:p.stat().st_size for p in DR.iterdir() if p.suffix in (".gf17rib",".l1") or p.name.endswith(".gf17rib.l1")}
 grp=lambda pre:sum(v for k,v in F.items() if re.match(pre,k))
 C={"fact_levels":grp(r"fact_\d+\.gf17rib"),"fact_codebooks":sb["fact_hc"]+sb["fact_cb"]+sb["fact_cbx"],"qual_levels":grp(r"qual_\d+\.gf17rib"),"qual_codebooks":sb["qual_hc"]+sb["qual_cb"]+sb["qual_cbx"],"presence_sig_levels":grp(r"sig_\d+\.gf17rib"),"presence_sig_codebook":sb["sig_hc"]+sb["sig_cb"]+sb["sig_cbx"],"multi_counts":grp(r"cnt_\d+\.gf17rib")+sb["cnt_hc"]+sb["cnt_cb"]+sb["cnt_cbx"],"string_ids":grp(r"sid\.gf17rib"),"sitelinks":grp(r"sl_\d+\.gf17rib")+sb["sl_hc"]+sb["sl_cb"]+sb["sl_cbx"],"qid_map":sb["qid"]+sb["qidx"],"meta":sb["meta"],"text_model":sb["tm"],"labels_titles":sb["ta"]+sb["ta_x"],"aliases":sb["tb"]+sb["tb_x"],"strings":sb["tc"]+sb["tc_x"],f"name_table_f{fd}":grp(rf"name_f{fd}(_end)?\.gf17rib")}
 C={k:int(round(v)) for k,v in C.items()};tot=sum(C.values());v3=sum(p.stat().st_size for p in V3.iterdir());B=json.loads((DR/"build.json").read_text())
 nf=B["facts"];fb=C["fact_levels"]+C["fact_codebooks"]
 r={"fd":fd,"components":C,"total_bytes":tot,"v3_bytes":v3,"ratio_vs_v3":round(tot/v3,5),"gate_GB2_max_ratio":0.15,"gate_GB2_pass":tot/v3<=0.15,"container_bytes":cont,"container_pages_4096sq":B["container_pages"],"other_name_tables_bytes":{k:v for k,v in B["components"].items() if k.startswith("name_f")},"facts":nf,"fact_bits_per_fact_incl_codebooks":round(fb*8/nf,3),"fact_levels_bits_per_fact":round(C["fact_levels"]*8/nf,3),"all_fact_side_bits_per_fact":round((fb+C["qual_levels"]+C["qual_codebooks"]+C["presence_sig_levels"]+C["presence_sig_codebook"]+C["multi_counts"]+C["string_ids"])*8/nf,3),"derived_facts_not_stored":B["derived_facts"],"roundtrip_bitexact":B["roundtrip_bitexact"]}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(r,indent=1));print(json.dumps(r,indent=1))
if __name__=="__main__":main(int(sys.argv[1]) if len(sys.argv)>1 else 0)

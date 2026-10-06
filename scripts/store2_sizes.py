import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/"data"/"store2"/"drive";R=ROOT/"data"/"store2"/"results";NF=55062201;V3=4629551616;S1=410155258
def main(fd:int):
 B=json.loads((D/"build.json").read_text());C=B["components"];sz=lambda p:(D/p).stat().st_size if (D/p).exists() else 0
 nt=sum(sz(f"name_f{fd}{x}") for x in (".gf17rib",".gf17rib.l1","_end.gf17rib"));cont=sz("store2.gf17pg");tot=cont+nt
 cb=B["core_bits"];fsh=cb[2]/(cb[1]+cb[2]);fact=(C["sc"]+C["xc"])*fsh;tables=C["cb"]+C["cbx"]+C["cbd"]+C["ctx"]
 comp={"fact_stream":round(fact),"signatures_and_counts":round((C["sc"]+C["xc"])*(1-fsh)),"header_qid_gaps_sitelinks":C["sh"]+C["xh"],"fact_codebooks":C["cb"]+C["cbx"]+C["cbd"],"fact_context_lists":C["ctx"],"qualifiers":C["sq"]+C["xq"]+C["qcb"]+C["qcbx"]+C["qcbd"],"labels_titles":C["sl"]+C["xl"],"aliases_names":C["sn"]+C["xn"],"strings":C["ss"]+C["xs"],"model_tables":C["tabs"]+C["pcls"],"piece_table":C["bpe"],"hot_table":C["hot"]+C["hotx"],"meta":C["meta"],f"name_table_f{fd}":nt}
 o={"fd":fd,"total_bytes":tot,"container_bytes":cont,"name_table_bytes":nt,"v3_bytes":V3,"ratio_vs_v3":round(tot/V3,4),"store1_bytes":S1,"ratio_vs_store1":round(tot/S1,4),"gate_G1_max_bytes":300000000,"gate_G1_pass":tot<=300000000,"facts":NF,"fact_bits_per_fact":round(8*fact/NF,3),"fact_bits_with_tables":round(8*(fact+tables)/NF,3),"gate_G3_max_bits":7.0,"gate_G3_pass":8*fact/NF<=7.0,"fact_stream_note":"core stream (sc + its block index xc) holds signatures, counts and facts; the fact share is the encoder's measured fact-symbol bits over core bits (build.json core_bits[2]/(core_bits[1]+core_bits[2])), applied to the stored bytes including rANS flush and index","core_bits":dict(zip(["header","signature_counts","facts","label","title"],cb)),"components":comp,"sections":C,"roundtrip_bitexact":B["roundtrip_bitexact"]}
 R.mkdir(parents=True,exist_ok=True);(R/f"sizes_f{fd}.json").write_text(json.dumps(o,indent=1));print(json.dumps({k:v for k,v in o.items() if k not in ("sections",)},indent=1))
if __name__=="__main__":main(int(sys.argv[1]) if len(sys.argv)>1 else 2)

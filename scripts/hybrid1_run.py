import os,sys,json,time,argparse,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ap=argparse.ArgumentParser();ap.add_argument("--bake",default="");ap.add_argument("--opts",default="{}");ap.add_argument("--set",default="dev");ap.add_argument("--out",required=True);ap.add_argument("--name",default="");ap.add_argument("--cmd",default="");ap.add_argument("--meas",action="store_true");ap.add_argument("--ppl",action="store_true");a=ap.parse_args()
os.environ["AMNI_BLENDER_OPTS"]=a.opts
import torch,adam_t_probe8 as PR,adam_t_eval as A,hybrid1_measure as HM,hybrid1_ppl as HP
from amni.serve import blender_atex as B
cmd=a.cmd or "atex:"+a.bake;at=cmd.startswith("atex:");SETS=[s for s in a.set.split(",") if s in ("dev","probe")];OUT={s:a.out.format(set=s) for s in SETS}
v=B.Vram();v.start();T={};t=time.time();b=A.proto(cmd);T["load_s"]=round(time.time()-t,1);ph={"load":v.pk}
def phase(k,f):
    v.pk=v.cur();t=time.time();r=f();ph[k]=v.pk;T[k+"_s"]=round(time.time()-t,1);return r
X={"meas":phase("meas",lambda:HM.meas(b.e,v)) if a.meas and at else None,"ppl":phase("ppl",lambda:HP.ppl(b.e)) if a.ppl and at else None};A.proto=lambda c:b
def run(s):PR.P8=ROOT/"data"/"adam_t"/"eval"/("hybrid1_dev" if s=="dev" else "probe8");return PR.run(argparse.Namespace(proto_cmd=cmd,e2e=False,name=a.name or os.path.basename(a.bake.rstrip("/")),step=a.opts,out=OUT[s]))
O={s:phase("score_"+s,lambda s=s:run(s)) for s in SETS};v.stop()
BL={"bake":a.bake,"opts":json.loads(a.opts),"engine_opts":getattr(getattr(b,"e",None),"o",None),"sets":SETS,"outputs":OUT,**X,"proc_vram_peak_GB":{k:round(x/1e9,3) for k,x in ph.items()},"proc_vram_peak_all_GB":round(max(ph.values())/1e9,3),"times":T,"same_process":True,"alloc_conf_env":os.environ.get("PYTORCH_HIP_ALLOC_CONF",""),"torch":torch.__version__,"hip":torch.version.hip,"python":platform.python_version(),"throttle_obeyed_in_scoring":not os.environ.get("AMNI_BLENDER_NOTHR")}
for s in SETS:R=json.load(open(OUT[s]));R["blender2"]=BL;json.dump(R,open(OUT[s],"w"),indent=1)
SETS or (Path(a.out).parent.mkdir(parents=True,exist_ok=True),json.dump({"name":a.name,"blender2":BL},open(a.out,"w"),indent=1))
g=lambda d,k:(d.get(k) or {}).get("p")
for s,o in O.items():S=o["summary"];c=(S.get("chat_probe") or {}).get("all") or {};bp=S.get("blend_probe") or {};print(json.dumps({"name":o["name"],"set":s,"faithful_para":g(bp,"faithful_para"),"no_new_para":g(bp,"no_new_content_para"),"hedge_uncov":g(bp,"hedge_uncov"),"blend_coherent":g(bp,"coherent"),"chat_answered":g(c,"answered"),"chat_coherent":g(c,"coherent"),"chat_decline":g(c,"decline"),"chat_em_fp":g(c,"em_fp"),"distinct2":(S.get("diversity") or {}).get("distinct2_chat_probe"),"opener3_repeat":(S.get("diversity") or {}).get("opener3_repeat_rate"),"wall_s":o["wall_s"]}))
print(json.dumps({"name":a.name,"ppl":(X["ppl"] or {}).get("ppl"),"meas_peak_GB":(X["meas"] or {}).get("peak_proc_vram_GB"),"decode_tok_s":(X["meas"] or {}).get("decode_tok_s"),"prefill_tok_s":(X["meas"] or {}).get("prefill_tok_s"),"peak_all_GB":BL["proc_vram_peak_all_GB"],"phase_peaks":BL["proc_vram_peak_GB"]}))

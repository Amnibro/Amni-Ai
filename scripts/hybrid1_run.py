import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
ap=argparse.ArgumentParser();ap.add_argument("--bake",default="");ap.add_argument("--opts",default="{}");ap.add_argument("--set",default="dev",choices=("dev","probe"));ap.add_argument("--out",required=True);ap.add_argument("--name",default="");ap.add_argument("--cmd",default="");a=ap.parse_args()
os.environ["AMNI_BLENDER_OPTS"]=a.opts
import adam_t_probe8 as PR
PR.P8=ROOT/"data"/"adam_t"/"eval"/("hybrid1_dev" if a.set=="dev" else "probe8")
o=PR.run(argparse.Namespace(proto_cmd=a.cmd or "atex:"+a.bake,e2e=False,name=a.name or os.path.basename(a.bake),step=a.opts,out=a.out))
S=o["summary"];c=S["chat_probe"]["all"];b=S["blend_probe"]
print(json.dumps({"name":o["name"],"opts":a.opts,"set":a.set,"faithful_para":b["faithful_para"]["p"],"no_new_para":b["no_new_content_para"]["p"],"hedge_uncov":b["hedge_uncov"]["p"],"blend_coherent":b["coherent"]["p"],"chat_answered":c["answered"]["p"],"chat_coherent":c["coherent"]["p"],"chat_decline":c["decline"]["p"],"chat_em_fp":c["em_fp"]["p"],"chat_canned":c["canned"]["p"],"distinct2":S["diversity"]["distinct2_chat_probe"],"opener3_repeat":S["diversity"]["opener3_repeat_rate"],"wall_s":o["wall_s"]}))

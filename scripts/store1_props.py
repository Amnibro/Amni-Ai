import json,time,hashlib,urllib.request,urllib.parse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"data"/"store1"/"props_en.json";API="https://www.wikidata.org/w/api.php"
def keep()->list:
 t=(ROOT/"crates"/"adam-wikifact"/"src"/"lib.rs").read_text();a=t.index("KEEP: [u16; 189] = [")+len("KEEP: [u16; 189] = [");b=t.index("]",a);q=t.index("QKEEP: [u16; 12] = [")+len("QKEEP: [u16; 12] = [");r=t.index("]",q)
 return sorted({int(x) for x in t[a:b].split(",")}|{int(x) for x in t[q:r].split(",")})
def fetch(ids:list)->dict:
 u=API+"?"+urllib.parse.urlencode({"action":"wbgetentities","ids":"|".join(f"P{i}" for i in ids),"props":"labels|aliases|descriptions|datatype","languages":"en","format":"json"});rq=urllib.request.Request(u,headers={"User-Agent":"Amni-STORE1-build/1.0 (CC0 property labels; one-off build fetch)"})
 with urllib.request.urlopen(rq,timeout=60) as r:b=r.read()
 return json.loads(b),hashlib.sha256(b).hexdigest(),u
def main():
 K=keep();E={};H=[]
 for i in range(0,len(K),50):
  j,h,u=fetch(K[i:i+50]);H.append({"url":u,"sha256":h});E.update(j["entities"]);time.sleep(1)
 P={k:{"label":v.get("labels",{}).get("en",{}).get("value",""),"aliases":[a["value"] for a in v.get("aliases",{}).get("en",[])],"description":v.get("descriptions",{}).get("en",{}).get("value",""),"datatype":v.get("datatype","")} for k,v in E.items()}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps({"license":"CC0 1.0 (Wikidata)","source":API,"fetched":time.strftime("%Y-%m-%dT%H:%M:%S"),"requests":H,"props":P},indent=1,ensure_ascii=False));print(len(P),sum(len(v["aliases"]) for v in P.values()))
if __name__=="__main__":main()

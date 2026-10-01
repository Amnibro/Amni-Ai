import os,numpy as np
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","facts","a5_cc0_facts.tsv")
A=["Ka","Lo","Mi","Ve","Zu","Ta","Ri","No","Sa","Fe","Gu","Di","Be","Ho","Ya","Qu"];B=["rel","van","sik","dor","mun","tal","pes","gri","lon","vek","sar","bim","tor","nel","quo","fal"];C=["a","en","is","o","ur","eth","ix","an"]
CTRY=["Q30","Q142","Q183","Q145","Q38","Q29","Q17","Q155","Q16","Q408"];CONT={"Q30":"Q49","Q16":"Q49","Q155":"Q18","Q17":"Q48","Q408":"Q3960"};OCC=["Q901","Q36180","Q82955","Q639669","Q1028181","Q81096","Q39631","Q37226"]
def main():
 r=np.random.default_rng(1700);seen=set();names=[]
 while len(names)<100:
  n=A[r.integers(16)]+B[r.integers(16)]+C[r.integers(8)]
  if n not in seen:seen.add(n);names.append(n)
 ppl=["person:"+n.capitalize()+" "+B[r.integers(16)].capitalize()+C[r.integers(8)] for n in names[:40]];twn=["town:"+n.capitalize()+"ford" for n in names[40:70]];org=["org:"+n.capitalize()+" Works" for n in names[70:85]];riv=["river:"+n.capitalize()+" River" for n in names[85:100]]
 f=[]
 for i,p in enumerate(ppl):
  c=CTRY[r.integers(10)];f+= [(p,"P31","Q5"),(p,"P19",twn[r.integers(30)]),(p,"P27",c),(p,"P106",OCC[r.integers(8)])]
  f.append((p,"P106",OCC[(OCC.index(f[-1][2])+1+r.integers(7))%8]) if i<10 else (p,"P21",["Q6581097","Q6581072"][r.integers(2)]))
 for i,t in enumerate(twn):
  c=CTRY[r.integers(10)];f+=[(t,"P31","Q3957"),(t,"P17",c),(t,"P206",riv[r.integers(15)]),(t,"P30",CONT.get(c,"Q46")),(t,"P47",twn[(i+1+r.integers(29))%30])]
 for i,o in enumerate(org):
  c=CTRY[r.integers(10)];f+=[(o,"P31",["Q3918","Q4830453"][r.integers(2)]),(o,"P159",twn[r.integers(30)]),(o,"P17",c),(o,"P112",ppl[r.integers(40)]),(o,"P749",org[(i+1+r.integers(14))%15])]
 for i,v in enumerate(riv):
  c=CTRY[r.integers(10)];f+=[(v,"P31","Q4022"),(v,"P17",c),(v,"P30",CONT.get(c,"Q46")),(v,"P403",riv[(i+1+r.integers(14))%15]),(v,"P885",twn[r.integers(30)])]
 assert len(f)==500 and len({x[0] for x in f})==100
 with open(OUT,"w") as w:
  w.write("# 500 invented facts about 100 invented entities (people, towns, organisations, rivers). Written for TORUS-A gate A5, 2026-09-30. Dedicated to the public domain: CC0 1.0.\n# subject\tproperty\tobject (Wikidata QID, or an invented entity named in this file)\n")
  [w.write(f"{a}\t{b}\t{c}\n") for a,b,c in f]
 print(len(f),"facts",OUT)
if __name__=="__main__":main()

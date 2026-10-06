import os,sys,json,collections
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
from amni.compute import decontam as D
OUT="data/adam_t/eval/hybrid1_dev";LIC="CC0-1.0, written for mission HYBRID-1 (2026-10-05)"
P=[("Marlowe Bridge","Marlowe Bridge is a stone arch bridge over the River Tane in the county of Eastmere. It carries a single lane of traffic and was closed to lorries after a survey of its piers.","What is Marlowe Bridge?"),
("Quillon Island","Quillon Island is a low sandy island off the southern coast of Varnia. Its only settlement is a fishing hamlet of about forty houses.","Tell me about Quillon Island."),
("Drabble Hill Tunnel","Drabble Hill Tunnel is a railway tunnel on the Ashby to Corran line. It is 1.2 km long and has two parallel bores.","How long is the Drabble Hill tunnel?"),
("Ostrander Lake","Ostrander Lake is a glacial lake in the Pellam Range. The lake freezes from December to April and is fed by three small streams.","Where is Ostrander Lake?"),
("Fennimore Observatory","Fennimore Observatory is an astronomical observatory built on a ridge above the town of Halcombe. It houses a 60 cm reflecting telescope used by a local society.","What does Fennimore Observatory have?"),
("Tolliver Press","Tolliver Press is a small publishing house based in Brightwater. It prints poetry pamphlets and regional history books.","What does Tolliver Press publish?"),
("Greyfriars Market","Greyfriars Market is a covered market hall in the centre of Dunmere. Traders sell cheese, bread and second-hand books on weekdays.","What can you buy at Greyfriars Market?"),
("Helvering Canal","Helvering Canal is a narrow canal that links the River Oxe with the port of Kelsey. It has nine locks and is now used mainly by pleasure boats.","How many locks does the Helvering Canal have?"),
("Pimlott Abbey","Pimlott Abbey is a ruined abbey on a headland in the north of Tarn County. Only the nave walls and a bell tower remain standing.","What is left of Pimlott Abbey?"),
("Ravenscar Mine","Ravenscar Mine was a slate mine in the hills above Lornton. It employed around two hundred men and closed in the middle of the last century.","Who worked at Ravenscar Mine?"),
("Wexcombe Choir","Wexcombe Choir is a mixed amateur choir founded by teachers from the village school. It performs at the harvest festival and sings carols each winter.","When does the Wexcombe Choir perform?"),
("Ambrose Kettle","Ambrose Kettle was a clockmaker from the town of Yarrow. He built a tall case clock that still stands in the town hall.","What did Ambrose Kettle make?"),
("Nilsson Falls","Nilsson Falls is a waterfall on the Ogden River that drops about thirty metres over a basalt ledge. A footpath leads to a viewing platform.","How high are Nilsson Falls?"),
("Corvin Lighthouse","Corvin Lighthouse is a white tower on a rocky point near Saltmoor. Its lamp was automated and the keepers' cottages are now holiday lets.","What happened to the keepers at Corvin Lighthouse?"),
("Sable Cove","Sable Cove is a sheltered bay with black pebbles on the north coast of Merrin Island. Seals often rest on the rocks at the western end.","What animals can be seen at Sable Cove?"),
("Brantley Academy","Brantley Academy is a secondary school in the town of Oakhurst. It teaches about nine hundred pupils aged eleven to eighteen.","How many pupils attend Brantley Academy?"),
("Calder Orchard","Calder Orchard is a community orchard planted with old varieties of apple and pear. Volunteers pick the fruit each autumn and press it into juice.","What do volunteers do at Calder Orchard?"),
("Isken Pass","Isken Pass is a mountain pass at 2,100 metres between the valleys of Rho and Seld. It is closed by snow for most of the winter.","How high is Isken Pass?"),
("Mossgiel Farm","Mossgiel Farm is a hill farm that keeps a flock of black-faced sheep. The farmhouse dates from the early eighteenth century.","What animals live at Mossgiel Farm?"),
("Harbour Light Cafe","Harbour Light Cafe is a family-run cafe on the quay at Pellow. It is known for its crab sandwiches and its view of the fishing boats.","What is the Harbour Light Cafe known for?"),
("Calculator","The answer is 84. Working: 12 x 7 = 84.","What is 12 times 7?"),
("Calculator","The answer is 19. Working: 45 - 26 = 19.","What is 45 minus 26?"),
("Calculator","The answer is 9. Working: 63 / 7 = 9.","What is 63 divided by 7?"),
("Calculator","The answer is 150. Working: 25 x 6 = 150.","How much is 25 times 6?"),
("Memory","My sister is called Odile and she lives in Lyon.","What is my sister called?"),
("Memory","I keep two cats, one grey and one ginger.","How many cats do I have?"),
("Memory","I start work at seven on Mondays and finish at three.","When do I start work on Mondays?"),
("Memory","My favourite tea is jasmine and I drink it without sugar.","What tea do I like?"),
("Verrick, capital of","Verrick is the capital of the republic of Altmark.","What is Verrick the capital of?"),
("Lindqvist Prize, awarded for","The Lindqvist Prize is awarded for achievement in marine biology.","What is the Lindqvist Prize awarded for?"),
("Tobias Renner, occupation","Tobias Renner worked as a cartographer.","What was Tobias Renner's job?"),
("Karsten River, mouth","The Karsten River flows into the Bay of Orlo.","Where does the Karsten River end?"),
("Pellucid Glass Company","Pellucid Glass Company is a manufacturer of laboratory glassware. It was set up in 1921 in the town of Ashgrove and supplies universities across the region.","What does Pellucid Glass Company make?"),
("Anselm Gardens","Anselm Gardens is a botanical garden with a heated palm house and a rose walk. It is open daily and has a small tea room.","What is in Anselm Gardens?"),
("Roddick Viaduct","Roddick Viaduct is a railway viaduct with eleven arches that spans the Wend valley. It was built from local red brick.","What is Roddick Viaduct built from?"),
("Fairlight Crossing","Fairlight Crossing is a small village at the junction of two old drove roads. It has a pub, a chapel and a bus stop.","What is there in Fairlight Crossing?"),
("Zeller Theatre","Zeller Theatre is a playhouse with four hundred seats in the city of Brandt. It stages new writing and a pantomime every December.","How many seats does Zeller Theatre have?")]
U=[("none","What are the opening hours of Marlowe Bridge Museum?",[]),
("none","Please outline what the Glennor Accord says.",[]),
("none","Who designed the Pelham Tower?",[]),
("none","What is the population of Westerholm?",[]),
("none","Tell me what Cobb Valley Railway is.",[]),
("none","How deep is Lake Insterburg?",[]),
("other","What is the tallest peak in the Dovrek Range?",["Fenwick Dairy: Fenwick Dairy is a cheese maker in the village of Thorpe Marsh. It sells soft cheese to local shops."]),
("other","Who wrote the novel The Salt Ledger?",["Iverson Reef: Iverson Reef is a coral reef off the east coast of Calla. Divers visit it between May and September."]),
("other","What does Hobart Lane Primary School teach?",["Ketteridge Bypass: Ketteridge Bypass is a dual carriageway around the town of Ketteridge. It opened to traffic last spring."]),
("other","Where is the Sandor Gallery?",["Wren Hollow Pottery: Wren Hollow Pottery is a studio that makes hand thrown mugs and bowls. It runs weekend classes."]),
("other","How old is the Elmgrove Oak?",["Tindall Rowing Club: Tindall Rowing Club is a rowing club on the River Sedge. Members train before dawn on weekdays."]),
("when","When was Tarvin Hall built?",["Tarvin Hall: Tarvin Hall is a manor house on the edge of Ludby. It has a walled garden and a gatehouse."]),
("when","When was the Corbel Society founded?",["Corbel Society: The Corbel Society is a group that studies church architecture. It publishes a yearly journal."]),
("when","When did Pettigrew Harbour open?",["Pettigrew Harbour: Pettigrew Harbour is a small tidal harbour with a stone quay. Fishing boats unload there each morning."])]
C=[("greet","Hello there, Adam! How is your afternoon going?",["afternoon","good","going","hello","hi"]),
("greet","Hey Adam, I just made a big pot of soup for the week.",["soup","pot","week","nice","made"]),
("greet","Morning! I am about to head out for a bike ride.",["morning","bike","ride","out","enjoy"]),
("greet","Hi again, Adam, I finally finished painting the spare room.",["paint*","room","finish*","spare","nice"]),
("small","What a grey and drizzly day it is outside.",["grey","drizzl*","day","rain*","weather"]),
("small","I think I will try baking bread this weekend.",["bak*","bread","weekend","try","fun"]),
("small","My neighbour's dog keeps barking at the postman.",["dog","bark*","neighbo*","postman","noisy"]),
("vent","I am so tired of my commute taking two hours every day.",["commute","tired","hours","long","hard"]),
("vent","My laptop crashed and I lost the report I was writing.",["laptop","lost","report","crash*","frustrat*"]),
("vent","In meetings my suggestions seem to get ignored by everybody.",["listen*","ideas","meeting*","frustrat*","ignored"]),
("vent","I have had a headache since lunchtime and I cannot focus.",["headache","focus","lunch*","rest","sorry"]),
("vent","The builders next door started drilling at six this morning.",["builder*","drill*","morning","noise","early"]),
("idiom","It is raining cats and dogs where I live.",["rain*","cats","dogs","pour*","heavy"]),
("idiom","I am feeling under the weather today.",["weather","under","sick","unwell","rest"])]
def main():
    os.makedirs(OUT,exist_ok=True);B=[];G=[]
    for i,(t,x,q) in enumerate(P):
        m=f"{t}: {x}" if t in("Calculator","Memory") or "," not in t else f"{t}: {x}"
        B.append({"id":f"d1b{i:03d}","kind":"para","sub":"para","material":[m],"q":q,"ref":x,"title":t,"license":LIC})
    for i,(s,q,m) in enumerate(U):B.append({"id":f"d1u{i:03d}","kind":"uncov","sub":"unc_"+s,"material":m,"q":q,"title":m[0].split(":")[0] if m else "none","license":LIC})
    for i,(s,q,k) in enumerate(C):G.append({"id":f"d1g{i:03d}","cat":"chat","sub":s,"q":q,"turns":[],"kw":k,"gold":[],"ctx":[],"license":LIC})
    S=Path('/home/anmire/ai/Amni-Ai');fs=sorted({p.resolve() for pat in D.PATS for p in S.glob(pat) if p.is_file() and not D.SKIP.search(p.relative_to(S).as_posix()) and '.decontam_cache' not in p.parts});ix=D.index(6,files=fs,cache=None);bad=[(r["id"],D.dirty(" ".join(r.get("material",[])+[r["q"]]),ix=ix)) for r in B+G if D.dirty(" ".join(r.get("material",[])+[r["q"]]),ix=ix)]
    assert len(B)+len(G)>=40 and not bad,bad
    open(f"{OUT}/blend_probe.jsonl","w").write("".join(json.dumps(r)+"\n" for r in B));open(f"{OUT}/greet_vent_probe.jsonl","w").write("".join(json.dumps(r)+"\n" for r in G))
    json.dump({"license":LIC,"n_blend":len(B),"n_chat":len(G),"overlap_ngram":6,"overlap_flagged":len(bad),"checked_against":f"amni.compute.decontam.index(6) over {len(fs)} eval source files of the shared tree (probe8, chat suites, h4, harness, cognition, concepts eval)"},open(f"{OUT}/meta.json","w"),indent=1);print(len(B),len(G),"overlap flagged",len(bad))
if __name__=="__main__":main()

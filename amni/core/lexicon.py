import numpy as np, hashlib, struct, json as _json
from typing import Dict,List,Tuple,Optional,Set
from pathlib import Path
from collections import defaultdict
try:
    import amni_kernels as _ak
    _HAS_AK=True
except ImportError:
    _HAS_AK=False
POS_MAP = {"n":0,"v":1,"a":2,"s":2,"r":3,"prep":4,"conj":5,"det":6,"pron":7,"num":8,"interj":9,"other":10}
POS_NAMES = {0:"noun",1:"verb",2:"adj",3:"adv",4:"prep",5:"conj",6:"det",7:"pron",8:"num",9:"interj",10:"other"}
DOMAIN_MAP = {"general":0,"science":1,"math":2,"code":3,"language":4,"art":5,"history":6,"philosophy":7,"economics":8,"law":9,"medicine":10,"engineering":11,"geography":12,"music":13,"literature":14,"religion":15,"politics":16,"technology":17,"nature":18,"food":19,"sports":20,"military":21,"creative":22,"logic":23,"technical":24}
DOMAIN_NAMES = {v:k for k,v in DOMAIN_MAP.items()}
N_POS = len(POS_MAP)
N_DOMAINS = len(DOMAIN_MAP)
REL_TYPES = {"synonym":0,"antonym":1,"hypernym":2,"hyponym":3,"meronym":4,"holonym":5,"co_occurs":6,"similar":7,"derived":8,"causes":9,"entails":10,"also_see":11}
REL_NAMES = {v:k for k,v in REL_TYPES.items()}
N_RELS = len(REL_TYPES)
WN_DOMAIN_KEYWORDS = {
    "science":{"physics","chemistry","biology","geology","astronomy","molecule","atom","cell","organism","species","evolution","electron","proton","neutron","quantum","energy","force","mass","velocity","acceleration","gravity","photon","wave","particle","reaction","compound","element","nucleus","enzyme","protein","dna","rna","chromosome","gene","mutation","fossil","mineral","crystal","magnet","circuit"},
    "math":{"number","equation","formula","theorem","proof","algebra","calculus","geometry","matrix","vector","integral","derivative","function","set","graph","prime","factorial","logarithm","exponent","polynomial","fraction","ratio","proportion","probability","statistics","mean","median","variance","sum","product","quotient","remainder","divisor","multiple","factor"},
    "medicine":{"disease","symptom","treatment","diagnosis","surgery","medicine","therapy","infection","virus","bacteria","vaccine","drug","dose","patient","hospital","doctor","nurse","organ","tissue","blood","bone","muscle","nerve","brain","heart","lung","liver","kidney","stomach","intestine","skin","immune","allergy","cancer","tumor","inflammation"},
    "law":{"court","judge","jury","trial","verdict","sentence","law","statute","regulation","contract","tort","plaintiff","defendant","attorney","lawyer","prosecution","defense","witness","evidence","testimony","appeal","jurisdiction","legislation","constitution","amendment","rights","liability","damages","negligence","crime","felony","misdemeanor"},
    "economics":{"market","price","supply","demand","inflation","deflation","gdp","trade","tariff","tax","budget","debt","credit","interest","investment","stock","bond","currency","exchange","profit","loss","revenue","cost","wage","salary","employment","unemployment","monopoly","competition","subsidy","commodity","capital","asset","liability","equity"},
    "history":{"war","battle","empire","dynasty","revolution","colony","independence","treaty","alliance","civilization","ancient","medieval","renaissance","industrial","monarchy","republic","democracy","conquest","migration","settlement","archaeology","artifact","monument","era","century","decade","period","reign","coronation","abdication"},
    "philosophy":{"ethics","morality","logic","reason","truth","knowledge","existence","consciousness","mind","soul","virtue","justice","freedom","beauty","reality","perception","belief","doubt","argument","premise","conclusion","metaphysics","epistemology","ontology","aesthetics","utilitarianism","determinism","empiricism","rationalism"},
    "art":{"painting","sculpture","drawing","sketch","canvas","color","pigment","brush","palette","composition","perspective","portrait","landscape","abstract","impressionism","cubism","surrealism","expressionism","baroque","gothic","classical","modern","contemporary","gallery","museum","exhibition","aesthetic","visual","design","illustration"},
    "music":{"melody","harmony","rhythm","tempo","note","chord","scale","key","octave","pitch","tone","instrument","guitar","piano","violin","drum","orchestra","symphony","sonata","concerto","opera","jazz","blues","rock","classical","composer","conductor","singer","vocal","bass","treble","beat","measure","bar"},
    "literature":{"novel","poem","story","essay","drama","fiction","nonfiction","prose","verse","stanza","metaphor","simile","allegory","irony","satire","tragedy","comedy","narrative","plot","character","setting","theme","protagonist","antagonist","author","poet","playwright","chapter","paragraph","sentence"},
    "technology":{"computer","software","hardware","algorithm","database","network","internet","server","cloud","processor","memory","storage","program","code","binary","digital","analog","wireless","bluetooth","wifi","encryption","protocol","interface","browser","application","operating","system","virtual","artificial","intelligence"},
    "engineering":{"bridge","building","structure","material","steel","concrete","load","stress","strain","pressure","voltage","current","resistance","circuit","motor","engine","turbine","generator","pump","valve","gear","bearing","shaft","weld","bolt","rivet","beam","column","foundation","mechanical","electrical","civil","chemical"},
    "geography":{"continent","ocean","river","mountain","valley","desert","forest","island","peninsula","plateau","climate","weather","temperature","precipitation","latitude","longitude","equator","hemisphere","arctic","tropical","temperate","coastal","inland","urban","rural","population","border","territory","region","landscape"},
    "nature":{"tree","flower","plant","animal","bird","fish","insect","mammal","reptile","amphibian","forest","jungle","ocean","river","lake","mountain","desert","grassland","wetland","coral","reef","ecosystem","habitat","biodiversity","endangered","species","predator","prey","migration","hibernation","photosynthesis","pollination"},
    "food":{"cook","bake","fry","boil","grill","roast","recipe","ingredient","spice","herb","flour","sugar","salt","butter","oil","meat","vegetable","fruit","grain","bread","pasta","rice","cheese","egg","milk","cream","sauce","soup","salad","dessert","beverage","wine","beer","coffee","tea"},
    "sports":{"game","match","tournament","championship","league","team","player","coach","referee","score","goal","point","win","lose","draw","ball","field","court","track","race","swim","run","jump","throw","kick","bat","racket","net","basket","penalty","foul","record","medal","trophy"},
    "military":{"army","navy","air","force","marine","soldier","officer","general","admiral","captain","sergeant","battalion","regiment","division","brigade","weapon","rifle","cannon","missile","bomb","tank","helicopter","submarine","aircraft","carrier","base","camp","fort","strategy","tactics","combat","deployment","intelligence","reconnaissance"},
    "religion":{"god","faith","prayer","worship","church","temple","mosque","synagogue","bible","quran","torah","scripture","prophet","saint","angel","heaven","hell","soul","spirit","sacred","holy","divine","blessing","sin","salvation","redemption","resurrection","pilgrimage","ritual","ceremony","clergy","priest","monk","nun"},
    "politics":{"government","president","minister","parliament","congress","senate","election","vote","party","policy","law","regulation","democracy","republic","monarchy","dictatorship","diplomacy","foreign","domestic","campaign","debate","legislation","executive","judicial","legislative","constitution","amendment","bill","resolution","committee"},
}
class WordNonce:
    __slots__ = ('word','nonce_id','vector','pos_id','domain_id','freq','synset_ids')
    def __init__(self,word:str,nonce_id:int,vector:np.ndarray,pos_id:int=10,domain_id:int=0,freq:float=0.0,synset_ids:Optional[List[str]]=None):
        self.word=word
        self.nonce_id=nonce_id
        self.vector=vector
        self.pos_id=pos_id
        self.domain_id=domain_id
        self.freq=freq
        self.synset_ids=synset_ids or []
    def to_dict(self)->dict:
        return {"w":self.word,"id":self.nonce_id,"pos":self.pos_id,"dom":self.domain_id,"freq":self.freq,"syns":self.synset_ids}
def word_to_hash_vector(word:str,dim:int=512,seed:int=42)->np.ndarray:
    if _HAS_AK and hasattr(_ak,'word_to_hash_vector'):
        return np.array(_ak.word_to_hash_vector(word,dim,seed),dtype=np.float32)
    h=hashlib.sha256(f"{seed}:{word.lower().strip()}".encode()).digest()
    rng=np.random.RandomState(int.from_bytes(h[:4],'little'))
    v=rng.randn(dim).astype(np.float32)
    return v/np.linalg.norm(v).clip(min=1e-8)
def encode_structured_nonce(base_vec:np.ndarray,pos_id:int,domain_id:int,freq:float=0.0)->np.ndarray:
    if _HAS_AK and hasattr(_ak,'encode_structured_nonce'):
        return np.array(_ak.encode_structured_nonce(base_vec.tolist(),pos_id,domain_id,freq),dtype=np.float32)
    dim=len(base_vec)
    v=base_vec.copy().astype(np.float32)
    pos_start=dim-N_POS-N_DOMAINS-1
    dom_start=pos_start+N_POS
    freq_idx=dom_start+N_DOMAINS
    v[pos_start:pos_start+N_POS]*=0.5
    v[pos_start+pos_id]+=0.3
    v[dom_start:dom_start+N_DOMAINS]*=0.5
    v[dom_start+domain_id]+=0.3
    v[freq_idx]=min(freq/100.0,1.0)*0.2 if freq_idx<dim else v[freq_idx]
    return v/np.linalg.norm(v).clip(min=1e-8)
class LexEntry:
    __slots__ = ('word','definitions','pos_tags','domains','synonyms','antonyms','hypernyms','hyponyms','meronyms','holonyms','co_occurs','freq')
    def __init__(self,word:str):
        self.word=word
        self.definitions:List[str]=[]
        self.pos_tags:Set[int]=set()
        self.domains:Set[int]=set()
        self.synonyms:Set[str]=set()
        self.antonyms:Set[str]=set()
        self.hypernyms:Set[str]=set()
        self.hyponyms:Set[str]=set()
        self.meronyms:Set[str]=set()
        self.holonyms:Set[str]=set()
        self.co_occurs:Dict[str,float]={}
        self.freq:float=0.0
    def primary_pos(self)->int:
        return min(self.pos_tags) if self.pos_tags else POS_MAP["other"]
    def primary_domain(self)->int:
        return min(self.domains) if self.domains else DOMAIN_MAP["general"]
    def all_relations(self)->Dict[int,Set[str]]:
        return {REL_TYPES["synonym"]:self.synonyms,REL_TYPES["antonym"]:self.antonyms,REL_TYPES["hypernym"]:self.hypernyms,REL_TYPES["hyponym"]:self.hyponyms,REL_TYPES["meronym"]:self.meronyms,REL_TYPES["holonym"]:self.holonyms}
class Lexicon:
    def __init__(self,dim:int=512):
        self.dim=dim
        self.entries:Dict[str,LexEntry]={}
        self.word_nonces:Dict[str,WordNonce]={}
        self.nonce_to_word:Dict[int,str]={}
        self.vocab_size=0
        self._next_id=0
    def add_entry(self,entry:LexEntry)->WordNonce:
        w=entry.word.lower().strip()
        if w in self.word_nonces:
            existing=self.entries[w]
            existing.pos_tags|=entry.pos_tags
            existing.domains|=entry.domains
            existing.synonyms|=entry.synonyms
            existing.antonyms|=entry.antonyms
            existing.hypernyms|=entry.hypernyms
            existing.hyponyms|=entry.hyponyms
            existing.meronyms|=entry.meronyms
            existing.holonyms|=entry.holonyms
            existing.definitions.extend(entry.definitions)
            existing.freq=max(existing.freq,entry.freq)
            wn=self.word_nonces[w]
            wn.pos_id=existing.primary_pos()
            wn.domain_id=existing.primary_domain()
            wn.freq=existing.freq
            wn.vector=encode_structured_nonce(word_to_hash_vector(w,self.dim),wn.pos_id,wn.domain_id,wn.freq)
            return wn
        self.entries[w]=entry
        nid=self._next_id
        self._next_id+=1
        base_vec=word_to_hash_vector(w,self.dim)
        vec=encode_structured_nonce(base_vec,entry.primary_pos(),entry.primary_domain(),entry.freq)
        wn=WordNonce(w,nid,vec,entry.primary_pos(),entry.primary_domain(),entry.freq)
        self.word_nonces[w]=wn
        self.nonce_to_word[nid]=w
        self.vocab_size=self._next_id
        return wn
    def lookup(self,word:str)->Optional[WordNonce]:
        return self.word_nonces.get(word.lower().strip())
    def lookup_id(self,nonce_id:int)->Optional[WordNonce]:
        w=self.nonce_to_word.get(nonce_id)
        return self.word_nonces.get(w) if w else None
    def find_similar(self,word:str,top_k:int=10)->List[Tuple[str,float]]:
        w=word.lower().strip()
        entry=self.entries.get(w)
        if not entry: return []
        cands=[]
        for s in entry.synonyms: cands.append((s,0.9))
        for h in entry.hypernyms: cands.append((h,0.7))
        for h in entry.hyponyms: cands.append((h,0.6))
        for co,sc in entry.co_occurs.items(): cands.append((co,min(sc*0.5,0.55)))
        for m in entry.meronyms: cands.append((m,0.4))
        for h in entry.holonyms: cands.append((h,0.4))
        seen=set()
        deduped=[]
        for wd,sc in sorted(cands,key=lambda x:-x[1]):
            wl=wd.lower().strip()
            if wl!=w and wl not in seen and wl in self.entries:
                seen.add(wl)
                deduped.append((wl,sc))
        return deduped[:top_k]
    def words_by_domain(self,domain_id:int)->List[str]:
        return [w for w,wn in self.word_nonces.items() if wn.domain_id==domain_id]
    def words_by_pos(self,pos_id:int)->List[str]:
        return [w for w,wn in self.word_nonces.items() if wn.pos_id==pos_id]
    def get_nonce_matrix(self)->np.ndarray:
        n=self.vocab_size
        mat=np.zeros((n,self.dim),dtype=np.float32)
        for w,wn in self.word_nonces.items():
            mat[wn.nonce_id]=wn.vector
        return mat
    def save(self,path:str):
        p=Path(path)
        p.mkdir(parents=True,exist_ok=True)
        meta={"dim":self.dim,"vocab_size":self.vocab_size,"n_pos":N_POS,"n_domains":N_DOMAINS,"n_rels":N_RELS}
        with open(str(p/"lexicon_meta.json"),'w') as f: _json.dump(meta,f)
        entries_data={}
        for w,e in self.entries.items():
            entries_data[w]={"defs":e.definitions[:3],"pos":list(e.pos_tags),"dom":list(e.domains),"syn":list(e.synonyms)[:20],"ant":list(e.antonyms)[:10],"hyper":list(e.hypernyms)[:10],"hypo":list(e.hyponyms)[:10],"mero":list(e.meronyms)[:10],"holo":list(e.holonyms)[:10],"freq":e.freq}
        with open(str(p/"lexicon_entries.json"),'w') as f: _json.dump(entries_data,f)
        nonce_data={w:wn.to_dict() for w,wn in self.word_nonces.items()}
        with open(str(p/"lexicon_nonces.json"),'w') as f: _json.dump(nonce_data,f)
        mat=self.get_nonce_matrix()
        mat.astype(np.float16).tofile(str(p/"lexicon_vectors.bin"))
        print(f"lexicon saved: {self.vocab_size} words, {self.dim}d vectors to {p}")
    @classmethod
    def load(cls,path:str)->'Lexicon':
        p=Path(path)
        with open(str(p/"lexicon_meta.json")) as f: meta=_json.load(f)
        lex=cls(dim=meta["dim"])
        with open(str(p/"lexicon_entries.json")) as f: entries_data=_json.load(f)
        with open(str(p/"lexicon_nonces.json")) as f: nonce_data=_json.load(f)
        vecs=np.fromfile(str(p/"lexicon_vectors.bin"),dtype=np.float16).reshape(meta["vocab_size"],meta["dim"]).astype(np.float32)
        for w,nd in nonce_data.items():
            nid=nd["id"]
            ed=entries_data.get(w,{})
            entry=LexEntry(w)
            entry.definitions=ed.get("defs",[])
            entry.pos_tags=set(ed.get("pos",[10]))
            entry.domains=set(ed.get("dom",[0]))
            entry.synonyms=set(ed.get("syn",[]))
            entry.antonyms=set(ed.get("ant",[]))
            entry.hypernyms=set(ed.get("hyper",[]))
            entry.hyponyms=set(ed.get("hypo",[]))
            entry.meronyms=set(ed.get("mero",[]))
            entry.holonyms=set(ed.get("holo",[]))
            entry.freq=ed.get("freq",0.0)
            lex.entries[w]=entry
            wn=WordNonce(w,nid,vecs[nid].copy(),nd["pos"],nd["dom"],nd.get("freq",0.0),nd.get("syns",[]))
            lex.word_nonces[w]=wn
            lex.nonce_to_word[nid]=w
            lex._next_id=max(lex._next_id,nid+1)
        lex.vocab_size=lex._next_id
        return lex
def classify_synset_domain(synset)->int:
    name=synset.name().lower()
    defn=(synset.definition() or "").lower()
    combined=f"{name} {defn}"
    best_domain,best_score="general",0
    for domain,keywords in WN_DOMAIN_KEYWORDS.items():
        score=sum(1 for kw in keywords if kw in combined)
        lexname=getattr(synset,'lexname',lambda:"")()
        if isinstance(lexname,str):
            ln=lexname.lower()
            if domain=="science" and any(x in ln for x in ["physics","chemistry","biology","geology"]): score+=3
            elif domain=="medicine" and any(x in ln for x in ["body","medicine","health"]): score+=3
            elif domain=="food" and "food" in ln: score+=3
            elif domain=="nature" and any(x in ln for x in ["plant","animal","weather"]): score+=3
            elif domain=="art" and any(x in ln for x in ["art","color"]): score+=3
            elif domain=="music" and "music" in ln: score+=3
            elif domain=="sports" and "sport" in ln: score+=3
            elif domain=="military" and "military" in ln: score+=3
            elif domain=="religion" and "religion" in ln: score+=3
            elif domain=="law" and "law" in ln: score+=3
            elif domain=="economics" and any(x in ln for x in ["commerce","money","business"]): score+=3
            elif domain=="politics" and any(x in ln for x in ["government","politic"]): score+=3
        if score>best_score:
            best_score=score
            best_domain=domain
    return DOMAIN_MAP.get(best_domain,0)
def ingest_wordnet(lexicon:Lexicon,max_words:int=0)->int:
    from nltk.corpus import wordnet as wn
    count=0
    seen_words=set()
    for synset in wn.all_synsets():
        domain_id=classify_synset_domain(synset)
        pos_id=POS_MAP.get(synset.pos(),POS_MAP["other"])
        defn=synset.definition() or ""
        hypers={h.lemmas()[0].name().lower().replace('_',' ') for h in synset.hypernyms() if h.lemmas()}
        hypos={h.lemmas()[0].name().lower().replace('_',' ') for h in synset.hyponyms() if h.lemmas()}
        meros=set()
        for m in synset.part_meronyms()+synset.substance_meronyms()+synset.member_meronyms():
            if m.lemmas(): meros.add(m.lemmas()[0].name().lower().replace('_',' '))
        holos=set()
        for h in synset.part_holonyms()+synset.substance_holonyms()+synset.member_holonyms():
            if h.lemmas(): holos.add(h.lemmas()[0].name().lower().replace('_',' '))
        also={s.lemmas()[0].name().lower().replace('_',' ') for s in synset.also_sees() if s.lemmas()}
        lemma_names=[l.name().lower().replace('_',' ') for l in synset.lemmas()]
        ant_names=set()
        for l in synset.lemmas():
            for a in l.antonyms():
                ant_names.add(a.name().lower().replace('_',' '))
        syns=set(lemma_names)
        for lname in lemma_names:
            if max_words>0 and count>=max_words: break
            if lname in seen_words:
                entry=lexicon.entries.get(lname)
                if entry:
                    entry.pos_tags.add(pos_id)
                    entry.domains.add(domain_id)
                    entry.definitions.append(defn)
                    entry.synonyms|=(syns-{lname})
                    entry.antonyms|=ant_names
                    entry.hypernyms|=hypers
                    entry.hyponyms|=hypos
                    entry.meronyms|=meros
                    entry.holonyms|=holos
                    entry.synonyms|=also
                    wn_obj=lexicon.word_nonces.get(lname)
                    if wn_obj:
                        wn_obj.synset_ids.append(synset.name())
                        wn_obj.pos_id=entry.primary_pos()
                        wn_obj.domain_id=entry.primary_domain()
                        wn_obj.vector=encode_structured_nonce(word_to_hash_vector(lname,lexicon.dim),wn_obj.pos_id,wn_obj.domain_id,wn_obj.freq)
                continue
            seen_words.add(lname)
            entry=LexEntry(lname)
            entry.definitions.append(defn)
            entry.pos_tags.add(pos_id)
            entry.domains.add(domain_id)
            entry.synonyms=(syns-{lname})|also
            entry.antonyms=ant_names
            entry.hypernyms=hypers
            entry.hyponyms=hypos
            entry.meronyms=meros
            entry.holonyms=holos
            wn_obj=lexicon.add_entry(entry)
            wn_obj.synset_ids.append(synset.name())
            count+=1
        if max_words>0 and count>=max_words: break
    return count
def ingest_corpus_frequencies(lexicon:Lexicon,corpus_words:List[str])->None:
    freq_map=defaultdict(int)
    for w in corpus_words:
        freq_map[w.lower().strip()]+=1
    total=len(corpus_words)
    for w,cnt in freq_map.items():
        wn=lexicon.lookup(w)
        if wn:
            freq_ppm=cnt/total*1e6
            wn.freq=freq_ppm
            entry=lexicon.entries.get(w)
            if entry: entry.freq=freq_ppm
            wn.vector=encode_structured_nonce(word_to_hash_vector(w,lexicon.dim),wn.pos_id,wn.domain_id,wn.freq)
def build_co_occurrence(lexicon:Lexicon,sentences:List[List[str]],window:int=5,max_cooc_per_word:int=50)->Dict[int,Dict[int,float]]:
    cooc_sparse:Dict[int,Dict[int,float]]=defaultdict(lambda:defaultdict(float))
    for si,sent in enumerate(sentences):
        ids=[]
        for w in sent:
            wn=lexicon.lookup(w.lower().strip())
            if wn: ids.append(wn.nonce_id)
        for i,a in enumerate(ids):
            for j in range(max(0,i-window),min(len(ids),i+window+1)):
                if i!=j:
                    cooc_sparse[a][ids[j]]+=1.0/abs(i-j)
        if (si+1)%20000==0: print(f"    co-occurrence: {si+1}/{len(sentences)} sentences")
    for nid in cooc_sparse:
        total=sum(cooc_sparse[nid].values())
        if total>0:
            cooc_sparse[nid]={k:v/total for k,v in sorted(cooc_sparse[nid].items(),key=lambda x:-x[1])[:max_cooc_per_word]}
    print(f"  co-occurrence: {len(cooc_sparse)} words with neighbors")
    return dict(cooc_sparse)
class TextureMap:
    __slots__=('domain_id','domain_name','nonce_ids','vectors','path')
    def __init__(self,domain_id:int,domain_name:str,nonce_ids:List[int],vectors:np.ndarray,path:str=""):
        self.domain_id=domain_id
        self.domain_name=domain_name
        self.nonce_ids=nonce_ids
        self.vectors=vectors
        self.path=path
    def save(self,dir_path:str):
        p=Path(dir_path)
        p.mkdir(parents=True,exist_ok=True)
        fname=f"texture_{self.domain_name}.bin"
        self.path=str(p/fname)
        with open(self.path,'wb') as f:
            np.array([self.domain_id,len(self.nonce_ids),self.vectors.shape[1]],dtype=np.int32).tofile(f)
            np.array(self.nonce_ids,dtype=np.int32).tofile(f)
            self.vectors.astype(np.float16).tofile(f)
    @classmethod
    def load(cls,path:str)->'TextureMap':
        with open(path,'rb') as f:
            h=np.fromfile(f,dtype=np.int32,count=3)
            dom_id,n_nonces,dim=int(h[0]),int(h[1]),int(h[2])
            nonce_ids=np.fromfile(f,dtype=np.int32,count=n_nonces).tolist()
            vecs=np.fromfile(f,dtype=np.float16,count=n_nonces*dim).reshape(n_nonces,dim).astype(np.float32)
        return cls(dom_id,DOMAIN_NAMES.get(dom_id,"unknown"),nonce_ids,vecs,path)
def build_texture_maps(lexicon:Lexicon,out_dir:str)->Dict[int,TextureMap]:
    maps={}
    domain_groups=defaultdict(list)
    for w,wn in lexicon.word_nonces.items():
        domain_groups[wn.domain_id].append(wn)
    for dom_id,nonces in domain_groups.items():
        nonce_ids=[wn.nonce_id for wn in nonces]
        vecs=np.stack([wn.vector for wn in nonces])
        tm=TextureMap(dom_id,DOMAIN_NAMES.get(dom_id,"unknown"),nonce_ids,vecs)
        tm.save(out_dir)
        maps[dom_id]=tm
        print(f"  texture [{DOMAIN_NAMES.get(dom_id,'?')}]: {len(nonce_ids)} nonces -> {tm.path}")
    return maps
class DomainExpert:
    __slots__=('domain_id','domain_name','texture','active')
    def __init__(self,domain_id:int,texture:Optional[TextureMap]=None):
        self.domain_id=domain_id
        self.domain_name=DOMAIN_NAMES.get(domain_id,"unknown")
        self.texture=texture
        self.active=False
    def activate(self,texture:TextureMap):
        self.texture=texture
        self.active=True
    def deactivate(self):
        self.texture=None
        self.active=False
    def lookup_nonces(self,query_vec:np.ndarray,top_k:int=10)->List[Tuple[int,float]]:
        if not self.active or self.texture is None: return []
        sims=self.texture.vectors@query_vec
        top_idx=np.argsort(-sims)[:top_k]
        return [(self.texture.nonce_ids[i],float(sims[i])) for i in top_idx]
class CreativeExpert:
    __slots__=('role','delta_weights','inference_cache','learn_rate')
    def __init__(self,role:str,learn_rate:float=0.01):
        self.role=role
        self.delta_weights:Dict[int,np.ndarray]={}
        self.inference_cache:Dict[str,np.ndarray]={}
        self.learn_rate=learn_rate
    def infer(self,context_vec:np.ndarray,known_nonces:List[Tuple[int,float]],lexicon:Lexicon)->np.ndarray:
        if self.role=="inferrer":
            return self._logical_chain(context_vec,known_nonces,lexicon)
        elif self.role=="generator":
            return self._novel_combine(context_vec,known_nonces,lexicon)
        return self._document(context_vec,known_nonces,lexicon)
    def _logical_chain(self,ctx:np.ndarray,nonces:List[Tuple[int,float]],lex:Lexicon)->np.ndarray:
        result=np.zeros_like(ctx)
        for nid,score in nonces:
            wn=lex.lookup_id(nid)
            if not wn: continue
            entry=lex.entries.get(wn.word)
            if not entry: continue
            for hyper in entry.hypernyms:
                hwn=lex.lookup(hyper)
                if hwn: result+=hwn.vector*score*0.5
            for hypo in entry.hyponyms:
                hwn=lex.lookup(hypo)
                if hwn: result+=hwn.vector*score*0.3
            if nid in self.delta_weights:
                result+=self.delta_weights[nid]*score
        norm=np.linalg.norm(result).clip(min=1e-8)
        return result/norm
    def _novel_combine(self,ctx:np.ndarray,nonces:List[Tuple[int,float]],lex:Lexicon)->np.ndarray:
        if len(nonces)<2: return ctx
        vecs=[]
        for nid,score in nonces[:5]:
            wn=lex.lookup_id(nid)
            if wn: vecs.append(wn.vector*score)
        if not vecs: return ctx
        combined=np.mean(vecs,axis=0)
        spread=np.std(vecs,axis=0) if len(vecs)>1 else np.zeros_like(combined)
        result=combined+spread*0.1
        if any(nid in self.delta_weights for nid,_ in nonces):
            for nid,score in nonces:
                if nid in self.delta_weights:
                    result+=self.delta_weights[nid]*score*0.5
        norm=np.linalg.norm(result).clip(min=1e-8)
        return result/norm
    def _document(self,ctx:np.ndarray,nonces:List[Tuple[int,float]],lex:Lexicon)->np.ndarray:
        for nid,score in nonces:
            if score>0.5:
                delta=ctx*self.learn_rate
                self.delta_weights[nid]=self.delta_weights.get(nid,np.zeros_like(ctx))+delta
        return ctx
    def save_deltas(self,path:str):
        if not self.delta_weights: return
        p=Path(path)
        p.parent.mkdir(parents=True,exist_ok=True)
        with open(str(p),'wb') as f:
            np.array([len(self.delta_weights)],dtype=np.int32).tofile(f)
            for nid,dw in self.delta_weights.items():
                np.array([nid,len(dw)],dtype=np.int32).tofile(f)
                dw.astype(np.float16).tofile(f)
    def load_deltas(self,path:str):
        if not Path(path).exists(): return
        with open(path,'rb') as f:
            n=int(np.fromfile(f,dtype=np.int32,count=1)[0])
            for _ in range(n):
                h=np.fromfile(f,dtype=np.int32,count=2)
                nid,dim=int(h[0]),int(h[1])
                self.delta_weights[nid]=np.fromfile(f,dtype=np.float16,count=dim).astype(np.float32)
class ExpertTeam:
    def __init__(self,lexicon:Lexicon):
        self.lexicon=lexicon
        self.domain_experts:Dict[int,DomainExpert]={did:DomainExpert(did) for did in DOMAIN_MAP.values()}
        self.creative_experts=[CreativeExpert("inferrer"),CreativeExpert("generator"),CreativeExpert("documenter")]
        self.texture_maps:Dict[int,TextureMap]={}
    def load_textures(self,texture_dir:str):
        p=Path(texture_dir)
        for f in p.glob("texture_*.bin"):
            tm=TextureMap.load(str(f))
            self.texture_maps[tm.domain_id]=tm
            if tm.domain_id in self.domain_experts:
                self.domain_experts[tm.domain_id].activate(tm)
        print(f"loaded {len(self.texture_maps)} texture maps")
    def detect_domains(self,words:List[str],top_k:int=3)->List[int]:
        domain_scores=defaultdict(float)
        for w in words:
            wn=self.lexicon.lookup(w)
            if wn: domain_scores[wn.domain_id]+=1.0
        if not domain_scores: return [DOMAIN_MAP["general"]]
        sorted_doms=sorted(domain_scores.items(),key=lambda x:-x[1])
        return [d for d,_ in sorted_doms[:top_k]]
    def query(self,context_vec:np.ndarray,words:List[str],top_k:int=20)->Tuple[np.ndarray,List[Tuple[int,float]]]:
        active_domains=self.detect_domains(words)
        all_nonces=[]
        for did in active_domains:
            expert=self.domain_experts.get(did)
            if expert and expert.active:
                all_nonces.extend(expert.lookup_nonces(context_vec,top_k=top_k//len(active_domains)))
        all_nonces.sort(key=lambda x:-x[1])
        all_nonces=all_nonces[:top_k]
        inf_vec=self.creative_experts[0].infer(context_vec,all_nonces,self.lexicon)
        gen_vec=self.creative_experts[1].infer(context_vec,all_nonces,self.lexicon)
        self.creative_experts[2].infer(context_vec,all_nonces,self.lexicon)
        result=context_vec*0.4+inf_vec*0.3+gen_vec*0.3
        norm=np.linalg.norm(result).clip(min=1e-8)
        return result/norm,all_nonces
    def save(self,path:str):
        p=Path(path)
        p.mkdir(parents=True,exist_ok=True)
        for i,ce in enumerate(self.creative_experts):
            ce.save_deltas(str(p/f"creative_{ce.role}_deltas.bin"))
        meta={"n_domain_experts":len(self.domain_experts),"n_creative_experts":len(self.creative_experts),"active_domains":[did for did,de in self.domain_experts.items() if de.active],"creative_roles":[ce.role for ce in self.creative_experts]}
        with open(str(p/"expert_team_meta.json"),'w') as f: _json.dump(meta,f)
    def load(self,path:str):
        p=Path(path)
        for ce in self.creative_experts:
            dp=p/f"creative_{ce.role}_deltas.bin"
            if dp.exists(): ce.load_deltas(str(dp))

import os,re,time,zlib,struct
import numpy as np
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
from amni.serve import user_memory as U
WS=('what','where','when','why','how');AT=('','name','color','model','job','breed','kind','favorite','age','home','work','location','origin','destination','likes','allergy','birthday');WI={w:i for i,w in enumerate(WS)};AI={a:i for i,a in enumerate(AT)}
MAGIC=b'AMTO';VER=1;RW=48;UW=32;NK=14;TM=4913;HDR=struct.Struct('<4sHHHIII')
ZD=b" actually sorry wait now next last every morning evening night week month year monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december because so that to be for with by from the in on at and my is was it's he she they his her their our dog cat car bike house apartment flight appointment meeting birthday name called named lives live moved works work job drives parked keep favorite color blue red green black white sister brother mom dad wife husband son daughter friend best boyfriend girlfriend partner cousin aunt uncle grandma grandpa boss coworker neighbor roommate I'm I've I "
S_=lambda s:frozenset(s.split())
VB=S_("locate base be have do go get make take see come know think look want give use find tell ask work seem feel try leave call keep let begin help talk turn start show hear play run move live believe hold bring happen write sit stand lose pay meet include continue set learn change lead understand watch follow stop create speak read spend grow open walk win offer remember love consider appear buy wait serve send expect build stay fall cut reach remain suggest raise pass sell require report decide pull save visit catch cook bake eat drink drive ride fly travel study teach sing dance draw paint swim bike cycle hike fish shop clean fix repair rent own wear sleep rest relax celebrate attend join marry adopt train practice prepare plan pick drop park store hide put hang care miss enjoy like hate prefer need afford avoid improve earn commute jog skate ski surf climb camp garden sew knit film record type text email mail order deliver ship pack book reserve cancel renew replace upgrade trade lend borrow host throw support feed impress surprise treat protect escape retire quit graduate major relocate reside head return arrive land depart schedule reschedule postpone name adore dislike detest volunteer coach babysit tutor manage check carpool bus transfer switch sign enroll apply register donate collect listen dive row sail kayak paddle wash polish charge plug lock unlock share waste grab brew roast grill fry boil steam blend mix knead chop meditate pray exercise stretch lift fit stash tuck keep bury hang leave lock order pay renovate paint shower feed walk groom brush")
NOUNISH=S_("work visit call date plan trip train ride show run practice lesson race drive cook dance book order party check flight text email mail record film garden camp host coach tutor treat support type fish paint drink set change turn store park name hike swim shop ship pick lift walk bus rest sleep dive sail row charge lock plug share exercise stretch roast grill boil mix blend love care schedule return transfer surf trade appointment meeting spot")
IRR={'went':'go','gone':'go','got':'get','gotten':'get','bought':'buy','brought':'bring','kept':'keep','left':'leave','ran':'run','took':'take','taken':'take','taught':'teach','sold':'sell','made':'make','began':'begin','drove':'drive','driven':'drive','flew':'fly','flown':'fly','rode':'ride','ridden':'ride','ate':'eat','drank':'drink','met':'meet','paid':'pay','said':'say','saw':'see','seen':'see','found':'find','thought':'think','told':'tell','gave':'give','given':'give','came':'come','wrote':'write','written':'write','sat':'sit','stood':'stand','lost':'lose','spent':'spend','grew':'grow','won':'win','built':'build','fell':'fall','held':'hold','heard':'hear','felt':'feel','sent':'send','slept':'sleep','swam':'swim','sang':'sing','threw':'throw','wore':'wear','lent':'lend','hung':'hang','caught':'catch','chose':'choose','broke':'break','forgot':'forget','hid':'hide','hidden':'hide','located':'locate','situated':'locate','based':'base','is':'be','are':'be','was':'be','were':'be','am':'be',"'s":'be',"'m":'be',"'re":'be','been':'be','has':'have','had':'have','does':'do','did':'do','done':'do','leaves':'leave','lives':'live','studies':'study','flies':'fly','carries':'carry','tries':'try','marries':'marry','married':'marry','studied':'study','tried':'try','majored':'major','traveled':'travel','travelled':'travel','commutes':'commute'}
CLASS={'move':'live','relocate':'live','reside':'live','stay':'live','base':'live','fly':'go','travel':'go','head':'go','commute':'go','get':'go','return':'go','depart':'go','arrive':'go','ride':'drive','park':'keep','store':'keep','hide':'keep','put':'keep','stash':'keep','tuck':'keep','hang':'keep','call':'name','name':'name','leave':'keep','reschedule':'be','schedule':'be','postpone':'be','adore':'like','love':'like','enjoy':'like','major':'study'}
MOTION=S_("go move walk bike cycle drive ride fly run commute head travel get come return relocate carpool jog skate swim bus hike sail row transfer switch take")
MODE=S_("walk bike cycle drive ride jog skate swim carpool bus run fly")
CATEN=S_("want need have get plan hope try like love decide forget remember use learn start begin continue agree promise refuse seem appear expect afford manage fail tend wish choose prefer hate able about supposed gonna wanna ought going used")
MEANS=S_("bus train car bike bicycle plane ferry boat subway metro tram taxi cab uber lyft scooter motorcycle motorbike foot hand phone mail email text post check cheque cash card transfer app zoom skype video carpool rail air sea road coach shuttle helicopter truck van skateboard skates bank venmo paypal")
PLACEN=S_("level floor spot space row section gate terminal platform lot garage basement attic kitchen office gym school work home house apartment flat room bedroom bathroom closet drawer cabinet shelf desk table counter fridge freezer pantry shed yard garden porch driveway lot deck cafe restaurant bar pub park library hospital clinic airport station church college university campus downtown city town village suburbs countryside beach lake mountains farm studio store shop mall market bank hotel street avenue road building center centre warehouse factory lab bakery salon museum theater theatre stadium arena pool court field cottage cabin condo neighborhood trunk glovebox bag backpack purse wallet pocket box locker safe case jar sink couch sofa bed nightstand dresser hallway hall entryway door window corner coast state country abroad overseas upstairs downstairs outside inside island barn cellar loft porch balcony patio rack hook bin crate tin pouch folder binder envelope vault headquarters branch site plant clubhouse dorm")
BAREW=S_("home downtown abroad overseas upstairs downstairs nearby uptown offshore")
DET=S_("the a an my our his her their this that these those your its some")
PREPW=S_("in at to from near inside outside behind under into onto on by across off beside beneath above below between around within")
ADV=S_("right originally initially really also just still now finally usually always often recently currently actually never sometimes only even already mostly normally typically generally totally definitely probably apparently basically honestly literally seriously officially")
AUX=S_("is are am was were 's 'm 're be been being will 'll would 'd has have had 've do does did not n't don't doesn't didn't isn't wasn't aren't won't can can't could should might must may gonna wanna")
DAYS=S_("monday tuesday wednesday thursday friday saturday sunday mondays tuesdays wednesdays thursdays fridays saturdays sundays tues thurs")
MONS=S_("january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec")
TW=DAYS|MONS|S_("today tomorrow tonight yesterday weekend weekends weekday weekdays morning mornings afternoon afternoons evening evenings night nights noon midnight week weeks month months year years summer winter spring fall autumn christmas easter thanksgiving halloween holidays holiday am pm a.m p.m o'clock ago later soon weekly daily annually monthly nightly hours hour minutes minute days day decade dawn dusk sunrise sunset lunchtime")
TGL=S_("all whole these end beginning middle start next this last every each on at in by until till since before after around about from the of a an early late mid other coming following to and or first second third fourth fifth")
NUMW=S_("one two three four five six seven eight nine ten eleven twelve twenty thirty half couple few")
ATTRN={'name':('what','name'),'color':('what','color'),'colour':('what','color'),'job':('what','job'),'occupation':('what','job'),'profession':('what','job'),'breed':('what','breed'),'age':('what','age'),'model':('what','model'),'make':('what','model'),'brand':('what','model'),'address':('where','home'),'time':('when',''),'date':('when',''),'day':('when',''),'location':('where','location'),'reason':('why',''),'favorite':('what','favorite'),'birthday':('when','birthday')}
ADJ=S_("new old little big older younger baby current first second third next last upcoming usual regular new-ish lovely dear sweet whole same")
PRONS={'he','she','they','it','him','her','them'}
_TOK=re.compile(r"\d{1,2}:\d{2}|\d+(?:st|nd|rd|th)?|[A-Za-z]+(?:[.'-][A-Za-z]+)*\.?|'[a-z]+|[^\w\s]")
_CT=re.compile(r"(?i)\b(i|you|he|she|it|we|they|that|there|what|where|who|when|why|how)'(s|m|re|ve|ll|d)\b")
_NT=re.compile(r"(?i)\b(do|does|did|is|was|are|were|ca|wo|could|would|should|have|has|had)n't\b")
_LEAD=re.compile(r"(?i)^\W*(?:(?:ugh|argh|dang|darn|shoot|welp|oh no|actually|oh|so|well|also|and|but|btw|by the way|fun fact|update|correction|sorry|wait|oops|ok(?:ay)?|hey|hi|um|uh|yeah|yes|no|nope|right|scratch that|i misspoke|i meant|i mean|my (?:bad|mistake)|i got that wrong|i was wrong|just so you know|fyi|heads up|anyway|hmm|ah|guess what|quick update|small correction|change of plans?|not anymore|new info)\b[\s,:;.!-]*)+")
_CORR=re.compile(r"(?i)\b(?:wait|misread|mixed (?:it|that|them) up|typo|i meant|make that|make it|turns out|actually|correction|sorry|oops|scratch that|misspoke|my (?:bad|mistake)|got that wrong|was wrong|wrong|not anymore|any ?more|no longer|changed|switched|instead|update|rescheduled|postponed|moved|now)\b")
_NOT=re.compile(r"(?i)^\s*not\s+[^,]+,\s*|,?\s*\bnot\s+(?:the\s+|a\s+|an\s+|on\s+|in\s+|at\s+)?[\w' -]+?\s*(?=$|[.!?])")
_QPRE=re.compile(r"(?i)^\W*(?:(?:hey|so|ok(?:ay)?|um|uh|and|also|wait|quick question|adam|btw|sorry|hmm|oh|quick one)\W+)*(?:(?:can|could|would) you (?:please )?(?:tell me|remind me|say)(?: again)?\W*|(?:do|did) you (?:still )?(?:remember|recall|know)\W*|remind me(?: again)?\W*|tell me(?: again)?\W*|i forgot\W*|i forget\W*)?")
_QSAY=re.compile(r"(?i)\b(?:did|do) i (?:say|tell you|mention|say that|tell you that)(?: that)?\b|\bi (?:told|said to) you\b|\bdid i\b(?= (?:say|tell))|\bagain\b|\bexactly\b|\bthat\b(?=\s*\?)")
def h32(s:str)->int:return U.bh(s.lower().encode('utf-8','replace'))
def lemma(w:str)->str:
    w=w.lower().strip(".'")
    if w in IRR:return IRR[w]
    if w in VB:return w
    for c in ([w[:-3]+'y'] if w.endswith(('ies','ied')) else [])+([w[:-3],w[:-3]+'e',w[:-4]] if w.endswith('ing') else [])+([w[:-2],w[:-1],w[:-3]] if w.endswith('ed') else [])+([w[:-2],w[:-1]] if w.endswith('es') else [w[:-1]] if w.endswith('s') else []):
        if c in VB:return c
    return w
isv0=lambda t:t in IRR and IRR[t] not in ('be','have','do') or lemma(t) in VB
isv=lambda t:t not in NOUNISH and t not in DET and (t in IRR and IRR[t] not in ('be','have','do') or lemma(t) in VB and (t in VB or t.endswith(('s','ed','ing'))))
pcls=lambda p:CLASS.get(p,p)
iscol=lambda t:U.GX.get(U.bh(t.lower().encode())&U.WM,(0,0))[0]==2048
isper=lambda t:bool(U.CLS.get(U.bh(t.lower().encode())&U.WM,0)&32) or t.lower() in ('kid','kids','child','baby','fiance','fiancee','husband','wife','teacher','doctor','dentist','therapist','landlady','grandmother','grandfather','stepson','stepdaughter','twin','bestie','colleague','sibling','parents')
def toks(s:str):
    s=_NT.sub(lambda m:m[1]+' not',_CT.sub(lambda m:m[1]+" '"+m[2].lower(),s));return s,[(m[0],m.start(),m.end()) for m in _TOK.finditer(s)]
def _istime(T,i):
    t=T[i][0].lower().rstrip('.')
    return (t in TW and (t!='may' or T[i][0][0]=='M')) or bool(re.fullmatch(r"(?:1[89]|20)\d\d|\d{1,2}:\d{2}",t)) or (bool(re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)",t)) and (i>0 and T[i-1][0].lower().rstrip('.') in MONS|{'the'} or i+1<len(T) and T[i+1][0].lower().rstrip('.') in MONS|{'of'})) or (bool(re.fullmatch(r"\d{1,2}",t)) and (i+1<len(T) and T[i+1][0].lower().rstrip('.') in ('am','pm','a.m','p.m',"o'clock") or i>0 and T[i-1][0].lower() in MONS or i>0 and T[i-1][0].lower() in ('at','around','by','until','till') and 0<int(t)<=12)) or (t in NUMW|{'a','couple'} or t.isdigit()) and i+1<len(T) and T[i+1][0].lower() in TW
def tspan(T,i,n):
    if not (_istime(T,i) or T[i][0].lower() in TGL-{'a','an','of','to','and','or'}):return None
    j=i;last=-1
    while j<n and (_istime(T,j) or T[j][0].lower() in TGL|NUMW or T[j][0].isdigit()):
        last=j if _istime(T,j) else last;j+=1
        if j-i>9:break
    if last<0 or last-i>5 and not any(_istime(T,k) for k in range(i,i+3)):return None
    k=last+1
    if k<n and T[k][0].lower()=='old':return None
    while k<n and T[k][0].lower() in ('ago','later'):k+=1
    return (i,k)
FAV={'favourite':'favorite','fave':'favorite','go-to':'favorite','fav':'favorite'}
NPSTOP=S_("and or but because so who which that where when why how what if as than then too again named called due not")
def _np(T,i,n,chain=True):
    j=i;lab=[];name=[];attr=None
    while j<n and T[j][0].lower() in ADJ:j+=1
    while j<n and len(lab)<4:
        t=T[j][0];tl=FAV.get(t.lower().rstrip('.'),t.lower().rstrip('.'))
        if not re.fullmatch(r"[A-Za-z][\w'-]*\.?",t) or tl in AUX or tl in PREPW or tl in DET or tl in ADV or tl in NPSTOP or lab and tl in BAREW:break
        if lab and t[0].isupper() and (isper(lab[-1]) or lab[-1] in ('friend','partner','pet','puppy','kitten')):
            while j<n and T[j][0][:1].isupper() and T[j][0].lower() not in AUX:name.append(T[j][0].rstrip('.'));j+=1
            break
        if lab and tl not in ATTRN and not lab[-1].endswith("'s") and (isv(tl) or tl in NOUNISH and isv0(tl) and (isper(lab[-1]) or j+1>=n or T[j+1][0].lower() in PREPW|{'?','.','!',','})):break
        if tl.endswith("'s") and chain:
            nx=T[j+1][0].lower() if j+1<n else ''
            if nx and nx not in DET and nx not in PREPW and nx not in AUX and nx not in ADV and not iscol(nx) and not isv(nx) and nx not in ('not','called','named','so','very','gonna','going','due') and re.fullmatch(r"[a-z][\w-]*",nx):
                lab.append(tl[:-2]+"'s");j+=1;continue
            lab.append(tl[:-2]);T[j]=("'s",T[j][1]+len(t)-2,T[j][2]);break
        lab.append(tl);j+=1
    if lab and lab[-1] in ATTRN and not (len(lab)==2 and lab[0]=='favorite') and lab!=['favorite']:
        attr=ATTRN[lab[-1]];lab=lab[:-1]
    lab=[x[:-2] if x.endswith("'s") and k==len(lab)-1 else x for k,x in enumerate(lab)]
    return ' '.join(lab),name,attr,j
class Store:
    def __init__(s,sl=None,ut=None,head=0):s.s=sl if sl is not None else np.zeros((0,RW),np.uint16);s.u=ut if ut is not None else np.zeros((0,UW),np.uint16);s.head=head;s.names={};s._x=None;s.lp=s.lo=''
    def ents(s)->dict:
        d={}
        for r in s.s:
            l,_,v=s.text(r);d.setdefault(l,int(r[0])|int(r[1])<<16)
            (int(r[3])%17)==AI['name'] and int(r[3])//17==0 and s.names.__setitem__(v.lower(),l)
        return d
    @staticmethod
    def text(r)->tuple:
        b=r[8:].tobytes()[:int(r[6])].decode('utf-8','replace').split('\x1f');return tuple((b+['','',''])[:3])
    def age(s,c):return (s.head%TM-np.asarray(c,np.int64))%TM
    def put(s,ent,verb,w,a,val,conf,src=0):
        tb=ent.encode()[:28]+b'\x1f'+verb.encode()[:12]+b'\x1f';tb=(tb+val.encode())[:80];r=np.zeros(RW,np.uint16);e=h32(ent);r[0]=e&0xFFFF;r[1]=e>>16;r[2]=h32(pcls(lemma(verb)) if verb else '')&0xFFFF;r[3]=WI[w]*17+AI.get(a,0);r[4]=s.head%TM;r[7]=h32(lemma(verb) if verb else '')&0xFFFF;r[5]=min(16,max(0,round(conf*16)))+17*src;r[6]=len(tb);r[8:8+(len(tb)+1)//2]=np.frombuffer(tb+b'\0'*(len(tb)&1),'<u2');s.s=np.vstack([s.s,r[None]])
    def putu(s,text,keys):
        z=zlib.compressobj(9,zlib.DEFLATED,-15,9,zlib.Z_DEFAULT_STRATEGY,ZD);b=z.compress(text.encode('utf-8'))+z.flush();ks=list(dict.fromkeys(keys))[:NK];p=np.frombuffer(b+b'\0'*(len(b)&1),'<u2');nr=-(-len(p)//UW);hd=np.zeros(UW,np.uint16);hd[0]=nr;hd[1]=s.head%TM;hd[2]=len(ks);hd[3]=len(b)
        for i,k in enumerate(ks):hd[4+2*i]=k&0xFFFF;hd[5+2*i]=k>>16
        pr=np.zeros((nr,UW),np.uint16);pr.reshape(-1)[:len(p)]=p;s.u=np.vstack([s.u,hd[None],pr]);s._x=None
    def uidx(s):
        if s._x is None:
            st=[];i=0
            while i<len(s.u):st.append(i);i+=1+int(s.u[i,0])
            st=np.array(st,np.int64);H=s.u[st] if len(st) else np.zeros((0,UW),np.uint16);K=(H[:,4:4+2*NK:2].astype(np.uint32)|H[:,5:5+2*NK:2].astype(np.uint32)<<16) if len(st) else np.zeros((0,NK),np.uint32);K[np.arange(NK)[None,:]>=H[:,2:3].astype(np.int64)]=0;s._x=(st,K)
        return s._x
    def utext(s,i)->str:
        st,_=s.uidx();o=int(st[i]);hd=s.u[o];d=zlib.decompressobj(-15,ZD);return (d.decompress(s.u[o+1:o+1+int(hd[0])].reshape(-1).tobytes()[:int(hd[3])])+d.flush()).decode('utf-8','replace')
    def udrop(s,ix):
        st,_=s.uidx();keep=np.ones(len(s.u),bool)
        for i in ix:o=int(st[i]);keep[o:o+1+int(s.u[o,0])]=False
        s.u=s.u[keep];s._x=None
    def rebase(s):
        if not len(s.s) and not len(s.u):s.head=0;return
        st,_=s.uidx();c=np.concatenate([s.s[:,4],s.u[st,1]]).astype(np.int64);a=s.age(c)
        if a.max()<TM//2:return
        o=np.argsort(-a,kind='stable');r=np.empty(len(a),np.int64);r[o]=np.arange(len(a));_,inv=np.unique(a,return_inverse=True);nc=(s.head-inv)%TM
        s.s[:,4]=nc[:len(s.s)];s.u[st,1]=nc[len(s.s):]
def _pk(a:np.ndarray)->np.ndarray:return encode_fp16_to_rgba4(np.ascontiguousarray(a,np.uint16).reshape(-1).view(np.float16))
def encode(st:Store)->bytes:
    st.rebase();pay=np.concatenate([_pk(st.s).reshape(-1),_pk(st.u).reshape(-1)]).tobytes();h=HDR.pack(MAGIC,VER,RW,UW,len(st.s),len(st.u),st.head)
    return h+struct.pack('<II',zlib.crc32(h),zlib.crc32(pay))+pay
def decode(raw:bytes)->Store:
    mg,ver,rw,uw,ns,nu,head=HDR.unpack_from(raw);c1,c2=struct.unpack_from('<II',raw,HDR.size);pay=raw[HDR.size+8:]
    if mg!=MAGIC or ver!=VER or rw!=RW or uw!=UW or c1!=zlib.crc32(raw[:HDR.size]) or c2!=zlib.crc32(pay):raise ValueError('toroid store corrupt')
    d=np.frombuffer(pay,np.uint8).reshape(-1,4)
    if (d>=17).any():raise ValueError('toroid page corrupt')
    u=decode_rgba4_to_fp16(d).view(np.uint16);return Store(u[:ns*RW].reshape(ns,RW).copy(),u[ns*RW:].reshape(nu,UW).copy(),head)
_cache={}
def enabled()->bool:return os.environ.get('AMNI_UMEM_TOROID','0')=='1' and U.enabled()
def path(key:str)->str:return os.path.join(U.root(),re.sub(r'[^0-9a-f]','',key)[:32]+'.torus')
def load(key:str)->Store:
    p=path(key)
    with U._lock:
        st=os.stat(p).st_mtime_ns if os.path.exists(p) else None;c=_cache.get(key)
        if c and c[0]==st:return decode(c[1])
        raw=open(p,'rb').read() if st else None;_cache[key]=(st,raw) if raw else (st,encode(Store()));return decode(raw) if raw else Store()
def save(key:str,st:Store):
    p=path(key)
    with U._lock:
        os.makedirs(U.root(),mode=0o700,exist_ok=True);_cache.pop(key,None)
        if not len(st.s) and not len(st.u):os.path.exists(p) and os.remove(p);return
        t=p+f'.{os.getpid()}.tmp';fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'wb') as fh:fh.write(encode(st));fh.flush();os.fsync(fh.fileno())
        os.replace(t,p)
_sg=lambda w:w[:-1] if len(w)>3 and w.endswith('s') and not w.endswith(('ss','us','is')) else w
def _resolve(lab:str,known)->str:
    if not lab or lab=='me' or lab in known:return lab
    L=lab.split();tl={_sg(x) for x in L};c=[k for k in known if k!='me' and (_sg(k.split()[-1])==_sg(L[-1]) and tl<={_sg(x) for x in k.split()} or k.startswith(lab+' ') or k.startswith(lab+"'s "))]
    return c[-1] if c else lab
REPV=S_("say show confirm mean read list");COOKV=S_("cook make bake brew fry grill roast steam boil prepare");PART=S_("up out off down back over away around")
PURP=S_("health safety sake job work fun business school money love exercise research career studies family kids savings retirement budget relaxation comfort convenience privacy emergencies emergency reasons reason wedding birthday anniversary graduation funeral reunion conference interview meeting vacation holiday trip environment")
_FILL=re.compile(r"(?i)\b(?:after all|these days|nowadays|lately|right now|at the moment|for now|for sure|lol|haha|tbh)\b")
_REP=re.compile(r"(?i)^(?:(?:it\s+)?turns out|apparently|(?:the\s+)?[a-z]+(?:\s+[a-z]+)?\s+(?:says|said|confirmed|told me|thinks)|i (?:just )?(?:found out|heard|learned|realized|noticed|checked))(?:\s+that)?[\s,]+(?=(?:my|the|our|his|her|it|it's|he|she|they|i|we)\b)")
_SPL=re.compile(r"(?<=[.!?;])\s+|\s*;\s*|,?\s+(?:and|but|while|whereas|plus)\s+(?=(?:my|i|i'm|he|she|they|it|it's|we|our|his|her|their|the)\b)|,\s+(?=(?:[Mm]y|I|I'm|[Hh]e|[Ss]he|[Tt]hey|[Ww]e|[Ii]t|[Ii]t's|[Hh]is|[Hh]er|[Tt]heir|[Oo]ur)\b)|,\s+(?=(?:and\s+)?(?:[Tt]he\s+|[Mm]y\s+|[Oo]ur\s+)?[A-Za-z]+(?:\s+[a-z]+)?\s+(?:is|are|was|were)\b)")
def clauses(msg:str):
    m=msg.strip();m=_NOT.sub(' ',m) if re.search(r"(?i)\bnot\b",m) and (_CORR.search(m) or re.search(r"(?i),\s*not\b|^\W*(?:no\W+)?not\b",m)) else m;m=_FILL.sub(' ',m)
    for c in _SPL.split(m):
        c=_LEAD.sub('',c.strip())
        while (c2:=_REP.sub('',c))!=c:c=_LEAD.sub('',c2)
        c=re.sub(r"(?i)^(?:make (?:that|it)|change (?:that|it) to|i meant)\s+",'',c).strip(' ,.!-')
        c and (yield c)
_placey=lambda ow:bool(ow) and (ow[-1] in PLACEN or len(ow)>1 and ow[-2] in PLACEN and re.fullmatch(r"[a-z]|\d+",ow[-1]) is not None)
PETS=S_("dog cat puppy kitten pet horse rabbit parrot hamster ferret turtle gecko goldfish hedgehog iguana cockatiel chinchilla tortoise snake budgie pup")
def _vtype(ow,TT):
    o=[t for t in ow if t not in DET and t not in ADV]
    return None if not o else ('what','color') if all(iscol(t) or t in ('dark','light','bright','pale','deep') for t in o) and any(iscol(t) for t in o) else ('what','age') if re.fullmatch(r"(?:\d{1,3}|"+'|'.join(NUMW)+r")(?: (?:years?|months?|weeks?) old)?",' '.join(o)) else ('where','location') if _placey(o) else ('what','name') if len(o)<=2 and all(t[0][:1].isupper() for t in TT if t[0].lower() in o) else None
def _obj(T,lw,a,b,known,lo):
    b=next((q for q in range(a,b) if lw[q]==','),b)
    while a<b and lw[a] in ('for','into','about','on','to','with','and','of'):a+=1
    a0=a
    while a<b and (lw[a] in DET or lw[a] in ADJ):a+=1
    if a<b and lw[a] in ('it','them','that','this'):return lo if b-a==1 else ''
    if not 0<b-a<=4 or any(not re.fullmatch(r"[a-z][\w'-]*",x) or x in AUX or x in PREPW or x in PRONS or x in ADV or x in ('i','me','you','we','us','all','right','out','back') or a==a0 and isv(x) and not x.endswith('ing') or _istime(T,a+q) or iscol(x) and q<b-a-1 for q,x in enumerate(lw[a:b])):return ''
    return _resolve(' '.join(FAV.get(x,x) for x in lw[a:b]),known)
def analyze(c:str,last:str='',known=(),names=None,lp='',cor=False):
    c,T=toks(c);n=len(T);names=names or {};lw=[t[0].lower() for t in T]
    if not n:return None
    i=0;ent='';name=[];pa=None;weak=False
    if lw[0] in ('my','our'):
        ent,name,pa,i=_np(T,1,n);lw=[t[0].lower() for t in T];ent=ent or ('me' if pa else '')
        if ent and i<n and lw[i] in PREPW|{'with','for','of'}:
            q=i+1
            while q<n and q<i+6 and lw[q] not in AUX and not isv(lw[q]):q+=1
            i=q if q<n and q<i+6 and (lw[q] in AUX or isv0(lw[q])) else i
    elif lw[0] in ('his','her','their') and n>1:
        lab,name,pa,i=_np(T,1,n);lw=[t[0].lower() for t in T];b=lp or last;ent=(b if not lab else f"{b}'s {lab}") if b else lab
    elif lw[0] in ('i','we'):ent,i='me',1
    elif lw[0] in ('he','she','they'):ent,i=lp or last,1
    elif lw[0] in ('it','that','this'):ent,i=('?' if cor else last),1
    elif T[0][0][:1].isupper() and lw[0] in names:ent,i=names[lw[0]],1
    elif lw[0]=='the' or lw[0] not in ADV and lw[0] not in AUX and lw[0] not in PREPW and lw[0] not in TW and (U.bh(lw[0].encode())&U.WM) not in U.FUN and not isv(lw[0]) and lw[0] not in ('ugh','hmm','yeah','cool','thanks','nice','any','some','no','yes','ok','okay'):
        lab,_,p2,j=_np(T,1 if lw[0]=='the' else 0,n);lw=[t[0].lower() for t in T];jj=j
        if lab and jj<n and lw[jj] in PREPW|{'with','for','of'}:
            q=jj+1
            while q<n and q<jj+6 and lw[q] not in AUX and not isv(lw[q]):q+=1
            jj=q if q<n and q<jj+6 else jj
        if lab and jj<n and (lw[jj] in AUX or isv0(lw[jj])):ent,i,weak,pa=_resolve(lab,known),jj,_resolve(lab,known) not in known,p2
    elif isv(lw[0]) and lw[0].endswith('ing') or lw[0] in ('been','got','just'):ent='me'
    ent=_resolve(ent,known) if ent and ent!='?' else ent
    j=i;cop=False;hv=False
    while j<n:
        t=lw[j]
        if t in ('going','used') and j+2<n and lw[j+1]=='to' and lw[j+2] in VB and lw[j+2] not in NOUNISH:j+=2;continue
        if t in AUX or t in ADV or t=='gonna':cop=cop or lemma(t)=='be' or t in ("'s","'m","'re");hv=hv or lemma(t)=='have';j+=1;continue
        break
    verb=T[j][0] if j<n and isv0(lw[j]) and not (cop and lw[j] in ('called','named')) and not (cop and not lw[j].endswith(('ed','ing','en')) and lw[j] not in IRR) else ''
    pred=lemma(verb) if verb else 'be' if cop else 'have' if hv else ''
    k=j+1 if verb else j
    while verb and k<n and lw[k] in PART and not (k+1<n and lw[k+1] in DET and lw[k] in ('over','around')):k+=1
    subj=i>0 or ent=='me'
    if weak and pred in REPV:ent,weak,subj='?',False,False
    sp=[];x=k;sv=[];cv=verb
    while x<n:
        t=lw[x];nx=lw[x+1] if x+1<n else '';y=None
        if t=='and' and nx and (isv(nx) or nx in IRR and IRR[nx] not in ('be','have','do')) and x+1>k:cv=T[x+1][0];x+=2;continue
        if t in ('because','cause',"'cause",'coz','cuz') or t=='since' and x+1<n and not _istime(T,x+1) and not nx.isdigit() or t=='so' and nx in ('that','i','we','he','she','they','my','it','the','you','her','his') or t=='in' and nx=='order' or t=='to' and nx in VB and not (nx in NOUNISH|{'work','school','bed','church','class','practice','college','town','camp','market','lunch','dinner','breakfast','brunch','court'} and (lemma(lw[x-1]) in MOTION or nx in ('work','school','bed','church','class','college','town','market','lunch','dinner','breakfast','brunch','court','camp'))) and x>0 and lemma(lw[x-1]) not in CATEN and lw[x-1] not in CATEN and lw[x-1] not in DET or t=='for' and any(q in PURP for q in lw[x+1:x+4]) and not (x+1<n and _istime(T,x+1)):y=('why','',x,n)
        elif (tt:=tspan(T,x,n)):y=('when','',tt[0],tt[1])
        elif t=='by' and (e:=next((q for q in range(x+1,min(n,x+4)) if lw[q] in MEANS or lw[q].endswith('ing') and q==x+1),None)) is not None and all(lw[q] not in PREPW|AUX for q in range(x+1,e)):y=('how','',x,e+1)
        elif t=='on' and nx=='foot' or t=='from' and nx=='scratch' or t=='in' and nx in ('cash','person','bulk'):y=('how','',x,x+2)
        elif t=='on' and nx in DET and x+2<n and lw[x+2] in MEANS or t=='over' and nx=='the' and x+2<n and lw[x+2] in ('phone','internet'):y=('how','',x,x+3)
        elif t=='over' and nx in MEANS|{'zoom','facetime','whatsapp','slack','chat'}:y=('how','',x,x+2)
        elif t in ('online','via'):y=('how','',x,x+(2 if t=='via' else 1))
        elif t in ('with','using') and (nx in DET and x+2<n and not isper(lw[x+2]) and lw[x+2] not in PRONS or nx and nx not in DET and nx not in PRONS and not isper(nx) and re.fullmatch(r"[a-z][\w-]*",nx) and nx not in ('me','us','you')):
            z=x+(2 if nx in DET else 1);e=z
            while e<n and e<z+3 and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','because','so','to','every','each','on'} and not _istime(T,e):e+=1
            while e<n and e<z+4 and T[e][0][:1].isupper() and lw[e] not in AUX|{'i'}:e+=1
            y=('how','',x,e) if e>z else None
        elif t in PREPW or t in BAREW:
            z=x+(0 if t in BAREW else 1);dt=z<n and lw[z] in DET
            z+=dt;e=z
            if e<n and T[e][0][:1].isupper() and not _istime(T,e):
                while e<n and (T[e][0][:1].isupper() and lw[e] not in AUX|{'i'} and not _istime(T,e) or lw[e] in ('of','de','la','upon','on','&','del','van','von') and e+1<n and T[e+1][0][:1].isupper()):e+=1
            else:
                while e<n and e<z+4 and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|PRONS|{'and','or','because','so','to','every','i','me','you'} and not _istime(T,e) and not (isv(lw[e]) and not lw[e].endswith('ing')):
                    e+=1
                    if (lw[e-1] in PLACEN or lw[e-1] in BAREW) and not (dt and e<n and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|PRONS|TGL and not isv(lw[e])):break
                e=e if e>z and (lw[e-1] in PLACEN or lw[e-1] in BAREW or dt and (t!='to' or pcls(lemma(verb)) in ('go','live') or lemma(verb) in MOTION)) else z
            while e>z and e<n and e<z+6 and (re.fullmatch(r"[A-Z]|\d+",T[e][0]) and lw[e-1] in PLACEN):e+=1
            if e>z and e+1<n and lw[e]=='of':
                f=e+1+(lw[e+1] in DET)
                while f<n and f<e+5 and (T[f][0][:1].isupper() or re.fullmatch(r"[a-z][\w-]*",lw[f]) and lw[f] not in PREPW|AUX|DET|{'and','or','because','so','to'}) and not _istime(T,f):f+=1
                e=f if f>e+1+(lw[e+1] in DET) else e
            if e>z and e+2<n and lw[e] in ('near','by','behind','beside','next','across','under','off','on','in','at') and not _istime(T,e+1):
                f=e+1+(lw[e+1] in ('to','from'));g=f+(f<n and lw[f] in DET);h=g
                while h<n and h<g+3 and (T[h][0][:1].isupper() or re.fullmatch(r"[a-z][\w-]*",lw[h]) and lw[h] not in PREPW|AUX|DET|{'and','or','because','so','to'}) and not _istime(T,h) and not isv(lw[h]):h+=1
                e=h if h>g and f<n and lw[f] in DET|{'the'} or h>g and T[g][0][:1].isupper() else e
            y=('where','',x,e) if e>z or t in BAREW else None
        if y:
            sp.append(y);sv.append(cv);x=y[3]
            if y[0]=='why':break
        else:x+=1
    return dict(T=T,c=c,ent=ent,name=name,pa=pa,verb=verb,pred=pred,cop=cop,k=k,sp=sp,sv=sv,subj=subj,weak=weak)
def _val(c,T,a,b,w):
    while a<b and (T[a][0].lower() in (DET|ADV|{'as','is','be'}) and w=='what' or w in ('where','when') and T[a][0].lower() in PREPW|{'on','at','in'} and a+1<b):a+=1
    while b>a and T[b-1][0].lower() in ADV|{'too','again','though','lol','anymore','now','as','well','here','there','and','or','but','then','with','of','for','the','a','an',',','instead'}:b-=1
    return re.sub(r"\s+",' ',c[T[a][1]:T[b-1][2]]).strip(" ,.;:!?\"") if b>a else ''
def _wattr(w,a,pred,verb,lw,x):
    if w!='where':return w,a
    pc=pcls(pred);return ('how','') if pred in COOKV else (w,'home' if pc=='live' else 'work' if pc=='work' else 'origin' if lw[x]=='from' else 'destination' if pc=='go' or pred in MOTION and lw[x] in ('to','into') else 'location')
def extract(msg:str,last:str='',known=(),names=None,ls=('what',''),lp='',lo=''):
    names=dict(names or {});known=list(known);F=[];cor=bool(_CORR.search(msg)) or bool(re.search(r"(?i),\s*not\b|^\W*(?:no\W+)?not\b",msg))
    for c in clauses(msg):
        A=analyze(c,last,known,names,lp,cor)
        if not A:continue
        T,ent,verb,pred,sp,k,cc=A['T'],A['ent'],A['verb'],A['pred'],A['sp'],A['k'],A['c'];lw=[t[0].lower() for t in T];n=len(T)
        if not ent and (cor or not A['subj']) and last:ent='?' if cor else last
        if not ent or A['weak'] and not any(s[0]!='what' for s in sp) and not A['pa'] and not (A['cop'] and k<n and (lw[k] in ('a','an') or T[k][0][:1].isupper())):continue
        add=lambda e,v,w,a,val,cf:val and len(val)<=72 and e and F.append((e,v,w,a,val,cf,cor))
        A['name'] and add(ent,'','what','name',' '.join(A['name']),0.9)
        A['name'] and names.__setitem__(A['name'][0].lower(),ent)
        e0=sp[0][2] if sp else n;prev=k;obs=[]
        for w,a,x,y in sp:
            ob=_obj(T,lw,prev,x,known,lo) if pred not in ('be','') else '';ds=_obj(T,lw,x+1,y,known,'') if w=='where' and lw[x] in ('to','into') and (lemma(verb) in MOTION or pcls(lemma(verb))=='go') and not T[min(y-1,n-1)][0][:1].isupper() else ''
            obs.append(ob or (obs[-1] if obs else '') or ds);prev=y
        for (w,a,x,y),ob,vb in zip(sp,obs,A['sv']):
            v=_val(cc,T,x,y,w);pr_=lemma(vb) if vb else pred;w2,a2=_wattr(w,a,pr_,vb,lw,x);verb=vb
            if A['pa'] and A['pa'][0]==w2 and not ob:a2=A['pa'][1]
            add(ent,verb,w2,a2,v,0.8);ob and ob!=ent and ob!=v.lower() and add(ob,verb,w2,'location' if w2=='where' and pcls(pred) in ('keep','be') else a2,v,0.75)
            w2=='where' and a2=='location' and ent=='me' and pred=='park' and not ob and add('car',verb,'where','location',v,0.6)
        if pred in MODE and not any(s[0]=='how' for s in sp) and any(s[0]=='where' and lw[s[2]]=='to' for s in sp):add(ent,verb,'how','',verb,0.6)
        ob0=obs[0] if obs else (_obj(T,lw,k,n,known,lo) if pred not in ('be','') else '')
        cmp_=(k,e0) if e0>k else None
        if A['pa'] and A['pa'][0]!='what' and cmp_ and not sp:add(ent,verb,A['pa'][0],A['pa'][1],_val(cc,T,*cmp_,A['pa'][0]),0.8);cmp_=None
        if cmp_ and (ent=='?' or not A['subj'] and not A['verb'] and not A['cop']):
            ow=[t for t in lw[cmp_[0]:cmp_[1]] if t not in DET|ADV];v=_val(cc,T,*cmp_,'what')
            vt=_vtype(ow,T[cmp_[0]:cmp_[1]]);add(ent,'',*(vt or ls),v,0.6) if cor and len(ow)<=5 else None;cmp_=None
        if cmp_:
            a0,b0=cmp_;ow=lw[a0:b0];v=_val(cc,T,a0,b0,'what')
            if pred in ('name','call') and ow and ow[0] in ('him','her','it','them') and last:add(lo or last,'','what','name',_val(cc,T,a0+1,b0,'what'),0.9)
            elif ow and ow[0] in ('called','named') or pred=='name' or lw[max(0,k-1)] in ('called','named'):add(ent,verb,'what','name',_val(cc,T,a0+(ow[0] in ('called','named')),b0,'what'),0.9)
            elif ow and ow[0]=='allergic':add(ent,verb,'what','allergy',_val(cc,T,a0+2 if b0>a0+1 and ow[1]=='to' else a0+1,b0,'what'),0.9)
            elif pred=='be' or A['pa']:
                at=A['pa'][1] if A['pa'] else 'color' if all(iscol(t) or t in ('dark','light','bright','pale','deep','and','navy','sky','forest','hot') for t in ow if t not in DET|ADV) and any(iscol(t) for t in ow) else 'age' if re.fullmatch(r"(?:\d{1,3}|"+'|'.join(NUMW)+r")(?: (?:years?|months?|weeks?) old)?",v.lower()) else 'breed' if ent.split()[-1] in PETS and ow and ow[0] in ('a','an') else 'job' if isper(ent.split()[-1] if ent!='me' else 'mom') and ow and ow[0] in ('a','an') else 'name' if isper(ent.split()[-1] if ent!='me' else 'x') and v[:1].isupper() and len(v.split())<=2 and ow[0] not in DET else 'where' if cor and _placey([t for t in ow if t not in DET]) else 'kind'
                add(ent,verb,'where' if at=='where' else 'what','location' if at=='where' else at,v,0.9 if at!='kind' else 0.7)
            elif pred=='work' and ow and ow[0]=='as':add(ent,verb,'what','job',_val(cc,T,a0+1,b0,'what'),0.9)
            elif pred in ('drive','ride'):add(ent,verb,'what','model',v,0.8);ent=='me' and pred=='drive' and add('car','','what','model',v,0.5)
            else:
                nb=[q for q,t in enumerate(ow) if t in ('named','called')];nm=[T[a0+q][0] for q in range(nb[0]+1,len(ow)) if T[a0+q][0][:1].isupper()] if nb else []
                he=_obj(T,lw,a0,a0+nb[0],known,lo) if nb else ob0
                if he and he!=ent and pred in ('have','own','adopt','get','buy','play','use','wear','start','book','schedule'):
                    add(he,'','what','kind',_val(cc,T,a0,a0+(nb[0] if nb else b0-a0),'what'),0.6);nm and add(he,'','what','name',' '.join(nm),0.9);[add(he,'','what','color',t,0.7) for t in ow if iscol(t)]
                    cm=[q for q,t in enumerate(ow) if t==','];cm and add(he,'','what','model',_val(cc,T,a0+cm[0]+1,b0,'what'),0.7)
                elif pred:add(ent,verb,'what','likes' if pcls(pred)=='like' else '',v,0.5)
            if pred=='be' and ent!='me' and not A['pa']:
                col=[T[a0+q][0] for q,t in enumerate(ow) if iscol(t)]
                if col and not all(iscol(t) or t in DET|ADV|{'dark','light','bright','pale','deep','navy'} for t in ow):
                    q0=[q for q,t in enumerate(ow) if iscol(t)][0];md=ow[q0-1] if q0>0 and ow[q0-1] in ('dark','light','bright','pale','deep') else '';add(ent,'','what','color',(md+' ' if md else '')+col[0],0.7)
        ent!='?' and (last:=ob0 or ent);ent not in ('?','me') and isper(ent.split()[-1]) and (lp:=ent);lo=ob0 or lo
        for e in {f[0] for f in F}:e not in ('me','?') and e not in known and known.append(e)
    return F,last,names,lp,lo
_QN=re.compile(r"(?i)\b([a-z]+)'s\b(?=\s+(?:my|the|your|our|his|her|it|this|that|a|an|i|we|due|being|getting|going|coming|scheduled|happening|located|held|parked|kept|stored)\b)")
_QWHY=re.compile(r"(?i)\b(?:what(?:'s| is| was)? the (?:point|reason|purpose|story)|for what reason|what for|how come|what made (?:me|us|him|her|them))\b")
def ask(q:str,last:str='',known=(),names=None,lp=''):
    names=names or {};m=_QSAY.sub(' ',_QPRE.sub('',_QN.sub(lambda x:x[1]+' is',q.strip())));m=re.sub(r"\s+",' ',m).strip(' ?.!,');m,T=toks(m);lw=[t[0].lower() for t in T];n=len(T)
    if not n:return None
    wi=next((i for i,t in enumerate(lw) if t in ('what','which','who','whom','whose','where','when','why','how')),None)
    if wi is None:return None
    t=lw[wi];nx=lw[wi+1] if wi+1<n else '';nx2=lw[wi+2] if wi+2<n else '';w={'what':'what','which':'what','who':'what','whom':'what','whose':'what','where':'where','when':'when','why':'why','how':'how'}[t];a='';cand=''
    if _QWHY.search(q):w='why'
    elif t in ('what','which') and nx in ('time','day','date','year','month','hour','night','week'):w='when'
    elif t=='how' and nx=='long' and 'ago' in lw:w='when'
    elif t=='what' and 'for' in lw[-1:]:w='why'
    elif t=='how' and nx=='old':w,a='what','age'
    elif t=='how' and nx in ('many','much','far','long','often'):w='what'
    elif t in ('what','which') and nx in ('color','colour'):a='color';cand=lw[wi+3] if nx2=='of' and wi+3<n else ''
    elif t in ('what','which') and nx in ('kind','type','sort','make','model','brand','breed'):a='model' if nx in ('make','model','brand') else 'breed' if nx=='breed' else 'kind';cand=lw[wi+3] if nx2=='of' and wi+3<n else ''
    elif t in ('who','whom'):a='name'
    elif t in ('what','which') and nx in ('job','name','age'):a=nx
    elif t in ('what','which') and nx and nx not in ATTRN and nx not in AUX and nx not in DET:cand=nx
    ents=[];pa=None;pred='';i=0;used=set();sec=[]
    while i<n:
        if lw[i] in ('my','our'):
            lab,_,p2,j=_np(T,i+1,n);lw=[x[0].lower() for x in T];used|=set(range(i,j))
            if lab:ents.append(_resolve(lab,known))
            elif p2:ents.append('me')
            pa=pa or p2;i=max(j,i+1)
        elif lw[i] in ('i','we','me','us'):ents.append('me');i+=1
        elif lw[i] in ('he','she','they','him','her','them') and i!=wi and (lp or last):ents.append(lp or last);i+=1
        elif lw[i]=='it' and i!=wi and last:ents.append(last);i+=1
        elif lw[i] in names and T[i][0][:1].isupper():ents.append(names[lw[i]]);i+=1
        elif lw[i]=='the' and i+1<n:
            lab,_,p2,j=_np(T,i+1,n);lw=[x[0].lower() for x in T];used|=set(range(i,j))
            lab and ents.append(_resolve(lab,known));i=max(j,i+1)
        else:
            for L in (2,1):
                g=' '.join(FAV.get(x,x) for x in lw[i:i+L])
                if i+L<=n and not used&set(range(i,i+L)) and lw[i] not in AUX|DET|PREPW|{'what','where','when','why','how','which','who'} and (r:=_resolve(g,known)) in known and r!='me':sec.append(r);used|=set(range(i,i+L));break
            i+=1
    if pa and (not a or a==pa[1]):w=pa[0] if w=='what' else w;a=pa[1] if pa[0]==w else a
    for j in range(wi+1,n):
        if j in used or lw[j] in AUX or lw[j] in ADV or lw[j] in ('my','i','we','you','the','usually','again','me','us') or lw[j] in DET:continue
        if isv0(lw[j]) and lw[j] not in ('called','named') and lw[j]!=cand:pred=lemma(lw[j]);break
    if 'called' in lw or 'named' in lw:a=a or ('name' if w=='what' else a)
    pc=pcls(pred)
    if w=='what' and not a and re.search(r"\bdo(?: for (?:a )?(?:living|work|a job))?$",' '.join(lw)):a,pred='job','work'
    if w=='what' and pc in ('drive',) and not a:a='model'
    if w=='where' and not a:a='home' if pc=='live' else 'work' if pc=='work' else 'origin' if 'from' in lw[-1:] or pred in ('','be','come','grow') and 'from' in lw else 'destination' if pc=='go' and pred!='get' or 'to' in lw[-1:] or pred in MOTION and pred not in ('get','take') else 'location'
    nk=_resolve(cand,known) if cand else ''
    ents=([nk] if nk in known else [])+ents+sec;ents=list(dict.fromkeys(ents)) or ([last] if last else [])
    return dict(ents=ents,w=w,attr=a,pred=pred)
KB={AI['kind'],AI['breed'],AI['model']};KA={'kind','breed','model'}
def lookup(st:Store,qa:dict,known=None):
    if not len(st.s) or not qa or not qa['ents']:return []
    s=st.s;e=s[:,0].astype(np.uint32)|s[:,1].astype(np.uint32)<<16;w=s[:,3]//17;a=s[:,3]%17;pr=s[:,2];lm=s[:,7];ql=h32(qa['pred'])&0xFFFF if qa['pred'] else None;cf=(s[:,5]%17)/16.0;ag=st.age(s[:,4]).astype(np.float64)
    qe=np.array([h32(x) for x in qa['ents']],np.uint32);m=np.isin(e,qe)&(w==WI[qa['w']])
    if not m.any():return []
    qp=h32(pcls(qa['pred']))&0xFFFF if qa['pred'] else None;bh=h32('be')&0xFFFF;h0=h32('')&0xFFFF;me=h32('me');ix=np.flatnonzero(m)
    gk=[(int(e[i]),int(w[i]),int(a[i]),int(lm[i]) if int(e[i])==me or int(w[i])==0 and int(pr[i]) not in (bh,h0) else 0) for i in ix];best={}
    for i,g in zip(ix,gk):best[g]=i if g not in best or ag[i]<ag[best[g]] else best[g]
    qa_=AI.get(qa['attr'],0);out=[];mv={h32(pcls(x))&0xFFFF for x in MOTION|MODE}|{h32(x)&0xFFFF for x in MOTION|MODE};qp=h32('go')&0xFFFF if qa['pred'] in GEN and qa['w']=='how' else qp
    for i in best.values():
        if int(e[i])==me and qp not in (None,bh) and int(pr[i]) not in (qp,bh,h0) and not (qa['pred'] in GEN and qa['w']=='how' and int(pr[i]) in mv):continue
        cp=qa_ in KB and int(a[i]) in KB;sc=0.5*(qa_==0 and int(a[i]) in (0,AI['kind'],AI['model']))+(3.0 if int(e[i])==int(qe[0]) else 2.0)+2.0*(qa_>0 and int(a[i])==qa_)+1.0*(cp and int(a[i])!=qa_)-2.5*(qa_>0 and int(a[i])>0 and int(a[i])!=qa_ and not cp)-0.5*(qa_>0 and int(a[i])==0)+1.5*(qp is not None and int(pr[i])==qp)+0.5*(ql is not None and int(lm[i])==ql)+float(cf[i])-0.002*float(ag[i])
        out.append((sc,int(i)))
    return sorted(out,reverse=True)
def keys(msg:str,F,known=()):
    ks=[e for e,*_ in F]+[x for c in clauses(msg) for T in [toks(c)[1]] for x in [_np(T,j+1,len(T))[0] for j,t in enumerate(T) if t[0].lower() in ('my','our')]]+(['me'] if re.search(r"(?i)\b(?:i|i'm|i've|i'd|i'll|me|we)\b",msg) else [])
    ks=[_resolve(k,known) for k in ks if k and k!='?'];nn=[t for t in re.findall(r"[a-z][a-z'-]{2,}",msg.lower()) if t not in U._STOPW.split() and t not in AUX|ADV|DET|PREPW and not isv(t)]
    return [h32(k) for k in dict.fromkeys(ks+nn)]
FW=dict(ent=2.0,own=1.0,typ=2.5,att=1.0,att_bad=1.0,pred=1.0,pmis=1.5,rec=0.35,sup=3.0,slot=1.5,subj=0.5,null=4.5,tau=0.75)
GEN=S_("get go commute travel come")
def spans(text:str,known=(),names=None,last=''):
    F=extract(text,last,known,names)[0];out=[dict(v=v,w=w,a=a,ent=last if e=='?' else e,pred=lemma(vb) if vb else '',subj=cf>=0.75) for e,vb,w,a,v,cf,_ in F]
    for c in clauses(text):
        A=analyze(c,last,known,names)
        if not A:continue
        T=A['T'];lw=[t[0].lower() for t in T];ent=A['ent'] if A['ent'] not in ('','?') else last
        [out.append(dict(v=T[q][0],w='what',a='color',ent=ent,pred=A['pred'],subj=False)) for q,t in enumerate(lw) if iscol(t)]
        [out.append(dict(v=_val(A['c'],T,x,y,w),w=w,a='',ent=ent,pred=lemma(vb) if vb else A['pred'],subj=False)) for (w,a,x,y),vb in zip(A['sp'],A['sv'])]
        last=ent or last
    return [o for o in out if o['v']]
def _hd(x:str)->str:return _sg(x.split()[-1]) if x else ''
def soft(st:Store,qa:dict,known=(),names=None,slot:str='',fw=None):
    fw={**FW,**(fw or {})}
    if not qa or not qa['ents'] or not len(st.u):return []
    stt,K=st.uidx();qk=np.array([h32(x) for x in qa['ents']],np.uint32);m=np.isin(K,qk).any(1)
    if not m.any():return []
    ix=np.flatnonzero(m);ag=st.age(st.u[stt[ix],1]).astype(np.float64);order=ix[np.argsort(ag,kind='stable')];C=[];inv={v:k for k,v in st.ents().items()}
    for r,i in enumerate(order):
        for sp in spans(st.utext(int(i)),known,names,inv.get(int(K[i][0]),'')):sp['ui']=int(i);C.append((r,sp))
    if not C:return []
    qa_=qa['attr'];qp=pcls(qa['pred']) if qa['pred'] else '';pe=qa['ents'][0];qh={_hd(x) for x in qa['ents']};seen={};S=[]
    for r,sp in C:
        g=(sp['ent'],sp['w'],sp['a']);sup=g in seen and seen[g]<r;seen.setdefault(g,r)
        sl=bool(slot) and (sp['v'].lower() in slot.lower() or slot.lower() in sp['v'].lower())
        pm=sp['ent']=='me' and pe=='me' and bool(qp) and qp not in ('be','have','do') and bool(sp['pred']) and pcls(sp['pred']) not in (qp,'be','have') and not (qa['pred'] in GEN and qa['w']=='how' and (sp['pred'] in MOTION|MODE))
        s=-fw['pmis']*pm+fw['ent']*(sp['ent']==pe)+fw['own']*(sp['ent'] in qa['ents'] or _hd(sp['ent']) in qh)+fw['typ']*(sp['w']==qa['w'])+fw['att']*(bool(qa_) and (sp['a']==qa_ or qa_ in KA and sp['a'] in KA))-fw['att_bad']*(bool(qa_) and bool(sp['a']) and sp['a']!=qa_ and not (qa_ in KA and sp['a'] in KA))+fw['pred']*(bool(qp) and pcls(sp['pred'])==qp)-fw['rec']*r-fw['sup']*sup+fw['slot']*sl+fw['subj']*sp['subj']
        S.append((s,sp))
    z=np.array([s for s,_ in S]+[fw['null']])/fw['tau'];p=np.exp(z-z.max());p/=p.sum();agg={'':(float(p[-1]),None)};p=p[:-1]
    for (s,sp),pp in zip(S,p):k=sp['v'].lower();agg[k]=(agg[k][0]+pp,agg[k][1]) if k in agg else (pp,sp)
    return sorted([(float(pp),sp) for pp,sp in agg.values()],key=lambda x:-x[0])
def _top(S):
    T=[x for x in S if x[1] is not None];return T,(T[0] if T and T[0][0]>=max([x[0] for x in S if x[1] is None] or [0]) else None)
def label(ent:str)->str:return 'you' if ent=='me' else 'your '+ent
def line(ent:str,w:str,a:str,val:str,verb:str='',unsure:str='')->str:
    lb=label(ent)+(f" ({verb.lower()})" if ent=='me' and verb and w!='what' else '');return f"Memory: {lb}, {a or w}: {val}{' or '+unsure+' (unsure)' if unsure else ''}."
def _row(st,i):
    r=st.s[i];e,v,val=st.text(r);return dict(ent=e,verb=v,w=WS[int(r[3])//17],a=AT[int(r[3])%17],v=val,conf=(int(r[5])%17)/16.0)
def _li(st:Store,m=None):
    ix=np.flatnonzero(m) if m is not None else np.arange(len(st.s));return int(ix[np.argmin(st.age(st.s[ix,4]).astype(np.int64)*len(st.s)-ix)]) if len(ix) else None
def _last(st:Store)->str:i=_li(st);return st.text(st.s[i])[0] if i is not None else ''
def _ea(st:Store):return st.s[:,0].astype(np.uint32)|st.s[:,1].astype(np.uint32)<<16
def _typed(st:Store,w:str,a:str,noself:bool=True)->str:
    if not len(st.s):return ''
    i=_li(st,(st.s[:,3]//17==WI[w])&((st.s[:,3]%17==AI.get(a,0)) if a else True)&((_ea(st)!=h32('me')) if noself else True));return st.text(st.s[i])[0] if i is not None else ''
def _lper(st:Store)->str:
    i=_li(st,np.array([isper(st.text(r)[0].split()[-1]) and st.text(r)[0]!='me' for r in st.s],bool)) if len(st.s) else None;return st.text(st.s[i])[0] if i is not None else ''
def _lobj(st:Store)->str:
    i=_li(st,(_ea(st)!=h32('me'))&(st.s[:,4]==st.head%TM)) if len(st.s) else None;return st.text(st.s[i])[0] if i is not None else ''
def ingest(st:Store,msg:str,B=None)->list:
    B=B or bus(st);kn=st.ents();i=_li(st);ls=(WS[int(st.s[i,3])//17],AT[int(st.s[i,3])%17]) if i is not None else ('what','');F,_,_,lp,lo=extract(msg,_last(st),list(kn),st.names,ls,st.lp or _lper(st),st.lo or _lobj(st));st.head+=1;R=[]
    for e,v,w,a,val,cf,co in F:
        e=e if e!='?' else _typed(st,w,a) or _typed(st,w,'') or _typed(st,w,'',False) or _last(st) or 'me'
        if co and w!='what' and (not v or lemma(v)=='be') and len(st.s) and (j:=_li(st,(_ea(st)==h32(e))&(st.s[:,3]//17==WI[w]))) is not None:v,a=st.text(st.s[j])[1],AT[int(st.s[j,3])%17]
        j=_li(st,(_ea(st)==h32(e))&(st.s[:,3]==WI[w]*17+AI.get(a,0))) if len(st.s) else None;v=v or (st.text(st.s[j])[1] if j is not None else '')
        if co and j is not None:
            ov=st.text(st.s[j])[2];c0=int(st.s[j,4])
            for q in np.flatnonzero((st.s[:,3]==st.s[j,3])&(st.s[:,4]==c0)&(_ea(st)!=h32(e))):
                oe,ovb,ovv=st.text(st.s[q])
                if ovv==ov and (q2:=_li(st,(_ea(st)==h32(oe))&(st.s[:,3]==st.s[j,3])))==q and _learn(B,st,oe,ovb,w,a,val,cf):R.append((oe,ovb,w,a,val,cf))
        _learn(B,st,e,v,w,a,val,cf) and R.append((e,v,w,a,val,cf))
    st.lp,st.lo=lp,lo;st.putu(msg.strip()[:600],keys(msg,R,list(kn)));return R
SAY={('where','home'):'{S} {live} in {v}.',('where','work'):'{S} {work} at {v}.',('where','origin'):'{S} {be} from {v}.',('where','destination'):'{S} {be} going to {v}.',('where','location'):'{S} {be} at {v}.',('what','name'):'{P} name is {v}.',('what','job'):'{S} {work} as {n}.',('what','age'):'{S} {be} {v}.',('what','favorite'):'{P} favorite is {v}.'}
_SW={'my':'your','me':'you','i':'you',"i'm":"you're",'our':'your','we':'you','us':'you','mine':'yours','myself':'yourself',"i've":"you've","i'll":"you'll","i'd":"you'd"}
def _you(t:str)->str:return re.sub(r"(?i)\b(?:i'm|i've|i'll|i'd|my|me|i|our|we|us|mine|myself)\b",lambda m:_SW[m[0].lower()],t)
def say(r:dict)->str:
    e,w,a,v,vb=r['ent'],r['w'],r['a'],_you(r['v']),r.get('verb','');me=e=='me';S='You' if me else f'Your {e}';P='Your' if me else f"Your {e}'s";n=v if re.match(r"(?i)(?:a|an|the)\b",v) else ('an ' if v[:1].lower() in 'aeiou' else 'a ')+v
    k=SAY.get((w,a))
    if k:return k.format(S=S,P=P,v=v,n=n,be='are' if me else 'is',live='live' if me else 'lives',work='work' if me else 'works')
    if w in ('why','how'):return (v[:1].upper()+v[1:]+'.') if not (w=='how' and v.lower()==vb.lower()) else f"{S} {v.lower()}{'' if me else 's'}."
    if w=='when':return (f"{S} {vb.lower()} {'in ' if re.fullmatch(r'(?:1[89]|20)\d\d|[A-Z][a-z]+',v) else ''}{v}." if me and vb else f"It's {v}.")
    return f"{S} {vb.lower()} {v}." if me and vb and lemma(vb)!='be' else f"{S} {'are' if me else 'is'} {v if a!='kind' or v[:1].isupper() else n}."
class SlotLUT:
    def __init__(s,st:Store,gate:float):s.st,s.gate,s.pend,s.last=st,gate,[],None
    def store(s,key,value,source='',meta=None,track_recent=True):
        e,v,w,a,cf=key[6:].split('\x1f');s.st.put(e,v,w,a,value,float(cf));s.pend.append(len(s.st.s)-1)
    def commit_recent(s):s.pend=[]
    def rollback_recent(s):
        m=np.ones(len(s.st.s),bool);m[s.pend]=False;s.st.s=s.st.s[m];s.pend=[]
    def lookup(s,q):
        if q.startswith('slot::'):e,v,w,a,_=q[6:].split('\x1f');x=_slotval(s.st,e,v,w,a);return {'a':x} if x else None
        s.last=(q,recall(s.st,q,'both'));r=s.last[1];return {'a':r['slot']['v']} if r['slot'] and r['p']>=s.gate else None
class SoftLUT:
    def __init__(s,st:Store,sl:SlotLUT):s.st,s.sl=st,sl
    @property
    def _raw(s):return len(s.st.u)>0
    def lookup_soft(s,q,k=1,return_diag=True):
        r=s.sl.last[1] if s.sl.last and s.sl.last[0]==q else recall(s.st,q,'soft');S=r['soft']
        return (S[0][1]['v'],None,S[0][0],S[1][0] if len(S)>1 else None) if S and r['line'] and not r['slot'] else (None,None,None,None)
def bus(st:Store,gate:float=None):
    from amni.serve.memory_bus import MemoryBus,_CONF_DIRECT
    g=_CONF_DIRECT if gate is None else gate;sl=SlotLUT(st,g);return MemoryBus(adam=None,answer_lut=sl,sem_lut=SoftLUT(st,sl),kb=None,learning_atlas=None,ledger_path=os.devnull)
def _slotval(st:Store,e,v,w,a)->str:
    if not len(st.s):return ''
    i=_li(st,(_ea(st)==h32(e))&(st.s[:,3]==WI[w]*17+AI.get(a,0))&((st.s[:,7]==(h32(lemma(v))&0xFFFF)) if v else True));return st.text(st.s[i])[2] if i is not None else ''
def _learn(B,st:Store,e,v,w,a,val,cf)->bool:
    pv=_slotval(st,e,v,w,a);return bool(B.record_learning('slot::'+'\x1f'.join((e,v,w,a,str(round(cf,4)))),val,kind='user_fact',provenance='user:toroid',exactness='exact',supersedes=pv if pv and pv!=val else None)['stored'])
def respond(st:Store,q:str,gate:float=None,hint:float=0.5,unsure:float=0.25,abstain:bool=False,unk:bool=False)->dict:
    B=bus(st,gate);v,home,c=B.recall(q);sl=B.answer_lut;r=sl.last[1] if sl.last and sl.last[0]==q else recall(st,q,'both');qa=r['qa']
    if v and qa and qa['ents']:
        if home=='tier0_atex_override':d=r['slot']
        else:sp=r['soft'][0][1];d=dict(ent=qa['ents'][0],w=qa['w'],a=qa['attr'] or sp['a'],v=v,verb=sp['pred'] if qa['ents'][0]=='me' else '')
        return dict(kind='direct',text=say(d),conf=float(c),home=home,line=r['line'],r=r)
    if r['line'] and r['p']>=hint:return dict(kind='hint',text='',conf=r['p'],home='hint',line=r['line'],r=r)
    if abstain and qa and qa['ents'] and not r['slot'] and r.get('pnull',0)>=(gate or 0.9) and (qa['ents'][0] in st.ents() or qa['ents'][0]=='me'):return dict(kind='direct',text="You haven't told me that yet.",conf=r['pnull'],home='abstain',line='',r=r)
    S=r['soft']
    if unk and qa and qa['ents'] and not r['slot'] and (qa['ents'][0] in st.ents() or qa['ents'][0]=='me'):return dict(kind='unknown',text='',conf=r.get('pnull',1.0),home='unknown',line=line(qa['ents'][0],qa['w'],qa['attr'],'unknown'),r=r)
    return dict(kind='unsure',text='',conf=r['p'],home='unsure',line=line(qa['ents'][0],qa['w'],qa['attr'] or S[0][1]['a'],S[0][1]['v'],'',S[1][1]['v']),r=r) if qa and qa['ents'] and len(S)>1 and S[0][0]>=unsure else dict(kind='none',text='',conf=r['p'],home='none',line='',r=r)
def observe(st:Store,msg:str,**kw):
    m=str(msg).strip();f=forget_st(st,m)
    return ('forget',f) if f is not None else ('ask',respond(st,m,**kw)) if U._isq(m) else ('skip',None) if not m or U._REQ.search(m) else ('fact',ingest(st,m))
def remember(key:str,msg:str)->int:
    with U._lock:st=load(key);F=ingest(st,msg);save(key,st);return len(F)
def forget_st(st:Store,msg:str):
    f=U._fg(str(msg).strip())
    if not f:return None
    if f[1]:n=len(st.s)+len(st.uidx()[0]);st.s,st.u,st.head,st._x,st.lp,st.lo=Store().s,Store().u,0,None,'','';return ('all',n)
    kn=st.ents();qa=ask(f[0],_last(st),list(kn),st.names,st.lp)
    if not qa or not qa['ents']:return (None,0)
    wa=None if re.match(r"(?i)\s*what about\b",f[0]) and qa['w']=='what' and not qa['attr'] else qa['w'];e=h32(qa['ents'][0]);es=_ea(st);wm=(st.s[:,3]//17==WI[wa]) if wa else np.ones(len(st.s),bool)
    qp=h32(qa['pred'])&0xFFFF if qa['pred'] and qa['pred'] not in ('be','have','do') else None;qc=h32(pcls(qa['pred']))&0xFFFF if qp is not None else None
    pm=((st.s[:,7]==qp) if (st.s[:,7]==qp).any() else (st.s[:,2]==qc)) if qp is not None and qa['ents'][0]=='me' else None
    dm=(es==e)&wm&(pm if pm is not None else True)|(wm&pm if pm is not None else False)
    for q in np.flatnonzero(dm):dm|=(st.s[:,4]==st.s[q,4])&(st.s[:,3]==st.s[q,3])&np.array([st.text(r)[2]==st.text(st.s[q])[2] for r in st.s],bool)
    stt,K=st.uidx();um=np.flatnonzero((K==e).any(1))
    um=[i for i in um if wa is None or any(sp['w']==wa for sp in spans(st.utext(int(i)),list(kn),st.names))]
    n=int(np.sum(dm))+len(um);st.s=st.s[~dm];st.udrop(um);st.lo='' if st.lo==qa['ents'][0] else st.lo;return (qa['ents'][0],n)
def forget(key:str,msg:str):
    with U._lock:
        st=load(key);r=forget_st(st,msg)
        r is not None and r[1] and save(key,st);return r
def recall(st:Store,q:str,tier:str='both',unk:bool=False,theta:float=0.5,fw=None)->dict:
    kn=st.ents();qa=ask(q,_last(st),list(kn),st.names,st.lp);out=dict(qa=qa,slot=None,soft=[],line='',p=0.0)
    if not qa or not qa['ents']:return out
    L=lookup(st,qa) if tier in ('slot','both') else [];out['slot']=_row(st,L[0][1]) if L and L[0][0]>=2.0 else None;sv=out['slot']['v'] if out['slot'] else ''
    S=soft(st,qa,list(kn),st.names,sv if tier=='both' else '',fw) if tier in ('soft','both') else []
    out['pnull']=max([x[0] for x in S if x[1] is None] or [1.0]);S,top=_top(S);out['soft']=S;out['p']=top[0] if top else (out['slot']['conf'] if out['slot'] else 0.0);ent=qa['ents'][0];pa=qa['attr']
    ag=sum(pp for pp,sp in S if sv and (sv.lower() in sp['v'].lower() or sp['v'].lower() in sv.lower()));out['p']=(ag if S else out['slot']['conf']) if tier=='both' and out['slot'] else out['p']
    if tier=='slot' or tier=='both' and out['slot']:
        r=out['slot'];out['line']=line(r['ent'],r['w'],r['a'],r['v'],r['verb']) if r else '';u=np.flatnonzero(st.u[st.uidx()[0],1]==int(st.s[L[0][1],4])) if r else [];out['src']=st.utext(int(u[-1])) if len(u) else ''
    elif top:
        sp=top[1];out['src']=st.utext(sp['ui']);out['line']=line(ent,qa['w'],pa or sp['a'],sp['v'],sp['pred'] if ent=='me' and qa['w']!='what' else '',S[1][1]['v'] if top[0]<theta and len(S)>1 else '')
    if not out['line'] and unk and (ent in kn or ent=='me'):out['line']=line(ent,qa['w'],pa,'unknown')
    return out
def turn(message:str,history=None,user=None,session=None,dry:bool=False)->dict:
    t0=time.perf_counter();out={'lines':[],'forgot':None,'facts':0,'p':0.0,'ms':0.0,'direct':'','kind':''}
    if not enabled():return out
    m=str(message).strip();E=os.environ.get;dr=E('AMNI_UMEM_TOROID_DIRECT','1')=='1'
    try:
        with U._lock:
            key=U.user_key(user);st=load(key);k,r=observe(st,m,hint=float(E('AMNI_UMEM_TOROID_HINT','0.5')),unsure=float(E('AMNI_UMEM_TOROID_UNSURE','0.25')),abstain=E('AMNI_UMEM_TOROID_ABSTAIN','0')=='1',unk=E('AMNI_UMEM_TOROID_UNK','0')=='1')
            k=='forget' and out.update(forgot=r);k=='fact' and out.update(facts=len(r))
            k=='ask' and out.update(kind=r['kind'],p=r['conf'],direct=r['text'] if dr and r['kind']=='direct' else '',lines=[r['line']] if r['line'] and not (dr and r['kind']=='direct') and r['kind']!='none' else [])
            not dry and (k=='fact' or k=='forget' and r[1]) and save(key,st)
    except (OSError,ValueError) as e:out['err']=type(e).__name__
    out['ms']=(time.perf_counter()-t0)*1000;return out

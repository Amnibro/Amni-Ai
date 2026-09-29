import os,re,hmac,stat,time,zlib,struct,secrets,functools
import numpy as np
from amni.compute.reffelt4 import encode_fp16_to_rgba4,decode_rgba4_to_fp16
from amni.serve import user_memory as U
from amni.serve.amni_chat_client import _hkdf_sha256,_aenc,_adec
WS=('what','where','when','why','how');AT=('','name','color','model','job','breed','kind','favorite','age','home','work','location','origin','destination','likes','allergy','birthday');WI={w:i for i,w in enumerate(WS)};AI={a:i for i,a in enumerate(AT)}
MAGIC=b'AMTO';VER=1;RW=48;UW=32;NK=14;TM=4913;HDR=struct.Struct('<4sHHHIII')
ZD=b" actually sorry wait now next last every morning evening night week month year monday tuesday wednesday thursday friday saturday sunday january february march april may june july august september october november december because so that to be for with by from the in on at and my is was it's he she they his her their our dog cat car bike house apartment flight appointment meeting birthday name called named lives live moved works work job drives parked keep favorite color blue red green black white sister brother mom dad wife husband son daughter friend best boyfriend girlfriend partner cousin aunt uncle grandma grandpa boss coworker neighbor roommate I'm I've I "
S_=lambda s:frozenset(s.split())
VB=S_("locate base be have do go get make take see come know think look want give use find tell ask work seem feel try leave call keep let begin help talk turn start show hear play run move live believe hold bring happen write sit stand lose pay meet include continue set learn change lead understand watch follow stop create speak read spend grow open walk win offer remember love consider appear buy wait serve send expect build stay fall cut reach remain suggest raise pass sell require report decide pull save visit catch cook bake eat drink drive ride fly travel study teach sing dance draw paint swim bike cycle hike fish shop clean fix repair rent own wear sleep rest relax celebrate attend join marry adopt train practice prepare plan pick drop park store hide put hang care miss enjoy like hate prefer need afford avoid improve earn commute jog skate ski surf climb camp garden sew knit film record type text email mail order deliver ship pack book reserve cancel renew replace upgrade trade lend borrow host throw support feed impress surprise treat protect escape retire quit graduate major relocate reside head return arrive land depart schedule reschedule postpone name adore dislike detest volunteer coach babysit tutor manage check carpool bus transfer switch sign enroll apply register donate collect listen dive row sail kayak paddle wash polish charge plug lock unlock share waste grab brew roast grill fry boil steam blend mix knead chop meditate pray exercise stretch lift fit stash tuck keep bury hang leave lock order pay renovate paint shower feed walk groom brush")
NOUNISH=S_("work visit call date plan trip train ride show run practice lesson race drive cook dance book order party check flight text email mail record film garden camp host coach tutor treat support type fish paint drink set change turn store park name hike swim shop ship pick lift walk bus rest sleep dive sail row charge lock plug share exercise stretch roast grill boil mix blend love care schedule return transfer surf trade appointment meeting spot")
IRR={**dict(x.split(':') for x in "rang:ring rung:ring sang:sing sung:sing swum:swim drunk:drink begun:begin became:become overcame:overcome forgave:forgive forgiven:forgive fell:fall fallen:fall fled:flee forgotten:forget froze:freeze frozen:freeze grown:grow knelt:kneel knew:know known:know laid:lay led:lead lent:lend lain:lie meant:mean ridden:ride risen:rise sold:sell shook:shake shaken:shake shone:shine shot:shoot showed:show shown:show shrank:shrink slid:slide spoken:speak spun:spin sprang:spring stole:steal stolen:steal stuck:stick stung:sting struck:strike swore:swear sworn:swear swept:sweep swung:swing tore:tear torn:tear understood:understand woke:wake woken:wake worn:wear bitten:bite bled:bleed blew:blow blown:blow broken:break burnt:burn chosen:choose clung:cling crept:creep dealt:deal dug:dig drew:draw drawn:draw dreamt:dream fed:feed fought:fight forbade:forbid leapt:leap learnt:learn spilt:spill sank:sink sunk:sink sought:seek sewn:sew slung:sling sped:speed stank:stink strode:stride strung:string trod:tread wept:weep withdrew:withdraw wrung:wring overslept:oversleep upset:upset mistook:mistake mistaken:mistake undertook:undertake withheld:withhold rebuilt:rebuild repaid:repay resold:resell rewrote:rewrite outgrew:outgrow".split()),'went':'go','gone':'go','got':'get','gotten':'get','bought':'buy','brought':'bring','kept':'keep','left':'leave','ran':'run','took':'take','taken':'take','taught':'teach','sold':'sell','made':'make','began':'begin','drove':'drive','driven':'drive','flew':'fly','flown':'fly','rode':'ride','ridden':'ride','ate':'eat','drank':'drink','met':'meet','paid':'pay','said':'say','saw':'see','seen':'see','found':'find','thought':'think','told':'tell','gave':'give','given':'give','came':'come','wrote':'write','written':'write','sat':'sit','stood':'stand','lost':'lose','spent':'spend','grew':'grow','won':'win','built':'build','fell':'fall','held':'hold','heard':'hear','felt':'feel','sent':'send','slept':'sleep','swam':'swim','sang':'sing','threw':'throw','wore':'wear','lent':'lend','hung':'hang','caught':'catch','chose':'choose','broke':'break','forgot':'forget','hid':'hide','hidden':'hide','located':'locate','situated':'locate','based':'base','is':'be','are':'be','was':'be','were':'be','am':'be',"'s":'be',"'m":'be',"'re":'be','been':'be','has':'have','had':'have','does':'do','did':'do','done':'do','leaves':'leave','lives':'live','studies':'study','flies':'fly','carries':'carry','tries':'try','marries':'marry','married':'marry','studied':'study','tried':'try','majored':'major','traveled':'travel','travelled':'travel','commutes':'commute'}
CLASS={'retire':'live','settle':'live','emigrate':'live','move':'live','relocate':'live','reside':'live','stay':'live','base':'live','fly':'go','travel':'go','head':'go','commute':'go','get':'go','return':'go','depart':'go','arrive':'go','ride':'drive','park':'keep','store':'keep','hide':'keep','put':'keep','stash':'keep','tuck':'keep','hang':'keep','call':'name','name':'name','leave':'keep','reschedule':'be','schedule':'be','postpone':'be','adore':'like','love':'like','enjoy':'like','major':'study'}
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
TGL=S_("all whole these end beginning middle start next this last every each on at in by until till since before after around about from the of a an early late mid other coming following to and or past quarter first second third fourth fifth")
NUMW=S_("one two three four five six seven eight nine ten eleven twelve twenty thirty half couple few")
ATTRN={'name':('what','name'),'color':('what','color'),'colour':('what','color'),'job':('what','job'),'occupation':('what','job'),'profession':('what','job'),'breed':('what','breed'),'age':('what','age'),'model':('what','model'),'make':('what','model'),'brand':('what','model'),'address':('where','home'),'time':('when',''),'date':('when',''),'day':('when',''),'location':('where','location'),'reason':('why',''),'favorite':('what','favorite'),'birthday':('when','birthday')}
ADJ=S_("new old little big older younger baby current first second third next last upcoming usual regular new-ish lovely dear sweet whole same")
PRONS={'he','she','they','it','him','her','them'}
_TOK=re.compile(r"\d{1,2}:\d{2}|\d+(?:st|nd|rd|th|s\b)?|[A-Za-z]+(?:[.'-][A-Za-z]+)*\.?|'[a-z]+|[^\w\s]")
_CT=re.compile(r"(?i)\b(i|you|he|she|it|we|they|that|there|what|where|who|when|why|how)'(s|m|re|ve|ll|d)\b")
_NT=re.compile(r"(?i)\b(do|does|did|is|was|are|were|ca|wo|could|would|should|have|has|had)n't\b")
_LEAD=re.compile(r"(?i)^\W*(?:(?:ugh|argh|dang|darn|shoot|welp|oh no|actually|oh|so|well|also|and|but|btw|by the way|fun fact|funny story|update(?: on [^:,]{1,40})?|correction|confession|sorry|wait|oops|ok(?:ay)?|hey|hi|um|uh|yeah|yea|yes|no|nope|right|lol|lmao|haha+|ngl|tbh|omg|yo|nvm|never ?mind|hmm+|ah+|ooh|anyway|honestly|basically|good news|bad news|so anyway|scratch that|i misspoke|i meant|i mean|my (?:bad|mistake)|i got that wrong|i was wrong|just so you know|fyi|heads up|anyway|hmm|ah|guess what|quick update|small correction|change of plans?|not anymore|new info)\b[\s,:;.!-]*)+")
_CORR=re.compile(r"(?i)\b(?:wait|misread|mixed (?:it|that|them) up|typo|i meant|make that|make it|turns out|actually|correction|sorry|oops|scratch that|misspoke|my (?:bad|mistake)|got that wrong|was wrong|wrong|not anymore|any ?more|no longer|changed|switched|instead|update|rescheduled|postponed|moved|now)\b")
_COR2=re.compile(r"(?i)^\W*\*|^\W*(?:no|nope|nah|wait|oops|sorry|correction|update|not|nvm)\b|\b(?:correction|scratch that|i meant|make that|make it|turns out|misread|mixed (?:it|that|them) up|typo|misspoke|my (?:bad|mistake)|got (?:that|it) wrong|was wrong|change of plans?|never ?mind|nvm|rescheduled|postponed|pushed back|(?:moved|pushed|bumped|shifted) (?:it|that|them)\b|(?:got|been|was|is|'s)\s+(?:moved|pushed|bumped|shifted|changed|rescheduled|postponed)|swapped|instead|not anymore|no longer|update)\b|\b(?:actually|really)\s*,?\s*(?:it'?s|it was|it is|its|they'?re|he'?s|she'?s|that'?s|we'?re|i'?m|no)\b|\bit'?s\s+actually\b|,\s*not\s+\w+")
_NOT=re.compile(r"(?i)^\s*not\s+[^,]+,\s*|,?\s*\bnot\s+(?:the\s+|a\s+|an\s+|on\s+|in\s+|at\s+)?[\w' -]+?\s*(?=$|[.!?])")
_QPRE=re.compile(r"(?i)^\W*(?:(?:hey|yo|so|ok(?:ay)?|um|uh|and|also|wait|quick question|quick q|adam|btw|sorry|hmm|oh|quick one|any idea|remind me|real quick)\W+)*(?:(?:can|could|would) you (?:please )?(?:tell me|remind me|say)(?: again)?\W*|(?:do|did) you (?:still )?(?:remember|recall|know)\W*|remind me(?: again)?\W*|tell me(?: again)?\W*|i forgot\W*|i forget\W*)?")
_QSAY=re.compile(r"(?i)\b(?:did|do|had|have) (?:i|we) (?:say|said|tell you|told you|mention|mentioned|say that|tell you that)(?: that| about| was| is)?\b|\bi (?:told|said to|mentioned to) you\b|\bdid i\b(?= (?:say|tell))|\bagain\b|\bexactly\b|\bthat\b(?=\s*\?)|,?\s*(?:remind me|do you remember|did i mention|lol|again)\s*(?=[?.!]*$)")
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
isplace=lambda t:t in PLACEN or len(t)>2 and t.isalpha() and t not in DET|PREPW|AUX|PRONS and bool({'loc','sto'}&wcls(t)) and t not in TW|NOUNISH|{'work','home','way','side','end','point'}
ismeans=lambda t:t in MEANS or len(t)>2 and t.isalpha() and t not in DET|PREPW|AUX|PRONS and bool({'veh','comm','pay'}&wcls(t,1))
isper=lambda t:bool(U.CLS.get(U.bh(t.lower().encode())&U.WM,0)&32) or 'per' in wcls(t.lower(),1) or t.lower() in ('kid','kids','child','baby','fiance','fiancee','husband','wife','teacher','doctor','dentist','therapist','landlady','grandmother','grandfather','stepson','stepdaughter','twin','bestie','colleague','sibling','parents')
CAS=dict(x.split(':') for x in "bc:because cuz:because coz:because cos:because bcos:because bcuz:because becuz:because becuase:because becasue:because beacuse:because bday:birthday b-day:birthday bro:brother sis:sister mum:mom mam:mom mummy:mom gf:girlfriend bf:boyfriend hubby:husband appt:appointment appts:appointments tmrw:tomorrow tmr:tomorrow tmrow:tomorrow tomoz:tomorrow tonite:tonight thurs:thursday thur:thursday thu:thursday tues:tuesday tue:tuesday weds:wednesday fri:friday yr:year yrs:years wk:week wks:weeks hr:hour hrs:hours mins:minutes rn:now pls:please plz:please ur:your teh:the hte:the adn:and nad:and taht:that waht:what wiht:with jsut:just thx:thanks ppl:people u:you til:until favourite:favorite fave:favorite fav:favorite colour:color colours:colors neighbour:neighbor rember:remember remeber:remember wensday:wednesday wendsday:wednesday tommorrow:tomorrow tommorow:tomorrow thru:through".split())
DAB={'sat':'saturday','sun':'sunday','wed':'wednesday','mon':'monday','mar':'march','jan':'january','feb':'february','apr':'april','jun':'june','jul':'july','aug':'august','sept':'september','sep':'september','oct':'october','nov':'november','dec':'december'}
_CZ=[(re.compile(r"(?i)\b(what|where|when|who|how|that|there|she)s\b"),r"\1's"),(re.compile(r"\b([Ii])(m|ve)\b"),r"\1'\2"),(re.compile(r"(?i)\b(do|does|did|is|was|are|were|ca|could|would|should|have|has|had)nt\b"),r"\1n't"),(re.compile(r"(?i)\bb/c\b|\bb/c(?=\s)"),'because'),(re.compile(r"(?i)\bw/(?=\s)"),'with'),(re.compile(r"(?i)(?<=\w) n (?=\w)"),' and '),(re.compile(r"(?i)\bw (?=(?:the|my|a|an|his|her|their|our|apple|google|cash|card)\b)"),'with '),(re.compile(r"(?i)^(\W*)y (?=(?:did|do|does|is|was|are|were|am|would|should|can)\b)"),r"\1why "),(re.compile(r"(?i)\b(\d{1,2}(?::\d{2})?)(am|pm)\b"),r"\1 \2"),(re.compile(r"\b([A-Za-z]+s)'(?=\s+[a-z])"),r"\1's"),(re.compile(r"\b(\d{1,2})\.(\d{2})(?=\s*(?:am|pm|a\.m|p\.m)?\b)(?!\.\d)"),r"\1:\2")]
@functools.lru_cache(maxsize=1)
def _cog():
    try:
        from amni.serve import cognition as C;return (C,C._g()) if os.environ.get('AMNI_UMEM_TOROID_WN','1')=='1' else None
    except Exception:return None
LEX=None
def _lex()->frozenset:
    global LEX
    LEX=LEX or frozenset(w for v in globals().values() if isinstance(v,frozenset) for w in v)|frozenset(IRR)|frozenset(CAS.values())|frozenset(ATTRN)|frozenset(U._STOPW.split())|frozenset("during through throughout toward towards without within upon among amongst despite unless whilst onto via per versus beyond besides except plus minus near nearby inside outside till since around across behind below beneath beside between whose whom whichever whoever each every both few many much more most less least several such own same other another ought shall everything anything something nothing everyone someone anyone nobody somebody everybody somewhere anywhere everywhere nowhere myself yourself himself herself themselves ourselves itself whatever whenever wherever however though although whether either neither okay lol haha yeah yep nope gonna gotta wanna kinda sorta because tomorrow again until remember forget forgot friend girlfriend boyfriend brother sister mother father husband wife cousin daughter son parents birthday appointment week month year morning evening night afternoon".split());return LEX
def _vf(w:str)->tuple:return tuple(dict.fromkeys([w]+([w[:-3]+'y'] if w.endswith(('ies','ied')) else [])+([w[:-3],w[:-3]+'e']+([w[:-4]] if len(w)>5 and w[-4]==w[-5] else []) if w.endswith('ing') and len(w)>5 else [])+([w[:-2],w[:-1]]+([w[:-3]] if len(w)>4 and w[-3]==w[-4] else []) if w.endswith('ed') and len(w)>4 else [])+([w[:-2],w[:-1]] if w.endswith('es') and len(w)>3 else [w[:-1]] if w.endswith('s') and len(w)>3 else [])+([w[:-2]] if w.endswith("'s") else [])))
def isknown(w:str)->bool:
    c=_cog();return lemma(w) in VB or any(v in _lex() or c is not None and v in c[1]['W'] for v in _vf(w))
@functools.lru_cache(maxsize=65536)
def fix(w:str)->str:
    if len(w)<4 or not w.isalpha() or isknown(w):return w
    L='abcdefghijklmnopqrstuvwxyz';E={**{w[:i]+w[i+1:]:3 if i and w[i]==w[i-1] else 0 for i in range(len(w))},**{w[:i]+c+w[i+1:]:1 for i in range(len(w)) for c in L},**{w[:i]+w[i+1]+w[i]+w[i+2:]:2 for i in range(len(w)-1)},**{w[:i]+c+w[i:]:3 for i in range(len(w)+1) for c in L}};c=_cog()
    K=[e for e in E if e!=w and isknown(e)];K=[e for e in K if any(v in _lex() for v in _vf(e))] or ([e for e in K if len(w)>=5] if c else [])
    return max(K,key=lambda e:(any(v in _lex() for v in _vf(e)),E[e],sum(c[1]['W'].get(v,(0,0))[1]-c[1]['W'].get(v,(0,0))[0] for v in _vf(e)) if c else 0)) if K else w
def _nt1(T,i):
    t=T[i][0];l=t.lower()
    if l.endswith("'s") and len(l)>3 and l[:-2] in CAS:return (CAS[l[:-2]][:1].upper()+CAS[l[:-2]][1:] if t[:1].isupper() else CAS[l[:-2]])+"'s"
    r=CAS.get(l) or (DAB[l] if l in DAB and (i>0 and (T[i-1][0][:1].isdigit() or T[i-1][0].lower() in ('on','this','next','last','every','till','until','by','sat','sun','mon','wed','fri','in','of','early','late','mid','end','since')) or i+1<len(T) and (T[i+1][0].lower() in ('morning','night','evening','afternoon','at','the') or T[i+1][0][:1].isdigit())) else None) or (fix(l) if t.isalpha() and (t.islower() or len(t)>1 and t[1:].islower() and i==0) else None) or l
    return t if r==l else (r[:1].upper()+r[1:] if t[:1].isupper() else r)
def norm(s:str)->str:
    for R,b in _CZ:s=R.sub(b,s)
    return s
def toks(s:str):
    s=_NT.sub(lambda m:m[1]+' not',_CT.sub(lambda m:m[1]+" '"+m[2].lower(),norm(s)));T=[(m[0],m.start(),m.end()) for m in _TOK.finditer(s)];return s,[(_nt1(T,i),a,b) for i,(_,a,b) in enumerate(T)]
def _istime(T,i):
    t=T[i][0].lower().rstrip('.')
    return t in NUMW|{'twelve','eleven','ten','nine','eight','seven','six','five','four','three','two','one','noon'} and i>0 and T[i-1][0].lower() in ('at','around','by','until','till','past','from','to') and (T[i-1][0].lower()!='to' or i>1 and T[i-2][0].lower() in NUMW|{'quarter','half'} or i>1 and _istime(T,i-2)) or 'tim' in wcls(t,1) and t not in ('term','period','spell','moment','while','stage','season','run','shift','turn','time') or (t in TW and (t!='may' or T[i][0][0]=='M')) or bool(re.fullmatch(r"(?:1[89]|20)\d\d|\d{1,2}:\d{2}",t)) or (bool(re.fullmatch(r"\d{1,2}(?:st|nd|rd|th)",t)) and (i>0 and T[i-1][0].lower().rstrip('.') in MONS|{'the'} or i+1<len(T) and T[i+1][0].lower().rstrip('.') in MONS|{'of'})) or (bool(re.fullmatch(r"\d{1,2}",t)) and (i+1<len(T) and T[i+1][0].lower().rstrip('.') in ('am','pm','a.m','p.m',"o'clock") or i>0 and T[i-1][0].lower() in MONS or i>0 and T[i-1][0].lower() in ('at','around','by','until','till') and 0<int(t)<=12)) or (t in NUMW|{'a','couple'} or t.isdigit()) and i+1<len(T) and T[i+1][0].lower() in TW
def tspan(T,i,n):
    if not (_istime(T,i) or T[i][0].lower() in TGL-{'a','an','of','to','and','or'} or T[i][0].lower() in ('half','quarter')):return None
    j=i;last=-1
    while j<n and (_istime(T,j) or T[j][0].lower() in TGL|NUMW|{'quarter'} or T[j][0].isdigit()):
        last=j if _istime(T,j) else last;j+=1
        if j-i>9:break
    if last<0 or last-i>5 and not any(_istime(T,k) for k in range(i,i+3)):return None
    k=last+1
    if k<n and T[k][0].lower()=='old':return None
    while k<n and T[k][0].lower() in ('ago','later'):k+=1
    k=k+2 if k+1<n and T[k][0].lower() in ('after','before') and T[k+1][0].lower() in ('that','then','this','next') else k
    return (i,k)
FAV={'favourite':'favorite','fave':'favorite','go-to':'favorite','fav':'favorite'}
NPSTOP=S_("and or but because so who which that where when why how what if as than then too again named called due not")
def _np(T,i,n,chain=True):
    j=i;lab=[];name=[];attr=None
    while j<n and T[j][0].lower() in ADJ:j+=1
    while j<n and len(lab)<4:
        t=T[j][0];tl=FAV.get(t.lower().rstrip('.'),t.lower().rstrip('.'))
        if lab and t==',' and j+2<n and T[j+1][0][:1].isupper() and T[j+2][0] in (',','.',';') and (isper(lab[-1]) or lab[-1] in ('friend','partner','pet','puppy','kitten')) and T[j+1][0].lower() not in TW|{'i','i\'m'}:name.append(T[j+1][0]);j+=3;break
        if not re.fullmatch(r"[A-Za-z][\w'-]*\.?",t) or tl in AUX or tl in PREPW or tl in DET or tl in ADV or tl in NPSTOP or tl in PRONS or tl in ('i','we','you','me','us','with','for','about','during','after','before','since','until') or lab and (tl in BAREW or tl in FRW and not t[:1].isupper() and len(lab)>0 and tl not in ATTRN or wpos(tl)=={2} or wpos(tl)=={2,3} or 1 in wpos(tl) and j+1<n and T[j+1][0].lower() in PART and tl not in ATTRN):break
        if lab and (t[0].isupper() or tl.isalpha() and len(tl)>2 and not isknown(tl.replace("'s",''))) and (isper(lab[-1]) or lab[-1] in ('friend','partner','pet','puppy','kitten')):
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
    lab=[x[:-2] if x.endswith("'s") and k==len(lab)-1 else x for k,x in enumerate(lab)];lab=lab[:-1]+[KIN.get(lab[-1],lab[-1])] if lab else lab
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
_cache={};EHDR=struct.Struct('<4sHH8s16s12s');EVER=2;NF=getattr(os,'O_NOFOLLOW',0)
def enabled()->bool:return os.environ.get('AMNI_UMEM_TOROID','0')=='1' and U.enabled()
def path(key:str)->str:return os.path.join(U.root(),re.sub(r'[^0-9a-f]','',key)[:32]+'.torus')
def kdir()->str:return os.environ.get('AMNI_UMEM_KEY_DIR') or os.path.join(U.root(),'keys')
def kpath(key:str)->str:return os.path.join(kdir(),re.sub(r'[^0-9a-f]','',key)[:32]+'.key')
def _strict(p:str,d:bool):
    s=os.lstat(p)
    if not (stat.S_ISDIR if d else stat.S_ISREG)(s.st_mode) or os.name=='posix' and (stat.S_IMODE(s.st_mode)&0o077 or s.st_uid!=os.getuid()):raise PermissionError('toroid key must be a private file (0600) in a private directory (0700)')
def _kread(p:str)->bytes:
    _strict(os.path.dirname(p),True);_strict(p,False);fd=os.open(p,os.O_RDONLY|NF)
    with os.fdopen(fd,'rb') as f:k=f.read(33)
    if len(k)!=32:raise ValueError('toroid key invalid')
    return k
def _dsync(d:str):
    try:fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
    except OSError:pass
def _scrub(fd:int):
    n=os.fstat(fd).st_size;os.lseek(fd,0,0);[os.write(fd,secrets.token_bytes(min(65536,n-i))) for i in range(0,n,65536)];os.fsync(fd)
def _wfile(p:str,b:bytes,shred:bool=False):
    d=os.path.dirname(p);os.makedirs(d,mode=0o700,exist_ok=True);t=p+f'.{os.getpid()}.tmp';fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|NF,0o600)
    with os.fdopen(fd,'wb') as fh:fh.write(b);fh.flush();os.fsync(fh.fileno())
    o=os.open(p,os.O_RDWR|NF) if shred and os.name=='posix' and os.path.isfile(p) and not os.path.islink(p) else None
    try:os.replace(t,p);o is not None and _scrub(o)
    finally:o is not None and os.close(o)
    _dsync(d)
def _shred(p:str)->bool:
    if not os.path.lexists(p):return False
    try:
        fd=os.open(p,os.O_RDWR|NF)
        try:_scrub(fd)
        finally:os.close(fd)
    except OSError:pass
    os.remove(p);_dsync(os.path.dirname(p));return True
def _uk(kf:bytes,key:str)->bytes:return _hkdf_sha256(kf,32,b'amni-toroid-v2',b'amni-toroid-user\x00'+key.encode())
def _kid(uk:bytes)->bytes:return _hkdf_sha256(uk,8,b'\x00'*32,b'amni-toroid-kid')
def seal(inner:bytes,uk:bytes)->bytes:
    sa,nc=secrets.token_bytes(16),secrets.token_bytes(12);h=EHDR.pack(MAGIC,EVER,0,_kid(uk),sa,nc);return h+_aenc(_hkdf_sha256(uk,32,sa,b'amni-toroid-page'),nc,inner,h)
def unseal(raw:bytes,uk:bytes)->bytes:
    if len(raw)<EHDR.size+16:raise ValueError('toroid store corrupt')
    mg,ver,_,kd,sa,nc=EHDR.unpack_from(raw);h=raw[:EHDR.size]
    if mg!=MAGIC or ver!=EVER or not hmac.compare_digest(kd,_kid(uk)):raise ValueError('toroid store key mismatch')
    try:return _adec(_hkdf_sha256(uk,32,sa,b'amni-toroid-page'),nc,raw[EHDR.size:],h)
    except Exception:raise ValueError('toroid store authentication failed') from None
def _kcommit(kp:str):_shred(kp);os.replace(kp+'.new',kp);_dsync(os.path.dirname(kp))
def _ukey(key:str,kd:bytes)->bytes:
    kp=kpath(key)
    for p in (kp,kp+'.new'):
        uk=_uk(_kread(p),key) if os.path.lexists(p) else None
        if uk and hmac.compare_digest(_kid(uk),kd):(_kcommit(kp) if p!=kp else _shred(kp+'.new'));return uk
    raise ValueError('toroid key missing or wrong')
def _head(p:str):
    try:
        with open(p,'rb') as f:b=f.read(EHDR.size)
    except FileNotFoundError:return None,False
    return (b[8:16] if len(b)==EHDR.size and b[:4]==MAGIC and struct.unpack_from('<H',b,4)[0]==EVER else None),b[:4]==MAGIC and len(b)>=6 and struct.unpack_from('<H',b,4)[0]==VER
def load(key:str)->Store:
    p=path(key)
    with U._lock:
        st=os.stat(p).st_mtime_ns if os.path.exists(p) else None;c=_cache.get(key)
        if c and c[0]==st:return decode(c[1])
        if not st:_cache[key]=(None,encode(Store()));return Store()
        raw=open(p,'rb').read()
        if raw[:4]==MAGIC and len(raw)>=6 and struct.unpack_from('<H',raw,4)[0]==VER:s=decode(raw);save(key,s);return s
        if len(raw)<EHDR.size or raw[:4]!=MAGIC:raise ValueError('toroid store corrupt')
        inner=unseal(raw,_ukey(key,raw[8:16]));_cache[key]=(st,inner);return decode(inner)
def save(key:str,st:Store,rotate:bool=False):
    p=path(key)
    with U._lock:
        os.makedirs(U.root(),mode=0o700,exist_ok=True);_cache.pop(key,None)
        if not len(st.s) and not len(st.u):wipe(key);return
        kd,leg=_head(p);kp=kpath(key);fresh=rotate or leg or kd is None and not os.path.lexists(kp)
        if fresh:kf=secrets.token_bytes(32);_wfile(kp+'.new',kf);_strict(kdir(),True);uk=_uk(kf,key)
        else:uk=_ukey(key,kd) if kd is not None else _uk(_kread(kp),key)
        _wfile(p,seal(encode(st),uk),shred=rotate or leg);fresh and _kcommit(kp)
def wipe(key:str)->bool:
    with U._lock:
        _cache.pop(key,None);r=_shred(path(key));[_shred(x) for x in (kpath(key),kpath(key)+'.new')];return r
_sg=lambda w:w[:-1] if len(w)>3 and w.endswith('s') and not w.endswith(('ss','us','is')) else w
def _resolve(lab:str,known)->str:
    if not lab or lab=='me' or lab in known:return lab
    L=lab.split();tl={_sg(x) for x in L};c=[k for k in known if k!='me' and (_sg(k.split()[-1])==_sg(L[-1]) and tl<={_sg(x) for x in k.split()} or k.startswith(lab+' ') or k.startswith(lab+"'s "))]
    return c[-1] if c else lab
KIN=dict(x.split(':') for x in "boy:son lad:son girl:daughter nan:grandma nana:grandma gran:grandma granny:grandma grandmother:grandma grandad:grandpa granddad:grandpa grandfather:grandpa gramps:grandpa hubby:husband missus:wife bestie:friend mate:friend bff:friend mother:mom mama:mom papa:dad pops:dad father:dad fiancee:fiance gf:girlfriend bf:boyfriend flatmate:roommate housemate:roommate".split())
REPV=S_("say show confirm mean read list");COOKV=S_("cook make bake brew fry grill roast steam boil prepare");PART=S_("up out off down back over away around")
PURP=S_("health safety sake job work fun business school money love exercise research career studies family kids savings retirement budget relaxation comfort convenience privacy emergencies emergency reasons reason wedding birthday anniversary graduation funeral reunion conference interview meeting vacation holiday trip environment")
_FILL=re.compile(r"(?i)\b(?:after all|these days|nowadays|lately|right now|at the moment|for now|for sure|lol|lmao|haha+|tbh|ngl|imo|af|fr|i guess|i think|i reckon|honestly|literally|basically|tho|rn|lowkey|highkey)\b")
_REP=re.compile(r"(?i)^(?:(?:it\s+)?turns out|apparently|(?:the\s+)?[a-z]+(?:\s+[a-z]+)?\s+(?:says|said|confirmed|told me|thinks)|i (?:just )?(?:found out|heard|learned|realized|noticed|checked))(?:\s+that)?[\s,]+(?=(?:my|the|our|his|her|it|it's|he|she|they|i|we)\b)")
_SPL=re.compile(r"(?<=[.!?;])\s+|\s*:\s+(?=[a-z])|,?\s+(?:while|whereas|whilst)\s+(?=(?:little|big|old|young|baby)\s+[A-Z])|,\s+(?=[A-Z][a-z]+(?:'s|\s+(?:is|was|has|had|does|did|lives|works|moved|went))\b)|,?\s+(?:and|but|while|whereas|whilst)\s+(?=[A-Z][a-z]+\s+(?!and\b|or\b)[a-z]+)|\s*;\s*|\s+(?:saying|to say|says|said|told me|reckons|thinks)\s+(?:that\s+)?(?=(?:it|it's|he|she|they|the|my|our|his|her|their|we)\b)|,\s+so\s+(?=(?:the|my|our|his|her|their|it|that|this)\b)|,?\s+(?:and|but|while|whereas|plus)\s+(?=(?:my|i|i'm|he|she|they|it|it's|we|our|his|her|their|the)\b)|,\s+(?=(?:[Mm]y|I|I'm|[Hh]e|[Ss]he|[Tt]hey|[Ww]e|[Ii]t|[Ii]t's|[Hh]is|[Hh]er|[Tt]heir|[Oo]ur)\b)|,\s+(?=(?:and\s+)?(?:[Tt]he\s+|[Mm]y\s+|[Oo]ur\s+)?[A-Za-z]+(?:\s+[a-z]+)?\s+(?:is|are|was|were)\b)")
_RW=[(re.compile(r"(?i)^(?:you know|remember|did i (?:tell|mention to) you|guess what|fun fact|so basically)\W+(?:how\s+|that\s+|about\s+)?(?=(?:my|our|i|i'm|we|he|she|they|the)\b)"),''),(re.compile(r"(?i)^(?:so\s+|and\s+)?the\s+(?:whole\s+|real\s+|main\s+|only\s+|actual\s+|big\s+)?reason\s+(?:why\s+|that\s+)?(.+?),?\s+(?:is|was|'s)\s+(?:(?:just|simply|basically|honestly|really)\s+)?(?:that\s+|because\s+|bc\s+|cuz\s+|cause\s+)?(.+)$"),r"\1 because \2"),(re.compile(r"(?i)^(.+?),?\s+(?:so|which is why|that's why|that is why|and that's why|hence|and so)\s+((?:i|we|he|she|they|there|it|that|this|nobody|everyone|my\s+[\w'-]+|our\s+[\w'-]+|(?-i:(?!I')[A-Z][\w-]+(?:'s)?))(?:'s)?\s+(?!(?:can|could|will|would|might|may|won't|wouldn't|can't|don't|have to|need to)\b).+)$"),r"\1. \2 because \1"),(re.compile(r"(?i)^the\s+way\s+(.+?)\s+(?:is|was|'s)\s+(?:(?:just|basically|mostly|always|still)\s+)?(?:by\s+)?(.+)$"),r"\1 by \2"),(re.compile(r"(?i)^((?:my|our|his|her|their)\s+[\w-]+)(?:'s)?\s+(?:approach|method|system|trick|technique|routine|strategy|way|solution|go-to)\s+(?:is|was|'s)\s+(?:(?:just|basically|mostly|always|still)\s+)?(.+)$"),r"\1 does it by \2"),(re.compile(r"(?i)^(?:with|now that|ever since|given that|seeing as)\s+(.+?),\s+(.+)$"),r"\1. \2")]
_EMO=re.compile(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
DROPX=S_("love hope need want like thank thanks guess remember forget let please check look see imagine make try get go tell call text send wish mind")
def _pro(c:str)->str:
    s0,T=toks(c);n=len(T);l0=T[0][0].lower() if n else ''
    if n>2 and (l0 in IRR and IRR[l0] not in ('be','have','do') and l0 not in ('left','lost','found','made','told','said') or l0.endswith('ed') and l0 not in ('used','based','located','married') and wv(l0) or isv(l0) and l0 not in NOUNISH|DROPX and not l0.endswith(('s','ing'))) and T[1][0].lower() not in AUX|{'is','are','was','to'}:return 'I '+c
    tt=tspan(T,0,n) if n>3 else None
    if tt and tt[1]<n-1 and (T[tt[1]][0]==',' or T[tt[1]][0].lower() in ('my','our','i','we','he','she','they','his','her','their') or T[tt[1]][0][:1].isupper() and T[tt[1]][0].lower() not in AUX):k=tt[1]+(T[tt[1]][0]==',');return s0[T[k][1]:]+' '+s0[:T[tt[1]][1]].strip(' ,')
    return c
_vless=lambda x:len(x.split())<=4 and re.match(r"(?i)(?:my|our|his|her|their)\s",x) is not None and not any(isv(w.lower()) or w.lower() in AUX or wv(w.lower()) for w in x.split()[1:])
def _merge(P:list)->list:
    R=[]
    for x in P:(R.__setitem__(-1,R[-1]+' and '+x) if R and _vless(R[-1]) and not re.match(r"(?i)\s*(?:my|our|i|we)\b",x) else R.append(x))
    return [x for x in R if x]
def clauses(msg:str):
    m=_EMO.sub(' ',msg.strip());m=_NOT.sub(' ',m) if re.search(r"(?i)\bnot\b",m) and (_CORR.search(m) or re.search(r"(?i),\s*not\b|^\W*(?:no\W+)?not\b",m)) else m
    for sn in re.split(r"(?<=[.!?])\s+",m):
        sn=norm(_LEAD.sub('',sn.strip()).rstrip(' .!?'))
        for R,b in _RW:sn=R.sub(b,sn)
        for c in _merge([x for x in _SPL.split(_FILL.sub(' ',sn)) if x]):
            c=_LEAD.sub('',c.strip())
            while (c2:=_REP.sub('',c))!=c:c=_LEAD.sub('',c2)
            c=re.sub(r"(?i)^(?:make (?:that|it)|change (?:that|it) to|i meant)\s+",'',c).strip(' ,.!-')
            c=_pro(c) if c else c
            c and (yield c)
_placey=lambda ow:bool(ow) and (isplace(ow[-1]) or len(ow)>1 and ow[-2] in PLACEN and re.fullmatch(r"[a-z]|\d+",ow[-1]) is not None)
PETS=S_("dog cat puppy kitten pet horse rabbit parrot hamster ferret turtle gecko goldfish hedgehog iguana cockatiel chinchilla tortoise snake budgie pup")
def _vtype(ow,TT):
    o=[t for t in ow if t not in DET and t not in ADV]
    return None if not o else ('what','color') if all(iscol(t) or t in ('dark','light','bright','pale','deep') for t in o) and any(iscol(t) for t in o) else ('what','age') if re.fullmatch(r"(?:\d{1,3}|"+'|'.join(NUMW)+r")(?: (?:years?|months?|weeks?) old)?",' '.join(o)) else ('when','') if TT and any(_istime(TT,i) for i in range(len(TT))) and not any(t[0][:1].isupper() and 'loc' in wcls(t[0],1) for t in TT) else ('where','location') if _placey(o) or len(o)<=3 and 'loc' in wcls(o[-1],1) and TT and TT[-1][0][:1].isupper() else ('what','name') if len(o)<=2 and all(t[0][:1].isupper() for t in TT if t[0].lower() in o) else None
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
    elif lw[0] in ('the','some','a','an','every','another','this') or lw[0] not in ADV and lw[0] not in AUX and lw[0] not in PREPW and lw[0] not in TW and (U.bh(lw[0].encode())&U.WM) not in U.FUN and not isv(lw[0]) and lw[0] not in ('ugh','hmm','yeah','cool','thanks','nice','any','no','yes','ok','okay'):
        lab,_,p2,j=_np(T,1 if lw[0] in ('the','some','a','an','every','another','this') else 0,n);lw=[t[0].lower() for t in T];jj=j
        if lab and jj+1<n and lw[jj] in ('i','we','you','he','she','they') and (isv0(lw[jj+1]) or wv(lw[jj+1],lw[jj])):jj+=2+(jj+2<n and lw[jj+2] in PREPW|{'for','with','about','at','to'})
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
        if t in ('because','cause',"'cause",'coz','cuz') or t=='since' and x+1<n and not _istime(T,x+1) and not nx.isdigit() or t=='so' and (nx=='that' or nx in ('i','we','he','she','they','my','it','the','you','her','his') and any(q in ("can","could","would","will","'ll","'d","might","may","wo","ca","wouldn't","won't","can't") for q in lw[x+2:x+4])) or t=='in' and nx=='order' or t=='to' and (nx in VB or 1 in wpos(nx) and 0 not in wpos(nx) and nx.isalpha()) and nx not in ('say','tell','ask','let','check','confirm','see','know','hear') and not (nx in NOUNISH|{'work','school','bed','church','class','practice','college','town','camp','market','lunch','dinner','breakfast','brunch','court'} and (lemma(lw[x-1]) in MOTION or nx in ('work','school','bed','church','class','college','town','market','lunch','dinner','breakfast','brunch','court','camp'))) and x>0 and lemma(lw[x-1]) not in CATEN and lw[x-1] not in CATEN and lw[x-1] not in DET or t=='for' and any(q in PURP for q in lw[x+1:x+4]) and not (x+1<n and _istime(T,x+1)):y=('why','',x,n)
        elif (tt:=tspan(T,x,n)):y=('when','',tt[0],tt[1])
        elif t=='by' and (e:=next((q for q in range(x+1,min(n,x+4)) if ismeans(lw[q]) or lw[q].endswith('ing') and q==x+1),None)) is not None and all(lw[q] not in PREPW|AUX for q in range(x+1,e)):y=('how','',x,e+1)
        elif t=='on' and nx=='foot' or t=='from' and nx=='scratch' or t=='in' and nx in ('cash','person','bulk'):y=('how','',x,x+2)
        elif t=='on' and nx in DET and x+2<n and ismeans(lw[x+2]) or t=='over' and nx=='the' and x+2<n and lw[x+2] in ('phone','internet'):y=('how','',x,x+3)
        elif t=='over' and (ismeans(nx) or nx in ('zoom','facetime','whatsapp','slack','chat','video')):y=('how','',x,x+2)
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
                    if (isplace(lw[e-1]) or lw[e-1] in BAREW) and not (dt and e<n and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|PRONS|TGL and not isv(lw[e])):break
                e=e if e>z and (isplace(lw[e-1]) or lw[e-1] in BAREW or dt and (t!='to' or pcls(lemma(verb)) in ('go','live') or lemma(verb) in MOTION)) else z
            while e>z and e<n and e<z+6 and (re.fullmatch(r"[A-Z]|\d+",T[e][0]) and isplace(lw[e-1])):e+=1
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
    b=next((q for q in range(a+1,b) if T[q][0]==',' and q+1<b and T[q+1][0].lower() in ('which','who','whom','whose','but','though','although','whereas','while','so') or T[q][0] in (':',';') or T[q][0]==',' and q+2<b and T[q+1][0].lower()=='and' and T[q+2][0].lower() in ('then','i','we','he','she','they','my','our','the','his','her','their')),b) if w in ('why','how','what','where') else b
    b=next((q for q in range(a+1,b) if T[q][0].lower() in ('before','until','till','while','whenever','whereas','although','though') or T[q][0].lower() in ('when','because','since') and w!='why'),b) if w in ('how','what','where') else b
    while b>a and T[b-1][0].lower() in ADV|{'too','again','though','lol','anymore','now','as','well','here','there','and','or','but','then','with','of','for','the','a','an',',','instead'}:b-=1
    return re.sub(r"\s+",' ',c[T[a][1]:T[b-1][2]]).strip(" ,.;:!?\"") if b>a else ''
def _wattr(w,a,pred,verb,lw,x,v=''):
    if w!='where':return w,a
    pc=pcls(pred);hv=v.lower().split()[-1] if v else '';return ('how','') if pred in COOKV else (w,'location' if pc=='live' and hv and 'sto' in wcls(hv) else 'home' if pc=='live' else 'work' if pc=='work' else 'origin' if lw[x]=='from' else 'destination' if pc=='go' or pred in MOTION and lw[x] in ('to','into') else 'location')
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
            v=_val(cc,T,x,y,w);pr_=lemma(vb) if vb else pred;w2,a2=_wattr(w,a,pr_,vb,lw,x,v);verb=vb
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
                at=A['pa'][1] if A['pa'] else 'model' if ent!='me' and {'veh','tool'}&wcls(ent.split()[-1],1) and any(t[0][:1].isupper() or t[0][:1].isdigit() for t in T[a0:b0]) else 'color' if all(iscol(t) or t in ('dark','light','bright','pale','deep','and','navy','sky','forest','hot') for t in ow if t not in DET|ADV) and any(iscol(t) for t in ow) else 'age' if re.fullmatch(r"(?:\d{1,3}|"+'|'.join(NUMW)+r")(?: (?:years?|months?|weeks?) old)?",v.lower()) else 'breed' if (ent.split()[-1] in PETS or 'ani' in wcls(ent.split()[-1],1)) and ow and ow[0] in ('a','an') else 'job' if isper(ent.split()[-1] if ent!='me' else 'mom') and ow and ow[0] in ('a','an') else 'name' if isper(ent.split()[-1] if ent!='me' else 'x') and v[:1].isupper() and len(v.split())<=2 and ow[0] not in DET else 'where' if cor and _placey([t for t in ow if t not in DET]) else 'kind'
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
FRW={**{w:'why' for w in "reason reasons point purpose motive motivation thinking logic rationale excuse cause story".split()},**{w:'how' for w in "way method system approach trick technique routine process setup strategy means commute ride recipe hack".split()},**{w:'when' for w in "time date day night week month year hour deadline".split()},**{w:'where' for w in "place city town country spot address location street neighborhood neighbourhood venue area".split()}}
WHYV=S_("brought bring brings make made makes caused cause led lead prompted prompt pushed push drove drive convinced convince inspired inspire triggered trigger motivated motivate")
FSK=S_("real actual main whole exact usual big specific go-to deal thing")
WNA=dict(per='person.n.01',ani='animal.n.01',loc='location.n.01 structure.n.01 room.n.01 facility.n.01 establishment.n.04 geographical_area.n.01 region.n.03',sto='container.n.01 furniture.n.01 storage_space.n.01',veh='conveyance.n.03',tim='time_period.n.01 time_unit.n.01 clock_time.n.01 calendar_day.n.01',col='color.n.01',food='food.n.01 food.n.02',occ='worker.n.01 professional.n.01 skilled_worker.n.01 expert.n.01 employee.n.01 creator.n.02',evt='social_event.n.01 meeting.n.01 gathering.n.01 ceremony.n.01 examination.n.01 appointment.n.01',tool='instrumentality.n.03 implement.n.01 device.n.01',comm='communication.n.02 communication.n.01 telecommunication.n.01',act='activity.n.01',pay='currency.n.01 payment.n.01 medium_of_exchange.n.01',dog='dog.n.01',cat='cat.n.01')
@functools.lru_cache(maxsize=1)
def _wna()->dict:
    c=_cog();return {k:frozenset(c[1]['ix'][x] for x in v.split() if x in c[1]['ix']) for k,v in WNA.items()} if c else {}
@functools.lru_cache(maxsize=65536)
def _syn(v:str,pos:int)->tuple:
    c=_cog()
    if not c or v not in c[1]['W']:return ()
    G=c[1];r=G['W'][v];return tuple(sorted((int(x) for x in G['ws'][r[0]:r[1]] if int(G['pos'][int(x)])==pos),key=lambda i:(G['nm'][i].rsplit('.',2)[0].replace('_',' ')!=v.replace('-',' '),int(G['nm'][i].rsplit('.',1)[1]))))
@functools.lru_cache(maxsize=65536)
def wcls(w:str,k:int=2)->frozenset:
    c=_cog()
    if not c or not w:return frozenset()
    C,G=c;A=_wna();S=[i for v in dict.fromkeys([w.lower(),w.lower().replace('-',' ')]+list(_vf(w.lower()))) for i in _syn(v,0)][:k]
    return frozenset(nm for nm,a in A.items() if any(a&set(C.anc(int(i))) for i in S))
@functools.lru_cache(maxsize=65536)
def wpos(w:str)->frozenset:
    c=_cog()
    if not c or not w:return frozenset()
    G=c[1];return frozenset(int(G['pos'][int(i)]) for v in _vf(w.lower()) for r in [G['W'].get(v)] if r for i in G['ws'][r[0]:r[1]])
def wv(t:str,prev:str='')->bool:
    P=wpos(t);return isv0(t) or 1 in P and t not in DET|AUX|PREPW|PRONS and (t.endswith(('ing','ed')) or 0 not in P or prev in AUX|PRONS|{'i','we','you','to'})
_QN=re.compile(r"(?i)\b([a-z]+)'s\b(?=\s+(?:my|the|your|our|his|her|it|this|that|a|an|i|we|due|being|getting|going|coming|scheduled|happening|located|held|parked|kept|stored)\b)")
_QWHY=re.compile(r"(?i)\b(?:what(?:'s| is| was)? the (?:point|reason|purpose|story)|for what reason|what for|how come|what made (?:me|us|him|her|them))\b")
def ask(q:str,last:str='',known=(),names=None,lp=''):
    names=names or {};fp=bool(re.search(r"(?i)\b(?:did|do|had|have) (?:i|we) (?:say|said|tell|told|mention)|\bi (?:said|told|mentioned)\b",q));m=_QSAY.sub(' ',_QPRE.sub('',_QN.sub(lambda x:x[1]+' is',norm(q.strip()))));m=re.sub(r"\s+",' ',m).strip(' ?.!,');m,T=toks(m);lw=[t[0].lower() for t in T];n=len(T);rw=[m[a:b].lower() for _,a,b in T];nm=lambda i:rw[i] if rw[i] in names else rw[i][:-2] if rw[i].endswith("'s") and rw[i][:-2] in names else None
    if not n:return None
    wi=next((i for i,t in enumerate(lw) if t in ('what','which','who','whom','whose','where','when','why','how')),None)
    if wi is None and not {'allergic','allergy'}&set(lw):return None
    wi=0 if wi is None else wi;t=lw[wi];nx=lw[wi+1] if wi+1<n else '';nx2=lw[wi+2] if wi+2<n else '';w={'what':'what','which':'what','who':'what','whom':'what','whose':'what','where':'where','when':'when','why':'why','how':'how'}.get(t,'what');a='';cand='';fu=set()
    j=wi+1
    while j<n and j<wi+7 and (lw[j] in AUX|DET|ADV|FSK|PRONS or lw[j] in ('i','we','you','say','said','mention','mentioned','tell','told','the','usual')) and lw[j] not in FRW:j+=1
    fr=lw[j] if j<n and t in ('what','which') and (lw[j] in FRW or j==wi+1 and 'loc' in wcls(lw[j],1)) else '';fr and fu.add(j);jv=j
    while jv<n and jv<wi+6 and (lw[jv] in AUX or lw[jv] in ('i','we','say','said','mention','mentioned','tell','told','that','it') or lw[jv] in DET|ADV):jv+=1
    if _QWHY.search(q) or t=='what' and (FRW.get(fr)=='why' or jv<n and lw[jv] in WHYV and jv+1<n and (lw[jv+1] in ('me','us','him','her','them','my','our','his','her','their') or lw[jv+1] in names)):w='why'
    elif fr and t in ('what','which') and FRW.get(fr,'where')!='what':w=FRW.get(fr,'where')
    elif t in ('what','which') and _sg(nx) in ('time','day','date','year','month','hour','night','week'):w='when'
    elif t=='how' and nx=='long' and 'ago' in lw:w='when'
    elif t=='what' and 'for' in lw[-1:]:w='why'
    elif t=='how' and nx=='old':w,a='what','age'
    elif t=='how' and nx in ('many','much','far','long','often'):w='what'
    elif t in ('what','which') and nx in ('color','colour'):a='color';cand=lw[wi+3] if nx2=='of' and wi+3<n else ''
    elif t in ('what','which') and nx in ('kind','type','sort','make','model','brand','breed'):a='model' if nx in ('make','model','brand') else 'breed' if nx=='breed' or nx2=='of' and wi+3<n and 'ani' in wcls(lw[wi+3],1) else 'kind';cand=lw[wi+3] if nx2=='of' and wi+3<n else ''
    elif t in ('who','whom'):a='name'
    elif t in ('what','which') and nx in ('job','name','age'):a=nx
    elif t in ('what','which') and nx and nx not in ATTRN and nx not in AUX and nx not in DET:cand=nx
    J=' '.join(lw)
    if w=='what' and not a:a='allergy' if {'allergic','allergy','allergies'}&set(lw) else 'likes' if re.search(r"\b(?:into|goes for|go for|obsessed with|crazy about|fan of|likes?|loves?|enjoys?|a fan of)\b(?: \w+)?(?: \w+)?$",J) or re.search(r"\b(?:into|goes for|go for|obsessed with)\b",J) else 'job' if re.search(r"\bdo(?:es)?\b(?: \w+){0,3} (?:do|doing)(?: for (?:a )?(?:living|work|a job|money))?(?: these days| now| nowadays| all those years)?$|\bfor (?:work|a living)\b|\btrain\w* to (?:become|be)\b",J) else 'name' if re.search(r"\b(?:call|calling|called|name|naming|named)\b",J) else 'favorite' if 'favorite' in lw else a
    a='favorite' if a=='likes' and {'best','most'}&set(lw) else a
    cn=_np(T,wi+1,n,False)[0] if t in ('what','which') else ''
    if w=='what' and not a and t=='which' and cn and ({'tool','veh'}&wcls(cn.split()[-1]) or cn.split()[-1] in ('model','make','brand')):a,cand='model',cn
    if w=='what' and not a and re.search(r"\b(?:use|using|used)\b|\bwork(?:s|ed)? (?:best )?for\b",J):w='how'
    if w=='when' and 'birthday' in lw:a='birthday'
    ents=[];pa=None;pred='';i=0;used=set(fu);sec=[]
    while i<n:
        if i in fu:i+=1;continue
        if lw[i] in ('my','our'):
            lab,_,p2,j=_np(T,i+1,n);lw=[x[0].lower() for x in T];used|=set(range(i,j))
            lab=re.sub(r"\s+(?:thing|stuff)$",'',lab)
            if lab:ents.append(_resolve(lab,known))
            elif p2:ents.append('me')
            pa=pa or p2;i=max(j,i+1)
        elif lw[i] in ('i','we','me','us'):ents.append('me');i+=1
        elif lw[i] in ('he','she','they','him','her','them') and i!=wi and (lp or last):ents.append(lp or last);i+=1
        elif lw[i]=='it' and i!=wi and last:ents.append(last);i+=1
        elif nm(i):ents.append(names[nm(i)]);used.add(i);i+=1
        elif lw[i]=='the' and i+1<n and any(k in fu for k in range(i+1,min(n,i+4))):i+=1
        elif lw[i]=='the' and i+1<n:
            lab,_,p2,j=_np(T,i+1,n);lw=[x[0].lower() for x in T];used|=set(range(i,j));lab=re.sub(r"\s+(?:thing|stuff)$",'',lab)
            lab and lab not in FRW and ents.append(_resolve(lab,known));i=max(j,i+1)
        elif i!=wi and i>0 and re.fullmatch(r"[a-z][a-z-]+(?:'s)?",rw[i]) and (T[i][0][:1].isupper() or rw[i].endswith("'s")) and not isknown(rw[i].replace("'s",'')) and rw[i] not in names:ents.append(rw[i].replace("'s",''));used.add(i);i+=1
        elif i!=wi and isper(lw[i]) and lw[i] not in PRONS|AUX and not isv0(lw[i]) and 0 in wpos(lw[i]) and lw[i] not in ('who','i'):ents.append(_resolve(lw[i],known));used.add(i);i+=1
        else:
            for L in (2,1):
                g=' '.join(FAV.get(x,x) for x in lw[i:i+L])
                if i+L<=n and not used&set(range(i,i+L)) and lw[i] not in AUX|DET|PREPW|{'what','where','when','why','how','which','who'} and (r:=_resolve(g,known)) in known and r!='me':sec.append(r);used|=set(range(i,i+L));break
            i+=1
    if pa and (not a or a==pa[1]):w=pa[0] if w=='what' else w;a=pa[1] if pa[0]==w else a
    fa=next((k for k in range(wi+1,n) if FRW.get(lw[k]) in ('why','how','when') and k not in used and not (k+1<n and re.fullmatch(r"[a-z][a-z-]+",lw[k+1]) and lw[k+1] not in PREPW|AUX|DET|PRONS|FUN_Q|{'i','we','you','that','behind','with','for','of','on'})),None) if w=='what' and not a else None
    if fa is not None:w=FRW[lw[fa]];used.add(fa);ents=[e for e in ents if e!=lw[fa]]
    for j in range(wi+1,n):
        if j in used or lw[j] in AUX or lw[j] in ADV or lw[j] in ('my','i','we','you','the','usually','again','me','us','end','ended','ending') or lw[j] in DET or lw[j] in FRW:continue
        if (isv0(lw[j]) or wv(lw[j],lw[j-1])) and lw[j] not in ('called','named') and lw[j]!=cand:pred=lemma(lw[j]) if isv0(lw[j]) else next((v for v in _vf(lw[j]) if 1 in wpos(v)),lw[j]);break
    if 'called' in lw or 'named' in lw:a=a or ('name' if w=='what' else a)
    pc=pcls(pred)
    if w=='what' and not a and re.search(r"\bdo(?: for (?:a )?(?:living|work|a job))?$",' '.join(lw)):a,pred='job','work'
    if w=='what' and pc in ('drive',) and not a:a='model'
    if w=='where' and not a:a='work' if {'job','work','working','teach','employed','hired','office'}&set(lw) else 'destination' if pred in ('send','ship','deliver','post','mail','move') and 'from' not in lw else 'home' if pc=='live' else 'work' if pc=='work' else 'origin' if 'from' in lw[-1:] or pred in ('','be','come','grow') and 'from' in lw else 'destination' if pc=='go' and pred!='get' or 'to' in lw[-1:] or pred in MOTION and pred not in ('get','take') else 'location'
    nk=_resolve(cand,known) if cand else ''
    ents=(['me'] if fp and not ents else [])+([nk] if nk in known else [])+ents+sec;ents=list(dict.fromkeys(ents)) or ([last] if last else [])
    cw=[x for x in dict.fromkeys(_sg(lemma(x[:-2] if x.endswith("'s") else x)) for k,x in enumerate(lw) if k not in fu and re.fullmatch(r"[a-z][a-z'-]+",x) and x not in AUX|DET|PREPW|ADV|PRONS|FUN_Q and len(x)>2)]
    ents=ents or ([names[x] for x in cw if x in names][:1]) or [x for x in cw if x in known][:1] or ([cw[0]] if cw and not ents else ents)
    return dict(ents=ents,w=w,attr=a,pred=pred,cw=cw)
FUN_Q=S_("what which who whom whose where when why how me my mine you your yours we our us i remind remember recall tell told say said mention mentioned again lol now these those days anymore still ever usually normally currently exactly end ended up thing stuff know think the its did does can could would should will")
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
        cp=qa_ in KB and int(a[i]) in KB;sc=0.5*(qa_==0 and int(a[i]) in (0,AI['kind'],AI['model']))+(3.0 if int(e[i])==int(qe[0]) else 2.0)+2.0*(qa_>0 and int(a[i])==qa_)+1.0*(cp and int(a[i])!=qa_)-2.5*(qa_>0 and int(a[i])>0 and int(a[i])!=qa_ and not cp and not _acomp(AT[qa_],AT[int(a[i])]))+0.5*_acomp(AT[qa_],AT[int(a[i])])-0.5*(qa_>0 and int(a[i])==0)+1.5*(qp is not None and int(pr[i])==qp)+0.5*(ql is not None and int(lm[i])==ql)+float(cf[i])-0.002*float(ag[i])
        out.append((sc,int(i)))
    return sorted(out,reverse=True)
def keys(msg:str,F,known=()):
    ks=[e for e,*_ in F]+[x for c in clauses(msg) for T in [toks(c)[1]] for x in [_np(T,j+1,len(T))[0] for j,t in enumerate(T) if t[0].lower() in ('my','our')]]+(['me'] if re.search(r"(?i)\b(?:i|i'm|i've|i'd|i'll|me|we)\b",msg) else [])
    ks=[_resolve(k,known) for k in ks if k and k!='?'];cw=[_lm(t) for t,_,_ in toks(msg)[1] if re.fullmatch(r"[a-z][a-z'-]{2,}",t.lower()) and t.lower() not in U._STOPW.split() and t.lower() not in AUX|ADV|DET|PREPW|PRONS|FUN_Q|TGL];nn=[t for t in cw if not isv(t) and not wv(t)]+[t for t in cw if isv(t) or wv(t)]
    return [h32(k) for k in dict.fromkeys(ks+nn)]
WHA=S_("home location work destination origin")
_acomp=lambda a,b:a!=b and bool(a) and bool(b) and a in WHA and b in WHA and 'location' in (a,b) and 'origin' not in (a,b)
FW=dict(ent=0.806,own=0.335,att=1.224,attc=0.706,att_bad=-1.0,pred=-0.14,pmis=-1.231,rec=-0.943,sup=-2.464,slot=0.482,subj=0.748,null=0.947,tau=1.0,ca=0.3,cb=0.8,co=0.098,kb=0.049,k_lp=0.888,k_ent=0.04,k_att=0.311,k_slot=0.336,k_pri=-1.306,k_cor=0.825,k_mg=0.005,k_nv=1.062,k_wild=-0.074,k_sup=-0.412,cu=0.341,pri=1.153,own2=0.022,csup=-2.086,wild=-0.662,cor=1.024,ci=0.334,vlen=0.102,ment=0.632,nco=-0.257,psyn=0.543,xtyp=-0.27,tonly=-0.837)
GEN=S_("get go commute travel come")
MEANV=S_("take catch grab ride drive use get hop board book")
MODEV=S_("walk cycle bike carpool hitchhike drive jog run fly sail skate scoot commute paddle row swim cab")
COMV=S_("text call phone email message write facetime skype whatsapp dm post mail ring video visit")
CMOD=S_("dark light bright pale deep matte navy sky forest hot baby lime olive mint sage royal electric neon pastel dusty burnt rose jet")
_LIKE=re.compile(r"(?i)\b(?:into|loves?|loved|loving|likes?|liked|enjoys?|enjoyed|adores?|obsessed with|mad about|crazy about|fan of|excited about|swears by|hooked on|got into|gotten into|getting into|passionate about)\s+(?:the\s+|a\s+|an\s+|his\s+|her\s+|their\s+|my\s+)?([^,.;!?]{2,48})")
_AGE=re.compile(r"(?i)\b(?:is|'s|are|was|turned|turns|turning|being|aged|just)\s+(\d{1,3}|"+'|'.join(NUMW)+r")(?:\s+(?:years?|yrs?|months?|weeks?)(?:\s+old)?)?\b(?!\s*(?:am|pm|minutes|min|hours|dollars|quid|bucks|pounds|euros|%|miles|km))")
_ALG=re.compile(r"(?i)\ballerg(?:ic|y|ies)\s+(?:is\s+)?to\s+([^,.;!?]{2,40})")
def _lm(x:str)->str:
    x=x.lower();x=x[:-2] if x.endswith("'s") else x;v=lemma(x);v=v if v!=x else next((u for u in _vf(x)[1:] if u in VB or 1 in wpos(u) and 0 not in wpos(x)),x);return _sg(v)
def cwords(lw)->set:return {_lm(x) for x in lw if re.fullmatch(r"[a-z][a-z'-]+",x) and len(x)>2 and x not in AUX|DET|PREPW|ADV|PRONS|FUN_Q|TGL and x not in U._STOPW.split()}
def _det(T,lw,c):
    n=len(T);R=[];V=lambda a,b,w='what':_val(c,T,a,b,w)
    for k in range(n):
        t=lw[k];p=lw[k-1] if k else ''
        if t in ('a','an') and p in ('is',"'s",'was','as','be','become','becoming','being','are',"'re",'am',"'m"):
            e=k+1
            while e<n and e<k+5 and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','who','which','that','but','so'} and not (e>k+1 and lw[e].endswith('ing') and wv(lw[e])):e+=1
            h=lw[e-1] if e>k+1 else '';hc=wcls(h,1) if h else frozenset()
            not any(T[q][0][:1].isupper() for q in range(k+1,e)) and ('occ' in hc or 'per' in hc and h not in ('person','guy','girl','woman','man','kid','child','baby','friend','fan','one','lady','bloke','dude')) and R.append(('what','job',k,e,1.0 if 'occ' in hc else 0.6));{'dog','cat'}&hc and h not in ('dog','cat','puppy','kitten') and R.append(('what','breed',k+1,e,1.0))
        if p in ('is',"'s",'was','are',"'re",'am',"'m") and t not in DET|PRONS|ADV|{'not','a','an'} and re.fullmatch(r"[a-z][\w-]*",t):
            e=k
            while e<n and e<k+3 and re.fullmatch(r"[a-z][\w-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|ADV|{'and','or','but','so'}:e+=1
            e>k and any({'comm','pay','veh'}&wcls(lw[q],1) for q in range(k,e)) and R.append(('how','',k,e,0.7))
        if lemma(t) in ('put','stuff','keep','store','save','stash','tuck','hide','drop','throw','pour','file','shove','cram','pack','load','move') and k+1<n:
            q=next((x for x in range(k+1,min(n,k+6)) if lw[x] in ('into','in','inside','onto')),None)
            if q is not None and q+1<n:
                e=q+1+(lw[q+1] in DET|{'my','our','his','her','their'})
                while e<n and e<q+6 and re.fullmatch(r"[a-z][\w'-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','but','because','so','which'} and not _istime(T,e):e+=1
                e>q+1 and R.append(('how','',q,e,0.5))
        if (lemma(t) in ('have','own','adopt','get','buy','keep','bring','pick','choose','order','rescue') and k+1<n and lw[k+1] in ('a','an','this','some') or lemma(t) in ('go','opt','settle','land','decide') and k+2<n and lw[k+1] in ('with','for','on') and lw[k+2] in ('a','an','the','this','some')) and k+1<n:
            k2=k+(lw[k+1] in ('with','for','on'));e=k2+2
            while e<n and e<k+6 and re.fullmatch(r"[a-z][\w'-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','but','because','so','who','which','that','named','called'} and not _istime(T,e):e+=1
            e>k2+2 and R.append(('what','kind',k2+1,e,0.5))
        if lemma(t) in MEANV and k+1<n:
            e=k+1+(lw[k+1] in DET|{'my','our','his','her','their'})
            while e<n and e<k+6 and re.fullmatch(r"[a-z0-9][\w'-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','to','because','so'}:e+=1
            e>k+1 and (ismeans(lw[e-1]) or 'veh' in wcls(lw[e-1])) and R.append(('how','',k,e,0.8))
        if _lm(t) in COMV and (t.endswith(('ing','s','ed')) or p in PRONS|{'i','we','just','usually','always','still','to','will','can','mostly','only'}):
            e=k+1
            while e<n and e<k+4 and re.fullmatch(r"[a-z][\w'-]*",lw[e]) and lw[e] not in PREPW|AUX|DET|{'and','or','but','because','so','all','every','each'} and not _istime(T,e):e+=1
            R.append(('how','',k,e,0.6))
        if (lemma(t) in MODEV or t.endswith('ing') and t[:-3] in MODEV|{'carpool','hitchhik','cycl','bik','driv','jogg','runn','skat','scoot'}) and (t.endswith(('ing','s','ed')) or p in PRONS|{'i','we','just','usually','always','to','will','can'}):R.append(('how','',k,k+1,0.6))
        if t in ('over','through','via','on','in','using','into') and k+1<n:
            e=k+1+(lw[k+1] in DET|{'my','our','his','her','their'})
            while e<n and e<k+6 and re.fullmatch(r"[a-z0-9][\w'-]*",lw[e],re.I) and lw[e] not in PREPW|AUX|DET|{'and','or','to','because','so','i','we','he','she'}:e+=1
            h=lw[e-1] if e>k+1 else '';hc=wcls(h,1) if h else frozenset()
            p in ('is',"'s",'was','are',"'re",'am',"'m",'be') and t not in DET and e>k and {'comm','pay','veh'}&wcls(lw[e-1],1) and R.append(('how','',k,e,0.6))
            h=h if len(h)>2 and h.isalpha() else '';hc=hc if h else frozenset()
            h and ({'comm','pay','veh'}&hc or t in ('via','through','using') and not {'loc'}&hc or T[k+1][0][:1].isupper() and t=='on') and R.append(('how','',k,e,0.7));h and t in ('in','on','with','into') and ('tool' in hc or 'sto' in hc and t=='into') and not {'comm','pay','veh'}&hc and R.append(('how','',k,e,0.5))
        if (iscol(t) or t in ('cream','plum','mustard','teal','ivory','olive','coral','peach','rust','charcoal','khaki')) and not (k+1<n and lw[k+1] in ('pepper','wine','cheese','juice','sauce','card','sox')):
            b=k-(p in CMOD);R.append(('what','color',b,k+1,0.9 if iscol(t) else 0.6))
        if T[k][0][:1].isupper() and k>0 and t not in TW and not _istime(T,k) and (p in DET|{'my','our','his','her','their','is',"'s",'was','for','on','with','got','bought','have','has','drive','drives','use','uses'} or re.fullmatch(r"(?:19|20)\d\d",p)) and not (k>0 and lw[k-1] in ('i',)) and 'loc' not in wcls(t,1) and t not in PRONS|{'i'}:
            e=k
            while e<n and (T[e][0][:1].isupper() or T[e][0][:1].isdigit()) and lw[e] not in AUX|{'i'}:e+=1
            b=k-1 if re.fullmatch(r"(?:19|20)\d\d",p) else k;R.append(('what','model',b,e,0.4))
    for R_,w,a,pr in ((_LIKE,'what','likes',0.8),(_ALG,'what','allergy',1.0),(_AGE,'what','age',0.9)):
        for m in R_.finditer(c):
            b=next((i for i in range(n) if T[i][1]>=m.start(1)),None);e=next((i for i in range(n) if T[i][2]>=m.end(1)),n-1)+1
            b is not None and R.append((w,a,b,min(e,n),pr))
    return R
_DUR=re.compile(r"(?:for\s+)?(?:about\s+|around\s+|like\s+|nearly\s+|almost\s+|over\s+)?(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|a few|a couple of|several|\d+|a whole)\s+(?:minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|nights?|decades?)(?:\s+(?:long|straight|now|or so))?")
def spans(text:str,known=(),names=None,last=''):
    F=extract(text,last,known,names)[0];CL=[];out=[];lp='';lo='';Q=list(clauses(text));tcor=bool(_COR2.search(text));t0=last
    while Q:
        c=Q.pop(0);sub=isinstance(c,tuple);c,pc=(c if sub else (c,None));A=analyze(c,(t0 if tcor and t0 not in ("","me") else lo if lo else last) if re.match(r"(?i)(?:it|that|this)\b",c) else last,known,names,lp)
        if not A:continue
        T=A['T'];lw=[t[0].lower() for t in T];ent=A['ent'] if A['ent'] not in ('','?') else (pc[3] if sub else last);cw=cwords(lw)|(pc[1] if sub else set());es={ent}|{_resolve(_np(T,j+1,len(T))[0],known) for j,t in enumerate(lw) if t in ('my','our')}|{names[t] for t in lw if t in names}|(pc[2] if sub else set())
        CL.append((A['c'].lower(),cw,es,ent,A['pred']))
        [out.append(dict(v=_val(A['c'],T,x,y,w),w=w,a='',ent=ent,pred=lemma(vb) if vb else A['pred'],subj=False,cw=cw,ents=es,pri=0.5)) for (w,a,x,y),vb in zip(A['sp'],A['sv'])]
        [out.append(dict(v=_val(A['c'],T,x,y,w),w=w,a=a,ent=ent,pred=A['pred'],subj=False,cw=cw,ents=es,pri=pr)) for w,a,x,y,pr in _det(T,lw,A['c'])]
        Q[:0]=[(z,CL[-1]) for (w,a,x,y) in A['sp'] if w=='why' and not sub and y-x>3 for z in clauses(re.sub(r"(?i)^\W*(?:because|cause|'cause|coz|cuz|since|so that|so|in order to|to)\s+",'',A['c'][T[x][1]:T[y-1][2]]))]
        ob=_obj(T,lw,A['k'],A['sp'][0][2] if A['sp'] else len(T),known,'') if A['pred'] not in ('be','') and not re.match(r"(?i)(?:which|who|that|whom)\b",c) else '';lo=ob or lo
        sub or (last:=ent or last);ent not in ('me','') and isper(ent.split()[-1]) and (lp:=ent)
    out=[o for o in out if not (o['w']=='when' and _DUR.fullmatch(o['v'].lower().strip()))];ucw=set().union(*[x[1] for x in CL]) if CL else set()
    if CL and _COR2.search(text):
        for m in re.finditer(r"(?i)(?:^\W*\*\s*|\b(?:it's|it is|it was|its|it'll be|that's|make it|make that|actually)\s+(?:actually\s+|really\s+|now\s+)?)([^,.;!?]{1,60})",_EMO.sub(' ',text)):
            v=re.split(r"\s+(?:he|she|they|i|we|you|not|now|instead|which|because|bc|cuz|since|so|and then|until|till|while|before|after|when|though|but)\b",re.sub(r"(?i)^(?:going to be|gonna be|gonna|going to|actually|really|now)\s+",'',m[1].strip()),maxsplit=1)[0].strip(' *');T_=toks(v)[1];vt=_vtype([t[0].lower() for t in T_],T_) if T_ else None;vt=None if vt and vt[1] in ('age','name') else vt
            v and not re.match(r"(?i)(?:because|bc|cuz|cause|since|to|so|going|gonna|been|a bit|fine|ok|okay|all|just|not|no)\b",v) and out.append(dict(v=v,w=vt[0] if vt else '*',a=vt[1] if vt else '',ent=CL[0][3],pred='',subj=False,cw=CL[0][1],ents=CL[0][2],pri=0.8))
        lc=re.sub(r"\s+",' ',CL[-1][0]).strip(' ,.');lc and len(lc)<=72 and len(lc.split())>=3 and out.append(dict(v=re.sub(r"(?i)^(?:i|we|he|she|they)\s+(?:told me|said|say|says|found out|heard)(?:\s+\w+)?\s*,?\s*",'',next((x for x in clauses(text) if x.lower()==CL[-1][0]),CL[-1][0])),w='*',a='',ent=CL[-1][3],pred=CL[-1][4],subj=False,cw=CL[-1][1],ents=CL[-1][2],pri=0.3))
    for e,vb,w,a,v,cf,_ in F:
        cl=next((x for x in CL if v.lower() in x[0]),None);out.append(dict(v=v,w=w,a=a,ent=last if e=='?' else e,pred=lemma(vb) if vb else '',subj=cf>=0.75,cw=cl[1] if cl else ucw,ents=(cl[2] if cl else set())|{last if e=='?' else e},pri=1.0))
    [o.__setitem__('ucw',ucw) for o in out];return [o for o in out if o['v'] and not (o['w']=='when' and _DUR.fullmatch(o['v'].lower().strip()))]
def _hd(x:str)->str:return _sg(x.split()[-1]) if x else ''
@functools.lru_cache(maxsize=65536)
def _wsn(w:str)->frozenset:
    return frozenset(i for v in dict.fromkeys([w]+list(_vf(w))) for i in _syn(v,0)[:3])
@functools.lru_cache(maxsize=65536)
def _whn(w:str)->frozenset:
    c=_cog();return frozenset(j for i in _wsn(w) for j in c[0].anc(int(i),(1,),2)) if c else frozenset()
KINW=S_("mom dad mother father sister brother son daughter wife husband aunt uncle cousin niece nephew grandma grandpa grandmother grandfather parent parents sibling siblings partner boyfriend girlfriend fiance fiancee friend roommate boss coworker colleague neighbor neighbour stepdad stepmom stepson stepdaughter twin kid kids child children baby")
_kin=lambda t:_sg(t) in KINW or t in KIN or t in KIN.values()
def _heq(a:str,b:str)->bool:return a==b or KIN.get(a,a)==KIN.get(b,b) or bool(a) and bool(b) and a not in ('me','thing','one') and b not in ('me','thing','one') and (bool(_wsn(a)&_wsn(b)) or not _kin(a) and not _kin(b) and (bool(_wsn(a)&_whn(b)) or bool(_wsn(b)&_whn(a)))) or len(a)>=5 and len(b)>=5 and abs(len(a)-len(b))<=1 and (sum(x!=y for x,y in zip(a,b))<=1 if len(a)==len(b) else any(a[:i]+a[i+1:]==b for i in range(len(a))) or any(b[:i]+b[i+1:]==a for i in range(len(b))))
FK=('pmis','ent','own','att','attc','att_bad','pred','rec','sup','slot','subj','co','cu','pri','own2','csup','wild','cor','ci','vlen','ment','nco','psyn','xtyp','tonly')
@functools.lru_cache(maxsize=65536)
def _vsn(v:str)->frozenset:
    return frozenset(i for x in dict.fromkeys([v]+list(_vf(v))) for i in _syn(x,1)) if v and v not in ('be','have','do','get','go','make','take','say') else frozenset()
def cands(st:Store,qa:dict,known=(),names=None,slot:str=''):
    if not qa or not qa['ents'] or not len(st.u):return []
    stt,K=st.uidx();qk=np.array([h32(x) for x in qa['ents']]+[h32(x) for x in qa.get('cw',())],np.uint32);m=np.isin(K,qk).any(1)
    if not m.any():return []
    _cont=lambda tx:bool(_COR2.search(tx)) or re.match(r"(?i)\W*(?:(?:and|but|so|also|oh|ok|well|yeah|yea|tbh|lol|update|anyway)\W+)*(?:he|she|they|their|his|her|its|it|them)\b",tx) is not None
    ix=np.flatnonzero(m);ix=np.unique(np.concatenate([ix,[i+1 for i in ix if i+1<len(stt) and not m[i+1] and _cont(st.utext(int(i+1)))]]).astype(np.int64));ag=st.age(st.u[stt[ix],1]).astype(np.float64);order=ix[np.argsort(ag,kind='stable')];C=[];inv={v:k for k,v in st.ents().items()}
    for r,i in enumerate(order):
        tx=st.utext(int(i));cr=bool(_COR2.search(tx));ci=_cont(tx) and not m[i] and i>0 and bool(m[i-1])
        pv=inv.get(int(K[i-1][0]),'') if cr and i>0 else '';pv=pv if pv!='me' or inv.get(int(K[i][0]),'')=='me' else ''
        for sp in spans(tx,known,names,pv or inv.get(int(K[i][0]),'')):sp['ui']=int(i);sp['cor']=cr;sp['ci']=ci and sp['ent'] in ('','?',pv or inv.get(int(K[i][0]),''));C.append((r,sp))
    qa_=qa['attr'];qp=pcls(qa['pred']) if qa['pred'] else '';pe=qa['ents'][0];qh={_hd(x) for x in qa['ents']};seen={};S=[];Q=set(qa.get('cw',()))-{'thing','stuff'};rc=min([r for r,sp in C if sp.get('cor') and sp['w'] in (qa['w'],'*')]+[10**6]);EW=set(' '.join(qa['ents']).split())
    for r,sp in C:
        g=(sp['ent'],sp['w'],sp['a']);sup=g in seen and seen[g]<r;seen.setdefault(g,r)
        wild=sp['w']=='*';sp=({**sp,'w':qa['w'],'a':qa_} if wild else sp)
        sp=({**sp,'a':'favorite'} if sp['w']=='what' and sp['ent'].startswith('favorite') and sp['a'] in ('','kind') else sp)
        ek=pe=='me' and sp['ent'] not in ('','me','?') and isper(_hd(sp['ent'])) and not sp.get('ci') or sp['ent'] not in ('','me','?') and pe not in ('me',) and not any(_heq(_hd(sp['ent']),h) for h in qh) and not EW&set(sp['ent'].split()) and (isper(_hd(sp['ent'])) or _hd(sp['ent']) in {_hd(k) for k in known}) and (isper(_hd(pe)) or pe in known)
        nco={x for x in Q&sp.get('cw',set())-EW if not isv(x) and not wv(x)};hm=any(_heq(_hd(sp['ent']),h) for h in qh);o2=bool(set(qa['ents'])&sp.get('ents',set()) or qh&{_hd(x) for x in sp.get('ents',()) if x});mt=bool((qh|EW)-{'me','thing','stuff','one','place','time','day'}&sp.get('cw',set()))
        xt=qa['w']=='how' and sp['w']=='where' and (pcls(qa['pred'])=='keep' or qa['pred'] in COOKV|{'store','keep','save'});sp=({**sp,'w':'how','a':''} if xt else sp)
        to=bool(qa_) and qa_ not in ('','kind','likes') and sp['a']==qa_ and sp['ent'] in ('','?','it','me') and sp['w']==qa['w']
        if ek or sp['w']!=qa['w'] or qa_ and sp['a'] and sp['a']!=qa_ and not (qa_ in KA and sp['a'] in KA) and not _acomp(qa_,sp['a']) or not (to or sp.get('ci') or sp['ent']==pe or hm or mt or o2 or nco or EW&set(' '.join(x for x in sp.get('ents',()) if x and x!='me').split())-{'thing','stuff','one'}):continue
        pm=sp['ent']=='me' and pe=='me' and bool(qp) and qp not in ('be','have','do') and bool(sp['pred']) and pcls(sp['pred']) not in (qp,'be','have') and not (qa['pred'] in GEN and qa['w']=='how' and (sp['pred'] in MOTION|MODE))
        f=dict(pmis=float(pm),ent=float(sp['ent']==pe),own=float(sp['ent'] in qa['ents'] or hm),att=float(bool(qa_) and (sp['a']==qa_ or qa_ in KA and sp['a'] in KA)),attc=float(bool(qa_) and _acomp(qa_,sp['a'])),att_bad=float(bool(qa_) and bool(sp['a']) and sp['a']!=qa_ and not (qa_ in KA and sp['a'] in KA) and not _acomp(qa_,sp['a'])),pred=float(bool(qp) and pcls(sp['pred'])==qp),rec=float(r),sup=float(sup),slot=float(bool(slot) and (sp['v'].lower() in slot.lower() or slot.lower() in sp['v'].lower())),subj=float(sp['subj']),co=float(min(3,len(Q&sp.get('cw',set())))),cu=float(min(3,len(Q&sp.get('ucw',set())))),pri=float(sp.get('pri',0.5)),own2=float(o2),csup=float(r>rc),wild=float(wild),cor=float(bool(sp.get('cor'))),ci=float(bool(sp.get('ci'))),vlen=min(12,len(sp['v'].split()))/6.0,ment=float(mt),nco=float(min(2,len(nco))),tonly=float(to and not (sp['ent']==pe or hm or mt or o2)),xtyp=float(xt),psyn=float(bool(qa['pred']) and bool(sp['pred']) and qa['pred']!=sp['pred'] and bool(_vsn(qa['pred'])&_vsn(sp['pred']))))
        sp['f']=f;S.append((f,sp))
    return S
def soft(st:Store,qa:dict,known=(),names=None,slot:str='',fw=None):
    fw={**FW,**(fw or {})};S=cands(st,qa,known,names,slot)
    if not qa or not qa['ents'] or not len(st.u):return []
    if not S:return [(1.0,None)]
    z=np.array([sum(fw[k]*f[k] for k in FK) for f,_ in S]+[fw['null']])/fw['tau'];p=np.exp(z-z.max());p/=p.sum();agg={'':(float(p[-1]),None)};p=p[:-1]
    for (f,sp),pp in zip(S,p):k=sp['v'].lower();agg[k]=(agg[k][0]+pp,agg[k][1]) if k in agg else (pp,sp)
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
        s.last=(q,recall(s.st,q,'both'));r=s.last[1];return {'a':r['val']} if r.get('val') and r['p']>=s.gate else None
class SoftLUT:
    def __init__(s,st:Store,sl:SlotLUT):s.st,s.sl=st,sl
    @property
    def _raw(s):return len(s.st.u)>0
    def lookup_soft(s,q,k=1,return_diag=True):
        r=s.sl.last[1] if s.sl.last and s.sl.last[0]==q else recall(s.st,q,'soft');S=r['soft']
        return (S[0][1]['v'],None,r['p'],S[1][0]*r['p']/max(S[0][0],1e-9) if len(S)>1 else None) if S and r['line'] and not r['slot'] else (None,None,None,None)
def bus(st:Store,gate:float=None):
    from amni.serve.memory_bus import MemoryBus,_CONF_DIRECT
    g=_CONF_DIRECT if gate is None else gate;sl=SlotLUT(st,g);return MemoryBus(adam=None,answer_lut=sl,sem_lut=SoftLUT(st,sl),kb=None,learning_atlas=None,ledger_path=os.devnull)
def _slotval(st:Store,e,v,w,a)->str:
    if not len(st.s):return ''
    i=_li(st,(_ea(st)==h32(e))&(st.s[:,3]==WI[w]*17+AI.get(a,0))&((st.s[:,7]==(h32(lemma(v))&0xFFFF)) if v else True));return st.text(st.s[i])[2] if i is not None else ''
def _fit(e:str,v:str,val:str)->str:
    k=80-len(e.encode()[:28])-len(v.encode()[:12])-2;b=val.encode()
    return val if len(b)<=k else b[:k].decode('utf-8','ignore').rsplit(' ',1)[0].rstrip(' ,;')
def _learn(B,st:Store,e,v,w,a,val,cf)->bool:
    val=_fit(e,v,val);pv=_slotval(st,e,v,w,a);return bool(B.record_learning('slot::'+'\x1f'.join((e,v,w,a,str(round(cf,4)))),val,kind='user_fact',provenance='user:toroid',exactness='exact',supersedes=pv if pv and pv!=val else None)['stored'])
def respond(st:Store,q:str,gate:float=None,hint:float=0.5,unsure:float=0.25,abstain:bool=False,unk:bool=False)->dict:
    gate=float(os.environ.get('AMNI_UMEM_TOROID_GATE','0.8')) if gate is None else gate;B=bus(st,gate);v,home,c=B.recall(q,gate=gate);sl=B.answer_lut;r=sl.last[1] if sl.last and sl.last[0]==q else recall(st,q,'both');qa=r['qa']
    if v and qa and qa['ents']:
        if home=='tier0_atex_override' and r['slot']:d=r['slot']
        else:sp=r['soft'][0][1];d=dict(ent=qa['ents'][0],w=qa['w'],a=qa['attr'] or sp['a'],v=v,verb=sp['pred'] if qa['ents'][0]=='me' else '')
        return dict(kind='direct',text=say(d),v=d['v'],conf=float(c),home=home,line=r['line'],r=r)
    if r['line'] and r['p']>=hint:return dict(kind='hint',text='',conf=r['p'],home='hint',line=r['line'],r=r)
    if abstain and qa and qa['ents'] and not r['slot'] and r.get('pnull',0)>=(gate or 0.9) and (qa['ents'][0] in st.ents() or qa['ents'][0]=='me'):return dict(kind='direct',text="You haven't told me that yet.",conf=r['pnull'],home='abstain',line='',r=r)
    S=r['soft']
    if unk and qa and qa['ents'] and not r['slot'] and (qa['ents'][0] in st.ents() or qa['ents'][0]=='me'):return dict(kind='unknown',text='',conf=r.get('pnull',1.0),home='unknown',line=line(qa['ents'][0],qa['w'],qa['attr'],'unknown'),r=r)
    return dict(kind='unsure',text='',conf=r['p'],home='unsure',line=line(qa['ents'][0],qa['w'],qa['attr'] or S[0][1]['a'],S[0][1]['v'],'',S[1][1]['v']),r=r) if qa and qa['ents'] and len(S)>1 and S[0][0]>=unsure else dict(kind='none',text='',conf=r['p'],home='none',line='',r=r)
_TAGQ=re.compile(r"(?i)^\W*(?:you know|remember|did i (?:tell|mention to) you|guess what)\W+(?:how\s+|that\s+|about\s+)?(?:my|our|i|i'm|we|he|she|they|the)\b|[^?]{20,}[,.;!]\s*(?:\w+\W*){1,3}\?\s*$")
def observe(st:Store,msg:str,**kw):
    m=str(msg).strip();f=forget_st(st,m)
    return ('forget',f) if f is not None else ('ask',respond(st,m,**kw)) if U._isq(m) and not (_TAGQ.search(m) and not re.match(r"(?i)^\W*(?:(?:so|ok|okay|and|wait|hey|um)\W+)*(?:what|when|where|why|how|who|which|do|does|did|is|are|can|could|would|will)\b",m)) else ('skip',None) if not m or U._REQ.search(m) else ('fact',ingest(st,m))
def remember(key:str,msg:str)->int:
    with U._lock:st=load(key);F=ingest(st,msg);save(key,st);return len(F)
_FGV=r"forget|delete|erase|remove|wipe|scrub|scrap|drop|clear|purge|bin|ditch|lose|get rid of|take out|strike|unlearn"
_FGK=r"remember|store|keep|save|hold onto|hang onto|hold on to|retain|track|log|record|have"
_FG2=re.compile(r"(?i)^\W*(?:(?:actually|ok(?:ay)?|so|oh|hmm+|anyway|also|and|but|please|pls|wait|on (?:second thought|reflection)(?:[^,]{0,40})?)\W+)*(?:(?:can|could|would|will) you\s+(?:please\s+)?(?:(?P<n1>not|stop)\s+)?(?:(?:just|also)\s+)?|please\s+|let'?s\s+(?P<n2>not)\s+|(?P<n3>do not|don't|dont|no need to|stop|no longer|never)\s+|i'?d rather you (?P<n4>not|didn't)\s+|)(?:(?:just|also|really)\s+)?(?P<v>"+_FGV+"|"+_FGK+r")(?:ing)?\s+(?P<x>.+)$")
_FGM=re.compile(r"(?i)\b(?:thing|stuff|bit|details?|info(?:rmation)?|part|about|reason|date|time|memory|memories|records?|stored|saved|colou?r|name|address|job)\b")
def _fg2(m:str):
    G=[g for sg in re.split(r"[.;!?]+|,\s*(?:so\s+|but\s+|and\s+)?",' '.join(t for t,_,_ in toks(_EMO.sub(' ',m))[1]) if m else '') for g in [_FG2.match(sg.strip())] if g and bool(g['n1'] or g['n2'] or g['n3'] or g['n4'])==(re.fullmatch(_FGK,g['v'].lower()) is not None) and not (g['v'].lower()=='have' and not (g['n1'] or g['n2'] or g['n3'] or g['n4']))]
    return _fgx(G[0]['x'],m) if G else None
def _fgx(x:str,m:str=''):
    x=re.split(r"\s*[.,;!?]\s*",x.strip())[0];x=re.sub(r"(?i)\s+(?:from|out of|in|off|on)\s+(?:your\s+|the\s+|my\s+)?(?:memory|memories|records?|notes|file|files|system|head|list)\b.*$|\s+(?:anymore|any more|ok|okay|though|please|for me|stored|saved|anywhere|about me)$",'',x).strip()
    if not x or re.match(r"(?i)to\s+\w",x) or not (re.match(r"(?i)(?:how|why|where|when|what|who|which)\b",x) or _FGM.search(x) or re.search(r"(?i)\b(?:memory|stored|saved|on file)\b",m)):return None
    x=re.sub(r"(?i)^(?:(?:that|the|this|those|these|all|any)\s+)?(?:bit|part|thing|stuff|info(?:rmation)?|details?)\s+(?:about|of|on|regarding|with)\s+|^what i (?:told you|said|mentioned)(?: about)?\s+|^(?:everything|anything) (?:about|on)\s+",'',x).strip()
    x=re.sub(r"(?i)\s+(?:thing|stuff|details?|info|bit|part)(?=\s+about\b|$)",'',x).strip();x=re.sub(r"(?i)^(?:the|that|this)\s+(\w+)\s+about\s+(him|her|them|me)$",r"\2",x)
    fr=next((FRW[w] for w in re.findall(r"[a-z]+",x.lower())[-1:] if w in FRW),None)
    return (x+'?',False) if re.match(r"(?i)(?:how|why|where|when|what|who|which)\b",x) else ({'why':'Why','when':'When','how':'How','where':'Where'}[fr]+' '+re.sub(r"\s+\w+$",'',x)+'?',False) if fr and fr!='what' and len(x.split())>1 else (f"What about {x}?",False)
def forget_st(st:Store,msg:str):
    f=U._fg(str(msg).strip());f=((_fgx(re.sub(r"(?i)^what about\s+",'',f[0]).rstrip('?')) or (f[0],False))[0],False) if f and not f[1] else f or _fg2(str(msg).strip())
    if not f:return None
    if f[1]:n=len(st.s)+len(st.uidx()[0]);st.s,st.u,st.head,st._x,st.lp,st.lo=Store().s,Store().u,0,None,'','';return ('all',n)
    kn=st.ents();qa=ask(f[0],_last(st),list(kn),st.names,st.lp)
    if not qa or not qa['ents']:return (None,0)
    qa['ents']=[e if e in kn or e=='me' else next((k for k in kn if k!='me' and (set(e.split())-{'thing','stuff'})&set(k.split())),e) for e in qa['ents']]
    wa=None if re.match(r"(?i)\s*what about\b",f[0]) and qa['w']=='what' and not qa['attr'] else qa['w'];e=h32(qa['ents'][0]);es=_ea(st);wm=(st.s[:,3]//17==WI[wa]) if wa else np.ones(len(st.s),bool)
    qp=h32(qa['pred'])&0xFFFF if qa['pred'] and qa['pred'] not in ('be','have','do') else None;qc=h32(pcls(qa['pred']))&0xFFFF if qp is not None else None
    pm=((st.s[:,7]==qp) if (st.s[:,7]==qp).any() else (st.s[:,2]==qc)) if qp is not None and qa['ents'][0]=='me' else None
    dm=(es==e)&wm&(pm if pm is not None else True)|(wm&pm if pm is not None else False)
    for q in np.flatnonzero(dm):dm|=(st.s[:,4]==st.s[q,4])&(st.s[:,3]==st.s[q,3])&np.array([st.text(r)[2]==st.text(st.s[q])[2] for r in st.s],bool)
    stt,K=st.uidx();um=np.flatnonzero((K==e).any(1))
    um=[i for i in um if wa is None or any(sp['w']==wa for sp in spans(st.utext(int(i)),list(kn),st.names))];qc2=set(qa.get('cw',()))-{qa['pred'],'thing','stuff','forget','delete'}
    um=[i for i in um if cwords([t[0].lower() for t in toks(st.utext(int(i)))[1]])&qc2] or um if qa['ents'][0]=='me' and qc2 else um
    n=int(np.sum(dm))+len(um);st.s=st.s[~dm];st.udrop(um);st.lo='' if st.lo==qa['ents'][0] else st.lo;return (qa['ents'][0],n)
def forget(key:str,msg:str):
    with U._lock:
        f=U._fg(str(msg).strip())
        if f and f[1]:return ('all',int(wipe(key)))
        st=load(key);r=forget_st(st,msg);r is not None and r[1] and save(key,st,rotate=True);return r
CK=('lp','ent','att','slot','pri','cor','mg','nv','wild','sup')
def cfeat(S,top,slot=None)->dict:
    if top is None:c=min(max(float(slot['conf']),1e-4),1-1e-4);return dict(lp=float(np.log(c/(1-c))),ent=1.0,att=1.0,slot=1.0,pri=1.0,cor=0.0,mg=c,nv=0.0,wild=0.0,sup=0.0)
    f=top[1].get('f',{});p=min(max(float(top[0]),1e-4),1-1e-4);p2=max([x[0] for x in S if x is not top] or [0.0])
    return dict(lp=float(np.clip(np.log(p/(1-p)),-9,9)),ent=f.get('ent',0.0),att=f.get('att',0.0),slot=f.get('slot',0.0),pri=f.get('pri',0.0),cor=f.get('cor',0.0),mg=float(top[0]-p2),nv=float(np.log(max(1,len(S)))),wild=f.get('wild',0.0),sup=f.get('sup',0.0)+f.get('csup',0.0))
def _cal(x:dict,fw:dict)->float:return float(1/(1+np.exp(-(fw['kb']+sum(fw['k_'+k]*x[k] for k in CK)))))
def recall(st:Store,q:str,tier:str='both',unk:bool=False,theta:float=0.5,fw=None)->dict:
    kn=st.ents();qa=ask(q,_last(st),list(kn),st.names,st.lp);out=dict(qa=qa,slot=None,soft=[],line='',p=0.0)
    if not qa or not qa['ents']:return out
    L=lookup(st,qa) if tier in ('slot','both') else [];out['slot']=_row(st,L[0][1]) if L and L[0][0]>=2.0 else None;sv=out['slot']['v'] if out['slot'] else ''
    S=soft(st,qa,list(kn),st.names,sv if tier=='both' else '',fw) if tier in ('soft','both') else []
    out['pnull']=max([x[0] for x in S if x[1] is None] or [1.0]);S,top=_top(S);out['soft']=S;out['p']=top[0] if top else (out['slot']['conf'] if out['slot'] else 0.0);ent=qa['ents'][0];pa=qa['attr']
    ag=sum(pp for pp,sp in S if sv and (sv.lower() in sp['v'].lower() or sp['v'].lower() in sv.lower()));out['p']=(ag if S else out['slot']['conf']) if tier=='both' and out['slot'] else out['p']
    out['cx']=cfeat(S,top) if top else cfeat(S,None,out['slot']) if out['slot'] else None
    if tier=='both' and top:out['slot']=None;out['p']=top[0]
    if tier=='slot' or tier=='both' and out['slot']:
        r=out['slot'];out['line']=line(r['ent'],r['w'],r['a'],r['v'],r['verb']) if r else '';u=np.flatnonzero(st.u[st.uidx()[0],1]==int(st.s[L[0][1],4])) if r else [];out['src']=st.utext(int(u[-1])) if len(u) else ''
    elif top:
        sp=top[1];out['src']=st.utext(sp['ui']);out['line']=line(ent,qa['w'],pa or sp['a'],sp['v'],sp['pred'] if ent=='me' and qa['w']!='what' else '',S[1][1]['v'] if top[0]<theta and len(S)>1 else '')
    fw={**FW,**(fw or {})};out['p']=_cal(out['cx'],fw) if out['line'] and tier!='slot' and out.get('cx') else out['p'];out['val']=(out['slot']['v'] if out['slot'] else top[1]['v'] if top else '') if out['line'] and ' (unsure).' not in out['line'] else ''
    if not out['line'] and unk and (ent in kn or ent=='me'):out['line']=line(ent,qa['w'],pa,'unknown')
    return out
def turn(message:str,history=None,user=None,session=None,dry:bool=False)->dict:
    t0=time.perf_counter();out={'lines':[],'forgot':None,'facts':0,'p':0.0,'ms':0.0,'direct':'','kind':''}
    if not enabled():return out
    m=str(message).strip();E=os.environ.get;dr=E('AMNI_UMEM_TOROID_DIRECT','1')=='1'
    try:
        with U._lock:
            key=U.user_key(user);f=U._fg(m)
            if f and f[1] and not dry:out.update(forgot=('all',int(wipe(key))));out['ms']=(time.perf_counter()-t0)*1000;return out
            st=load(key);k,r=observe(st,m,hint=float(E('AMNI_UMEM_TOROID_HINT','0.5')),unsure=float(E('AMNI_UMEM_TOROID_UNSURE','0.25')),abstain=E('AMNI_UMEM_TOROID_ABSTAIN','0')=='1',unk=E('AMNI_UMEM_TOROID_UNK','0')=='1')
            k=='forget' and out.update(forgot=r);k=='fact' and out.update(facts=len(r))
            k=='ask' and out.update(kind=r['kind'],p=r['conf'],direct=r['text'] if dr and r['kind']=='direct' else '',lines=[r['line']] if r['line'] and not (dr and r['kind']=='direct') and r['kind']!='none' else [])
            not dry and (k=='fact' or k=='forget' and r[1]) and save(key,st,rotate=k=='forget')
    except (OSError,ValueError) as e:out['err']=type(e).__name__
    out['ms']=(time.perf_counter()-t0)*1000;return out

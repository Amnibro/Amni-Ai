import contextvars,hashlib,os,re,secrets,struct,threading,time,zlib
from collections import namedtuple
import numpy as np
from amni.compute.reffelt4 import decode_rgba4_to_fp16,pack_rgba4_page
WM=(1<<20)-1;SELFK=1;ISINK=3;NR=64;VER=1;PW=PH=64;MAGIC=b'AMUM';HDR=struct.Struct('<4sHHHHIII');REC=struct.Struct('<IBIdH8sBB')
BJOB,BQLOCK,BSOL,BFRO,BVOP,BART,BNEG,BDO,BWHE,BWHO,BWHEN,BMETI,BSKMU,BSKNO,BACO,BKINL,BHAV,BPRE,BENTK,BCALL,BSTUD,BMOST,BFAV=(1<<21,1<<22,1<<20,1<<19,1<<18,1<<17,1<<16,1<<15,1<<14,1<<13,1<<12,1<<11,1<<10,1<<9,1<<8,1<<7,1<<6,1<<5,1<<4,1<<3,1<<2,1<<1,1)
BKEEP=BDO|BWHE|BWHO|BWHEN|BMETI|BQLOCK
F=namedtuple('F','ent rel clk ts turn sid val name')
def bh(w:bytes)->int:
    h=0
    for c in w:h=(h*83492791+c+1)&0xFFFFFFFF
    return h
H_TO,H_IN,H_A,H_HOW,H_YEARS,H_CRAZY,H_ALL,H_FAN,H_GO,H_GOES,H_WENT,H_GOING,H_DEGREE,H_SCHOOL,H_AROUND,H_AT,H_FROM,H_MAJOR,H_NOT,H_LET,H_TELL,H_ASK=(bh(w.encode()) for w in 'to in a how years crazy all fan go goes went going degree school around at from major not let tell ask'.split())
CLS,RID,FUN,GX={},{},set(),{}
def _both(w:str)->list:return [bh(w.encode())&WM]+([bh((w[0].upper()+w[1:]).encode())&WM] if w[0].islower() else [])
def _put(w:str,bit:int,r:int=0):
    for i in _both(w):CLS[i]=CLS.get(i,0)|bit;RID[i]=r or RID.get(i,0)
[FUN.update(_both(w)) for w in "a an the and or of to in on for with as at by from into about over under is are was were be been being am do does did my your his her their you it its this that these those he she they them him we us actually oh well now really anyway okay ok yeah hey btw what why how when where who whom which can could would should will shall may might must just also some any there here so if not no yes please tell give than then fun fact guess though recently hi hello".split()]
[_put(w,b) for ws,b in (("again just still year years trade while last every most way really lately recently home grad alum baby days day out back today ever night anything more down easily around away other always than seriously mostly new big whole own same very all even quite pretty quite super totally already never often usually sometimes ago together hands believe context record case curious wrong earlier few couple bit lots long several much many these those now please though then also too actually here there so if while during with without per via next each another any some fact know people family thing things stuff pretty stay stays crazy we wed older comes falls badly because",1),("wait sorry oops hang scratch let update correction honestly heads hi hey so yes no thanks please hmm ah wow ooh right alright cool nice great good ok okay and but or um uh yep yeah nope gotcha noted awesome perfect sure exactly absolutely definitely maybe perhaps anyway meanwhile also plus then oh actually well fun fact just",2),("a an that old my really just still very seriously mostly new big whole own same more",4),("did do does can could have has had would should",8),("say said tell told mention mentioned remember remembered recall recalled remind know",16),("sister mom aunt grandma niece daughter wife girlfriend stepmom stepsister godmother goddaughter brother dad uncle grandpa nephew son husband boyfriend stepdad stepbrother godfather godson cousin coworker boss landlord neighbor roommate bandmate classmate teammate mentor tutor partner friend manager dog cat parrot hamster rabbit ferret turtle gecko horse goldfish hedgehog iguana cockatiel chinchilla tortoise rat snake budgie buddy pal pig dragon law guinea father mother",32),("january february march april may june july august september october november december",64),("de del la el da di van von st",128)) for w in ws.split()]
[_put(w,b,r) for d,b in (("color:10 colors:10 colour:10 food:11 meal:11 hobby:7 hobbies:7 job:5 jobs:5 car:9 cars:9 ride:9 allergy:8 allergies:8 birthday:15 born:15 month:15 age:14 name:6 school:13 college:13 university:13 degree:12 city:4 town:4 apartment:4 anything:11 everything:11 trade:5",256),("live:4 lives:4 lived:4 living:4 based:4 base:4 moved:4 move:4 moving:4 work:5 works:5 worked:5 working:5 allergic:8 react:8 reacts:8 reacted:8 major:12 majored:12 graduate:13 graduated:13 turned:14 turn:14 went:13 go:13 goes:13 going:13",512),("like:7 likes:7 liked:7 love:7 loves:7 loved:7 drive:9 drives:9 drove:9 driving:9 eat:11 eats:11 eating:11 order:11 orders:11 ordered:11 studied:12 study:12 studying:12 bought:9 buy:9 pick:7 picks:7 into:7",1024)) for w,r in (x.split(':') for x in d.split()) for r in [int(r)]]
[GX.setdefault(i,(int(b),int(r))) for w,b,r in [x.split(':') for x in "employed:512:5 lands:512:15 passionate:1024:7 alma:256:13 profession:256:5 professions:256:5 occupation:256:5 career:256:5 shade:256:10 hue:256:10 birth:256:15 located:512:4 reside:512:4 resides:512:4 enjoy:1024:7 enjoys:1024:7 enjoyed:1024:7 hate:1024:2 hates:1024:2 hated:1024:2 dislike:1024:2 dislikes:1024:2 detest:1024:2 detests:1024:2 loathe:1024:2 loathes:1024:2".split()]+[(c,2048,10) for c in "red orange yellow green blue purple pink brown black white gray grey silver gold golden beige teal navy maroon violet scarlet crimson indigo turquoise magenta lavender burgundy charcoal cerulean chartreuse cobalt emerald fuchsia ivory khaki lilac mauve ochre periwinkle sapphire sepia taupe terracotta ultramarine vermilion aquamarine azure bronze copper cyan mahogany chestnut gunmetal pewter platinum amethyst goldenrod seafoam aqua celadon puce umber russet cornflower heliotrope".split()] for i in _both(w)]
H_FAMILY,H_RELATIVES,H_FOLKS=(bh(w.encode()) for w in ('family','relatives','folks'))
WQS=[bh(w.encode()) for w in 'what where who when which how why whose do does did can could would will remind is are'.split()];WIM={bh(w.encode()) for w in 'give tell list write explain describe create make find identify classify provide generate summarize translate compare suggest recommend show define calculate build reply label choose name rewrite extract categorize imagine'.split()};WCM={bh(w.encode()) for w in 'actually correction scratch update updated wrong oops misspoke fix mixed instead now moved changed switched longer anymore traded sorry wait corrected new'.split()}
_LTC={}
def _pflag(p:str)->bool:
    j=(p[:-8] if p.endswith('_sid.npy') else p[:-5] if p.endswith('.gf17') else p)+'.json'
    if j not in _LTC:
        try:_LTC[j]='"latest": 1' in open(j,encoding='utf-8',errors='replace').read()
        except OSError:_LTC[j]=False
    return _LTC[j]
def _lt()->bool:
    v=os.environ.get('AMNI_UMEM_LT')
    if v is not None:return v!='0'
    from amni.compute.ray_field import DEFAULT_PACK
    return _pflag(os.environ.get('AMNI_RAY_PACK') or DEFAULT_PACK)
WCH=frozenset(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_')
_TRIM=(b" last year",b" this year",b" the most",b" these days",b" by the way",b" the",b" of",b" now",b" again");_TRIML=_TRIM+(b" the moment",b" right")
class _P:
    def __init__(s,lt=None):
        s.flg=s.pre=s.subj=s.meta=s.ante=s.own=s.pbd=s.inv=s.poss=s.entc=s.relc=s.gapc=s.qent=s.qrel=s.clk=s.nps=s.we=s.ph=s.pph=s.sw=s.cw=s.hw=0;s.lav=-9;s.wq=s.tcm=False;s.lt=_lt() if lt is None else lt;s.vb=bytearray();s.facts=[];s.names={SELFK:'I'};s.kin={};s.ments=set();s.t=b''
    def b(s,m):return s.flg&m!=0
    def on(s,*m):
        for x in m:s.flg|=x
    def off(s,*m):
        for x in m:s.flg&=~x
    def reset(s):s.poss=s.entc=s.relc=s.gapc=s.inv=s.meta=s.subj=0;s.vb.clear();s.flg&=BKEEP
    def latest(s,pred):return max((f for f in s.facts if pred(f)),key=lambda f:f[3],default=None)
    def commit(s):
        for _ in range(4):
            t=next((t for t in (_TRIML if s.lt else _TRIM) if len(s.vb)>len(t) and s.vb.endswith(t)),None)
            if t:del s.vb[-len(t):]
        rel=s.relc;ent=s.subj if rel==ISINK else s.entc
        if s.b(BSOL) and s.vb and rel==0 and (s.b(BFAV) or s.b(BMOST)):f=s.latest(lambda f:f[0]==ent and f[1]>=4);rel=f[1] if f else rel;s.relc=rel
        if s.b(BSOL) and rel==7 and s.b(BMOST):f=s.latest(lambda f:f[0]==ent and f[1] in (10,11));rel=f[1] if f else rel;s.relc=rel
        if s.b(BSOL) and s.vb and (2 if s.lt else 3)<=rel<=15 and ent and (rel==ISINK or ent==SELFK or s.b(BENTK)):s.clk+=1;s.facts.append([ent,rel,bytes(s.vb),s.clk,s.names.get(ent,'')])
        s.reset()
    def setrel(s,r,noun,force):
        s.relc=r if force or noun or s.relc==0 else s.relc
        if noun:s.qrel=r;s.on(BQLOCK)
        elif (not s.b(BQLOCK) and not (s.lt and s.qrel==2)) or (s.lt and r==2):s.qrel=r
        if not s.b(BSOL) and s.b(BPRE):s.vb.clear()
        s.off(BJOB,BHAV,BPRE)
    def person(s,e,kin):
        if e==0:return
        if not s.b(BFRO) and not s.b(BSOL):s.vb.clear()
        s.entc=e;s.poss=0;s.inv=0;s.own=e if e!=SELFK else SELFK;s.ante=e if e!=SELFK else s.ante;s.qent=e;e!=SELFK and s.ments.add(e)
        s.on(BENTK) if kin or e==SELFK else s.off(BENTK)
    def push(s,sep,w):
        if sep is not None and 0<len(s.vb)<24:s.vb.append(sep)
        s.vb+=w[:16][:max(0,24-len(s.vb))]
    def addv(s,w,cap,cls):
        if s.lt and not s.vb and s.relc==0 and not cap and 5<=len(w)<=16 and w.endswith(b'ing'):s.on(BFRO);return
        if s.b(BFRO) or (not s.b(BSOL) and (cls&4 or cls&1) and not cap) or (s.b(BSOL) and cls&1 and not cap):return
        s.push(45 if s.pbd==1 else 39 if s.pbd==2 else 32,w);s.lav=s.sw;s.on(BSOL);s.off(BART,BPRE)
        if s.b(BJOB) and s.relc==0:s.setrel(13 if cap else 5,False,False)
    def freeze(s):
        s.on(BSOL,BFRO,BVOP);s.inv=3;s.off(BPRE);s.entc=0 if s.entc==0 or s.entc==s.subj else s.entc
    def there(s):
        f=s.latest(lambda f:f[2][:1].isupper())
        if f is None:return
        s.vb[:]=f[2][:24];s.on(BSOL,BFRO,BVOP);s.relc=s.relc or 4
    def cmp(s,p,cap):
        s.ph=((s.pph*83492791)&0xFFFFFFFF)^p if s.lt and s.pph and not cap else p;return s.ph
    def name(s,e,kin,w=None):s.names[e]=(w or s.t[s.nps:s.we]).decode('latin-1').strip(' ,');s.kin[e]=kin
    def step(s,w,whw,p4w):
        n=len(w);lw=w[:16].lower();ix=whw&WM;cls,rid=GX.get(ix,(CLS.get(ix,0),RID.get(ix,0))) if s.lt else (CLS.get(ix,0),RID.get(ix,0));fun,ew=ix in FUN,s.hw if s.lt else p4w;cap=n>=2 and 65<=w[0]<=90;num=n>0 and w[:16].isdigit();pre=s.pre;b=s.b
        P=lambda k,p:n==k and lw.startswith(p)
        hd=cls&32 or (n>=3 and not fun and cls&2015==0);r=1
        if b(BSKMU) and not (P(3,b'you') or P(2,b'me') or P(5,b'ab') or P(2,b'to') or P(4,b'that')):s.off(BSKMU)
        if s.meta==1 and not P(1,b'i') and not P(3,b'you'):s.meta=0
        if s.meta in (2,3) and not cls&16:
            was=s.meta;s.meta=0
            if was==2:s.person(SELFK,True)
        if s.meta==8 and not (P(2,b'me') or P(3,b'him') or P(3,b'her') or P(4,b'them')):s.meta=0
        if s.meta>=9:
            s.meta=0
            if hd and s.pbd!=2:s.poss=1;s.own=s.ante or s.own
            else:s.person(s.ante or s.entc,True)
        if b(BACO) and P(3,b'not'):s.on(BSKNO);s.off(BACO)
        elif b(BACO) and not s.lt:s.off(BACO)
        elif b(BSKNO):r=1
        elif s.meta==4:s.meta=0
        elif P(6,b'rem'):s.meta=4
        elif b(BSKMU):s.off(BSKMU)
        elif s.meta==1 and P(1,b'i'):s.meta=2
        elif s.meta==1 and P(3,b'you'):s.meta=3
        elif s.meta in (2,3) and cls&16:
            s.on(BSKMU);s.meta==2 and s.on(BMETI);s.meta=0
        elif s.meta==5:r=1
        elif P(2,b'if') and not b(BSOL):s.meta=6
        elif s.meta==6 and P(3,b'y'):s.meta=7
        elif s.meta==6:s.meta=0
        elif s.meta==7 and P(3,b'ask'):s.meta=8
        elif s.meta==7 and (cls&16 or s.pbd==2):s.meta=5
        elif s.meta==7:s.meta=0
        elif s.meta==8 and P(2,b'me'):s.person(SELFK,True);s.meta=0
        elif s.meta==8 and n in (3,4):s.person(s.ante,True);s.meta=0
        elif s.pbd==2 and b(BSOL) and not b(BFRO):s.push(39,w)
        elif s.pbd==2 and P(1,b't'):s.on(BNEG)
        elif s.pbd==2 and n==1 and (lw in (b'm',b'r') or (lw==b's' and s.poss==0)):
            s.gapc=1;s.inv=2 if s.inv==1 else s.inv
            if b(BPRE) and s.vb:s.on(BSOL,BFRO,BVOP);s.off(BPRE)
        elif s.pbd==2 and lw==b's' and s.poss:s.gapc=1;s.own=s.entc or s.own
        elif s.pbd==2 and n==1:r=1
        elif cls&2 and not b(BSOL) and not b(BVOP) and not P(3,b'f') and not P(2,b'in'):r=1
        elif P(2,b'my') and s.relc==9 and pre==H_TO and not b(BSOL):r=1
        elif P(2,b'my') or P(4,b'your') or P(1,b'i'):
            if not P(1,b'i') and s.inv==2 and s.vb:s.freeze()
            if b(BSOL) and not b(BFRO) and s.relc==0:s.on(BFRO)
            elif b(BSOL) and not b(BFRO):s.commit()
            if P(1,b'i'):s.person(SELFK,True)
            else:s.poss=1;s.own=SELFK if n==2 else s.ante;s.nps=s.we;s.qent=s.qent or (SELFK if n==2 else 0)
        elif P(2,b'me') and pre in (H_LET,H_TELL,H_ASK):r=1
        elif P(2,b'me'):
            if b(BSOL) and s.relc==0:s.on(BFRO)
            s.person(SELFK,True)
        elif P(2,b'he') or P(3,b'she') or P(4,b'they') or P(2,b'it') or P(3,b'him'):
            e=s.own if P(2,b'it') and s.own else s.ante
            if e:s.person(e,True)
        elif (n==3 and lw[:1]==b'h' and lw[2:3]==b'r') or P(3,b'his') or P(5,b'thei'):
            if s.inv==2 and s.vb:s.freeze()
            s.meta=9 if n==3 and lw[2:3]==b'r' else 10 if n==3 else 11;s.nps=s.we
        elif s.lt and cls&2048 and s.entc and s.gapc==1 and s.relc==0 and not b(BSOL):s.poss=0;s.setrel(10,False,False);s.addv(w,cap,0)
        elif s.lt and s.entc==SELFK and not b(BSOL) and s.hw in (H_FAMILY,H_RELATIVES,H_FOLKS):s.name(H_FAMILY,True,b'family');s.person(H_FAMILY,True)
        elif num:
            if s.relc in (0,14):s.setrel(14,True,False)
            s.addv(w,False,0)
        elif P(3,b'old'):
            if pre in (H_HOW,H_YEARS) or (not b(BVOP) and not b(BSOL)):s.setrel(14,True,False)
        elif cls&512 and rid==4 and pre==H_A and n>=6:s.setrel(5,True,False);s.poss=0
        elif P(4,b'call') or (n==5 and lw[:2]==b'ca' and lw[3:4]==b'l'):s.vb.clear();s.inv=0;s.poss=0;s.setrel(6,False,False);s.on(BCALL)
        elif (b(BSOL) or b(BVOP)) and not b(BFRO) and P(4,b'work') and pre!=H_TO and not (s.lt and s.wq):s.addv(w,False,cls)
        elif P(6,b'call') or P(5,b'name'):s.setrel(6,n==5,False);s.poss=0;s.on(BVOP if s.gapc==1 else BCALL)
        elif P(4,b'home'):
            if s.inv==2 and s.vb:s.freeze();s.setrel(4,False,False);r=0
            elif b(BPRE) or b(BCALL) or s.inv==1:s.on(BSOL,BFRO,BVOP);s.relc=4
            elif s.relc==0:s.setrel(4,False,False)
            if not b(BQLOCK):s.qrel=4
        elif P(5,b'when') and b(BSOL):r=1
        elif P(5,b'wher') and s.inv==2 and s.vb:s.freeze();s.setrel(4,False,False);s.on(BWHE);r=0
        elif P(5,b'wher'):s.on(BWHE);s.setrel(4,False,False)
        elif P(4,b'when'):s.on(BWHEN);s.setrel(15,False,False)
        elif P(3,b'who'):s.on(BWHO);s.setrel(6,False,False)
        elif cls&64 and cap and s.entc==0 and s.gapc==0 and not b(BVOP) and not b(BART) and s.inv==0:s.vb[:]=w[:16];s.subj=p4w;s.inv=1
        elif cls&64 and (b(BART) or s.gapc==1 or b(BVOP) or (s.entc and s.relc in (0,15,4))):
            if s.relc in (0,15) or (s.relc==4 and not b(BSOL)):s.setrel(15,True,False)
            s.addv(w,cap,0)
        elif b(BSTUD) and not b(BFRO) and n>2 and not fun and not cls&2 and not cls&8 and not P(6,b'sc') and not P(7,b'col') and not P(10,b'u'):s.addv(w,cap,0)
        elif cls&256 and n in (6,7,10) and (b(BSTUD) or s.relc in (12,13)) and pre in (H_AT,H_IN,H_FROM,H_TO):r=1
        elif cls&256 and n in (6,7) and s.entc==0 and not b(BSOL) and pre==H_IN:r=1
        elif cls&256 and s.qrel==12 and pre==H_IN and not b(BSTUD):r=1
        elif cls&256 and cap and b(BSOL) and n in (4,6,7,10):s.addv(w,True,0)
        elif s.lt and cls&256 and b(BVOP) and not b(BFRO) and s.relc and s.relc!=rid and not s.poss and not b(BSOL) and not s.wq:s.addv(w,cap,0)
        elif cls&256:
            if s.lt and b(BSOL) and not b(BFRO) and s.lav==s.sw-1 and s.vb[:1].islower() and 32 not in s.vb:s.vb.clear();s.off(BSOL)
            if s.poss:s.person(s.own or SELFK,True)
            if rid in (10,11) and b(BFAV) and not b(BSOL):
                pe=s.entc or s.own or s.ante;f=s.latest(lambda f:pe and f[0]==pe and f[1]==7)
                if f:f[1]=rid
            s.setrel(rid,True,False);s.poss=0
        elif cls&512:
            if n==5 and lw[:1]==b'm':s.on(BSTUD)
            s.setrel(5 if n>=6 and pre==H_A and rid==4 else rid,False,False);s.poss=0
        elif cls&1024:
            if n>=5 and lw[:2]==b'st':s.on(BSTUD)
            s.setrel(rid,False,rid==11 and s.relc==7);s.poss=0;s.on(BVOP);s.off(BCALL)
        elif P(3,b'fan'):s.setrel(7,False,False);s.poss=0
        elif P(4,b'baby'):s.setrel(15,True,False)
        elif (P(4,b'gr') and (not s.lt or lw==b'grad')) or P(4,b'alu'):s.setrel(13,True,False)
        elif P(6,b'ar') and (b(BNEG) or s.relc==8):s.setrel(8,False,False);s.on(BVOP);s.poss=0
        elif P(4,b'away'):s.setrel(8,False,False)
        elif cls&8 and not b(BSOL) and not b(BVOP) and s.gapc==0 and not P(3,b'had'):
            if (P(4,b'hav') or P(3,b'has')) and (s.entc or s.poss):s.on(BHAV);s.poss=0
            else:
                if P(2,b'd') or P(3,b'di') or P(4,b'do'):s.on(BDO)
                s.meta=1
        elif (P(3,b'got') or P(4,b'have') or P(6,b'gott')) and s.entc==0 and not b(BSOL) and s.poss==0:s.meta=1
        elif P(3,b'got') or P(4,b'have') or P(3,b'has') or P(3,b'had') or P(6,b'gott'):s.on(BHAV);s.poss=0
        elif P(4,b'been'):s.off(BHAV);s.gapc=1
        elif b(BART) and b(BHAV) and cls&32 and not cap:s.name(ew,True,w);s.person(ew,True);s.off(BHAV,BART)
        elif P(4,b'most'):s.on(BMOST)
        elif P(8,b'fav'):s.on(BFAV)
        elif s.poss and (P(3,b'bi') or P(4,b'be') or (P(6,b'l') and (not s.lt or lw[1:3] in (b'it',b'ov',b'on',b'at'))) or P(3,b'new')):r=1
        elif s.poss and s.pbd==1 and P(2,b'in'):r=1;s.ph=s.pph
        elif s.poss and hd and not (b(BKINL) and not cls&32 and s.pbd!=1):
            e=s.cmp(ew,cap);s.name(e,bool(cls&32));s.person(e,bool(cls&32) or s.lt);s.poss=1
            if cls&32 or s.lt:s.on(BKINL)
        elif s.poss and hd:s.poss=0;r=0
        elif P(2,b'is') or P(3,b'are') or P(3,b'was') or P(4,b'were') or P(2,b'am'):
            s.gapc=1;s.inv=2 if s.inv==1 else s.inv
            if b(BPRE) and s.vb:s.on(BSOL,BFRO,BVOP);s.off(BPRE)
        elif P(1,b'a') or P(2,b'an'):
            s.on(BART)
            if (s.gapc==1 or b(BHAV)) and s.relc==0:s.on(BJOB)
            if s.gapc==1 or s.relc or s.entc:s.on(BVOP)
        elif s.inv==2 and s.vb and (P(4,b'over') or P(3,b'all') or P(3,b'the') or P(3,b'way')):r=1
        elif s.inv==2 and s.vb and P(2,b'in'):
            fw=bh(bytes(c for c in s.vb[:4] if c!=32));s.name(fw,False,bytes(s.vb));s.subj=s.entc=fw;s.relc=3;s.vb.clear();s.off(BSOL,BFRO);s.on(BVOP);s.inv=0
        elif s.inv==2 and s.vb:s.freeze();r=0
        elif s.inv==1 and (cap or (cls&128 and s.vb)):
            s.push(32,w)
            if cap and s.subj==0:s.subj=p4w
        elif P(2,b'in') and b(BSOL):s.addv(w,False,0)
        elif P(4,b'into'):s.setrel(7,False,True);s.on(BVOP)
        elif P(4,b'than') and s.relc==0:s.setrel(11,False,True);s.on(BVOP)
        elif P(5,b'ab') and pre==H_CRAZY:s.setrel(11,True,False);s.on(BVOP)
        elif P(5,b'ab') and pre==H_ALL:s.setrel(7,False,True);s.on(BVOP)
        elif P(2,b'of') and pre==H_FAN:s.setrel(7,False,True);s.on(BVOP)
        elif P(2,b'by') and pre in (H_GO,H_GOES):s.setrel(6,True,False);s.on(BVOP);s.off(BSOL);s.vb.clear()
        elif P(2,b'in') and pre in (H_DEGREE,H_MAJOR):s.setrel(12,True,False);s.on(BVOP)
        elif (P(2,b'at') or P(4,b'fro')) and s.relc==12 and not b(BSOL):s.setrel(13,True,False);s.on(BVOP)
        elif P(3,b'for') and pre==H_SCHOOL:s.setrel(12,True,False);s.off(BSOL);s.vb.clear();s.on(BVOP)
        elif P(2,b'to') and pre in (H_WENT,H_GO,H_GOING):s.setrel(13,False,False);s.on(BVOP)
        elif P(2,b'in') and pre==H_AROUND:s.setrel(9,False,True);s.on(BVOP)
        elif P(3,b'for') and pre==H_IN:s.setrel(9,False,True);s.on(BVOP);s.off(BSOL);s.vb.clear()
        elif P(2,b'in') and s.relc==0 and s.entc and s.inv==0 and not b(BSOL):s.setrel(4,False,False);s.on(BVOP)
        elif P(3,b'for') or P(2,b'in') or P(2,b'to') or (P(4,b'fr') and (not s.lt or lw==b'from')) or (P(5,b'ab') and (not s.lt or lw==b'about')) or P(2,b'of') or (P(4,b'in') and (not s.lt or lw==b'into')) or P(2,b'as') or P(2,b'by'):
            if b(BSOL) and b(BVOP) and not b(BFRO) and (P(2,b'in') or P(2,b'of') or (P(4,b'in') and (not s.lt or lw==b'into'))):s.addv(w,False,0)
            elif b(BSOL):s.off(BVOP)
            elif s.entc==0 and s.inv==0 and not b(BVOP) and s.relc==0:r=1
            elif not b(BFRO):
                if b(BPRE):s.vb.clear();s.off(BPRE)
                s.on(BVOP)
            s.inv=0 if s.inv==1 else s.inv
        elif P(5,b'ther') or P(4,b'here'):
            if s.entc and not b(BSOL):s.there()
        elif b(BHAV) and cap and not b(BFRO) and s.relc in (0,9):s.setrel(9,False,False);s.addv(w,True,0)
        elif b(BCALL) and cap and not b(BFRO):
            if s.relc==0:s.setrel(6,False,False)
            s.addv(w,True,0)
        elif P(3,b'the') and b(BVOP) and not b(BSOL) and s.relc!=3:s.addv(w,False,0)
        elif s.entc and s.poss==0 and not b(BSOL) and s.gapc==0 and not b(BVOP) and not b(BFRO) and not b(BCALL) and not fun and not cls&11 and hd:
            if not s.vb:s.push(None,w)
            s.on(BPRE)
        elif P(5,b'fa') and not b(BSOL) and not b(BVOP):r=1
        elif b(BSOL) and not b(BFRO) and P(3,b'and'):s.push(32,w)
        elif b(BSOL) and b(BVOP) and not b(BFRO) and (P(3,b'the') or P(3,b'and') or P(2,b'of') or P(2,b'in')):s.addv(w,False,0)
        elif P(5,b'ev') and b(BSOL) and s.relc==0:s.setrel(11,True,False)
        elif (b(BVOP) or s.gapc==1 or b(BCALL)) and not b(BFRO) and (not fun or cap) and (not cls&11 or cap):
            if cap and s.gapc==1 and not b(BART) and s.relc in (0,6):s.setrel(6,False,False)
            s.addv(w,cap,cls)
        elif b(BSOL) and b(BVOP) and not b(BFRO) and not cls&11 and not fun:s.addv(w,cap,cls)
        elif cap and s.entc==0 and s.poss==0 and not fun:s.vb[:]=w[:16];s.subj=p4w;s.inv=1
        elif b(BPRE) and cap:s.push(32,w)
        s.lt and s.off(BACO)
        return r
    def word(s,w):
        whw,p4w=bh(w),bh(w[:4]);s.hw=bh(w[:16].lower())
        s.pph,s.ph,s.wq,s.sw,s.cw,s.tcm=s.ph,0,(s.hw in WQS[:16] or s.hw in WIM or (s.sw==0 and s.hw in WQS)) if s.cw==0 else s.wq,s.sw+1,s.cw+1,s.tcm or (s.lt and s.hw in WCM)
        for _ in range(2):
            if s.step(w,whw,p4w)==1:break
        s.pre=bh(w[:16].lower())
    def finish_q(s):
        s.qent=s.qent or (SELFK if s.b(BMETI) else 0);s.qrel=s.qrel or (6 if s.b(BWHO) else 15 if s.b(BWHEN) else 4 if s.b(BWHE) else 5 if s.b(BDO) else 0)
    def punct(s,c):
        end,nq=c in (46,33,63,10),s.lt and (c==63 or s.wq)
        if s.meta==5 and c==44:s.meta=0
        elif s.b(BSKNO) and end:s.off(BSKNO);s.reset()
        elif c==44 and not s.b(BSOL):s.poss=0;s.off(BKINL,BVOP,BPRE);s.gapc=0;s.inv=0;s.vb.clear()
        elif c==63 and not s.b(BSOL):s.finish_q();s.off(BVOP);s.gapc=0;s.inv=0;s.vb.clear()
        elif c==44:s.reset() if nq else (s.commit(),s.on(BACO))
        elif end:
            if c==63:s.finish_q()
            s.reset() if nq else s.commit()
        if s.lt and (end or c==44):s.sw,s.cw,s.wq=0 if end else s.sw,0,False
    def turn(s,t):
        s.qent=s.qrel=s.flg=s.pbd=s.ph=s.sw=s.cw=0;s.wq=s.tcm=False;s.reset();s.ments=set();s.t=t=t+b'\n';w=bytearray();f0=len(s.facts)
        for i,c in enumerate(t):
            if c in WCH:w.append(c);continue
            if w:s.we=i;s.word(bytes(w));w.clear();s.pbd=1 if c==45 else 2 if c==39 else 0
            if c in (46,44,33,63,10):s.punct(c)
        return s.facts[f0:]
_enc=lambda x:(x if isinstance(x,bytes) else str(x).encode('utf-8','replace'))
_FORGET=re.compile(r"(?i)^\W*(?:(?:please|ok(?:ay)?|hey|adam|and|also|now)\W+)*(?:forget|erase|delete|remove|wipe)\b(?!\W*(?:it|about it|that)\W*$)")
_ALL=re.compile(r"(?i)\b(?:everything|all)\b")
_PRON=re.compile(r"(?i)^\W*(?:actually,? |also,? |and )?(?:he|she|they)(?:'s| is| was| has| drives| works| lives| moved| likes| loves| studied| goes)\b")
_REQ=re.compile(r"(?i)^\W*(?:(?:please|hello|hi|hey|ok(?:ay)?)\W+)*(?:give|make|list|classify|write|extract|name me|pretend|create|generate|provide|help|find|show|explain|describe|suggest|recommend|summari[sz]e|translate|tell me|can|could|would)\b")
_BADV=re.compile(r"(?i)^(?:(?:bulleted |numbered |check)?list|bullet|looking|learning|trying|wondering|thinking|planning|going|doing|find|start|located|write|use|read)\b|\b(?:and|or|in|of|the|a|an|to|for|with|at|on|by)$")
def keep(fs:list,msg)->list:
    m=_enc(msg);ss=re.split(rb'(?<=[.!?])\s+',m.strip());return [] if _REQ.search(m.decode('utf-8','replace')) else [f for f in fs if len(f[2])<24 and f[2] in m and not _BADV.search(f[2].decode('latin-1')) and not any(f[2] in x and (_isq(x.decode('utf-8','replace')) or _REQ.search(x.decode('utf-8','replace'))) for x in ss)]
_Q=re.compile(r"(?i)^\W*(?:(?:so|and|ok(?:ay)?|hey|um|wait|also)\W+)*(?:who|whom|whose|what|when|where|which|why|how|do|does|did|is|are|was|were|can|could|would|should|will|am|remind me)\b")
_isq=lambda m:m.endswith('?') or bool(_Q.search(m)) and not m.endswith(('.','!'))
_SELFQ=re.compile(r"(?i)\b(?:do|did|am|was|have|had) i\b(?!')|\bhow old am i\b|\bremind me\b|\bdo you (?:remember|know)\b|\bmy (?:name|job|age|birthday|favou?rite|car|allerg\w*|hobb\w*|home|city|town|school|college|degree|major)\b|\bi(?:'m| am) (?:in|from|based)\b|\bwhere i (?:live|work)\b")
_TPL={3:("{e} is in {v}.",),4:("I live in {v}.","My {e} lives in {v}."),5:("I work as a {v}.","My {e} works as a {v}.","I'm a {v}.","My {e} is a {v}."),6:("My name is {v}.","My {e}'s name is {v}.","My {e} is called {v}."),7:("I love {v}.","My {e} loves {v}.","I really like {v}.","My {e} really likes {v}."),8:("I'm allergic to {v}.","My {e} is allergic to {v}."),9:("I drive {v}.","My {e} drives {v}.","I drive a {v}.","My {e} drives a {v}."),10:("My favorite color is {v}.","My {e}'s favorite color is {v}.","My {e} is {v}.","When it comes to colors, I love {v} the most.","When it comes to colors, my {e} loves {v} the most."),11:("My favorite food is {v}.","My {e}'s favorite food is {v}.","Out of everything, I love {v} the most.","Out of everything, my {e} loves {v} the most."),12:("I studied {v}.","My {e} studied {v}.","I majored in {v}.","My {e} majored in {v}."),13:("I went to {v}.","My {e} went to {v}.","I graduated from {v}.","My {e} graduated from {v}."),2:("I hate {v}.","My {e} hates {v}.","I dislike {v}.","My {e} dislikes {v}."),14:("I am {v}.","My {e} is {v}.","I'm {v} years old.","My {e} is {v} years old."),15:("My birthday is in {v}.","My {e}'s birthday is in {v}.","I was born in {v}.","My {e} was born in {v}.")}
_CV=contextvars.ContextVar('amni_user_memory',default=(None,None));_lock=threading.RLock();_cache={};_rcache={}
def enabled()->bool:return os.environ.get('AMNI_USER_MEMORY','1')!='0'
def root()->str:return os.environ.get('AMNI_UMEM_DIR') or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'experiences','user_memory')
def bind(user=None,session=None):return _CV.set((user,session))
def bound(it,user=None,session=None):
    while True:
        tk=_CV.set((user,session))
        try:x=next(it)
        except StopIteration:return
        finally:_CV.reset(tk)
        yield x
def _salt()->bytes:
    p=os.path.join(root(),'.salt')
    with _lock:
        if not os.path.exists(p):os.makedirs(root(),mode=0o700,exist_ok=True);fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.write(fd,secrets.token_bytes(32));os.close(fd)
        return open(p,'rb').read()
def user_key(user=None)->str:
    u=user if user is not None else _CV.get()[0];u=u if u is not None else os.environ.get('AMNI_USER_ID','owner')
    return hashlib.sha256(_salt()+b'\0'+_enc(u)).hexdigest()[:32]
def path(key:str)->str:return os.path.join(root(),re.sub(r'[^0-9a-f]','',key)[:32]+'.ptex')
def encode(fs:list)->bytes:
    pay=b''.join(REC.pack(f.ent,f.rel,f.clk,f.ts,min(f.turn,65535),f.sid,len(f.val),len(nb))+f.val+nb for f in fs for nb in [f.name.encode('utf-8')[:255]])
    u=np.frombuffer(pay+b'\0'*(len(pay)&1),'<u2');px,_=pack_rgba4_page(u.view(np.float16),page_w=PW);npg=-(-px.shape[0]//PH) if len(pay) else 0
    pg=np.zeros((npg*PH,PW,4),np.uint8);pg[:px.shape[0]]=px;pg=pg.reshape(npg,PH,PW,4);pb=PW*PH*2
    d=b''.join(struct.pack('<II',min(pb,len(pay)-i*pb),zlib.crc32(pg[i].tobytes())) for i in range(npg));h=HDR.pack(MAGIC,VER,PW,PH,npg,len(fs),len(pay),max((f.clk for f in fs),default=0))+d
    return h+struct.pack('<I',zlib.crc32(h))+pg.tobytes()
def decode(raw:bytes)->list:
    mg,ver,pw,ph,npg,nrec,nb,_=HDR.unpack_from(raw);dl=HDR.size+8*npg
    if mg!=MAGIC or ver!=VER or struct.unpack_from('<I',raw,dl)[0]!=zlib.crc32(raw[:dl]):raise ValueError('user memory header corrupt')
    pg=np.frombuffer(raw,np.uint8,npg*ph*pw*4,dl+4).reshape(npg,ph,pw,4)
    if any(struct.unpack_from('<II',raw,HDR.size+8*i)[1]!=zlib.crc32(pg[i].tobytes()) for i in range(npg)) or (pg>=17).any():raise ValueError('user memory page corrupt')
    pay=decode_rgba4_to_fp16(pg).view('<u2').tobytes()[:nb];fs=[];o=0
    for _ in range(nrec):
        e,r,c,ts,tn,sid,vl,nl=REC.unpack_from(pay,o);o+=REC.size;fs.append(F(e,r,c,ts,tn,sid,pay[o:o+vl],pay[o+vl:o+vl+nl].decode('utf-8','replace')));o+=vl+nl
    return fs
_EID={}
def _eid(name:str,lt:bool):
    k=(name,lt)
    if k not in _EID:fs=_P(lt=lt).turn(f"My {name} lives in Xq.".encode('latin-1','replace'));_EID[k]=fs[-1][0] if fs and fs[-1][1]==4 else None
    return _EID[k]
def _mig(fs:list)->list:return [f._replace(ent=n) if f.ent!=SELFK and f.rel!=ISINK and f.name and _eid(f.name,False)==f.ent and (n:=_eid(f.name,True)) else f for f in fs] if _lt() else fs
def load(key:str)->list:
    p=path(key)
    with _lock:
        st=os.stat(p).st_mtime_ns if os.path.exists(p) else None;c=_cache.get(key)
        if c and c[0]==st:return list(c[1])
        fs=_mig(decode(open(p,'rb').read())) if st else [];_cache[key]=(st,fs);return list(fs)
def save(key:str,fs:list):
    p=path(key)
    with _lock:
        os.makedirs(root(),mode=0o700,exist_ok=True);_cache.pop(key,None)
        if not fs:os.path.exists(p) and os.remove(p);return
        t=p+f'.{os.getpid()}.tmp';fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'wb') as fh:fh.write(encode(fs));fh.flush();os.fsync(fh.fileno())
        os.replace(t,p)
def latest(fs:list)->list:
    d={}
    for f in fs:d[(f.ent,f.rel)]=f if (f.ent,f.rel) not in d or f.clk>d[(f.ent,f.rel)].clk else d[(f.ent,f.rel)]
    return sorted(d.values(),key=lambda f:f.clk)
def remember(key:str,facts:list,session=None,turn:int=0)->int:
    with _lock:
        fs=load(key);lt={(f.ent,f.rel):f for f in latest(fs)};clk=max((f.clk for f in fs),default=0);sid=hashlib.sha256(_enc(session or '')).digest()[:8];n=0
        for e,r,v,_,nm in facts:
            if (e,r) in lt and lt[(e,r)].val==v:continue
            clk+=1;f=F(e,r,clk,time.time(),turn,sid,v[:24],nm);fs.append(f);lt[(e,r)]=f;n+=1
        n and save(key,fs);return n
def forget(key:str,ent=None,rel=None)->int:
    with _lock:
        fs=load(key);keep=[f for f in fs if ent is not None and not (f.ent==ent and (not rel or f.rel==rel))];save(key,keep);return len(fs)-len(keep)
def render(f)->str:
    k=(f.ent,f.rel,f.val,f.name)
    if k not in _rcache:
        v,e=f.val.decode('latin-1'),f.name or 'friend';ts=_TPL.get(f.rel,())
        ts=sorted([t for t in ts if ('{e}' in t)==(f.ent!=SELFK)],key=lambda t:f.rel==10 and not CLS.get(bh(e.split()[-1].encode('latin-1','replace'))&WM,0)&32 and 'favorite' in t)
        _rcache[k]=next((x for t in ts for x in [t.format(e=e,v=v)] if [g[:3] for g in _P().turn(x.encode('latin-1','replace'))][-1:]==[[f.ent,f.rel,f.val]]),None)
    return _rcache[k]
def _vp4(v:bytes)->int:return bh(bytes(c for c in v[:4] if c!=32))
def preamble(fs:list,p,hk:set,hn:int=0)->str:
    q,qr=p.qent,p.qrel;fs=[f for f in latest(fs) if (f.ent,f.rel) not in hk];vp={_vp4(f.val) for f in fs if f.ent==q}
    rv=lambda f:2 if f.ent==q and (not qr or f.rel==qr) else 1 if f.ent==q or (f.rel==ISINK and f.ent in vp) else 0
    cap=max(8,min(int(os.environ.get('AMNI_UMEM_CAP','64')),NR-hn));pick=sorted(fs,key=lambda f:(rv(f),f.clk))[-cap:]
    return ''.join(f'User: {t}\nAdam: Got it.\n\n' for f in pick for t in [render(f)] if t)
def _target(p,m:str):return 'all' if _ALL.search(m) and p.qent in (0,SELFK) and not re.search(r'(?i)\bmy\b',m) else (p.qent,p.qrel) if p.qent else None
def parse(message:str,history=None):
    p=_P();turns=[];hf=[];hents=set()
    for u,_ in history or []:
        fs=keep(p.turn(_enc(u).strip()),u);fg=_FORGET.search(str(u)) and _target(p,str(u));turns.append((fs,fg))
        if not _isq(str(u).strip()):hents|=p.ments
    drop={i for i,(fs,_) in enumerate(turns) for j in range(i+1,len(turns)) for t in [turns[j][1]] if t and fs and (t=='all' or any(f[0]==t[0] and (not t[1] or f[1]==t[1]) for f in fs))}
    for i,(fs,_) in enumerate(turns):hf+=[] if i in drop else fs
    mf=keep(p.turn(_enc(message).strip()),message);return p,mf,hf,hents,drop
def turn(message:str,history=None,user=None,session=None,dry:bool=False)->dict:
    hist=list(history or []);win=hist[-int(os.environ.get('AMNI_RAY_HIST','24')):];m=str(message).strip();q=_isq(m);p,mf,hf,hents,drop=parse(m if m.endswith('?') or not q else m+'?',win)
    on=enabled();key=user_key(user) if on else None;fs=load(key) if on else [];sess=session if session is not None else _CV.get()[1]
    hwin=[x for i,x in enumerate(win) if i not in drop];out={'route':False,'pre':'','hist':hwin,'reply':None,'facts':[f[:3] for f in mf],'user_key':key}
    tg=_FORGET.search(m) and not q and _target(p,m)
    if tg and not dry:
        nm=p.names.get(tg[0],'') if tg!='all' else '';n=(save(key,[]) or len(fs)) if on and tg=='all' else forget(key,tg[0],tg[1] if any(f.ent==tg[0] and f.rel==tg[1] for f in fs) else 0) if on else 0
        out['reply']="Okay, I've forgotten everything you told me." if tg=='all' else (f"Okay, I've forgotten what you told me about {'yourself' if tg[0]==SELFK else 'your '+nm}." if n or any(f[0]==tg[0] for f in hf) else "I don't have anything saved about that.");return out
    if not q:
        out['route']=bool(mf) or bool(_PRON.search(m))
        if on and mf and not dry:remember(key,mf,sess,len(hist))
        return out
    allf=[(f.ent,f.rel) for f in fs]+[(f[0],f[1]) for f in hf];e=p.qent
    out['route']=bool(e) and (bool(_SELFQ.search(m)) and any(a==SELFK for a,_ in allf) if e==SELFK else e in {a for a,_ in allf}|hents or p.kin.get(e,False))
    out['pre']=preamble(fs,p,{(f[0],f[1]) for f in hf},len(hf)) if on and out['route'] and fs else ''
    return out

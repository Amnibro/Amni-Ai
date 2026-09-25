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
BSAME,BQDONE,BSHR=1<<23,1<<25,1<<26;BKEEP|=BQDONE;GK=0x9E3779B1
H_SAME,H_THE,H_WHAT,H_WHICH,H_NEVER,H_NOTHING,H_OLD,H_AS,H_WITH,H_PART,H_OF,H_ON,H_NOT2,H_GO2,H_GOES2,H_WENT2,H_GOING2=(bh(w.encode()) for w in 'same the what which never nothing old as with part of on not go goes went going'.split())
STOPV={bh(w.encode()) for w in 'but although whenever whereas unless until since though'.split()};SHARE={bh(w.encode()) for w in 'share shares shared sharing'.split()};RELW={bh(w.encode()) for w in 'which who where whose'.split()};POSW={bh(w.encode()) for w in 'my your his her their s'.split()}
_STOPW="the a an and or of to in on for with is are was were be been being do does did what why how whom which when where can could would should will shall may might must i you me my your it its this that these those there their they them he she his her we our us as at by from about into than then so if not no yes please tell give explain describe there here just also some any my i'm i is are was in at to of the a an and called named works as from actually oh well now really anyway okay ok yeah hey btw recently fun fact guess what though"
_G=None
def _gtab():
    global _G
    if _G is None:
        rows=[(r[1],r[2],int(r[3]),int(r[4])) for r in (l.split('\t') for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'crates','adam-ray','src','v19_rel.tsv'),encoding='utf-8').read().splitlines()) if len(r)>=5]
        stop={i for w in _STOPW.split() for i in _both(w)};prot=set("home work call called calls calling name names named where when who old into than fan baby grad alum most favorite favourite fav been have has got had going went go goes born birth same".split())
        known=lambda w:(lambda i:CLS.get(i,0)!=0 or i in FUN or i in stop or w in prot or i in GX)(bh(w.encode())&WM);gt={}
        for p,w,c,r in rows:
            if c<4096 and (p or not known(w)):gt.setdefault(bh(w.encode()) if not p else ((bh(p.encode())*GK)&0xFFFFFFFF)^bh(w.encode()),(c,r))
        vt={}
        for p,w,c,r in rows:
            if c==8192:vt.setdefault(bh(w.encode()),r)
        _G=(gt,vt,{bh(w.encode()) for p,w,c,r in rows if c==4096 and not known(w)},{bh(w.encode()) for p,w,c,r in rows if c==16384},{bh(w.encode()) for p,w,c,r in rows if c==32768},{bh(w.encode()) for p,w,c,r in rows if c==65536 and not known(w)}|{H_NOTHING})
    return _G
WQS=[bh(w.encode()) for w in 'what where who when which how why whose do does did can could would will remind is are'.split()];WIM={bh(w.encode()) for w in 'give tell list write explain describe create make find identify classify provide generate summarize translate compare suggest recommend show define calculate build reply label choose name rewrite extract categorize imagine'.split()};WCM={bh(w.encode()) for w in 'actually correction scratch update updated wrong oops misspoke fix mixed instead now moved changed switched longer anymore traded sorry wait corrected new'.split()}
_LTC={}
def _pflag(p:str)->bool:
    j=(p[:-8] if p.endswith('_sid.npy') else p[:-5] if p.endswith('.gf17') else p)+'.json'
    if j not in _LTC:
        try:_LTC[j]='"latest": 1' in open(j,encoding='utf-8',errors='replace').read()
        except OSError:_LTC[j]=False
    return _LTC[j]
def _gn()->bool:
    v=os.environ.get('AMNI_UMEM_GN')
    if v is not None:return v!='0'
    from amni.compute.ray_field import DEFAULT_PACK
    p=os.environ.get('AMNI_RAY_PACK') or DEFAULT_PACK;j=(p[:-8] if p.endswith('_sid.npy') else p[:-5] if p.endswith('.gf17') else p)+'.json'
    if j+'#gen' not in _LTC:
        try:_LTC[j+'#gen']='"gen": 1' in open(j,encoding='utf-8',errors='replace').read()
        except OSError:_LTC[j+'#gen']=False
    return _pflag(p) and _LTC[j+'#gen']
def _lt()->bool:
    v=os.environ.get('AMNI_UMEM_LT')
    if v is not None:return v!='0'
    from amni.compute.ray_field import DEFAULT_PACK
    return _pflag(os.environ.get('AMNI_RAY_PACK') or DEFAULT_PACK)
def _g2()->bool:
    v=os.environ.get('AMNI_UMEM_G2')
    if v is not None:return v!='0'
    from amni.compute.ray_field import DEFAULT_PACK
    p=os.environ.get('AMNI_RAY_PACK') or DEFAULT_PACK;j=(p[:-8] if p.endswith('_sid.npy') else p[:-5] if p.endswith('.gf17') else p)+'.json'
    if j+'#g2' not in _LTC:
        try:_LTC[j+'#g2']='"gen2": 1' in open(j,encoding='utf-8',errors='replace').read()
        except OSError:_LTC[j+'#g2']=False
    return _gn() and _LTC[j+'#g2']
WCH=frozenset(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_')
_TRIM=(b" last year",b" this year",b" the most",b" these days",b" by the way",b" the",b" of",b" now",b" again");_TRIML=_TRIM+(b" the moment",b" right")
class _HD2(dict):
    def __missing__(s,k):v=bh(k.encode());s[k]=v;return v
_H2=_HD2()
def _pk(hs):
    k=hs[0]
    for h in hs[1:]:k=((k*GK)&0xFFFFFFFF)^h
    return k
def _fb(h,b):return G2F.get(h,0)>>b&1
_MON2,_FAV2,_ADV2,_DIT2=({bh(w.encode()) for w in x.split()} for x in ("january february march april may june july august september october november december","favourite favorite fave fav go-to all-time top","really just also always still truly totally absolutely seriously pretty super so very actually even currently usually sometimes often mostly definitely honestly literally basically","give gave feed fed buy bought get got bring brought make made send sent offer hand cook cooked"))
G2C,G2R,G2T,G2HD,G2V,G2M,G2S,G2X,G2Y,G2K,G2J,G2F,G2CF,G2A={},{},{},set(),set(),set(),set(),set(),set(),set(),set(),{},set(),set()
def _g2init():
    if G2F:return
    for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'crates','adam-ray','src','v19_gen2.tsv'),encoding='utf-8'):
        r=l.rstrip('\n').split('\t')
        if len(r)<5:continue
        c,rl,f=int(r[2]),int(r[3]),int(r[4]);ws=r[1].split(' ');k=_pk([bh(w.encode()) for w in ws])
        if c==1:G2C.setdefault(k,[]).append((rl,f,len(ws)));G2CF.add(bh(ws[0].encode()))
        elif c==2:G2R.setdefault(k,rl)
        elif c==3:G2T.setdefault(k,(rl,f))
        elif c==12:G2F[k]=G2F.get(k,0)|(1<<rl)
        elif c in (4,5,6,7,8,9,10,11,13):{4:G2HD,5:G2V,6:G2M,7:G2S,8:G2X,9:G2Y,10:G2K,11:G2J,13:G2A}[c].add(k)
class _G2:
    def __init__(s):_g2init();s.lm=[];s.tk=[];s.ante=0;s.links=[];s.lastp=b'';s.pend=None;s.lastv=None;s.tf=[];s.qe=s.qr=0;s.nm={}
    def newturn(s):s.pend=None;s.lastv=None;s.tf=[];s.qe=s.qr=0
    def word(s,w,n,pbd):
        w=w[:16];lw=w.lower();h=bh(lw);T=s.tk;p=T[-1] if T and T[-1][6]==0 else None
        if pbd==2 and p:
            if lw==b's':
                if p[0] in (_H2['he'],_H2['she'],_H2['it'],_H2['that'],_H2['there'],_H2['what'],_H2['where'],_H2['who'],_H2['when'],_H2['how'],_H2['here']):len(T)<48 and T.append([_H2['is'],b's',False,False,False,False,0])
                else:p[4]=True;p[1]=(p[1]+b"'s")[:24]
                return
            if lw==b't':p[5]=True;return
            if lw in (b'm',b're'):len(T)<48 and T.append([_H2['am'] if lw==b'm' else _H2['are'],lw,False,False,False,False,0]);return
            if lw==b'd':len(T)<48 and T.append([_H2['would'],lw,False,False,False,False,0]);return
            if lw in (b've',b'll'):len(T)<48 and T.append([_H2['have'] if lw==b've' else _H2['will'],lw,False,False,False,False,0]);return
            p[1]=(p[1]+b"'"+w)[:24];p[0]=bh(p[1].lower());return
        if pbd==1 and p:p[1]=(p[1]+b'-'+w)[:24];p[0]=bh(p[1].lower());p[3]=p[3] and w.isdigit();return
        if len(T)<48:T.append([h,bytes(w),65<=w[0]<=90,w.isdigit(),False,False,0])
    def comma(s):
        if s.tk and len(s.tk)<48:s.tk.append([0,b',',False,False,False,False,1])
    def end(s,nq,mem):
        T=s.tk;s.tk=[]
        if T:s.run(T,nq,mem)
    def run(s,T,nq,mem):
        n=len(T);hs=[t[0] for t in T];bd=[t[6] for t in T];cap=[t[2] for t in T];num=[t[3] for t in T];capx=[t[2] or (not t[6] and 48<=t[1][0]<=57) for t in T]
        wd=lambda i:0<=i<n and not bd[i]
        cl=[0]*n;c=0
        for i in range(n):
            if bd[i] or (hs[i] in (_H2['and'],_H2['but'],_H2['so'],_H2['while'],_H2['whereas']) and i+1<n and (hs[i+1] in (_H2['my'],_H2['our'],_H2['i'],_H2['he'],_H2['she'],_H2['they'],_H2['his'],_H2['her'],_H2['their']))):c+=1
            cl[i]=c
        neg=[t[5] or _fb(t[0],10) for t in T]
        ifr=[False]*n;inif=False
        for i in range(n):
            inif=(inif and not bd[i]) or hs[i]==_H2['if'];ifr[i]=inif
        hyp=[False]*n
        for i in range(n):hyp[i]=(i>0 and hyp[i-1] and cl[i]==cl[i-1]) or hs[i] in G2Y
        cov=[0]*n;men=[];rnh=[];mself=False
        def rnmatch(j):
            k=j;fav=False
            while True:
                for L in (4,3,2,1):
                    if k+L<=n and all(wd(x) for x in range(k,k+L)):
                        r=G2R.get(_pk(hs[k:k+L]))
                        if r is not None and not (r==0 and not fav and hs[k] in (_H2['thing'],_H2['one'])):return (k+L,7 if r==0 and fav and hs[k] in (_H2['thing'],_H2['one']) else r,fav)
                if k<n and wd(k) and hs[k] in G2M and k-j<3:fav=fav or hs[k] in _FAV2;k+=1
                else:break
            if fav and (k>=n or not wd(k) or _fb(hs[k],5) or hs[k] in G2S):return (k,0,fav)
            return None
        def npend(j,inv):
            k=j;hd=False
            while k<n and k-j<4 and wd(k):
                h=hs[k];r=T[k][1]
                if cap[k] or num[k] or G2F.get(h,0) or h in G2S or neg[k] or (hd and h not in G2HD):break
                if k==j and (h in G2V or h in G2CF) and not T[k][4] and not (k+1<n and wd(k+1) and not G2F.get(hs[k+1],0) and hs[k+1] not in G2S and not cap[k+1]):break
                if k>j:
                    isv=h in G2V or h in G2CF or (k>j and h not in G2HD and ((r.endswith(b's') and not r.endswith(b'ss')) or r.endswith(b'ed') or r.endswith(b'ing'))) or (inv and k>j and h in G2A and (k+1>=n or not wd(k+1) or ((G2F.get(hs[k+1],0) or hs[k+1] in G2S) and not _fb(hs[k+1],6) and not _fb(hs[k+1],5))))
                    if inv and hd and h in G2A and h not in G2HD:break
                    if isv and not (k+1<n and wd(k+1) and hs[k+1] in G2HD and not G2F.get(hs[k+1],0) and not cap[k+1] and not T[k][4] and not (inv and (k+2>=n or not wd(k+2)))):break
                hd=hd or h in G2HD;k+=1
                if T[k-1][4]:break
            return k
        def resolve(i,kind):
            ks=[m[2] for m in men if m[2]!=SELFK]
            if kind==2:
                sub=next((m[2] for m in men if cl[m[0]]==cl[i] and m[3]!=3),None);ks2=[x for x in ks if x!=sub]
                if ks2:return ks2[-1]
            return ks[-1] if ks else s.ante
        i=0
        while i<n:
            h=hs[i]
            if not wd(i):i+=1;continue
            ocue=i>0 and any(f&1 for r,f,L in G2C.get(hs[i-1],()))
            if h in (_H2['my'],_H2['our']) or (h in (_H2['his'],_H2['her'],_H2['their'],_H2['its']) and wd(i+1) and not _fb(hs[i+1],1) and not (h==_H2['her'] and ocue) and not (h==_H2['her'] and G2F.get(hs[i+1],0))):
                own=SELFK if h in (_H2['my'],_H2['our']) else resolve(i,1);j=i+1;key=None;inv=i>0 and _fb(hs[i-1],6);dit=i>0 and hs[i-1] in _DIT2
                while j<n:
                    r=rnmatch(j)
                    if r and r[1]==9 and T[r[0]-1][4]:r=None
                    if r:
                        rnh.append((j,r[0],r[1],key if key is not None else own,r[2],i));[cov.__setitem__(x,2) for x in range(j,r[0])]
                        if men and men[-1][1]==j and men[-1][3]==3:men[-1][3]=0
                        j=r[0];break
                    k=npend(j,inv);k=j+1 if dit and k>j+1 and not (k<n and wd(k) and _fb(hs[k],1)) else k
                    if k==j:break
                    for x in range(j,k):key=hs[x] if key is None else ((key*83492791)&0xFFFFFFFF)^hs[x]
                    men.append([j,k,key,0]);[cov.__setitem__(x,1) for x in range(j,k)];s.nm[key]=b' '.join(T[x][1] for x in range(i+1,k)).removesuffix(b"'s");j=k
                    if not T[k-1][4]:break
                    men[-1][3]=3
                if men and men[-1][3]==3 and men[-1][1]==j:men[-1][3]=0
                cov[i]=cov[i] or 3;i=max(j,i+1);continue
            if h in (_H2['i'],_H2['me'],_H2['myself'],_H2['mine']):
                meta=any(wd(x) and hs[x] in G2X for x in (i+1,i+2) if x<n and not (x==i+2 and not _fb(hs[i+1],6)))
                objimp=h==_H2['me'] and i>0 and hs[i-1] in (_H2['remind'],_H2['tell'],_H2['let'],_H2['ask'],_H2['give'],_H2['show'])
                if not meta and not objimp:men.append([i,i+1,SELFK,1]);cov[i]=1
                elif meta:mself=True
                i+=1;continue
            if h in (_H2['he'],_H2['she'],_H2['they'],_H2['him'],_H2['them']) or (h==_H2['her'] and not (wd(i+1) and not G2F.get(hs[i+1],0) and not ocue)) or (h==_H2['it'] and (ocue or (wd(i+1) and (hs[i+1] in G2CF or _fb(hs[i+1],5))))):
                rp=i>=2 and hs[i-1] in (_H2['says'],_H2['said'],_H2['thinks'],_H2['reckons'],_H2['told']) and len(men)>=2 and men[-1][3]==0 and men[-2][3]==3 and men[-2][1]==men[-1][0]
                e=men[-2][2] if rp else resolve(i,2 if h in (_H2['him'],_H2['them'],_H2['her']) or (h==_H2['it'] and ocue) else 1)
                if e:men.append([i,i+1,e,4 if h==_H2['it'] else 2]);cov[i]=1
                i+=1;continue
            i+=1
        cues=[];i=0
        while i<n:
            if not wd(i) or cov[i] in (1,3):i+=1;continue
            hit=None;ix=[i]
            while len(ix)<6 and ix[-1]+1<n and wd(ix[-1]+1) and cov[ix[-1]+1] in (0,2):
                q=ix[-1]+1
                while q<n and wd(q) and hs[q] in _ADV2 and q-ix[-1]<3:q+=1
                if q<n and wd(q) and cov[q] in (0,2):ix.append(q)
                else:break
            for L in range(len(ix),0,-1):
                if not all(cov[x]==2 for x in ix[:L]):
                    rs=[x for x in G2C.get(_pk([hs[x] for x in ix[:L]]),()) if nq or not x[1]&16]
                    if rs and not (L==1 and i>0 and _fb(hs[i-1],1) and hs[i-1]!=_H2['that']):hit=(i,ix[L-1]+1,rs);break
            if hit:cues.append(hit);i=hit[1]
            else:i+=1
        free=[]
        for i in range(n):
            if wd(i) and not cov[i] and (i==0 or not wd(i-1) or _fb(hs[i-1],1) or _fb(hs[i-1],9) or _fb(hs[i-1],7) or hs[i-1] in (_H2['same'],_H2['other'],_H2['best']) or hs[i-1] in G2M):
                r=rnmatch(i)
                if r and r[1]:free.append((i,r[0],r[1]))
        alg=any(h in (_H2['allergy'],_H2['allergic'],_H2['allergies'],_H2['epipen'],_H2['hives']) for h in hs)
        def content(k):return wd(k) and not G2F.get(hs[k],0) and hs[k] not in G2S and not neg[k] and not cov[k]
        cst={c[0] for c in cues if not all(x[1]&2 for x in c[2])}
        def vspan(j,rel,cut=None):
            k=j
            while k<n and wd(k) and not cov[k]:
                if cut is not None and k in cst:return -1
                r=rnmatch(k)
                if r and r[1] and k==j and not cap[k] and r[0]<n and wd(r[0]) and (capx[r[0]] or G2T.get(hs[r[0]],(0,0))[0]==r[1]):k=r[0];continue
                if hs[k]==_H2['the'] and k+1<n and wd(k+1) and not cov[k+1]:
                    r=rnmatch(k+1)
                    if r and r[1]:k=r[0];continue
                if not capx[k] and ((hs[k] in G2K and not (hs[k]==_H2['the'] and rel in (2,7,8,11) and k+1<n and content(k+1) and not cap[k+1])) or hs[k] in G2S or hs[k] in G2M or (hs[k] in (_H2['on'],_H2['in'],_H2['at'],_H2['with'],_H2['as'],_H2['to'],_H2['of'],_H2['from'],_H2['about'],_H2['for'],_H2['by'],_H2['into']) and k==j)):k+=1;continue
                break
            if k>=n or not wd(k) or cov[k]:return None
            k0=k
            if cap[k] and G2F.get(hs[k],0)&0b1100111110:return None
            if capx[k] or num[k]:
                while k<n and wd(k) and not cov[k] and k-k0<4 and (capx[k] or num[k] or (hs[k] in G2J and k+1<n and wd(k+1) and cap[k+1])):k+=1
                while k<n and k-k0<4 and content(k) and not cap[k] and hs[k] in G2T and G2T[hs[k]][0] in (11,5,7,9):k+=1
            else:
                for L in (6,5,4,3,2):
                    if k0+L<=n and all(wd(x) for x in range(k0,k0+L)) and _pk(hs[k0:k0+L]) in G2T:return (k0,k0+L)
                if hs[k]==_H2['the']:k+=1
                while k<n and k-k0<4:
                    if content(k) or (wd(k) and cap[k] and not cov[k]):k+=1;continue
                    if wd(k) and hs[k] in (_H2['in'],_H2['on'],_H2['at'],_H2['from'],_H2['with']) and any(T[x][1].endswith(b'ing') for x in range(k0,k)) and content(k+1):k+=2;continue
                    if wd(k) and hs[k] in (_H2['and'],_H2['of']) and content(k+1) and _pk(hs[k0:k+2]) in G2T:k+=2;continue
                    break
            return (k0,k) if k>k0 and not (k-k0==1 and hs[k0]==_H2['the']) else None
        def vback(i):
            k=i-1;g=0
            while k>=0 and g<7 and not (content(k) or (wd(k) and capx[k] and not cov[k])):k-=1;g+=1
            if k<0 or g>=7 or used[k]:return None
            k1=k+1
            if cap[k] and G2F.get(hs[k],0)&0b1100111110:return None
            if capx[k] or num[k]:
                while k>=0 and wd(k) and not cov[k] and k1-k<=4 and (capx[k] or num[k] or hs[k] in G2J) and not (cap[k] and G2F.get(hs[k],0)&0b1100111110):k-=1
            else:
                while k>=0 and k1-k<=4 and (content(k) or (wd(k) and cap[k] and not cov[k])):k-=1
            k0=k+1
            while k0<k1 and hs[k0] in G2J and not cap[k0]:k0+=1
            return (k0,k1) if k1>k0 else None
        def vtype(a,b):
            if num[a]:return 14
            if hs[a] in _MON2:return 15
            if cap[a]:
                for L in range(b-a,0,-1):
                    t=G2T.get(_pk(hs[a:a+L]))
                    if t and t[1]&1:return t[0]
                if any(G2T.get(hs[x],(0,0))[1]==3 for x in range(a,b)):return 13
            t=G2T.get(_pk(hs[a:b])) or G2T.get(hs[b-1])
            if t and not (t[1]&1 and not cap[a]):return t[0]
            if T[b-1][1].endswith(b'ing') and not cap[a] and len(T[b-1][1])>=5:return 7
            t=G2T.get(hs[a])
            return t[0] if t and not (t[1]&1 and not cap[a]) else 0
        def txt(a,b):return b' '.join(T[x][1] for x in range(a,b))[:24]
        def near(i,j):
            c0=cl[i];ms=[m for m in men if not (m[0]<=i<m[1]) and m[3]!=3]
            L=[m for m in ms if m[1]<=i and cl[m[0]]==c0];R=[m for m in ms if m[0]>=j and cl[m[0]]==c0];L2=[m for m in ms if m[1]<=i]
            return L[-1][2] if L else R[0][2] if R else L2[-1][2] if L2 else 0
        def after_men(j):
            q=j
            if q<n and cov[q]==3:q+=1
            m=next((m for m in men if m[0]==q and m[3]!=3),None)
            if m is None and any(x[0]==q and x[2]==9 for x in rnh):
                x=next(x for x in rnh if x[0]==q and x[2]==9);return (_pk([hs[z] for z in range(x[0],x[1]) if hs[z] not in G2M]),x[1])
            if m is None:
                m=next((m for m in men if m[0]==q),None)
                while m is not None and m[3]==3:
                    m2=next((x for x in men if x[0]==m[1]),None)
                    if m2 is None:break
                    m=m2
            return (m[2],m[1]) if m else (0,q)
        used=[0]*n;out=[]
        def put(e,r,a,b,vt):
            if not e or r<2 or r>15:return False
            v=txt(a,b)
            if r in (4,6,13) and not capx[a]:return False
            if r==14 and not num[a]:return False
            if r==15 and not (hs[a] in _MON2 or cap[a]):return False
            if r==9 and not (cap[a] or vt==9):return False
            for x in range(a,b):used[x]=1
            out.append((e,r,v,vt));return True
        same=[i for i in range(n) if wd(i) and (hs[i] in (_H2['same'],_H2['share'],_H2['shares'],_H2['shared'],_H2['sharing']) or (hs[i]==_H2['as'] and i>0 and hs[i-1]==_H2['old']))]
        scl={cl[i] for i in same}
        if not nq:
            for i in range(n-3):
                if wd(i) and hs[i]==_H2['it'] and _fb(hs[i+1],5):
                    sp=vspan(i+2,0)
                    if sp and sp[1]+2<n and not wd(sp[1]) and neg[sp[1]+1]:
                        sp2=vspan(sp[1]+2,0)
                        if sp2:
                            er=mem.findval(txt(*sp2))
                            if er:put(er[0],er[1],sp[0],sp[1],vtype(*sp))
        if not nq and s.pend and n>=3 and hs[0] in (_H2['it'],_H2['that'],_H2['he'],_H2['she'],_H2['they']) and _fb(hs[1],5):
            sp=vspan(2,s.pend[1])
            if sp:put(s.pend[0],s.pend[1],sp[0],sp[1],vtype(*sp))
        s.pend=None
        if not nq:
            for (j0,j1,rel,own,fav,st) in rnh+[(x[0],x[1],x[2],0,False,x[0]) for x in free]:
                if cl[j0] in scl or any(j0<=c[0]<j1 for c in cues):continue
                k=j1;sk=0
                while k<n and wd(k) and not cov[k] and not cap[k] and (_fb(hs[k],5) or _fb(hs[k],6) or hs[k] in G2S or hs[k] in (_H2['in'],_H2['on'],_H2['at'],_H2['falls'],_H2['comes'],_H2['lands'],_H2['around'],_H2['every'],_H2['been'])) and k-j1<5:k+=1;sk+=1
                sp=vspan(k,rel) if sk and (st!=j0 or any(_fb(hs[x],5) or hs[x] in (_H2['has'],_H2['have'],_H2['had'],_H2['falls'],_H2['comes'],_H2['lands']) for x in range(j1,k))) else None
                if sp is None and k<n and not wd(k) and k+1<n and hs[k+1] in (_H2['if'],_H2['for'],_H2['in'],_H2['as'],_H2['just'],_H2['by']):
                    q=k+1
                    while q<n and wd(q) and q-k<7:q+=1
                    if q+2<n and hs[q+1]==_H2['is']:sp=vspan(q+2,rel)
                if sp is None and (k>=n or not wd(k)) and k+2<n and hs[k+1]==_H2['it'] and (_fb(hs[k+2],5) or hs[k+2]==_H2['would']):sp=vspan(k+3,rel)
                q=st-1
                while q>=0 and wd(q) and hs[q] in (_H2['the'],_H2['best'],_H2['a'],_H2['an']):q-=1
                if sp is None and q>=0 and _fb(hs[q],5):sp=vback(q)
                if sp and own==0 and not men and st>=2 and hs[st-1]==_H2['best']:own=SELFK
                if sp is None and rel and st>=2 and hs[st-2] in (_H2['it'],_H2['that']) and _fb(hs[st-1],5) and own:
                    tv=next((x for x in reversed(s.tf) if x[0]==own),None)
                    if tv and tv[1]!=rel:out.append((own,rel,tv[2],tv[3]))
                    continue
                if not sp or any(used[x] for x in range(*sp)):continue
                a,b=sp;vt=vtype(a,b);r2=rel or vt
                if r2==0 and fav:r2=mem.lrel(own or near(j0,j1),(7,10,11)) or 0
                if rel==9 and vt in (10,14):own=_pk([hs[x] for x in range(j0,j1) if hs[x] not in G2M]);r2=vt
                if ifr[a] or hyp[j0] or any(neg[x] and cl[x]==cl[j0] for x in range(n)):continue
                put(own or near(j0,j1),r2,a,b,vt)
            for (i0,i1,rs) in cues:
                if cl[i0] in scl:continue
                ng=any(neg[x] for x in range(i0,i1)) or any(neg[x] and cl[x]==cl[i0] for x in range(0,i0))
                rows=[x for x in rs if not x[1]&16 and (bool(x[1]&(4|128))==ng)] or [x for x in rs if not x[1]&16 and not x[1]&(4|128)]
                if not rows:continue
                rel,f,_=rows[0];e0=0
                if f&64:sp=vback(i0)
                else:
                    e0,j=after_men(i1)
                    j2=j
                    while j2<n and wd(j2) and (_fb(hs[j2],5) or hs[j2] in G2S) and e0:j2+=1
                    sp=vspan(j2 if e0 else j,rel,True)
                    if sp==-1:continue
                    jb=j
                    while jb<n and wd(jb) and (hs[jb] in G2K or hs[jb] in G2S or hs[jb] in G2M):jb+=1
                    if sp is None and (jb>=n or not wd(jb)):
                        q=jb+1
                        if q+1<n and hs[q]==_H2['it'] and (_fb(hs[q+1],5) or hs[q+1]==_H2['would']):sp=vspan(q+2,rel)
                        elif q+1<n and hs[q]==_H2['that'] and _fb(hs[q+1],5):sp=vspan(q+2,rel)
                        elif not f&1 and (j>=n or not wd(j)):sp=vback(i0)
                    if sp is None and rel==8 and (j>=n or not wd(j)):
                        e=e0 or near(i0,i1)
                        tv=next((x for x in reversed(s.tf) if x[0]==e and x[1]==2),None) or (s.lastv and (e,2,s.lastv,0))
                        if e and tv:out.append((e,8,tv[2],tv[3]))
                        continue
                if i1<n and wd(i1) and hs[i1] in (_H2['there'],_H2['here']) and s.lastp and rel==4:
                    e=e0 or near(i0,i1)
                    if e:out.append((e,4,s.lastp,4))
                    used[i1]=1;continue
                if not sp or any(used[x] for x in range(*sp)):continue
                a,b=sp;vt=vtype(a,b)
                if f&2 and not (vt==rel or (rel==14 and num[a]) or (rel==15 and hs[a] in _MON2) or (rel==12 and vt in (12,13)) or (rel==9 and cap[a] and a>0 and hs[a-1] in (_H2['a'],_H2['an']) and hs[a] not in _MON2)):continue
                if f&2 and rel==12 and vt==13:rel=13
                if f&8:
                    fr=[x[2] for x in free if x[2] in (10,11)]
                    rel=fr[0] if fr else 10 if vt==10 else 11 if vt==11 else 7 if vt==7 else rel
                if f&32:rel=8 if alg else rel
                if rel==4 and vt==15:continue
                if rel==15 and vt not in (15,0) and cap[a]:rel=4
                if rel==6 and b<n and wd(b) and hs[b]==_H2['home']:rel=4
                if ng and not f&(4|128):
                    if rel in (7,10,11):rel=2
                    else:continue
                if (ifr[a] and ifr[i0]) or hyp[i0]:continue
                put(e0 or near(i0,i1),rel,a,b,vt)
            for c0 in sorted(set(cl)):
                if c0 in scl or any(cl[x]==c0 and used[x] for x in range(n)):continue
                idx=[x for x in range(n) if cl[x]==c0]
                for m in [m for m in men if cl[m[0]]==c0 and m[3]!=3]:
                    k=m[1]
                    while k<n and wd(k) and _fb(hs[k],6) and k-m[1]<2:k+=1
                    if k<n and wd(k) and _fb(hs[k],5) and not any(neg[x] for x in idx) and not any(hyp[x] or ifr[x] for x in idx):
                        art=k+1<n and hs[k+1] in (_H2['a'],_H2['an'])
                        sp=vspan(k+1,0)
                        if sp:
                            a,b=sp;vt=vtype(a,b)
                            r=vt if vt in (10,14,15,5) else 5 if art and not cap[a] and vt in (0,7) else 6 if vt==0 and cap[a] and b-a<=2 and not art else 0
                            r=r if m[3]!=4 or r in (10,14,15) else 0
                            if r:put(m[2],r,a,b,vt)
                        break
            for p0 in range(n):
                if not (capx[p0] and not cov[p0] and (p0==0 or not wd(p0-1) or hs[p0-1]==_H2['and'])):continue
                k=p0
                while k<n and wd(k) and (capx[k] or (hs[k] in G2J and k+1<n and wd(k+1) and cap[k+1] and k>p0)) and k-p0<4 and not (k>p0 and T[k-1][4]):k+=1
                imp=T[k-1][4] and k<n and wd(k) and hs[k] in (_H2['in'],_H2['a'],_H2['part'],_H2['located'],_H2['over'])
                if not (k<n and wd(k) and (_fb(hs[k],5) or imp)):continue
                q=k if imp else k+1
                while q<n and wd(q) and hs[q] in (_H2['an'],_H2['a'],_H2['city'],_H2['town'],_H2['village'],_H2['located'],_H2['over'],_H2['part'],_H2['situated'],_H2['right'],_H2['just'],_H2['up'],_H2['down'],_H2['out'],_H2['actually'],_H2['also']):q+=1
                if not (q<n and hs[q] in (_H2['in'],_H2['of'])):continue
                q+=1
                while q<n and wd(q) and hs[q] in (_H2['the'],_H2['state'],_H2['country'],_H2['region'],_H2['province'],_H2['prefecture'],_H2['county'],_H2['of']):q+=1
                if q<n and wd(q) and cap[q]:
                    sp=vspan(q,4)
                    if sp:s.isk(mem,txt(p0,k)[:-2] if imp else txt(p0,k),txt(*sp))
            for e,r,v,vt in out:mem.put(e,r,v,vt);s.fire(e,r,v,mem);s.tf=(s.tf+[(e,r,v,vt)])[-16:];s.qe,s.qr=e,r
            lv=[x for x in range(n) if content(x) and not used[x]]
            if lv:
                q=lv[-1]
                while q>0 and content(q-1) and not used[q-1] and lv[-1]-q<3:q-=1
                s.lastv=txt(q,lv[-1]+1)
            for i in range(n-2):
                if bd[i] and wd(i+1):
                    pv=next(((e,r,v) for e,r,v,vt in out if r==4),None)
                    q=i+1
                    if pv and hs[q] in (_H2['a'],_H2['which'],_H2['in']):
                        if hs[q]==_H2['in'] and q+1<n and cap[q+1]:
                            sp=vspan(q+1,4)
                            if sp:s.isk(mem,pv[2],txt(*sp))
                            continue
                        while q<n and wd(q) and hs[q] in (_H2['a'],_H2['city'],_H2['town'],_H2['which'],_H2['is'],_H2['located'],_H2['part']):q+=1
                        if q<n and hs[q] in (_H2['in'],_H2['of']) and q+1<n and cap[q+1]:
                            sp=vspan(q+1,4)
                            if sp:s.isk(mem,pv[2],txt(*sp))
            for c0 in sorted(scl):
                ms=[m for m in men if cl[m[0]]==c0 and m[3]!=3];sm=[x for x in same if cl[x]==c0][0]
                if not ms:continue
                rr=[x[2] for x in rnh+[(f[0],f[1],f[2],0,0) for f in free] if cl[x[0]]==c0 and x[2]]
                cr=[x[2][0][0] for x in cues if cl[x[0]]==c0 and not x[2][0][1]&(16|2)]
                rel=14 if hs[sm]==_H2['as'] else rr[0] if rr else cr[0] if cr else 0
                src=[m for m in ms if m[0]>sm];tg=[m for m in ms if m[0]<sm]
                own=[x[3] for x in rnh if cl[x[0]]==c0 and x[3]]
                if not src and own and tg and own[-1]!=tg[0][2]:src=[(0,0,own[-1],0)]
                if not tg and own and src:tg=[(0,0,own[0],0)]
                if not tg or not src or not rel:continue
                pairs=[(tg[0][2],src[-1][2])]
                if len(tg)>=2 and tg[1][2]!=tg[0][2]:pairs=[(tg[0][2],tg[1][2]),(tg[1][2],tg[0][2])]
                elif hs[sm] in (_H2['share'],_H2['shares'],_H2['shared'],_H2['sharing']) and _H2['with'] in hs:pairs.append((src[-1][2],tg[0][2]))
                for t_,s_ in pairs:
                    if t_==s_:continue
                    s.links=(s.links+[(t_,rel,s_)])[-8:]
                    v=mem.latest(s_,rel) or next((mem.latest(s_,r) for r in (7,11,10) if rel in (7,10,11) and r!=rel and mem.latest(s_,r) is not None),None)
                    if v is not None:mem.put(t_,rel,v[0],v[1]);s.fire(t_,rel,v[0],mem)
        else:
            owners=[m for m in men if m[3]!=3]+[(x[0],x[1],x[3],0) for x in rnh if x[3]]
            owners.sort(key=lambda m:m[0])
            ns=[m for m in owners if m[2]!=SELFK];e=ns[-1][2] if ns else SELFK if owners or mself else 0
            v9=[x for x in rnh if x[2]==9]
            if v9 and (any(x[0]!=9 and not x[1]&2 for c in cues for x in c[2][:1]) or any(x[2]!=9 for x in free)):
                e=_pk([hs[x] for x in range(v9[-1][0],v9[-1][1]) if hs[x] not in G2M]);rnh=[x for x in rnh if x[2]!=9]
            r1=[x[2] for x in rnh if x[3]==e and x[2]]
            ng=any(neg)
            cr=[];ep=max([m[0] for m in owners if m[2]==e],default=-1)
            for (i0,i1,rs) in sorted(cues,key=lambda c:(c[0]<ep,c[0])):
                rows=[x for x in rs if not x[1]&2 and bool(x[1]&(4|128))==ng] or [x for x in rs if not x[1]&2 and not x[1]&(4|128)]
                if rows:
                    r,f,_=rows[0];fr=[x[2] for x in free if x[2] in (10,11)]
                    r=fr[0] if f&8 and fr and r in (7,11) else r
                    r=2 if ng and r in (7,11) and not f&(4|128) else r
                    cr.append(8 if f&32 and (alg or (mem.latest(e,8) is not None and mem.latest(e,2) is None)) else r)
            fr=[x[2] for x in free]
            wh=[h for h in hs if _fb(h,9)]
            r4=4 if _H2['where'] in wh else 15 if _H2['when'] in wh else 14 if (_H2['how'] in wh and _H2['old'] in hs) or _H2['age'] in hs else 6 if _H2['who'] in wh else 0
            if not r4 and not r1 and not cr and not fr and any(h in (_H2['do'],_H2['does'],_H2['did'],_H2['doing']) for h in hs[-2:]) and _H2['what'] in wh:r4=5
            rel=r1[-1] if r1 else cr[0] if cr else fr[0] if fr else r4
            if _H2['where'] in wh and rel in (12,0,13):rel=13 if rel in (12,13) or any(h in (_H2['study'],_H2['studied'],_H2['degree'],_H2['school'],_H2['college'],_H2['university'],_H2['enrolled']) for h in hs) else 4
            if r1 and e:s.pend=(e,r1[-1])
            s.qe,s.qr=e,rel
        lp=[x for x in range(1,n) if capx[x] and not cov[x] and wd(x-1) and hs[x-1] in (_H2['in'],_H2['to'],_H2['at'],_H2['from'])]
        if lp:
            q=lp[-1]
            while q<n and wd(q) and (capx[q] or (hs[q] in G2J and q+1<n and wd(q+1) and cap[q+1])) and q-lp[-1]<4:q+=1
            s.lastp=txt(lp[-1],q)
        ms=[m[2] for m in men if m[2]!=SELFK and m[3]!=3];s.lm=ms
        if ms:s.ante=ms[-1]
    def isk(s,mem,c,v):s.nm[_f4(c)]=c;mem.put(_f4(c),ISINK,v,4)
    def fire(s,e,r,v,mem,d=0):
        for (t_,rel,s_) in list(s.links):
            if s_==e and (rel==r or (rel in (7,10,11) and r in (7,10,11))) and d<2:mem.put(t_,r,v,0);s.fire(t_,r,v,mem,d+1)
def _f4(v):
    h=0
    for c in v[:4]:
        if c!=32:h=(h*83492791+c+1)&0xFFFFFFFF
    return h
class _GM:
    def __init__(s,p):s.p=p
    def _m(s,f):return max((x for x in s.p.facts[-64:] if f(x)),key=lambda x:x[3],default=None)
    def put(s,e,r,v,vs):p=s.p;p.clk+=1;p.facts.append([e,r,bytes(v),p.clk,p.names.get(e,'')]);p.vsm[p.clk]=vs
    def latest(s,e,r):f=s._m(lambda x:x[0]==e and x[1]==r);return (f[2],s.p.vsm.get(f[3],0)) if f else None
    def lrel(s,e,rs):f=s._m(lambda x:x[0]==e and x[1] in rs);return f[1] if f else 0
    def findval(s,v):f=s._m(lambda x:x[2].lower()==bytes(v).lower());return (f[0],f[1]) if f else None
class _P:
    def __init__(s,lt=None,gn=None,g2=None):
        s.g2=(_g2() if gn is None else False) if g2 is None else g2;s.gn=(_gn() if gn is None else gn) or s.g2;s.vsm={};s.ga=_G2() if s.g2 else None;s.gm=_GM(s);s.lt=s.lt_=(_lt() if lt is None else lt) or s.gn;s.sms=s.prn=s.apv=s.cwt=s.vty=s.vwn=0;s.cms=s.vms=s.pcm=s.gbg=False;s.vws=[0]*8;s.vwt=[0]*8
        s.flg=s.pre=s.subj=s.meta=s.ante=s.own=s.pbd=s.inv=s.poss=s.entc=s.relc=s.gapc=s.qent=s.qrel=s.clk=s.nps=s.we=s.ph=s.pph=s.sw=s.cw=s.hw=0;s.lav=-9;s.wq=s.tcm=False;s.vb=bytearray();s.facts=[];s.names={SELFK:'I'};s.kin={};s.ments=set();s.t=b''
    def b(s,m):return s.flg&m!=0
    def on(s,*m):
        for x in m:s.flg|=x
    def off(s,*m):
        for x in m:s.flg&=~x
    def reset(s):s.poss=s.entc=s.relc=s.gapc=s.inv=s.meta=s.subj=0;s.vb.clear();s.flg&=BKEEP
    def latest(s,pred):return max((f for f in s.facts if pred(f)),key=lambda f:f[3],default=None)
    def commit(s):
        if s.b(BSAME):
            src,rel=s.entc,s.relc;f=s.latest(lambda f:src and src!=s.sms and f[0]==src and (f[1]==rel or rel==0))
            if f:s.vb[:]=f[2];s.relc=f[1];s.entc=s.sms;s.on(BSOL,BENTK)
            else:s.vb.clear()
        for _ in range(4):
            t=next((t for t in (_TRIML if s.lt else _TRIM) if len(s.vb)>len(t) and s.vb.endswith(t)),None)
            if t:del s.vb[-len(t):]
        if s.gn and s.b(BSOL) and s.vb and s.vwn>1 and s.relc in (14,15) and s.vws[s.vwn-1]<=len(s.vb):
            i=s.vwn
            while i>0 and s.vwt[i-1]==s.relc:i-=1
            if 0<i<s.vwn:del s.vb[:s.vws[i]]
        if s.gn and s.b(BSOL) and s.vb:s.relc=s.vty if s.relc==0 else 4 if s.relc==15 and s.vty not in (14,15) and s.vb[:1].isupper() else s.relc
        rel=s.relc;ent=s.subj if rel==ISINK else s.entc
        if s.b(BSOL) and s.vb and rel==0 and (s.b(BFAV) or s.b(BMOST)):f=s.latest(lambda f:f[0]==ent and f[1]>=4);rel=f[1] if f else rel;s.relc=rel
        if s.b(BSOL) and rel==7 and s.b(BMOST):f=s.latest(lambda f:f[0]==ent and f[1] in (10,11));rel=f[1] if f else rel;s.relc=rel
        if not s.g2 and s.b(BSOL) and s.vb and (2 if s.lt else 3)<=rel<=15 and not (s.gn and s.b(BNEG) and rel not in (2,8,ISINK)) and ent and (rel==ISINK or ent==SELFK or s.b(BENTK)):s.clk+=1;s.facts.append([ent,rel,bytes(s.vb),s.clk,s.names.get(ent,'')])
        s.reset()
    def setrel(s,r,noun,force):
        if s.gn and r==13 and not noun and s.b(BWHE) and s.hw in (H_GO2,H_GOES2,H_WENT2,H_GOING2):return
        r=2 if s.gn and r in (7,11) and s.b(BNEG) else r
        s.relc=r if force or noun or s.relc==0 else s.relc
        if s.gn and s.b(BQDONE):pass
        elif noun:s.qrel=r;s.on(BQLOCK)
        elif (not s.b(BQLOCK) and not (s.lt and s.qrel==2)) or (s.lt and r==2):s.qrel=r
        if not s.b(BSOL) and s.b(BPRE):s.vb.clear()
        s.off(BJOB,BHAV,BPRE)
    def person(s,e,kin):
        if e==0:return
        if not s.b(BFRO) and not s.b(BSOL):s.vb.clear()
        s.entc=e;s.poss=0;s.inv=0;s.own=e if e!=SELFK else SELFK;s.ante=e if e!=SELFK else s.ante;s.qent=s.qent if s.gn and s.b(BQDONE) else e;e!=SELFK and s.ments.add(e)
        s.on(BENTK) if kin or e==SELFK else s.off(BENTK)
    def push(s,sep,w):
        s.vwn=0 if not s.vb else s.vwn
        if s.gbg and s.vwn>0:s.vwt[s.vwn-1]=s.cwt
        if s.vwn<8:s.vws[s.vwn],s.vwt[s.vwn]=min(len(s.vb)+(1 if s.vb and sep is not None else 0),24),s.cwt;s.vwn+=1
        if sep is not None and 0<len(s.vb)<24:s.vb.append(sep)
        s.vb+=w[:16][:max(0,24-len(s.vb))];s.vty,s.vms=s.cwt,len(s.vb)==min(len(w),16) and s.cms
    def addv(s,w,cap,cls):
        if s.lt and not s.vb and s.relc==0 and not cap and 5<=len(w)<=16 and w.endswith(b'ing'):
            if s.gn and s.b(BFAV):s.setrel(7,False,False)
            else:s.on(BFRO);return
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
    def rnoun(s,rid):
        if s.lt and s.b(BSOL) and not s.b(BFRO) and s.lav==s.sw-1 and s.vb[:1].islower() and 32 not in s.vb:s.vb.clear();s.off(BSOL)
        if s.poss:s.person(s.own or SELFK,True)
        if rid in (10,11) and s.b(BFAV) and not s.b(BSOL):
            pe=s.entc or s.own or s.ante;f=s.latest(lambda f:pe and f[0]==pe and f[1]==7)
            if f:f[1]=rid
        s.setrel(rid,True,False);s.poss=0
    def rflush(s):
        pr,c=s.prn&255,s.prn>>8;s.prn=0
        if c&256:s.rnoun(pr)
        elif c&512:s.setrel(pr,False,False);s.poss=0
        else:s.setrel(pr,False,pr==11 and s.relc==7);s.poss=0;s.on(BVOP);s.off(BCALL)
    def name(s,e,kin,w=None):s.names[e]=(w or s.t[s.nps:s.we]).decode('latin-1').strip(' ,');s.kin[e]=kin
    def step(s,w,whw,p4w):
        n=len(w);lw=w[:16].lower();ix=whw&WM;hw=s.hw;G=_gtab() if s.gn else None;bk=((s.pre*GK)&0xFFFFFFFF)^hw;s.gbg=bool(s.gn and bk in G[0])
        gg=(G[0].get(bk) or G[0].get(hw)) if s.gn else None;gg=gg if gg and (not gg[0]&1536 or not s.b(BSOL) or s.b(BFRO)) and (gg[0]!=256 or s.poss or s.b(BFAV) or s.pre in (H_WHAT,H_WHICH,H_SAME)) else None
        cls,rid=((CLS.get(ix,0)&~3840)|gg[0],gg[1]) if gg else (1,0) if s.gn and CLS.get(ix,0)==0 and hw in G[5] else GX.get(ix,(CLS.get(ix,0),RID.get(ix,0))) if s.lt_ else (CLS.get(ix,0),RID.get(ix,0))
        neg=s.b(BNEG) or s.pre in (H_NEVER,H_NOT);cls,rid=((cls,2) if neg else (cls&~1024,0)) if s.gn and rid==102 else (cls,rid)
        fun,ew=ix in FUN,s.hw if s.lt else p4w;cap=n>=2 and 65<=w[0]<=90;num=n>0 and w[:16].isdigit();pre=s.pre;b=s.b
        P=lambda k,p:n==k and lw.startswith(p)
        hd=cls&32 or (n>=3 and not fun and cls&2015==0);r=1
        if s.gn:
            s.cms=hw in G[4]
            if hw in (H_NOT2,H_NEVER):s.on(BNEG)
            s.cwt=14 if num else 15 if cls&64 else 10 if cls&2048 else G[1].get(hw,0)
            if s.prn:
                if s.poss and (cls&32 or hw in G[3]) and not cls&3840 and not cap and hw not in G[2]:s.prn=0
                else:s.rflush()
            if b(BSOL) and not b(BFRO) and s.vb and not b(BSAME) and ((((s.vb[:1].isupper() and not cap and n>=4 and not cls&128 and s.relc in (4,6,13,15)) or s.vb[:1].isdigit()) and not fun and not num) or (cls&1 and not cap) or hw in STOPV):s.on(BFRO)
        if b(BSKMU) and not (P(3,b'you') or P(2,b'me') or P(5,b'ab') or P(2,b'to') or P(4,b'that')):s.off(BSKMU)
        if s.meta==1 and not P(1,b'i') and not P(3,b'you'):s.meta=0
        if s.meta in (2,3) and not cls&16:
            was=s.meta;s.meta=0
            if was==2:s.person(SELFK,True)
        if s.meta==8 and not (P(2,b'me') or P(3,b'him') or P(3,b'her') or P(4,b'them')):s.meta=0
        if s.meta>=9:
            s.meta=0
            if hd and s.pbd!=2 and not (s.gn and cap):s.poss=1;s.own=s.ante or s.own
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
        elif s.gn and hw in (H_ON,H_WITH) and s.relc and not b(BSOL) and not b(BFRO):
            if b(BPRE):s.vb.clear();s.off(BPRE)
            s.on(BVOP)
        elif s.gn and s.vms and b(BSOL) and not b(BFRO) and lw==b'of':s.vb.clear();s.vms=False;s.off(BSOL)
        elif s.gn and s.apv and s.entc==0 and lw==b'in' and not b(BSOL):s.subj=s.entc=s.apv;s.relc=ISINK;s.vb.clear();s.apv=0;s.off(BSOL,BFRO);s.on(BVOP)
        elif s.gn and not b(BSAME) and ((hw==H_AS and pre==H_OLD) or (hw==H_WITH and b(BSHR))):
            if hw==H_AS:s.relc=14
            s.on(BSAME);s.sms=s.sms or s.entc or SELFK;s.vb.clear();s.off(BSOL,BFRO,BPRE)
        elif s.gn and hw in SHARE and s.entc and not b(BSOL):s.on(BSHR);s.sms=s.entc
        elif s.gn and s.inv==2 and s.vb and hw==H_OF and pre==H_PART:
            fw=bh(bytes(c for c in s.vb[:4] if c!=32));s.name(fw,False,bytes(s.vb));s.subj=s.entc=fw;s.relc=3;s.vb.clear();s.off(BSOL,BFRO);s.on(BVOP);s.inv=0
        elif s.gn and s.inv==2 and s.vb and not cap and (hw in G[4] or hw==H_PART):r=1
        elif s.gn and hw==H_SAME and pre==H_THE and not b(BSAME):s.on(BSAME);s.sms=s.entc or SELFK;s.vb.clear();s.off(BSOL,BFRO,BPRE)
        elif s.pbd==2 and b(BSOL) and not b(BFRO):s.push(39,w)
        elif s.pbd==2 and P(1,b't'):s.on(BNEG)
        elif s.pbd==2 and n==1 and (lw in (b'm',b'r') or (lw==b's' and s.poss==0)):
            s.gapc=1;s.inv=2 if s.inv==1 else s.inv
            if b(BPRE) and s.vb:s.on(BSOL,BFRO,BVOP);s.off(BPRE)
        elif s.pbd==2 and lw==b's' and s.poss:s.gapc=1;s.own=s.entc or s.own;s.ph=s.pph if s.gn else s.ph
        elif s.pbd==2 and n==1:r=1
        elif cls&2 and not b(BSOL) and not b(BVOP) and not P(3,b'f') and not P(2,b'in'):r=1
        elif P(2,b'my') and s.relc==9 and pre==H_TO and not b(BSOL):r=1
        elif P(2,b'my') or P(4,b'your') or P(1,b'i'):
            if not P(1,b'i') and s.inv==2 and s.vb:s.freeze()
            if b(BSOL) and not b(BFRO) and s.relc==0 and not b(BSAME):s.on(BFRO)
            elif b(BSOL) and not b(BFRO) and not b(BSAME):s.reset() if s.gn and s.wq and not P(1,b'i') else s.commit()
            if s.gn and not P(1,b'i'):s.off(BKINL)
            if P(1,b'i'):s.person(SELFK,True)
            else:s.poss=1;s.own=SELFK if n==2 else s.ante;s.nps=s.we;s.qent=s.qent or (SELFK if n==2 else 0)
        elif P(2,b'me') and pre in (H_LET,H_TELL,H_ASK):r=1
        elif P(2,b'me'):
            if b(BSOL) and s.relc==0:s.on(BFRO)
            s.person(SELFK,True)
        elif P(2,b'he') or P(3,b'she') or P(4,b'they') or P(2,b'it') or P(3,b'him'):
            e=s.own if P(2,b'it') and s.own else s.ante
            if e:s.person(e,True)
        elif (n==3 and lw[:1]==b'h' and lw[2:3]==b'r') or P(3,b'his') or P(5,b'thei') or (s.gn and P(3,b'its')):
            if s.inv==2 and s.vb:s.freeze()
            s.meta=9 if n==3 and lw[2:3]==b'r' else 10 if n==3 else 11;s.nps=s.we
        elif s.lt and cls&2048 and s.entc and s.gapc==1 and s.relc==0 and not b(BSOL):s.poss=0;s.setrel(10,False,False);s.addv(w,cap,0)
        elif s.lt and s.entc==SELFK and not b(BSOL) and s.hw in (H_FAMILY,H_RELATIVES,H_FOLKS):s.name(H_FAMILY,True,b'family');s.person(H_FAMILY,True)
        elif num:
            if s.gn and b(BPRE):s.vb.clear();s.off(BPRE)
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
        elif cls&64 and cap and s.entc==0 and s.gapc==0 and not b(BVOP) and not b(BART) and s.inv==0:s.vb.clear();s.push(None,w);s.subj=p4w;s.inv=1
        elif cls&64 and (b(BART) or s.gapc==1 or b(BVOP) or (s.entc and s.relc in (0,15,4))):
            if s.relc in (0,15) or (s.relc==4 and not b(BSOL)):s.setrel(15,True,False)
            s.addv(w,cap,0)
        elif b(BSAME) and cls&256 and s.entc==s.sms:s.setrel(rid,True,True);s.poss=0
        elif b(BSTUD) and not b(BFRO) and n>2 and not fun and not cls&2 and not cls&8 and not P(6,b'sc') and not P(7,b'col') and not P(10,b'u'):s.addv(w,cap,0)
        elif cls&256 and n in (6,7,10) and (b(BSTUD) or s.relc in (12,13)) and pre in (H_AT,H_IN,H_FROM,H_TO):r=1
        elif cls&256 and n in (6,7) and s.entc==0 and not b(BSOL) and pre==H_IN:r=1
        elif cls&256 and s.qrel==12 and pre==H_IN and not b(BSTUD):r=1
        elif cls&256 and cap and b(BSOL) and n in (4,6,7,10):s.addv(w,True,0)
        elif s.lt and cls&256 and b(BVOP) and not b(BFRO) and s.relc and s.relc!=rid and not s.poss and not b(BSOL) and not s.wq:s.addv(w,cap,0)
        elif s.gn and cls&1792 and s.poss and s.pbd==0 and not b(BSOL) and not s.prn and pre in POSW:s.prn=rid|(cls&1792)<<8;s.ph=hw
        elif cls&256:s.rnoun(rid)
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
        elif P(8,b'fav') or (s.gn and n==9 and lw==b'favourite'):s.on(BFAV)
        elif s.poss and (P(3,b'bi') or P(4,b'be') or (P(6,b'l') and (not s.lt or lw[1:3] in (b'it',b'ov',b'on',b'at'))) or P(3,b'new')):r=1
        elif s.poss and s.pbd==1 and P(2,b'in'):r=1;s.ph=s.pph
        elif s.poss and hd and not (b(BKINL) and not cls&32 and s.pbd!=1 and not (s.gn and s.pph and hw in G[3])):
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
        elif s.gn and hw in G[2] and s.entc and s.relc==0 and not b(BSOL) and not b(BVOP) and s.poss==0 and s.gapc==0:s.on(BVOP)
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
        elif cap and s.entc==0 and s.poss==0 and not fun:s.vb.clear();s.push(None,w);s.subj=p4w;s.inv=1
        elif b(BPRE) and cap:s.push(32,w)
        s.lt and s.off(BACO)
        return r
    def word(s,w):
        whw,p4w=bh(w),bh(w[:4]);s.hw=bh(w[:16].lower())
        s.pph,s.ph,s.wq,s.sw,s.cw,s.tcm=s.ph,0,(s.hw in WQS[:16] or s.hw in WIM or (s.sw==0 and s.hw in WQS)) if s.cw==0 and not (s.gn and s.pcm and s.hw in RELW) else s.wq,s.sw+1,s.cw+1,s.tcm or (s.lt and s.hw in WCM);s.pcm=False
        for _ in range(2):
            if s.step(w,whw,p4w)==1:break
        s.pre=bh(w[:16].lower())
        if s.g2:s.ga.word(w,len(w),s.pbd)
    def finish_q(s):
        if s.gn and s.b(BQDONE):return
        s.qent=s.qent or (SELFK if s.b(BMETI) else 0);s.qrel=s.qrel or (6 if s.b(BWHO) else 15 if s.b(BWHEN) else 4 if s.b(BWHE) else 5 if s.b(BDO) else 0)
    def punct(s,c):
        if s.prn:s.rflush()
        end,nq,qd0=c in (46,33,63,10),s.lt and (c==63 or s.wq),s.b(BQDONE)
        if s.g2 and c==44:s.ga.comma()
        elif s.g2 and end:s.ga.end(nq,s.gm);s.names.update({k:v.decode('latin-1') for k,v in s.ga.nm.items()});[s.kin.__setitem__(k,True) for k in s.ga.lm];s.ments|=set(s.ga.lm)
        if s.meta==5 and c==44:s.meta=0
        elif s.b(BSKNO) and end:s.off(BSKNO);s.reset()
        elif c==44 and not s.b(BSOL):
            s.poss=0;s.off(BKINL,BVOP,BPRE);s.gapc=0;s.gn and s.off(BNEG)
            if not (s.gn and s.inv==1):s.inv=0;s.vb.clear()
        elif c==63 and not s.b(BSOL):s.finish_q();s.off(BVOP);s.gapc=0;s.inv=0;s.vb.clear()
        elif c==44:
            ap=bh(bytes(x for x in s.vb[:4] if x!=32)) if s.gn and s.relc==4 and s.vb[:1].isupper() else 0
            s.reset() if nq else (s.commit(),s.on(BACO));s.apv=ap
        elif end:
            if c==63:s.finish_q()
            s.reset() if nq else s.commit()
        if s.gn and c==63:s.on(BQDONE)
        if s.lt and (end or c==44):s.sw,s.cw,s.wq=0 if end else s.sw,0,False
        if end:s.apv=0
        s.pcm=c==44
        if s.g2 and end and not qd0:s.qent,s.qrel=s.ga.qe,s.ga.qr
    def turn(s,t):
        s.qent=s.qrel=s.flg=s.pbd=s.ph=s.sw=s.cw=0;s.wq=s.tcm=False;s.reset();s.ments=set();s.t=t=t+b'\n';w=bytearray();f0=len(s.facts);s.g2 and s.ga.newturn()
        for i,c in enumerate(t):
            if c in WCH:w.append(c);continue
            if w:s.we=i;s.word(bytes(w));w.clear();s.pbd=1 if c==45 else 2 if c==39 else 0
            if c in (46,44,33,63,10):s.punct(c)
            elif s.g2 and c in (58,59):s.ga.comma()
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

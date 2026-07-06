"""SkillRegistry — extensible tool layer for AmniAgent. Each skill: (name, fn, gate, desc, schema).
Built-ins: time, calc, mem, web, file_read, file_write, code_edit, shell, scan.
Asimov gating: every call passes through `_gate(args, ctx)` returning rejection reason or None.
calc / web / mem are thin aliases over Adam's existing tiers — file_*/shell/code_edit/scan are I/O primitives.
v6.1.0: file gates use a roots LIST instead of single workdir. `unrestricted=True` adds drive roots so Adam reaches any file."""
import os,re,ast,json,time,string,subprocess,shlex,math
from pathlib import Path
from dataclasses import dataclass,asdict,field
from typing import Callable,Dict,Any,Optional,List
_SHELL_ALLOW={'ls','dir','cat','type','pwd','cd','git','python','python3','py','pip','pip3','where','which','echo','head','tail','wc','find'}
_SHELL_BLOCK_ARGS={'rm','del','rmdir','rd','format','mkfs','dd','shutdown','reboot','kill','taskkill','--force','-rf','-f'}
_SHELL_META=re.compile(r'[;&|<>`$\n\r]')
_SHELL_WINENV=re.compile(r'%\w+%')
_PY_BASES={'python','python3','py'}
_PIP_BASES={'pip','pip3'}
_PY_SAFE_FLAGS={'--version','-v','--help','-h'}
_PIP_SAFE_SUB={'list','show','freeze','check'}
_TEXT_EXT={'.txt','.md','.markdown','.rst','.py','.js','.ts','.tsx','.jsx','.html','.htm','.css','.json','.yaml','.yml','.toml','.ini','.cfg','.csv','.tsv','.log','.sh','.ps1','.bat','.c','.h','.cpp','.hpp','.cs','.go','.rs','.java','.kt','.swift','.rb','.php','.lua','.r','.sql','.tex','.bib','.xml','.svg','.gitignore','.dockerignore','.env','.example'}
def _enumerate_drives():
    return [Path(f'{d}:/') for d in string.ascii_uppercase if Path(f'{d}:/').exists()] if os.name=='nt' else [Path('/')]
@dataclass
class SkillResult:
    ok:bool
    output:Any=None
    error:Optional[str]=None
    skill:str=''
    elapsed_ms:int=0
    def to_dict(self)->Dict[str,Any]:return asdict(self)
@dataclass
class _Skill:
    name:str
    fn:Callable
    gate:Optional[Callable]
    desc:str
    schema:Dict[str,Any]=field(default_factory=dict)
class SkillRegistry:
    def __init__(self,workdir:Optional[str]=None,roots:Optional[List[str]]=None,audit_log:Optional[str]=None,unrestricted:bool=False):
        self._skills:Dict[str,_Skill]={}
        rs:List[Path]=[]
        if roots:rs.extend(Path(r).resolve() for r in roots)
        if workdir:rs.append(Path(workdir).resolve())
        if not rs and not unrestricted:rs.append(Path(os.getcwd()).resolve())
        if unrestricted:rs.extend(_enumerate_drives())
        seen=set();self.roots=[];_=[(self.roots.append(p),seen.add(str(p))) for p in rs if str(p) not in seen]
        self.workdir=self.roots[0] if self.roots else Path(os.getcwd()).resolve()
        self.unrestricted=unrestricted
        self.audit_log=Path(audit_log) if audit_log else None
        if self.audit_log:self.audit_log.parent.mkdir(parents=True,exist_ok=True)
    def register(self,name:str,fn:Callable,gate:Optional[Callable]=None,desc:str='',schema:Optional[Dict]=None):
        self._skills[name]=_Skill(name=name,fn=fn,gate=gate,desc=desc,schema=schema or {})
    def list_skills(self)->List[Dict[str,Any]]:
        return [{'name':s.name,'desc':s.desc,'schema':s.schema} for s in self._skills.values()]
    def has(self,name:str)->bool:return name in self._skills
    def call(self,name:str,args:Dict[str,Any],ctx:Optional[Dict]=None)->SkillResult:
        t0=time.time()
        ctx=ctx or {}
        if name not in self._skills:
            r=SkillResult(ok=False,error=f'unknown skill: {name}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
            self._audit(name,args,r);return r
        s=self._skills[name]
        if s.gate is not None:
            reason=s.gate(args,ctx,self)
            if reason:
                r=SkillResult(ok=False,error=f'gated: {reason}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
                self._audit(name,args,r);return r
        try:out=s.fn(args,ctx,self);r=SkillResult(ok=True,output=out,skill=name,elapsed_ms=int((time.time()-t0)*1000))
        except Exception as e:r=SkillResult(ok=False,error=f'{type(e).__name__}: {e}',skill=name,elapsed_ms=int((time.time()-t0)*1000))
        self._audit(name,args,r);return r
    def _audit(self,name:str,args:Dict,r:SkillResult):
        if not self.audit_log:return
        try:from amni.serve.code_safety import scrub_egress as _se
        except Exception:_se=lambda x:x
        rec={'ts':time.time(),'skill':name,'args':{k:_se(str(v)[:500]) for k,v in args.items()},'ok':r.ok,'error':_se(r.error or ''),'elapsed_ms':r.elapsed_ms}
        try:
            with open(self.audit_log,'a',encoding='utf-8') as f:f.write(json.dumps(rec)+'\n')
        except Exception:pass
    def _abs(self,p):
        pp=Path(p)
        a=pp if pp.is_absolute() else self.workdir/pp
        if not a.exists() and ' ' in str(p):
            ds=str(p).replace(' ','');dsp=Path(ds) if Path(ds).is_absolute() else self.workdir/ds
            if dsp.exists():return dsp
        return a
    def _in_workdir(self,p:str)->bool:return self._in_allowed_roots(p)
    def _in_allowed_roots(self,p:str)->bool:
        try:
            rp=self._abs(p).resolve()
            for r in self.roots:
                rr=Path(r).resolve()
                if rp==rr:return True
                try:
                    if rp.is_relative_to(rr):return True
                except AttributeError:
                    try:
                        if os.path.commonpath([str(rp),str(rr)])==str(rr):return True
                    except Exception:pass
            return False
        except Exception:return False
_SECRET_NAMES={'.env','.netrc','.pgpass','.htpasswd','credentials','.git-credentials','id_rsa','id_dsa','id_ecdsa','id_ed25519','.npmrc','.pypirc','.dockercfg','.boto'}
_SECRET_EXTS={'.pem','.key','.pfx','.p12','.keystore','.jks','.ppk'}
def _is_secret_file(path)->bool:
    pp=Path(path);n=pp.name.lower()
    return (n in _SECRET_NAMES) or n.startswith('.env') or (pp.suffix.lower() in _SECRET_EXTS)
def _secret_blocked(path)->bool:
    return _is_secret_file(path) and os.environ.get('AMNI_ALLOW_SECRET_FILES','0')!='1'
def _law_protected(path)->bool:
    if os.environ.get('AMNI_ALLOW_LAW_EDIT','0')=='1':return False
    p=Path(path);n=p.name.lower();parent=p.parent.name.lower()
    return (n=='asimov.py' and parent in ('a1','inference')) or (n=='integrity.py' and parent=='learning')
_SECURITY_CORE={'code_safety.py','federated.py','pii_egress.py'}
def _security_protected(path)->bool:
    if os.environ.get('AMNI_ALLOW_SECURITY_EDIT','0')=='1':return False
    p=Path(path)
    return p.name.lower() in _SECURITY_CORE and p.parent.name.lower()=='serve'
def _write_protected(path):
    if _law_protected(path):return 'Asimov/integrity law file'
    if _security_protected(path):return 'core security-enforcement module'
    return None
def _gate_path(args,ctx,reg:'SkillRegistry')->Optional[str]:
    p=args.get('path')
    if not p:return 'missing path arg'
    if not reg._in_allowed_roots(p):return f'path outside allowed roots ({len(reg.roots)} configured): {p}'
    if _secret_blocked(p):return f'refused: {Path(p).name!r} looks like a secret/credential file — set AMNI_ALLOW_SECRET_FILES=1 to override'
    return None
def _gate_shell(args,ctx,reg:'SkillRegistry')->Optional[str]:
    cmd=args.get('cmd','')
    if not cmd:return 'missing cmd arg'
    if _SHELL_META.search(cmd) or _SHELL_WINENV.search(cmd):return 'shell metacharacters not allowed (no chaining ; | & / redirect < > / substitution $() ` / env-expansion $VAR %VAR%)'
    try:parts=shlex.split(cmd,posix=False)
    except Exception:return 'unparseable cmd'
    if not parts:return 'empty cmd'
    base=parts[0].lower().split('\\')[-1].split('/')[-1].removesuffix('.exe')
    _extra={x.strip().lower() for x in os.environ.get('AMNI_SHELL_EXTRA_ALLOW','').split(',') if x.strip()}
    if base not in _SHELL_ALLOW and base not in _extra:return f'command not in allowlist: {base}'
    if os.environ.get('AMNI_SHELL_ALLOW_EXEC','0')!='1':
        if base in _PY_BASES and not (parts[1:] and {a.lower() for a in parts[1:]}<=_PY_SAFE_FLAGS):return 'python via shell is limited to --version/--help — use the sandboxed run_python skill to execute code (set AMNI_SHELL_ALLOW_EXEC=1 to override)'
        if base in _PIP_BASES:
            _sub=next((a.lower() for a in parts[1:] if not a.startswith('-')),'')
            if _sub not in _PIP_SAFE_SUB and not ({a.lower() for a in parts[1:]}<=_PY_SAFE_FLAGS):return 'pip via shell is limited to read-only subcommands (list/show/freeze/check); install/uninstall/download run arbitrary setup code (set AMNI_SHELL_ALLOW_EXEC=1 to override)'
    for p in parts[1:]:
        if p.lower() in _SHELL_BLOCK_ARGS:return f'blocked arg: {p}'
        if '..' in p or p.startswith('/') or (len(p)>1 and p[1]==':'):
            if not reg._in_allowed_roots(p):return f'arg path outside allowed roots: {p}'
    return None
def _gate_code_edit(args,ctx,reg:'SkillRegistry')->Optional[str]:
    g=_gate_path(args,ctx,reg)
    if g:return g
    _wp=_write_protected(args.get('path',''))
    if _wp:return f'refused: {_wp} is write-protected (set AMNI_ALLOW_LAW_EDIT/AMNI_ALLOW_SECURITY_EDIT=1 to override)'
    if not args.get('find') or args.get('replace') is None:return 'missing find/replace'
    return None
def _skill_time(args,ctx,reg):return {'iso':time.strftime('%Y-%m-%dT%H:%M:%S'),'epoch':int(time.time())}
_WORD_OPS={'divided by':'/','to the power of':'**','to the power':'**','plus':'+','minus':'-','times':'*','multiplied by':'*','over':'/'}
def _norm_math(e:str)->str:
    e=re.sub(r'(?i)\b(\d+\.?\d*)\s*(?:percent|%)\s+of\s+(\d+\.?\d*)','(\\1/100*\\2)',e)
    e=re.sub(r'(\d+\.?\d*)\s*%','(\\1/100)',e)
    for w,sym in _WORD_OPS.items():e=re.sub(r'\b'+re.escape(w)+r'\b',sym,e,flags=re.IGNORECASE)
    e=re.sub(r'(\d)\s*[x×]\s*(?=\d)','\\1*',e)
    return e
def _try_python_eval(expr:str)->Optional[float]:
    e=_norm_math(expr.strip().lower())
    if '=' in e or re.search(r'[a-z]',e):return None
    e=re.sub(r'[^0-9+\-*/().\s%]','',e)
    if not e.strip() or not re.search(r'[\d]',e):return None
    if not re.search(r'[+\-*/%]',e):return None
    try:v=eval(e,{'__builtins__':{}},{});return float(v) if isinstance(v,(int,float)) else None
    except Exception:return None
_CALC_VERBS={'solve':'solve','expand':'expand','factor':'factor','simplify':'simplify','derivative':'diff','differentiate':'diff','diff':'diff','integrate':'integrate','integral':'integrate'}
def _sympy_calc(expr:str)->Optional[str]:
    try:import sympy as sp
    except Exception:return None
    el=expr.lower();e=_norm_math(expr.strip())
    e=re.sub(r'(?i)\b(what\s+is|whats|compute|calculate|evaluate|the\s+value\s+of|find)\b','',e).strip(' ?.')
    verb=next((v for k,v in _CALC_VERBS.items() if re.search(r'\b'+k+r'\b',el)),None)
    try:
        if verb:
            body=re.sub(r'(?i)\b('+'|'.join(_CALC_VERBS)+r')\b','',e).strip(' ?=.')
            if verb=='solve':
                lhs,_,rhs=body.partition('=');eq=sp.Eq(sp.sympify(lhs),sp.sympify(rhs)) if rhs.strip() else sp.sympify(body);sol=sp.solve(eq);return str(sol)
            if verb=='diff':
                parts=[p.strip() for p in body.split(',')];f=sp.sympify(parts[0]);v=sp.Symbol(parts[1]) if len(parts)>1 else (sorted(f.free_symbols,key=str)[0] if f.free_symbols else sp.Symbol('x'));return str(sp.diff(f,v))
            if verb=='integrate':
                parts=[p.strip() for p in body.split(',')];f=sp.sympify(parts[0]);v=sp.Symbol(parts[1]) if len(parts)>1 else (sorted(f.free_symbols,key=str)[0] if f.free_symbols else sp.Symbol('x'));return str(sp.integrate(f,v))
            return str(getattr(sp,verb)(sp.sympify(body)))
        if '=' in e and re.search(r'[a-zA-Z]',e):
            lhs,_,rhs=e.partition('=');return str(sp.solve(sp.Eq(sp.sympify(lhs),sp.sympify(rhs))))
        val=sp.sympify(e)
        if val.free_symbols:return str(val)
        f=float(sp.N(val));return str(int(f)) if f==int(f) else str(round(f,8))
    except Exception:return None
def _skill_calc(args,ctx,reg):
    adam=ctx.get('adam')
    expr=args.get('expr') or args.get('query','')
    if not expr:return {'error':'missing expr'}
    fast=_try_python_eval(expr)
    if fast is not None:
        out=int(fast) if fast==int(fast) else round(fast,8)
        return {'value':str(out),'tier':'fast_eval','tokens':0}
    sym=_sympy_calc(expr)
    if sym is not None:return {'value':sym,'tier':'sympy','tokens':0}
    if adam is None:return {'error':'symbolic expr requires Adam: '+expr}
    r=adam.ask(f'Compute: {expr}',writeback=False)
    return {'value':r.get('answer'),'tier':r.get('tier'),'tokens':r.get('tokens')}
_UNITS={'length':{'m':1,'km':1000,'cm':.01,'mm':.001,'um':1e-6,'nm':1e-9,'mi':1609.344,'mile':1609.344,'yd':.9144,'ft':.3048,'in':.0254,'inch':.0254,'nmi':1852,'ly':9.4607e15,'au':1.496e11},'mass':{'kg':1,'g':.001,'mg':1e-6,'ug':1e-9,'lb':.45359237,'oz':.028349523,'t':1000,'tonne':1000,'st':6.35029,'ton':1000},'time':{'s':1,'ms':.001,'us':1e-6,'min':60,'h':3600,'hr':3600,'day':86400,'week':604800,'year':31557600,'yr':31557600},'data':{'b':1,'byte':1,'kb':1e3,'mb':1e6,'gb':1e9,'tb':1e12,'pb':1e15,'kib':1024,'mib':1048576,'gib':1073741824,'tib':1.0995e12,'bit':.125},'energy':{'j':1,'kj':1000,'mj':1e6,'cal':4.184,'kcal':4184,'wh':3600,'kwh':3.6e6,'ev':1.602176634e-19,'btu':1055.06},'power':{'w':1,'kw':1000,'mw':1e6,'hp':745.7},'pressure':{'pa':1,'kpa':1000,'mpa':1e6,'bar':1e5,'mbar':100,'atm':101325,'psi':6894.757,'mmhg':133.322,'torr':133.322},'volume':{'l':1,'ml':.001,'m3':1000,'cm3':.001,'gal':3.785412,'qt':.946353,'pt':.473176,'cup':.236588,'floz':.0295735,'tbsp':.0147868,'tsp':.00492892},'speed':{'mps':1,'kmh':.2777778,'mph':.44704,'kn':.514444,'knot':.514444,'fps':.3048},'area':{'m2':1,'km2':1e6,'cm2':1e-4,'mm2':1e-6,'ft2':.092903,'in2':.00064516,'acre':4046.856,'ha':10000,'mi2':2589988},'angle':{'rad':1,'deg':math.pi/180,'grad':math.pi/200,'arcmin':math.pi/10800,'arcsec':math.pi/648000},'frequency':{'hz':1,'khz':1e3,'mhz':1e6,'ghz':1e9},'force':{'n':1,'kn':1000,'lbf':4.448222,'dyn':1e-5},'data_rate':{'bps':1,'kbps':1e3,'mbps':1e6,'gbps':1e9}}
_UAL={'meter':'m','meters':'m','metre':'m','kilometers':'km','kilometres':'km','kilometer':'km','centimeter':'cm','centimeters':'cm','millimeter':'mm','millimeters':'mm','miles':'mi','feet':'ft','foot':'ft','inches':'in','yard':'yd','yards':'yd','kilogram':'kg','kilograms':'kg','kilo':'kg','kilos':'kg','gram':'g','grams':'g','grammes':'g','pound':'lb','pounds':'lb','lbs':'lb','ounce':'oz','ounces':'oz','seconds':'s','sec':'s','secs':'s','second':'s','minute':'min','minutes':'min','mins':'min','hours':'h','hour':'h','days':'day','weeks':'week','years':'year','celsius':'c','centigrade':'c','fahrenheit':'f','kelvin':'k','degc':'c','degf':'f','bytes':'b','kilobytes':'kb','megabytes':'mb','gigabytes':'gb','terabytes':'tb','bits':'bit','liter':'l','litre':'l','liters':'l','litres':'l','milliliter':'ml','milliliters':'ml','gallon':'gal','gallons':'gal','quart':'qt','pint':'pt','calorie':'cal','calories':'cal','kilocalorie':'kcal','joule':'j','joules':'j','watt':'w','watts':'w','kilowatt':'kw','pascal':'pa','pascals':'pa','atmosphere':'atm','atmospheres':'atm','degree':'deg','degrees':'deg','radian':'rad','radians':'rad','hertz':'hz','newton':'n','newtons':'n','kph':'kmh','mihr':'mph'}
def _temp_to(v,fr,to):
    fr,to=fr.lower(),to.lower();k={'c':v+273.15,'f':(v-32)*5/9+273.15,'k':v}.get(fr)
    if k is None:return None
    return {'c':k-273.15,'f':(k-273.15)*9/5+32,'k':k}.get(to)
def _skill_units(args,ctx,reg):
    q=(args.get('query') or '').strip();v=args.get('value');fr=args.get('from');to=args.get('to')
    if q and v is None:
        m=re.match(r'(?i)^\s*(-?\d+\.?\d*)\s*([a-z°/0-9]+)\s*(?:to|in|->|as)\s*([a-z°/0-9]+)\s*$',q.replace('°',' deg').replace('  ',' '))
        if m:v,fr,to=float(m.group(1)),m.group(2),m.group(3)
    if v is None or not fr or not to:return {'error':'need value, from, to (or query "10 km to miles")'}
    norm=lambda u:_UAL.get(str(u).strip().lower(),str(u).strip().lower())
    fr,to=norm(fr),norm(to)
    if fr in('c','f','k') or to in('c','f','k'):
        r=_temp_to(float(v),fr,to);return {'value':round(r,6),'from':fr,'to':to,'unit':to} if r is not None else {'error':'temperature units are c/f/k'}
    for cat,tbl in _UNITS.items():
        if fr in tbl and to in tbl:
            r=float(v)*tbl[fr]/tbl[to];return {'value':round(r,10),'from':fr,'to':to,'category':cat}
    return {'error':f'unknown or mismatched units: {fr} -> {to}'}
def _skill_datetime(args,ctx,reg):
    from datetime import datetime,timedelta
    try:from dateutil import parser as dp,relativedelta as rd
    except Exception:dp=None
    act=(args.get('action') or '').lower();q=args.get('query','')
    P=lambda s:dp.parse(str(s)) if dp else datetime.fromisoformat(str(s))
    try:
        if not act and q:
            mb=re.search(r'(?i)between\s+(.+?)\s+and\s+(.+)',q);ma=re.search(r'(?i)(\d+)\s*(day|week|month|year|hour|minute)s?\s*(?:from|after|before|ago)\s*(.+)?',q)
            if mb:act,args['a'],args['b']='diff',mb.group(1),mb.group(2)
            elif ma:act='add';args['date']=(ma.group(3) or 'today').strip();sign=-1 if re.search(r'(?i)before|ago',q) else 1;args[ma.group(2)+'s' if ma.group(2) in('day','week','hour','minute') else ma.group(2)]=sign*int(ma.group(1))
            else:act='parse';args['date']=q
        if act=='diff':
            a,b=P(args['a'] if str(args.get('a','')).lower()!='today' else datetime.now().isoformat()),P(args['b'] if str(args.get('b','')).lower()!='today' else datetime.now().isoformat());d=abs((b-a).total_seconds())
            return {'days':round(d/86400,4),'seconds':int(d),'human':f'{int(d//86400)} days, {int(d%86400//3600)} hours'}
        if act=='add':
            base=datetime.now() if str(args.get('date','today')).lower() in('today','now','') else P(args['date'])
            base=base+timedelta(days=args.get('days',0) or 0,weeks=args.get('weeks',0) or 0,hours=args.get('hours',0) or 0,minutes=args.get('minutes',0) or 0)
            if dp and (args.get('months') or args.get('years')):base=base+rd.relativedelta(months=args.get('months',0) or 0,years=args.get('years',0) or 0)
            return {'result':base.isoformat(timespec='seconds'),'weekday':base.strftime('%A'),'date':base.strftime('%Y-%m-%d')}
        d=P(args['date']) if args.get('date') and str(args['date']).lower() not in('today','now') else datetime.now()
        return {'iso':d.isoformat(timespec='seconds'),'weekday':d.strftime('%A'),'date':d.strftime('%Y-%m-%d'),'time':d.strftime('%H:%M:%S'),'day_of_year':int(d.strftime('%j'))}
    except Exception as e:return {'error':f'datetime: {e}'}
_CONST={'c':(299792458,'m/s','speed of light'),'g':(9.80665,'m/s^2','standard gravity'),'G':(6.67430e-11,'m^3/kg/s^2','gravitational constant'),'h':(6.62607015e-34,'J*s','Planck'),'hbar':(1.054571817e-34,'J*s','reduced Planck'),'k_B':(1.380649e-23,'J/K','Boltzmann'),'N_A':(6.02214076e23,'1/mol','Avogadro'),'R':(8.314462618,'J/mol/K','gas constant'),'F':(96485.33212,'C/mol','Faraday'),'e':(1.602176634e-19,'C','elementary charge'),'epsilon_0':(8.8541878128e-12,'F/m','vacuum permittivity'),'m_e':(9.1093837015e-31,'kg','electron mass'),'m_p':(1.67262192369e-27,'kg','proton mass'),'sigma':(5.670374419e-8,'W/m^2/K^4','Stefan-Boltzmann'),'atm':(101325,'Pa','standard atmosphere')}
_EQ={'ohms_law':('V = I * R','voltage = current * resistance'),'power_electrical':('P = V * I','also P = I^2 * R = V^2 / R'),'ideal_gas':('P * V = n * R * T','ideal gas law'),'nernst':('E = E0 - (R * T / (n * F)) * ln(Q)','electrode potential; F=Faraday, n=electrons, Q=reaction quotient'),'faraday_electrolysis':('m = (Q * M) / (n * F)','mass deposited; Q=charge, M=molar mass, n=electrons'),'newton_second':('F = m * a','force'),'kinetic_energy':('KE = 0.5 * m * v^2',''),'grav_pe':('PE = m * g * h',''),'kinematics_v':('v = u + a * t',''),'kinematics_s':('s = u*t + 0.5*a*t^2',''),'coulomb':('F = k * q1 * q2 / r^2','k = 8.9875e9'),'density':('rho = m / V',''),'molarity':('M = n / V','mol per liter'),'ph':('pH = -log10([H+])',''),'arrhenius':('k = A * exp(-Ea / (R * T))','reaction rate'),'compound_interest':('A = P * (1 + r/n)^(n*t)',''),'ohm_power':('P = I^2 * R',''),'hookes_law':('F = -k * x','spring'),'half_life':('N = N0 * (1/2)^(t/T)','')}
_ELEM={'h':('Hydrogen',1,1.008),'he':('Helium',2,4.0026),'li':('Lithium',3,6.94),'c':('Carbon',6,12.011),'n':('Nitrogen',7,14.007),'o':('Oxygen',8,15.999),'f':('Fluorine',9,18.998),'na':('Sodium',11,22.990),'mg':('Magnesium',12,24.305),'al':('Aluminium',13,26.982),'si':('Silicon',14,28.085),'p':('Phosphorus',15,30.974),'s':('Sulfur',16,32.06),'cl':('Chlorine',17,35.45),'k':('Potassium',19,39.098),'ca':('Calcium',20,40.078),'fe':('Iron',26,55.845),'cu':('Copper',29,63.546),'zn':('Zinc',30,65.38),'ag':('Silver',47,107.868),'au':('Gold',79,196.967),'pb':('Lead',82,207.2)}
def _skill_formula(args,ctx,reg):
    q=str(args.get('query') or args.get('name') or '').strip();ql=q.lower()
    if q in _CONST:v,u,d=_CONST[q];return {'constant':q,'value':v,'unit':u,'description':d}
    el=_ELEM.get(ql) or next((v for kk,v in _ELEM.items() if v[0].lower()==ql),None)
    if el:return {'element':el[0],'atomic_number':el[1],'atomic_mass':el[2]}
    key=ql.replace(' ','_').replace("'",'')
    if key in _EQ:f,d=_EQ[key];return {'equation':key,'formula':f,'note':d}
    if len(ql)>=3:
        for k,(v,u,d) in _CONST.items():
            if ql in d.lower():return {'constant':k,'value':v,'unit':u,'description':d}
        hits={k:v[0] for k,v in _EQ.items() if ql in k or ql in v[1].lower()}
        if hits:return {'matches':hits}
    return {'constants':list(_CONST),'equations':list(_EQ),'hint':'query a constant (c,R,F,N_A,k_B...), equation (nernst,ideal_gas,ohms_law,arrhenius...), or element (Fe,Cu,O...)'}
def _skill_codec(args,ctx,reg):
    import base64,hashlib,uuid,urllib.parse,json as _j
    op=(args.get('op') or '').lower();t=args.get('text','');
    try:
        if op in('base64_encode','b64'):return {'result':base64.b64encode(str(t).encode()).decode()}
        if op in('base64_decode','b64d'):return {'result':base64.b64decode(str(t)).decode('utf-8','replace')}
        if op=='hex_encode':return {'result':str(t).encode().hex()}
        if op=='hex_decode':return {'result':bytes.fromhex(str(t)).decode('utf-8','replace')}
        if op=='url_encode':return {'result':urllib.parse.quote(str(t))}
        if op=='url_decode':return {'result':urllib.parse.unquote(str(t))}
        if op=='hash':a=(args.get('algo') or 'sha256').lower();return {'algo':a,'result':hashlib.new(a,str(t).encode()).hexdigest()}
        if op=='uuid':return {'result':str(uuid.uuid4())}
        if op in('json_pretty','json'):return {'result':_j.dumps(_j.loads(t) if isinstance(t,str) else t,indent=2,sort_keys=bool(args.get('sort')))}
        if op=='json_minify':return {'result':_j.dumps(_j.loads(t),separators=(',',':'))}
        if op=='json_validate':
            try:_j.loads(t);return {'valid':True}
            except Exception as e:return {'valid':False,'error':str(e)}
        if op=='regex':
            pat=args.get('pattern','');return {'matches':re.findall(pat,str(t))}
        if op=='base':
            frm=int(args.get('from_base',10));to=int(args.get('to_base',16));n=int(str(t).strip(),frm)
            return {'result':{2:bin,8:oct,16:hex}.get(to,lambda x:str(x))(n) if to in(2,8,16) else _to_base(n,to),'decimal':n}
        if op=='bitwise':
            a,b=int(args.get('a',0)),int(args.get('b',0));o=args.get('bitop','and')
            return {'result':{'and':a&b,'or':a|b,'xor':a^b,'shl':a<<b,'shr':a>>b,'not':~a}.get(o)}
        if op=='hmac':
            import hmac as _h;key=str(args.get('key','')).encode();a=(args.get('algo') or 'sha256').lower();return {'algo':a,'result':_h.new(key,str(t).encode(),a).hexdigest()}
        if op=='crc32':
            import zlib;return {'result':format(zlib.crc32(str(t).encode())&0xffffffff,'08x')}
        return {'error':'op one of: base64_encode/decode, hex_encode/decode, url_encode/decode, hash, hmac, crc32, uuid, json_pretty/minify/validate, regex, base, bitwise'}
    except Exception as e:return {'error':f'codec: {e}'}
def _to_base(n,b):
    if n==0:return '0'
    dig='0123456789abcdefghijklmnopqrstuvwxyz';s='';neg=n<0;n=abs(n)
    while n:s=dig[n%b]+s;n//=b
    return ('-' if neg else '')+s
def _skill_stats(args,ctx,reg):
    import statistics as st
    nums=args.get('numbers');data=args.get('data')
    if nums is None and data:nums=[float(x) for x in re.findall(r'-?\d+\.?\d*',str(data))]
    if not nums:
        x=args.get('x');y=args.get('y')
        if x and y:
            try:
                import numpy as np;x=np.array(x,float);y=np.array(y,float);m,b=np.polyfit(x,y,1);r=float(np.corrcoef(x,y)[0,1])
                return {'slope':round(float(m),6),'intercept':round(float(b),6),'correlation':round(r,6),'fit':f'y = {m:.4g}*x + {b:.4g}'}
            except Exception as e:return {'error':f'fit: {e}'}
        return {'error':'need numbers:[...] or data:"1,2,3" (or x:[],y:[] for linear fit)'}
    nums=[float(n) for n in nums]
    try:
        return {'count':len(nums),'sum':round(sum(nums),8),'mean':round(st.mean(nums),8),'median':st.median(nums),'mode':(st.mode(nums) if len(set(nums))<len(nums) else None),'stdev':(round(st.pstdev(nums),8)),'variance':round(st.pvariance(nums),8),'min':min(nums),'max':max(nums),'range':round(max(nums)-min(nums),8)}
    except Exception as e:return {'error':f'stats: {e}'}
_MASS={'H':1.008,'He':4.0026,'Li':6.94,'Be':9.0122,'B':10.81,'C':12.011,'N':14.007,'O':15.999,'F':18.998,'Ne':20.18,'Na':22.99,'Mg':24.305,'Al':26.982,'Si':28.085,'P':30.974,'S':32.06,'Cl':35.45,'Ar':39.95,'K':39.098,'Ca':40.078,'Sc':44.956,'Ti':47.867,'V':50.942,'Cr':51.996,'Mn':54.938,'Fe':55.845,'Co':58.933,'Ni':58.693,'Cu':63.546,'Zn':65.38,'Ga':69.723,'Ge':72.63,'As':74.922,'Se':78.971,'Br':79.904,'Kr':83.798,'Rb':85.468,'Sr':87.62,'Y':88.906,'Zr':91.224,'Nb':92.906,'Mo':95.95,'Tc':98,'Ru':101.07,'Rh':102.91,'Pd':106.42,'Ag':107.868,'Cd':112.41,'In':114.82,'Sn':118.71,'Sb':121.76,'Te':127.6,'I':126.9,'Xe':131.29,'Cs':132.91,'Ba':137.33,'La':138.91,'Ce':140.12,'Pr':140.91,'Nd':144.24,'Pm':145,'Sm':150.36,'Eu':151.96,'Gd':157.25,'Tb':158.93,'Dy':162.5,'Ho':164.93,'Er':167.26,'Tm':168.93,'Yb':173.05,'Lu':174.97,'Hf':178.49,'Ta':180.95,'W':183.84,'Re':186.21,'Os':190.23,'Ir':192.22,'Pt':195.08,'Au':196.97,'Hg':200.59,'Tl':204.38,'Pb':207.2,'Bi':208.98,'Po':209,'At':210,'Rn':222,'Fr':223,'Ra':226,'Ac':227,'Th':232.04,'Pa':231.04,'U':238.03,'Np':237,'Pu':244}
def _parse_formula(f):
    f=str(f).replace('·','.').replace('*','.').replace(' ','');total={}
    def parse(s):
        i=0;c={}
        while i<len(s):
            if s[i]=='(' or s[i]=='[':
                depth=1;j=i+1
                while j<len(s) and depth:depth+=(s[j] in '([')-(s[j] in ')]');j+=1
                inner=parse(s[i+1:j-1]);m=re.match(r'\d+',s[j:]);mult=int(m.group()) if m else 1
                for k,v in inner.items():c[k]=c.get(k,0)+v*mult
                i=j+(len(m.group()) if m else 0)
            elif s[i].isupper():
                m=re.match(r'[A-Z][a-z]?',s[i:]);sym=m.group();i+=len(sym);n=re.match(r'\d+',s[i:]);cnt=int(n.group()) if n else 1;i+=len(n.group()) if n else 0;c[sym]=c.get(sym,0)+cnt
            else:i+=1
        return c
    for part in f.split('.'):
        if not part:continue
        mh=re.match(r'^(\d+)(.+)',part);mult=1
        if mh:mult=int(mh.group(1));part=mh.group(2)
        for k,v in parse(part).items():total[k]=total.get(k,0)+v*mult
    return total
def _skill_chem(args,ctx,reg):
    f=args.get('formula') or args.get('query','')
    try:
        comp=_parse_formula(f);bad=[s for s in comp if s not in _MASS]
        if bad:return {'error':'unknown element(s): '+','.join(bad)}
        if not comp:return {'error':'no formula parsed (try H2O, NaCl, C6H12O6, CuSO4.5H2O)'}
        mm=sum(_MASS[s]*c for s,c in comp.items())
        return {'formula':f,'molar_mass':round(mm,4),'unit':'g/mol','composition':comp,'percent_by_mass':{s:round(_MASS[s]*c/mm*100,2) for s,c in comp.items()}}
    except Exception as e:return {'error':f'chem: {e}'}
def _skill_finance(args,ctx,reg):
    op=(args.get('op') or '').lower();g=lambda k,d=None:float(args[k]) if k in args and args[k] is not None else d
    try:
        if op in('compound','compound_interest','fv'):
            P=g('principal',0) or g('present_value',0);r=g('rate')/100;n=g('n',12);t=g('years');A=P*(1+r/n)**(n*t);return {'future_value':round(A,2),'interest':round(A-P,2)}
        if op in('loan','mortgage','payment'):
            P=g('principal');r=g('rate')/100/12;n=g('years')*12;pay=(P*r/(1-(1+r)**-n)) if r else P/n;return {'monthly_payment':round(pay,2),'total_paid':round(pay*n,2),'total_interest':round(pay*n-P,2)}
        if op in('pct_change','change','roi'):
            a=g('from') or g('begin') or g('cost');b=g('to') or g('end') or g('value');return {'percent_change':round((b-a)/a*100,4)}
        if op=='cagr':
            b=g('begin') or g('from');e=g('end') or g('to');y=g('years');return {'cagr_percent':round(((e/b)**(1/y)-1)*100,4)}
        if op in('pv','present_value'):
            FV=g('future_value');r=g('rate')/100;t=g('years');return {'present_value':round(FV/(1+r)**t,2)}
        return {'error':'op: compound|loan|pct_change|cagr|pv'}
    except Exception as e:return {'error':f'finance: {e} (check args)'}
def _skill_password(args,ctx,reg):
    import secrets,string as _s
    n=max(4,min(256,int(args.get('length',20))));kind=(args.get('kind') or 'strong').lower()
    if kind=='token':return {'token':secrets.token_urlsafe(n)}
    if kind=='hextoken':return {'token':secrets.token_hex(n)}
    pool={'strong':_s.ascii_letters+_s.digits+'!@#$%^&*-_=+?','alnum':_s.ascii_letters+_s.digits,'hex':'0123456789abcdef','pin':_s.digits,'letters':_s.ascii_letters}.get(kind,_s.ascii_letters+_s.digits+'!@#$%^&*-_=+?')
    return {'password':''.join(secrets.choice(pool) for _ in range(n)),'length':n,'kind':kind}
def _skill_color(args,ctx,reg):
    import colorsys;c=str(args.get('color') or args.get('query','')).strip()
    try:
        if args.get('r') is not None:r,g,b=int(args['r']),int(args['g']),int(args['b'])
        else:
            h=c.lstrip('#');rgb=re.findall(r'\d+',c)
            if re.match(r'^[0-9a-fA-F]{6}$',h):r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
            elif len(rgb)>=3:r,g,b=int(rgb[0]),int(rgb[1]),int(rgb[2])
            else:return {'error':'give hex (#1e90ff) or rgb (30,144,255) or r,g,b'}
        hl,ll,sl=colorsys.rgb_to_hls(r/255,g/255,b/255)
        lum=0.2126*r/255+0.7152*g/255+0.0722*b/255
        return {'hex':'#%02x%02x%02x'%(r,g,b),'rgb':[r,g,b],'hsl':[round(hl*360),round(sl*100),round(ll*100)],'luminance':round(lum,3),'on_color':'black' if lum>.5 else 'white'}
    except Exception as e:return {'error':f'color: {e}'}
def _skill_geo(args,ctx,reg):
    try:
        la1,lo1,la2,lo2=[math.radians(float(args[k])) for k in ('lat1','lon1','lat2','lon2')]
        R=6371.0088;a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2;d=2*R*math.asin(math.sqrt(a))
        br=(math.degrees(math.atan2(math.sin(lo2-lo1)*math.cos(la2),math.cos(la1)*math.sin(la2)-math.sin(la1)*math.cos(la2)*math.cos(lo2-lo1)))+360)%360
        return {'distance_km':round(d,3),'distance_mi':round(d*0.621371,3),'bearing_deg':round(br,1)}
    except Exception as e:return {'error':'need lat1,lon1,lat2,lon2'}
def _skill_net(args,ctx,reg):
    import ipaddress;q=str(args.get('query') or args.get('cidr') or args.get('ip','')).strip()
    try:
        if '/' in q:
            net=ipaddress.ip_network(q,strict=False);hosts=net.num_addresses
            return {'network':str(net.network_address),'broadcast':str(getattr(net,'broadcast_address','')),'netmask':str(net.netmask),'prefix':net.prefixlen,'num_addresses':hosts,'usable_hosts':max(0,hosts-2) if net.version==4 else hosts,'version':net.version}
        ip=ipaddress.ip_address(q)
        return {'ip':str(ip),'version':ip.version,'is_private':ip.is_private,'is_global':ip.is_global,'is_loopback':ip.is_loopback,'reverse_dns':ip.reverse_pointer}
    except Exception as e:return {'error':f'net: {e} (give an IP or CIDR like 192.168.1.0/24)'}
def _skill_text(args,ctx,reg):
    import codecs;op=(args.get('op') or 'stats').lower();t=str(args.get('text',''))
    try:
        if op in('stats','count'):return {'chars':len(t),'chars_no_spaces':len(t.replace(' ','')),'words':len(t.split()),'lines':t.count('\n')+1 if t else 0,'sentences':len(re.findall(r'[.!?]+',t))}
        if op=='upper':return {'result':t.upper()}
        if op=='lower':return {'result':t.lower()}
        if op=='title':return {'result':t.title()}
        if op=='capitalize':return {'result':t.capitalize()}
        if op=='reverse':return {'result':t[::-1]}
        if op=='slug':return {'result':re.sub(r'[^a-z0-9]+','-',t.lower()).strip('-')}
        if op=='rot13':return {'result':codecs.encode(t,'rot_13')}
        if op in('snake',):return {'result':re.sub(r'[\s\-]+','_',t.strip().lower())}
        if op in('camel',):
            parts=re.split(r'[\s_\-]+',t.strip());return {'result':parts[0].lower()+''.join(w.capitalize() for w in parts[1:])}
        return {'error':'op: stats|upper|lower|title|capitalize|reverse|slug|rot13|snake|camel'}
    except Exception as e:return {'error':f'text: {e}'}
def _skill_random(args,ctx,reg):
    import secrets,random as _r;op=(args.get('op') or 'int').lower();items=args.get('items') or []
    try:
        if op=='dice':n=max(1,int(args.get('n',1)));s=max(2,int(args.get('sides',6)));rolls=[secrets.randbelow(s)+1 for _ in range(n)];return {'rolls':rolls,'total':sum(rolls)}
        if op=='coin':return {'result':'heads' if secrets.randbelow(2) else 'tails'}
        if op in('int','number'):lo=int(args.get('min',1));hi=int(args.get('max',100));return {'result':secrets.randbelow(hi-lo+1)+lo}
        if op in('pick','choice'):return {'result':items[secrets.randbelow(len(items))]} if items else {'error':'need items'}
        if op=='shuffle':x=list(items);_r.SystemRandom().shuffle(x);return {'result':x}
        if op=='sample':k=min(int(args.get('k',1)),len(items));return {'result':_r.SystemRandom().sample(items,k)} if items else {'error':'need items'}
        if op=='uuid':import uuid;return {'result':str(uuid.uuid4())}
        return {'error':'op: dice|coin|int|pick|shuffle|sample|uuid'}
    except Exception as e:return {'error':f'random: {e}'}
def _skill_solve(args,ctx,reg):
    import sympy as sp
    raw=str(args.get('equation') or args.get('name') or args.get('query','')).strip()
    knowns=args.get('values') or {};target=args.get('solve_for') or args.get('for')
    eqstr=(_EQ[raw.lower()][0] if raw.lower() in _EQ else raw).replace('^','**')
    if '=' not in eqstr:return {'error':'need an equation with = , or a named one: '+', '.join(list(_EQ)[:8])}
    try:
        _fns={'ln','log','exp','sin','cos','tan','sqrt','asin','acos','atan','pi','abs'}
        _loc={n:sp.Symbol(n) for n in set(re.findall(r'[A-Za-z_]\w*',eqstr)) if n not in _fns}
        lhs,_,rhs=eqstr.partition('=');eq=sp.Eq(sp.sympify(lhs,locals=_loc),sp.sympify(rhs,locals=_loc))
        subs={sp.Symbol(k):float(v) for k,v in knowns.items()}
        for cn,(cv,_,_) in _CONST.items():
            s=sp.Symbol(cn)
            if s in eq.free_symbols and cn not in knowns:subs[s]=cv
        eq2=eq.subs(subs);unk=sorted(eq2.free_symbols,key=str)
        if target:tgt=sp.Symbol(target)
        elif len(unk)==1:tgt=unk[0]
        else:return {'error':'specify solve_for; unknowns: '+','.join(map(str,unk))}
        sol=sp.solve(eq2,tgt);vals=[]
        for s in sol:
            try:vals.append(round(float(sp.N(s)),6) if not s.free_symbols else str(s))
            except Exception:vals.append(str(s))
        return {'equation':eqstr,'solve_for':str(tgt),'solution':(vals[0] if len(vals)==1 else vals),'used':{str(k):v for k,v in subs.items()}}
    except Exception as e:return {'error':f'solve: {e}'}
def _skill_roman(args,ctx,reg):
    v=str(args.get('value') or args.get('query','')).strip().upper()
    RR=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    try:
        if v.lstrip('-').isdigit():
            n=int(v);out=''
            for val,sym in RR:
                while n>=val:out+=sym;n-=val
            return {'roman':out}
        m={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000};tot=0;prev=0
        for ch in reversed(v):cur=m[ch];tot+=(-cur if cur<prev else cur);prev=cur
        return {'integer':tot}
    except Exception as e:return {'error':f'roman: {e}'}
_MORSE={'A':'.-','B':'-...','C':'-.-.','D':'-..','E':'.','F':'..-.','G':'--.','H':'....','I':'..','J':'.---','K':'-.-','L':'.-..','M':'--','N':'-.','O':'---','P':'.--.','Q':'--.-','R':'.-.','S':'...','T':'-','U':'..-','V':'...-','W':'.--','X':'-..-','Y':'-.--','Z':'--..','0':'-----','1':'.----','2':'..---','3':'...--','4':'....-','5':'.....','6':'-....','7':'--...','8':'---..','9':'----.','.':'.-.-.-',',':'--..--','?':'..--..','/':'-..-.','-':'-....-','(':'-.--.',')':'-.--.-',' ':'/'}
def _skill_cipher(args,ctx,reg):
    op=(args.get('op') or '').lower();t=str(args.get('text',''));sh=int(args.get('shift',3))
    try:
        if op=='morse_encode':return {'result':' '.join(_MORSE.get(c.upper(),'?') for c in t)}
        if op=='morse_decode':
            inv={v:k for k,v in _MORSE.items()};return {'result':''.join(inv.get(c,'?') for c in t.split(' '))}
        if op in('caesar','rot','shift'):
            r=''.join(chr((ord(c)-b+sh)%26+b) if (b:=(65 if c.isupper() else 97)) and c.isalpha() else c for c in t);return {'result':r,'shift':sh}
        if op=='atbash':return {'result':''.join(chr((25-(ord(c.lower())-97))+(65 if c.isupper() else 97)) if c.isalpha() else c for c in t)}
        if op=='binary_encode':return {'result':' '.join(format(ord(c),'08b') for c in t)}
        if op=='binary_decode':return {'result':''.join(chr(int(b,2)) for b in t.split())}
        if op=='reverse':return {'result':t[::-1]}
        return {'error':'op: morse_encode/decode | caesar(shift) | atbash | binary_encode/decode'}
    except Exception as e:return {'error':f'cipher: {e}'}
def _skill_numtheory(args,ctx,reg):
    import sympy as sp
    op=(args.get('op') or '').lower()
    try:
        n=int(args['n']) if args.get('n') is not None else None;k=int(args['k']) if args.get('k') is not None else None
        if op in('isprime','prime'):return {'n':n,'is_prime':bool(sp.isprime(n))}
        if op in('factorize','factor','factors'):return {'n':n,'factorization':{int(p):int(e) for p,e in sp.factorint(n).items()}}
        if op=='gcd':return {'gcd':int(math.gcd(n,k))}
        if op=='lcm':return {'lcm':abs(n*k)//math.gcd(n,k)}
        if op in('ncr','choose','combination'):return {'result':int(sp.binomial(n,k))}
        if op in('npr','permutation'):return {'result':int(sp.factorial(n)//sp.factorial(n-k))}
        if op=='factorial':return {'result':int(sp.factorial(n))}
        if op in('fib','fibonacci'):return {'result':int(sp.fibonacci(n))}
        if op in('nextprime','next_prime'):return {'result':int(sp.nextprime(n))}
        if op in('totient','phi'):return {'result':int(sp.totient(n))}
        if op=='divisors':return {'divisors':[int(d) for d in sp.divisors(n)]}
        return {'error':'op: isprime|factorize|gcd|lcm|ncr|npr|factorial|fib|nextprime|totient|divisors (args n, k?)'}
    except Exception as e:return {'error':f'numtheory: {e}'}
def _skill_matrix(args,ctx,reg):
    import numpy as np
    op=(args.get('op') or '').lower()
    try:
        A=np.array(args.get('a') if args.get('a') is not None else args.get('matrix'),float)
        B=np.array(args['b'],float) if args.get('b') is not None else None
        if op in('multiply','matmul','dot'):return {'result':(A@B).tolist()}
        if op=='add':return {'result':(A+B).tolist()}
        if op in('transpose','t'):return {'result':A.T.tolist()}
        if op in('determinant','det'):return {'result':round(float(np.linalg.det(A)),8)}
        if op in('inverse','inv'):return {'result':np.linalg.inv(A).round(8).tolist()}
        if op=='rank':return {'result':int(np.linalg.matrix_rank(A))}
        if op in('eigenvalues','eig'):return {'eigenvalues':[round(float(x),6) for x in np.linalg.eigvals(A).real]}
        if op in('solve','linsolve'):return {'result':np.linalg.solve(A,B).round(8).tolist()}
        return {'error':'op: multiply|add|transpose|det|inverse|rank|eig|solve (args a:[[...]], b:[[...]])'}
    except Exception as e:return {'error':f'matrix: {e}'}
def _skill_calendar(args,ctx,reg):
    from datetime import date
    import calendar as _cal
    try:from dateutil import parser as _dp
    except Exception:_dp=None
    P=lambda s:(_dp.parse(str(s)).date() if _dp else date.fromisoformat(str(s)))
    op=(args.get('op') or 'weekday').lower()
    try:
        if op=='weekday':return {'date':str(P(args['date'])),'weekday':P(args['date']).strftime('%A')}
        if op in('leap','leap_year'):y=int(args['year']) if args.get('year') else P(args['date']).year;return {'year':y,'is_leap':_cal.isleap(y)}
        if op in('days_in_month','month_length'):d=P(args['date']) if args.get('date') else date(int(args['year']),int(args['month']),1);return {'days':_cal.monthrange(d.year,d.month)[1]}
        if op=='age':b=P(args.get('birthdate') or args['date']);t=date.today();return {'age_years':t.year-b.year-((t.month,t.day)<(b.month,b.day))}
        if op in('day_of_year','doy'):return {'day_of_year':P(args['date']).timetuple().tm_yday}
        return {'error':'op: weekday|leap_year|days_in_month|age|day_of_year'}
    except Exception as e:return {'error':f'calendar: {e}'}
def _skill_validate(args,ctx,reg):
    op=(args.get('op') or '').lower();v=str(args.get('value',''))
    def _luhn(s):
        s=re.sub(r'\D','',s)
        if not s:return False
        d=[int(c) for c in s][::-1];return (sum(d[0::2])+sum(sum(divmod(x*2,10)) for x in d[1::2]))%10==0
    try:
        if op in('luhn','credit_card','card'):return {'valid':_luhn(v)}
        if op=='email':return {'valid':bool(re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$',v))}
        if op=='url':return {'valid':bool(re.match(r'^https?://[^\s]+$',v))}
        if op=='isbn':
            d=re.sub(r'[^0-9Xx]','',v)
            if len(d)==10:return {'valid':sum((10-i)*(10 if c in 'Xx' else int(c)) for i,c in enumerate(d))%11==0}
            if len(d)==13:return {'valid':sum((1 if i%2==0 else 3)*int(c) for i,c in enumerate(d))%10==0}
            return {'valid':False,'error':'ISBN must be 10 or 13 chars'}
        if op=='json':
            import json as _j
            try:_j.loads(v);return {'valid':True}
            except Exception as e:return {'valid':False,'error':str(e)}
        if op in('ipv4','ip'):
            p=v.split('.');return {'valid':len(p)==4 and all(x.isdigit() and 0<=int(x)<=255 for x in p)}
        return {'error':'op: luhn|email|url|isbn|json|ipv4'}
    except Exception as e:return {'error':f'validate: {e}'}
def _skill_dataformat(args,ctx,reg):
    import json as _j,csv as _csv,io
    op=(args.get('op') or '').lower();t=args.get('text','')
    try:
        if op=='csv_to_json':return {'result':list(_csv.DictReader(io.StringIO(str(t))))}
        if op=='json_to_csv':
            data=_j.loads(t) if isinstance(t,str) else t;data=data if isinstance(data,list) else [data];out=io.StringIO();w=_csv.DictWriter(out,fieldnames=list(data[0].keys()));w.writeheader();w.writerows(data);return {'result':out.getvalue()}
        if op=='flatten':
            def fl(o,p=''):
                it={}
                for k,vv in (o.items() if isinstance(o,dict) else enumerate(o)):
                    nk=f'{p}.{k}' if p else str(k);it.update(fl(vv,nk) if isinstance(vv,(dict,list)) else {nk:vv})
                return it
            return {'result':fl(_j.loads(t) if isinstance(t,str) else t)}
        return {'error':'op: csv_to_json|json_to_csv|flatten'}
    except Exception as e:return {'error':f'dataformat: {e}'}
def _skill_diff(args,ctx,reg):
    import difflib
    a=str(args.get('a','')).splitlines();b=str(args.get('b','')).splitlines()
    try:
        d=list(difflib.unified_diff(a,b,lineterm='',n=int(args.get('context',2))))
        return {'diff':'\n'.join(d),'similarity':round(difflib.SequenceMatcher(None,'\n'.join(a),'\n'.join(b)).ratio(),4),'added':sum(1 for x in d if x[:1]=='+' and x[:3]!='+++'),'removed':sum(1 for x in d if x[:1]=='-' and x[:3]!='---')}
    except Exception as e:return {'error':f'diff: {e}'}
_O1=['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
_T1=['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
def _skill_num2words(args,ctx,reg):
    try:n=int(args['n'])
    except Exception:return {'error':'need integer n'}
    def three(x):
        s=''
        if x>=100:s+=_O1[x//100]+' hundred';x%=100;s+=' ' if x else ''
        if x>=20:s+=_T1[x//10]+('-'+_O1[x%10] if x%10 else '');x=0
        if x>0:s+=_O1[x]
        return s
    if n==0:return {'words':'zero'}
    sc=['','thousand','million','billion','trillion','quadrillion'];neg=n<0;n=abs(n);parts=[];i=0
    while n and i<len(sc):
        if n%1000:parts.insert(0,three(n%1000)+(' '+sc[i] if sc[i] else ''))
        n//=1000;i+=1
    return {'words':('negative ' if neg else '')+' '.join(parts).strip()}
def _skill_resistor(args,ctx,reg):
    col={'black':0,'brown':1,'red':2,'orange':3,'yellow':4,'green':5,'blue':6,'violet':7,'purple':7,'grey':8,'gray':8,'white':9}
    mult={'black':1,'brown':10,'red':100,'orange':1e3,'yellow':1e4,'green':1e5,'blue':1e6,'violet':1e7,'grey':1e8,'gray':1e8,'white':1e9,'gold':.1,'silver':.01}
    tol={'brown':1,'red':2,'green':.5,'blue':.25,'violet':.1,'gold':5,'silver':10}
    bands=args.get('bands') or str(args.get('query','')).lower().split()
    try:
        bands=[b.lower().strip() for b in bands if str(b).strip()]
        if len(bands) not in (4,5):return {'error':'give 4 or 5 color bands, e.g. bands=["brown","black","red","gold"]'}
        if len(bands)==4:val=(col[bands[0]]*10+col[bands[1]])*mult[bands[2]];t=tol.get(bands[3])
        else:val=(col[bands[0]]*100+col[bands[1]]*10+col[bands[2]])*mult[bands[3]];t=tol.get(bands[4])
        v,unit=val,'ohm'
        for u,dd in [('Mohm',1e6),('kohm',1e3)]:
            if val>=dd:v,unit=val/dd,u;break
        return {'resistance_ohms':val,'display':f'{round(v,3)} {unit}','tolerance_percent':t}
    except Exception as e:return {'error':f'resistor: bad color band ({e})'}
def _skill_quote(args,ctx,reg):
    import requests
    sym=str(args.get('symbol') or args.get('query','')).strip().lower();kind=(args.get('kind') or '').lower()
    idmap={'btc':'bitcoin','eth':'ethereum','sol':'solana','doge':'dogecoin','ada':'cardano','xrp':'ripple','bnb':'binancecoin','ltc':'litecoin'}
    try:
        if kind=='crypto' or sym in idmap or sym in idmap.values():
            cid=idmap.get(sym,sym);r=requests.get('https://api.coingecko.com/api/v3/simple/price',params={'ids':cid,'vs_currencies':'usd','include_24hr_change':'true'},timeout=8).json()
            if cid in r:return {'symbol':cid,'price_usd':r[cid]['usd'],'change_24h_pct':round(r[cid].get('usd_24h_change',0),2),'source':'coingecko'}
        m=re.match(r'([a-z]{3})\s*(?:to|/|->|,| )+\s*([a-z]{3})',sym)
        if m:
            base=m.group(1).upper();tgt=m.group(2).upper();amt=float(args.get('amount',1));r=requests.get('https://api.frankfurter.app/latest',params={'from':base,'to':tgt,'amount':amt},timeout=8).json()
            if r.get('rates',{}).get(tgt) is not None:return {'from':base,'to':tgt,'amount':amt,'result':r['rates'][tgt],'source':'frankfurter.app (ECB)'}
        return {'error':'give a crypto (btc/eth) or an FX pair (e.g. "usd eur")'}
    except Exception as e:return {'error':f'quote: network unavailable ({type(e).__name__})'}
def _skill_define(args,ctx,reg):
    import requests
    w=str(args.get('word') or args.get('query','')).strip()
    try:
        r=requests.get(f'https://api.dictionaryapi.dev/api/v2/entries/en/{w}',timeout=8).json()
        if isinstance(r,list) and r:
            e=r[0];defs=[m['definition'] for mn in e.get('meanings',[]) for m in mn.get('definitions',[])][:3]
            return {'word':w,'phonetic':e.get('phonetic'),'part_of_speech':[m['partOfSpeech'] for m in e.get('meanings',[])],'definitions':defs}
        return {'word':w,'error':'not found'}
    except Exception as e:return {'error':f'define: network unavailable ({type(e).__name__})'}
def _skill_password_strength(args,ctx,reg):
    p=str(args.get('password') or args.get('value',''))
    pool=(26 if re.search(r'[a-z]',p) else 0)+(26 if re.search(r'[A-Z]',p) else 0)+(10 if re.search(r'\d',p) else 0)+(33 if re.search(r'[^a-zA-Z0-9]',p) else 0)
    ent=len(p)*math.log2(pool) if pool else 0
    rate='very weak' if ent<28 else 'weak' if ent<36 else 'reasonable' if ent<60 else 'strong' if ent<128 else 'very strong'
    return {'length':len(p),'charset_size':pool,'entropy_bits':round(ent,1),'rating':rate}
def _skill_timestamp(args,ctx,reg):
    from datetime import datetime,timezone
    v=str(args.get('value') or args.get('query','')).strip()
    try:
        if v.replace('.','',1).isdigit():
            ts=float(v);ts=ts/1000 if ts>1e12 else ts
            return {'unix':ts,'utc':datetime.fromtimestamp(ts,timezone.utc).isoformat(),'local':datetime.fromtimestamp(ts).isoformat(timespec='seconds')}
        try:from dateutil import parser as _dp;d=_dp.parse(v)
        except Exception:d=datetime.fromisoformat(v)
        return {'iso':d.isoformat(timespec='seconds'),'unix':int(d.timestamp())}
    except Exception as e:return {'error':f'timestamp: {e}'}
def _skill_unicode(args,ctx,reg):
    import unicodedata
    t=str(args.get('text') or args.get('query',''))
    try:
        if t.lower().startswith('u+') or args.get('codepoint') is not None:
            cp=int(str(args.get('codepoint') or t[2:]),16);ch=chr(cp);return {'char':ch,'codepoint':f'U+{cp:04X}','name':unicodedata.name(ch,'?'),'category':unicodedata.category(ch)}
        return {'chars':[{'char':c,'codepoint':f'U+{ord(c):04X}','name':unicodedata.name(c,'?')} for c in t[:20]]}
    except Exception as e:return {'error':f'unicode: {e}'}
_LOREM='lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua enim ad minim veniam quis nostrud exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure in reprehenderit voluptate velit esse cillum'.split()
def _skill_lorem(args,ctx,reg):
    import random as _r
    n=max(1,int(args.get('n',30)));unit=(args.get('unit') or 'words').lower();rng=_r.Random(int(args.get('seed',42)))
    sent=lambda:(' '.join(rng.choice(_LOREM) for _ in range(rng.randint(6,14)))).capitalize()+'.'
    if unit.startswith('word'):return {'text':' '.join(rng.choice(_LOREM) for _ in range(n)).capitalize()}
    if unit.startswith('sent'):return {'text':' '.join(sent() for _ in range(n))}
    if unit.startswith('para'):return {'text':'\n\n'.join(' '.join(sent() for _ in range(rng.randint(3,6))) for _ in range(n))}
    return {'error':'unit: words|sentences|paragraphs'}
def _skill_jwt(args,ctx,reg):
    import base64,json as _j
    tok=str(args.get('token') or args.get('query','')).strip()
    try:
        parts=tok.split('.')
        if len(parts)<2:return {'error':'not a JWT (need header.payload.signature)'}
        dec=lambda s:_j.loads(base64.urlsafe_b64decode(s+'='*(-len(s)%4)))
        h=dec(parts[0]);p=dec(parts[1]);out={'header':h,'payload':p,'signature_present':len(parts)>2}
        if 'exp' in p:
            from datetime import datetime,timezone;out['expires']=datetime.fromtimestamp(p['exp'],timezone.utc).isoformat();out['expired']=p['exp']<datetime.now(timezone.utc).timestamp()
        return out
    except Exception as e:return {'error':f'jwt decode: {e}'}
def _skill_url(args,ctx,reg):
    from urllib.parse import urlparse,parse_qs
    u=str(args.get('url') or args.get('query',''))
    try:
        p=urlparse(u);return {'scheme':p.scheme,'host':p.hostname,'port':p.port,'path':p.path,'query':parse_qs(p.query),'fragment':p.fragment or None}
    except Exception as e:return {'error':f'url: {e}'}
def _skill_semver(args,ctx,reg):
    a=str(args.get('a','')).lstrip('vV');b=str(args.get('b','')).lstrip('vV')
    pr=lambda s:(lambda m:tuple(int(x) for x in m.groups()) if m else None)(re.match(r'(\d+)\.(\d+)\.(\d+)',s))
    try:
        pa,pb=pr(a),pr(b)
        if not pa or not pb:return {'error':'need two semver strings like 1.2.3'}
        c=(pa>pb)-(pa<pb);return {'a':a,'b':b,'comparison':{1:'a > b',0:'a == b',-1:'a < b'}[c],'result':c}
    except Exception as e:return {'error':f'semver: {e}'}
def _skill_cron(args,ctx,reg):
    c=str(args.get('expression') or args.get('query','')).strip()
    try:
        parts=c.split()
        if len(parts)!=5:return {'error':'cron needs 5 fields: minute hour day-of-month month day-of-week'}
        names=['minute','hour','day-of-month','month','day-of-week']
        def d(f,n):
            return f'every {n}' if f=='*' else f'every {f[2:]} {n}s' if f.startswith('*/') else (f'{n} in {f}' if (',' in f or '-' in f) else f'at {n} {f}')
        return {'expression':c,'fields':dict(zip(names,parts)),'description':'; '.join(d(parts[i],names[i]) for i in range(5))}
    except Exception as e:return {'error':f'cron: {e}'}
def _skill_translate(args,ctx,reg):
    import requests
    t=str(args.get('text',''));to=(args.get('to') or 'es').lower();frm=(args.get('from') or 'en').lower()
    if not t:return {'error':'need text'}
    try:
        r=requests.get('https://api.mymemory.translated.net/get',params={'q':t,'langpair':f'{frm}|{to}'},timeout=8).json()
        return {'translation':r['responseData']['translatedText'],'from':frm,'to':to,'source':'mymemory'}
    except Exception as e:return {'error':f'translate: network unavailable ({type(e).__name__})'}
_RXTOK={'^':'start of string','$':'end of string','.':'any character','\\d':'a digit','\\D':'a non-digit','\\w':'a word char','\\W':'a non-word char','\\s':'whitespace','\\S':'non-whitespace','\\b':'word boundary','*':'zero or more','+':'one or more','?':'optional (0 or 1)','|':'OR','[':'character class start',']':'class end','(':'group start',')':'group end','{':'repetition start','}':'repetition end'}
def _skill_regex_explain(args,ctx,reg):
    p=str(args.get('pattern') or args.get('query',''));found=[];i=0
    while i<len(p):
        two=p[i:i+2]
        if two in _RXTOK:found.append({'token':two,'means':_RXTOK[two]});i+=2;continue
        if p[i] in _RXTOK:found.append({'token':p[i],'means':_RXTOK[p[i]]});i+=1;continue
        i+=1
    out={'pattern':p,'components':found}
    if args.get('text') is not None:
        try:out['matches']=re.findall(p,str(args['text']))
        except Exception as e:out['regex_error']=str(e)
    return out
def _skill_mem(args,ctx,reg):
    adam=ctx.get('adam')
    q=args.get('query','')
    k=int(args.get('k',3))
    if adam is None or not q:return {'error':'missing adam or query'}
    sl=getattr(adam,'sem_lut',None)
    n=len(sl._raw) if sl is not None else 0
    hits=[]
    if n>0:
        try:
            eff_margin=sl.auto_margin()
            soft=sl.lookup_soft(q,margin=eff_margin)
            if soft:hits.append({'q':'(soft-lookup)','a':soft,'score':None,'method':'soft_margin','store':'sem_lut'})
        except Exception as e:hits.append({'error':f'soft-lookup: {e}'})
        try:
            if hasattr(sl,'_ensure_encoder') and hasattr(sl,'_raw') and sl._raw:
                import numpy as np
                enc=sl._ensure_encoder()
                qv=enc([q])[0].astype('float32')
                kv=getattr(sl,'_stored_embs',None)
                if kv is None or len(kv)!=len(sl._raw):kv=enc([r[0] for r in sl._raw]).astype('float32')
                scores=kv@qv
                order=np.argsort(-scores)[:k]
                for i in order:
                    hits.append({'q':sl._raw[int(i)][0][:200],'a':sl._raw[int(i)][1][:400],'score':float(scores[int(i)]),'method':'flat_cosine','store':'sem_lut'})
        except Exception as e:hits.append({'error':f'flat-cosine: {e}'})
    loop=getattr(adam,'adam',None)
    try:
        lt=getattr(loop,'lut',None)
        ex=lt.lookup(q) if lt is not None else None
        if ex:hits.append({'q':(ex.get('q') or '')[:200],'a':(ex.get('a') or '')[:400],'score':1.0,'method':'answer_lut_exact','store':'atex_lut'})
        li=getattr(loop,'lut_index',None)
        if li is not None and li.size()>0:
            for k2,q2,a2,s2,c2 in li.search(q,k=k,min_score=0.5):hits.append({'q':q2[:200],'a':a2[:400],'score':round(float(s2),3),'method':'lut_embed','store':'atex_lut'})
    except Exception as e:hits.append({'error':f'atex-lut: {e}'})
    try:
        at=getattr(ctx.get('agent'),'atlas',None)
        if at is not None:
            for r in at.recall(q,session_id=str(args.get('session_id') or ''),k=min(k,3)):hits.append({'q':(r.get('user') or '')[:200],'a':(r.get('assistant') or '')[:400],'score':None,'method':f"conv_atlas_r{r.get('cell_radius')}",'store':'conversation_atlas'})
    except Exception as e:hits.append({'error':f'conv-atlas: {e}'})
    _li=getattr(loop,'lut_index',None)
    return {'hits':hits,'lessons_n':n,'stores':{'sem_lut':n,'atex_lut':len(getattr(getattr(loop,'lut',None),'_index',None) or {}),'lut_index':_li.size() if _li is not None else 0}}
def _scrub_pii_from_query(q:str,agent=None,source:str='web')->str:
    """Strip personal-info from any text before external HTTP. Delegates to the central pii_egress choke-point (single source of truth)."""
    try:
        from amni.serve.pii_egress import scrub as _scrub
        return _scrub(q,agent=agent,source=source)
    except Exception:return q
def _skill_web(args,ctx,reg):
    adam=ctx.get('adam')
    agent=ctx.get('agent')
    q_raw=args.get('query','')
    if not q_raw:return {'error':'missing query'}
    if adam is None or not hasattr(adam,'adam') or adam.adam.crawler is None:return {'error':'web crawler not available'}
    q=_scrub_pii_from_query(q_raw,agent=agent)
    try:
        ans,sources,n=adam.adam.crawler.crawl_and_distill(q,subject=None,letter_only=False)
        _sani=False
        try:
            from amni.serve.code_safety import sanitize_ingest as _si
            ans,_f=_si(ans or '');_sani=bool(_f)
        except Exception:pass
        return {'answer':ans,'sources':sources[:5],'tokens':n,'query_used':q,'pii_scrubbed':q!=q_raw,'sanitized':_sani}
    except Exception as e:return {'error':str(e),'query_used':q}
def _skill_find(args,ctx,reg):
    """Fast substring/regex search across the workdir. Returns top N hits as file:line + snippet."""
    import re as _re,fnmatch
    query=str(args.get('query') or '').strip()
    if not query:return {'error':'missing query'}
    use_regex=bool(args.get('regex',False))
    case_sensitive=bool(args.get('case_sensitive',False))
    glob_pat=str(args.get('glob') or '').strip()
    max_hits=int(args.get('max_hits',30) or 30)
    max_chars=int(args.get('max_chars',180) or 180)
    workdir=getattr(reg,'workdir',None) or '.'
    root=Path(workdir).resolve()
    if not root.exists():return {'error':f'workdir does not exist: {root}'}
    try:
        if use_regex:pat=_re.compile(query,0 if case_sensitive else _re.IGNORECASE)
        else:pat=_re.compile(_re.escape(query),0 if case_sensitive else _re.IGNORECASE)
    except _re.error as e:return {'error':f'bad regex: {e}'}
    _IGNORE_DIRS={'.git','.venv','venv','__pycache__','node_modules','.pytest_cache','.mypy_cache','.idea','.vscode','dist','build','.next','.nuxt','.adam-venvs','bakes','models','downloaded_models','hf_cache','ptex_hf','textures','checkpoints','eval_reports','full_lexicon_atlas','archive','environment_files'}
    _BINARY_EXTS={'.png','.jpg','.jpeg','.gif','.webp','.ico','.pdf','.zip','.tar','.gz','.7z','.exe','.dll','.so','.dylib','.bin','.npz','.safetensors','.onnx','.pt','.pth','.pyc','.pyo','.mp4','.mp3','.wav','.ogg','.woff','.woff2','.ttf','.eot'}
    hits=[];files_scanned=0;total_matches=0
    for p in root.rglob('*'):
        if not p.is_file():continue
        if any(part in _IGNORE_DIRS for part in p.relative_to(root).parts[:-1]):continue
        if p.suffix.lower() in _BINARY_EXTS:continue
        try:rel=str(p.relative_to(root)).replace('\\','/')
        except Exception:continue
        if glob_pat and not fnmatch.fnmatch(rel,glob_pat):continue
        try:size=p.stat().st_size
        except Exception:continue
        if size>500_000:continue
        try:text=p.read_text(encoding='utf-8',errors='ignore')
        except Exception:continue
        files_scanned+=1
        for ln,line in enumerate(text.splitlines(),1):
            if pat.search(line):
                snippet=line.strip()
                if len(snippet)>max_chars:
                    m=pat.search(snippet);mid=m.start() if m else 0
                    start=max(0,mid-max_chars//2);end=min(len(snippet),start+max_chars)
                    snippet=('…' if start>0 else '')+snippet[start:end]+('…' if end<len(snippet) else '')
                hits.append({'path':rel,'line':ln,'snippet':snippet})
                total_matches+=1
                if len(hits)>=max_hits:break
        if len(hits)>=max_hits:break
    return {'query':query,'regex':use_regex,'case_sensitive':case_sensitive,'glob':glob_pat or None,'hits':hits,'n_hits':len(hits),'files_scanned':files_scanned,'truncated':len(hits)>=max_hits and total_matches>=max_hits}
def _skill_file_read(args,ctx,reg):
    p=args['path'];max_bytes=int(args.get('max_bytes',65536));_pp=reg._abs(p)
    if _pp.is_dir():
        ents=sorted([e for e in _pp.iterdir()],key=lambda e:(e.is_file(),e.name.lower()))
        exts={}
        for e in ents:
            if e.is_file():k=(e.suffix.lower() or '<none>');exts[k]=exts.get(k,0)+1
        listing=[{'name':e.name,'kind':'dir' if e.is_dir() else 'file','ext':(e.suffix.lstrip('.') if e.is_file() else ''),'bytes':(e.stat().st_size if e.is_file() else 0)} for e in ents[:500]]
        return {'path':str(_pp),'is_dir':True,'entry_count':len(ents),'n_dirs':sum(1 for e in ents if e.is_dir()),'n_files':sum(1 for e in ents if e.is_file()),'ext_summary':dict(sorted(exts.items(),key=lambda kv:-kv[1])),'entries':listing}
    _offset=int(args.get('offset',0));_limit=int(args.get('limit',0));_line_offset=int(args.get('line_offset',args.get('line_0ffset',args.get('lineoffset',args.get('start_line',0)))));_line_limit=int(args.get('line_limit',args.get('line_1imit',args.get('lines',0))))
    raw=_pp.read_text(encoding='utf-8',errors='replace')
    if _line_offset or _line_limit:
        lines=raw.splitlines(keepends=True)
        sliced=lines[_line_offset:_line_offset+_line_limit] if _line_limit else lines[_line_offset:]
        data=''.join(sliced)[:max_bytes]
        return {'path':str(p),'content':data,'bytes':len(data),'total_lines':len(lines),'returned_lines':len(sliced),'line_offset':_line_offset}
    if _offset or _limit:
        end=_offset+_limit if _limit else _offset+max_bytes
        data=raw[_offset:end][:max_bytes]
        return {'path':str(p),'content':data,'bytes':len(data),'total_bytes':len(raw),'offset':_offset}
    data=raw[:max_bytes]
    return {'path':p,'content':data,'bytes':len(data)}
def _file_change_stats(before:str,after:str,max_preview_lines:int=10):
    bl=before.splitlines() if before else [];al=after.splitlines()
    return {'lines_before':len(bl),'lines_after':len(al),'lines_added':max(0,len(al)-len(bl)),'lines_removed':max(0,len(bl)-len(al)),'bytes_before':len(before),'bytes_after':len(after),'preview':'\n'.join(al[:max_preview_lines])+(f'\n... ({len(al)-max_preview_lines} more)' if len(al)>max_preview_lines else ''),'before_preview':'\n'.join(bl[:max_preview_lines])+(f'\n... ({len(bl)-max_preview_lines} more)' if len(bl)>max_preview_lines else ''),'diff_unified':_unified_diff(bl,al,max_lines=max_preview_lines*4)}
def _unified_diff(before_lines,after_lines,max_lines:int=40):
    import difflib
    if not before_lines and not after_lines:return ''
    if before_lines==after_lines:return ''
    raw=list(difflib.unified_diff(before_lines or [],after_lines or [],lineterm='',n=2))
    if len(raw)>max_lines:raw=raw[:max_lines]+[f'... ({len(raw)-max_lines} more diff lines)']
    return '\n'.join(raw)
def _skill_file_write(args,ctx,reg):
    from amni.serve.edit_verifier import verify_edit
    _wp=_write_protected(args.get('path',''))
    if _wp:return {'error':f'refused: {_wp} is write-protected (set AMNI_ALLOW_LAW_EDIT/AMNI_ALLOW_SECURITY_EDIT=1 to override)'}
    p=reg._abs(args['path']);content=args.get('content','')
    existed=p.exists();before=p.read_text(encoding='utf-8',errors='ignore') if existed else ''
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(content,encoding='utf-8')
    op='create' if not existed else 'overwrite'
    return {'path':str(p),'bytes_written':len(content),'ext':p.suffix.lstrip('.') or 'txt','created':not existed,'change':_file_change_stats(before,content),'verification':verify_edit(str(p),content,op=op)}
def _skill_code_edit(args,ctx,reg):
    from amni.serve.edit_verifier import verify_edit
    p=reg._abs(args['path']);find=args['find'];replace=args['replace'];count=int(args.get('count',1))
    src=p.read_text(encoding='utf-8')
    if find not in src:return {'error':'find string not present','path':str(p)}
    _fi=src.find(find);_aft=src[_fi+len(find):_fi+len(find)+1];_bef=src[_fi-1:_fi] if _fi>0 else ''
    _isw=lambda c:bool(c) and (c.isalnum() or c=='_')
    if _isw(find[-1:]) and _isw(_aft):return {'error':f'refused: your find ends mid-identifier — "...{find[-22:]}" is immediately followed in the file by "{_aft}", so applying it would split/duplicate a word and corrupt the code. Extend your find to end at a clean word boundary (include the whole identifier).','path':str(p)}
    if _isw(find[:1]) and _isw(_bef):return {'error':f'refused: your find begins mid-identifier (preceded by "{_bef}"). Start your find at a clean word boundary.','path':str(p)}
    new=src.replace(find,replace,count) if count>0 else src.replace(find,replace)
    if p.suffix=='.py':
        try:ast.parse(new)
        except SyntaxError as e:return {'error':f'syntax error after edit: {e}','path':str(p)}
    p.write_text(new,encoding='utf-8')
    return {'path':str(p),'replacements':src.count(find) if count==0 else min(count,src.count(find)),'ext':p.suffix.lstrip('.') or 'txt','change':_file_change_stats(src,new),'verification':verify_edit(str(p),new,op='edit')}
def _skill_shell(args,ctx,reg):
    from amni.serve.shell_audit import log_shell_run
    _g=_gate_shell(args,ctx,reg)
    if _g:return {'error':f'blocked: {_g}','cmd':args.get('cmd','')}
    cmd=args['cmd'];timeout=int(args.get('timeout',15));t0=time.time()
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
    dur=round(time.time()-t0,3)
    log_shell_run('shell',cmd,r.returncode,r.stdout,r.stderr,str(reg.workdir),dur)
    return {'cmd':cmd,'returncode':r.returncode,'stdout':r.stdout[:8000],'stderr':r.stderr[:4000],'duration_s':dur}
_GIT_SAFE_CMDS={'status','log','diff','branch','blame','show','ls-files','remote','config','rev-parse','describe','tag','shortlog','reflog'}
def _skill_git(args,ctx,reg):
    cmd=(args.get('cmd') or '').strip()
    if not cmd:return {'error':'missing cmd. Allowed: '+','.join(sorted(_GIT_SAFE_CMDS))}
    head=cmd.split()[0].lower()
    if head not in _GIT_SAFE_CMDS:return {'error':f'cmd {head!r} not in safe allowlist (read-only git ops only). Allowed: '+','.join(sorted(_GIT_SAFE_CMDS))}
    parts=['git']+cmd.split()
    if args.get('file'):parts.append(str(args['file']))
    if args.get('n') and head in ('log','shortlog','reflog'):parts[2:2]=['-n',str(int(args['n']))]
    try:
        from amni.serve.shell_audit import log_shell_run
        t0=time.time();r=subprocess.run(parts,capture_output=True,text=True,timeout=20,cwd=str(reg.workdir));dur=round(time.time()-t0,3)
        log_shell_run('git',' '.join(parts),r.returncode,r.stdout,r.stderr,str(reg.workdir),dur)
        return {'cmd':' '.join(parts),'returncode':r.returncode,'stdout':r.stdout[:6000],'stderr':r.stderr[:1500],'duration_s':dur}
    except FileNotFoundError:return {'error':'git not installed or not in PATH'}
    except subprocess.TimeoutExpired:return {'error':'git command exceeded 20s timeout'}
    except Exception as e:return {'error':f'{type(e).__name__}: {e}'}
def _apply_unified_diff(content:str,diff_text:str):
    lines=content.splitlines(keepends=True);out=lines[:];hunks=[]
    cur=None
    for ln in diff_text.splitlines():
        if ln.startswith('@@'):
            m=re.match(r'@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@',ln)
            if not m:continue
            cur={'old_start':int(m.group(1)),'old_count':int(m.group(2) or 1),'new_start':int(m.group(3)),'new_count':int(m.group(4) or 1),'lines':[]}
            hunks.append(cur)
        elif cur is not None and (ln.startswith(' ') or ln.startswith('+') or ln.startswith('-')):cur['lines'].append(ln)
    if not hunks:return None,'no @@ hunks found in diff'
    offset=0
    for h in hunks:
        old_lines=[(l[1:]+'\n' if not l[1:].endswith('\n') else l[1:]) for l in h['lines'] if l.startswith((' ','-'))]
        new_lines=[(l[1:]+'\n' if not l[1:].endswith('\n') else l[1:]) for l in h['lines'] if l.startswith((' ','+'))]
        start=h['old_start']-1+offset
        actual=out[start:start+len(old_lines)]
        if ''.join(actual).rstrip()!=''.join(old_lines).rstrip():return None,f"hunk at line {h['old_start']} doesn't match — file content drifted"
        out[start:start+len(old_lines)]=new_lines
        offset+=len(new_lines)-len(old_lines)
    return ''.join(out),None
_FORMATTERS={'.py':[('ruff format','ruff'),('black','black')],'.rs':[('rustfmt','rustfmt')],'.js':[('prettier --write','prettier')],'.jsx':[('prettier --write','prettier')],'.ts':[('prettier --write','prettier')],'.tsx':[('prettier --write','prettier')],'.go':[('gofmt -w','gofmt')],'.json':[('prettier --write','prettier')],'.html':[('prettier --write','prettier')],'.css':[('prettier --write','prettier')]}
_CODE_EXTS=('.py','.rs','.js','.jsx','.ts','.tsx','.mjs','.go','.cpp','.cc','.c','.h','.hpp','.java','.kt','.rb','.php','.swift','.cs','.scala')
def _skill_rename_symbol(args,ctx,reg):
    old=(args.get('old') or '').strip();new=(args.get('new') or '').strip()
    if not old or not new:return {'error':'missing old/new'}
    if not re.match(r'^[A-Za-z_]\w*$',old):return {'error':f'old name {old!r} not a valid identifier'}
    if not re.match(r'^[A-Za-z_]\w*$',new):return {'error':f'new name {new!r} not a valid identifier'}
    root=Path(reg.workdir);glob=args.get('glob');dry_run=bool(args.get('dry_run'))
    if glob:targets=[p for p in root.glob(glob) if p.is_file()]
    else:
        exts=set(args.get('exts') or _CODE_EXTS)
        targets=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in exts and '.venv' not in p.parts and 'node_modules' not in p.parts and '.git' not in p.parts and '__pycache__' not in p.parts]
    if len(targets)>1000:return {'error':f'too many files to scan ({len(targets)}); pass glob= or exts= to narrow'}
    pat=re.compile(r'\b'+re.escape(old)+r'\b')
    files_changed=[];total_replacements=0
    for fp in targets:
        try:src=fp.read_text(encoding='utf-8',errors='replace')
        except Exception:continue
        if old not in src:continue
        new_src,n=pat.subn(new,src)
        if n>0:
            files_changed.append({'path':str(fp.relative_to(root)),'replacements':n})
            total_replacements+=n
            if not dry_run:
                try:fp.write_text(new_src,encoding='utf-8')
                except Exception as e:files_changed[-1]['write_error']=str(e)[:200]
    return {'old':old,'new':new,'files_scanned':len(targets),'files_changed':len(files_changed),'total_replacements':total_replacements,'changes':files_changed[:30],'dry_run':dry_run}
def _skill_prune_sessions(args,ctx,reg):
    older_than_days=int(args.get('older_than_days',30));keep_n=int(args.get('keep_n',50));dry_run=bool(args.get('dry_run',False))
    conv_root=Path(reg.workdir)/'experiences'/'conversations'
    if not conv_root.exists():
        for cand in (Path.cwd()/'experiences'/'conversations',Path('experiences/conversations')):
            if cand.exists():conv_root=cand;break
    if not conv_root.exists():return {'error':'no conversations directory found'}
    files=sorted(conv_root.glob('*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
    cutoff=time.time()-(older_than_days*86400)
    keep=set(str(p) for p in files[:keep_n])
    candidates=[p for p in files[keep_n:] if p.stat().st_mtime<cutoff and str(p) not in keep]
    freed=sum(p.stat().st_size for p in candidates);deleted=[]
    if not dry_run:
        for p in candidates:
            try:p.unlink();deleted.append(p.name)
            except Exception:pass
    return {'total_sessions':len(files),'kept_recent':min(keep_n,len(files)),'candidates_for_prune':len(candidates),'deleted':len(deleted),'freed_bytes':freed,'freed_mb':round(freed/(1024*1024),2),'dry_run':dry_run,'older_than_days':older_than_days,'keep_n':keep_n,'samples':[p.name for p in candidates[:5]]}
def _skill_export_session(args,ctx,reg):
    """Dump a conversation session as markdown. Args: {session_id, out_path?, format?}"""
    import json as _j
    sid=args.get('session_id')
    if not sid:return {'error':'missing session_id'}
    fmt=(args.get('format') or 'markdown').lower()
    conv_root=Path(reg.workdir)/'experiences'/'conversations'
    if not conv_root.exists():
        for cand in (Path.cwd()/'experiences'/'conversations',Path('experiences/conversations')):
            if cand.exists():conv_root=cand;break
    fp=conv_root/f'{sid}.jsonl'
    if not fp.exists():return {'error':f'session {sid!r} not found at {fp}'}
    try:lines=fp.read_text(encoding='utf-8').strip().splitlines()
    except Exception as e:return {'error':f'read failed: {e}'}
    turns=[]
    for ln in lines:
        try:turns.append(_j.loads(ln))
        except Exception:continue
    if fmt=='json':
        out_text=_j.dumps(turns,indent=2,ensure_ascii=False)
    elif fmt=='text':
        out_text='\n\n'.join(f"[{t.get('role','?')}] {t.get('content','')}" for t in turns)
    else:
        lines_md=[f"# Conversation `{sid}`",f"_{len(turns)} turns from {fp.name}_\n"]
        for t in turns:
            role=t.get('role','?');content=(t.get('content') or '').strip()
            meta_bits=[f"tier={t['tier']}"] if t.get('tier') else []
            if t.get('persona'):meta_bits.append(f"persona={t['persona']}")
            if t.get('is_private'):meta_bits.append('private')
            if t.get('blocked'):meta_bits.append('blocked')
            meta=(' · '.join(meta_bits))
            if role=='user':lines_md.append(f"\n## 👤 user");lines_md.append(content)
            elif role=='assistant':lines_md.append(f"\n## 🤖 {t.get('persona','assistant')}"+(f" · *{meta}*" if meta else ''));lines_md.append(content)
            else:lines_md.append(f"\n## {role}\n{content}")
        out_text='\n'.join(lines_md)
    out_path=args.get('out_path')
    if out_path:
        gate_res=_gate_path({'path':out_path},ctx,reg)
        if gate_res:return {'error':gate_res}
        Path(out_path).write_text(out_text,encoding='utf-8')
        return {'session_id':sid,'turns':len(turns),'format':fmt,'path':out_path,'bytes':len(out_text)}
    return {'session_id':sid,'turns':len(turns),'format':fmt,'content':out_text[:8000],'truncated':len(out_text)>8000}
def _skill_parse_error(args,ctx,reg):
    """Parse a compiler/runtime error message and identify the root cause + likely fix."""
    txt=args.get('text') or args.get('error','')
    if not txt:return {'error':'missing text/error'}
    result={'language':None,'kind':None,'file':None,'line':None,'message':None,'likely_cause':None,'suggested_fix':None}
    pat_py=re.search(r'File "([^"]+)", line (\d+)(?:, in (\w+))?\s*\n[^\n]*\n\s*(\w+(?:Error|Exception|Warning))\s*:\s*(.+?)(?:\n|$)',txt,re.DOTALL)
    if pat_py:
        result.update({'language':'Python','file':pat_py.group(1),'line':int(pat_py.group(2)),'function':pat_py.group(3),'kind':pat_py.group(4),'message':pat_py.group(5).strip()})
    else:
        m=re.search(r'(\w+(?:Error|Exception))\s*:\s*(.+?)(?:\n|$)',txt)
        if m:result.update({'language':'Python','kind':m.group(1),'message':m.group(2).strip()})
    if not result['language']:
        m=re.search(r'error\[(E\d+)\]:\s*(.+?)\n.*?-->\s*([^\s:]+):(\d+):(\d+)',txt,re.DOTALL)
        if m:result.update({'language':'Rust','kind':m.group(1),'message':m.group(2).strip(),'file':m.group(3),'line':int(m.group(4)),'col':int(m.group(5))})
    if not result['language']:
        m=re.search(r'(\w+(?:Error)?):\s*(.+?)\n\s*at\s+[^\s]+\s+\(([^:)]+):(\d+):(\d+)\)',txt)
        if m:result.update({'language':'JavaScript','kind':m.group(1),'message':m.group(2).strip(),'file':m.group(3),'line':int(m.group(4)),'col':int(m.group(5))})
    if not result['language']:
        m=re.search(r'([^:]+\.go):(\d+):(\d+):\s*(.+?)(?:\n|$)',txt)
        if m:result.update({'language':'Go','file':m.group(1),'line':int(m.group(2)),'col':int(m.group(3)),'message':m.group(4).strip()})
    _CAUSE_PATTERNS={
        r'NameError.*not defined':('undefined name','add an import or check spelling'),
        r'AttributeError.*has no attribute':('wrong attribute or wrong object type','check the object type or the attribute name'),
        r'TypeError.*missing.*required.*argument':('missing function argument','add the missing argument to the call'),
        r'TypeError.*takes.*positional argument':('argument count mismatch','check the function signature'),
        r'ImportError|ModuleNotFoundError':('missing module','pip install the package or fix the import path'),
        r'IndentationError|TabError':('Python indentation','fix indentation (4 spaces, no tabs)'),
        r'SyntaxError.*invalid syntax':('Python syntax error','check for unmatched brackets, missing colons, or stray characters'),
        r'IndexError.*out of range':('list/array index out of bounds','add a length check before indexing'),
        r'KeyError':('dict key missing','use .get(key, default) or check membership first'),
        r'ZeroDivisionError':('division by zero','add a check that the divisor is non-zero'),
        r'RecursionError':('infinite recursion','add a base case or increase sys.setrecursionlimit'),
        r'cannot find function|cannot find type|cannot find macro|unresolved import':('Rust missing import','add use statement for the missing symbol'),
        r'mismatched types':('Rust type mismatch','check the function signature or add an explicit conversion'),
        r'borrow.*cannot be|borrowed.*moved':('Rust borrow checker','clone the value or restructure to avoid concurrent borrows'),
        r'cannot move out of':('Rust ownership move','use a reference, clone, or restructure to avoid move-after-borrow'),
        r'TS\d{4}':('TypeScript type error','check types and interface declarations'),
        r'cannot read propert(?:y|ies) of (?:null|undefined)':('JS null/undefined access','add a null check before accessing the property'),
        r'undefined: ':('Go undefined symbol','import the package or check spelling'),
    }
    for pat,(cause,fix) in _CAUSE_PATTERNS.items():
        if re.search(pat,txt,re.IGNORECASE):result['likely_cause']=cause;result['suggested_fix']=fix;break
    return {k:v for k,v in result.items() if v is not None}
def _skill_auto_import(args,ctx,reg):
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=reg._abs(path)
    if not p.exists():return {'error':f'file not found: {path}'}
    if p.suffix.lower()!='.py':return {'error':'only .py supported currently'}
    src=p.read_text(encoding='utf-8')
    try:tree=ast.parse(src)
    except SyntaxError as e:return {'error':f'parse error at line {e.lineno}: {e.msg}'}
    existing_imports=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            for n in node.names:existing_imports.add(n.asname or n.name)
        elif isinstance(node,ast.ImportFrom):
            for n in node.names:existing_imports.add(n.asname or n.name)
    referenced_names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load):referenced_names.add(node.id)
        elif isinstance(node,ast.Attribute):
            base=node
            while isinstance(base,ast.Attribute):base=base.value
            if isinstance(base,ast.Name):referenced_names.add(base.id)
    _BUILTINS={'print','len','range','str','int','float','bool','list','dict','tuple','set','frozenset','type','isinstance','issubclass','object','None','True','False','self','cls','enumerate','zip','map','filter','sorted','reversed','min','max','sum','abs','round','pow','open','input','iter','next','any','all','hash','id','repr','format','vars','dir','getattr','setattr','hasattr','delattr','globals','locals','super','property','staticmethod','classmethod','Exception','ValueError','TypeError','KeyError','IndexError','RuntimeError','OSError','FileNotFoundError','ImportError','AttributeError','NotImplementedError','StopIteration','ZeroDivisionError','NameError','Warning','DeprecationWarning'}
    _STDLIB_HINTS={'os':'os','sys':'sys','re':'re','json':'json','time':'time','math':'math','random':'random','datetime':'datetime','Path':'from pathlib import Path','defaultdict':'from collections import defaultdict','Counter':'from collections import Counter','OrderedDict':'from collections import OrderedDict','deque':'from collections import deque','dataclass':'from dataclasses import dataclass','field':'from dataclasses import field','asdict':'from dataclasses import asdict','partial':'from functools import partial','wraps':'from functools import wraps','lru_cache':'from functools import lru_cache','cache':'from functools import cache','reduce':'from functools import reduce','chain':'from itertools import chain','combinations':'from itertools import combinations','permutations':'from itertools import permutations','product':'from itertools import product','islice':'from itertools import islice','Optional':'from typing import Optional','List':'from typing import List','Dict':'from typing import Dict','Tuple':'from typing import Tuple','Any':'from typing import Any','Union':'from typing import Union','Callable':'from typing import Callable','Iterator':'from typing import Iterator','Sequence':'from typing import Sequence','Mapping':'from typing import Mapping'}
    undefined=referenced_names-existing_imports-_BUILTINS
    suggestions=[]
    for name in sorted(undefined):
        if name in _STDLIB_HINTS:suggestions.append({'name':name,'import':_STDLIB_HINTS[name]})
    return {'path':str(p),'undefined_references':sorted(undefined),'suggested_imports':suggestions,'note':'Apply manually via file_write or code_edit. AST-based detection only catches stdlib hints currently.'}
def _skill_format_code(args,ctx,reg):
    import shutil as _sh
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=reg._abs(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    ext=p.suffix.lower()
    if ext not in _FORMATTERS:return {'skipped':True,'reason':f'no formatter for {ext}'}
    for tool_cmd,binary in _FORMATTERS[ext]:
        if _sh.which(binary):
            try:
                r=subprocess.run(tool_cmd.split()+[str(p)],capture_output=True,text=True,timeout=15,cwd=str(reg.workdir))
                return {'path':str(p),'formatter':binary,'returncode':r.returncode,'stdout':r.stdout[:500],'stderr':r.stderr[:500]}
            except Exception as e:return {'error':f'{type(e).__name__}: {e}','formatter':binary}
    return {'skipped':True,'reason':f'no formatter installed for {ext} (tried: {[b for _,b in _FORMATTERS[ext]]})'}
def _skill_tts(args,ctx,reg):
    try:from amni.voice import speak,tts_backend,list_voices
    except Exception as e:return {'error':f'voice module import failed: {e}'}
    text=args.get('text','')
    if args.get('list_voices'):return {'backend':tts_backend(),'voices':list_voices()[:30]}
    if not text:return {'error':'missing text'}
    _voice=args.get('voice');_persona_key=None
    _ps=ctx.get('personas') or (getattr(ctx.get('agent'),'personas',None)) or (getattr(ctx.get('adam'),'personas',None))
    try:
        if _ps is not None:
            _sid=args.get('session_id') or args.get('sid')
            _name=None
            if _sid and hasattr(_ps,'session_persona'):_name=_ps.session_persona(_sid)
            if not _name and hasattr(_ps,'_default'):_name=_ps._default
            _cur=_ps.get(_name) if _name else None
            if _cur:
                _persona_key=(_cur.name or '').lower()
                if not _voice and hasattr(_cur,'tts_voice'):_voice=_cur.tts_voice
    except Exception as _pe:print(f'[tts] persona lookup: {_pe}',flush=True)
    audio=speak(text,backend=args.get('backend'),voice=_voice,persona=_persona_key)
    if not audio:return {'error':'TTS produced no audio','backend':tts_backend()}
    out_path=args.get('out_path')
    if out_path:
        gate_res=_gate_path({'path':out_path},ctx,reg)
        if gate_res:return {'error':gate_res}
        Path(out_path).write_bytes(audio)
        return {'path':out_path,'bytes':len(audio),'backend':tts_backend(),'voice_used':_voice}
    import base64
    return {'audio_base64':base64.b64encode(audio).decode('ascii'),'bytes':len(audio),'backend':tts_backend(),'mime':'audio/wav','voice_used':_voice}
def _skill_stt(args,ctx,reg):
    try:from amni.voice import transcribe,stt_backend
    except Exception as e:return {'error':f'voice module import failed: {e}'}
    audio=None
    if args.get('path'):
        gate_res=_gate_path({'path':args['path']},ctx,reg)
        if gate_res:return {'error':gate_res}
        try:audio=Path(args['path']).read_bytes()
        except Exception as e:return {'error':f'read failed: {e}'}
    elif args.get('audio_base64'):
        import base64
        try:audio=base64.b64decode(args['audio_base64'])
        except Exception as e:return {'error':f'base64 decode failed: {e}'}
    if not audio:return {'error':'missing path or audio_base64','backend':stt_backend()}
    return transcribe(audio,backend=args.get('backend'),model_size=args.get('model_size','base'))
def _skill_symbols(args,ctx,reg):
    path=args.get('path')
    if not path:return {'error':'missing path'}
    p=Path(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    src=p.read_text(encoding='utf-8',errors='replace')
    ext=p.suffix.lower();out={'path':str(p),'lang':None,'functions':[],'classes':[],'imports':[]}
    if ext=='.py':
        out['lang']='Python'
        try:
            tree=ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node,ast.FunctionDef) or isinstance(node,ast.AsyncFunctionDef):
                    out['functions'].append({'name':node.name,'line':node.lineno,'args':[a.arg for a in node.args.args]})
                elif isinstance(node,ast.ClassDef):
                    methods=[n.name for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                    out['classes'].append({'name':node.name,'line':node.lineno,'methods':methods[:20]})
                elif isinstance(node,ast.Import):
                    out['imports'].extend(n.name for n in node.names)
                elif isinstance(node,ast.ImportFrom):
                    out['imports'].append(f"{node.module}.{','.join(n.name for n in node.names)}")
        except SyntaxError as e:return {'error':f'parse error at line {e.lineno}: {e.msg}'}
    elif ext=='.rs':
        out['lang']='Rust'
        for m in re.finditer(r'^\s*(?:pub\s+)?fn\s+(\w+)\s*[<(]',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r'^\s*(?:pub\s+)?struct\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'struct'})
        for m in re.finditer(r'^\s*(?:pub\s+)?enum\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'enum'})
        for m in re.finditer(r'^\s*(?:pub\s+)?trait\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln,'kind':'trait'})
        for m in re.finditer(r'^\s*use\s+([\w:{}*,\s]+);',src,re.MULTILINE):
            out['imports'].append(m.group(1).strip())
    elif ext in ('.js','.ts','.jsx','.tsx','.mjs'):
        out['lang']='JavaScript/TypeScript'
        for m in re.finditer(r'^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r'^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['functions'].append({'name':m.group(1),'line':ln,'kind':'arrow'})
        for m in re.finditer(r'^\s*(?:export\s+)?class\s+(\w+)',src,re.MULTILINE):
            ln=src[:m.start()].count('\n')+1
            out['classes'].append({'name':m.group(1),'line':ln})
        for m in re.finditer(r"^\s*import\s+(.+?)\s+from\s+['\"](.+?)['\"]",src,re.MULTILINE):
            out['imports'].append(f"{m.group(1).strip()} from {m.group(2)}")
    else:return {'lang':None,'note':f'no symbol extractor for {ext} — use file_read instead'}
    return out
def _skill_project_info(args,ctx,reg):
    p=Path(reg.workdir);info={'workdir':str(p),'top_files':[],'dependencies':{},'languages':set(),'git':{}}
    try:
        for f in sorted(p.iterdir()):
            if f.is_file() and not f.name.startswith('.') and len(info['top_files'])<25:info['top_files'].append(f.name)
    except Exception:pass
    _LANG_EXT={'.py':'Python','.rs':'Rust','.js':'JavaScript','.ts':'TypeScript','.tsx':'TypeScript','.jsx':'JavaScript','.go':'Go','.cpp':'C++','.cc':'C++','.c':'C','.h':'C/C++','.hpp':'C++','.java':'Java','.kt':'Kotlin','.rb':'Ruby','.php':'PHP','.swift':'Swift','.cs':'C#','.scala':'Scala','.sh':'Shell','.html':'HTML','.css':'CSS','.sql':'SQL'}
    for f in p.rglob('*'):
        if f.is_file():
            s=f.suffix.lower()
            if s in _LANG_EXT:info['languages'].add(_LANG_EXT[s])
        if len(info['languages'])>=8:break
    info['languages']=sorted(info['languages'])
    for dep_file,key in (('Cargo.toml','cargo'),('package.json','npm'),('pyproject.toml','python'),('requirements.txt','pip'),('go.mod','go'),('Gemfile','ruby'),('pom.xml','maven'),('build.gradle','gradle'),('CMakeLists.txt','cmake'),('Makefile','make')):
        if (p/dep_file).exists():
            try:info['dependencies'][key]=(p/dep_file).read_text(encoding='utf-8',errors='replace')[:1500]
            except Exception:pass
    try:
        r=subprocess.run(['git','rev-parse','--abbrev-ref','HEAD'],capture_output=True,text=True,timeout=4,cwd=str(p))
        if r.returncode==0:info['git']['branch']=r.stdout.strip()
        r2=subprocess.run(['git','status','--porcelain'],capture_output=True,text=True,timeout=4,cwd=str(p))
        if r2.returncode==0:
            ch=r2.stdout.strip().splitlines();info['git']['dirty_files']=len(ch);info['git']['changes_preview']=ch[:10]
    except Exception:pass
    return info
def _gate_diff(args,ctx,reg):
    g=_gate_path(args,ctx,reg)
    if g:return g
    _wp=_write_protected(args.get('path',''))
    if _wp:return f'refused: {_wp} is write-protected (set AMNI_ALLOW_LAW_EDIT/AMNI_ALLOW_SECURITY_EDIT=1 to override)'
    return None
def _skill_code_diff(args,ctx,reg):
    path=args.get('path');diff=args.get('diff') or args.get('patch')
    if not path:return {'error':'missing path'}
    if not diff:return {'error':'missing diff/patch (unified diff format with @@ hunks)'}
    p=reg._abs(path)
    if not p.exists():return {'error':f'file does not exist: {path}'}
    try:src=p.read_text(encoding='utf-8')
    except Exception as e:return {'error':f'read failed: {e}'}
    new,err=_apply_unified_diff(src,diff)
    if err:return {'error':err,'path':str(p)}
    if args.get('dry_run'):return {'path':str(p),'dry_run':True,'preview_first_200':new[:200],'old_bytes':len(src),'new_bytes':len(new)}
    p.write_text(new,encoding='utf-8')
    return {'path':str(p),'old_bytes':len(src),'new_bytes':len(new),'applied':True}
def _detect_test_runner(workdir):
    p=Path(workdir)
    if (p/'Cargo.toml').exists():return ('cargo test --quiet','rust')
    if (p/'package.json').exists():
        try:
            import json as _j
            d=_j.loads((p/'package.json').read_text(encoding='utf-8'))
            if 'scripts' in d and 'test' in d['scripts']:return ('npm test --silent','js')
        except Exception:pass
    if (p/'pyproject.toml').exists() or (p/'pytest.ini').exists() or (p/'tox.ini').exists() or any(p.glob('test_*.py')) or any(p.glob('tests')):return ('pytest -x --tb=short -q','python')
    if (p/'go.mod').exists():return ('go test ./...','go')
    if (p/'Makefile').exists():
        try:
            mk=(p/'Makefile').read_text(encoding='utf-8',errors='replace')
            if re.search(r'^test\s*:',mk,re.MULTILINE):return ('make test','make')
        except Exception:pass
    return (None,None)
def _skill_test_run(args,ctx,reg):
    explicit=args.get('cmd');timeout=int(args.get('timeout',60))
    if explicit:cmd=explicit;flavor='custom'
    else:
        cmd,flavor=_detect_test_runner(reg.workdir)
        if not cmd:return {'error':'no test runner detected. Pass cmd= explicitly. Tried: Cargo.toml/package.json/pyproject.toml/go.mod/Makefile'}
    try:
        r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=timeout,cwd=str(reg.workdir))
        passed=r.returncode==0
        out=(r.stdout or '')[-3000:];err=(r.stderr or '')[-1500:]
        return {'cmd':cmd,'flavor':flavor,'passed':passed,'returncode':r.returncode,'stdout':out,'stderr':err}
    except subprocess.TimeoutExpired:return {'error':f'tests exceeded {timeout}s timeout','cmd':cmd,'flavor':flavor,'passed':False}
    except Exception as e:return {'error':f'{type(e).__name__}: {e}','cmd':cmd,'flavor':flavor,'passed':False}
_DANGEROUS_PYTHON=re.compile(r'\b(?:os\.system|subprocess\.|os\.remove|os\.unlink|os\.rmdir|shutil\.rmtree|__import__\([\'"]os|exec\s*\(|eval\s*\(|open\s*\([^)]*[\'"]w|requests\.|urllib\.|socket\.|os\.environ\[|os\.setuid|os\.fork)\b')
def _skill_run_python(args,ctx,reg):
    code=args.get('code') or args.get('expr','')
    timeout=int(args.get('timeout',8))
    if not code:return {'error':'no code provided'}
    if _DANGEROUS_PYTHON.search(code):return {'error':'rejected: code contains potentially dangerous operations (filesystem mutation, network, exec/eval, subprocess)'}
    try:from amni.serve.self_debug import run_in_sandbox as _sbx
    except Exception:_sbx=None
    if _sbx is None:return {'error':'sandbox unavailable'}
    res=_sbx(code,timeout=timeout)
    if res.get('blocked'):return {'error':'rejected (AST danger scan): '+', '.join(sorted({d['rule'] for d in res.get('dangers',[])})),'dangers':res.get('dangers',[])[:6]}
    if not res.get('ran'):return {'error':res.get('error','sandbox failed to run')}
    return {'stdout':(res.get('stdout') or '')[:6000],'stderr':(res.get('stderr') or '')[:2000],'returncode':res.get('returncode'),'timed_out':bool(res.get('timed_out')),'killed':res.get('killed'),'capped':bool(res.get('capped'))}
def _extract_synthetic_q(filename:str,chunk:str)->str:
    h=re.search(r'^#+\s+(.+?)$',chunk,re.MULTILINE)
    if h:return f'What does "{h.group(1).strip()}" describe in {filename}?'
    cls_or_def=re.search(r'^(?:class|def)\s+([A-Za-z_]\w*)',chunk,re.MULTILINE)
    if cls_or_def:return f'What is {cls_or_def.group(1)} in {filename}?'
    first_sent=re.match(r'^[^.\n]{20,180}[.!?]',chunk.strip())
    if first_sent:return f'In {filename}, what does this say: "{first_sent.group(0).strip()[:120]}..."?'
    first_words=' '.join(chunk.split()[:8])
    return f'What does {filename} say about "{first_words}"?'
def _chunk_text(text:str,max_chars:int=1500)->List[str]:
    paras=[p.strip() for p in re.split(r'\n\s*\n',text) if p.strip()]
    out=[];buf=''
    for p in paras:
        if len(buf)+len(p)+2<=max_chars:buf=f'{buf}\n\n{p}' if buf else p
        else:
            if buf:out.append(buf)
            if len(p)<=max_chars:buf=p
            else:
                for i in range(0,len(p),max_chars):out.append(p[i:i+max_chars])
                buf=''
    if buf:out.append(buf)
    return out
def _iter_files(root:Path,glob:str,max_files:int,exts:Optional[set])->List[Path]:
    if root.is_file():return [root][:max_files]
    out=[]
    for p in root.rglob(glob):
        if not p.is_file():continue
        if exts is not None and p.suffix.lower() not in exts:continue
        if _secret_blocked(p):continue
        if p.stat().st_size>2_000_000:continue
        out.append(p)
        if len(out)>=max_files:break
    return out
def _skill_scan(args,ctx,reg):
    adam=ctx.get('adam')
    if adam is None:return {'error':'scan requires Adam (no ctx adam)'}
    p=args.get('path')
    if not p:return {'error':'missing path arg'}
    root=reg._abs(p).resolve()
    if not root.exists():return {'error':f'path not found: {p}'}
    glob=args.get('glob','**/*');max_files=int(args.get('max_files',50));max_chars_per_file=int(args.get('max_chars_per_file',8000));distill=bool(args.get('distill',False));only_text=bool(args.get('only_text',True))
    exts=_TEXT_EXT if only_text else None
    files=_iter_files(root,glob,max_files,exts)
    if not files:return {'files_scanned':0,'lessons_added':0,'note':'no matching files'}
    n_lessons_before=len(adam.sem_lut._raw) if adam.sem_lut is not None else 0
    errors=[];scanned=[];pending=[]
    for f in files:
        try:txt=f.read_text(encoding='utf-8',errors='replace')[:max_chars_per_file]
        except Exception as e:errors.append({'file':str(f),'error':str(e)});continue
        chunks=_chunk_text(txt,max_chars=1500)
        for i,ch in enumerate(chunks):
            try:
                from amni.serve.federated import scrub_pii as _sp
                from amni.serve.code_safety import sanitize_ingest as _si
                ch=_si(_sp(ch)[0])[0]
            except Exception:pass
            q=None
            if distill:
                try:
                    sys_p='Generate ONE concise question whose answer is found in the text below. Format: "Q: ..."'
                    qresp=adam.adam.svc.chat(f'Text:\n{ch[:1200]}',system=sys_p,max_new_tokens=40,do_sample=False,kb_top_k=0)
                    qline=(qresp[0] if isinstance(qresp,tuple) else qresp).strip()
                    q=re.sub(r'^Q:\s*','',qline,flags=re.IGNORECASE).strip() or None
                except Exception as e:errors.append({'file':str(f),'chunk':i,'distill_error':str(e)})
            if not q:q=_extract_synthetic_q(f.name,ch)
            pending.append((q,ch))
        scanned.append({'file':str(f),'chunks':len(chunks),'bytes':len(txt)})
    if adam.sem_lut is not None and pending:
        for q,a in pending:adam.sem_lut.add(q,a)
        try:adam.sem_lut.fit()
        except Exception as e:errors.append({'fit_error':str(e)})
        try:adam.save_lessons()
        except Exception as e:errors.append({'save_error':str(e)})
    n_lessons_after=len(adam.sem_lut._raw) if adam.sem_lut is not None else 0
    return {'files_scanned':len(scanned),'lessons_added':n_lessons_after-n_lessons_before,'lessons_total':n_lessons_after,'distilled':distill,'errors':errors[:10],'files':scanned[:20],'bulk_fit':True}
def _skill_options(args,ctx,reg):
    tk=re.sub(r'[^A-Z.]','',str(args.get('ticker') or args.get('symbol') or args.get('query') or '').strip().upper().split(' ')[0])
    if not tk:return {'error':'need a ticker, e.g. {ticker:"NVDA"}'}
    tf=str(args.get('tf') or args.get('timeframe') or '1h').strip();lb=str(int(args.get('lookback',24)))
    PY=os.environ.get('AZNO_PY',r'C:\Users\antho\Documents\ai\azno-v2\.venv\Scripts\python.exe');CWD=os.environ.get('AZNO_DIR',r'C:\Users\antho\Documents\ai\azno-v2')
    try:
        p=subprocess.run([PY,os.path.join('tools','options_signal.py'),tk,tf,lb],cwd=CWD,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,env={**os.environ,'PYTHONIOENCODING':'utf-8'})
        o=(p.stdout or '').strip()
        return json.loads(o[o.index('{'):o.rindex('}')+1]) if ('{' in o and '}' in o) else {'error':'options engine produced no card','detail':(p.stderr or o)[:300]}
    except Exception as e:return {'error':f'options: {type(e).__name__}: {e}'}
def _fmt_options(c)->str:
    if not isinstance(c,dict) or not c.get('ok'):return 'Options: '+str((c or {}).get('error','no data / unknown ticker'))
    tk=c.get('ticker');tf=c.get('tf');px=c.get('price');h=c.get('history') or {}
    if not c.get('signal') or not c.get('options'):return f"{tk} {tf} @ ${px}: no fresh P-term reversal right now — no directional options setup. (backtest {int((h.get('win_rate') or 0)*100)}% hit-rate, {h.get('avg_swing_pct')}% avg swing.)"
    return (c.get('example') or '')+' — '+(c.get('verdict') or '')+'. '+(c.get('disclaimer') or '')
def options_command(text:str):
    m=re.match(r'(?i)^\s*/?(?:opt|opts|option|options)\b[\s:]*([A-Za-z.]{1,6})(?:[\s:]+([0-9]+[a-z]+|[a-z]+))?\s*$',text or '')
    return (m.group(1).upper(),(m.group(2) or '1h')) if m else None
_CHART_STOP={'send','me','the','a','an','for','of','my','our','please','pls','show','get','gimme','give','chart','charts','plot','graph','option','options','opt','opts','signal','signals','and','to','on','in','with','pull','up','can','could','you','i','want','wanna','see','latest','about','info','how','is','whats','what','do','does','did','adam','azno','us','here','there','this','that','now','today','some','any','all','help','stuff','things','price','are','was','were','be','or','if','it','at','by','as','we','they','will','new','good','best','data','sell','buy','hold','call','put','long','short','trade','live','real','need','look','both','tell','your','them','from','yes','hey','hi','thx','thanks','ok','okay','cool','nice','well','just','also','only','make','take','come','know','like','more','less','over','into','out','off','per','via','than','then','when','why','who','much','many','lots','current','give','got','check','checks','checking','market','markets','status','update','updates','news','hello','howdy','hiya','morning','afternoon','evening','tonight','yesterday','tomorrow','holdings','holding','portfolio','position','positions','account','balance','funds','money','watch','watchlist','again','back','soon','later','maybe','sure','right','wrong','done','start','open','close','high','low','last','first','next','value','worth','total','gain','loss','gains','losses','profit','risk','sorry','still','which','where','been','have','has','had','would','should','something','anything','nothing'}
def _chart_out_dir(sub='charts'):
    d=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'json',sub);os.makedirs(d,exist_ok=True);return d
def _extract_ticker(text):
    best=None
    for tok in re.findall(r'\$?\b[A-Za-z]{1,5}\b',text or ''):
        dollar=tok.startswith('$');w=tok.lstrip('$');up=w.isupper() and len(w)>=2
        if not (dollar or up) and w.lower() in _CHART_STOP:continue
        score=3 if dollar else (2 if w.isupper() else 1)
        if best is None or score>best[0]:best=(score,w.upper())
    return best[1] if best else None
def _extract_tf(text):
    m=re.search(r'(?i)\b(\d+)\s?(m|min|h|hr|hour|d|day|w|wk)\b',text or '')
    return f"{m.group(1)}{ {'m':'m','min':'m','h':'h','hr':'h','hour':'h','d':'d','day':'d','w':'w','wk':'w'}[m.group(2).lower()] }" if m else '1h'
def chart_command(text):
    t=(text or '').strip()
    if not re.search(r'(?i)\b(chart|charts|plot|graph|option|options|opt|opts)\b',t):return None
    tk=_extract_ticker(t)
    return (tk,_extract_tf(t),bool(re.search(r'(?i)\boption',t))) if tk else None
_TICKER_XSTOP={'LOL','OMG','WTF','BRB','IDK','IMO','IMHO','TBH','FYI','ASAP','LMAO','ROFL','SMH','NVM','THX','PLZ','BTW','GG','GL','GM','GN','GO','IRL','JK','NP','TY','YW','YEP','NOPE','NAH','WOW','HMM','HA','AH','OH','UM','EH','YO','SUP','WAIT','TEST','STOP','NO','SO','BUT','NOT','WHY','WHO','USA','USD','EUR','CEO','CTO','CFO','GPU','CPU','API','URL','FAQ','DIY','ETA','DNA','RIP','TV','PC','AI','FBI','NASA','HTML','JSON','HTTP','PIN','ID','AM','PM','EST','PST','UTC','GMT','CET','LMK','FWIW','IIRC','AFAIK','PDF','PNG','JPG','GIF','SQL','CSS','CSV','RAM','ROM','SSD','USB','WIFI','GPS','LED','LCD','TAPE','WAY','DUDE','BRO','MAN','SIR','MAAM','MEANT','MEAN','TABLE','ERROR','NAME','FILE','CODE','WORD'}|{'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','DC'}
def ticker_intent(text):
    t=(text or '').strip()
    def _ok(w):return bool(w) and w.upper() not in _TICKER_XSTOP and w.lower() not in _CHART_STOP
    m=re.match(r'^\$?([A-Z]{1,5})(?:\s+(\d+\s?(?:m|min|h|hr|hour|d|day|w|wk)))?\s*[?.!]*$',t)
    if m and (t.startswith('$') or m.group(2) or len(m.group(1))>=2) and _ok(m.group(1)):return (m.group(1),_extract_tf(t))
    for _p in (r'(?i:\b(?:data|info|information|stats|numbers|analysis|read|signal)\s+(?:on|for|about)\s+)\$?([A-Z]{1,5})\b',r'(?i:\b(?:ticker|symbol)\b[\s:,]*)\$?([A-Z]{1,5})\b',r'(?i:\bstock\b[\s:,]*)\$?([A-Z]{1,5})\b',r'\$?\b([A-Z]{1,5})\b(?i:\s+(?:is\s+)?(?:a\s+|the\s+)?(?:ticker|stock|symbol)\b)',r'(?i:\b(?:i\s+meant|i\s+mean|meant|referring\s+to|talking\s+about|asking\s+about)\s+(?:the\s+)?(?:ticker\s+|stock\s+|symbol\s+|company\s+)?)\$?([A-Z]{1,5})\b',r'(?i:\b(?:i\s+meant|i\s+mean|meant|referring\s+to)\b.{0,30}?\b(?:ticker|symbol|stock)\b[\s:,]*)\$?([A-Za-z]{1,5})\b',r'^(?i:no+[,!\s]+)\$?([A-Z]{2,5})\s*[?.!]*$'):
        m=re.search(_p,t)
        if m and _ok(m.group(1)):return (m.group(1).upper(),_extract_tf(t))
    return None
def _render_azno_charts(ticker,tf,want_options=True):
    PY=os.environ.get('AZNO_PY',r'C:\Users\antho\Documents\ai\azno-v2\.venv\Scripts\python.exe');CWD=os.environ.get('AZNO_DIR',r'C:\Users\antho\Documents\ai\azno-v2')
    outdir=_chart_out_dir();ts=str(int(time.time()));imgs=[];summary='';env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'}
    if want_options:
        op=os.path.join(outdir,f'{ticker}_{tf}_opt_{ts}.png')
        try:
            r=subprocess.run([PY,os.path.join('tools','options_chart.py'),ticker,tf,op,'stock'],cwd=CWD,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=160,env=env)
            lines=[l for l in (r.stdout or '').splitlines() if l.strip()];summary=lines[-1].strip() if lines else ''
            if os.path.exists(op):imgs.append(op)
        except Exception:pass
    pc=os.path.join(outdir,f'{ticker}_{tf}_price_{ts}.png')
    try:
        code='import sys;sys.path.insert(0,r"%s");from core.chart_export import export_chart_png;print(export_chart_png("%s","%s",r"%s",market="stock"))'%(CWD,ticker,tf,pc)
        subprocess.run([PY,'-c',code],cwd=CWD,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,env=env)
        if os.path.exists(pc):imgs.append(pc)
    except Exception:pass
    return {'ok':bool(imgs),'images':imgs,'summary':summary,'ticker':ticker,'tf':tf}
def _skill_chart(args,ctx,reg):
    tk=_extract_ticker(str(args.get('ticker') or args.get('symbol') or args.get('query') or ''))
    if not tk:return {'error':'need a ticker, e.g. {ticker:"TSLA"}'}
    tf=str(args.get('tf') or args.get('timeframe') or '1h').strip();r=_render_azno_charts(tk,tf,want_options=bool(args.get('options',True)))
    return {'ok':True,'ticker':tk,'tf':tf,'images':r['images'],'summary':r.get('summary') or f'{tk} {tf} chart'} if r.get('ok') else {'error':'chart render produced no image','detail':(r.get('summary') or '')[:200]}
_IMG_STOP={'conclusion','conclusions','comparison','comparisons','contrast','contrasts','distinction','distinctions','parallel','parallels','line','lines','blank','blanks','attention','inspiration','breath','up','on','it','that','this','out','near','from','back','away','closer','level','even','straws','lots','fire','blood','crowd','crowds'}
def image_intent(text):
    t=(text or '').strip()
    m=re.match(r'(?i)^/?(?:img|image|imagine|dream)\b[\s:]+(.+)$',t) or re.match(r"(?i)^(?:please\s+|hey\s+)?(?:adam[,\s]+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+)?(?:please\s+)?(?:draw|paint|sketch|illustrate|visualize)\s+(?:me\s+|us\s+)?(?:a\s+|an\s+|the\s+|some\s+)?(.+)$",t) or re.search(r"(?i)\b(?:generate|create|make|produce|render|gen)\s+(?:me\s+|us\s+)?(?:a\s+|an\s+|some\s+)?(?:image|picture|pic|photo|artwork|art|wallpaper|illustration|portrait|drawing|painting|poster|logo)\s*(?:of|for|about|showing|with|depicting|:)?\s*(.*)$",t)
    if not m:return None
    p=(m.group(1) or '').strip(' ?.!,')
    return p if p and p.split()[0].lower() not in _IMG_STOP else None
def _amni_gen_generate(prompt,negative='',steps=8,width=1024,height=1024,seed=-1,use_expansion=True,init_path=None,strength=0.6,timeout=900):
    import requests,base64
    base=os.environ.get('AMNI_GEN_URL','http://127.0.0.1:8763')
    try:requests.get(base+'/health',timeout=4)
    except Exception:return {'error':'amni-gen engine offline — start Amni-Gen (amni-ui) on :8763 and ask again'}
    body={'prompt':prompt,'negative':negative or 'blurry, low quality, bad anatomy, deformed, watermark','steps':int(steps),'guidance':7.0,'width':int(width),'height':int(height),'seed':int(seed),'use_expansion':bool(use_expansion),'save_images':False,'lift_method':'none','backend':'sdxl'}
    if init_path and os.path.exists(str(init_path)):body['init_image_b64']=base64.b64encode(open(init_path,'rb').read()).decode();body['strength']=float(strength)
    last=None
    with requests.post(base+'/generate',json=body,stream=True,timeout=(10,timeout)) as r:
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith('data: '):
                last=json.loads(line[6:])
                if last.get('type') in ('done','error'):break
    if not isinstance(last,dict) or last.get('type')!='done':return {'error':((last or {}).get('message') or 'amni-gen stream ended without a result')[:300]}
    out=os.path.join(_chart_out_dir('gen'),f"gen_{int(time.time())}_{re.sub(r'[^a-z0-9]+','_',prompt.lower())[:40].strip('_')}.png")
    open(out,'wb').write(base64.b64decode(last['image']))
    tm=last.get('timings') or {}
    return {'ok':True,'images':[out],'summary':f"🎨 \"{prompt[:70]}\" — {tm.get('gen_s','?')}s @ {int(width)}x{int(height)}"+(' (img2img)' if init_path else '')}
def _skill_image_gen(args,ctx,reg):
    p=str(args.get('prompt') or args.get('query') or args.get('text') or '').strip()
    if not p:return {'error':'need a prompt, e.g. {prompt:"a sunset over the mountains"}'}
    r=_amni_gen_generate(p,negative=str(args.get('negative') or ''),steps=int(args.get('steps') or 8),width=int(args.get('width') or 1024),height=int(args.get('height') or 1024),seed=int(args.get('seed') or -1),use_expansion=bool(args.get('expand',True)),init_path=args.get('init_path'),strength=float(args.get('strength') or 0.6))
    return {'ok':True,'prompt':p,'images':r['images'],'summary':r['summary']} if r.get('ok') else r
def default_registry(workdir:Optional[str]=None,roots:Optional[List[str]]=None,audit_log:Optional[str]='logs/agent_skill_calls.jsonl',unrestricted:bool=False,with_agentic:bool=True)->SkillRegistry:
    reg=SkillRegistry(workdir=workdir,roots=roots,audit_log=audit_log,unrestricted=unrestricted)
    scope='UNRESTRICTED (all drives)' if unrestricted else f'{len(reg.roots)} root(s)'
    reg.register('time',_skill_time,desc='Get current local time. Args: {}',schema={})
    reg.register('calc',_skill_calc,desc='Exact math via fast-eval + sympy (arithmetic, algebra, calculus, equations, percent, trig, gcd). Args: {expr}. e.g. "7*8", "solve x^2-4=0", "diff x^2", "17 percent of 200".',schema={'expr':'str'})
    reg.register('units',_skill_units,desc='Exact unit conversion (length/mass/temp/time/data/energy/power/pressure/volume/speed/area/angle/frequency/force). Args: {query:"10 km to miles"} or {value,from,to}.',schema={'query':'str?','value':'float?','from':'str?','to':'str?'})
    reg.register('datetime',_skill_datetime,desc='Date/time arithmetic. Args: {query:"days between 2026-01-01 and 2026-06-09"} or {action: diff|add|parse, a?,b?,date?,days?,weeks?,months?,years?,hours?}. Deterministic where weights fail.',schema={'query':'str?','action':'str?','a':'str?','b':'str?','date':'str?','days':'int?','weeks':'int?','months':'int?','years':'int?','hours':'int?','minutes':'int?'})
    reg.register('formula',_skill_formula,desc='Recall physical constants (c,R,F=Faraday,N_A,k_B,h,e...), named equations (nernst,ideal_gas,ohms_law,faraday_electrolysis,arrhenius,compound_interest...), and element data (Fe,Cu,O...). Args: {query}. Pairs with calc: recall the formula, then compute.',schema={'query':'str'})
    reg.register('codec',_skill_codec,desc='Exact encode/decode + hash + uuid + json + regex + base. Args: {op, text, ...}. op: base64_encode/decode, hex_encode/decode, url_encode/decode, hash(algo), uuid, json_pretty/minify/validate, regex(pattern), base(from_base,to_base), bitwise(a,b,bitop).',schema={'op':'str','text':'str?','algo':'str?','pattern':'str?','from_base':'int?','to_base':'int?','a':'int?','b':'int?','bitop':'str?','sort':'bool?'})
    reg.register('stats',_skill_stats,desc='Exact statistics on a number set: mean/median/mode/stdev/variance/min/max/range/sum. Args: {numbers:[...]} or {data:"1,2,3"}; or {x:[],y:[]} for linear fit + correlation.',schema={'numbers':'list?','data':'str?','x':'list?','y':'list?'})
    reg.register('chem',_skill_chem,desc='Exact molar mass + composition from a chemical formula (full periodic table, handles parens + hydrates). Args: {formula}. e.g. "H2O", "C6H12O6", "Ca(OH)2", "CuSO4.5H2O". Returns molar_mass g/mol, composition, percent_by_mass.',schema={'formula':'str'})
    reg.register('finance',_skill_finance,desc='Exact financial math. Args: {op, ...}. op: compound (principal,rate,years,n?), loan/mortgage (principal,rate,years), pct_change (from,to), cagr (begin,end,years), pv (future_value,rate,years). Rates in percent.',schema={'op':'str','principal':'float?','rate':'float?','years':'float?','n':'float?','from':'float?','to':'float?','begin':'float?','end':'float?','future_value':'float?'})
    reg.register('password',_skill_password,desc='Cryptographically-secure password/token generation (secrets). Args: {length?, kind?}. kind: strong (default) | alnum | hex | pin | letters | token (url-safe) | hextoken.',schema={'length':'int?','kind':'str?'})
    reg.register('color',_skill_color,desc='Color conversion + readability. Args: {color:"#1e90ff" or "30,144,255"} or {r,g,b}. Returns hex, rgb, hsl, luminance, on_color (black/white for contrast).',schema={'color':'str?','r':'int?','g':'int?','b':'int?'})
    reg.register('geo',_skill_geo,desc='Great-circle distance + bearing between two lat/lon points (haversine). Args: {lat1,lon1,lat2,lon2}. Returns distance_km, distance_mi, bearing_deg.',schema={'lat1':'float','lon1':'float','lat2':'float','lon2':'float'})
    reg.register('net',_skill_net,desc='IP/subnet calculator. Args: {query} = an IP (192.168.1.10 -> private?/version/reverse-dns) or CIDR (10.0.0.0/24 -> network/broadcast/netmask/usable hosts). IPv4 + IPv6.',schema={'query':'str'})
    reg.register('text',_skill_text,desc='Text utilities. Args: {op, text}. op: stats (char/word/line/sentence count) | upper | lower | title | capitalize | reverse | slug | rot13 | snake | camel.',schema={'op':'str','text':'str'})
    reg.register('random',_skill_random,desc='Secure randomness. Args: {op, ...}. op: dice (n,sides) | coin | int (min,max) | pick (items) | shuffle (items) | sample (items,k) | uuid.',schema={'op':'str','n':'int?','sides':'int?','min':'int?','max':'int?','items':'list?','k':'int?'})
    reg.register('solve',_skill_solve,desc='Recall-and-solve engine: takes a named equation (nernst/ideal_gas/ohms_law/arrhenius/compound_interest/kinematics_v...) or a raw "lhs = rhs", plugs in known {values} (and auto-fills physical constants R,F,c...), and solves for the unknown. Args: {equation|name, values:{var:num}, solve_for?}. e.g. {name:"ohms_law", values:{I:2,R:5}} -> V=10.',schema={'equation':'str?','name':'str?','values':'dict?','solve_for':'str?'})
    reg.register('roman',_skill_roman,desc='Roman numeral <-> integer. Args: {value} (a number -> roman, or roman string -> integer).',schema={'value':'str'})
    reg.register('cipher',_skill_cipher,desc='Classic encode/decode. Args: {op, text, shift?}. op: morse_encode/decode | caesar (shift) | atbash | binary_encode/decode | reverse.',schema={'op':'str','text':'str','shift':'int?'})
    reg.register('numtheory',_skill_numtheory,desc='Exact number theory. Args: {op, n, k?}. op: isprime | factorize | gcd | lcm | ncr | npr | factorial | fib | nextprime | totient | divisors.',schema={'op':'str','n':'int','k':'int?'})
    reg.register('matrix',_skill_matrix,desc='Exact linear algebra (numpy). Args: {op, a:[[...]], b?:[[...]]}. op: multiply | add | transpose | det | inverse | rank | eig | solve (Ax=b).',schema={'op':'str','a':'list','b':'list?'})
    reg.register('calendar',_skill_calendar,desc='Calendar facts. Args: {op, date?|year?|month?|birthdate?}. op: weekday | leap_year | days_in_month | age | day_of_year.',schema={'op':'str','date':'str?','year':'int?','month':'int?','birthdate':'str?'})
    reg.register('validate',_skill_validate,desc='Checksum/format validators. Args: {op, value}. op: luhn (credit card) | email | url | isbn (10/13) | json | ipv4.',schema={'op':'str','value':'str'})
    reg.register('dataformat',_skill_dataformat,desc='Data format conversion. Args: {op, text}. op: csv_to_json | json_to_csv | flatten (nested json -> dotted keys).',schema={'op':'str','text':'str'})
    reg.register('diff',_skill_diff,desc='Unified text diff + similarity between two strings. Args: {a, b, context?}. Returns diff, similarity ratio, added/removed line counts.',schema={'a':'str','b':'str','context':'int?'})
    reg.register('num2words',_skill_num2words,desc='Integer -> English words (e.g. 1234 -> "one thousand two hundred thirty-four"). Args: {n}.',schema={'n':'int'})
    reg.register('resistor',_skill_resistor,desc='Resistor color-band decoder. Args: {bands:["brown","black","red","gold"]} (4 or 5 bands) -> resistance + tolerance.',schema={'bands':'list?','query':'str?'})
    reg.register('quote',_skill_quote,desc='Live market price (no API key). Args: {symbol or query, kind?}. Crypto via CoinGecko (btc/eth/sol...) or FX via exchangerate.host ("usd eur"). Returns price + 24h change.',schema={'symbol':'str?','query':'str?','kind':'str?'})
    reg.register('options',_skill_options,desc='Options-trader read for a STOCK ticker from the azno P-term reversal signal: direction, conviction, historical hit-rate + swing, a horizon-matched expiry, an ATM single-leg AND a defined-risk vertical SPREAD (short leg parked at the P-term target), plus P-term expected move vs options-implied move (Black-Scholes). Informational, at-your-own-risk. Args: {ticker, tf?=1h}. e.g. {ticker:"NVDA"}.',schema={'ticker':'str','tf':'str?','lookback':'int?'})
    reg.register('chart',_skill_chart,desc='Render + SEND a price/signal chart AND an options-analysis chart (PNG images) for a STOCK ticker via the azno P-term engine. Use whenever the user asks to see/send/pull up a chart, plot, graph, or options chart. Args: {ticker, tf?=1h, options?=true}. e.g. {ticker:"TSLA"}. Images are delivered as chat attachments.',schema={'ticker':'str','tf':'str?','options':'bool?'})
    reg.register('image_gen',_skill_image_gen,desc='Generate (or transform via init_path) an AI image from a text prompt with the local Amni-Gen SDXL engine and SEND it as a chat attachment. Use whenever the user asks to draw/paint/sketch/generate/create an image, picture, photo, art, wallpaper, portrait, poster, or logo. Args: {prompt, width?, height?, steps?, seed?, negative?, expand?, init_path?, strength?}. e.g. {prompt:"a golden retriever astronaut on the moon"}.',schema={'prompt':'str','width':'int?','height':'int?','steps':'int?','seed':'int?','negative':'str?','expand':'bool?','init_path':'str?','strength':'float?'})
    reg.register('define',_skill_define,desc='Dictionary lookup (dictionaryapi.dev, no key). Args: {word}. Returns phonetic, part-of-speech, up to 3 definitions.',schema={'word':'str'})
    reg.register('password_strength',_skill_password_strength,desc='Estimate a password\'s entropy + strength rating. Args: {password}. Returns charset size, entropy bits, rating (very weak..very strong).',schema={'password':'str'})
    reg.register('timestamp',_skill_timestamp,desc='Unix epoch <-> date. Args: {value}. A number -> UTC/local ISO; a date string -> unix timestamp. Handles ms timestamps.',schema={'value':'str'})
    reg.register('unicode',_skill_unicode,desc='Unicode inspector. Args: {text} (per-char codepoint+name) or {codepoint:"1F600"} / "U+1F600" -> the char + name + category.',schema={'text':'str?','codepoint':'str?'})
    reg.register('lorem',_skill_lorem,desc='Placeholder text generator. Args: {n, unit?}. unit: words (default) | sentences | paragraphs.',schema={'n':'int?','unit':'str?'})
    reg.register('jwt',_skill_jwt,desc='Decode a JWT (header + payload, no verification). Args: {token}. Flags exp/expired. For inspection only - does NOT validate the signature.',schema={'token':'str'})
    reg.register('url',_skill_url,desc='Parse a URL into scheme/host/port/path/query/fragment. Args: {url}.',schema={'url':'str'})
    reg.register('semver',_skill_semver,desc='Compare two semantic versions. Args: {a, b}. Returns a>b / a==b / a<b.',schema={'a':'str','b':'str'})
    reg.register('cron',_skill_cron,desc='Parse + describe a 5-field cron expression. Args: {expression} e.g. "*/15 9-17 * * 1-5".',schema={'expression':'str'})
    reg.register('translate',_skill_translate,desc='Translate text (mymemory, no key). Args: {text, from?=en, to?=es}. Language codes like en/es/fr/de/ja.',schema={'text':'str','from':'str?','to':'str?'})
    reg.register('regex_explain',_skill_regex_explain,desc='Explain a regex pattern token-by-token, and optionally test it. Args: {pattern, text?}.',schema={'pattern':'str','text':'str?'})
    reg.register('mem',_skill_mem,desc="Query Adam's federated context stores: sem-lut lesson bank (soft+cosine) + ATEX answer-LUT (exact+embed index) + conversation atlas recall; hits tagged per-store. Args: {query, k?, session_id?}",schema={'query':'str','k':'int?','session_id':'str?'})
    def _skill_weight_correct(args,ctx,_reg):
        adam=ctx.get('adam')
        if adam is None:return {'error':'adam unavailable'}
        from amni.serve.correction_trace import CorrectionTrace
        ct=getattr(adam,'_correction_trace',None)
        if ct is None:ct=CorrectionTrace(adam,agent=ctx.get('agent'));adam._correction_trace=ct
        if args.get('action')=='history':return {'history':ct.history(int(args.get('n',20)))}
        p=args.get('prompt') or args.get('question') or '';c=args.get('correct') or args.get('correct_answer') or ''
        if not p or not c:return {'error':'need prompt + correct'}
        return ct.correct(p,args.get('wrong') or '',c,feedback=args.get('feedback') or '',retrain=bool(args.get('retrain',True)),steps=int(args.get('steps',240)))
    reg.register('weight_correct',_skill_weight_correct,desc='Tracer/weight-correction cycle ("toggle a switch"): documents the wrong answer + self-diagnosed reason to the PTEX ledger, flips the instant ATEX switch (MemoryBus record + anti-pattern suppress), identifies the nonce (sem-lut cell), trains a SELECTIVE cosine-gated page at that address (base weights untouched, cos=1), re-asks to verify the flip. Actions: correct {prompt, wrong?, correct, feedback?, retrain?, steps?} | history {n?}',schema={'action':'str?','prompt':'str?','wrong':'str?','correct':'str?','feedback':'str?','retrain':'bool?','steps':'int?','n':'int?'})
    reg.register('web',_skill_web,desc="DDG search + distill via Adam's crawler. Args: {query}",schema={'query':'str'})
    reg.register('file_read',_skill_file_read,gate=_gate_path,desc=f'Read a UTF-8 text file within {scope}. Args: {{path, max_bytes?}}',schema={'path':'str','max_bytes':'int?'})
    reg.register('find',_skill_find,desc=f'Fast substring/regex search across workdir text files. Skips binary + noise dirs (.git/.venv/__pycache__/etc). Args: {{query, regex?, case_sensitive?, glob?, max_hits?, max_chars?}}',schema={'query':'str','regex':'bool?','case_sensitive':'bool?','glob':'str?','max_hits':'int?','max_chars':'int?'})
    def _skill_reminder(args,ctx,reg_):
        """Manage time-aware reminders. Actions: add (text, due_at?, session_id?) | list (session_id?, limit?) | due | dismiss (id) | stats."""
        from amni.serve import reminders as _rm
        action=(args.get('action') or 'list').strip().lower()
        sid=(ctx.get('conv') and getattr(ctx['conv'],'session_id',None)) or args.get('session_id','')
        if action=='add':
            text=args.get('text','')
            if not text:return {'error':'text required'}
            return _rm.add(text=text,due_at=args.get('due_at'),session_id=sid)
        if action=='list':return {'reminders':_rm.list_active(session_id=args.get('session_id') or None,limit=int(args.get('limit',50)))}
        if action=='due':return {'due':_rm.list_due()}
        if action=='dismiss':
            rid=args.get('id','')
            if not rid:return {'error':'id required'}
            return _rm.dismiss(rid)
        if action=='stats':return _rm.stats()
        return {'error':f'unknown action {action!r}; valid: add|list|due|dismiss|stats'}
    reg.register('reminder',_skill_reminder,desc='Time-aware reminders / todo store. Actions: add (text, due_at?) | list (limit?) | due | dismiss (id) | stats. Add auto-parses "in N minutes/hours/days", "tomorrow", "at HH:MM(am|pm)" from the text.',schema={'action':'str?','text':'str?','due_at':'float?','id':'str?','session_id':'str?','limit':'int?'})
    def _skill_recall(args,ctx,reg_):
        """Search past conversation sessions for matching content. Args: query (str), max_sessions?, max_snippets_per_session?."""
        import re as _re,json as _json
        q=str(args.get('query') or '').strip()
        if not q:return {'error':'query required'}
        max_sessions=int(args.get('max_sessions',5));max_snip=int(args.get('max_snippets_per_session',2))
        conv_root=None
        try:
            store=ctx.get('store') or (ctx.get('agent') and ctx['agent'].store)
            if store is not None:conv_root=getattr(store,'root',None)
        except Exception:pass
        if conv_root is None:
            conv_root=Path('experiences/conversations')
        if not conv_root.exists():return {'query':q,'hits':[],'n_hits':0,'sessions_scanned':0}
        pat=_re.compile(_re.escape(q),_re.IGNORECASE)
        cur_sid=None
        try:
            if ctx.get('conv'):cur_sid=ctx['conv'].session_id
        except Exception:pass
        hits=[];sessions_scanned=0
        files=sorted(conv_root.glob('*.jsonl'),key=lambda p:-p.stat().st_mtime)
        for p in files:
            if '.archived.' in p.name:continue
            sid=p.stem
            if sid==cur_sid:continue
            sessions_scanned+=1;snips=[]
            try:lines=p.read_text(encoding='utf-8',errors='ignore').splitlines()
            except Exception:continue
            for ln in lines:
                ln=ln.strip()
                if not ln:continue
                try:t=_json.loads(ln)
                except Exception:continue
                content=str(t.get('content') or '')
                if pat.search(content):
                    role=t.get('role','?');snippet=content[:240]
                    snips.append({'role':role,'text':snippet,'ts':t.get('ts')})
                    if len(snips)>=max_snip:break
            if snips:
                ts=p.stat().st_mtime
                hits.append({'session_id':sid,'mtime':ts,'iso':time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(ts)),'snippets':snips})
                if len(hits)>=max_sessions:break
        return {'query':q,'hits':hits,'n_hits':len(hits),'sessions_scanned':sessions_scanned}
    reg.register('recall',_skill_recall,desc='Cross-session conversation recall. Searches all session JSONLs (skipping current session + archived) for substring matches. Args: query, max_sessions? (default 5), max_snippets_per_session? (default 2). Returns hits grouped by session with timestamp + role + 240-char snippet.',schema={'query':'str','max_sessions':'int?','max_snippets_per_session':'int?'})
    def _skill_note(args,ctx,reg_):
        """Quick text capture. Actions: add (text, tags?) | list (limit?, tag?, search?, session_only?) | delete (id) | tags | stats."""
        try:from amni.serve import notes as _nt
        except Exception as e:return {'error':f'notes module unavailable: {e}'}
        action=str(args.get('action') or 'add').lower().strip()
        sid=''
        try:
            if ctx.get('conv'):sid=ctx['conv'].session_id or ''
        except Exception:pass
        if action=='add':
            text=args.get('text') or ''
            tags=args.get('tags');tags=tags if isinstance(tags,list) else ([str(tags)] if tags else None)
            return _nt.add(text=text,tags=tags,session_id=sid)
        if action=='list':
            limit=int(args.get('limit',20));tag=args.get('tag');search=args.get('search','') or ''
            session_only=bool(args.get('session_only',False))
            return {'notes':_nt.list_recent(limit=limit,tag=tag,search=search,session_id=(sid if session_only else None))}
        if action=='delete':
            nid=args.get('id','')
            if not nid:return {'error':'id required'}
            return _nt.delete(nid)
        if action=='tags':return {'tags':_nt.all_tags()}
        if action=='stats':return _nt.stats()
        return {'error':f'unknown action {action!r}; valid: add|list|delete|tags|stats'}
    reg.register('note',_skill_note,desc='Quick text capture (3rd capture surface after bookmarks/reminders). Actions: add (text, tags?) | list (limit?, tag?, search?, session_only?) | delete (id) | tags | stats. #hashtags in text auto-extracted as tags.',schema={'action':'str?','text':'str?','tags':'list?','id':'str?','tag':'str?','search':'str?','limit':'int?','session_only':'bool?'})
    def _skill_pose_coach(args,ctx,reg_):
        """Physical-therapy form coach from body-pose landmarks. Actions: exercises | analyze (landmarks, exercise) | start (exercise) | frame (landmarks, exercise?) | stop | history (exercise?, limit?)."""
        try:from amni.serve import pose_coach as _pc
        except Exception as e:return {'error':f'pose_coach module unavailable: {e}'}
        action=str(args.get('action') or 'exercises').lower().strip()
        sid=''
        try:
            if ctx.get('conv'):sid=ctx['conv'].session_id or ''
        except Exception:pass
        sid=args.get('session_id') or sid
        if action in ('exercises','list'):return {'exercises':_pc.list_exercises()}
        if action=='analyze':
            lms=args.get('landmarks')
            if not lms:return {'error':'landmarks required (list of {x,y,visibility?})'}
            return _pc.analyze_frame(lms,str(args.get('exercise') or 'pushup'))
        if action=='start':return _pc.start_session(str(args.get('exercise') or 'pushup'),session_id=sid)
        if action=='frame':
            lms=args.get('landmarks')
            if not lms:return {'error':'landmarks required'}
            if not sid:return {'error':'session_id required for frame (start a session first)'}
            return _pc.feed_session(sid,lms,exercise=args.get('exercise'))
        if action in ('stop','summary'):
            if not sid:return {'error':'session_id required'}
            return _pc.stop_session(sid)
        if action=='history':return {'history':_pc.session_history(limit=int(args.get('limit',20)),exercise=args.get('exercise'))}
        return {'error':f'unknown action {action!r}; valid: exercises|analyze|start|frame|stop|history'}
    reg.register('pose_coach',_skill_pose_coach,desc='Physical-therapy / exercise form coach. Computes joint angles from MediaPipe-Pose 33-pt landmarks, counts reps, and gives form feedback (push-up, sit-up/crunch, squat, bicep curl). Actions: exercises | analyze (landmarks, exercise) | start (exercise) | frame (landmarks, exercise?) | stop | history. Landmarks: list of {x,y,visibility?} normalized [0,1].',schema={'action':'str?','exercise':'str?','landmarks':'list?','session_id':'str?','limit':'int?'})
    def _skill_pc_action(args,ctx,reg_):
        """Safe PC operation behind a propose->confirm gate. Actions: propose (action, target, args?) | confirm (token) | cancel (token) | pending | audit. Destructive patterns refused; everything audited. Nothing runs until confirmed."""
        try:from amni.serve import pc_actions as _pca
        except Exception as e:return {'error':f'pc_actions unavailable: {e}'}
        action=str(args.get('action') or 'pending').lower().strip()
        if action=='propose':
            pca=str(args.get('pc_action') or args.get('type') or '')
            tgt=str(args.get('target') or '') or ('full screen' if pca=='screenshot' else '')
            extra=args.get('args') if isinstance(args.get('args'),dict) else None
            if args.get('question') and pca=='screenshot':extra={**(extra or {}),'question':args.get('question')}
            return _pca.propose(pca,tgt,extra)
        if action=='confirm':
            tok=args.get('token','')
            if not tok:return {'error':'token required'}
            res=_pca.confirm(tok)
            if res.get('executed') and res.get('action')=='screenshot' and isinstance(res.get('result'),dict) and res['result'].get('screenshot_path'):
                vision=ctx.get('vision')
                if vision is not None:
                    try:
                        from pathlib import Path as _P
                        img=_P(res['result']['screenshot_path']).read_bytes()
                        q=str(args.get('question') or '').strip()
                        d=vision.caption_with_question(img,q) if q else vision.describe(img)
                        if isinstance(d,dict) and not d.get('error'):res['result']['vision']=d.get('caption') or d.get('answer')
                    except Exception:pass
            return res
        if action=='cancel':
            tok=args.get('token','')
            return _pca.cancel(tok) if tok else {'error':'token required'}
        if action=='pending':return _pca.list_pending()
        if action=='audit':return _pca.audit_recent(limit=int(args.get('limit',30)))
        return {'error':f'unknown action {action!r}; valid: propose|confirm|cancel|pending|audit'}
    def _skill_coding_ledger(args,ctx,reg_):
        """Record/recall coding attempts so retries do better. Actions: record (task, outcome?, approach?, errors?, lesson?, success?, files?) | recall (task) | brief (task) | stats | commit."""
        try:from amni.serve import coding_ledger as _cl
        except Exception as e:return {'error':f'coding_ledger unavailable: {e}'}
        action=str(args.get('action') or 'stats').lower().strip()
        sid=''
        try:
            if ctx.get('conv'):sid=ctx['conv'].session_id or ''
        except Exception:pass
        if action=='record':
            errs=args.get('errors');errs=errs if isinstance(errs,list) else ([str(errs)] if errs else None)
            fls=args.get('files');fls=fls if isinstance(fls,list) else ([str(fls)] if fls else None)
            return _cl.record(task=str(args.get('task') or ''),outcome=str(args.get('outcome') or ''),approach=str(args.get('approach') or ''),errors=errs,lesson=str(args.get('lesson') or ''),success=args.get('success'),files=fls,session_id=sid)
        if action=='recall':return {'attempts':_cl.recall(str(args.get('task') or ''),k=int(args.get('k',3))),'prior':_cl.attempts_for(str(args.get('task') or ''))}
        if action=='brief':return {'brief':_cl.brief(str(args.get('task') or ''))}
        if action=='stats':return _cl.stats()
        if action=='commit':return _cl.commit_to_ptex(adam=ctx.get('adam'))
        if action=='federate':return _cl.federation_export(limit=int(args.get('limit',200)),only_success=bool(args.get('only_success',True)))
        if action=='import':
            ents=args.get('entries');ents=ents if isinstance(ents,list) else []
            return _cl.federation_import(ents,source=str(args.get('source') or 'peer'))
        return {'error':f'unknown action {action!r}; valid: record|recall|brief|stats|commit|federate|import'}
    reg.register('coding_ledger',_skill_coding_ledger,desc='Coding-attempt learning loop: so a 2nd attempt beats the 1st. Actions: record (task, outcome?, approach?, errors?, lesson?, success?, files?) | recall (task — prior attempts ranked by Reffelt-nonce relevance) | brief (task) | stats | commit (to PTEX) | federate (export PII-scrubbed successful lessons — no raw tasks/paths/errors — for cross-instance sharing) | import (entries, source — merge a peer\'s scrubbed lessons, marked federated, never counted as first-party attempts). Records persist to data/coding_attempts.jsonl + lessons/coding_attempts_ptex; surfaced pre-response.',schema={'action':'str?','task':'str?','outcome':'str?','approach':'str?','errors':'list?','lesson':'str?','success':'bool?','files':'list?','k':'int?','limit':'int?','only_success':'bool?','entries':'list?','source':'str?'})
    def _skill_coding_runner(args,ctx,reg_):
        """Conductor for the SE loop. Actions: prepare (task) -> work order with prior attempts + located files; complete (run_id, success, outcome?, errors?, lesson?, approach?, files?) -> records + says whether to retry; status (run_id) | runs."""
        try:from amni.serve import coding_runner as _cr
        except Exception as e:return {'error':f'coding_runner unavailable: {e}'}
        action=str(args.get('action') or 'runs').lower().strip()
        if action=='prepare':return _cr.prepare(str(args.get('task') or ''),agent=ctx.get('agent'),max_attempts=int(args.get('max_attempts',3)))
        if action=='complete':
            rid=args.get('run_id','')
            if not rid:return {'error':'run_id required'}
            errs=args.get('errors');errs=errs if isinstance(errs,list) else ([str(errs)] if errs else None)
            fls=args.get('files');fls=fls if isinstance(fls,list) else ([str(fls)] if fls else None)
            return _cr.complete(rid,success=bool(args.get('success',False)),outcome=str(args.get('outcome') or ''),errors=errs,lesson=str(args.get('lesson') or ''),approach=str(args.get('approach') or ''),files=fls,agent=ctx.get('agent'))
        if action=='verify':
            rid=args.get('run_id','')
            if not rid:return {'error':'run_id required'}
            tr=reg_.call('test_run',{'cmd':args.get('cmd'),'timeout':int(args.get('timeout',60))},ctx=ctx)
            test_result=getattr(tr,'output',None) if tr is not None else None
            if not isinstance(test_result,dict):test_result=test_result if isinstance(test_result,dict) else {'error':'test_run produced no result','passed':False}
            return _cr.complete_from_test(rid,test_result,lesson=str(args.get('lesson') or ''),approach=str(args.get('approach') or ''),agent=ctx.get('agent'))
        if action=='status':return _cr.status(str(args.get('run_id') or ''))
        if action=='runs':return _cr.list_runs()
        return {'error':f'unknown action {action!r}; valid: prepare|complete|verify|status|runs'}
    reg.register('coding_runner',_skill_coding_runner,desc='Agentic SE conductor: prepare (task) bundles prior-attempt recall + located files (code map) into a work order with attempt#; verify (run_id, cmd?) RUNS the tests + records OBJECTIVE success (tests pass=success, failures=errors to learn) + returns will_retry; complete (run_id, success, ...) for manual outcomes; status | runs. The runner never edits disk — writes go through pc_action propose->confirm.',schema={'action':'str?','task':'str?','run_id':'str?','success':'bool?','outcome':'str?','errors':'list?','lesson':'str?','approach':'str?','files':'list?','cmd':'str?','timeout':'int?','max_attempts':'int?'})
    def _skill_code_index(args,ctx,reg_):
        """Train Adam on a codebase: build a PTEX-backed map of files+symbols, then query it. Actions: build (root?) | query (term) | semantic (q) | file (path) | stats."""
        try:from amni.serve import code_index as _ci
        except Exception as e:return {'error':f'code_index unavailable: {e}'}
        action=str(args.get('action') or 'stats').lower().strip()
        enc=getattr(getattr(ctx.get('adam'),'sem_lut',None),'encoder',None) if ctx.get('adam') is not None else None
        if action=='build':return _ci.build_index(root=args.get('root'),max_files=int(args.get('max_files',6000)),ptex=bool(args.get('ptex',True)),encoder=enc)
        if action=='query':return _ci.query(str(args.get('term') or args.get('query') or ''),limit=int(args.get('limit',25)))
        if action=='semantic':return _ci.semantic_query(str(args.get('q') or args.get('query') or ''),encoder=enc,k=int(args.get('k',5)))
        if action=='file':return _ci.file_info(str(args.get('path') or ''))
        if action=='map':return _ci.repo_map(root=args.get('root'),max_files=int(args.get('max_files',400)),per_file_syms=int(args.get('per_file_syms',12)),sub=args.get('sub') or args.get('dir'),lang=args.get('lang'),encoder=enc)
        if action=='stats':return _ci.stats()
        return {'error':f'unknown action {action!r}; valid: build|map|query|semantic|file|stats'}
    reg.register('code_index',_skill_code_index,desc='Train Adam on a codebase + locate code. Walks a tree extracting per-file language/symbols/summary into a PTEX-backed map (experiences/code_map_ptex). Actions: map (ONE compact pointer-index of every file->top symbols+summary — CALL THIS FIRST to see the whole repo cheaply, optional sub=path-substring/lang/per_file_syms, then file_read the one file you pick) | build (root?, max_files?) | query (term — substring over paths+symbols) | semantic (q — Reffelt-cell nearest file) | file (path — its symbols) | stats. Foundation for the coding loop: map -> locate -> edit -> verify -> iterate.',schema={'action':'str?','root':'str?','term':'str?','query':'str?','q':'str?','path':'str?','sub':'str?','dir':'str?','lang':'str?','per_file_syms':'int?','max_files':'int?','limit':'int?','k':'int?','ptex':'bool?'})
    reg.register('pc_action',_skill_pc_action,desc='Safe PC operation (Tier 5b). propose-then-confirm gate: nothing touches the OS until the owner confirms a token. pc_action one of: echo|notify|open_url|open_path|launch_app|screenshot|type_text|press_key|click|run. type_text target=text to type; press_key target=key or combo e.g. "ctrl+c"; click target="x,y" or args {x,y,button,clicks}; screenshot describes the screen via LOCAL vision (never sent off-box). Actions: propose | confirm (token) | cancel (token) | pending | audit. Destructive patterns refused outright; every step audited to logs/pc_actions.jsonl. Tip: screenshot first so you can see the target before click/type.',schema={'action':'str?','pc_action':'str?','target':'str?','args':'dict?','token':'str?','question':'str?','limit':'int?'})
    reg.register('file_write',_skill_file_write,gate=_gate_path,desc=f'Write/overwrite a UTF-8 text file within {scope}. Args: {{path, content}}',schema={'path':'str','content':'str'})
    reg.register('code_edit',_skill_code_edit,gate=_gate_code_edit,desc=f'Find-and-replace edit in a file within {scope}; .py edits ast-validated. Args: {{path, find, replace, count?}}',schema={'path':'str','find':'str','replace':'str','count':'int?'})
    reg.register('shell',_skill_shell,gate=_gate_shell,desc=f'Run a read-only allowlisted shell command from primary root. Scope: {scope}. Args: {{cmd, timeout?}}',schema={'cmd':'str','timeout':'int?'})
    reg.register('git',_skill_git,desc=f'Read-only git in {scope}. Args: {{cmd, file?, n?}}. cmd one of: status, log, diff, branch, blame, show, ls-files, remote, config, rev-parse, describe, tag, shortlog, reflog. Mutation ops (add/commit/push/etc) refused.',schema={'cmd':'str','file':'str?','n':'int?'})
    reg.register('test_run',_skill_test_run,desc=f'Auto-detect + run project tests in {scope}. Detects cargo/pytest/npm/go/make. Args: {{cmd?, timeout?}}. Returns passed:bool + stdout/stderr.',schema={'cmd':'str?','timeout':'int?'})
    reg.register('code_diff',_skill_code_diff,gate=_gate_diff,desc=f'Apply a unified diff (@@ hunks) to a file in {scope}. Safer than full-file rewrite. Args: {{path, diff, dry_run?}}.',schema={'path':'str','diff':'str','dry_run':'bool?'})
    reg.register('project_info',_skill_project_info,desc=f'Summarize the current workspace ({scope}): top files, detected languages, dependency manifests, git branch + dirty status. No args.',schema={})
    reg.register('format_code',_skill_format_code,gate=_gate_path,desc=f'Run the canonical formatter for a file in {scope} (.py:ruff/black, .rs:rustfmt, .js/.ts/.jsx/.tsx/.json/.html/.css:prettier, .go:gofmt). Args: {{path}}.',schema={'path':'str'})
    reg.register('symbols',_skill_symbols,gate=_gate_path,desc=f'Extract functions/classes/imports from a code file in {scope}. AST-based for Python; regex for Rust/JS/TS. Args: {{path}}.',schema={'path':'str'})
    reg.register('rename_symbol',_skill_rename_symbol,desc=f'Rename a symbol across all code files in {scope} (word-boundary regex). Args: {{old, new, glob?, exts?, dry_run?}}. Use dry_run first to preview.',schema={'old':'str','new':'str','glob':'str?','exts':'list?','dry_run':'bool?'})
    reg.register('auto_import',_skill_auto_import,gate=_gate_path,desc=f'Detect undefined names in a Python file and suggest stdlib imports. Args: {{path}}.',schema={'path':'str'})
    reg.register('parse_error',_skill_parse_error,desc='Parse a Python/Rust/JS/TS/Go compiler or runtime error/stack-trace. Returns {language, kind, file, line, message, likely_cause, suggested_fix}. Args: {text}.',schema={'text':'str'})
    reg.register('export_session',_skill_export_session,desc='Dump a session as markdown/text/json. Args: {session_id, out_path?, format?}. format = markdown (default) | text | json.',schema={'session_id':'str','out_path':'str?','format':'str?'})
    reg.register('prune_sessions',_skill_prune_sessions,desc='Delete old session jsonl files. Keeps N most-recent; only deletes >older_than_days old. Args: {older_than_days?=30, keep_n?=50, dry_run?=false}.',schema={'older_than_days':'int?','keep_n':'int?','dry_run':'bool?'})
    reg.register('tts',_skill_tts,desc='Text-to-speech. Returns WAV audio (base64) or writes to out_path. Backends auto-detect: piper > pyttsx3 (Windows SAPI / espeak). Args: {text, out_path?, backend?, voice?, list_voices?}.',schema={'text':'str','out_path':'str?','backend':'str?','voice':'str?','list_voices':'bool?'})
    reg.register('stt',_skill_stt,desc='Speech-to-text. Accepts path (workdir-scoped WAV) or audio_base64. Backends: faster-whisper (recommended) > vosk. Args: {path? | audio_base64?, model_size?, backend?}.',schema={'path':'str?','audio_base64':'str?','model_size':'str?','backend':'str?'})
    reg.register('run_python',_skill_run_python,desc='Execute a Python snippet in a hardened sandbox: isolated `python -I -B`, stripped env (no secrets), throwaway temp cwd, CPU/mem/output caps, and network egress disabled. AST + regex danger-scan rejects fs-mutation/subprocess/exec/eval/network. For pure computation, not file/workdir access. Returns stdout/stderr/returncode. Args: {code, timeout?}',schema={'code':'str','timeout':'int?'})
    reg.register('scan',_skill_scan,gate=_gate_path,desc=f'Walk path (file or dir + glob), chunk text, teach each chunk to Adam. Args: {{path, glob?, max_files?, max_chars_per_file?, distill?, only_text?}}',schema={'path':'str','glob':'str?','max_files':'int?','max_chars_per_file':'int?','distill':'bool?','only_text':'bool?'})
    def _skill_chain(args,ctx,reg_):
        steps=args.get('steps') or [];max_steps=int(args.get('max_steps',8))
        if not isinstance(steps,list) or not steps:return {'error':'chain needs steps: list of {skill, args} dicts'}
        if len(steps)>max_steps:return {'error':f'chain capped at {max_steps} steps (got {len(steps)})'}
        results=[];prev=None;ok=True
        for i,step in enumerate(steps):
            if not isinstance(step,dict):results.append({'step':i,'error':'step must be a dict','ok':False});ok=False;break
            name=(step.get('skill') or '').strip();sargs=step.get('args') or {}
            if not name:results.append({'step':i,'error':'step missing "skill" name','ok':False});ok=False;break
            if name=='chain':results.append({'step':i,'error':'nested chain not allowed (prevent recursion)','ok':False});ok=False;break
            try:
                if isinstance(sargs,dict) and prev is not None:
                    sargs={k:(json.dumps(prev,default=str)[:4000] if v=='$prev' else (str(prev)[:4000] if v=='$prev_str' else v)) for k,v in sargs.items()}
            except Exception:pass
            r=reg_.call(name,sargs,ctx=ctx)
            d=r.to_dict() if hasattr(r,'to_dict') else {'ok':bool(r and not getattr(r,'error',None)),'output':r}
            results.append({'step':i,'skill':name,'args':sargs,**d})
            if not d.get('ok',False):
                if step.get('continue_on_error'):ok=False;continue
                else:ok=False;break
            prev=d.get('output')
        return {'ok':ok,'n_steps':len(results),'results':results,'final':results[-1].get('output') if results else None}
    reg.register('chain',_skill_chain,desc='Run a sequence of skills in order. Args: {steps:[{skill, args, continue_on_error?},...], max_steps?=8}. Use "$prev" / "$prev_str" in args to inject previous step output. Nested chain disallowed.',schema={'steps':'list','max_steps':'int?'})
    def _skill_self_inspect(args,ctx,reg_):
        """Read Adam's own source for self-reflection. Returns summary digest the LLM can reason over to identify hotspots / propose improvements."""
        from pathlib import Path as _P
        repo_root=_P(__file__).resolve().parents[2]
        subsystem=(args.get('subsystem') or 'amni/serve').strip().lstrip('/')
        max_files=int(args.get('max_files',12));max_chars=int(args.get('max_chars_per_file',2400))
        base=(repo_root/subsystem).resolve()
        try:base.relative_to(repo_root.resolve())
        except Exception:return {'error':f'subsystem must be inside repo root; got {subsystem}'}
        if not base.exists():return {'error':f'subsystem path not found: {subsystem}'}
        targets=[]
        if base.is_file():targets=[base]
        else:
            for p in sorted(base.rglob('*.py')):
                if p.name.startswith('_') and p.name!='__init__.py':continue
                if '__pycache__' in p.parts:continue
                targets.append(p)
                if len(targets)>=max_files:break
        snippets=[]
        for p in targets:
            try:
                txt=p.read_text(encoding='utf-8',errors='ignore')
                lines=txt.splitlines();line_count=len(lines)
                if len(txt)>max_chars:txt=txt[:max_chars]+f'\n... ({line_count} total lines, truncated)'
                snippets.append({'path':str(p.relative_to(repo_root)),'lines':line_count,'bytes':p.stat().st_size,'preview':txt})
            except Exception:continue
        return {'subsystem':subsystem,'files_inspected':len(snippets),'snippets':snippets,'hint':'Read the snippets, then propose specific improvements via the self_improvement skill (action=propose). Reference file paths + line numbers in your rationale.'}
    reg.register('self_inspect',_skill_self_inspect,desc='Read Adam\'s own source files for self-reflection. Args: {subsystem?=\'amni/serve\', max_files?=12, max_chars_per_file?=2400}. Scope-checked to repo root.',schema={'subsystem':'str?','max_files':'int?','max_chars_per_file':'int?'})
    def _skill_self_improvement(args,ctx,reg_):
        """Record + manage self-improvement proposals. Actions: propose | list | get | transition | stats. The actual code changes flow through file_write+code_edit (with v6.10.16 verify + v6.10.19 auto-pytest). This skill is the notebook, not the robot arm."""
        from amni.serve import self_improvement as _si
        action=(args.get('action') or 'list').strip().lower()
        if action=='propose':
            return _si.propose(title=args.get('title',''),rationale=args.get('rationale',''),planned_change=args.get('planned_change',''),files_touched=args.get('files_touched',[]),category=args.get('category','enhancement'),author=args.get('author','adam'))
        if action=='list':return {'proposals':_si.list_proposals(status=args.get('status'),category=args.get('category'),limit=int(args.get('limit',50)),include_history=bool(args.get('include_history')))}
        if action=='get':
            pid=args.get('id');
            if not pid:return {'error':'need id for action=get'}
            p=_si.get_proposal(pid);return p if p else {'error':f'no proposal {pid!r}'}
        if action=='transition':
            pid=args.get('id');new_status=args.get('status');
            if not pid or not new_status:return {'error':'need id + status for transition'}
            return _si.transition(pid,new_status,notes=args.get('notes',''),author=args.get('author','adam'))
        if action=='stats':return _si.stats()
        return {'error':f'unknown action {action!r}; valid: propose|list|get|transition|stats'}
    reg.register('self_improvement',_skill_self_improvement,desc='Record and query Adam\'s self-improvement proposals. Actions: propose (title, rationale, planned_change, files_touched?, category?) | list (status?, category?, limit?, include_history?) | get (id) | transition (id, status: proposed|attempted|validated|deployed|declined|reverted, notes?) | stats. Proposals are an append-only audit log.',schema={'action':'str','title':'str?','rationale':'str?','planned_change':'str?','files_touched':'list?','category':'str?','id':'str?','status':'str?','notes':'str?','limit':'int?','include_history':'bool?'})
    def _skill_venv(args,ctx,reg_):
        """Manage sandboxed Python virtual environments under <workdir>/.adam-venvs/. Actions: list | create | install | run | remove."""
        from amni.serve import venv_manager as _vm
        action=(args.get('action') or '').strip().lower()
        wd=reg_.workdir
        if action=='list':return {'venvs':_vm.list_venvs(wd)}
        name=(args.get('name') or '').strip()
        if action=='create':return _vm.create(wd,name)
        if action=='install':return _vm.install(wd,name,args.get('packages') or [])
        if action=='run':return _vm.run(wd,name,args.get('cmd') or '',timeout=int(args.get('timeout',300)))
        if action=='remove':return _vm.remove(wd,name)
        return {'error':f'unknown action {action!r}; valid: list|create|install|run|remove'}
    reg.register('venv',_skill_venv,desc='Sandboxed Python venv management for Adam\'s experiments. Actions: list | create (name) | install (name, packages: list) | run (name, cmd, timeout?) | remove (name). All venvs live under <workdir>/.adam-venvs/; name must match [a-z0-9_-]{1,32}; cap of 8 concurrent venvs; pip install validates package specs; run refuses obviously-destructive cmds.',schema={'action':'str','name':'str?','packages':'list?','cmd':'str?','timeout':'int?'})
    def _skill_self_reflect(args,ctx,reg_):
        """Run / inspect Adam's periodic self-reflection cycle. Actions: status | run | enable | disable."""
        from amni.serve import self_reflection as _sr
        action=(args.get('action') or 'status').strip().lower()
        if action=='status':return _sr.status()
        if action=='run':return _sr.run_cycle(force=bool(args.get('force',False)),dry_run=bool(args.get('dry_run',False)),notify=bool(args.get('notify',True)))
        if action=='enable':return _sr.set_enabled(True)
        if action=='disable':return _sr.set_enabled(False)
        return {'error':f'unknown action {action!r}; valid: status|run|enable|disable'}
    reg.register('self_reflect',_skill_self_reflect,desc='Adam\'s daily self-reflection cycle. Rotates through subsystems (amni/serve, amni/storage, amni/agent, amni/skills, amni/cli, scripts), scans for heuristic signals (TODOs, large files, missing tests, missing docstrings), drops up to 3 proposals/cycle into the self_improvement log. Actions: status | run (force?, dry_run?, notify?) | enable | disable. Cap: one cycle per ~20h unless force=True.',schema={'action':'str?','force':'bool?','dry_run':'bool?','notify':'bool?'})
    def _skill_proposal_attempt(args,ctx,reg_):
        """Auto-attempt a low-risk (category=documentation) self-improvement proposal via deterministic handler. NEVER deploys — human approval required."""
        from amni.serve import proposal_attempter as _pa
        action=(args.get('action') or 'attempt').strip().lower()
        if action=='handlers':return {'handlers':_pa.list_handlers()}
        if action=='attempt':
            pid=(args.get('id') or '').strip()
            if not pid:return {'error':'id required for action=attempt'}
            return _pa.attempt(pid,dry_run=bool(args.get('dry_run',False)),notify=bool(args.get('notify',True)))
        if action=='attempt_next':return _pa.attempt_next_eligible(max_attempts=int(args.get('max',1)),dry_run=bool(args.get('dry_run',False)))
        return {'error':f'unknown action {action!r}; valid: handlers|attempt|attempt_next'}
    reg.register('proposal_attempt',_skill_proposal_attempt,desc='Auto-attempt LOW-RISK (category=documentation) self-improvement proposals via deterministic handlers. Pipeline: backup -> apply -> ast.parse + sha256 readback + sibling pytest -> transition state. NEVER marks deployed (human approval required). Actions: handlers (list known) | attempt (id, dry_run?) | attempt_next (max?, dry_run?).',schema={'action':'str?','id':'str?','dry_run':'bool?','notify':'bool?','max':'int?'})
    def _skill_metrics_snapshot(args,ctx,reg_):
        """Adam's daily metrics snapshot for trend tracking. Actions: status | snapshot | history | trend | enable | disable."""
        from amni.serve import metrics_snapshot as _ms
        action=(args.get('action') or 'status').strip().lower()
        if action=='status':return _ms.status()
        if action=='snapshot':return _ms.snapshot(force=bool(args.get('force',False)),notify=bool(args.get('notify',False)))
        if action=='collect':return {'snapshot':_ms.collect()}
        if action=='history':return {'history':_ms.history(limit=int(args.get('limit',30)))}
        if action=='trend':return _ms.trend(days=int(args.get('days',7)))
        if action=='enable':return _ms.set_enabled(True)
        if action=='disable':return _ms.set_enabled(False)
        return {'error':f'unknown action {action!r}; valid: status|snapshot|collect|history|trend|enable|disable'}
    reg.register('metrics_snapshot',_skill_metrics_snapshot,desc='Daily metric snapshots of Adam\'s own behavior (skill latency, daemon throughput, coach streak, verification pass rate, proposal mix, queue depth). One row/day to data/metrics_snapshots.jsonl. Actions: status | snapshot (force?, notify?) | collect (read-only sample) | history (limit?) | trend (days?) | enable | disable.',schema={'action':'str?','force':'bool?','notify':'bool?','days':'int?','limit':'int?'})
    try:
        from amni.serve import widgets as _w
        def _skill_weather(args,ctx,reg_):
            d=_w.fetch_weather(location=args.get('location',''),lat=args.get('lat'),lon=args.get('lon'))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('weather',d,title=f"Weather — {d.get('location','?')}",icon='🌤')
            return d
        def _skill_system_stats(args,ctx,reg_):
            d=_w.fetch_system_stats()
            d['widget']=_w.make_widget_envelope('system',d,title='System',icon='⚙')
            return d
        def _skill_time_card(args,ctx,reg_):
            d=_w.fetch_time_card(tz_name=args.get('tz'))
            d['widget']=_w.make_widget_envelope('time',d,title='Time',icon='🕐')
            return d
        reg.register('weather',_skill_weather,desc='Current weather + forecast for a location via Open-Meteo (no API key). Emits a weather widget. Args: {location?:str, lat?:float, lon?:float}',schema={'location':'str?','lat':'float?','lon':'float?'})
        reg.register('system_stats',_skill_system_stats,desc='CPU/memory/disk/GPU snapshot via psutil + torch. Emits a system widget.',schema={})
        reg.register('time_card',_skill_time_card,desc='Time + timezone + weekday as a time widget. Args: {tz?:str like America/New_York}',schema={'tz':'str?'})
        def _skill_news(args,ctx,reg_):
            d=_w.fetch_news(query=args.get('query',''),n=int(args.get('n',6)))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('news',d,title=f"News — {d.get('query','top')}",icon='📰')
            return d
        def _skill_stock(args,ctx,reg_):
            d=_w.fetch_stock(symbols=args.get('symbols','') or args.get('symbol',''))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('stock',d,title=f"Stock — {d.get('symbols','?')}",icon='📈')
            return d
        def _skill_file_preview(args,ctx,reg_):
            d=_w.fetch_file_preview(args.get('path',''),max_lines=int(args.get('max_lines',40)),max_chars=int(args.get('max_chars',2400)))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('file',d,title=f"File — {(d.get('path') or '').split('/')[-1].split(chr(92))[-1]}",icon='📄')
            return d
        def _skill_disk(args,ctx,reg_):
            d=_w.fetch_disk()
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('disk',d,title='Disk usage',icon='💾')
            return d
        def _skill_git_status(args,ctx,reg_):
            d=_w.fetch_git_status(workdir=args.get('workdir'))
            if d.get('_error'):return {'error':d['_error']}
            d['widget']=_w.make_widget_envelope('git',d,title=f"Git — {d.get('branch','?')}",icon='⎇')
            return d
        reg.register('news',_skill_news,desc='Top news headlines for a topic via DuckDuckGo news. Emits a news widget. Args: {query?:str, n?:int=6}',schema={'query':'str?','n':'int?'})
        reg.register('stock',_skill_stock,desc='Stock quote(s) via Yahoo Finance. Emits a stock widget. Args: {symbols:str e.g. "AAPL,MSFT,GOOG"}',schema={'symbols':'str','symbol':'str?'})
        reg.register('file_preview',_skill_file_preview,gate=_gate_path,desc=f'Read first N lines of a text file in {scope} and render as a file widget. Args: {{path, max_lines?=40, max_chars?=2400}}',schema={'path':'str','max_lines':'int?','max_chars':'int?'})
        reg.register('disk_widget',_skill_disk,desc='Per-partition disk usage via psutil. Emits a disk widget.',schema={})
        reg.register('git_status',_skill_git_status,desc='git branch + dirty count + recent commits + remote + ahead/behind. Emits a git widget. Args: {workdir?:str}',schema={'workdir':'str?'})
    except Exception as _we:print(f'[skills] widget skills register failed: {_we}',flush=True)
    try:
        from amni.serve.coach import coach_skill as _coach_skill
        reg.register('coach',_coach_skill,desc='Socratic coaching/tutor mode. Actions: start <topic> [seed_question?] | ask | answer <text> | hint | skip | summary | status. Tracks per-topic mastery in coach_atlas. Args: {action, topic?, session_id?, answer?, difficulty?, seed_question?, seed_model_answer?, seed_hint?}',schema={'action':'str','topic':'str?','session_id':'str?','answer':'str?','difficulty':'int?','seed_question':'str?','seed_model_answer':'str?','seed_hint':'str?'})
    except Exception as _ce:print(f'[skills] coach skill register failed: {_ce}',flush=True)
    try:
        from amni.serve.scheduler import schedule_loop_skill as _sched_skill
        reg.register('schedule_loop',_sched_skill,desc='Adam-driven recurring jobs. Actions: add (kind, payload, cadence_s, label?, start_in_s?) | list | get <id> | cancel <id> | enable <id> | disable <id> | runs <id> | run_now <id> | stats. Kinds: skill (payload={name,args}), prompt (payload={text,system?}), webpoll (payload={url|query}). Args: {action, kind?, payload?, cadence_s?, id?, label?, start_in_s?}',schema={'action':'str','kind':'str?','payload':'dict?','cadence_s':'int?','id':'str?','label':'str?','start_in_s':'int?'})
    except Exception as _se:print(f'[skills] schedule_loop skill register failed: {_se}',flush=True)
    try:
        from amni.serve import ingest as _ingest
        _ingest.register(reg)
    except Exception as _ie:print(f'[skills] ingest skills register failed: {_ie}',flush=True)
    try:
        from amni.serve.learning_daemon import learning_daemon_skill as _ld_skill
        reg.register('learning_daemon',_ld_skill,desc='Inspect/control Adam\'s 24/7 self-improvement daemon AND run user-directed learning missions. Actions: stats | curiosity_tick | sleep_pass | repetition_pass | pause | resume | queue_topic <topic> | atlas_verified | atlas_debated | mission_start (topic=what to master, stop=stop condition, timeframe?) | mission_status | mission_stop | mission_resume. A mission makes Adam autonomously web-crawl to master the topic, decomposing it into subtopics and looping (going deeper each round) until the user stops it or it self-assesses the stop condition is met. Args: {action, topic?, stop?, timeframe?, limit?}',schema={'action':'str','topic':'str?','stop':'str?','stop_condition':'str?','timeframe':'str?','mission':'str?','limit':'int?'})
    except Exception as _le:print(f'[skills] learning_daemon skill register failed: {_le}',flush=True)
    try:
        from amni.serve.federation_store import federation_skill as _fed_skill
        reg.register('federation',_fed_skill,desc='Federate Adam\'s learned lessons as portable PTEX learning packs (git/HuggingFace stored) and fetch them on demand. Actions: stats | list (the manifest \'page\' of available packs; remote=URL to read a peer\'s page) | export (publish a pack; source=provenance filter e.g. code-corpus, domain?, languages?, push? to git+HF) | fetch (download+merge a pack into the rapid-access sem_lut; pack_id? | url? | domain? | languages?). Local-first; HF push needs AMNI_LEARNINGS_HF_REPO+HF_TOKEN, git push needs AMNI_LEARNINGS_GIT_REMOTE. Args: {action, source?, domain?, languages?, pack_id?, url?, remote?, name?, push?, limit?}',schema={'action':'str','source':'str?','domain':'str?','languages':'list?','pack_id':'str?','id':'str?','url':'str?','remote':'str?','name':'str?','push':'bool?','limit':'int?'})
    except Exception as _fede:print(f'[skills] federation skill register failed: {_fede}',flush=True)
    try:
        from amni.serve.kg_query import kg_query_skill as _kg_skill
        reg.register('kg_query',_kg_skill,desc='Query Adam\'s knowledge graph (SPO triples). Actions: stats | neighbors <subject> | out <subject> | in <subject> | predicate <p> | path <from> <to> [max_hops] | search <q> | add s,p,o | forget [s|p|o]. Args: {action, subject?, q?, predicate?, p?, a?, b?, from?, to?, max_hops?, s?, o?, object?, source?, confidence?, limit?, direction?}',schema={'action':'str','subject':'str?','q':'str?','predicate':'str?','p':'str?','s':'str?','o':'str?','a':'str?','b':'str?','from':'str?','to':'str?','max_hops':'int?','limit':'int?','direction':'str?','source':'str?','confidence':'float?'})
    except Exception as _kge:print(f'[skills] kg_query skill register failed: {_kge}',flush=True)
    try:
        from amni.serve.vision import describe_image_skill as _vis_skill
        reg.register('describe_image',_vis_skill,desc='Describe an image via BLIP. Provide image_base64 OR path. Optional question for VQA mode ("what color is the dog?"). Args: {image_base64?, path?, question?}',schema={'image_base64':'str?','path':'str?','question':'str?'})
    except Exception as _ve:print(f'[skills] describe_image skill register failed: {_ve}',flush=True)
    try:
        from amni.storage.file_watcher import watch_skill as _watch_skill
        reg.register('watch',_watch_skill,desc='File/folder change watcher. Actions: add (path, glob?, recursive?=true, label?, on_change_skill?, on_change_args?, coalesce_s?=2.0) | list | get <id> | cancel <id> | enable <id> | disable <id> | events <id> | tick | stats. Args: {action, path?, glob?, recursive?, label?, on_change_skill?, on_change_args?, coalesce_s?, id?, limit?}',schema={'action':'str','path':'str?','glob':'str?','recursive':'bool?','label':'str?','on_change_skill':'str?','on_change_args':'dict?','coalesce_s':'float?','id':'str?','limit':'int?'})
    except Exception as _we:print(f'[skills] watch skill register failed: {_we}',flush=True)
    if with_agentic:
        try:from amni.serve.agentic import register as _reg_agentic;_reg_agentic(reg)
        except Exception as e:print(f'[skills] agentic register failed: {e}',flush=True)
    return reg

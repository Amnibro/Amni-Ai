import os,hmac,stat,struct,hashlib,secrets,threading
from amni.serve.amni_chat_client import _hkdf_sha256,_aenc,_adec
NF=getattr(os,'O_NOFOLLOW',0);EHDR=struct.Struct('<4sHH8s16s12s');EVER=2;_lock=threading.RLock()
def dsync(d:str):
    try:fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
    except OSError:pass
def scrub(fd:int):
    n=os.fstat(fd).st_size;os.lseek(fd,0,0);[os.write(fd,secrets.token_bytes(min(65536,n-i))) for i in range(0,n,65536)];os.fsync(fd)
def wfile(p:str,b:bytes,shred:bool=False):
    d=os.path.dirname(p);os.makedirs(d,mode=0o700,exist_ok=True);t=p+f'.{os.getpid()}.tmp';fd=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|NF,0o600)
    with os.fdopen(fd,'wb') as fh:fh.write(b);fh.flush();os.fsync(fh.fileno())
    o=os.open(p,os.O_RDWR|NF) if shred and os.name=='posix' and os.path.isfile(p) and not os.path.islink(p) else None
    try:os.replace(t,p);o is not None and scrub(o)
    finally:o is not None and os.close(o)
    dsync(d)
def shred(p:str)->bool:
    if not os.path.lexists(p):return False
    try:
        fd=os.open(p,os.O_RDWR|NF)
        try:scrub(fd)
        finally:os.close(fd)
    except OSError:pass
    os.remove(p);dsync(os.path.dirname(p));return True
class Vault:
    def __init__(s,n:str,magic:bytes,ver=None):s.n,s.m,s.v=n,magic,ver;s.lb=tuple(f'amni-{n}-{x}'.encode() for x in ('v2','user\0','page','kid'))
    def strict(s,p:str,d:bool):
        t=os.lstat(p)
        if not (stat.S_ISDIR if d else stat.S_ISREG)(t.st_mode) or os.name=='posix' and (stat.S_IMODE(t.st_mode)&0o077 or t.st_uid!=os.getuid()):raise PermissionError(f'{s.n} key must be a private file (0600) in a private directory (0700)')
    def kread(s,p:str)->bytes:
        s.strict(os.path.dirname(p),True);s.strict(p,False);fd=os.open(p,os.O_RDONLY|NF)
        with os.fdopen(fd,'rb') as f:k=f.read(33)
        if len(k)!=32:raise ValueError(f'{s.n} key invalid')
        return k
    def uk(s,kf:bytes,name:str)->bytes:return _hkdf_sha256(kf,32,s.lb[0],s.lb[1]+name.encode())
    def kid(s,uk:bytes)->bytes:return _hkdf_sha256(uk,8,b'\x00'*32,s.lb[3])
    def seal(s,inner:bytes,uk:bytes)->bytes:
        sa,nc=secrets.token_bytes(16),secrets.token_bytes(12);h=EHDR.pack(s.m,EVER,0,s.kid(uk),sa,nc);return h+_aenc(_hkdf_sha256(uk,32,sa,s.lb[2]),nc,inner,h)
    def unseal(s,raw:bytes,uk:bytes)->bytes:
        if len(raw)<EHDR.size+16:raise ValueError(f'{s.n} store corrupt')
        mg,ver,_,kd,sa,nc=EHDR.unpack_from(raw);h=raw[:EHDR.size]
        if mg!=s.m or ver!=EVER or not hmac.compare_digest(kd,s.kid(uk)):raise ValueError(f'{s.n} store key mismatch')
        try:return _adec(_hkdf_sha256(uk,32,sa,s.lb[2]),nc,raw[EHDR.size:],h)
        except Exception:raise ValueError(f'{s.n} store authentication failed') from None
    def env(s,b:bytes)->bool:return b[:4]==s.m and len(b)>=6 and struct.unpack_from('<H',b,4)[0]==EVER
    def plain(s,b:bytes)->bool:return b[:4]==s.m and len(b)>=6 and struct.unpack_from('<H',b,4)[0]==s.v if s.v is not None else not s.env(b)
    def kcommit(s,kp:str):shred(kp);os.replace(kp+'.new',kp);dsync(os.path.dirname(kp))
    def ukey(s,kp:str,name:str,kd:bytes)->bytes:
        for p in (kp,kp+'.new'):
            uk=s.uk(s.kread(p),name) if os.path.lexists(p) else None
            if uk and hmac.compare_digest(s.kid(uk),kd):(s.kcommit(kp) if p!=kp else shred(kp+'.new'));return uk
        raise ValueError(f'{s.n} key missing or wrong')
    def head(s,p:str):
        try:
            with open(p,'rb') as f:b=f.read(EHDR.size)
        except FileNotFoundError:return None,False
        return (b[8:16] if len(b)==EHDR.size and s.env(b) else None),s.plain(b)
    def read(s,p:str,kp:str,name:str):
        with _lock:
            raw=open(p,'rb').read()
            if s.plain(raw):return raw,True
            if len(raw)<EHDR.size or raw[:4]!=s.m:raise ValueError(f'{s.n} store corrupt')
            return s.unseal(raw,s.ukey(kp,name,raw[8:16])),False
    def write(s,p:str,kp:str,name:str,inner:bytes,rotate:bool=False):
        with _lock:
            kd,leg=s.head(p);fresh=rotate or leg or kd is None and not os.path.lexists(kp)
            if fresh:kf=secrets.token_bytes(32);wfile(kp+'.new',kf);s.strict(os.path.dirname(kp),True);uk=s.uk(kf,name)
            else:uk=s.ukey(kp,name,kd) if kd is not None else s.uk(s.kread(kp),name)
            wfile(p,s.seal(inner,uk),shred=rotate or leg);fresh and s.kcommit(kp)
    def wipe(s,p:str,kp:str)->bool:
        with _lock:r=shred(p);[shred(x) for x in (kp,kp+'.new')];return r
    def fkey(s,p:str)->tuple:
        p=os.path.abspath(p);d,b=os.path.split(p);k=os.environ.get('AMNI_UMEM_KEY_DIR');return (os.path.join(k,s.n,hashlib.sha256(d.encode()).hexdigest()[:12]+'-'+b+'.key') if k else os.path.join(d,'keys',b+'.key')),b
    def get(s,p:str)->bytes:
        kp,b=s.fkey(p);inner,leg=s.read(p,kp,b);leg and s.write(p,kp,b,inner);return inner
    def put(s,p:str,inner:bytes,rotate:bool=False):kp,b=s.fkey(p);s.write(p,kp,b,inner,rotate)
    def drop(s,p:str)->bool:return s.wipe(p,s.fkey(p)[0])
    def add(s,p:str,b:bytes):
        with _lock:s.put(p,(s.get(p) if os.path.exists(p) else b'')+b)
ATLAS=Vault('atlas',b'AMAT')

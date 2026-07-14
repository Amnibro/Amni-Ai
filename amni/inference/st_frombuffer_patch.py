import json,struct,torch,safetensors,safetensors.torch as _stt
_DT={'F64':torch.float64,'F32':torch.float32,'F16':torch.float16,'BF16':torch.bfloat16,'I64':torch.int64,'I32':torch.int32,'I16':torch.int16,'I8':torch.int8,'U8':torch.uint8,'BOOL':torch.bool,'F8_E4M3':getattr(torch,'float8_e4m3fn',None),'F8_E5M2':getattr(torch,'float8_e5m2',None)}
_ORIG_SAFE_OPEN=safetensors.safe_open
class _SafeOpenPt:
    def __init__(s,fn,device='cpu'):
        s._dev=device;s._f=open(fn,'rb');hlen=struct.unpack('<Q',s._f.read(8))[0];hdr=json.loads(s._f.read(hlen));s._meta=hdr.pop('__metadata__',None);s._hdr=hdr;s._base=8+hlen
    def keys(s):return list(s._hdr.keys())
    def metadata(s):return s._meta
    def get_tensor(s,k):
        v=s._hdr[k];a,b=v['data_offsets'];s._f.seek(s._base+a)
        t=torch.frombuffer(bytearray(s._f.read(b-a)),dtype=_DT[v['dtype']]).reshape(v['shape'] if v['shape'] else [])
        return t if str(s._dev) in ('cpu','') else t.to(s._dev)
    def __enter__(s):return s
    def __exit__(s,*a):
        s._f.close();return False
def _safe_open(fn,framework='pt',device='cpu'):
    return _SafeOpenPt(fn,device) if framework=='pt' else _ORIG_SAFE_OPEN(fn,framework=framework,device=device)
def _build(hdr,base,reader,device):
    dev=device if isinstance(device,torch.device) else torch.device(device if device else 'cpu')
    out={}
    for k,v in hdr.items():
        if k=='__metadata__':continue
        a,b=v['data_offsets'];t=torch.frombuffer(bytearray(reader(base+a,b-a)),dtype=_DT[v['dtype']]).reshape(v['shape'] if v['shape'] else [])
        out[k]=t.clone() if dev.type=='cpu' else t.to(dev)
    return out
def _reader_file(f):
    def rd(o,n):
        f.seek(o);return f.read(n)
    return rd
def _load_file(filename,device='cpu'):
    with open(filename,'rb') as f:
        hlen=struct.unpack('<Q',f.read(8))[0];hdr=json.loads(f.read(hlen))
        return _build(hdr,8+hlen,_reader_file(f),device)
def _load(data,device='cpu'):
    buf=data if isinstance(data,(bytes,bytearray)) else bytes(data)
    hlen=struct.unpack('<Q',bytes(buf[:8]))[0];hdr=json.loads(bytes(buf[8:8+hlen]))
    return _build(hdr,8+hlen,lambda o,n:buf[o:o+n],device)
_stt.load_file=_load_file
_stt.load=_load
safetensors.safe_open=_safe_open
_stt.safe_open=_safe_open

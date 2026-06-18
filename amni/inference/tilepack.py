import numpy as np,torch
from pathlib import Path
TS=16
def unpack_tensor(packed,tb,EW,shape):
    R,C=shape;N=R*C;W=1+EW+7
    flat=np.unpackbits(packed)[:N*W].reshape(N,W);codes=np.zeros(N,np.uint64)
    for b in range(W):codes|=(flat[:,b].astype(np.uint64)<<np.uint64(W-1-b))
    sign=(codes>>np.uint64(EW+7)).astype(np.uint32);el=((codes>>np.uint64(7))&np.uint64((1<<EW)-1)).astype(np.int32);man=(codes&np.uint64(0x7F)).astype(np.uint32)
    tbf=np.repeat(np.repeat(tb.astype(np.int32),TS,0),TS,1)[:R,:C].reshape(-1);exp=(el+tbf).astype(np.uint32)
    return ((sign<<15)|(exp<<7)|man).astype(np.uint16).reshape(R,C)
def load_tilepack(bake_dir,info):
    raw=np.fromfile(Path(bake_dir)/info['ptex_path'],dtype=np.uint8,count=int(info['nbytes']))
    if info['fmt']=='raw16':u16=np.ascontiguousarray(raw.view(np.uint16).reshape(info['shape']))
    else:
        pl=int(info['pk_len']);tr,tc=info['tb_shape'];tb=raw[pl:pl+tr*tc].reshape(tr,tc);u16=unpack_tensor(raw[:pl],tb,int(info['EW']),info['shape'])
    return torch.from_numpy(u16.view(np.int16)).view(torch.bfloat16)

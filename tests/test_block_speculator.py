import os,sys
os.environ.setdefault('HIP_VISIBLE_DEVICES','1');os.environ['AMNI_HIP_GEMV_ON']='0';os.environ['AMNI_BLOCK_K']='12'
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(_ROOT))
import numpy as np
from amni.compute.reffelt4 import encode_ids_to_rgba2,decode_rgba2_to_ids,roundtrip_check_ids
def test_codec_roundtrip_full_vocab():
    ids=np.arange(0,100352,dtype=np.int64)
    assert roundtrip_check_ids(ids)
    edge=np.array([0,1,16,17,289,4912,4913,83520,83521,100255,100256,100257,100351],dtype=np.int64)
    assert np.array_equal(edge,decode_rgba2_to_ids(encode_ids_to_rgba2(edge),edge.size))
def test_fnv_stable():
    from amni.inference.block_speculator import fnv1a64
    assert fnv1a64((1,2,3))==fnv1a64((1,2,3)) and fnv1a64((1,2,3))!=fnv1a64((1,2,4))
def test_bank_lookup_roundtrip():
    from amni.inference.block_speculator import PTEXBlockBank
    b=PTEXBlockBank(None,None,h_sizes=(4,3),k_max=8,min_h=3)
    seq=[10,20,30,40,50,60,70,80,90,100]
    b.add_sequence(seq)
    hit=b.lookup([10,20,30,40])
    assert hit is not None and hit[1][0]==50
def test_prune_and_gate():
    from amni.inference.block_speculator import PTEXBlockBank,fnv1a64
    b=PTEXBlockBank(None,None,h_sizes=(4,),k_max=8,min_h=3)
    b._min_tries=16;b._min_ratio=0.5
    b.add_sequence([1,2,3,4,5,6,7,8,9,10,11,12])
    sig=fnv1a64((1,2,3,4))
    assert b.expected_gain_ok(sig) and b.lookup([0,1,2,3,4]) is not None
    b.record_propose(sig,8);b.record_propose(sig,8);b.record_accept(sig,1)
    assert not b.expected_gain_ok(sig) and b.lookup([0,1,2,3,4]) is None
    assert b.prune()>=1 and sig not in b._sig2off
def test_persistence_roundtrip():
    import tempfile,shutil
    from amni.inference.block_speculator import PTEXBlockBank
    d=tempfile.mkdtemp()
    try:
        os.environ['AMNI_BLOCK_PERSIST']='1'
        b=PTEXBlockBank(d,None,h_sizes=(4,3),k_max=8,min_h=3)
        seq=[5,6,7,8,9,10,11,12,13,14,15,16,17,100257,99]
        b.add_sequence(seq);assert b.save()
        b2=PTEXBlockBank(d,None,h_sizes=(4,3),k_max=8,min_h=3)
        h1=b.lookup([5,6,7,8]);h2=b2.lookup([5,6,7,8])
        assert h1 is not None and h2 is not None and list(h1[1])==list(h2[1])
        assert b2._toks==b._toks
    finally:shutil.rmtree(d,ignore_errors=True)
def _bake():
    for c in (_ROOT/'bakes'/'granite41_3b_gf17',Path('bakes/granite41_3b_gf17')):
        if (c/'manifest.json').exists():return str(c)
    return None
def test_exactness_and_acceptance():
    bk=_bake()
    if bk is None:print('SKIP exactness: no bake');return
    import torch
    os.environ['AMNI_BLOCK_SPEC']='1'
    from amni.inference.streaming_chat import StreamingChatService
    from amni.inference.streaming_linear import StreamingLinear
    svc=StreamingChatService(bk,bk,budget_mb=8000);dev=svc.device
    reg=next((m.registry for _,m in svc.model.named_modules() if isinstance(m,StreamingLinear)),None)
    if reg is not None:reg.autosize_budget(cap_bytes=int(14*1024**3));reg.pin_hot();reg.warmup()
    assert svc._block_bank is not None
    txt=svc.tok.apply_chat_template([{'role':'user','content':'Write six Python functions: add, subtract, multiply, divide, modulo, power. Each takes a,b and returns the result with a one-line docstring.'}],tokenize=False,add_generation_prompt=True)
    ii=svc.tok(txt,return_tensors='pt',add_special_tokens=False).input_ids.to(dev)
    am=torch.ones_like(ii)
    with torch.no_grad():base=svc.model.generate(input_ids=ii,attention_mask=am,max_new_tokens=200,do_sample=False,pad_token_id=svc.tok.pad_token_id)
    svc._block_bank.add_sequence(base[0].tolist())
    a0=svc._block_bank.accepted_tokens
    with torch.no_grad():spec=svc.model.generate(input_ids=ii,attention_mask=am,max_new_tokens=200,do_sample=False,pad_token_id=svc.tok.pad_token_id,prompt_lookup_num_tokens=12)
    accepted=svc._block_bank.accepted_tokens-a0
    print(f'[test] base_len={base.shape[1]-ii.shape[1]} spec_len={spec.shape[1]-ii.shape[1]} accepted_tokens={accepted} proposed_steps={svc._block_bank.proposed_steps}')
    assert base.shape==spec.shape and torch.equal(base,spec),'EXACT: block-spec output must equal baseline'
    assert accepted>0,'acceptance must fire (>=1 speculated token accepted)'
    print('[test] EXACT + acceptance fired OK')
if __name__=='__main__':
    test_codec_roundtrip_full_vocab();print('codec OK')
    test_fnv_stable();print('fnv OK')
    test_bank_lookup_roundtrip();print('bank OK')
    test_prune_and_gate();print('prune+gate OK')
    test_persistence_roundtrip();print('persistence OK')
    test_exactness_and_acceptance()

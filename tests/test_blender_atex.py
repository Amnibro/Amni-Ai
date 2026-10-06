import os,sys,json,subprocess,pytest,torch
from pathlib import Path
from amni.serve import blender_atex as B,adam_t_client as C
GPU=pytest.mark.skipif(not torch.cuda.is_available(),reason="needs a GPU")
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("AMNI_BLENDER",raising=False);assert not B.active()
def test_flag_on(monkeypatch):
    monkeypatch.setenv("AMNI_BLENDER","atex:/x");assert B.active()
def test_msgs_single_and_history():
    p=C.template("What is 2+2?",[("hi","hello")],0,(),["Calculator: The answer is 4."]," ")
    M=B.msgs(p);assert M[-1]=={"role":"user","content":"Calculator: The answer is 4. What is 2+2?"} and M[0]=={"role":"user","content":"hi"} and M[1]["content"]=="hello"
def test_notes_router():
    assert B.NOTE.match("Marlowe Bridge: It is a bridge. What is it?") and B.NOTE.match("Calculator: The answer is 4. What?") and not B.NOTE.match("Hi again, Adam, I finished painting.") and not B.NOTE.match("What is 2+2?")
def test_system_prompts_frozen():
    assert set(B.SYSM)>={"alone","split4"} and isinstance(B.SYSM["split4"],tuple) and "not sure" in B.SYSM["split4"][0]
@pytest.mark.parametrize("bits",[4,8])
def test_kv_quant_roundtrip(bits):
    L=B.kvlayer(bits)();torch.manual_seed(0);k=torch.randn(1,2,5,64);v=torch.randn(1,2,5,64);ko,vo=L.update(k,v)
    assert L.get_seq_length()==5 and (ko-k).norm()/k.norm()<(0.15 if bits==4 else 0.02)
    ko2,_=L.update(k[:,:,:1],v[:,:,:1]);assert ko2.shape[-2]==6
def _tiny_bake(tmp_path,n,bits=8,hybrid=False):
    from transformers import LlamaConfig,LlamaForCausalLM,Qwen3_5TextConfig,Qwen3_5ForCausalLM,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers
    torch.manual_seed(n);src=tmp_path/"src";out=tmp_path/"bake";cfg=Qwen3_5TextConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=n,num_attention_heads=4,num_key_value_heads=2,head_dim=64,linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=64,linear_value_head_dim=64,layer_types=[("full_attention" if i%4==3 else "linear_attention") for i in range(n)],tie_word_embeddings=True,eos_token_id=255) if hybrid else LlamaConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=n,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,tie_word_embeddings=False,eos_token_id=255,bos_token_id=254)
    (Qwen3_5ForCausalLM if hybrid else LlamaForCausalLM)(cfg).to(torch.bfloat16).save_pretrained(src);t=Tokenizer(models.WordLevel({f"w{i}":i for i in range(256)},unk_token="w0"));t.pre_tokenizer=pre_tokenizers.Whitespace();PreTrainedTokenizerFast(tokenizer_object=t,eos_token="w255").save_pretrained(src)
    subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/"scripts"/"bake_hybrid1.py"),"--src",str(src),"--out",str(out),"--bits",str(bits),"--gs","64"],check=True,capture_output=True);return str(out)
@GPU
@pytest.mark.parametrize("n",[3,4,5])
def test_stream_ring_bit_exact_vs_resident(tmp_path,n):
    bk=_tiny_bake(tmp_path,n);torch.manual_seed(1);ids=torch.randint(0,250,(1,40));R=[]
    for o in ({},{"stream":True},{"stream":True,"chunk":16},{"chunk":16}):
        e=B.Atex(bk,**o);lg=[];e.generate(ids,12,ign=True,lg_out=lg);R.append((list(e.last_ids),torch.stack(lg)));del e
    assert R[1][0]==R[0][0] and torch.equal(R[1][1],R[0][1]) and R[2][0]==R[3][0] and torch.equal(R[2][1],R[3][1]) and (R[2][1]-R[0][1]).abs().max()<0.05
@GPU
def test_reloads_with_expandable_segments_env_stay_exact():
    r=subprocess.run([sys.executable,str(Path(__file__).with_name("alloc_reload_probe.py")),"8"],env={**os.environ,"PYTORCH_HIP_ALLOC_CONF":"expandable_segments:True"},capture_output=True,text=True);assert json.loads(r.stdout.strip().splitlines()[-1])["corrupt_per_rep"]==[0]*8,r.stdout+r.stderr[-500:]
@GPU
@pytest.mark.parametrize("n",[5,8])
def test_gated_deltanet_hybrid_stream_chunk_kv8(tmp_path,n):
    bk=_tiny_bake(tmp_path,n,hybrid=True);torch.manual_seed(3);ids=torch.randint(0,250,(1,70));R=[]
    for o in ({},{"stream":True},{"chunk":16},{"stream":True,"chunk":16},{"kv":8,"chunk":16,"stream":True}):
        e=B.Atex(bk,**o);lg=[];e.generate(ids,10,ign=True,lg_out=lg);R.append((list(e.last_ids),torch.stack(lg),e.kvb,e.cache()));del e
    lin=[l for l in R[0][3].layers if not isinstance(l,B.DynamicLayer)];assert len(lin)==n-n//4 and all(isinstance(l,B.QKV) for l in R[4][3].layers if isinstance(l,B.DynamicLayer))
    assert R[1][0]==R[0][0] and torch.equal(R[1][1],R[0][1]) and torch.equal(R[3][1],R[2][1]) and (R[2][1]-R[0][1]).abs().max()<0.05 and (R[4][1]-R[2][1]).abs().max()<0.5 and 0<R[4][2]<R[0][2]

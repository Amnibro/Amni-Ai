"""Smoke: QwenAtexChatService from a tiny synthetic Gf17Atex bake (no Qwen3-8B weights, no GPU required)."""
import json,os,sys,tempfile,unittest
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(_ROOT))

from amni.inference.atex_select import (
    is_gf17_atex_bake,is_gf17_atex_manifest,looks_like_qwen3,select_svc_kind,
)

def _write_json(p,obj):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj),encoding='utf-8')


class TestAtexSelect(unittest.TestCase):
    def test_bake_name_qwen3_probe(self):
        self.assertEqual(select_svc_kind('bakes/qwen3_8b_gf17_atex_probe'),'qwen_atex')
        self.assertEqual(select_svc_kind('bakes/qwen3_8b_gf17_atex_probe','Qwen/Qwen3-8B'),'qwen_atex')
        self.assertTrue(looks_like_qwen3('bakes/qwen3_8b_gf17_atex_probe'))
        self.assertTrue(looks_like_qwen3('Qwen/Qwen3-8B'))

    def test_granite_atex_not_stolen(self):
        with tempfile.TemporaryDirectory() as td:
            _write_json(Path(td)/'bake_manifest.json',{'gs':128,'tensors':{'model.layers.0.x.weight':{'q':1,'shape':[8,8],'inf':8}}})
            _write_json(Path(td)/'config.json',{'model_type':'granite','architectures':['GraniteForCausalLM']})
            self.assertEqual(select_svc_kind(td),'granite_atex')

    def test_nvfp4_wins(self):
        with tempfile.TemporaryDirectory() as td:
            _write_json(Path(td)/'bake_manifest.json',{'format':'nvfp4_atex','tensors':{}})
            _write_json(Path(td)/'config.json',{'model_type':'qwen3','architectures':['Qwen3ForCausalLM']})
            self.assertEqual(select_svc_kind(td),'nvfp4')

    def test_qwen_config_json(self):
        with tempfile.TemporaryDirectory() as td:
            _write_json(Path(td)/'config.json',{'model_type':'qwen3','architectures':['Qwen3ForCausalLM']})
            _write_json(Path(td)/'bake_manifest.json',{'gs':128,'tensors':{'a.weight':{'q':1,'shape':[4,4],'inf':4}}})
            self.assertTrue(is_gf17_atex_bake(td))
            self.assertTrue(is_gf17_atex_manifest(json.loads((Path(td)/'bake_manifest.json').read_text())))
            self.assertEqual(select_svc_kind(td),'qwen_atex')

    def test_plain_hf_qwen_name(self):
        self.assertEqual(select_svc_kind('downloaded_models/Qwen3-8B'),'qwen_atex')


def _tiny_qwen3_cfg():
    return {
        'architectures':['Qwen3ForCausalLM'],
        'model_type':'qwen3',
        'hidden_size':128,
        'intermediate_size':256,
        'num_hidden_layers':1,
        'num_attention_heads':4,
        'num_key_value_heads':2,
        'head_dim':32,
        'vocab_size':64,
        'max_position_embeddings':32,
        'rms_norm_eps':1e-6,
        'rope_theta':10000.0,
        'tie_word_embeddings':True,
        'attention_dropout':0.0,
        'hidden_act':'silu',
        'initializer_range':0.02,
        'use_cache':True,
        'bos_token_id':1,
        'eos_token_id':2,
        'pad_token_id':0,
    }


def _write_synthetic_atex_bake(root):
    """Tiny Gf17Atex-shaped dir: fake codes/scale, no real 8B weights."""
    torch= __import__('torch')
    st=__import__('safetensors.torch',fromlist=['save_file'])
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    _write_json(root/'config.json',_tiny_qwen3_cfg())
    _write_json(root/'generation_config.json',{'eos_token_id':2})
    GS=128
    # q_proj 128x128 (divisible by GS); embed + norms bf16
    codes=torch.randint(0,16,(128,128),dtype=torch.uint8)
    scale=torch.full((128,1),0.02,dtype=torch.float32)
    embed=torch.zeros(64,128,dtype=torch.bfloat16)
    sd={
        'model.layers.0.self_attn.q_proj.weight.codes':codes,
        'model.layers.0.self_attn.q_proj.weight.scale':scale,
        'model.embed_tokens.weight':embed,
        'model.layers.0.self_attn.k_proj.weight':torch.zeros(64,128,dtype=torch.bfloat16),
        'model.layers.0.self_attn.v_proj.weight':torch.zeros(64,128,dtype=torch.bfloat16),
        'model.layers.0.self_attn.o_proj.weight':torch.zeros(128,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.gate_proj.weight':torch.zeros(256,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.up_proj.weight':torch.zeros(256,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.down_proj.weight':torch.zeros(128,256,dtype=torch.bfloat16),
        'model.layers.0.self_attn.q_norm.weight':torch.ones(32,dtype=torch.bfloat16),
        'model.layers.0.self_attn.k_norm.weight':torch.ones(32,dtype=torch.bfloat16),
        'model.layers.0.input_layernorm.weight':torch.ones(128,dtype=torch.bfloat16),
        'model.layers.0.post_attention_layernorm.weight':torch.ones(128,dtype=torch.bfloat16),
        'model.norm.weight':torch.ones(128,dtype=torch.bfloat16),
        'lm_head.weight':torch.zeros(64,128,dtype=torch.bfloat16),
    }
    man={'gs':GS,'arch':'Qwen3ForCausalLM','model_type':'qwen3','source':'Qwen/Qwen3-8B','packed':False,'tensors':{
        'model.layers.0.self_attn.q_proj.weight':{'q':1,'shape':[128,128],'inf':128},
        'model.embed_tokens.weight':{'q':0,'shape':[64,128]},
        'model.layers.0.self_attn.k_proj.weight':{'q':0,'shape':[64,128]},
        'model.layers.0.self_attn.v_proj.weight':{'q':0,'shape':[64,128]},
        'model.layers.0.self_attn.o_proj.weight':{'q':0,'shape':[128,128]},
        'model.layers.0.mlp.gate_proj.weight':{'q':0,'shape':[256,128]},
        'model.layers.0.mlp.up_proj.weight':{'q':0,'shape':[256,128]},
        'model.layers.0.mlp.down_proj.weight':{'q':0,'shape':[128,256]},
        'model.layers.0.self_attn.q_norm.weight':{'q':0,'shape':[32]},
        'model.layers.0.self_attn.k_norm.weight':{'q':0,'shape':[32]},
        'model.layers.0.input_layernorm.weight':{'q':0,'shape':[128]},
        'model.layers.0.post_attention_layernorm.weight':{'q':0,'shape':[128]},
        'model.norm.weight':{'q':0,'shape':[128]},
        'lm_head.weight':{'q':0,'shape':[64,128]},
    }}
    _write_json(root/'bake_manifest.json',man)
    st.save_file(sd,str(root/'model.safetensors'))
    return root


def _write_synthetic_hf(root):
    torch=__import__('torch')
    st=__import__('safetensors.torch',fromlist=['save_file'])
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    _write_json(root/'config.json',_tiny_qwen3_cfg())
    sd={
        'model.embed_tokens.weight':torch.zeros(64,128,dtype=torch.bfloat16),
        'model.layers.0.self_attn.q_proj.weight':torch.randn(128,128,dtype=torch.float32)*0.02,
        'model.layers.0.self_attn.k_proj.weight':torch.zeros(64,128,dtype=torch.bfloat16),
        'model.layers.0.self_attn.v_proj.weight':torch.zeros(64,128,dtype=torch.bfloat16),
        'model.layers.0.self_attn.o_proj.weight':torch.zeros(128,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.gate_proj.weight':torch.zeros(256,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.up_proj.weight':torch.zeros(256,128,dtype=torch.bfloat16),
        'model.layers.0.mlp.down_proj.weight':torch.zeros(128,256,dtype=torch.bfloat16),
        'model.layers.0.self_attn.q_norm.weight':torch.ones(32,dtype=torch.bfloat16),
        'model.layers.0.self_attn.k_norm.weight':torch.ones(32,dtype=torch.bfloat16),
        'model.layers.0.input_layernorm.weight':torch.ones(128,dtype=torch.bfloat16),
        'model.layers.0.post_attention_layernorm.weight':torch.ones(128,dtype=torch.bfloat16),
        'model.norm.weight':torch.ones(128,dtype=torch.bfloat16),
        'lm_head.weight':torch.zeros(64,128,dtype=torch.bfloat16),
    }
    st.save_file(sd,str(root/'model.safetensors'))
    return root


def _torch_ok():
    try:
        import torch,transformers,accelerate,safetensors  # noqa: F401
        from transformers import Qwen3Config  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_torch_ok(),'torch/transformers/Qwen3 not installed')
class TestQwenAtexServiceFixture(unittest.TestCase):
    def test_construct_from_synthetic_atex_bake(self):
        import torch
        from amni.inference.qwen_atex_svc import AtexLin,QwenAtexChatService
        with tempfile.TemporaryDirectory() as td:
            bake=_write_synthetic_atex_bake(td)
            self.assertTrue(is_gf17_atex_bake(bake))
            self.assertEqual(select_svc_kind(str(bake)),'qwen_atex')
            svc=QwenAtexChatService(path=str(bake))
            self.assertEqual(svc.source,'gf17_atex_bake')
            self.assertTrue(callable(svc.chat))
            self.assertTrue(callable(svc.chat_stream))
            q=svc.m.model.layers[0].self_attn.q_proj
            self.assertIsInstance(q,AtexLin)
            self.assertEqual(tuple(q.codes.shape),(128,128))
            self.assertEqual(tuple(q.scale.shape),(128,1))
            self.assertGreaterEqual(svc.n_quant,1)
            W=q.dequant()
            self.assertEqual(tuple(W.shape),(128,128))
            if not torch.cuda.is_available():
                x=torch.zeros(1,4,128)
                y=q(x)
                self.assertEqual(tuple(y.shape),(1,4,128))
                with self.assertRaises(RuntimeError):
                    svc.chat('hi',max_new_tokens=1)
            else:
                r,n=svc.chat('hi',max_new_tokens=2)
                self.assertIsInstance(r,str)
                self.assertGreaterEqual(n,0)

    def test_construct_from_plain_hf_dir(self):
        from amni.inference.qwen_atex_svc import AtexLin,QwenAtexChatService
        with tempfile.TemporaryDirectory() as td:
            hf=_write_synthetic_hf(td)
            self.assertFalse(is_gf17_atex_bake(hf))
            self.assertEqual(select_svc_kind(str(hf)),'qwen_atex')
            svc=QwenAtexChatService(path=str(hf))
            self.assertEqual(svc.source,'hf_int4_pack')
            q=svc.m.model.layers[0].self_attn.q_proj
            self.assertIsInstance(q,AtexLin)
            self.assertGreaterEqual(svc.n_quant,1)
            # embed stays dense (not AtexLin)
            self.assertFalse(isinstance(svc.m.model.embed_tokens,AtexLin))

    def test_chat_signature_matches_granite(self):
        import inspect
        from amni.inference.qwen_atex_svc import QwenAtexChatService
        from amni.inference.granite_atex_svc import GraniteAtexChatService
        g=inspect.signature(GraniteAtexChatService.chat)
        q=inspect.signature(QwenAtexChatService.chat)
        for name in ('user_msg','system','history','facts','max_new_tokens','do_sample'):
            self.assertIn(name,q.parameters)
            self.assertIn(name,g.parameters)


if __name__=='__main__':
    unittest.main()

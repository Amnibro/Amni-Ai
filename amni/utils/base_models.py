"""Supported upstream base models: one table every subsystem reads (installer, bake script, loader, registry, Ollama/OpenAI listings, VRAM tiers).
Facts below were read from the published Hugging Face config.json / tokenizer_config.json / chat_template.jinja (2026-11-16), not assumed:
- ibm-granite/granite-4.2-3b: GraniteForCausalLM, 40 layers, hidden 2560, GQA 40q/8kv, vocab 100352, untied embeddings, rope_theta 1e7,
  embedding/residual/logits multipliers all 1.0 (4.1 used 12.0/0.22/10.0). Chat format switched from <|start_of_role|>..<|end_of_text|> to ChatML
  <|im_start|>/<|im_end|> (eos id 100257 = <|im_end|>). Reasoning ON by default: generation prompt opens '<think>\\n'; enable_thinking=False emits '<think></think>'.
  <think>/</think> are NOT special tokens, so skip_special_tokens does not remove them -> callers must strip them (strip_thinking()).
- Qwen/Qwen3.8-27B: Qwen3_5ForConditionalGeneration (model_type qwen3_5, text model_type qwen3_5_text) - same hybrid Gated-DeltaNet (linear_attention x3 + full_attention x1)
  family as Qwen3.5, scaled to 64 layers / hidden 5120 / 24q-4kv heads, explicit output_gate_type=swish and partial_rotary_factor=0.25, untied, vocab 248320.
  eos ids [248046 <|im_end|>, 248044 <|endoftext|>]; tool calls are XML '<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>'.
"""
import os,re
from typing import Dict,List,Optional,Any
BASE_MODELS:List[Dict[str,Any]]=[
    {'key':'granite-4.2-3b','hf_id':'ibm-granite/granite-4.2-3b','gguf':['ibm-granite/granite-4.2-3b-GGUF','bartowski/granite-4.2-3b-GGUF'],'family':'granite','arch':'GraniteForCausalLM',
     'params_b':3.0,'layers':40,'hidden':2560,'ctx':131072,'chat_format':'chatml','thinking':True,'think_default':True,'stop':['<|im_end|>'],'tool_format':'json',
     'bake_dir':'granite42_3b_gf17_atex','ollama':'adam:granite4.2-3b','aliases':['granite-4.2-3b','granite4.2','granite4.2:3b','granite-4.2','granite42-3b','granite'],
     'vram_gb':{'bf16':7.5,'atex_int4':2.6,'gguf_q4_k_m':2.4},'ram_gb_cpu':8,'default':True,'status':'current','released':'2026-08-25'},
    {'key':'qwen3.8-27b','hf_id':'Qwen/Qwen3.8-27B','gguf':['unsloth/Qwen3.8-27B-GGUF'],'fp8':'Qwen/Qwen3.8-27B-FP8','family':'qwen3_5','arch':'Qwen3_5ForConditionalGeneration',
     'params_b':27.0,'layers':64,'hidden':5120,'ctx':262144,'chat_format':'chatml','thinking':True,'think_default':True,'stop':['<|im_end|>','<|endoftext|>'],'tool_format':'xml',
     'bake_dir':'qwen38_27b_gf17_atex','ollama':'adam:qwen3.8-27b','aliases':['qwen3.8-27b','qwen3.8','qwen3.8:27b','qwen38-27b','qwen-27b'],
     'vram_gb':{'bf16':56.0,'fp8':30.0,'atex_int4':19.0,'gguf_q4_k_m':17.5},'ram_gb_cpu':32,'default':False,'status':'current','released':'2026-08'},
    {'key':'granite-4.1-8b','hf_id':'ibm-granite/granite-4.1-8b','gguf':[],'family':'granite','arch':'GraniteForCausalLM','params_b':8.0,'chat_format':'granite_roles','thinking':False,'think_default':False,
     'stop':['<|end_of_text|>'],'tool_format':'json','bake_dir':'granite41_8b_gf17_atex','ollama':'adam:granite4.1-8b','aliases':['granite-4.1-8b','granite41-8b'],
     'vram_gb':{'atex_int4':6.0},'ram_gb_cpu':16,'default':False,'status':'legacy'},
    {'key':'granite-4.1-3b','hf_id':'ibm-granite/granite-4.1-3b','gguf':[],'family':'granite','arch':'GraniteForCausalLM','params_b':3.0,'chat_format':'granite_roles','thinking':False,'think_default':False,
     'stop':['<|end_of_text|>'],'tool_format':'json','bake_dir':'granite41_3b_gf17_atex','ollama':'adam:granite4.1-3b','aliases':['granite-4.1-3b','granite41-3b','granite4.1'],
     'vram_gb':{'atex_int4':2.6,'palette':4.0},'ram_gb_cpu':8,'default':False,'status':'legacy'},
    {'key':'qwen3.5-9b','hf_id':'Qwen/Qwen3.5-9B','gguf':[],'family':'qwen3_5','arch':'Qwen3_5ForConditionalGeneration','params_b':9.0,'chat_format':'chatml','thinking':True,'think_default':True,
     'stop':['<|im_end|>','<|endoftext|>'],'tool_format':'xml','bake_dir':'adam-qwen35-9b','ollama':'adam:qwen3.5-9b','aliases':['qwen3.5-9b','qwen35-9b'],
     'vram_gb':{'gguf_q4_k_m':5.5,'gguf_q8_0':9.5},'ram_gb_cpu':16,'default':False,'status':'legacy'},
]
_BY_KEY={m['key']:m for m in BASE_MODELS}
def default_model()->Dict[str,Any]:
    k=os.environ.get('AMNI_BASE_MODEL','').strip()
    return resolve(k) or next(m for m in BASE_MODELS if m.get('default'))
def resolve(name:Optional[str])->Optional[Dict[str,Any]]:
    """Map a key, HF id, Ollama tag, alias or local dir name (case-insensitive) to its BASE_MODELS entry."""
    if not name:return None
    raw=str(name).strip().lower().rstrip('/')
    cands=[raw,raw.split('/')[-1]]
    cands=[c.replace('-gguf','').replace('-fp8','') for c in cands]
    for m in BASE_MODELS:
        names={m['key'],m['hf_id'].lower(),m['hf_id'].lower().split('/')[-1],m['ollama'].lower(),m['bake_dir'].lower(),*[a.lower() for a in m['aliases']],*[g.lower().replace('-gguf','') for g in m.get('gguf',[])]}
        if m.get('fp8'):names.add(m['fp8'].lower().replace('-fp8',''))
        if any(c in names for c in cands):return m
    return None
def detect_from_config(cfg:Any)->Optional[Dict[str,Any]]:
    """Identify a base model from a transformers config (or dict). Granite 4.2 vs 4.1 differ by bos id / multipliers, Qwen3.8-27B vs 3.5 by depth/width."""
    g=(lambda k,d=None:cfg.get(k,d)) if isinstance(cfg,dict) else (lambda k,d=None:getattr(cfg,k,d))
    mt=str(g('model_type','') or '').lower()
    tc=g('text_config');tg=(lambda k,d=None:(tc.get(k,d) if isinstance(tc,dict) else getattr(tc,k,d))) if tc is not None else g
    if mt=='granite':
        if int(g('bos_token_id',0) or 0)==100283 or float(g('logits_scaling',1.0) or 1.0)==1.0:return _BY_KEY['granite-4.2-3b'] if int(g('hidden_size',0) or 0)==2560 else None
        return _BY_KEY['granite-4.1-3b'] if int(g('hidden_size',0) or 0)==2560 else _BY_KEY['granite-4.1-8b']
    if mt.startswith('qwen3_5'):
        h=int(tg('hidden_size',0) or 0);L=int(tg('num_hidden_layers',0) or 0)
        if h==5120 and L==64:return _BY_KEY['qwen3.8-27b']
        if h==4096:return _BY_KEY['qwen3.5-9b']
    return None
def thinking_enabled(model:Optional[Dict[str,Any]]=None)->bool:
    """Reasoning is OFF by default for chat (fast replies; small models burn the token budget inside <think>). AMNI_THINKING=1 turns it on."""
    v=os.environ.get('AMNI_THINKING','').strip().lower()
    if v in ('1','true','yes','on'):return True
    return False
def chat_template_kwargs(tok:Any=None)->Dict[str,Any]:
    """Extra kwargs for tokenizer.apply_chat_template. Only passes enable_thinking when the template actually reads it (Granite 4.2, Qwen3.5/3.8),
    so older templates render byte-identically."""
    tpl=getattr(tok,'chat_template',None) if tok is not None else None
    if isinstance(tpl,dict):tpl=' '.join(str(v) for v in tpl.values())
    if tpl is None or 'enable_thinking' in str(tpl):return {'enable_thinking':thinking_enabled()}
    return {}
_THINK_RE=re.compile(r'<think>.*?</think>',re.S)
def strip_thinking(text:Optional[str])->str:
    """Remove <think>..</think> reasoning (and an unterminated trailing <think> block, or a leading orphan '</think>' left when the prompt opened the block)."""
    if not text:return text or ''
    t=_THINK_RE.sub('',text)
    if '</think>' in t:t=t.split('</think>')[-1]
    if '<think>' in t:t=t.split('<think>')[0]
    return t.strip()
_QWEN_TOOL_RE=re.compile(r'<tool_call>\s*<function=(?P<name>[^>\n]+)>(?P<body>.*?)(?:</function>\s*)?(?:</tool_call>|$)',re.S)
_QWEN_PARAM_RE=re.compile(r'<parameter=(?P<key>[^>\n]+)>\n?(?P<value>.*?)\n?(?:</parameter>|(?=<parameter=)|(?=</function>)|$)',re.S)
def parse_xml_tool_calls(text:str)->List[Dict[str,Any]]:
    """Parse Qwen3.5/3.8 XML-style tool calls into [{'name':..,'arguments':{..}}]. JSON-style (Granite) calls are handled by the existing parser."""
    import json as _j
    out=[]
    for m in _QWEN_TOOL_RE.finditer(text or ''):
        args={}
        for p in _QWEN_PARAM_RE.finditer(m.group('body')):
            v=p.group('value').strip()
            try:args[p.group('key').strip()]=_j.loads(v)
            except Exception:args[p.group('key').strip()]=v
        out.append({'name':m.group('name').strip(),'arguments':args})
    return out
def ollama_tags()->List[str]:
    return [m['ollama'] for m in BASE_MODELS if m['status']=='current']
def summary()->List[Dict[str,Any]]:
    return [{k:m.get(k) for k in ('key','hf_id','gguf','params_b','status','default','thinking','vram_gb','ram_gb_cpu','bake_dir','ollama')} for m in BASE_MODELS]

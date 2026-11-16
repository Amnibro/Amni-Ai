"""Fast, network-free checks for the v6.21.0 model registry + offline switch. Run: python -m pytest tests/test_base_models_offline.py -q   (tests/ is gitignored by repo policy: git add -f to ship it)"""
import os,sys,time,importlib
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amni.utils import base_models as bm
# Values copied from the published config.json files (2026-11-16).
GRANITE42={'model_type':'granite','architectures':['GraniteForCausalLM'],'hidden_size':2560,'num_hidden_layers':40,'bos_token_id':100283,'eos_token_id':100257,'logits_scaling':1.0,'embedding_multiplier':1.0,'residual_multiplier':1.0,'tie_word_embeddings':False}
GRANITE41={'model_type':'granite','architectures':['GraniteForCausalLM'],'hidden_size':2560,'num_hidden_layers':40,'bos_token_id':100257,'eos_token_id':100257,'logits_scaling':10.0,'embedding_multiplier':12.0,'residual_multiplier':0.22,'tie_word_embeddings':True}
QWEN38={'model_type':'qwen3_5','architectures':['Qwen3_5ForConditionalGeneration'],'text_config':{'model_type':'qwen3_5_text','hidden_size':5120,'num_hidden_layers':64}}
QWEN35_9B={'model_type':'qwen3_5','architectures':['Qwen3_5ForConditionalGeneration'],'text_config':{'model_type':'qwen3_5_text','hidden_size':4096,'num_hidden_layers':32}}
def test_default_is_granite42():
    os.environ.pop('AMNI_BASE_MODEL',None)
    assert bm.default_model()['key']=='granite-4.2-3b'
def test_detect_from_config():
    assert bm.detect_from_config(GRANITE42)['key']=='granite-4.2-3b'
    assert bm.detect_from_config(GRANITE41)['key']=='granite-4.1-3b'
    assert bm.detect_from_config(QWEN38)['key']=='qwen3.8-27b'
    assert bm.detect_from_config(QWEN35_9B)['key']=='qwen3.5-9b'
def test_resolve_aliases():
    for name,key in [('ibm-granite/granite-4.2-3b','granite-4.2-3b'),('bartowski/granite-4.2-3b-GGUF','granite-4.2-3b'),('adam:granite4.2-3b','granite-4.2-3b'),
                     ('Qwen/Qwen3.8-27B','qwen3.8-27b'),('Qwen/Qwen3.8-27B-FP8','qwen3.8-27b'),('unsloth/Qwen3.8-27B-GGUF','qwen3.8-27b'),('qwen3.8','qwen3.8-27b'),('granite42_3b_gf17_atex','granite-4.2-3b')]:
        assert bm.resolve(name)['key']==key,name
    assert bm.resolve('llama-9000') is None
def test_strip_thinking():
    assert bm.strip_thinking('<think>\nlet me see\n</think>\n\n391')=='391'
    assert bm.strip_thinking('reasoning that started in the prompt</think>Paris')=='Paris'
    assert bm.strip_thinking('Answer<think>unterminated')=='Answer'
    assert bm.strip_thinking('plain')=='plain'
def test_chat_template_kwargs_only_when_template_uses_it():
    class T:pass
    old=T();old.chat_template="{{ '<|start_of_role|>' }}"
    new=T();new.chat_template="{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
    os.environ.pop('AMNI_THINKING',None)
    assert bm.chat_template_kwargs(old)=={}
    assert bm.chat_template_kwargs(new)=={'enable_thinking':False}
    os.environ['AMNI_THINKING']='1'
    try:assert bm.chat_template_kwargs(new)=={'enable_thinking':True}
    finally:os.environ.pop('AMNI_THINKING',None)
def test_qwen_xml_tool_calls():
    t='<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>'
    assert bm.parse_xml_tool_calls(t)==[{'name':'get_weather','arguments':{'city':'Paris','days':3}}]
def test_listings_include_new_models():
    from amni.serve import ollama_compat,openai_compat
    assert 'adam:granite4.2-3b' in ollama_compat._MODEL_NAMES and 'adam:qwen3.8-27b' in ollama_compat._MODEL_NAMES
    ids=[m['id'] for m in openai_compat.models_list()['data']]
    assert 'granite-4.2-3b' in ids and 'qwen3.8-27b' in ids
def _fresh_offline():
    import amni.utils.offline as off
    return importlib.reload(off)
def test_offline_flag_sets_hf_env(monkeypatch):
    for k in ('AMNI_OFFLINE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):monkeypatch.delenv(k,raising=False)
    off=_fresh_offline()
    assert off.init_from_args(True) is True
    assert os.environ['HF_HUB_OFFLINE']=='1' and os.environ['TRANSFORMERS_OFFLINE']=='1' and off.is_offline()
def test_offline_autodetect_is_fast(monkeypatch):
    monkeypatch.delenv('AMNI_OFFLINE',raising=False)
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:9')  # nothing listens on the discard port -> refused
    off=_fresh_offline()
    t=time.time();res=off.is_offline(refresh=True);dt=time.time()-t
    assert res is True and dt<3.0
def test_offline_forced_off(monkeypatch):
    monkeypatch.setenv('AMNI_OFFLINE','0')
    off=_fresh_offline()
    assert off.is_offline() is False
def test_network_skill_fails_soft(monkeypatch):
    monkeypatch.setenv('AMNI_OFFLINE','1')
    _fresh_offline()
    from amni.serve.skills import SkillRegistry
    reg=SkillRegistry(workdir='.')
    called=[]
    reg.register('web',lambda a,c,r:called.append(1) or 'x',desc='t')
    t=time.time();r=reg.call('web',{'query':'q'});dt=time.time()-t
    assert not r.ok and 'offline' in (r.error or '') and not called and dt<1.0
def test_tool_protocol_parses_native_formats():
    from amni.serve.tool_protocol import parse_tool_calls,strip_tool_calls
    q='Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>'
    r=parse_tool_calls(q);assert [(c['name'],c['arguments']) for c in r]==[('get_weather',{'city':'Paris'})]
    g='<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>'  # Granite 4.2 native JSON
    r=parse_tool_calls(g);assert [(c['name'],c['arguments']) for c in r]==[('get_weather',{'city':'Rome'})]
    assert strip_tool_calls(q)=='Let me check.'
    f='```tool_call\n{"name":"calc","arguments":{"expr":"2+2"}}\n```'  # Adam's own fenced grammar unchanged
    assert parse_tool_calls(f)[0]['name']=='calc'
def test_goal_stops_on_repeated_step_and_formats_readably():
    from amni.serve.agentic import run_goal
    from amni.serve.skills import SkillRegistry
    from amni.serve.agent import AmniAgent
    class Svc:
        n=0
        def chat(self,*a,**k):Svc.n+=1;return '{"tool":"echo","args":{"code":"print(1)"}}',10
    class Inner:svc=Svc()
    class A:adam=Inner()
    reg=SkillRegistry(workdir='.');reg.register('echo',lambda a,c,r:{'stdout':'1\n','stderr':'','returncode':0},desc='t')
    out=run_goal(A(),reg,'print one',max_steps=5)
    assert out['n_steps']==1 and out['stop_reason']=='repeated step (done)' and Svc.n==2
    txt=AmniAgent._format_skill_output(None,'goal',out)
    assert '```python' in txt and 'print(1)' in txt and 'Output:' in txt and '{"goal"' not in txt

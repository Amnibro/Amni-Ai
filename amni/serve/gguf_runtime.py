"""Talk to a GGUF server (llama-server or Ollama) with Adam's prompt + facts.

Does not replace the Granite bake. Activate with AMNI_GEN_BACKEND=gguf
or POST /models/activate.
"""
import json,os,urllib.request,urllib.error
from typing import Dict,Iterator,List,Optional,Tuple
def backend_name()->str:
    return (os.environ.get('AMNI_GEN_BACKEND') or os.environ.get('AMNI_GGUF_BACKEND') or 'bake').strip().lower()
def enabled()->bool:
    return backend_name() in ('gguf','ollama','llama')
def server_url()->str:
    if backend_name()=='ollama':
        return (os.environ.get('AMNI_OLLAMA_URL') or 'http://127.0.0.1:11434').rstrip('/')
    return (os.environ.get('AMNI_GGUF_URL') or os.environ.get('LLAMA_SERVER_URL') or 'http://127.0.0.1:8787').rstrip('/')
def model_id()->str:
    return (os.environ.get('AMNI_GGUF_MODEL') or os.environ.get('AMNI_OLLAMA_MODEL') or 'qwen38-27b-aggressive').strip()
def _post(url:str,body:dict,timeout:int=300)->dict:
    data=json.dumps(body).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))
def health()->Dict:
    url=server_url()
    try:
        req=urllib.request.Request(url+'/health' if backend_name()!='ollama' else url+'/api/tags')
        with urllib.request.urlopen(req,timeout=3) as resp:
            raw=resp.read().decode('utf-8')
        return {'ok':True,'url':url,'backend':backend_name(),'model':model_id(),'probe':raw[:200]}
    except Exception as e:
        return {'ok':False,'url':url,'backend':backend_name(),'model':model_id(),'error':str(e)[:200]}
def _messages(system:str,message:str,history:Optional[List[Tuple[str,str]]],facts:Optional[List[str]])->List[Dict]:
    sys=system or 'You are Adam.'
    if facts:
        sys+='\n\nFacts below this line are true — use them.\n'+'\n'.join(f'- {f}' for f in facts[:8])
    msgs=[{'role':'system','content':sys}]
    for u,a in (history or [])[-6:]:
        msgs.append({'role':'user','content':u})
        msgs.append({'role':'assistant','content':a})
    msgs.append({'role':'user','content':message})
    return msgs
def chat(message:str,system:str='',history=None,facts=None,max_new_tokens:int=512,do_sample:bool=True)->Dict:
    url=server_url()+'/v1/chat/completions'
    body={
        'model':model_id(),
        'messages':_messages(system,message,history,facts),
        'max_tokens':int(max_new_tokens),
        'temperature':0.7 if do_sample else 0.0,
        'top_p':0.8,
        'chat_template_kwargs':{'enable_thinking':False},
    }
    r=_post(url,body)
    choice=(r.get('choices') or [{}])[0]
    ans=((choice.get('message') or {}).get('content') or '').strip()
    usage=r.get('usage') or {}
    return {'answer':ans,'tier':'tier_gguf','tokens':int(usage.get('completion_tokens') or 0),'raw':r}
def chat_stream(message:str,system:str='',history=None,facts=None,max_new_tokens:int=512,do_sample:bool=True)->Iterator[str]:
    url=server_url()+'/v1/chat/completions'
    body={
        'model':model_id(),
        'messages':_messages(system,message,history,facts),
        'max_tokens':int(max_new_tokens),
        'temperature':0.7 if do_sample else 0.0,
        'stream':True,
        'chat_template_kwargs':{'enable_thinking':False},
    }
    data=json.dumps(body).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=300) as resp:
        buf=b''
        while True:
            chunk=resp.read(256)
            if not chunk:break
            buf+=chunk
            while b'\n' in buf:
                line,buf=buf.split(b'\n',1)
                s=line.decode('utf-8','ignore').strip()
                if not s.startswith('data:'):continue
                payload=s[5:].strip()
                if payload=='[DONE]':return
                try:obj=json.loads(payload)
                except Exception:continue
                delta=((obj.get('choices') or [{}])[0].get('delta') or {})
                t=delta.get('content') or ''
                if t:yield t
def ollama_modelfile(gguf_path:str,name:str='qwen38-27b-aggressive')->str:
    return f'FROM {gguf_path}\nPARAMETER temperature 0.7\nPARAMETER top_p 0.8\nPARAMETER top_k 20\nPARAMETER num_ctx 8192\n'
def activate(gguf_path:str,backend:str='gguf',model:str='qwen38-27b-aggressive',persist:bool=True)->Dict:
    os.environ['AMNI_GEN_BACKEND']=backend
    os.environ['AMNI_GGUF_MODEL']=model
    os.environ['AMNI_ACTIVE_GGUF']=gguf_path
    if persist:
        try:
            from amni.bootstrap import load_config,save_config
            cfg=load_config();cfg['gen_backend']=backend;cfg['active_gguf']=gguf_path;save_config(cfg)
        except Exception as e:
            return {'ok':True,'persisted':False,'error':str(e)[:160],'backend':backend,'model':model,'path':gguf_path}
    return {'ok':True,'persisted':bool(persist),'backend':backend,'model':model,'path':gguf_path}

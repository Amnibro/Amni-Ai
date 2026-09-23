"""Talk to a GGUF server (llama-server or Ollama) with Adam's prompt + facts.

Does not replace the Granite bake. Activate with AMNI_GEN_BACKEND=gguf
or POST /models/activate.
"""
import json,os,urllib.request,urllib.error
from typing import Dict,Iterator,List,Optional,Tuple
def backend_name()->str:
    return (os.environ.get('AMNI_GEN_BACKEND') or os.environ.get('AMNI_GGUF_BACKEND') or 'bake').strip().lower()
def enabled()->bool:
    return backend_name() in ('gguf','ollama','llama','ray')
def server_url()->str:
    if backend_name()=='ollama':
        return (os.environ.get('AMNI_OLLAMA_URL') or 'http://127.0.0.1:11434').rstrip('/')
    return (os.environ.get('AMNI_GGUF_URL') or os.environ.get('LLAMA_SERVER_URL') or 'http://127.0.0.1:8787').rstrip('/')
def model_id()->str:
    if backend_name()=='ray':return (os.environ.get('AMNI_RAY_MODEL') or 'adam:ray-v16').strip()
    return (os.environ.get('AMNI_GGUF_MODEL') or os.environ.get('AMNI_OLLAMA_MODEL') or 'qwen38-27b-aggressive').strip()
def _post(url:str,body:dict,timeout:int=300)->dict:
    data=json.dumps(body).encode('utf-8')
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))
def health()->Dict:
    if backend_name()=='ray':
        from amni.compute.ray_field import engine
        e=engine();return {'ok':os.path.exists(e.path),'backend':'ray','model':model_id(),'pack':e.path,'loaded':e.ready,'arch':e.arch}
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
_qa=[None]
def _ray_guide(e,message:str):
    from amni.compute.ray_qa import QA,guide,DEFAULT_QA
    p=os.environ.get('AMNI_RAY_QA',DEFAULT_QA)
    if not os.path.exists(p):return None
    _qa[0]=_qa[0] or QA(p);return guide(e,_qa[0],message)
def _ray_stream(message:str,history=None,max_new_tokens:int=512,do_sample:bool=True)->Iterator[str]:
    from amni.compute.ray_field import engine
    h=(history or [])[-1:];prompt=''.join(f'User: {u.strip()}\nAdam: {a.strip()}\n\n' for u,a in h)+f'User: {message.strip()}\nAdam:'
    e=engine();g=_ray_guide(e,message)
    if g is None and os.environ.get('AMNI_RAY_QA_DECLINE','1')=='1':yield "I'm not sure about that one yet.";return
    yield from e.stream(prompt,guide=(b' '+g) if g else None,beta=float(os.environ.get('AMNI_RAY_BETA','16')),max_bytes=max(16,min(max(int(max_new_tokens),900),int(os.environ.get('AMNI_RAY_MAX_BYTES','900')))),temp=float(os.environ.get('AMNI_RAY_TEMP','0.7')) if do_sample else 0.35,topk=int(os.environ.get('AMNI_RAY_TOPK','12')),stop=('\n\n','\nUser:','\nAdam:'))
def chat(message:str,system:str='',history=None,facts=None,max_new_tokens:int=512,do_sample:bool=True)->Dict:
    if backend_name()=='ray':
        t=''.join(_ray_stream(message,history,max_new_tokens,do_sample));return {'answer':t.strip(),'tier':'tier_ray','tokens':len(t.encode('utf-8'))}
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
    if backend_name()=='ray':
        yield from _ray_stream(message,history,max_new_tokens,do_sample);return
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
    if backend=='ray':os.environ['AMNI_RAY_PACK']=gguf_path;os.environ['AMNI_RAY_MODEL']=model
    os.environ['AMNI_GGUF_MODEL']=model
    os.environ['AMNI_ACTIVE_GGUF']=gguf_path
    if persist:
        try:
            from amni.bootstrap import load_config,save_config
            cfg=load_config();cfg['gen_backend']=backend;cfg['active_ray' if backend=='ray' else 'active_gguf']=gguf_path;save_config(cfg)
        except Exception as e:
            return {'ok':True,'persisted':False,'error':str(e)[:160],'backend':backend,'model':model,'path':gguf_path}
    return {'ok':True,'persisted':bool(persist),'backend':backend,'model':model,'path':gguf_path}
class Svc:
    def chat(self,user_msg:str,system=None,history=None,facts=None,max_new_tokens:int=80,do_sample:bool=False,**kw)->Tuple[str,int]:
        r=chat(user_msg,system=system or '',history=history,facts=facts,max_new_tokens=max_new_tokens,do_sample=do_sample);return (r.get('answer') or ''),int(r.get('tokens') or 0)
    def chat_stream(self,user_msg:str,system=None,history=None,facts=None,max_new_tokens:int=80,do_sample:bool=False,**kw)->Iterator[str]:
        yield from chat_stream(user_msg,system=system or '',history=history,facts=facts,max_new_tokens=max_new_tokens,do_sample=do_sample)
def svc()->Optional[Svc]:
    return Svc() if enabled() else None

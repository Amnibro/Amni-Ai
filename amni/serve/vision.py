"""VisionService — bolt-on vision encoder so Adam handles image input. Lazy-loads BLIP-base (~470MB, downloads on first use via HF). Optional dep: transformers + Pillow + torch. If unavailable, every method returns a clean error dict — never crashes the whole server.
describe(image_bytes)                   — caption: "a dog playing in the snow"
caption_with_question(image, question)  — VQA: "what color is the dog?" → "brown and white"
Two backends are tried in order: blip-vqa-base for Q&A, blip-image-captioning-base for free-form caption. Both run on CPU by default; CUDA if available."""
import io,time,threading
from typing import Dict,Any,Optional,Tuple
_DEFAULT_CAPTION_MODEL='Salesforce/blip-image-captioning-base'
_DEFAULT_VQA_MODEL='Salesforce/blip-vqa-base'
class VisionService:
    def __init__(self,caption_model:str=_DEFAULT_CAPTION_MODEL,vqa_model:str=_DEFAULT_VQA_MODEL,device:Optional[str]=None):
        self.caption_model_name=caption_model;self.vqa_model_name=vqa_model
        self.device=device;self._cap_proc=None;self._cap_mdl=None;self._vqa_proc=None;self._vqa_mdl=None
        self._lock=threading.Lock();self._available=None;self.init_error=None
    def _try_init_torch(self):
        try:
            import torch as _t
            self._torch=_t
            if self.device is None:self.device='cuda' if _t.cuda.is_available() else 'cpu'
            return True
        except Exception as e:self.init_error=f'torch unavailable: {e}';return False
    def _ensure_caption(self)->bool:
        if self._cap_mdl is not None:return True
        with self._lock:
            if self._cap_mdl is not None:return True
            if not self._try_init_torch():return False
            try:
                from transformers import BlipProcessor,BlipForConditionalGeneration
                self._cap_proc=BlipProcessor.from_pretrained(self.caption_model_name)
                self._cap_mdl=BlipForConditionalGeneration.from_pretrained(self.caption_model_name).to(self.device).eval()
                return True
            except Exception as e:self.init_error=f'caption model load failed: {type(e).__name__}: {e}';return False
    def _ensure_vqa(self)->bool:
        if self._vqa_mdl is not None:return True
        with self._lock:
            if self._vqa_mdl is not None:return True
            if not self._try_init_torch():return False
            try:
                from transformers import BlipProcessor,BlipForQuestionAnswering
                self._vqa_proc=BlipProcessor.from_pretrained(self.vqa_model_name)
                self._vqa_mdl=BlipForQuestionAnswering.from_pretrained(self.vqa_model_name).to(self.device).eval()
                return True
            except Exception as e:self.init_error=f'vqa model load failed: {type(e).__name__}: {e}';return False
    def is_available(self)->bool:
        if self._available is not None:return self._available
        try:
            from transformers import BlipProcessor;from PIL import Image
            self._available=True
        except Exception as e:self._available=False;self.init_error=f'optional deps missing: {e}'
        return self._available
    def _open_image(self,image_bytes:bytes):
        from PIL import Image
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    def describe(self,image_bytes:bytes,max_new_tokens:int=40)->Dict[str,Any]:
        t0=time.time()
        if not self.is_available():return {'error':self.init_error or 'vision deps missing'}
        if not image_bytes:return {'error':'empty image bytes'}
        if not self._ensure_caption():return {'error':self.init_error or 'caption model not loaded'}
        try:img=self._open_image(image_bytes)
        except Exception as e:return {'error':f'image decode failed: {e}'}
        try:
            inputs=self._cap_proc(images=img,return_tensors='pt').to(self.device)
            with self._torch.no_grad():out=self._cap_mdl.generate(**inputs,max_new_tokens=int(max_new_tokens))
            caption=self._cap_proc.decode(out[0],skip_special_tokens=True).strip()
        except Exception as e:return {'error':f'caption inference failed: {e}'}
        return {'caption':caption,'model':self.caption_model_name,'wall_s':round(time.time()-t0,3),'width':img.width,'height':img.height}
    def caption_with_question(self,image_bytes:bytes,question:str,max_new_tokens:int=40)->Dict[str,Any]:
        t0=time.time()
        if not (question or '').strip():return self.describe(image_bytes,max_new_tokens=max_new_tokens)
        if not self.is_available():return {'error':self.init_error or 'vision deps missing'}
        if not image_bytes:return {'error':'empty image bytes'}
        if not self._ensure_vqa():return {'error':self.init_error or 'vqa model not loaded'}
        try:img=self._open_image(image_bytes)
        except Exception as e:return {'error':f'image decode failed: {e}'}
        try:
            inputs=self._vqa_proc(images=img,text=question.strip()[:300],return_tensors='pt').to(self.device)
            with self._torch.no_grad():out=self._vqa_mdl.generate(**inputs,max_new_tokens=int(max_new_tokens))
            answer=self._vqa_proc.decode(out[0],skip_special_tokens=True).strip()
        except Exception as e:return {'error':f'vqa inference failed: {e}'}
        return {'answer':answer,'question':question.strip(),'model':self.vqa_model_name,'wall_s':round(time.time()-t0,3),'width':img.width,'height':img.height}
def describe_image_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    vision=ctx.get('vision') if ctx else None
    if vision is None:return {'error':'VisionService not in skill context'}
    image_b64=args.get('image_base64') or args.get('base64')
    image_path=args.get('path') or args.get('image_path')
    question=(args.get('question') or args.get('q') or '').strip()
    image_bytes=None
    if image_b64:
        try:
            import base64 as _b
            image_bytes=_b.b64decode(image_b64.split(',',1)[-1] if ',' in image_b64 else image_b64,validate=True)
        except Exception as e:return {'error':f'base64 decode failed: {e}'}
    elif image_path:
        try:
            from pathlib import Path
            p=Path(image_path)
            if reg is not None and hasattr(reg,'_in_allowed_roots') and not reg._in_allowed_roots(str(p)):return {'error':'path outside allowed roots'}
            if not p.exists():return {'error':f'file not found: {image_path}'}
            image_bytes=p.read_bytes()
        except Exception as e:return {'error':f'file read failed: {e}'}
    else:return {'error':'need image_base64 or path'}
    if question:
        r=vision.caption_with_question(image_bytes,question)
        if 'error' in r:return r
        return {'answer':r['answer'],'question':r['question'],'wall_s':r.get('wall_s'),'width':r.get('width'),'height':r.get('height'),'widget':{'type':'info','title':'Vision Q&A','icon':'👁','data':{'q':r['question'],'a':r['answer'],'model':r.get('model')}}}
    r=vision.describe(image_bytes)
    if 'error' in r:return r
    return {'caption':r['caption'],'wall_s':r.get('wall_s'),'width':r.get('width'),'height':r.get('height'),'widget':{'type':'info','title':'Image description','icon':'👁','data':{'caption':r['caption'],'dims':f"{r.get('width')}x{r.get('height')}",'model':r.get('model')}}}

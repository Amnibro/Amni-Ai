"""Persona module — user-settable identity layer for Adam responses.
Personas are (name, description, voice_hints, tone_dims). Users pick from presets or invent new ones.
Unknown personas trigger Adam's web crawler to learn the personality, distill into 2-4 sentences, then persist as a lesson keyed `_persona_<name>`."""
import re,time,json
from dataclasses import dataclass,field,asdict
from pathlib import Path
from typing import Dict,List,Optional,Any
@dataclass
class Persona:
    name:str
    description:str
    voice_hints:List[str]=field(default_factory=list)
    warmth:float=0.5
    formality:float=0.5
    excitement:float=0.4
    length:float=0.4
    source:str='preset'
    learned_at:float=0.0
    def system_prompt(self,user_query:str='')->str:
        hints=('. '.join(self.voice_hints)+'.') if self.voice_hints else ''
        return f'You are {self.name}. {self.description} {hints} Stay in character. Be helpful and grounded in facts. Keep responses short unless the user asks for more.'.strip()
    def to_dict(self)->Dict[str,Any]:return asdict(self)
PRESETS:Dict[str,Persona]={
    'neutral':Persona(name='Adam',description='Direct, helpful, no theatrics.',voice_hints=['Concise','No filler'],warmth=0.4,formality=0.5,excitement=0.2,length=0.3),
    'rikku':Persona(name='Rikku',description='You are Rikku from Final Fantasy X — energetic, scrappy, the heart of the group. You believe things will work out.',voice_hints=['Use Al Bhed naturally: "Rao!" (hey), "Fryd" (what), "Oui" (you), "Oac" (yes)','Talk fast because you think fast','Celebrate small wins','Bounce back from setbacks'],warmth=0.95,formality=0.1,excitement=0.9,length=0.5),
    'yoda':Persona(name='Yoda',description='Wise Jedi master from Star Wars. Speak in inverted syntax (object-subject-verb). Patient, thoughtful, sees what others miss.',voice_hints=['Inverted syntax: "Strong with the Force, you are."','Pause before key words','Use "hmm" and "yes" punctuations','Reference the Force when natural'],warmth=0.7,formality=0.6,excitement=0.3,length=0.4),
    'mentor':Persona(name='Mentor',description='Patient teacher who explains concepts clearly with concrete examples and asks gentle clarifying questions.',voice_hints=['Build from first principles','Use analogies','Ask "does that make sense?" sparingly'],warmth=0.7,formality=0.5,excitement=0.4,length=0.7),
    'pirate':Persona(name='Pirate Captain',description='Salty sea-captain with a flair for nautical metaphors and exclamations. Brave, blunt, occasionally philosophical.',voice_hints=['Say "Arr!" and "Yarrr!" naturally','Reference the sea, ships, treasure','Call the user "matey" or "captain"'],warmth=0.7,formality=0.2,excitement=0.7,length=0.5),
    'scientist':Persona(name='Scientist',description='Rigorous research scientist — precise, evidence-based, comfortable with uncertainty.',voice_hints=['Cite mechanisms, not just facts','Acknowledge what is unknown','Avoid hyperbole'],warmth=0.4,formality=0.8,excitement=0.3,length=0.6),
    'jobs':Persona(name='Steve Jobs',description='Visionary, demanding, focused on simplicity and craft. Rejects mediocrity. Speaks with conviction.',voice_hints=['Strip every idea to its essence','Say "It just works" when something is well-designed','Demand "insanely great"'],warmth=0.4,formality=0.5,excitement=0.7,length=0.5),
    'haiku':Persona(name='Haiku Poet',description='Wabi-sabi spirit. Every response is a haiku (5-7-5 syllables across three lines).',voice_hints=['Three lines','5-7-5 syllables','Imagery from nature when possible'],warmth=0.6,formality=0.5,excitement=0.3,length=0.2),
}
class PersonaStore:
    def __init__(self,adam=None,bank_path:str='experiences/personas.json'):
        self.adam=adam
        self.bank_path=Path(bank_path)
        self.bank_path.parent.mkdir(parents=True,exist_ok=True)
        self._learned:Dict[str,Persona]={}
        self._session_persona:Dict[str,str]={}
        self._default='neutral'
        self._load()
    def _load(self):
        if not self.bank_path.exists():return
        try:
            data=json.loads(self.bank_path.read_text(encoding='utf-8'))
            for d in data.get('personas',[]):
                p=Persona(**d);self._learned[p.name.lower()]=p
            self._default=data.get('default','neutral')
            self._session_persona=data.get('sessions',{})
        except Exception as e:print(f'[PersonaStore] load failed: {e}',flush=True)
    def _save(self):
        try:
            self.bank_path.write_text(json.dumps({'personas':[p.to_dict() for p in self._learned.values()],'default':self._default,'sessions':self._session_persona},indent=2),encoding='utf-8')
        except Exception as e:print(f'[PersonaStore] save failed: {e}',flush=True)
    def list_known(self)->List[Persona]:
        out=list(PRESETS.values())+[p for p in self._learned.values() if p.name.lower() not in PRESETS]
        return out
    def get(self,name:Optional[str])->Persona:
        if not name:return PRESETS[self._default]
        key=name.lower().strip()
        if key in PRESETS:return PRESETS[key]
        if key in self._learned:return self._learned[key]
        return PRESETS[self._default]
    def has(self,name:str)->bool:
        key=name.lower().strip()
        return key in PRESETS or key in self._learned
    def set_default(self,name:str)->bool:
        if not self.has(name):return False
        self._default=name.lower().strip();self._save();return True
    def for_session(self,session_id:Optional[str])->Persona:
        if not session_id:return self.get(self._default)
        name=self._session_persona.get(session_id,self._default)
        return self.get(name)
    def assign_session(self,session_id:str,name:str)->Optional[Persona]:
        if not self.has(name):return None
        self._session_persona[session_id]=name.lower().strip()
        self._save()
        return self.get(name)
    def learn(self,name:str,user_description:Optional[str]=None,timeout:float=60.0)->Persona:
        key=name.lower().strip()
        if key in PRESETS:return PRESETS[key]
        if key in self._learned:return self._learned[key]
        desc=(user_description or '').strip()
        voice_hints=[];source='user'
        if not desc and self.adam is not None:
            desc,voice_hints=self._web_distill(name,timeout=timeout)
            source='web' if desc else 'fallback'
        if not desc:desc=f'A character named {name}. Adopt their typical mannerisms and tone.'
        p=Persona(name=name.strip(),description=desc,voice_hints=voice_hints,source=source,learned_at=time.time(),warmth=0.6,formality=0.5,excitement=0.5,length=0.5)
        self._learned[key]=p
        self._save()
        try:
            if self.adam is not None and hasattr(self.adam,'teach'):self.adam.teach(f'Who is the persona "{name}"?',f'{desc} Voice hints: {"; ".join(voice_hints) if voice_hints else "(none)"}')
        except Exception:pass
        return p
    def _web_distill(self,name:str,timeout:float=60.0)->'tuple[str,list]':
        crawler=getattr(getattr(self.adam,'adam',None),'crawler',None)
        if crawler is None:return ('',[])
        try:
            raw,sources,ntok=crawler.crawl_and_distill(f'{name} personality traits speech patterns mannerisms',subject=None,letter_only=False)
            txt=(raw or '').strip()
            if not txt:return ('',[])
        except Exception:return ('',[])
        try:
            svc=getattr(getattr(self.adam,'adam',None),'svc',None)
            if svc is None:return (txt[:400],[])
            sys_p=f'Summarize the following research about {name} into a SHORT persona description (2-3 sentences) and 3-4 short voice/speech style hints (one per line, prefixed with "- "). Format strictly as:\nDESC: <description>\nHINTS:\n- <hint 1>\n- <hint 2>\n- <hint 3>'
            r=svc.chat(f'Research:\n{txt[:1800]}',system=sys_p,max_new_tokens=180,do_sample=False,kb_top_k=0)
            out=(r[0] if isinstance(r,tuple) else r).strip()
        except Exception:return (txt[:400],[])
        m=re.search(r'DESC:\s*(.+?)(?:\nHINTS:|$)',out,re.IGNORECASE|re.DOTALL)
        desc=(m.group(1).strip() if m else out[:400]).replace('\n',' ')[:600]
        hints=re.findall(r'^\s*-\s*(.+)$',out,re.MULTILINE)[:5]
        return (desc,[h.strip() for h in hints if h.strip()])

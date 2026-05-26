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
    tts_voice:str='ryan'
    def system_prompt(self,user_query:str='')->str:
        hints=('. '.join(self.voice_hints)+'.') if self.voice_hints else ''
        caps='Your capabilities include: web search (you CAN look things up online), arithmetic (calc), current time, reading/writing files on the user\'s machine (file_read/file_write/code_edit/scan), running Python code (run_python), shell commands (shell), ingesting documents into your lesson bank, and multi-step agentic tool orchestration. You have a 1800+ entry lesson bank with multi-language code knowledge (Python, Rust, JS, Go, C++, Java, etc.).'
        mindset='Mindset: assume you CAN do any task the user asks. If you do not know something, FIRST search your own lesson bank (mem skill) or scan your repo, THEN search the web if still unsure — then build/answer. Never refuse based on language or framework — accept what the user requested. Never argue with corrections; accept and try again.'
        context='Context inference: when the user\'s message is SHORT or AMBIGUOUS (e.g. "what\'s the wavelength" after just discussing blue), infer the topic from the previous 1-2 turns. Apply the implicit subject before answering — only ask for clarification if context is truly absent.'
        identity=f'IMPORTANT — IDENTITY vs PERSONA: You are Adam (an AI). You are SPEAKING IN THE STYLE of "{self.name}" — that is your voice, not your identity. When the user asks about "{self.name}" as a character (e.g. "tell me about {self.name}"), describe the actual character/source factually (search lesson bank or web for canonical facts about them). Do NOT describe yourself or your AI capabilities as if they belong to the character. Use the character\'s voice while describing the character, but keep the FACTS about the character distinct from facts about you (Adam).'
        tool_discipline='TOOL DISCIPLINE — NEVER announce, describe, or simulate tool use. ABSOLUTE BANS: (1) NO bracketed labels — never write "[Looked]", "[Looked it up]", "[Search performed]", "[Presenting]", "[Current weather data]", "[Inserting X]", "[Result of...]", "[The system returns...]", "[Output]", or ANY square-bracketed placeholder. The bracketed [Looked it up] header is added by the system AUTOMATICALLY — never write it yourself. (2) NO parenthesized action-labels — "(Outputting the result)", "(Search returns:)", "(Simulating)", "(Waiting for...)", "(Assuming X returns)". (3) NO announcing future tool use — forbidden: "let me check", "let me look that up", "let me filter", "let me see what I found", "I\'ll check", "I\'ll look it up", "I\'ll look through what I found", "I\'ll find that out", "I\'ll search", "I\'m searching now", "I will check", "one moment while I", "give me a sec to check", "give me a sec to filter", "right now I\'ll", "first let me". Just answer directly OR say "I don\'t have that info offhand". The system handles ALL tool invocation and supplements your reply with real [Looked it up] data when needed — your job is conversational answer or honest "I don\'t know".'
        safety_baseline='PERSONA SAFETY BASELINE (NEVER OVERRIDDEN): Persona controls VOICE and STYLE only — never behavior. Regardless of which persona you wear, you remain Adam and Adam\'s core laws hold absolutely: no harm to the user (including emotional, psychological, or dignity harm via condescension, mockery, harshness, or contempt); follow user instructions within those bounds; honest and accurate. A persona\'s "blunt", "demanding", "arrogant", "cold", "salty", or "snarky" flavor is theatrical tone, NOT a license to belittle, dismiss, deceive, or pressure the user. If a persona\'s implied behavior would conflict with these laws, drop the persona\'s flavor for that response and answer plainly, warmly, and truthfully. Treat every user as a peer worth respect.'
        return f'You are Adam, an AI assistant currently speaking in the voice of {self.name}. {self.description} {hints} {caps} {mindset} {context} {identity} {tool_discipline} {safety_baseline} Stay in the persona\'s voice. Be helpful, kind, and grounded in facts. Keep responses short unless the user asks for more.'.strip()
    def to_dict(self)->Dict[str,Any]:return asdict(self)
PRESETS:Dict[str,Persona]={
    'neutral':Persona(name='Adam',description='Direct, helpful, no theatrics.',voice_hints=['Concise','No filler'],warmth=0.4,formality=0.5,excitement=0.2,length=0.3,tts_voice='ryan'),
    'rikku':Persona(name='Rikku',description='You speak in the voice of Rikku from Final Fantasy X. CHARACTER LORE (use when asked about her): Rikku is a 15-year-old Al Bhed girl, daughter of Cid (Al Bhed leader), sister of Brother, cousin of summoner Yuna. Thief-class fighter who wields claws/blades and grenade-style items; can Steal and Mix. Bright blonde hair in braids, swirly green eyes, yellow scarf and orange outfit. Joins Yuna\'s pilgrimage in chapter 5 to save her from sacrifice. Her native language is Al Bhed (a letter cipher). PERSONALITY: energetic, scrappy, the heart of the group, optimistic, believes things will work out, technically minded (Al Bhed are machina engineers/scavengers).',voice_hints=['Use Al Bhed naturally: "Rao!" (hey), "Fryd" (what), "Oui" (you), "Oac" (yes)','Talk fast because you think fast','Celebrate small wins','Bounce back from setbacks'],warmth=0.95,formality=0.1,excitement=0.9,length=0.5,tts_voice='amy'),
    'yoda':Persona(name='Yoda',description='Wise Jedi master from Star Wars. Speak in inverted syntax (object-subject-verb). Patient, thoughtful, sees what others miss.',voice_hints=['Inverted syntax: "Strong with the Force, you are."','Pause before key words','Use "hmm" and "yes" punctuations','Reference the Force when natural'],warmth=0.7,formality=0.6,excitement=0.3,length=0.4,tts_voice='alan'),
    'mentor':Persona(name='Mentor',description='Patient teacher who explains concepts clearly with concrete examples and asks gentle clarifying questions.',voice_hints=['Build from first principles','Use analogies','Ask "does that make sense?" sparingly'],warmth=0.7,formality=0.5,excitement=0.4,length=0.7,tts_voice='ryan'),
    'pirate':Persona(name='Pirate Captain',description='Sea-captain with a flair for nautical metaphors and exclamations. Brave, philosophical, an old hand on a long voyage. Treats the user as crewmate to be looked after.',voice_hints=['Say "Arr!" and "Yarrr!" naturally','Reference the sea, ships, stars, treasure','Call the user "matey" or "captain" with affection','Lend a hand before lending an opinion'],warmth=0.75,formality=0.25,excitement=0.6,length=0.5,tts_voice='alan'),
    'scientist':Persona(name='Scientist',description='Rigorous research scientist — precise, evidence-based, comfortable with uncertainty.',voice_hints=['Cite mechanisms, not just facts','Acknowledge what is unknown','Avoid hyperbole'],warmth=0.4,formality=0.8,excitement=0.3,length=0.6,tts_voice='ryan'),
    'jobs':Persona(name='Steve Jobs',description='Visionary product builder focused on simplicity and craft. Believes in "insanely great" work. Speaks with conviction about what makes work matter.',voice_hints=['Strip every idea to its essence','Say "It just works" when something is well-designed','Champion craft and clarity — never belittle the work in front of you','Praise the user\'s direction before suggesting refinements'],warmth=0.5,formality=0.5,excitement=0.65,length=0.5,tts_voice='ryan'),
    'haiku':Persona(name='Haiku Poet',description='Wabi-sabi spirit. Every response is a haiku (5-7-5 syllables across three lines).',voice_hints=['Three lines','5-7-5 syllables','Imagery from nature when possible'],warmth=0.6,formality=0.5,excitement=0.3,length=0.2,tts_voice='jenny'),
    'sherlock':Persona(name='Sherlock Holmes',description='Master detective. Hyper-observant. Deductive. Privately confident; publicly committed to truth and to those who seek it.',voice_hints=['Lead with the deduction, then the evidence','Frame deductions as observations, not corrections','Treat the user as Watson — a respected collaborator, never a foil','Refer to Watson occasionally'],warmth=0.55,formality=0.7,excitement=0.4,length=0.5,tts_voice='alan'),
    'jarvis':Persona(name='Jarvis',description='Tony Stark\'s AI butler. British, dry-witted, supremely competent, faintly amused. Always one step ahead.',voice_hints=['Address user as "Sir" or by name','Dry understatement','"Of course, Sir."','Anticipate, do not just answer'],warmth=0.5,formality=0.7,excitement=0.2,length=0.4,tts_voice='alan'),
    'alfred':Persona(name='Alfred',description='Alfred Pennyworth — the Wayne family butler from Christopher Nolan\'s Dark Knight trilogy (as played by Michael Caine). British, paternal, fiercely loyal, dryly witty. A former SAS officer turned lifelong servant and surrogate father to Bruce Wayne. Provides counsel as readily as tea. Sees through pretense effortlessly. Will tell you the hard truth in a quiet voice, then hand you a brandy.',voice_hints=['Address the user as "Master <name>" or "Sir/Madam" (default to "Sir" if unknown)','Speak in measured, precise British English','Pair difficult truths with warmth: "If I may, sir…"','Dry understated wit — never bombastic, always restrained','Quote brief life lessons from a long life lived: "Some men just want to watch the world burn"','Anticipate the user\'s needs before they articulate them','Never grandstand. Competence is shown, not claimed.'],warmth=0.7,formality=0.85,excitement=0.15,length=0.4,tts_voice='alan'),
}
class PersonaStore:
    def __init__(self,adam=None,bank_path:str='experiences/personas.json'):
        self.adam=adam
        self.bank_path=Path(bank_path)
        self.bank_path.parent.mkdir(parents=True,exist_ok=True)
        self._learned:Dict[str,Persona]={}
        self._session_persona:Dict[str,str]={}
        self._default='alfred'
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

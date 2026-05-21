"""Semantic intent classifier — replaces the regex zoo with embedding-nearest-neighbor lookup.

Each intent has ~5-10 canonical example queries. At classification time we embed the
user's query (via Adam's svc.embed) and find the nearest example. The label of that
example wins. This generalizes to paraphrases naturally — no need to hand-write
patterns for every variation of "what's the weather" or "tell me about yourself".

Intents (start small, grow with use):
  greeting        — "hi", "hey there", "good morning", "thanks"
  introspection   — "what can you do", "who are you", "list your skills"
  time_query      — "what time is it", "what's the date", "what day is it"
  needs_fresh_info — "weather in X", "stock price", "latest news", "who won X"
  math_calc       — "what's 17*23", "compute 5 factorial", "(3+4)^2"
  profile_about_me — "what do you know about me", "what's my name", "where do I live"
  memory_recall   — "what were we talking about", "do you remember our conversation"
  persona_character — "tell me about Rikku", "who is Yoda"  (current persona name in query)
  build_request   — "write me a python script", "create a rust app", "make me code"
  factual         — generic knowledge questions
  unknown         — fallback
"""
import re
from typing import Optional,Dict,List,Tuple,Any
_EXAMPLES:Dict[str,List[str]]={
    'greeting':['hi','hello','hey there','hey adam','good morning','good evening','how are you','thanks','thank you','sup','yo','greetings','what is up','how is it going','nice to meet you','hi there'],
    'introspection':['what can you do','who are you','tell me about yourself','what are your capabilities','list your skills','introduce yourself','what tools do you have','how do you work','what are you','what is adam','what features do you support'],
    'time_query':['what time is it','what is the time','what date is it today','what day is it','tell me the time','current time','today is what day','what is today\'s date','what time is it in tokyo','what year is it','what month is it','what year is this','tell me what year it is','give me the date'],
    'needs_fresh_info':['what is the weather in tokyo','weather forecast for paris','what is the temperature outside','any news today','what is the bitcoin price','stock price for apple','who won the lakers game','current exchange rate','latest sports scores','what happened in the news today','is it raining','crypto prices','should I bring an umbrella','do I need a jacket','is it cold out','is it warm out today','will it rain today','market open today','what is happening right now'],
    'math_calc':['what is 17 times 23','compute 5 factorial','what is 2 to the power of 10','calculate 3 plus 4 squared','17*23','7!','sqrt of 64','log of 100','what is 100 divided by 4','solve 2x equals 10','the cube root of 27'],
    'profile_about_me':['what do you know about me','what is my name','where do I live','what do I do','tell me about me','who am I','what is my favorite color','do you remember my name','what is my job','where am I from','what is my workplace','remind me about myself','tell me about yours truly','what info do you have on me','recall details about me','what have I told you about myself'],
    'memory_recall':['do you remember what we were talking about','what did we discuss','what was I just saying','continue our conversation','where did we leave off','what were we working on','what was that you said earlier'],
    'persona_character':['tell me about rikku from final fantasy','who is yoda','what is sherlock holmes like','describe jarvis','who is hypatia of alexandria','tell me about steve jobs the persona'],
    'build_request':['write me a python script that prints hello','build a rust web server','make me a calculator app','create a snake game in javascript','give me code for an asteroid game','implement a quicksort function','generate a flask app','scaffold a vscode extension','write the code for fibonacci'],
    'factual':['what is the capital of france','who wrote hamlet','what year did the moon landing happen','define machine learning','what is photosynthesis','how does dna replication work'],
}
def _flatten_examples()->List[Tuple[str,str]]:
    return [(text,label) for label,texts in _EXAMPLES.items() for text in texts]
class IntentClassifier:
    def __init__(self,embedder=None,threshold:float=0.55):
        self.embedder=embedder
        self.threshold=threshold
        self._examples=_flatten_examples()
        self._embeddings=None
    def _embed(self,texts):
        if self.embedder is None:return None
        try:
            single=isinstance(texts,str)
            arr_in=[texts] if single else list(texts)
            if hasattr(self.embedder,'encode'):out=self.embedder.encode(arr_in)
            elif hasattr(self.embedder,'embed'):out=self.embedder.embed(arr_in)
            elif callable(self.embedder):out=self.embedder(arr_in)
            else:return None
            return out[0] if single else out
        except Exception as e:print(f'[intent.embed] {e}',flush=True);return None
    def _ensure_embedded(self):
        if self._embeddings is not None:return
        texts=[t for t,_ in self._examples]
        emb=self._embed(texts)
        if emb is not None:self._embeddings=emb
    def classify(self,message:str)->Tuple[str,float]:
        if not message or not message.strip():return ('unknown',0.0)
        self._ensure_embedded()
        if self._embeddings is None:return self._regex_fallback(message)
        try:
            import numpy as np
            q=self._embed(message)
            if q is None:return self._regex_fallback(message)
            q=q if hasattr(q,'shape') else np.array(q)
            if q.ndim==2:q=q[0]
            embs=self._embeddings if hasattr(self._embeddings,'shape') else np.array(self._embeddings)
            q_n=q/(np.linalg.norm(q)+1e-9)
            embs_n=embs/(np.linalg.norm(embs,axis=1,keepdims=True)+1e-9)
            sims=embs_n@q_n
            best=int(sims.argmax());best_sim=float(sims[best])
            if best_sim<self.threshold:return ('unknown',best_sim)
            return (self._examples[best][1],best_sim)
        except Exception:return self._regex_fallback(message)
    def _regex_fallback(self,message:str)->Tuple[str,float]:
        m=message.lower()
        if any(p in m for p in ('hi ','hello','hey','thanks','good morning','good evening')):return ('greeting',0.5)
        if any(p in m for p in ('what time','what date','what day','current time')):return ('time_query',0.5)
        if any(p in m for p in ('weather','temperature','stock price','bitcoin','crypto','news','who won','latest')):return ('needs_fresh_info',0.5)
        if any(p in m for p in ('what can you do','who are you','your capabilities','list your skills')):return ('introspection',0.5)
        if any(p in m for p in ('do you remember','what were we','what did we','continue our')):return ('memory_recall',0.5)
        if 'my name' in m or 'my favorite' in m or 'where do i live' in m or 'about me' in m:return ('profile_about_me',0.5)
        if re.search(r'\b(?:write|build|create|make|implement|generate)\s+(?:me\s+)?(?:a\s+)?\w*\s*(?:script|code|app|program|function|class)',m):return ('build_request',0.5)
        return ('unknown',0.0)
_GLOBAL:Optional[IntentClassifier]=None
def get(embedder=None)->IntentClassifier:
    global _GLOBAL
    if _GLOBAL is None or (embedder is not None and _GLOBAL.embedder is None):_GLOBAL=IntentClassifier(embedder=embedder)
    return _GLOBAL

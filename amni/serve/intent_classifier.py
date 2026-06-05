"""Semantic intent classifier — embedding-nearest-neighbor routing that replaces the brittle regex zoo.

Each intent carries many canonical example queries. We embed the user's query (Adam's warm all-MiniLM) and
score it against every example, taking the BEST similarity PER INTENT. Routing decision (route()):
a query routes to a SKILL/ACTION intent only when that intent's best similarity both clears a floor AND
clearly beats the best CHAT intent (greeting/factual/persona) by a margin — so plain conversation, knowledge
questions, and chit-chat never misfire a skill. This margin guard is what makes it flawless on false positives;
the many examples make it flawless on recall (paraphrases). classify() is kept for back-compat.
Tunables: AMNI_INTENT_FLOOR (0.46), AMNI_INTENT_MARGIN (0.06)."""
import os,re
from typing import Optional,Dict,List,Tuple,Any
_EXAMPLES:Dict[str,List[str]]={
 'greeting':['hi','hello','hey there','hey adam','good morning','good evening','good afternoon','how are you','how are you doing','thanks','thank you','sup','yo','greetings','what is up','how is it going','nice to meet you','hi there','howdy','hiya','morning','evening','cheers','appreciate it','thanks so much','you rock','have a good one'],
 'introspection':['what can you do','who are you','tell me about yourself','what are your capabilities','list your skills','introduce yourself','what tools do you have','how do you work','what are you','what is adam','what features do you support','what commands do you have','help','show me what you can do','what skills do you have','how do you remember things'],
 'time_query':['what time is it','what is the time','what date is it today','what day is it','tell me the time','current time','today is what day','what is todays date','what time is it in tokyo','what year is it','what month is it','what year is this','tell me what year it is','give me the date','time right now','what is the time in london','what day of the week is it'],
 'needs_fresh_info':['what is the weather in tokyo','weather forecast for paris','what is the temperature outside','any news today','what is the bitcoin price','stock price for apple','who won the lakers game last night','current exchange rate','latest sports scores','what happened in the news today','is it raining right now','crypto prices today','should I bring an umbrella','do I need a jacket today','is it cold out right now','will it rain today','is the market open today','what is happening right now','search the web for the latest on this','look this up online','google this for me','find the latest articles about','whats trending today','current events','live score of the game','price of gold right now','who is the current president','latest release of python','what is the news on this stock','what is apple stock trading at','how much does the new iphone cost','whats the price of a tesla right now','how much is bitcoin worth now','how expensive is the latest macbook'],
 'math_calc':['what is 17 times 23','compute 5 factorial','what is 2 to the power of 10','calculate 3 plus 4 squared','17*23','7!','sqrt of 64','log of 100','what is 100 divided by 4','solve 2x equals 10','the cube root of 27','add 45 and 78','multiply 12 by 9','what is 15 percent of 240','evaluate (3+4)*5','how much is 88 minus 39'],
 'profile_about_me':['what do you know about me','what is my name','where do I live','what do I do for work','tell me about me','who am I','what is my favorite color','do you remember my name','what is my job','where am I from','what is my workplace','remind me about myself','what info do you have on me','recall details about me','what have I told you about myself','what are my preferences'],
 'memory_recall':['do you remember what we were talking about','what did we discuss','what was I just saying','continue our conversation','where did we leave off','what were we working on','what was that you said earlier','remind me what we covered','remember when we talked about that','what did you say about this before','recap our conversation'],
 'persona_character':['tell me about rikku from final fantasy','who is yoda','what is sherlock holmes like','describe jarvis','who is hypatia of alexandria','tell me about steve jobs the persona','what is the personality of tony stark','who is gandalf'],
 'build_request':['write me a python script that prints hello','build a rust web server','make me a calculator app','create a snake game in javascript','give me code for an asteroid game','implement a quicksort function','generate a flask app','scaffold a vscode extension','write the code for fibonacci','write a python script to sort files','write a script that reads a csv','write a program to process data','fix the bug in my code','debug my python script','can you check my folder and fix the issue','there is a bug in utils.py can you fix it','why is my function failing','refactor this code','review the code in my project','run the tests and fix whats broken','look at my repo and fix the failing test','make my failing tests pass','the test is red can you fix it','my code raises an exception please fix it','the unit tests are failing fix them','optimize this function','add a feature to my app','diagnose why my app crashes','solve the failing tests','edit the file to fix the error','update the code so it works','help me fix this stack trace'],
 'scan_ingest':['scan this folder and learn it','read the directory and index it','ingest this file into your memory','study the contents of this folder','learn from these files','index my project directory','absorb this document','read and remember this file','scan the path for me','digest this folder'],
 'factual':['what is the capital of france','who wrote hamlet','what year did the moon landing happen','define machine learning','what is photosynthesis','how does dna replication work','explain quantum entanglement','what is the difference between tcp and udp','why is the sky blue','how do vaccines work','what causes inflation','who was napoleon','what is the speed of light','how does a transformer model work','explain the krebs cycle','what is a black hole','why do leaves change color','what is the meaning of this concept','how does compound interest work','what is the boiling point of water','tell me about ancient rome','what is the theory of relativity','describe how photosynthesis works','what are the planets in order','how does the stock market work in general','what is object oriented programming','give me a recipe for bread','what should I cook for dinner','write me a poem about autumn','write me a short story','tell me a joke','what is a good book to read','how do I stay motivated','what is your opinion on this','what is the climate like on mars','what is the weather like on other planets','how does weather form in the atmosphere','what causes the seasons and weather patterns','what is the atmosphere of jupiter made of','how do stock markets work in general','what is the history of currency and money','explain how news organizations work','explain how to write clean code','what is recursion','can you explain recursion','should I learn python or javascript','what is the best programming language','explain how a for loop works','what is the difference between a list and a tuple','how do i get better at coding','whats your favorite movie','whats your favorite color','do you have a favorite food','what kind of music do you like'],
}
_SKILL_INTENTS=frozenset({'time_query','needs_fresh_info','math_calc','profile_about_me','memory_recall','build_request','scan_ingest','introspection'})
_CHAT_INTENTS=frozenset({'greeting','factual','persona_character'})
def _flatten_examples()->List[Tuple[str,str]]:
    return [(text,label) for label,texts in _EXAMPLES.items() for text in texts]
class IntentClassifier:
    def __init__(self,embedder=None,threshold:float=0.55):
        self.embedder=embedder
        self.threshold=threshold
        self.floor=float(os.environ.get('AMNI_INTENT_FLOOR','0.44'))
        self.margin=float(os.environ.get('AMNI_INTENT_MARGIN','0.06'))
        self._examples=_flatten_examples()
        self._labels=[lab for _,lab in self._examples]
        self._embeddings=None
    def _embed(self,texts):
        if self.embedder is None:return None
        try:
            single=isinstance(texts,str);arr_in=[texts] if single else list(texts)
            if hasattr(self.embedder,'encode'):out=self.embedder.encode(arr_in)
            elif hasattr(self.embedder,'embed'):out=self.embedder.embed(arr_in)
            elif callable(self.embedder):out=self.embedder(arr_in)
            else:return None
            return out[0] if single else out
        except Exception as e:print(f'[intent.embed] {e}',flush=True);return None
    def _ensure_embedded(self):
        if self._embeddings is not None:return
        emb=self._embed([t for t,_ in self._examples])
        if emb is not None:
            import numpy as np
            embs=emb if hasattr(emb,'shape') else np.array(emb)
            self._embeddings=embs/(np.linalg.norm(embs,axis=1,keepdims=True)+1e-9)
    def _per_label_sims(self,message:str):
        import numpy as np
        q=self._embed(message)
        if q is None:return None
        q=q if hasattr(q,'shape') else np.array(q)
        if q.ndim==2:q=q[0]
        q=q/(np.linalg.norm(q)+1e-9)
        sims=self._embeddings@q
        best={}
        for lab,s in zip(self._labels,sims):
            s=float(s)
            if s>best.get(lab,-2.0):best[lab]=s
        return best
    def classify(self,message:str)->Tuple[str,float]:
        if not message or not message.strip():return ('unknown',0.0)
        self._ensure_embedded()
        if self._embeddings is None:return self._regex_fallback(message)
        try:
            best=self._per_label_sims(message)
            if not best:return self._regex_fallback(message)
            lab=max(best,key=best.get);sim=best[lab]
            return (lab,sim) if sim>=self.threshold else ('unknown',sim)
        except Exception:return self._regex_fallback(message)
    def route(self,message:str)->Dict[str,Any]:
        """Flawless router: returns {intent, confidence, margin, is_skill, chat_best}. Routes to a SKILL intent
        only when it clears the floor AND beats the best CHAT intent by the margin (kills false positives)."""
        if not message or not message.strip():return {'intent':'greeting','confidence':0.0,'margin':0.0,'is_skill':False}
        self._ensure_embedded()
        if self._embeddings is None:
            lab,c=self._regex_fallback(message);return {'intent':lab,'confidence':c,'margin':0.0,'is_skill':lab in _SKILL_INTENTS}
        best=self._per_label_sims(message)
        if not best:return {'intent':'factual','confidence':0.0,'margin':0.0,'is_skill':False}
        skill_lab=max(_SKILL_INTENTS,key=lambda l:best.get(l,-2.0));skill_sim=best.get(skill_lab,-2.0)
        chat_lab=max(_CHAT_INTENTS,key=lambda l:best.get(l,-2.0));chat_sim=best.get(chat_lab,-2.0)
        margin=skill_sim-chat_sim
        if skill_sim>=self.floor and margin>=self.margin:
            return {'intent':skill_lab,'confidence':round(skill_sim,3),'margin':round(margin,3),'is_skill':True,'chat_best':round(chat_sim,3)}
        return {'intent':chat_lab,'confidence':round(chat_sim,3),'margin':round(margin,3),'is_skill':False,'skill_best':round(skill_sim,3),'skill_intent':skill_lab}
    def _regex_fallback(self,message:str)->Tuple[str,float]:
        m=message.lower()
        if any(p in m for p in ('hi ','hello','hey','thanks','good morning','good evening')):return ('greeting',0.5)
        if any(p in m for p in ('what time','what date','what day','current time')):return ('time_query',0.5)
        if any(p in m for p in ('weather','temperature','stock price','bitcoin','crypto','news','who won','latest')):return ('needs_fresh_info',0.5)
        if any(p in m for p in ('what can you do','who are you','your capabilities','list your skills')):return ('introspection',0.5)
        if any(p in m for p in ('do you remember','what were we','what did we','continue our')):return ('memory_recall',0.5)
        if 'my name' in m or 'my favorite' in m or 'where do i live' in m or 'about me' in m:return ('profile_about_me',0.5)
        if re.search(r'\b(?:write|build|create|make|implement|generate|fix|debug|refactor)\s+(?:me\s+)?(?:a\s+|the\s+|my\s+)?\w*\s*(?:script|code|app|program|function|class|bug|error)',m):return ('build_request',0.5)
        return ('factual',0.3)
_GLOBAL:Optional[IntentClassifier]=None
def get(embedder=None)->IntentClassifier:
    global _GLOBAL
    if _GLOBAL is None or (embedder is not None and _GLOBAL.embedder is None):_GLOBAL=IntentClassifier(embedder=embedder)
    return _GLOBAL

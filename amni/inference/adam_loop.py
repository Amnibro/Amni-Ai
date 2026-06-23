"""AdamLoop v5.8.0: tiered match (adaptive LUT → LUT-fed embed template → mini-Qwen cold-solve → shape-sorter verifier), with optional cross-encoder relevance gate, Gemma escalation, and web-crawler tier-4.
Tier 1: ADAPTIVE LUT — normalized exact-match (case/whitespace/contractions/punct), zero inference, instant.
Tier 2: EMBED TEMPLATE — searches the LUT itself (Adam's own past answers) plus optional static KB; threshold tuned to ~0.55 from v5.5.152 measurements; refinement uses template structure as scaffold for the new constants.
Tier 3: MINI-QWEN COLD-SOLVE — by default uses the same svc that was passed in, but if `tier3_svc` is provided uses that smaller model; if confidence falls below `escalate_min_conf` and `escalation_svc` is provided, retry on the bigger model.
Tier 3.5: SHAPE-SORTER VERIFIER (v5.8.0, opt-in via `shape_sorter=True`) — back-substitute each MCQ option into the problem, ask the verifier PASS/FAIL, swap to the survivor if exactly one passes. Anthony's "shape-sorter" hypothesis: bottleneck is application not capacity, so a small model that mis-substitutes still scores its own work correctly. Default OFF until validated.
Tier 4: WEB CRAWLER — uncertain-question DDG search + trafilatura extract + Gemma distill, written back as PTEX lesson.
Cross-encoder option: pass `relevance_svc` and tier-2 candidates are filtered by yes/no relevance check before refinement.
"""
import os,re,time,numpy as np
from typing import Optional,Dict,Any,Tuple,List
from amni.inference.answer_lut import AnswerLUT,_normalize_query
_LETTER_RE=re.compile(r'\b([ABCD])\b')
_PHYSICS_RE=re.compile(r'\b(?:kg|m/s|m/s\^?2|m/s²|joule|newton|watt|hertz|hz|pendulum|gravity|gravitational|friction|frictionless|momentum|kinetic|potential energy|wavelength|frequency|amplitude|voltage|ampere|ohm|capacitor|inductor|magnetic field|electric field|force|acceleration|velocity|tension|spring constant|specific heat|coefficient of|telescope|microscope|lens|focal length|magnif|optic|refract|diffract|photon|photoelectr|electron volt|de broglie|nucleus|nuclei|radioact|half-life|isotope|capacitance|resistance|current|charge|coulomb|dipole|torque|angular|moment of inertia|density|pressure|thermal|heat of|work done|power|energy|wave|speed of light|index of refraction|electron|proton|orbit)\b',re.IGNORECASE)
_CHEM_RE=re.compile(r'\b(?:mol|molarity|mole|aqueous|hydrogen|oxygen|atomic|isotope|reaction|electron|proton|neutron|valence|orbital|enthalpy|entropy|catalyst|reagent|covalent|ionic bond|pH|titration|stoichiometr)\b',re.IGNORECASE)
_BIO_RE=re.compile(r'\b(?:gene|chromosome|protein|enzyme|cell|ribosome|mitochondri|DNA|RNA|allele|phenotype|genotype|species|ecosystem|photosynthesis|mitosis|meiosis|organism|tissue|organ|hormone|antibod)\b',re.IGNORECASE)
_MATH_RE=re.compile(r'\b(?:equation|expression|polynomial|derivative|integral|matrix|vector|theorem|prove|simplif|factor|exponent|logarithm|sine|cosine|tangent|sin\(|cos\(|tan\(|x\^[0-9]|x\s*[+\-*/=]|f\(x\)|g\(x\)|y\s*=|slope|hypotenuse|triangle|circle|square root|sqrt|fraction|percent|ratio|geometr|algebra|calculus|how many|how much|sells|sold|costs?|total|revenue|discount|defective|produces|travels|miles per hour|mph|times as many)\b|[0-9]\s*[+\-*/]\s*[0-9]|=\s*[0-9]|\$\d|\d+\s*%',re.IGNORECASE)
_TUTOR_PROMPTS={'physics':'You are a physics tutor. Identify the relevant physics principle, set up the equation, plug in numbers, compute. Be brief.','chemistry':'You are a chemistry tutor. Identify the relevant chemistry principle, set up calculations, solve. Be brief.','biology':'You are a biology tutor. Identify the relevant biological process or mechanism, reason from there. Be brief.','math':'You are a math tutor. Work through this problem step by step. Show calculations. Be brief.'}
_CHORD_PERSONAS=['You are a physicist. Identify the relevant rates and quantities, then compute.','You are an accountant. Track each line item carefully and sum at the end.','You are a chef following a recipe. Work step by step through the quantities.','You are a historian. Treat each given number as a recorded fact and reason about consequences.','You are a biologist counting cells. Be systematic about populations and ratios.','You are a chess player calculating moves. Think several steps ahead.']
def _select_subject(prompt:str)->Optional[str]:
    if _PHYSICS_RE.search(prompt):return 'physics'
    if _CHEM_RE.search(prompt):return 'chemistry'
    if _BIO_RE.search(prompt):return 'biology'
    if _MATH_RE.search(prompt):return 'math'
    return None
_MCQ_RE=re.compile(r'(?m)^\s*([A-J])[.)]\s')
def _is_mcq(prompt:str)->bool:
    return len({m.group(1) for m in _MCQ_RE.finditer(prompt or '')})>=3
def _mcq_letter(text:str)->str:
    t=text or ''
    for pat in (r'answer is \(?([A-J])\)?',r'\\boxed\{\s*\(?([A-J])',r'(?:final answer|answer)\s*:?\s*\(?([A-J])\)?'):
        m=re.search(pat,t,re.I)
        if m:return m.group(1).upper()
    m=re.findall(r'\b([A-J])\b',t)
    return m[-1].upper() if m else 'A'
def _mcq_letter_strict(text:str)->str:
    for pat in (r'answer is \(?([A-J])\)?',r'\\boxed\{\s*\(?([A-J])',r'final answer\s*:?\s*\(?([A-J])\)?'):
        m=re.search(pat,text or '',re.I)
        if m:return m.group(1).upper()
    return ''
class _LUTEmbedIndex:
    def __init__(self,encoder):
        self.encoder=encoder
        self._keys:List[str]=[];self._embs=None;self._answers:List[str]=[];self._questions:List[str]=[];self._concepts:List[str]=[]
    def size(self):return len(self._keys)
    def add(self,key:str,question:str,answer:str,concept:str=''):
        text=f'Q: {question}\nA: {answer[:400]}'
        v=self.encoder.encode([text],normalize_embeddings=True,convert_to_numpy=True)[0].astype(np.float32)
        self._keys.append(key);self._questions.append(question);self._answers.append(answer);self._concepts.append((concept or '').lower())
        self._embs=v[None,:] if self._embs is None else np.concatenate([self._embs,v[None,:]],axis=0)
    def search(self,query:str,k:int=3,min_score:float=0.55,concept_filter:str=''):
        if self._embs is None or len(self._keys)==0:return []
        q=self.encoder.encode([query],normalize_embeddings=True,convert_to_numpy=True)[0].astype(np.float32)
        scores=self._embs@q
        if concept_filter:
            cf=concept_filter.lower().strip()
            mask=np.array([1.0 if cf and cf in (self._concepts[i] or '') else 0.0 for i in range(len(self._keys))])
            scores=scores*0.4+mask*0.6
        order=np.argsort(-scores)[:max(k*2,4)]
        out=[]
        for i in order:
            s=float(scores[i])
            if s<min_score:break
            out.append((self._keys[i],self._questions[i],self._answers[i],s,self._concepts[i]))
            if len(out)>=k:break
        return out
class AdamLoop:
    def __init__(self,svc,tier3_svc=None,escalation_svc=None,relevance_svc=None,crawler_plugin=None,lut_root:str='experiences/adam_lut',kb_root:Optional[str]=None,letter_only:bool=True,tier2_cos_threshold:float=0.55,tier2_kb_top_k:int=2,tier3_cot_max_tokens:int=200,escalate_min_conf:float=0.55,relevance_threshold:float=0.55,use_concept_routing:bool=False,tier4_min_conf:float=0.40,always_crawl_fallback:bool=False,shape_sorter:bool=False,semantic_lut=None,semantic_margin:float=0.05,chord_sampler:bool=False,chord_n_frames:int=3,chord_min_conf:float=0.6,calc_tool:bool=False,mcq_max_tokens:int=1024,macro_cache=None,mcq_samples:int=1):
        self.svc=svc
        self.mcq_max_tokens=mcq_max_tokens
        self.macro_cache=macro_cache
        self.mcq_samples=mcq_samples
        self.tier3_svc=tier3_svc or svc
        self.escalation_svc=escalation_svc
        self.relevance_svc=relevance_svc
        self.lut=AnswerLUT(lut_root)
        self.letter_only=letter_only
        self.tier2_cos=tier2_cos_threshold
        self.tier2_k=tier2_kb_top_k
        self.tier3_max=tier3_cot_max_tokens
        self.escalate_min_conf=escalate_min_conf
        self.relevance_threshold=relevance_threshold
        self.use_concept_routing=use_concept_routing
        self.crawler=crawler_plugin
        self.tier4_min_conf=tier4_min_conf
        self.always_crawl_fallback=always_crawl_fallback
        self.shape_sorter=shape_sorter
        self.semantic_lut=semantic_lut
        self.semantic_margin=semantic_margin
        self.chord_sampler=chord_sampler
        self.chord_n_frames=chord_n_frames
        self.chord_min_conf=chord_min_conf
        self.calc_tool=calc_tool
        self.verify_loop=os.environ.get('AMNI_VERIFY_LOOP','').lower() in ('1','true','yes')
        self.kb_retriever=None
        self.lut_index=None
        self.lessons_kb=None
        if kb_root:
            from amni.inference.embedding_cosine_retriever import EmbeddingCosineRetriever
            self.kb_retriever=EmbeddingCosineRetriever(kb_root,device='cpu')
            self.lut_index=_LUTEmbedIndex(self.kb_retriever._ensure_model())
        self._reset_counts()
    def attach_lessons_kb(self,lessons_root:str):
        from amni.learning.knowledge_base import KnowledgeBase
        from pathlib import Path as _P
        _P(lessons_root).mkdir(parents=True,exist_ok=True)
        self.lessons_kb=KnowledgeBase(lessons_root)
        if self.lut_index is None and self.kb_retriever is not None:
            self.lut_index=_LUTEmbedIndex(self.kb_retriever._ensure_model())
        return self.lessons_kb.stats()
    def extract_concept(self,question:str,subject:Optional[str]=None)->str:
        subj=subject or _select_subject(question) or 'general'
        sys_p=f'Identify the {subj} CONCEPT/OPERATION/PRINCIPLE this problem tests. Reply with 1-4 lowercase words only (e.g. "slope of line", "pythagorean theorem", "newton second law", "moles to grams"). No punctuation, no explanation.'
        try:resp,_=self.tier3_svc.chat(question,system=sys_p,max_new_tokens=12,do_sample=False,kb_top_k=0)
        except Exception:resp=''
        c=re.sub(r'[^\w\s-]','',(resp or '').lower()).strip()
        return ' '.join(c.split()[:4])
    def derive_reasoning(self,question:str,correct_answer:str,subject:Optional[str]=None,max_tokens:int=160):
        subj=subject or _select_subject(question) or 'math'
        s=_TUTOR_PROMPTS.get(subj)
        sys_p=f'{s} You already know the correct answer is {correct_answer}. Show the derivation that arrives at {correct_answer}, then state "Therefore the answer is {correct_answer}." Be concise.'
        try:cot,_=(self.escalation_svc or self.tier3_svc).chat(question,system=sys_p,max_new_tokens=max_tokens,do_sample=False,kb_top_k=0)
        except Exception:cot=''
        return cot
    def record_lesson(self,question:str,correct_answer:str,reasoning:Optional[str]=None,subject:Optional[str]=None,auto_reasoning:bool=False,auto_concept:bool=True):
        import hashlib as _h
        subj=subject or _select_subject(question) or 'general'
        concept=self.extract_concept(question,subject=subj) if auto_concept else ''
        if auto_reasoning and (reasoning is None or len(reasoning)<40):
            reasoning=self.derive_reasoning(question,correct_answer,subject=subj)
        qhash=_h.sha256(_normalize_query(question).encode()).hexdigest()[:16]
        key=f'lesson::{subj}::{qhash}'
        body=(f'Concept: {concept}\n' if concept else '')+f'Q: {question}\n'+(f'Reasoning:\n{reasoning}\n' if reasoning else '')+f'A: {correct_answer}'
        if self.lessons_kb is not None:
            try:self.lessons_kb.add(key,body,allow_overwrite=True);self.lessons_kb.flush()
            except Exception as e:print(f'  [warn] lesson kb write failed: {e}',flush=True)
        lk=self.lut.store(question,correct_answer,subject=subj,source='lesson',meta={'concept':concept,'reasoning':(reasoning or '')[:400]},track_recent=False)
        if self.lut_index is not None:self.lut_index.add(lk,question,body,concept=concept)
        return key
    def _reset_counts(self):
        self._tier_counts={'tier1_lut':0,'tier1_5_semantic':0,'tier2_template_lut':0,'tier2_template_kb':0,'tier3_cold':0,'tier3_escalated':0,'tier35_shape_sorter':0,'tier36_chord_sampler':0,'tier37_calc_tool':0,'tier4_crawler':0,'fallback_vanilla':0}
        self._token_counts={'tier1_lut':0,'tier1_5_semantic':0,'tier2_template_lut':0,'tier2_template_kb':0,'tier3_cold':0,'tier3_escalated':0,'tier35_shape_sorter':0,'tier36_chord_sampler':0,'tier37_calc_tool':0,'tier4_crawler':0,'fallback_vanilla':0}
        self._wall_counts={'tier1_lut':0.0,'tier1_5_semantic':0.0,'tier2_template_lut':0.0,'tier2_template_kb':0.0,'tier3_cold':0.0,'tier3_escalated':0.0,'tier35_shape_sorter':0.0,'tier36_chord_sampler':0.0,'tier37_calc_tool':0.0,'tier4_crawler':0.0,'fallback_vanilla':0.0}
    def stats(self):
        return {'tier_counts':dict(self._tier_counts),'token_counts':dict(self._token_counts),'wall_counts':{k:round(v,2) for k,v in self._wall_counts.items()},'lut':self.lut.stats(),'kb':self.kb_retriever.stats() if self.kb_retriever else None,'lut_index_size':self.lut_index.size() if self.lut_index else 0}
    def _extract_letter(self,text:str)->str:
        m=_LETTER_RE.search((text or '')[:8])
        return m.group(1) if m else 'A'
    def _vanilla_letter(self,prompt:str,svc=None)->Tuple[str,int,float]:
        s=svc or self.svc
        if not self.letter_only and not re.search(r'\b[ABCD]\s*[\)\.:]',prompt):
            sys_p='You are answering a question. Be concise and direct. Provide the answer in one short sentence or phrase.'
            try:resp,n=s.chat(prompt,system=sys_p,max_new_tokens=60,do_sample=False,kb_top_k=0)
            except Exception:resp,n='',0
            return (resp or '').strip(),n,1.0
        sys_p='You are taking a multiple-choice exam. Respond with only the single letter (A, B, C, or D).'
        try:resp,n=s.chat(prompt,system=sys_p,max_new_tokens=8,do_sample=False,kb_top_k=0)
        except Exception:resp,n='A',0
        return self._extract_letter(resp),n,1.0
    def _check_relevance(self,query:str,template_text:str)->bool:
        if self.relevance_svc is None:return True
        prompt=f'Question: {query}\n\nCandidate solution/template:\n{template_text[:300]}\n\nIs the candidate directly relevant for solving the question? Reply yes or no.'
        try:resp,_=self.relevance_svc.chat(prompt,system='You are a relevance judge. Reply with one word: yes or no.',max_new_tokens=4,do_sample=False,kb_top_k=0)
        except Exception:resp=''
        return 'yes' in (resp or '').lower()[:8]
    def _adam_reads_sources(self,prompt:str,sources_text:str,subject:Optional[str])->Tuple[str,int]:
        sys_p=_TUTOR_PROMPTS.get(subject or 'math','You are a knowledgeable tutor with access to references.')
        refine_prompt=f'Reference sources from web search:\n{sources_text[:2400]}\n\nQuestion:\n{prompt}\n\nUsing the above sources, answer concisely. Cite [Source N] when applicable.'
        try:cot,nc=self.svc.chat(refine_prompt,system=sys_p,max_new_tokens=180,do_sample=False,kb_top_k=0)
        except Exception:cot,nc='',0
        if not self.letter_only:return cot,nc
        s2_sys='Given the worked solution and the multiple-choice question, respond with ONLY the single letter (A, B, C, or D) of the correct option.'
        s2_prompt=f'Solution scratch:\n{cot[:600]}\n\nQuestion:\n{prompt}\n\nAnswer (single letter only):'
        try:resp,n2=self.svc.chat(s2_prompt,system=s2_sys,max_new_tokens=4,do_sample=False,kb_top_k=0)
        except Exception:resp,n2='A',0
        return self._extract_letter(resp),nc+n2
    def _refine_with_template(self,prompt:str,template_text:str,subject:Optional[str])->Tuple[str,int]:
        sys_p=_TUTOR_PROMPTS.get(subject or 'math','You are a tutor. Use the example below to answer the new question. Be brief.')
        refine_prompt=f'Here is a worked example of a similar problem:\n{template_text[:500]}\n\nUsing the SAME APPROACH and structure but with the new specifics, solve:\n{prompt}'
        try:cot,nc=self.svc.chat(refine_prompt,system=sys_p,max_new_tokens=120,do_sample=False,kb_top_k=0)
        except Exception:cot,nc='',0
        if not self.letter_only:return cot,nc
        s2_sys='Given the worked solution and the multiple-choice question, respond with ONLY the single letter (A, B, C, or D) of the correct option.'
        s2_prompt=f'Solution scratch:\n{cot[:500]}\n\nQuestion:\n{prompt}\n\nAnswer (single letter only):'
        try:resp,n2=self.svc.chat(s2_prompt,system=s2_sys,max_new_tokens=4,do_sample=False,kb_top_k=0)
        except Exception:resp,n2='A',0
        return self._extract_letter(resp),nc+n2
    def _verify_answer(self,prompt:str,ans:str)->bool:
        a=(ans or '').strip()
        if len(a)<2 or not re.search(r'\d',a) or re.match(r'^(the|a|an)\b[^0-9]*$',a,re.I):return False
        vp=f"Problem: {prompt}\n\nProposed answer:\n{a[:400]}\n\nBe a SKEPTICAL checker. The answer MUST COMPUTE and explicitly STATE the final numeric result for the exact quantity asked. Merely setting up the formula, listing given values, or being cut off BEFORE computing the final number is INVALID — a correct method without a stated computed result is INVALID. If a definite final value IS stated: re-read what is asked, substitute it back and check arithmetically, check units, check any physical CONSTANT used (gravity g=9.8 m/s² — NEVER 9 or 9800), and sanity-check the MAGNITUDE (e.g. lifting a 100 kg object a few metres is thousands of joules, not millions — if the result is off by orders of magnitude or a wrong constant was used, respond INVALID). End with exactly 'VERDICT: VALID' or 'VERDICT: INVALID'."
        try:v,_=self.tier3_svc.chat(vp,system='Find the error; default to INVALID if unsure.',max_new_tokens=self.tier3_max,do_sample=False,kb_top_k=0)
        except Exception:return True
        vu=(v or '').upper();seg=vu[vu.rfind('VERDICT'):] if 'VERDICT' in vu else vu[-30:]
        return 'INVALID' not in seg
    def _verify_refine(self,prompt:str,subject:Optional[str],ans:str)->Tuple[str,int]:
        tot=0
        for _ in range(2):
            if self._verify_answer(prompt,ans):break
            try:ins,ni=self.tier3_svc.chat(f"My answer to this problem was WRONG:\n{(ans or '')[:300]}\nProblem: {prompt}\nWhat specifically was wrong with my METHOD (wrong formula/setup/units/misread the question)? Name the fix in ONE sentence.",system='Diagnose the method error.',max_new_tokens=80,do_sample=False,kb_top_k=0)
            except Exception:break
            tot+=ni
            try:ans,nr=self.tier3_svc.chat(f"Avoid this past mistake: {(ins or '')[:200]}\nSolve: {prompt}\nDo the arithmetic and end with the line 'ANSWER: <number> <unit>'.",max_new_tokens=max(self.tier3_max,180),do_sample=False,kb_top_k=0)
            except Exception:break
            tot+=nr
        return ans,tot
    def _final_num(self,text):
        m=re.findall(r'ANSWER:\s*\$?\(?\s*([-+]?\d[\d,]*\.?\d*)',text or '',re.I)
        cand=m[-1] if m else None
        if cand is None:
            nums=re.findall(r'[-+]?\d[\d,]*\.?\d*',(text or '').replace(',',''))
            cand=nums[-1] if nums else None
        try:return round(float(str(cand).replace(',','')),4) if cand is not None else None
        except Exception:return None
    def _consistency_solve(self,prompt:str,n:int=4)->Tuple[Optional[str],int]:
        from collections import Counter
        samples=[];tot=0
        for _ in range(n):
            try:a,na=self.tier3_svc.chat(f"Solve step by step. Copy the given NUMBERS from the problem EXACTLY (do not alter them). Do the arithmetic carefully and end with 'ANSWER: <number> <unit>'.\nProblem: {prompt}",max_new_tokens=max(self.tier3_max,200),do_sample=True,temperature=0.6,kb_top_k=0)
            except Exception:continue
            tot+=na;v=self._final_num(a)
            if v is not None:samples.append((v,a))
        if not samples:return None,tot
        win=Counter(v for v,_ in samples).most_common(1)[0][0]
        for v,a in samples:
            if v==win:return a,tot
        return None,tot
    def _tier37_calc_tool(self,prompt:str,subject:Optional[str])->Tuple[Optional[str],int]:
        import re as _re
        s1_sys='You are solving a word problem. Reason step by step. Identify what arithmetic computation is needed. Be concise.'
        try:reasoning,n1=self.tier3_svc.chat(prompt,system=s1_sys,max_new_tokens=self.tier3_max,do_sample=False,kb_top_k=0)
        except Exception:reasoning,n1='',0
        s2_sys='Given a word problem and reasoning, write ONLY a single Python arithmetic expression that computes the answer. Use only digits, +, -, *, /, **, parens, decimal points. No variables, no functions, no units, no text. Just the expression. Example: 50 * 0.10\nRespond with the expression and nothing else.'
        s2_prompt=f'Problem: {prompt}\n\nReasoning: {(reasoning or "")[:500]}\n\nPython expression:'
        try:expr_resp,n2=self.tier3_svc.chat(s2_prompt,system=s2_sys,max_new_tokens=40,do_sample=False,kb_top_k=0)
        except Exception:expr_resp,n2='',0
        total_n=n1+n2
        if not expr_resp:return None,total_n
        for line in (expr_resp or '').split('\n'):
            line=line.strip()
            if not line:continue
            m=_re.search(r'([0-9][0-9+\-*/.()\s]*[0-9\)])',line)
            if not m:continue
            expr=m.group(1).strip()
            if not _re.match(r'^[0-9+\-*/.()\s]+$',expr) or not _re.search(r'\d',expr):continue
            try:result=eval(expr,{'__builtins__':{}},{})
            except Exception:continue
            if isinstance(result,bool):continue
            if isinstance(result,float):
                if abs(result-round(result))<1e-9:result=int(round(result))
                else:result=round(result,2)
            return str(result),total_n
        return None,total_n
    def _tier36_chord_sample(self,prompt:str,baseline_ans:str,subject:Optional[str])->Tuple[str,int]:
        import random as _r,re as _re
        from collections import Counter as _Counter
        rng=_r.Random(hash(prompt)&0xffff)
        frames=rng.sample(_CHORD_PERSONAS,min(self.chord_n_frames,len(_CHORD_PERSONAS)))
        base_sys='You are solving a problem. Show your reasoning briefly.'
        ans_format=' On the final line, write only: Answer: <number>'
        m=_re.search(r'(?i)answer\s*[:=]\s*\$?(\d+(?:\.\d+)?)',baseline_ans or '')
        baseline_num=m.group(1).rstrip('.') if m else (_re.findall(r'\$?(\d+(?:\.\d+)?)',baseline_ans or '')[-1].rstrip('.') if _re.findall(r'\$?(\d+(?:\.\d+)?)',baseline_ans or '') else baseline_ans)
        attempts=[baseline_num,baseline_num];total_n=0
        for frame in frames:
            sys_p=f'{frame} {base_sys}{ans_format}'
            try:resp,n=self.tier3_svc.chat(prompt,system=sys_p,max_new_tokens=self.tier3_max,do_sample=False,kb_top_k=0)
            except Exception:resp,n='',0
            total_n+=n
            m=list(_re.finditer(r'(?i)answer\s*[:=]\s*\$?(\d+(?:\.\d+)?)',resp or ''))
            if m:attempts.append(m[-1].group(1).rstrip('.'))
            elif (resp or '').strip():
                nums=_re.findall(r'\$?(\d+(?:\.\d+)?)',resp or '')
                if nums:attempts.append(nums[-1].rstrip('.'))
        cleaned=[a for a in attempts if a is not None and a!='']
        if not cleaned:return baseline_ans,total_n
        counts=_Counter(cleaned)
        top,top_count=counts.most_common(1)[0]
        n_total=len(cleaned)
        if top_count*2<=n_total:return baseline_num,total_n
        return top,total_n
    def _tier35_shape_sorter(self,prompt:str,initial_letter:str,subject:Optional[str])->Tuple[str,int]:
        sys_p='You are a verifier. Substitute the proposed option back into the problem and check whether ALL stated constraints are satisfied. Reply with one word: PASS or FAIL.'
        verdicts={};total_n=0
        for letter in 'ABCD':
            q=f'Problem:\n{prompt}\n\nProposed answer: option {letter}\n\nDoes option {letter} satisfy all the constraints in the problem? Reply PASS or FAIL.'
            try:resp,nn=self.tier3_svc.chat(q,system=sys_p,max_new_tokens=4,do_sample=False,kb_top_k=0)
            except Exception:resp,nn='',0
            total_n+=nn
            verdicts[letter]='pass' in (resp or '').lower()[:8]
        passers=[k for k,v in verdicts.items() if v]
        return (passers[0],total_n) if len(passers)==1 else (initial_letter,total_n)
    def _tier3_cold(self,prompt:str,subject:Optional[str])->Tuple[str,int,float,bool]:
        s1_sys=_TUTOR_PROMPTS.get(subject or 'math')
        if not self.letter_only and s1_sys is not None:s1_sys=s1_sys+' On the FINAL line of your response, write only: Answer: <number>'
        try:cot,nc=self.tier3_svc.chat(prompt,system=s1_sys,max_new_tokens=self.tier3_max,do_sample=False,kb_top_k=0)
        except Exception:cot,nc='',0
        confidence=1.0
        try:
            if hasattr(self.tier3_svc,'_next_token_top_prob'):confidence=float(self.tier3_svc._next_token_top_prob(prompt) or 1.0)
        except Exception:pass
        escalated=False
        if self.escalation_svc is not None and confidence<self.escalate_min_conf:
            try:cot2,nc2=self.escalation_svc.chat(prompt,system=s1_sys,max_new_tokens=self.tier3_max,do_sample=False,kb_top_k=0);cot,nc,escalated=cot2,nc+nc2,True
            except Exception:pass
        if not self.letter_only:return cot,nc,confidence,escalated
        s2_sys='Given the worked solution and the multiple-choice question, respond with ONLY the single letter (A, B, C, or D) of the correct option.'
        s2_prompt=f'Solution scratch:\n{cot[:600]}\n\nQuestion:\n{prompt}\n\nAnswer (single letter only):'
        try:resp,n2=(self.escalation_svc or self.tier3_svc).chat(s2_prompt,system=s2_sys,max_new_tokens=4,do_sample=False,kb_top_k=0)
        except Exception:resp,n2='A',0
        return self._extract_letter(resp),nc+n2,confidence,escalated
    def _mcq_solve_verify(self,prompt:str):
        solve_sys='You are an expert. Solve this multiple-choice problem step by step, then end with "The answer is (X)." where X is the correct letter.'
        chk_sys='You are a STRICT checker. Take the proposed option, substitute it back into the problem, and verify it satisfies EVERY stated condition/equation. Show the check briefly. Default to FAIL if it does not clearly hold. End with exactly "VERDICT: PASS" or "VERDICT: FAIL".'
        n=0
        try:cot,c=self.tier3_svc.chat(prompt,system=solve_sys,max_new_tokens=self.mcq_max_tokens,do_sample=False,kb_top_k=0)
        except Exception:cot,c='',0
        n+=c;cand=_mcq_letter(cot);work=cot
        for _ in range(2):
            try:v,cv=self.tier3_svc.chat(f'Problem:\n{prompt}\n\nProposed answer: option ({cand}). Plug option ({cand}) back into the problem and check it satisfies every stated condition.',system=chk_sys,max_new_tokens=400,do_sample=False,kb_top_k=0)
            except Exception:v,cv='',0
            n+=cv;vu=(v or '').upper();seg=vu[vu.rfind('VERDICT'):] if 'VERDICT' in vu else vu[-24:]
            if 'FAIL' not in seg:return cand,work,n,True
            try:cot,c=self.tier3_svc.chat(f'Problem:\n{prompt}\n\nYour answer ({cand}) FAILED a substitution check:\n{(v or "")[-280:]}\nSolve again carefully, avoiding that mistake, and end with "The answer is (X)."',system=solve_sys,max_new_tokens=self.mcq_max_tokens,do_sample=False,kb_top_k=0)
            except Exception:cot,c='',0
            n+=c;new=_mcq_letter(cot)
            if not new or new==cand:break
            cand=new;work=cot
        return cand,work,n,False
    def _answer_mcq(self,prompt:str,writeback:bool,t0:float)->Tuple[str,str,int]:
        cached=self.lut.lookup(prompt)
        if cached is not None:
            self._tier_counts['tier1_lut']+=1;self._wall_counts['tier1_lut']+=time.time()-t0
            return cached['a'],'tier1_lut',0
        if self.semantic_lut is not None:
            try:
                eff=self.semantic_lut.auto_margin() if self.semantic_margin=='auto' else self.semantic_margin
                sem=self.semantic_lut.lookup_soft(prompt,margin=eff)
            except Exception:sem=None
            if sem is not None:
                self._tier_counts['tier1_5_semantic']+=1;self._wall_counts['tier1_5_semantic']+=time.time()-t0
                return sem,'tier1_5_semantic',0
        subj=_select_subject(prompt)
        if subj in ('math','physics','chemistry','biology'):
            letter,cot,n,verified=self._mcq_solve_verify(prompt)
        else:
            sys_p='You are an expert taking a multiple-choice exam. Think step by step, then end your response with "The answer is (X)." where X is the correct letter.'
            try:cot,n=self.tier3_svc.chat(prompt,system=sys_p,max_new_tokens=self.mcq_max_tokens,do_sample=False,kb_top_k=0)
            except Exception:cot,n='',0
            letter=_mcq_letter(cot);verified=False
        ans=letter if self.letter_only else cot
        tier='tier3_verify' if verified else 'tier3_cold'
        self._tier_counts['tier3_cold']+=1;self._token_counts['tier3_cold']+=n;self._wall_counts['tier3_cold']+=time.time()-t0
        if writeback:
            try:
                k=self.lut.store(prompt,ans,subject=subj,source=tier,meta={'tokens':n,'reasoning':(cot or '')[:600],'verified':verified})
                if self.lut_index is not None:self.lut_index.add(k,prompt,ans)
                if self.semantic_lut is not None:
                    try:self.semantic_lut.add(prompt,ans);self.semantic_lut.fit()
                    except Exception:pass
                if self.macro_cache is not None:self.macro_cache.intern(ans if isinstance(ans,str) else str(ans))
            except Exception:pass
        return ans,tier,n
    def answer(self,prompt:str,writeback:bool=True)->Tuple[str,str,int]:
        t0=time.time()
        if _is_mcq(prompt):return self._answer_mcq(prompt,writeback,t0)
        vg=self.verify_loop and not self.letter_only and _select_subject(prompt) in ('math','physics','chemistry','engineering')
        cached=self.lut.lookup(prompt)
        if cached is not None and (not vg or self._verify_answer(prompt,cached['a'])):
            self._tier_counts['tier1_lut']+=1;self._wall_counts['tier1_lut']+=time.time()-t0
            return cached['a'],'tier1_lut',0
        if self.semantic_lut is not None and not vg:
            try:
                eff_margin=self.semantic_lut.auto_margin() if self.semantic_margin=='auto' else self.semantic_margin
                sem_ans=self.semantic_lut.lookup_soft(prompt,margin=eff_margin)
            except Exception:sem_ans=None
            if sem_ans is not None and (not vg or self._verify_answer(prompt,sem_ans)):
                self._tier_counts['tier1_5_semantic']+=1;self._wall_counts['tier1_5_semantic']+=time.time()-t0
                return sem_ans,'tier1_5_semantic',0
        subject=_select_subject(prompt)
        concept=self.extract_concept(prompt,subject=subject) if self.use_concept_routing else ''
        if self.lut_index is not None and self.lut_index.size()>0:
            res=self.lut_index.search(prompt,k=self.tier2_k,min_score=self.tier2_cos,concept_filter=concept)
            for k,past_q,past_a,cos,past_concept in res:
                tmpl_text=f'Q: {past_q}\nA: {past_a}'
                if not self._check_relevance(prompt,tmpl_text):continue
                ans,n=self._refine_with_template(prompt,tmpl_text,subject)
                self._tier_counts['tier2_template_lut']+=1;self._token_counts['tier2_template_lut']+=n;self._wall_counts['tier2_template_lut']+=time.time()-t0
                if writeback:
                    nk=self.lut.store(prompt,ans,subject=subject,source='tier2_template_lut',meta={'template_key':k,'cos':float(cos),'concept':concept,'past_concept':past_concept,'tokens':n})
                    self.lut_index.add(nk,prompt,ans,concept=concept)
                return ans,'tier2_template_lut',n
        if subject is not None and self.kb_retriever is not None:
            results=self.kb_retriever.retrieve(prompt,k=self.tier2_k,min_score=self.tier2_cos)
            for key,template,cos in results:
                if not self._check_relevance(prompt,template):continue
                ans,n=self._refine_with_template(prompt,template,subject)
                self._tier_counts['tier2_template_kb']+=1;self._token_counts['tier2_template_kb']+=n;self._wall_counts['tier2_template_kb']+=time.time()-t0
                if writeback:
                    nk=self.lut.store(prompt,ans,subject=subject,source='tier2_template_kb',meta={'kb_key':key,'cos':float(cos),'tokens':n})
                    if self.lut_index is not None:self.lut_index.add(nk,prompt,ans)
                return ans,'tier2_template_kb',n
        if subject is None:
            if self.crawler is not None and self.always_crawl_fallback:
                try:
                    vanilla_ans,vn,_=self._vanilla_letter(prompt)
                    crawl_ans,sources,cn=self.crawler.crawl_and_distill(prompt,subject=None,letter_only=self.letter_only)
                    if crawl_ans and self.letter_only:
                        crawl_letter=self._extract_letter(crawl_ans)
                        agree=crawl_letter==vanilla_ans
                        if agree:
                            final=crawl_letter;source_tag='tier4_crawler_consensus'
                            self._tier_counts['tier4_crawler']+=1;self._token_counts['tier4_crawler']+=cn+vn;self._wall_counts['tier4_crawler']+=time.time()-t0
                            if writeback:
                                self.lut.store(prompt,final,subject=None,source=source_tag,meta={'tokens':cn+vn,'sources':sources[:3],'vanilla':vanilla_ans,'crawl':crawl_letter,'agree':True})
                            return final,'tier4_crawler',cn+vn
                        else:
                            self._tier_counts['fallback_vanilla']+=1;self._token_counts['fallback_vanilla']+=vn;self._wall_counts['fallback_vanilla']+=time.time()-t0
                            if writeback:
                                self.lut.store(prompt,vanilla_ans,subject=None,source='vanilla_kept_disagree',meta={'tokens':vn,'crawl_disagreed':crawl_letter,'crawl_tokens':cn,'sources':sources[:3]})
                            return vanilla_ans,'fallback_vanilla',vn
                except Exception as e:print(f'  [adam_loop] crawler-fallback failed: {e}',flush=True)
            ans,n,_=self._vanilla_letter(prompt)
            self._tier_counts['fallback_vanilla']+=1;self._token_counts['fallback_vanilla']+=n;self._wall_counts['fallback_vanilla']+=time.time()-t0
            if writeback:
                self.lut.store(prompt,ans,subject=None,source='vanilla',meta={'tokens':n})
            return ans,'fallback_vanilla',n
        ans,n,conf,escalated=self._tier3_cold(prompt,subject)
        sorter_swapped=False;chord_used=False;calc_used=False;initial_ans=ans
        if self.verify_loop and not self.letter_only and subject in ('math','physics','chemistry','engineering'):
            cvans,cvn=self._consistency_solve(prompt);n+=cvn
            if cvans is not None:ans=cvans;calc_used=True
        if self.shape_sorter and self.letter_only:
            swapped,sn=self._tier35_shape_sorter(prompt,ans,subject)
            n+=sn
            if swapped!=ans:ans=swapped;sorter_swapped=True
        if self.calc_tool and not self.letter_only and not calc_used and subject in ('math','physics','chemistry','engineering'):
            calc_ans,cn=self._tier37_calc_tool(prompt,subject)
            n+=cn
            if calc_ans is not None:
                mnum=self._final_num(ans)
                try:agree=mnum is None or abs(float(calc_ans)-mnum)<=1e-6*max(1.0,abs(mnum))
                except Exception:agree=mnum is None
                if agree:ans=calc_ans;calc_used=True
        if self.chord_sampler and not self.letter_only and not calc_used and conf<self.chord_min_conf:
            chord_ans,cn=self._tier36_chord_sample(prompt,ans,subject)
            n+=cn
            if chord_ans!=ans:ans=chord_ans;chord_used=True
        verify_used=False
        if self.verify_loop and not self.letter_only and subject in ('math','physics','chemistry','engineering'):
            ans2,vn=self._verify_refine(prompt,subject,ans);n+=vn
            if ans2 and ans2!=ans:ans=ans2;verify_used=True
        if self.crawler is not None and conf<self.tier4_min_conf and not (self.verify_loop and (calc_used or verify_used)):
            try:
                crawl_ans,sources,cn=self.crawler.crawl_and_distill(prompt,subject=subject,letter_only=self.letter_only)
                if crawl_ans:
                    self._tier_counts['tier4_crawler']+=1;self._token_counts['tier4_crawler']+=cn;self._wall_counts['tier4_crawler']+=time.time()-t0
                    if writeback:
                        k=self.lut.store(prompt,self._extract_letter(crawl_ans) if self.letter_only else crawl_ans,subject=subject,source='tier4_crawler',meta={'tokens':cn,'sources':sources[:3],'conf':conf})
                        if self.lut_index is not None:self.lut_index.add(k,prompt,crawl_ans)
                    return (self._extract_letter(crawl_ans) if self.letter_only else crawl_ans),'tier4_crawler',cn
            except Exception as e:print(f'  [adam_loop] crawler escalation failed: {e}',flush=True)
        bucket='tier37_calc_tool' if calc_used else ('tier36_chord_sampler' if chord_used else ('tier35_shape_sorter' if sorter_swapped else ('tier3_escalated' if escalated else 'tier3_cold')))
        self._tier_counts[bucket]+=1;self._token_counts[bucket]+=n;self._wall_counts[bucket]+=time.time()-t0
        if writeback and ((not vg) or self._verify_answer(prompt,ans)):
            k=self.lut.store(prompt,ans,subject=subject,source=bucket,meta={'tokens':n,'conf':conf,'escalated':escalated,'sorter_swap':sorter_swapped,'chord_used':chord_used,'initial':initial_ans})
            if self.lut_index is not None:self.lut_index.add(k,prompt,ans)
            if self.macro_cache is not None:self.macro_cache.intern(ans if isinstance(ans,str) else str(ans))
            if self.semantic_lut is not None and not vg and bucket in ('tier37_calc_tool','tier36_chord_sampler','tier35_shape_sorter','tier3_escalated','tier3_cold'):
                try:self.semantic_lut.add(prompt,ans);self.semantic_lut.fit()
                except Exception as e:print(f'  [adam_loop] semantic_lut writeback failed: {e}',flush=True)
        return ans,bucket,n

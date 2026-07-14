"""CorrectionTrace — the 'toggle a switch' weight-correction cycle (Anthony 2026-07-02): Prompt -> wrong answer + feedback -> self-diagnosed REASON -> PTEX documentation (ledger + lesson homes) -> INSTANT switch (MemoryBus record_learning + anti-pattern suppress of the wrong answer -> tier0_atex_override flips the reply immediately) -> NONCE identification (sem-lut cell = the semantic address) -> SELECTIVE retrain (GatedPageBank.add_domain_supervised: additive low-rank page on late MLP, cosine-gated AT the query's address — activates near the nonce, inert everywhere else, base weights untouched cos=1) -> re-ask VERIFICATION -> post-documentation. Literal in-place base-weight swapping is forbidden by the paradigm (cos=1 invariant + AsimovLayer integrity); the gated page IS the switch."""
import os,json,time,hashlib
from typing import Dict,Any
class CorrectionTrace:
    def __init__(s,adam,agent=None,root='experiences/corrections'):
        s.adam=adam;s.agent=agent;s.root=root;os.makedirs(root,exist_ok=True)
        s.ledger=os.path.join(root,'trace_ledger.jsonl')
    def _bus(s):return getattr(s.agent,'memory_bus',None) or getattr(s.adam,'bus',None)
    def _log(s,rec):
        rec['ts']=time.time()
        open(s.ledger,'a',encoding='utf-8').write(json.dumps(rec,ensure_ascii=False)+'\n');return rec
    def _diagnose(s,prompt,wrong,correct,feedback):
        try:
            r=s.adam.ask(f'You previously answered this question WRONG.\nQ: {prompt}\nYour wrong answer: {wrong}\nCorrect answer: {correct}\n'+(f'Feedback: {feedback}\n' if feedback else '')+'In ONE sentence, state the most likely reason the wrong answer was produced (misread constant, wrong formula, stale fact, retrieval miss, ambiguity).',writeback=False)
            a=(r.get('answer') or '') if isinstance(r,dict) else str(r or '')
            return a.strip()[:300] or '(no diagnosis)'
        except Exception as e:return f'(diagnosis unavailable: {e})'[:200]
    def _nonce(s,prompt):
        sl=getattr(s.adam,'sem_lut',None)
        try:
            cell,_,_=sl._project(prompt);return [int(c) for c in cell]
        except Exception:return None
    def correct(s,prompt:str,wrong:str,correct:str,feedback:str='',retrain:bool=True,steps:int=240,lr:float=3e-4,verify:bool=True)->Dict[str,Any]:
        cid=hashlib.sha256(f'{prompt}|{correct}'.encode('utf-8','ignore')).hexdigest()[:10]
        reason=s._diagnose(prompt,wrong,correct,feedback)
        s._log({'id':cid,'phase':'documented','prompt':prompt[:400],'wrong':(wrong or '')[:400],'correct':correct[:400],'feedback':(feedback or '')[:300],'inferred_reason':reason,'nonce':s._nonce(prompt)})
        bus=s._bus();switched=False
        if bus is not None:
            try:
                bus.record_learning(prompt,correct,kind='correction',provenance=f'correction_trace:{cid}',exactness='exact')
                if wrong:bus.suppress(wrong,reason=reason,q=prompt)
                switched=True
            except Exception:pass
        if not switched:
            try:s.adam.teach(prompt,correct,source=f'correction_trace:{cid}');switched=True
            except Exception:pass
        page=None
        if retrain:
            try:
                bank=s.adam._ensure_gated_bank()
                lf=bank.add_domain_supervised(f'corr_{cid}',[prompt],[correct],steps=steps,lr=lr) if hasattr(bank,'add_domain_supervised') else s.adam.teach_weight(f'corr_{cid}',[f'Q: {prompt}\nA: {correct}'],steps=steps,lr=lr).get('final_loss')
                page={'domain':f'corr_{cid}','final_loss':round(float(lf),3) if lf is not None else None}
            except Exception as e:page={'domain':f'corr_{cid}','error':str(e)[:200]}
        ver={'ran':False}
        if verify:
            try:
                r=s.adam.ask(prompt,writeback=False)
                a=(r.get('answer') or '') if isinstance(r,dict) else str(r or '')
                ver={'ran':True,'tier':r.get('tier') if isinstance(r,dict) else None,'flipped':correct.lower()[:60] in a.lower(),'answer_head':a[:160]}
            except Exception as e:ver={'ran':False,'error':str(e)[:160]}
        return s._log({'id':cid,'phase':'corrected','instant_switch':switched,'nonce':s._nonce(prompt),'gated_page':page,'verify':ver,'inferred_reason':reason})
    def history(s,n:int=20):
        try:lines=open(s.ledger,encoding='utf-8').read().splitlines()[-int(n):]
        except Exception:return []
        out=[]
        for l in lines:
            try:out.append(json.loads(l))
            except Exception:pass
        return out

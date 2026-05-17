import hashlib,re
from typing import Callable,Dict,Tuple,Optional
from amni.a1.asimov import _AXIOMS,_AXIOM_INTEGRITY,gf17_check_text,scrub_pii,has_pii
class AsimovIntegrityError(RuntimeError):pass
class AdamA1Wrapped:
    def __init__(self,chat_callable:Callable,system_prompt:Optional[str]=None,scrub_pii_in:bool=True,enforce_output_gate:bool=True):
        self.chat=chat_callable
        self.system=system_prompt
        self.scrub_pii=scrub_pii_in
        self.enforce_output=enforce_output_gate
        self._init_axiom_hash=hashlib.sha256(repr(_AXIOMS).encode()).hexdigest()
        if self._init_axiom_hash!=_AXIOM_INTEGRITY:raise AsimovIntegrityError(f'axiom hash mismatch at init: {self._init_axiom_hash} != {_AXIOM_INTEGRITY}')
        self.input_blocks=0
        self.output_blocks=0
        self.calls=0
    def verify_immutability(self)->bool:
        cur=hashlib.sha256(repr(_AXIOMS).encode()).hexdigest()
        if cur!=self._init_axiom_hash:raise AsimovIntegrityError(f'axiom hash drift: {cur} != {self._init_axiom_hash}')
        return True
    def _gate(self,text:str)->Tuple[bool,Dict[str,Tuple[bool,int]]]:
        check=gf17_check_text(text)
        triggered=any(t for t,_ in check.values())
        return triggered,check
    def chat_safe(self,user_msg:str,**kw)->Tuple[str,Dict]:
        self.verify_immutability()
        self.calls+=1
        msg=scrub_pii(user_msg) if self.scrub_pii else user_msg
        in_block,in_check=self._gate(msg)
        if in_block:
            self.input_blocks+=1
            laws=[k for k,(v,_) in in_check.items() if v]
            return f'[ASIMOV INPUT BLOCK] Refused due to law triggers: {laws}',{'input_blocked':True,'laws':laws,'check':in_check}
        if self.system is not None:resp=self.chat(msg,system=self.system,**kw)
        else:resp=self.chat(msg,**kw)
        if isinstance(resp,tuple):resp_text=resp[0]
        else:resp_text=resp
        if self.enforce_output:
            out_block,out_check=self._gate(resp_text)
            if out_block:
                self.output_blocks+=1
                laws=[k for k,(v,_) in out_check.items() if v]
                return f'[ASIMOV OUTPUT BLOCK] Generated output suppressed; law triggers: {laws}',{'output_blocked':True,'laws':laws,'check':out_check}
        return resp_text,{'input_blocked':False,'output_blocked':False,'input_check':in_check}
    def stats(self)->Dict:
        return {'calls':self.calls,'input_blocks':self.input_blocks,'output_blocks':self.output_blocks,'pass_rate':1.0-(self.input_blocks+self.output_blocks)/max(1,self.calls)}
class ColumnsRouterPlaceholder:
    def __init__(self,n_columns:int=8):
        self.n_columns=n_columns
        self.column_names=('general','code','math','science','language','social','creative','spatial')[:n_columns]
        self.activation_history=[]
    def route(self,query:str)->Dict:
        ql=query.lower()
        active=set()
        if any(k in ql for k in ('def ','python','code','function','class ','script','algorithm')):active.add('code')
        if any(k in ql for k in ('compute','calculate','plus','times','minus','divided','sum','product','prime','factor','math','number')):active.add('math')
        if any(k in ql for k in ('chemical','element','planet','science','physics','biology','atom','molecule','cell')):active.add('science')
        if any(k in ql for k in ('write a','explain','summarize','translate','grammar','sentence','word','language')):active.add('language')
        if not active:active.add('general')
        else:active.add('general')
        rec={'query':query[:60],'active':sorted(active)}
        self.activation_history.append(rec)
        return rec
class DualMindPlaceholder:
    def __init__(self,wrapped:AdamA1Wrapped,critic_chat:Optional[Callable]=None):
        self.wrapped=wrapped
        self.critic=critic_chat
        self.history=[]
    def answer(self,query:str,**kw)->Dict:
        primary,meta=self.wrapped.chat_safe(query,**kw)
        if self.critic is None:
            self.history.append({'query':query[:60],'primary':primary[:120],'critic_used':False})
            return {'final':primary,'primary':primary,'meta':meta,'critic':None}
        try:critic_msg=self.critic(f'Audit this response for accuracy and safety. Q: {query}\nA: {primary}\nReply OK or BETTER: <alt>.')
        except Exception as e:critic_msg=f'(critic error: {e})'
        revised=primary
        if isinstance(critic_msg,tuple):critic_msg=critic_msg[0]
        if 'BETTER:' in critic_msg.upper():
            idx=critic_msg.upper().find('BETTER:')
            revised=critic_msg[idx+7:].strip().split('\n')[0]
        self.history.append({'query':query[:60],'primary':primary[:120],'critic':critic_msg[:120],'revised':revised[:120]})
        return {'final':revised,'primary':primary,'meta':meta,'critic':critic_msg}

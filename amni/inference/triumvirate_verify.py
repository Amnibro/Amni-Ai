import re
_PROPOSER_SYS=("You are the Proposer. Answer the user's question briefly and directly. "
               "If facts are provided in the system context, use them as authoritative ground truth. "
               "Output only the answer, no preamble.")
_CRITIC_SYS=("You are the Critic. The user asked a question and got a proposed answer. "
             "Your job is to look at the proposed answer and identify any factual errors, ambiguities, or inconsistencies. "
             "Be terse. If the proposed answer looks correct and consistent, just say 'OK'. "
             "Otherwise list concerns in one short sentence.")
_REVISER_SYS=("You are the Reviser. Given the question, the Proposer's answer, and the Critic's concerns, "
              "produce the final best answer. If the Critic said 'OK', output the Proposer's answer unchanged. "
              "Otherwise output the corrected answer. Output only the final answer, no preamble.")
_CRITIC_SYS_COMPARATIVE=("You are the Critic. Given the question, the retrieved facts shown above, and a proposed answer, "
                         "perform two checks: (1) Logical test: is the proposed answer directly supported by the retrieved facts? "
                         "(2) Better-approach test: would a different, more grounded answer follow more directly from the facts? "
                         "If the proposed answer passes both, reply with exactly 'OK'. "
                         "Only if a clearly better fact-grounded alternative exists, reply with 'BETTER: <the alternative answer in <=12 words>'. "
                         "Do not invent claims beyond the retrieved facts. Do not flag stylistic concerns.")
_REVISER_SYS_COMPARATIVE=("You are the Reviser. The Critic provided an alternative answer that is more directly grounded in the retrieved facts. "
                          "Output the Critic's alternative as the final answer, in one short sentence. Do not add or speculate.")
def _is_critic_ok(text):
    if not text:return True
    t=text.strip().lower()
    if t.startswith('ok') or t=='ok.' or t=='ok!':return True
    if 'no concerns' in t or 'looks correct' in t or 'looks accurate' in t:return True
    if 'no issues' in t or 'no errors' in t or 'consistent' in t.split(',')[0]:return True
    return False
def _parse_comparative(text):
    if not text:return True,None
    t=text.strip()
    tl=t.lower()
    if tl.startswith('ok') or tl=='ok.' or tl=='ok!':return True,None
    m=re.search(r'better\s*:\s*(.+)',t,re.IGNORECASE|re.DOTALL)
    if m:
        alt=m.group(1).strip().split('\n',1)[0].strip()
        if alt:return False,alt
    return True,None
class TriumvirateVerifier:
    def __init__(self,svc,mode='classic'):
        assert mode in ('classic','comparative'),f'bad mode: {mode}'
        self.svc=svc;self.mode=mode
    def answer(self,question,facts=None,max_new_tokens=60,return_trace=False):
        proposed,_=self.svc.chat(question,system=_PROPOSER_SYS,facts=facts,max_new_tokens=max_new_tokens,do_sample=False)
        proposed=proposed.strip()
        if self.mode=='classic':
            critic_q=f'Question: {question}\nProposed answer: {proposed}'
            critique,_=self.svc.chat(critic_q,system=_CRITIC_SYS,facts=facts,max_new_tokens=80,do_sample=False)
            critique=critique.strip()
            if _is_critic_ok(critique):
                final=proposed;revised=False
            else:
                reviser_q=f'Question: {question}\nProposer answer: {proposed}\nCritic concerns: {critique}'
                final,_=self.svc.chat(reviser_q,system=_REVISER_SYS,facts=facts,max_new_tokens=max_new_tokens,do_sample=False)
                final=final.strip();revised=True
        else:
            critic_q=f'Question: {question}\nProposed answer: {proposed}'
            critique,_=self.svc.chat(critic_q,system=_CRITIC_SYS_COMPARATIVE,facts=facts,max_new_tokens=80,do_sample=False)
            critique=critique.strip()
            ok,alt=_parse_comparative(critique)
            if ok:
                final=proposed;revised=False
            else:
                reviser_q=f'Question: {question}\nProposer answer: {proposed}\nCritic alternative: {alt}'
                final,_=self.svc.chat(reviser_q,system=_REVISER_SYS_COMPARATIVE,facts=facts,max_new_tokens=max_new_tokens,do_sample=False)
                final=final.strip();revised=True
        if return_trace:return final,{'proposed':proposed,'critique':critique,'final':final,'revised':revised,'mode':self.mode}
        return final

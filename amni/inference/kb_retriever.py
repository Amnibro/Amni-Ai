"""KBRetriever: query-time retrieval over a PTEX KnowledgeBase.

Wraps amni.learning.knowledge_base.KnowledgeBase with retrieval helpers usable from
the inference path. Designed to be lightweight (no embedding model required for the
v1 substring/keyword approach) — pure mmap reads against the KB pages.

Usage:
    retr = KBRetriever('experiences/kb_canonical')
    results = retr.retrieve('How do I read a file in Python?', k=3, max_chars_per=600)
    # results = [(key, text, score), ...]

    context_block = retr.format_as_context(results)
    # 'Reference docs:\n--- python~3.12::library/pathlib\n<text>\n--- ...'

Scoring:
    v1 = simple TF-style: count of query keyword occurrences in (key + text).
    Future: embedding-based via a small sidecar model.
"""
import re
from typing import List,Tuple,Optional
from amni.learning.knowledge_base import KnowledgeBase
_TOK_RE=re.compile(r"[a-zA-Z][a-zA-Z0-9_\-\.]*")
_STOP={'the','a','an','and','or','of','to','in','for','is','are','what','how','do','i','my','this','that','with','on','at','by','as','it','be','can','use','using','from','some','any','have','has','will','would','should','could','make','get','set','put','show','give','tell','please','help'}
def _tokenize(text:str)->List[str]:
    return [t.lower() for t in _TOK_RE.findall(text or '') if t.lower() not in _STOP and len(t)>=2]
class KBRetriever:
    def __init__(self,kb_root:str):
        self.kb=KnowledgeBase(kb_root)
    def retrieve(self,query:str,k:int=3,max_chars_per:int=600,min_score:int=1,slug:Optional[str]=None)->List[Tuple[str,str,int]]:
        q_tokens=_tokenize(query)
        if not q_tokens:return []
        scored=[]
        keys=self.kb.keys()
        slug_prefix=f'{slug}::' if slug else None
        for key in keys:
            if slug_prefix and not key.startswith(slug_prefix):continue
            key_l=key.lower()
            score=sum(1 for t in q_tokens if t in key_l)
            if score==0:
                txt=self.kb.lookup(key) or ''
                if not txt:continue
                txt_l=txt.lower()
                score=sum(1 for t in q_tokens if t in txt_l)
            if score<min_score:continue
            scored.append((score,key))
        scored.sort(reverse=True)
        out=[]
        for score,key in scored[:k]:
            txt=self.kb.lookup(key) or ''
            if len(txt)>max_chars_per:txt=txt[:max_chars_per]+'...'
            out.append((key,txt,score))
        return out
    def format_as_context(self,results:List[Tuple[str,str,int]])->str:
        if not results:return ''
        lines=['Reference docs (retrieved from knowledge base):']
        for key,txt,score in results:
            lines.append(f'--- {key} (score={score})')
            lines.append(txt)
        return '\n'.join(lines)
    def stats(self):return self.kb.stats()
    def close(self):self.kb.close()

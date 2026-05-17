"""FunctionPrefixMatcher: structural-token bridge for personal_functions KB.
Per the maintainer's design: detect when the model is starting to generate a known function (matches prefix of an entry in the KB), inject the full function from KB instead of letting the model finish token-by-token. Saves tokens on boilerplate, lets compute go to specific/unique parts.
"""
import hashlib
from pathlib import Path
from typing import Optional,List,Tuple
from amni.learning.knowledge_base import KnowledgeBase
class FunctionPrefixMatcher:
    def __init__(self,kb_root,min_prefix_len=20,max_prefix_len=80):
        self.kb=KnowledgeBase(kb_root)
        self.min_prefix=min_prefix_len
        self.max_prefix=max_prefix_len
        self._prefix_index={}
        self._build_index()
    _GENERIC=frozenset(('def main():','def main()','def __init__(self):','def __init__','class foo:','def f():','def test():','def run():','if __name__','return','pass','def setup','def fn(','def f(','def g(','def h('))
    def _build_index(self):
        keys=self.kb.keys()
        plens=[self.max_prefix,60,40,30,self.min_prefix]
        for key in keys:
            content=self.kb.lookup(key) or ''
            if len(content)<self.min_prefix:continue
            for plen in plens:
                if len(content)<plen:continue
                pfx=content[:plen].strip().lower()
                if not pfx or pfx in self._GENERIC:continue
                if any(pfx.startswith(g) and len(pfx)<len(g)+8 for g in self._GENERIC):continue
                if pfx not in self._prefix_index:self._prefix_index[pfx]=(key,len(content))
        self._prefix_lens=sorted(set(len(p) for p in self._prefix_index),reverse=True)
    def find_match(self,text):
        text_low=text.lower()
        for plen in self._prefix_lens:
            if plen>len(text_low):continue
            tail=text_low[-plen:].strip()
            if tail in self._prefix_index:
                key,full_len=self._prefix_index[tail]
                content=self.kb.lookup(key)
                return {'key':key,'content':content,'tokens_saved_estimate':max(0,full_len-plen)//4}
        return None
    def stats(self):
        return {'kb_entries':len(self.kb.keys()),'indexed_prefixes':len(self._prefix_index),'prefix_lens':self._prefix_lens[:5]}

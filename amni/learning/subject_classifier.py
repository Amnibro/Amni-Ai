"""SubjectClassifier — keyword-based query subject routing for Adam-1.
Single-subject-per-query policy: each query maps to one subject overlay (or 'global' fallback).
Avoids the GF(17)-sum collapse observed when activating multiple subject overlays simultaneously.
Usage:
    cls = SubjectClassifier()
    subject = cls.classify("What is 2 + 2?")    # -> 'math'
    subject = cls.classify("Write Python code") # -> 'code'
    subject = cls.classify("How are you today?") # -> 'global' (no clear match)
"""
import re
from typing import Dict,List,Optional,Tuple
DEFAULT_KEYWORDS={
    'math':('integer','algebra','equation','solve','compute','calculat','sum','product','divid','multipl','add','subtract','squar','cube','root','factor','prime','geometr','triangle','circle','angle','derivative','integral','matrix','probabilit','statistic','percent','fraction','decimal','arithmetic','theorem','proof','math','formula','digit','plus','minus','times','equal','number','count','how many','how much','how far','how long','how old','how fast','gives','buys','spends','costs','dollar','cents','price','sale','discount','mph','kph','miles per','kilomet','meter','distance','speed','velocity','weight','height','length','width','perimeter','area','volume','minute','second','hour','day','month','year','apples','marbles','pencils','candies','cookies','total','remain','left','altogether'),
    'code':('python','javascript','java','c++','c#','rust','golang','typescript','function','method','variable','loop','recursion','algorithm','array','list','dictionary','hash','sort','search','complexity','code','program','script','debug','compile','runtime','syntax','library','framework','api','endpoint','database','sql','json','xml','html','css','regex','git','github','docker','kubernetes','linux','bash','shell','def ','import ','return','class '),
    'science':('biology','chemistry','physics','molecul','atom','electron','proton','dna','rna','protein','cell','organism','ecosystem','evolution','energy','force','momentum','velocity','acceleration','gravity','quantum','relativity','reaction','catalyst','element','compound','periodic','genome','enzyme','species','climate','planet','star','galaxy','universe','theory','experiment','hypothesis','photosynth','mitochondri','chromosome','gene','virus','bacteri'),
    'language':('grammar','sentence','noun','verb','adjective','adverb','tense','conjug','translat','language','linguistics','etymolog','pronunci','spelling','phrase','idiom','metaphor','synonym','antonym','poem','poetry','novel','literature','author','plot','theme','prose','rhetoric','vocabular'),
    'history':('history','historical','ancient','medieval','renaissance','revolution',' war ','battle','empire','kingdom','civilizat','dynasty','century','treaty','president','king ','queen','emperor','pharaoh','coloniz','independence','democracy','monarchy','republic','reform','prehistor','archaeolog','artifact','heritage','tradition'),
    'reasoning':('therefore','because','if then','suppose','given that','it follows','infer','deduce','conclude','impl','assume','premise','contradicts','consistent','valid','invalid','syllogism','logic','reasoning','argument','fallacy','rebut','counter-argument'),
}
_MATH_OP_RE=re.compile(r'(\d+\s*[+\-*/×÷=]\s*\d+)|(\d+\s*(?:plus|minus|times|divided by|multiplied by|over|to the power of)\s*\d+)|(how many\s+\w+\s+(?:in|are|do))|((?:square|cube)\s+root)|(\d+\s*\^\s*\d+)|(\d+\s*%\s*of)',re.IGNORECASE)
_CODE_PATTERN_RE=re.compile(r'(```|^\s*(?:def |class |import |from |if __name__|public class|fn |func |let |const |var ))|(\.py\b|\.js\b|\.ts\b|\.rs\b|\.go\b|\.cpp\b)',re.IGNORECASE|re.MULTILINE)
class SubjectClassifier:
    def __init__(self,keywords:Optional[Dict[str,Tuple[str,...]]]=None,fallback='global',min_score=1):
        self.keywords=keywords or DEFAULT_KEYWORDS
        self.fallback=fallback
        self.min_score=min_score
        self._patterns={}
        for s,kws in self.keywords.items():
            patterns=[]
            for kw in kws:
                if kw.startswith(' ') or kw.endswith(' '):patterns.append(re.compile(r'(?<![a-zA-Z])'+re.escape(kw.strip())+r'(?![a-zA-Z])',re.IGNORECASE))
                else:patterns.append(re.compile(r'(?<![a-zA-Z])'+re.escape(kw),re.IGNORECASE))
            self._patterns[s]=patterns
    def score_all(self,text:str)->Dict[str,int]:
        out={}
        for subject,patterns in self._patterns.items():
            score=sum(1 for p in patterns if p.search(text))
            out[subject]=score
        if 'math' in out and _MATH_OP_RE.search(text):out['math']=out.get('math',0)+2
        if 'code' in out and _CODE_PATTERN_RE.search(text):out['code']=out.get('code',0)+2
        return out
    def classify(self,text:str)->str:
        if not text or not text.strip():return self.fallback
        scores=self.score_all(text)
        best=max(scores.items(),key=lambda kv:kv[1])
        return best[0] if best[1]>=self.min_score else self.fallback
    def classify_with_confidence(self,text:str)->Tuple[str,int,Dict[str,int]]:
        scores=self.score_all(text)
        best=max(scores.items(),key=lambda kv:kv[1])
        chosen=best[0] if best[1]>=self.min_score else self.fallback
        return chosen,best[1],scores
    def list_subjects(self)->List[str]:return list(self.keywords.keys())

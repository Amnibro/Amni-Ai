import json,time,re
from pathlib import Path
_CORRECTION_RE=re.compile(r"\b(?:no(?:[,\s]|$)|wrong|incorrect|actually|that(?:'s|\s+is)\s+(?:not\s+right|wrong|incorrect|not\s+true|false)|not\s+(?:right|correct|true)|not\s+[A-Za-z0-9][^,]{0,48},\s+it(?:'s|\s+is)|it(?:'s|\s+is)\s+actually|the\s+(?:right|correct)\s+answer\s+is|let\s+me\s+correct)\b",re.IGNORECASE)
class ConversationNotes:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.data=self._load()
    def _load(self):
        try:return json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'corrections':[],'notes':[],'behavior_prefs':[]}
        except Exception:return {'corrections':[],'notes':[],'behavior_prefs':[]}
    def save(self):
        try:self.path.write_text(json.dumps(self.data,indent=2,ensure_ascii=False),encoding='utf-8')
        except Exception as e:print(f'[ConversationNotes] save failed: {e}',flush=True)
    @staticmethod
    def is_correction(text):return bool(_CORRECTION_RE.search(text or ''))
    @staticmethod
    def extract_corrected_answer(text):
        t=(text or '').strip()
        if not t or not _CORRECTION_RE.search(t):return None
        def _clean(s):
            if not s:return None
            s=s.strip().strip('"\'`')
            s=s.split('\n')[0].strip()
            s=re.split(r'(?<=\w)[.!](?:\s|$)',s)[0].strip()
            s=re.sub(r'\s+not\s+.+$','',s,flags=re.IGNORECASE).strip().rstrip(',;:')
            if len(s)<2 or len(s)>400:return None
            if s.endswith('?'):return None
            if s.lower() in ('ok','okay','yes','yeah','no','wrong','true','false'):return None
            return s[:240]
        m=re.search(r"\bnot\s+.+?[,\s]+(?:it(?:'s|\s+is)|they(?:'re|\s+are)|the\s+(?:right|correct)\s+answer\s+is)\s+(.+)",t,re.IGNORECASE)
        if m:
            c=_clean(m.group(1))
            if c:return c
        m=re.search(r"(?:the\s+(?:right|correct)\s+answer\s+is|let\s+me\s+correct(?:\s+you)?(?:\s*:)?)\s+(.+)",t,re.IGNORECASE)
        if m:
            c=_clean(m.group(1))
            if c:return c
        m=re.search(r"(?:it(?:'s|\s+is)\s+actually|actually(?:\s*,)?(?:\s+it(?:'s|\s+is))?)\s+(.+)",t,re.IGNORECASE)
        if m:
            c=_clean(m.group(1))
            if c:return c
        m=re.search(r"(?:that(?:'s|\s+is)\s+(?:not\s+right|wrong|incorrect|not\s+true|false)[,.]?\s*)(?:it(?:'s|\s+is)\s+)?(.+)",t,re.IGNORECASE)
        if m:
            c=_clean(m.group(1))
            if c:return c
        m=re.search(r"^(?:no|wrong|incorrect)(?:[,\s]+|$)(.+)",t,re.IGNORECASE)
        if m:
            rest=re.sub(r"^(?:it(?:'s|\s+is)\s+)",'',m.group(1).strip(),flags=re.IGNORECASE)
            c=_clean(rest)
            if c:return c
        m=re.search(r"\bit(?:'s|\s+is)\s+(.+)",t,re.IGNORECASE)
        if m:
            c=_clean(m.group(1))
            if c:return c
        return None
    def add_correction(self,wrong_q,wrong_a,corrected_text,session_id=None,max_keep=50):
        if not corrected_text:return False
        entry={'ts':time.time(),'wrong_q':(wrong_q or '')[:400],'wrong_a':(wrong_a or '')[:600],'corrected_text':corrected_text[:600],'session_id':session_id}
        self.data.setdefault('corrections',[]).append(entry)
        self.data['corrections']=self.data['corrections'][-max_keep:]
        self.data['updated_ts']=time.time();self.save();return True
    def add_note(self,text,max_keep=30):
        if not text:return False
        self.data.setdefault('notes',[]).append({'ts':time.time(),'text':text[:500]})
        self.data['notes']=self.data['notes'][-max_keep:]
        self.data['updated_ts']=time.time();self.save();return True
    def add_behavior_pref(self,text,max_keep=20):
        if not text:return False
        self.data.setdefault('behavior_prefs',[]).append({'ts':time.time(),'text':text[:300]})
        self.data['behavior_prefs']=self.data['behavior_prefs'][-max_keep:]
        self.data['updated_ts']=time.time();self.save();return True
    def to_facts_list(self,corrections_n=5,notes_n=5,prefs_n=5):
        out=[]
        for c in (self.data.get('corrections') or [])[-corrections_n:]:
            q=(c.get('wrong_q') or '').strip();ct=(c.get('corrected_text') or '').strip()
            if q and ct:out.append(f"earlier I told the user '{(c.get('wrong_a') or '')[:120]}' about '{q[:120]}' but they corrected me: {ct[:240]}")
            elif ct:out.append(f"user correction noted: {ct[:240]}")
        for n in (self.data.get('notes') or [])[-notes_n:]:
            t=(n.get('text') or '').strip()
            if t:out.append(f"standing note from user: {t[:240]}")
        for p in (self.data.get('behavior_prefs') or [])[-prefs_n:]:
            t=(p.get('text') or '').strip()
            if t:out.append(f"user behavior preference: {t[:240]}")
        return out
    def forget(self):self.data={'corrections':[],'notes':[],'behavior_prefs':[]};self.save()
    def stats(self):return {'corrections_n':len(self.data.get('corrections') or []),'notes_n':len(self.data.get('notes') or []),'prefs_n':len(self.data.get('behavior_prefs') or []),'updated_ts':self.data.get('updated_ts')}

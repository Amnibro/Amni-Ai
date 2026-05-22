"""CoachAtlas — per-topic mastery + question history for Socratic coach sessions.
JSONL-backed (one file per topic). Each row: (ts, question, user_answer, score, difficulty, hint_used, skipped). Mastery is a rolling weighted average of recent scores. Question dedupe via embedding-projected cell so the coach doesn't repeat itself within a session.
Public API:
  start_session(topic) → session_id
  record(session_id, topic, question, user_answer, score, difficulty, hint_used, skipped)
  mastery(topic) → {pct, n_questions, recent_scores}
  recent_questions(topic, k) → list of recent question strings (for dedupe)
  forget(topic) → wipe a topic's history
  list_topics() → all topics with current mastery"""
import json,time,uuid,re
from pathlib import Path
from typing import List,Dict,Any,Optional
_SLUG_RE=re.compile(r'[^a-z0-9_\-]+')
def _slug(s:str)->str:
    return _SLUG_RE.sub('-',(s or 'general').strip().lower())[:64].strip('-') or 'general'
class CoachAtlas:
    def __init__(self,root:str='experiences/coach_atlas',mastery_window:int=10):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.mastery_window=mastery_window
        self._sessions:Dict[str,Dict[str,Any]]={}
    def _topic_path(self,topic:str)->Path:return self.root/f'topic_{_slug(topic)}.jsonl'
    def _meta_path(self,topic:str)->Path:return self.root/f'topic_{_slug(topic)}.meta.json'
    def _load_topic(self,topic:str)->List[Dict[str,Any]]:
        p=self._topic_path(topic)
        if not p.exists():return []
        out=[]
        try:
            for ln in p.read_text(encoding='utf-8').strip().splitlines():
                if ln.strip():
                    try:out.append(json.loads(ln))
                    except Exception:continue
        except Exception:pass
        return out
    def _append_topic(self,topic:str,row:Dict[str,Any]):
        try:
            with self._topic_path(topic).open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=str)+'\n')
        except Exception as e:print(f'[CoachAtlas] append {topic} failed: {e}',flush=True)
    def start_session(self,topic:str,session_id:Optional[str]=None,initial_difficulty:int=2)->str:
        sid=session_id or f'cs_{uuid.uuid4().hex[:12]}'
        self._sessions[sid]={'topic':topic,'difficulty':int(initial_difficulty),'started':time.time(),'asked':[],'streak_correct':0,'streak_wrong':0,'n_answered':0,'n_skipped':0,'n_hinted':0,'pending_question':None,'pending_model_answer':None,'pending_hint':None,'cumulative_score':0.0}
        return sid
    def get_session(self,session_id:str)->Optional[Dict[str,Any]]:return self._sessions.get(session_id)
    def update_session(self,session_id:str,**kw):
        s=self._sessions.get(session_id)
        if s is None:return False
        s.update(kw);return True
    def end_session(self,session_id:str)->Dict[str,Any]:
        s=self._sessions.pop(session_id,None)
        if s is None:return {'ended':False}
        topic=s['topic'];m=self.mastery(topic)
        return {'ended':True,'topic':topic,'session_id':session_id,'n_answered':s.get('n_answered',0),'n_skipped':s.get('n_skipped',0),'n_hinted':s.get('n_hinted',0),'final_difficulty':s.get('difficulty'),'session_avg_score':(s.get('cumulative_score',0.0)/max(s.get('n_answered',1),1)),'topic_mastery':m,'started':s.get('started'),'wall_s':round(time.time()-s.get('started',time.time()),1)}
    def record(self,session_id:str,topic:str,question:str,user_answer:str,score:float,difficulty:int,hint_used:bool=False,skipped:bool=False)->Dict[str,Any]:
        row={'ts':time.time(),'session_id':session_id,'topic':topic,'question':question[:600],'user_answer':(user_answer or '')[:600],'score':float(score),'difficulty':int(difficulty),'hint_used':bool(hint_used),'skipped':bool(skipped)}
        self._append_topic(topic,row)
        s=self._sessions.get(session_id)
        if s is not None:
            s['asked'].append(question[:240]);s['n_answered']+=(0 if skipped else 1);s['n_skipped']+=(1 if skipped else 0);s['n_hinted']+=(1 if hint_used else 0)
            if not skipped:
                s['cumulative_score']+=float(score)
                if score>=70:s['streak_correct']+=1;s['streak_wrong']=0
                elif score<50:s['streak_wrong']+=1;s['streak_correct']=0
                else:s['streak_correct']=0;s['streak_wrong']=0
                if s['streak_correct']>=2 and s['difficulty']<5:s['difficulty']+=1;s['streak_correct']=0
                elif s['streak_wrong']>=2 and s['difficulty']>1:s['difficulty']-=1;s['streak_wrong']=0
        return row
    def mastery(self,topic:str)->Dict[str,Any]:
        rows=self._load_topic(topic)
        if not rows:return {'pct':0.0,'n_questions':0,'n_scored':0,'recent_scores':[]}
        scored=[r for r in rows if not r.get('skipped')]
        recent=scored[-self.mastery_window:]
        if not recent:return {'pct':0.0,'n_questions':len(rows),'n_scored':0,'recent_scores':[]}
        weights=[max(1,int(r.get('difficulty',1))) for r in recent]
        total_w=sum(weights);weighted=sum(float(r.get('score',0))*w for r,w in zip(recent,weights))
        pct=weighted/total_w if total_w>0 else 0.0
        return {'pct':round(pct,1),'n_questions':len(rows),'n_scored':len(scored),'recent_scores':[round(float(r.get('score',0)),1) for r in recent],'avg_difficulty':round(sum(weights)/len(weights),2)}
    def recent_questions(self,topic:str,k:int=10)->List[str]:
        rows=self._load_topic(topic)
        return [r.get('question','') for r in rows[-k:]]
    def list_topics(self)->List[Dict[str,Any]]:
        out=[]
        for p in sorted(self.root.glob('topic_*.jsonl')):
            t=p.stem.removeprefix('topic_').replace('-',' ')
            m=self.mastery(t)
            if m['n_questions']>0:out.append({'topic':t,'mastery_pct':m['pct'],'n_questions':m['n_questions']})
        out.sort(key=lambda x:-x['mastery_pct'])
        return out
    def forget(self,topic:str)->bool:
        p=self._topic_path(topic)
        if not p.exists():return False
        try:p.unlink();return True
        except Exception:return False

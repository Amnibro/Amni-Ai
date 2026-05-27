"""pose_coach — physical-therapy form coaching from body-pose landmarks.
Stateless angle math + a stateful rep/form machine. Landmark indices follow MediaPipe Pose (BlazePose 33-pt) so the browser can POST landmarks straight from @mediapipe/pose.
Each landmark may be {x,y,z?,visibility?} or [x,y,v?] in normalized [0,1] image coords. No PII — body keypoints only; sessions persist to gitignored data/pose_sessions.jsonl (local, never federated)."""
import math,time,json,uuid,threading
from pathlib import Path
from typing import Dict,Any,List,Optional,Tuple
L={'nose':0,'l_shoulder':11,'r_shoulder':12,'l_elbow':13,'r_elbow':14,'l_wrist':15,'r_wrist':16,'l_hip':23,'r_hip':24,'l_knee':25,'r_knee':26,'l_ankle':27,'r_ankle':28}
EXERCISES={
 'pushup':{'label':'Push-up','primary':('shoulder','elbow','wrist'),'down_below':100.0,'up_above':150.0,'target_bottom':90.0,'min_vis':0.4,'form':[('back_straight',('shoulder','hip','knee'),155.0,'keep your back straight — hips sagging or piking')],'cue_down':'lower until your elbows reach ~90°','cue_up':'push all the way back up'},
 'situp':{'label':'Sit-up / Crunch','primary':('shoulder','hip','knee'),'down_below':75.0,'up_above':115.0,'target_bottom':55.0,'min_vis':0.4,'form':[],'cue_down':'curl up toward your knees','cue_up':'lower your back down with control'},
 'squat':{'label':'Squat','primary':('hip','knee','ankle'),'down_below':110.0,'up_above':160.0,'target_bottom':95.0,'min_vis':0.4,'form':[('chest_up',('shoulder','hip','knee'),65.0,'keep your chest up — avoid folding forward')],'cue_down':'sit down until your thighs are about parallel','cue_up':'drive up through your heels'},
 'bicep_curl':{'label':'Bicep curl','primary':('shoulder','elbow','wrist'),'down_below':65.0,'up_above':150.0,'target_bottom':45.0,'min_vis':0.4,'form':[('elbow_pinned',('hip','shoulder','elbow'),35.0,'keep your upper arm still — elbow is drifting forward')],'cue_down':'curl the weight up','cue_up':'lower under control to full extension'},
}
def _pt(lms,name):
    i=L.get(name)
    if i is None or lms is None or i>=len(lms):return None
    p=lms[i]
    try:
        if isinstance(p,dict):
            v=p.get('visibility',p.get('v',1.0));return (float(p['x']),float(p['y']),float(v if v is not None else 1.0))
        if isinstance(p,(list,tuple)):return (float(p[0]),float(p[1]),float(p[2]) if len(p)>2 else 1.0)
    except Exception:return None
    return None
def angle(a,b,c):
    if a is None or b is None or c is None:return None
    bax,bay=a[0]-b[0],a[1]-b[1];bcx,bcy=c[0]-b[0],c[1]-b[1]
    na=math.hypot(bax,bay);nc=math.hypot(bcx,bcy)
    if na<1e-9 or nc<1e-9:return None
    return math.degrees(math.acos(max(-1.0,min(1.0,(bax*bcx+bay*bcy)/(na*nc)))))
def _side_angle(lms,triplet,side,min_vis):
    pts=[_pt(lms,side+'_'+j) for j in triplet]
    if any(p is None for p in pts):return None
    if any(p[2]<min_vis for p in pts):return None
    return angle(pts[0],pts[1],pts[2])
def _joint_angle(lms,triplet,min_vis):
    la=_side_angle(lms,triplet,'l',min_vis);ra=_side_angle(lms,triplet,'r',min_vis)
    vals=[v for v in (la,ra) if v is not None]
    if not vals:return None,None
    return (sum(vals)/len(vals)),('both' if len(vals)==2 else ('left' if la is not None else 'right'))
def _form_issues(lms,spec):
    issues=[]
    for name,triplet,min_ok,msg in spec.get('form',[]):
        a,_=_joint_angle(lms,triplet,spec.get('min_vis',0.4))
        if a is not None and a<min_ok:issues.append({'name':name,'angle':round(a,1),'msg':msg})
    return issues
def analyze_frame(lms,exercise):
    spec=EXERCISES.get(exercise)
    if spec is None:return {'error':f'unknown exercise {exercise!r}; supported: {", ".join(EXERCISES)}'}
    a,side=_joint_angle(lms,spec['primary'],spec.get('min_vis',0.4))
    if a is None:return {'exercise':exercise,'angle':None,'side':None,'feedback':'I cannot see the joints clearly — step back so your whole body is in frame.','form_issues':[]}
    issues=_form_issues(lms,spec)
    tb=spec.get('target_bottom')
    depth=('at depth' if (tb is not None and a<=tb+8) else ('almost there' if (tb is not None and a<=tb+25) else 'extended'))
    fb=f'{spec["label"]}: primary angle {round(a,1)}° ({depth}).'
    if issues:fb+=' Form: '+'; '.join(i['msg'] for i in issues)
    return {'exercise':exercise,'label':spec['label'],'angle':round(a,1),'side':side,'target_bottom':tb,'depth':depth,'feedback':fb,'form_issues':issues}
class PoseCoach:
    def __init__(self,exercise,session_id=''):
        spec=EXERCISES.get(exercise)
        if spec is None:raise ValueError(f'unknown exercise {exercise!r}')
        self.exercise=exercise;self.spec=spec;self.session_id=session_id
        self.phase='up';self.reps=0;self.started_at=time.time();self.frames=0
        self.rep_min=None;self.rep_issues=set();self.rep_log=[];self.good_reps=0
        self.last_angle=None;self.peak_depth=None
    def feed(self,lms):
        self.frames+=1
        a,side=_joint_angle(lms,self.spec['primary'],self.spec.get('min_vis',0.4))
        if a is None:return {'exercise':self.exercise,'label':self.spec['label'],'angle':None,'side':None,'phase':self.phase,'reps':self.reps,'good_reps':self.good_reps,'feedback':'Lost tracking — get your whole body in frame.','form_issues':[],'rep_completed':False}
        self.last_angle=a
        frame_issues=_form_issues(lms,self.spec)
        for i in frame_issues:self.rep_issues.add(i['name'])
        spec=self.spec;tb=spec.get('target_bottom');rep_completed=False;rep_record=None
        if self.phase=='up' and a<spec['down_below']:self.phase='down';self.rep_min=a
        elif self.phase=='down':
            self.rep_min=a if self.rep_min is None else min(self.rep_min,a)
            self.peak_depth=self.rep_min if self.peak_depth is None else min(self.peak_depth,self.rep_min)
            if a>spec['up_above']:
                self.reps+=1;rep_completed=True
                depth_ok=(tb is None) or (self.rep_min<=tb+8)
                clean=depth_ok and not self.rep_issues
                if clean:self.good_reps+=1
                rep_record={'rep':self.reps,'min_angle':round(self.rep_min,1),'depth_ok':depth_ok,'form_issues':sorted(self.rep_issues),'clean':clean}
                self.rep_log.append(rep_record)
                self.phase='up';self.rep_min=None;self.rep_issues=set()
        if rep_completed:
            if rep_record['clean']:fb=f'Rep {self.reps} ✓ clean — depth {rep_record["min_angle"]}°. {spec["cue_down"].capitalize()} on the next one.'
            else:
                probs=[]
                if not rep_record['depth_ok']:probs.append(f'go deeper (reached {rep_record["min_angle"]}°, target ≤{tb:.0f}°)')
                for n,t,mo,msg in spec.get('form',[]):
                    if n in rep_record['form_issues']:probs.append(msg)
                fb=f'Rep {self.reps}: '+'; '.join(probs) if probs else f'Rep {self.reps} counted.'
        elif self.phase=='down':
            live=[]
            if tb is not None and a>tb+8:live.append(spec['cue_down'])
            for i in frame_issues:live.append(i['msg'])
            fb=f'Down — {round(a,1)}°. '+('; '.join(live) if live else 'good depth, now '+spec['cue_up'])
        else:fb=f'Ready — {round(a,1)}°. {spec["cue_down"].capitalize()}.'
        return {'exercise':self.exercise,'label':spec['label'],'angle':round(a,1),'side':side,'phase':self.phase,'reps':self.reps,'good_reps':self.good_reps,'target_bottom':tb,'rep_min_angle':(round(self.rep_min,1) if self.rep_min is not None else None),'feedback':fb,'form_issues':frame_issues,'rep_completed':rep_completed,'rep':rep_record}
    def summary(self):
        dur=round(time.time()-self.started_at,1)
        clean_rate=round(100.0*self.good_reps/self.reps,1) if self.reps else 0.0
        common=[]
        for n,t,mo,msg in self.spec.get('form',[]):
            c=sum(1 for r in self.rep_log if n in r.get('form_issues',[]))
            if c:common.append({'issue':n,'msg':msg,'count':c})
        return {'exercise':self.exercise,'label':self.spec['label'],'session_id':self.session_id,'reps':self.reps,'good_reps':self.good_reps,'clean_rate_pct':clean_rate,'duration_s':dur,'frames':self.frames,'peak_depth':(round(self.peak_depth,1) if self.peak_depth is not None else None),'common_issues':sorted(common,key=lambda x:-x['count']),'reps_log':self.rep_log}
_SESSIONS:Dict[str,PoseCoach]={}
_LOCK=threading.Lock()
def _log_path()->Path:
    p=Path(__file__).resolve().parents[2]/'data'/'pose_sessions.jsonl';p.parent.mkdir(parents=True,exist_ok=True);return p
def start_session(exercise,session_id=''):
    sid=session_id or ('pc_'+uuid.uuid4().hex[:10])
    try:coach=PoseCoach(exercise,session_id=sid)
    except ValueError as e:return {'error':str(e)}
    with _LOCK:_SESSIONS[sid]=coach
    return {'started':True,'session_id':sid,'exercise':exercise,'label':coach.spec['label'],'cue':coach.spec['cue_down']}
def feed_session(session_id,lms,exercise=None):
    with _LOCK:coach=_SESSIONS.get(session_id)
    if coach is None:
        if not exercise:return {'error':'no active session; pass exercise to auto-start'}
        r=start_session(exercise,session_id=session_id)
        if 'error' in r:return r
        with _LOCK:coach=_SESSIONS.get(session_id)
    if exercise and coach.exercise!=exercise:
        r=start_session(exercise,session_id=session_id)
        with _LOCK:coach=_SESSIONS.get(session_id)
    return coach.feed(lms)
def stop_session(session_id,persist=True):
    with _LOCK:coach=_SESSIONS.pop(session_id,None)
    if coach is None:return {'error':f'no active session {session_id!r}'}
    summ=coach.summary()
    if persist and coach.reps>0:
        try:
            rec=dict(summ);rec['ts']=time.time();rec['iso']=time.strftime('%Y-%m-%dT%H:%M:%S',time.localtime(rec['ts']))
            with _log_path().open('a',encoding='utf-8') as fh:fh.write(json.dumps(rec,default=str)+'\n')
        except Exception:pass
    return summ
def session_history(limit=20,exercise=None):
    p=_log_path()
    if not p.exists():return []
    out=[]
    try:
        for ln in p.read_text(encoding='utf-8').splitlines():
            ln=ln.strip()
            if not ln:continue
            try:r=json.loads(ln)
            except Exception:continue
            if exercise and r.get('exercise')!=exercise:continue
            out.append(r)
    except Exception:return []
    out.sort(key=lambda r:-float(r.get('ts') or 0));return out[:int(max(1,limit))]
def list_exercises():
    return [{'key':k,'label':v['label'],'tracks':'-'.join(v['primary']),'target_bottom':v.get('target_bottom'),'cue':v['cue_down']} for k,v in EXERCISES.items()]

"""coach skill — Socratic tutoring loop. Adam generates questions, grades answers, escalates difficulty, summarizes mastery.
Actions (passed in args['action']):
  start <topic>           → fresh session, returns first question
  ask                     → next question in current session (different difficulty if streak changed)
  answer <user_answer>    → grade, update mastery, return next question + feedback
  hint                    → reveal a hint without giving answer away
  skip                    → mark skip, next question (no mastery hit)
  summary                 → end session, return mastery snapshot
  status                  → current difficulty + asked + streaks
Adam-side calls use chat_persona with do_sample=False for deterministic JSON output."""
import json,re,time
from typing import Dict,Any,Optional,Tuple
from amni.storage.coach_atlas import CoachAtlas
_JSON_OBJ_RE=re.compile(r'\{(?:[^{}]|\{[^{}]*\})*\}',re.DOTALL)
_DIFFICULTY_LABEL={1:'introductory (one-fact recall)',2:'basic (definition + simple application)',3:'intermediate (multi-step reasoning)',4:'advanced (synthesis + edge cases)',5:'expert (proof, rigor, deep links)'}
_GEN_Q_PROMPT=(
    'Generate ONE single test question about the topic "{TOPIC}" at difficulty {DLABEL}.\n'
    'AVOID these recently-asked questions (do not repeat or paraphrase): {RECENT}\n\n'
    'Output ONLY this JSON object, no prose:\n'
    '{"question":"<the question, one line, ends with ?>", "model_answer":"<a concise correct answer, 1-3 sentences>", "hint":"<a non-revealing nudge, one short sentence>"}'
)
_GRADE_PROMPT=(
    'You are grading a student\'s answer to this question:\n'
    'QUESTION: {Q}\n'
    'MODEL CORRECT ANSWER: {MA}\n'
    'STUDENT ANSWER: {UA}\n\n'
    'Grade rules:\n'
    '- 90-100: complete + correct + precise.\n'
    '- 70-89:  correct main idea, minor omissions.\n'
    '- 50-69:  partially correct or vague.\n'
    '- 30-49:  some related content but mostly wrong.\n'
    '- 0-29:   wrong, blank, or irrelevant.\n\n'
    'Output ONLY this JSON object:\n'
    '{"score":0-100, "feedback":"<one or two sentences for the student>", "correct_facts":["..."], "missing_facts":["..."]}'
)
def _extract_json(text:str)->Optional[Dict[str,Any]]:
    if not text:return None
    m=_JSON_OBJ_RE.search(text)
    raw=m.group(0) if m else text.strip()
    try:return json.loads(raw)
    except Exception:pass
    try:return json.loads(raw.replace("'",'"'))
    except Exception:return None
def _call_adam_json(adam,prompt:str,max_new_tokens:int=400)->Optional[Dict[str,Any]]:
    if adam is None or not hasattr(adam,'chat_persona'):return None
    try:r=adam.chat_persona(prompt,system='You are a strict JSON-only generator. Output a single JSON object. No prose, no markdown fences.',max_new_tokens=max_new_tokens,do_sample=False)
    except Exception:return None
    ans=(r or {}).get('answer','') if isinstance(r,dict) else ''
    return _extract_json(ans)
_GEN_MA_PROMPT=(
    'You are reviewing a flashcard. Provide ONLY the correct answer to the question, in 1-3 concise sentences.\n'
    'QUESTION: {Q}\n\n'
    'Output ONLY this JSON object:\n'
    '{"model_answer":"<concise correct answer>", "hint":"<a non-revealing nudge for the student>"}'
)
def _gen_model_answer(adam,question:str)->Dict[str,str]:
    """Cheap LLM call: given a question (e.g. a v6.10.47 review-queue card), produce a model answer + hint so the grader has ground truth."""
    if not question.strip():return {'model_answer':'','hint':''}
    obj=_call_adam_json(adam,_GEN_MA_PROMPT.replace('{Q}',question),max_new_tokens=250)
    if not obj:return {'model_answer':'','hint':''}
    return {'model_answer':str(obj.get('model_answer','')).strip(),'hint':str(obj.get('hint','')).strip()}
def _gen_question(adam,atlas:CoachAtlas,topic:str,difficulty:int)->Optional[Dict[str,Any]]:
    recent=atlas.recent_questions(topic,k=6)
    recent_str=json.dumps(recent[-6:]) if recent else '[]'
    prompt=_GEN_Q_PROMPT.replace('{TOPIC}',topic).replace('{DLABEL}',_DIFFICULTY_LABEL.get(difficulty,'basic')).replace('{RECENT}',recent_str)
    obj=_call_adam_json(adam,prompt,max_new_tokens=350)
    if not obj or not obj.get('question'):return None
    return {'question':str(obj['question']).strip(),'model_answer':str(obj.get('model_answer','')).strip(),'hint':str(obj.get('hint','')).strip()}
def _grade_answer(adam,question:str,model_answer:str,user_answer:str)->Dict[str,Any]:
    if not (user_answer or '').strip():return {'score':0,'feedback':'(empty answer)','correct_facts':[],'missing_facts':[]}
    prompt=_GRADE_PROMPT.replace('{Q}',question).replace('{MA}',model_answer).replace('{UA}',user_answer[:600])
    obj=_call_adam_json(adam,prompt,max_new_tokens=350)
    if not obj:return {'score':50,'feedback':'(grader output not parseable; defaulted to 50)','correct_facts':[],'missing_facts':[]}
    score=int(obj.get('score',50));score=max(0,min(100,score))
    return {'score':score,'feedback':str(obj.get('feedback','')).strip(),'correct_facts':obj.get('correct_facts',[]) or [],'missing_facts':obj.get('missing_facts',[]) or []}
def _widget_envelope(session,mastery,question,feedback=None):
    data={'topic':session.get('topic'),'difficulty':session.get('difficulty'),'difficulty_label':_DIFFICULTY_LABEL.get(session.get('difficulty',2),'basic'),'streak_correct':session.get('streak_correct',0),'streak_wrong':session.get('streak_wrong',0),'n_answered':session.get('n_answered',0),'mastery_pct':(mastery or {}).get('pct',0.0),'mastery_recent':(mastery or {}).get('recent_scores',[]),'question':question}
    if feedback:data['feedback']=feedback
    return {'type':'info','title':f"Coach · {session.get('topic','?')}",'icon':'🎓','data':data}
def coach_skill(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    atlas=ctx.get('coach_atlas') if ctx else None
    adam=ctx.get('adam') if ctx else None
    if atlas is None:return {'error':'CoachAtlas not initialized in agent context'}
    if adam is None:return {'error':'Adam not in skill context'}
    action=(args.get('action') or '').strip().lower()
    topic=(args.get('topic') or '').strip()
    sid=(args.get('session_id') or args.get('sid') or '').strip() or None
    user_answer=(args.get('answer') or args.get('user_answer') or '').strip()
    if action in ('start','begin','new'):
        if not topic:return {'error':'need topic to start: coach start <topic>'}
        sid=atlas.start_session(topic,session_id=sid,initial_difficulty=int(args.get('difficulty',2)))
        s=atlas.get_session(sid)
        seed_q=(args.get('seed_question') or '').strip()
        if seed_q:
            seed_ma=(args.get('seed_model_answer') or '').strip()
            seed_hint=(args.get('seed_hint') or '').strip()
            if not seed_ma:gen=_gen_model_answer(adam,seed_q);seed_ma=gen['model_answer'];seed_hint=seed_hint or gen['hint']
            q={'question':seed_q,'model_answer':seed_ma,'hint':seed_hint}
        else:
            q=_gen_question(adam,atlas,topic,s['difficulty'])
            if not q:return {'error':'question generation failed (adam returned no valid JSON)','session_id':sid}
        atlas.update_session(sid,pending_question=q['question'],pending_model_answer=q['model_answer'],pending_hint=q['hint'])
        m=atlas.mastery(topic)
        return {'session_id':sid,'topic':topic,'difficulty':s['difficulty'],'question':q['question'],'mastery':m,'is_review':bool(seed_q),'widget':_widget_envelope(atlas.get_session(sid),m,q['question'])}
    if not sid:return {'error':'no session_id; start one with action=start, topic=<topic>'}
    s=atlas.get_session(sid)
    if s is None:return {'error':f'unknown session_id {sid}; sessions are in-memory; start a new one'}
    topic=s['topic']
    if action in ('ask','next'):
        q=_gen_question(adam,atlas,topic,s['difficulty'])
        if not q:return {'error':'question generation failed','session_id':sid}
        atlas.update_session(sid,pending_question=q['question'],pending_model_answer=q['model_answer'],pending_hint=q['hint'])
        m=atlas.mastery(topic)
        return {'session_id':sid,'topic':topic,'difficulty':s['difficulty'],'question':q['question'],'mastery':m,'widget':_widget_envelope(atlas.get_session(sid),m,q['question'])}
    if action=='answer':
        if not s.get('pending_question'):return {'error':'no pending question; ask one first'}
        q=s['pending_question'];ma=s.get('pending_model_answer','');grade=_grade_answer(adam,q,ma,user_answer)
        atlas.record(sid,topic,q,user_answer,grade['score'],s['difficulty'],hint_used=False,skipped=False)
        s_post=atlas.get_session(sid);m=atlas.mastery(topic)
        next_q=_gen_question(adam,atlas,topic,s_post['difficulty'])
        if next_q:atlas.update_session(sid,pending_question=next_q['question'],pending_model_answer=next_q['model_answer'],pending_hint=next_q['hint'])
        else:atlas.update_session(sid,pending_question=None,pending_model_answer=None,pending_hint=None)
        return {'session_id':sid,'topic':topic,'score':grade['score'],'feedback':grade['feedback'],'correct_facts':grade['correct_facts'],'missing_facts':grade['missing_facts'],'difficulty':s_post['difficulty'],'mastery':m,'next_question':next_q['question'] if next_q else None,'widget':_widget_envelope(s_post,m,next_q['question'] if next_q else '(end)',feedback=grade['feedback'])}
    if action=='hint':
        hint=s.get('pending_hint') or '(no hint available)'
        s['n_hinted']=s.get('n_hinted',0)+1
        return {'session_id':sid,'topic':topic,'hint':hint,'difficulty':s['difficulty']}
    if action=='skip':
        q=s.get('pending_question','');atlas.record(sid,topic,q,'',0,s['difficulty'],hint_used=False,skipped=True)
        next_q=_gen_question(adam,atlas,topic,s['difficulty'])
        if next_q:atlas.update_session(sid,pending_question=next_q['question'],pending_model_answer=next_q['model_answer'],pending_hint=next_q['hint'])
        m=atlas.mastery(topic);s_post=atlas.get_session(sid)
        return {'session_id':sid,'topic':topic,'skipped':True,'next_question':next_q['question'] if next_q else None,'mastery':m,'widget':_widget_envelope(s_post,m,next_q['question'] if next_q else '(end)')}
    if action in ('summary','end','stop','done'):
        summary=atlas.end_session(sid)
        return {'session_id':sid,**summary,'widget':{'type':'info','title':f"Coach · {summary.get('topic','?')} · session complete",'icon':'🏁','data':summary}}
    if action=='status':
        m=atlas.mastery(topic)
        return {'session_id':sid,'topic':topic,'difficulty':s['difficulty'],'pending_question':s.get('pending_question'),'streak_correct':s.get('streak_correct'),'streak_wrong':s.get('streak_wrong'),'n_answered':s.get('n_answered'),'n_skipped':s.get('n_skipped'),'n_hinted':s.get('n_hinted'),'mastery':m}
    return {'error':f'unknown action "{action}"; valid: start|ask|answer|hint|skip|summary|status'}

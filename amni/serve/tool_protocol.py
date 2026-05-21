"""Adam tool-call grammar — bridges OpenAI tool-call protocol to a 2B GF(17) model.
Strict fenced-JSON grammar (```tool_call ... ```), multi-call tolerant, malformed-JSON resilient.
build_system_prompt → strings the grammar into a system message.
parse_tool_calls    → extracts {name, arguments_str, id} from Adam's text.
flatten_history     → turns OpenAI messages[] (with tool_calls/tool roles) into a (history_pairs, last_user, system) tuple Adam understands.
tools_digest        → stable 8-byte hash of the tool list for cache keying.
"""
import re,json,hashlib,uuid,time
from typing import List,Dict,Any,Tuple,Optional
_FENCE_RE=re.compile(r"```tool_call\s*\n(.*?)\n```",re.DOTALL|re.IGNORECASE)
_LOOSE_FENCE_RE=re.compile(r"```(?:json|tool_call)?\s*\n?(\{[^`]*?\"name\"\s*:[^`]*?\})\s*\n?```",re.DOTALL|re.IGNORECASE)
_INLINE_JSON_RE=re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\"name\"\s*:\s*\"([a-zA-Z_][\w]*)\"(?:[^{}]|\{[^{}]*\})*\}",re.DOTALL)
def _safe_json(s:str)->Optional[Dict[str,Any]]:
    try:return json.loads(s)
    except Exception:pass
    s2=s.strip().rstrip(',').replace('\n',' ').replace('  ',' ')
    try:return json.loads(s2)
    except Exception:pass
    try:return json.loads(s2.replace("'",'"'))
    except Exception:return None
def parse_tool_calls(text:str)->List[Dict[str,Any]]:
    out=[];seen_spans=[]
    if not text:return out
    for m in _FENCE_RE.finditer(text):
        body=m.group(1).strip();obj=_safe_json(body)
        if obj and isinstance(obj,dict) and obj.get('name'):
            seen_spans.append((m.start(),m.end()))
            args=obj.get('arguments') if 'arguments' in obj else obj.get('args',{})
            if isinstance(args,str):args=_safe_json(args) or {}
            out.append({'id':f'call_{uuid.uuid4().hex[:16]}','name':obj['name'],'arguments':args or {}})
    for m in _LOOSE_FENCE_RE.finditer(text):
        if any(s<=m.start()<e for s,e in seen_spans):continue
        body=m.group(1).strip();obj=_safe_json(body)
        if obj and isinstance(obj,dict) and obj.get('name'):
            seen_spans.append((m.start(),m.end()))
            args=obj.get('arguments') if 'arguments' in obj else obj.get('args',{})
            if isinstance(args,str):args=_safe_json(args) or {}
            out.append({'id':f'call_{uuid.uuid4().hex[:16]}','name':obj['name'],'arguments':args or {}})
    if not out:
        for m in _INLINE_JSON_RE.finditer(text):
            if any(s<=m.start()<e for s,e in seen_spans):continue
            obj=_safe_json(m.group(0))
            if obj and isinstance(obj,dict) and obj.get('name'):
                args=obj.get('arguments') if 'arguments' in obj else obj.get('args',{})
                if isinstance(args,str):args=_safe_json(args) or {}
                out.append({'id':f'call_{uuid.uuid4().hex[:16]}','name':obj['name'],'arguments':args or {}})
    return out
def strip_tool_calls(text:str)->str:
    if not text:return text
    cleaned=_FENCE_RE.sub('',text)
    cleaned=_LOOSE_FENCE_RE.sub('',cleaned)
    return cleaned.strip()
def build_system_prompt(tools:List[Dict[str,Any]],cwd:Optional[str]=None,custom:Optional[str]=None,base_persona:Optional[str]=None)->str:
    if not tools:return base_persona or ''
    lines=[]
    if base_persona:lines.append(base_persona.strip());lines.append('')
    lines.append('You are an autonomous coding agent.'+(f' Working directory: {cwd}' if cwd else ''))
    lines.append('You have these tools available. Use them by emitting a fenced block exactly like this:')
    lines.append('```tool_call')
    lines.append('{"name":"<tool_name>","arguments":{"<arg>":"<value>"}}')
    lines.append('```')
    lines.append('Rules:')
    lines.append('1. Chain actions — never stop after one tool. Explore first, then act, then verify.')
    lines.append('2. Read files before editing them.')
    lines.append('3. Multiple tool calls per turn are OK — emit multiple fenced blocks.')
    lines.append('4. When the task is fully done, give a plain text final answer with NO tool_call fences.')
    lines.append('5. Tool outputs come back as OBSERVATION blocks — read them and decide next action.')
    lines.append('')
    lines.append('Tools:')
    for t in tools:
        f=t.get('function',t) if isinstance(t,dict) else {}
        name=f.get('name','?');desc=f.get('description','')
        params=f.get('parameters',{})
        props=params.get('properties',{}) if isinstance(params,dict) else {}
        req=params.get('required',[]) if isinstance(params,dict) else []
        arg_lines=[f"      {k}: {(v.get('type') or 'any') if isinstance(v,dict) else 'any'}"+(" (required)" if k in req else "")+((f' — {v.get("description")}') if isinstance(v,dict) and v.get('description') else '') for k,v in props.items()]
        lines.append(f'- {name}: {desc}')
        if arg_lines:lines.append('    arguments:');lines.extend(arg_lines)
    if custom:lines.append('');lines.append(f'Project notes:\n{custom.strip()}')
    return '\n'.join(lines)
def _format_observation(name:str,tool_call_id:str,content:str,max_chars:int=4000)->str:
    body=content if len(content)<=max_chars else content[:max_chars]+f'\n…[truncated {len(content)-max_chars} chars]'
    return f'OBSERVATION (tool_call_id={tool_call_id} tool={name}):\n{body}'
def flatten_history(messages:List[Dict[str,Any]])->Tuple[List[Tuple[str,str]],str,str]:
    if not messages:return [],'',''
    system='';user_last='';pairs=[]
    pending_user=None;pending_assistant=None;pending_tool_acc=[]
    tc_name_by_id={}
    for m in messages:
        role=m.get('role');content=(m.get('content') or '')
        if not isinstance(content,str):
            try:content=' '.join(c.get('text','') for c in content if isinstance(c,dict) and c.get('type')=='text')
            except Exception:content=str(content)
        if role=='system':system=(system+'\n\n'+content).strip() if system else content
        elif role=='user':
            if pending_user is not None and pending_assistant is not None:
                acc='\n\n'.join(pending_tool_acc) if pending_tool_acc else ''
                merged=(pending_assistant+('\n\n'+acc if acc else '')).strip()
                pairs.append((pending_user,merged))
            pending_user=content;pending_assistant=None;pending_tool_acc=[]
        elif role=='assistant':
            tc=m.get('tool_calls') or []
            if tc:
                summaries=[]
                for c in tc:
                    fn=(c.get('function') or {});nm=fn.get('name','?');aid=c.get('id','')
                    args=fn.get('arguments','{}')
                    if isinstance(args,(dict,list)):args=json.dumps(args)
                    tc_name_by_id[aid]=nm
                    summaries.append(f'Adam called {nm} with arguments: {args}')
                add=(content.strip()+'\n' if content.strip() else '')+'\n'.join(summaries)
                pending_assistant=(pending_assistant+'\n'+add).strip() if pending_assistant else add
            else:pending_assistant=(pending_assistant+'\n'+content).strip() if pending_assistant else content
        elif role=='tool':
            tcid=m.get('tool_call_id','?');nm=tc_name_by_id.get(tcid,'tool')
            pending_tool_acc.append(_format_observation(nm,tcid,content))
    if pending_user is not None and pending_assistant is None and not pending_tool_acc:user_last=pending_user
    elif pending_user is not None and (pending_assistant is not None or pending_tool_acc):
        if pending_tool_acc and pending_assistant is None:
            obs='\n\n'.join(pending_tool_acc)
            user_last=f'{pending_user}\n\n[Prior tool outputs you must now incorporate before responding to the user]\n{obs}'
        else:
            acc='\n\n'.join(pending_tool_acc) if pending_tool_acc else ''
            merged=(pending_assistant or '')+('\n\n'+acc if acc else '')
            pairs.append((pending_user,merged.strip()))
            user_last=''
    return pairs,user_last,system
def tools_digest(tools:Optional[List[Dict[str,Any]]])->str:
    if not tools:return 'no_tools'
    h=hashlib.blake2b(digest_size=8)
    for t in tools:
        f=(t.get('function') if isinstance(t,dict) else {}) or {}
        h.update((f.get('name','') or '').encode('utf-8','ignore'));h.update(b'\x1f')
    return h.hexdigest()
def build_openai_tool_calls(parsed:List[Dict[str,Any]])->List[Dict[str,Any]]:
    out=[]
    for p in parsed:
        args=p.get('arguments') or {}
        args_str=args if isinstance(args,str) else json.dumps(args)
        out.append({'id':p['id'],'type':'function','function':{'name':p['name'],'arguments':args_str}})
    return out
def openai_finish_reason(tool_calls:List[Dict[str,Any]])->str:return 'tool_calls' if tool_calls else 'stop'
def now_unix()->int:return int(time.time())

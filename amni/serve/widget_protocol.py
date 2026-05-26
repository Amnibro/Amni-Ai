"""Widget protocol — Adam emits inline data cards as fenced ```widget JSON blocks.
Frontend (or any UI) parses them out and renders as styled cards instead of dumping JSON in chat text. Mirror of tool_protocol shape — same grammar, same resilience.
Widget envelope:
  {"type":"<weather|system|time|news|stock|code|file|...>", "data":{...}, "title":"optional", "icon":"optional"}
Parser is multi-widget tolerant, malformed-JSON resilient, and case-insensitive on the fence keyword."""
import re,json,uuid
from typing import List,Dict,Any,Optional
_WIDGET_FENCE_RE=re.compile(r"```widget\s*\n(.*?)\n```",re.DOTALL|re.IGNORECASE)
_LOOSE_WIDGET_RE=re.compile(r"```(?:json|widget)?\s*\n?(\{[^`]*?\"type\"\s*:[^`]*?\})\s*\n?```",re.DOTALL|re.IGNORECASE)
_SUPPORTED_TYPES={'weather','system','time','news','stock','code','file','image','map','table','chart','progress','calendar','task','error','info','warning','success','disk','git','watch','file_change','skill_error'}
def _safe_json(s:str)->Optional[Dict[str,Any]]:
    try:return json.loads(s)
    except Exception:pass
    s2=s.strip().rstrip(',').replace('\n',' ').replace('  ',' ')
    try:return json.loads(s2)
    except Exception:pass
    try:return json.loads(s2.replace("'",'"'))
    except Exception:return None
def parse_widgets(text:str)->List[Dict[str,Any]]:
    out=[];seen_spans=[]
    if not text:return out
    for m in _WIDGET_FENCE_RE.finditer(text):
        body=m.group(1).strip();obj=_safe_json(body)
        if obj and isinstance(obj,dict) and obj.get('type'):
            seen_spans.append((m.start(),m.end()))
            t=str(obj['type']).strip().lower()
            out.append({'id':f'w_{uuid.uuid4().hex[:12]}','type':t,'data':obj.get('data',{}),'title':obj.get('title',''),'icon':obj.get('icon','')})
    for m in _LOOSE_WIDGET_RE.finditer(text):
        if any(s<=m.start()<e for s,e in seen_spans):continue
        body=m.group(1).strip();obj=_safe_json(body)
        if obj and isinstance(obj,dict) and obj.get('type'):
            t=str(obj['type']).strip().lower()
            if t not in _SUPPORTED_TYPES:continue
            seen_spans.append((m.start(),m.end()))
            out.append({'id':f'w_{uuid.uuid4().hex[:12]}','type':t,'data':obj.get('data',{}),'title':obj.get('title',''),'icon':obj.get('icon','')})
    return out
def strip_widgets(text:str)->str:
    if not text:return text
    cleaned=_WIDGET_FENCE_RE.sub('',text)
    cleaned=_LOOSE_WIDGET_RE.sub('',cleaned)
    return cleaned.strip()
def render_widget_text(w:Dict[str,Any])->str:
    t=w.get('type','?');d=w.get('data',{}) or {};title=w.get('title','') or t.title()
    if t=='weather':
        loc=d.get('location','?');temp=d.get('temp_c','?');desc=d.get('description','');hi=d.get('high_c','?');lo=d.get('low_c','?')
        return f"[{title}] {loc}: {temp}°C, {desc}. Today H {hi} / L {lo}."
    if t=='system':
        cpu=d.get('cpu_pct','?');mem=d.get('mem_pct','?');gpu=d.get('gpu_name','no GPU');gpu_vram=d.get('gpu_vram_used_gb','')
        return f"[{title}] CPU {cpu}% · MEM {mem}% · GPU {gpu}{(f' · VRAM {gpu_vram} GB' if gpu_vram else '')}"
    if t=='time':return f"[{title}] {d.get('iso','?')} ({d.get('tz','local')})"
    if t=='news':items=d.get('items',[]);return f"[{title}] "+ ' | '.join(f"{i.get('title','?')} ({i.get('source','')})" for i in items[:5])
    if t=='code':return f"[{title}] {d.get('lang','?')} code:\n{d.get('code','')[:500]}"
    if t=='file':return f"[{title}] {d.get('path','?')} — {d.get('size','?')} bytes"
    if t=='file_change':
        op=d.get('op','edit');p=d.get('path','?');la=d.get('lines_added',0);lr=d.get('lines_removed',0);return f"[{title}] {op}: {p} (+{la}/-{lr})"
    if t=='error':return f"[{title}] ❌ {d.get('message','?')}"
    if t=='info':return f"[{title}] {d.get('message','?')}"
    return f"[{title}] {json.dumps(d,default=str)[:300]}"
def build_system_prompt_addendum(available_widgets:List[str])->str:
    if not available_widgets:return ''
    lines=['When showing live data (weather, system stats, time, news, stock, etc.), prefer emitting a widget instead of a plain text dump. Format:','```widget','{"type":"<type>","data":{...},"title":"optional"}','```','','Supported widget types:']
    for w in available_widgets:lines.append(f'  - {w}')
    lines.append('')
    lines.append('Emit widgets when the user asks for current/live/right-now data (weather, time, system load, news, stock prices, file sizes). Otherwise use normal prose.')
    return '\n'.join(lines)
def widgets_summary(widgets:List[Dict[str,Any]])->str:
    if not widgets:return ''
    return ' | '.join(f"{w['type']}:{w.get('title','') or w['type']}" for w in widgets)

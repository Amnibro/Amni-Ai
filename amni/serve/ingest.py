"""Content ingestion skills — pull text from URLs / PDFs / YouTube transcripts, chunk, and teach Adam.
Each ingest_* skill returns:
  {url|path, chunks_taught, lessons_total_after, source_kind, title?, error?, chunks_preview?}
build_curriculum chains web search → ingest top-N → start a coach session on the topic.
Trafilatura is optional but in `[crawl]` extras. PyPDF + yt-dlp are conditional imports — clear error if missing."""
import re,time,hashlib,urllib.request,urllib.parse
from typing import Dict,Any,List,Optional,Tuple
_CHUNK_MAX_CHARS=900
_CHUNK_OVERLAP=120
_MIN_CHUNK_CHARS=80
_MAX_CHUNKS_PER_URL=24
def _safe_fetch(url:str,timeout:float=8.0,max_bytes:int=2_000_000)->Tuple[Optional[str],Optional[str]]:
    try:
        from amni.serve.code_safety import safe_urlopen
        raw,ct=safe_urlopen(url,timeout=timeout,max_bytes=max_bytes,headers={'User-Agent':'Amni-Ai/6.9.10 (educational ingest)','Accept':'text/html,application/xhtml+xml,*/*;q=0.6'})
        enc='utf-8'
        for tok in (ct or '').split(';'):
            if 'charset=' in tok:enc=tok.split('charset=',1)[1].strip().lower()
        try:return raw.decode(enc,errors='ignore'),ct
        except Exception:return raw.decode('utf-8',errors='ignore'),ct
    except Exception as e:return None,f'fetch_error: {type(e).__name__}: {e}'
def _strip_html_fallback(html:str)->str:
    if not html:return ''
    s=re.sub(r'<script[^>]*?>.*?</script>','',html,flags=re.DOTALL|re.IGNORECASE)
    s=re.sub(r'<style[^>]*?>.*?</style>','',s,flags=re.DOTALL|re.IGNORECASE)
    s=re.sub(r'<!--.*?-->','',s,flags=re.DOTALL)
    s=re.sub(r'<[^>]+>',' ',s)
    s=re.sub(r'&nbsp;|&#160;',' ',s);s=re.sub(r'&amp;','&',s);s=re.sub(r'&lt;','<',s);s=re.sub(r'&gt;','>',s);s=re.sub(r'&quot;','"',s)
    s=re.sub(r'\s+\n','\n',s);s=re.sub(r'\n\s+','\n',s);s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s)
    return s.strip()
def _distill(raw:str,is_html:bool)->Tuple[str,Optional[str]]:
    if not raw:return '',None
    if is_html:
        try:
            import trafilatura as _t
            extracted=_t.extract(raw,include_comments=False,include_tables=False,no_fallback=False)
            if extracted and len(extracted)>200:
                title=None
                try:
                    meta=_t.extract_metadata(raw);title=getattr(meta,'title',None)
                except Exception:pass
                return extracted,title
        except Exception:pass
        return _strip_html_fallback(raw),None
    return raw,None
def _chunk(text:str,max_chars:int=_CHUNK_MAX_CHARS,overlap:int=_CHUNK_OVERLAP)->List[str]:
    if not text:return []
    paras=[p.strip() for p in re.split(r'\n{2,}',text) if p.strip()]
    chunks=[];buf=''
    for p in paras:
        if len(p)>=max_chars:
            if buf:chunks.append(buf.strip());buf=''
            for i in range(0,len(p),max_chars-overlap):chunks.append(p[i:i+max_chars].strip())
            continue
        cand=(buf+'\n\n'+p).strip() if buf else p
        if len(cand)>max_chars:
            if buf:chunks.append(buf.strip())
            buf=p
        else:buf=cand
    if buf:chunks.append(buf.strip())
    return [c for c in chunks if len(c)>=_MIN_CHUNK_CHARS]
def _dedupe(chunks:List[str],seen:Optional[set]=None)->List[str]:
    seen=seen if seen is not None else set();out=[]
    for c in chunks:
        h=hashlib.blake2b(c.encode('utf-8','ignore'),digest_size=8).hexdigest()
        if h in seen:continue
        seen.add(h);out.append(c)
    return out
def _teach_chunks(adam,question_prefix:str,chunks:List[str],source_label:str)->int:
    if adam is None or not hasattr(adam,'teach'):return 0
    n=0
    for i,c in enumerate(chunks):
        q=f'{question_prefix} (chunk {i+1}/{len(chunks)})'
        try:adam.teach(q,c);n+=1
        except Exception as _e:break
    return n
def _ingest_url(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    url=(args.get('url') or '').strip()
    if not url:return {'error':'need url'}
    if not (url.startswith('http://') or url.startswith('https://')):url='https://'+url
    adam=ctx.get('adam') if ctx else None
    raw,err=_safe_fetch(url,timeout=float(args.get('timeout',8.0)))
    if raw is None:return {'error':err or 'fetch failed','url':url}
    is_html=('html' in (err or '').lower()) or url.lower().endswith(('.html','.htm')) or '<html' in raw[:512].lower()
    text,title=_distill(raw,is_html)
    if not text or len(text)<_MIN_CHUNK_CHARS:return {'error':'no extractable text','url':url,'title':title}
    chunks=_chunk(text,max_chars=int(args.get('chunk_max',_CHUNK_MAX_CHARS)))
    chunks=_dedupe(chunks)
    if int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))>0:chunks=chunks[:int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))]
    label=title or url
    taught=_teach_chunks(adam,f'From {label}',chunks,label) if not args.get('dry_run') else 0
    lessons_after=len(getattr(getattr(adam,'sem_lut',None),'_raw',[]) or []) if adam else 0
    return {'url':url,'title':title,'source_kind':'url','chunks_extracted':len(chunks),'chunks_taught':taught,'lessons_total_after':lessons_after,'chunks_preview':[c[:160] for c in chunks[:3]]}
def _ingest_pdf(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    path=(args.get('path') or '').strip()
    if not path:return {'error':'need path to local PDF'}
    try:
        from pathlib import Path as _P
        p=_P(path)
        if hasattr(reg,'_in_allowed_roots') and not reg._in_allowed_roots(str(p)):return {'error':'path outside allowed roots'}
        if not p.exists():return {'error':f'pdf not found: {path}'}
        try:from pypdf import PdfReader
        except ImportError:
            try:from PyPDF2 import PdfReader
            except ImportError:return {'error':'pdf reader not installed. pip install pypdf'}
        pages=[]
        with open(p,'rb') as f:
            reader=PdfReader(f);n_pages=len(reader.pages)
            for i,page in enumerate(reader.pages):
                if int(args.get('max_pages',60))>0 and i>=int(args.get('max_pages',60)):break
                try:t=page.extract_text() or ''
                except Exception:t=''
                if t.strip():pages.append(t)
    except Exception as e:return {'error':f'pdf parse failed: {type(e).__name__}: {e}'}
    text='\n\n'.join(pages)
    if not text or len(text)<_MIN_CHUNK_CHARS:return {'error':'no extractable text in pdf','path':path}
    chunks=_dedupe(_chunk(text,max_chars=int(args.get('chunk_max',_CHUNK_MAX_CHARS))))
    if int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))>0:chunks=chunks[:int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))]
    label=p.name
    adam=ctx.get('adam') if ctx else None
    taught=_teach_chunks(adam,f'From PDF {label}',chunks,label) if not args.get('dry_run') else 0
    lessons_after=len(getattr(getattr(adam,'sem_lut',None),'_raw',[]) or []) if adam else 0
    return {'path':str(p),'source_kind':'pdf','pages_read':len(pages),'chunks_extracted':len(chunks),'chunks_taught':taught,'lessons_total_after':lessons_after,'chunks_preview':[c[:160] for c in chunks[:3]]}
def _ingest_youtube(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    url=(args.get('url') or args.get('video') or '').strip()
    if not url:return {'error':'need youtube url or video id'}
    if 'youtu' not in url:url=f'https://www.youtube.com/watch?v={url}'
    try:
        try:from youtube_transcript_api import YouTubeTranscriptApi as _yta
        except ImportError:return {'error':'youtube_transcript_api not installed. pip install youtube-transcript-api'}
        m=re.search(r'(?:v=|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})',url)
        if not m:return {'error':f'could not extract video id from {url}'}
        vid=m.group(1)
        try:transcript=_yta.get_transcript(vid)
        except Exception as e:return {'error':f'transcript fetch failed: {type(e).__name__}: {e}'}
        text='\n\n'.join(seg.get('text','').strip() for seg in transcript if seg.get('text'))
    except Exception as e:return {'error':f'youtube ingest failed: {type(e).__name__}: {e}'}
    if not text or len(text)<_MIN_CHUNK_CHARS:return {'error':'transcript too short','url':url}
    chunks=_dedupe(_chunk(text,max_chars=int(args.get('chunk_max',_CHUNK_MAX_CHARS))))
    if int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))>0:chunks=chunks[:int(args.get('max_chunks',_MAX_CHUNKS_PER_URL))]
    label=f'YouTube {vid}'
    adam=ctx.get('adam') if ctx else None
    taught=_teach_chunks(adam,f'From {label}',chunks,label) if not args.get('dry_run') else 0
    lessons_after=len(getattr(getattr(adam,'sem_lut',None),'_raw',[]) or []) if adam else 0
    return {'url':url,'video_id':vid,'source_kind':'youtube','chunks_extracted':len(chunks),'chunks_taught':taught,'lessons_total_after':lessons_after,'chunks_preview':[c[:160] for c in chunks[:3]]}
def _build_curriculum(args:Dict[str,Any],ctx:Dict[str,Any],reg)->Dict[str,Any]:
    topic=(args.get('topic') or '').strip()
    if not topic:return {'error':'need topic'}
    if reg is None:return {'error':'skill registry not available'}
    coach_atlas=ctx.get('coach_atlas') if ctx else None
    if coach_atlas is None:return {'error':'CoachAtlas not in context'}
    task_reg=ctx.get('task_registry') if ctx else None
    max_sources=int(args.get('max_sources',3))
    web_query=str(args.get('query') or f'{topic} introduction tutorial')
    tid=task_reg.register('build_curriculum',label=f'Curriculum: {topic}',total=max_sources+2) if task_reg else None
    def _prog(done,msg=''):
        if task_reg and tid:task_reg.update(tid,done=done,message=msg)
    def _cancelled():return bool(task_reg and tid and task_reg.cancel_requested(tid))
    sources=[];ingest_outcomes=[]
    try:
        _prog(0,'searching the web for sources')
        if reg.has('web'):
            try:
                r=reg.call('web',{'query':web_query},ctx=ctx)
                if r.ok and r.output:sources=(r.output.get('sources') or [])[:max_sources]
            except Exception as e:
                if task_reg and tid:task_reg.fail(tid,f'web search: {e}')
                return {'error':f'web search failed: {e}'}
        _prog(1,f'{len(sources)} sources found, ingesting')
        for i,src in enumerate(sources):
            if _cancelled():
                if task_reg and tid:task_reg.mark_cancelled(tid)
                return {'cancelled':True,'topic':topic,'partial_sources':i,'ingest_outcomes':ingest_outcomes}
            if not isinstance(src,str) or not src.startswith('http'):continue
            try:ingest_outcomes.append(_ingest_url({'url':src,'max_chunks':int(args.get('chunks_per_source',8))},ctx,reg))
            except Exception as e:ingest_outcomes.append({'url':src,'error':str(e)[:200]})
            _prog(1+i+1,f'ingested {i+1}/{len(sources)}')
        coach_start={}
        if not args.get('skip_coach'):
            _prog(1+len(sources),'starting coach session')
            from amni.serve.coach import coach_skill as _coach_skill
            coach_start=_coach_skill({'action':'start','topic':topic,'difficulty':int(args.get('difficulty',2))},ctx,reg)
        total_taught=sum(o.get('chunks_taught',0) for o in ingest_outcomes if isinstance(o,dict))
        result={'topic':topic,'query':web_query,'n_sources':len(sources),'sources':sources,'ingest_outcomes':ingest_outcomes,'chunks_taught_total':total_taught,'coach_session':coach_start,'next_step':f'POST /skills/coach action=ask session_id={coach_start.get("session_id","?")}'}
        if task_reg and tid:task_reg.complete(tid,outcome={'taught':total_taught,'sources':len(sources)})
        return result
    except Exception as e:
        if task_reg and tid:task_reg.fail(tid,str(e)[:200])
        raise
def register(reg):
    reg.register('ingest_url',_ingest_url,desc='Fetch a webpage, distill via trafilatura (fallback to HTML strip), chunk, and teach Adam. Args: {url, max_chunks?=24, chunk_max?=900, timeout?=8, dry_run?=false}',schema={'url':'str','max_chunks':'int?','chunk_max':'int?','timeout':'float?','dry_run':'bool?'})
    reg.register('ingest_pdf',_ingest_pdf,desc='Read a local PDF (pypdf required), extract text per page, chunk, teach Adam. Args: {path, max_pages?=60, max_chunks?=24, chunk_max?=900, dry_run?=false}',schema={'path':'str','max_pages':'int?','max_chunks':'int?','chunk_max':'int?','dry_run':'bool?'})
    reg.register('ingest_youtube',_ingest_youtube,desc='Fetch a YouTube video transcript (youtube-transcript-api required), chunk, teach Adam. Args: {url|video, max_chunks?=24, chunk_max?=900, dry_run?=false}',schema={'url':'str?','video':'str?','max_chunks':'int?','chunk_max':'int?','dry_run':'bool?'})
    reg.register('build_curriculum',_build_curriculum,desc='Topic → web search → ingest top N URLs → start a coach session on the topic. Args: {topic, query?, max_sources?=3, chunks_per_source?=8, difficulty?=2, skip_coach?=false}',schema={'topic':'str','query':'str?','max_sources':'int?','chunks_per_source':'int?','difficulty':'int?','skip_coach':'bool?'})

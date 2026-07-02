"""Widget data fetchers — pure data, no rendering. Each fn returns a dict ready for widget_protocol envelope.
Skills wrap these and emit fenced ```widget JSON blocks for the frontend to render."""
import time,os,socket,subprocess,json,re,math
from typing import Dict,Any,Optional,List
def _safe_get(url:str,timeout:float=4.0,headers:Optional[Dict[str,str]]=None)->Optional[Dict[str,Any]]:
    try:
        import urllib.request as _u
        req=_u.Request(url,headers=headers or {'User-Agent':'Amni-Ai/6.9.6'})
        with _u.urlopen(req,timeout=timeout) as r:
            ct=r.headers.get('content-type','')
            raw=r.read().decode('utf-8','ignore')
            if 'json' in ct:return json.loads(raw)
            return {'_raw':raw}
    except Exception as e:return {'_error':str(e)[:200]}
def _geocode(query:str)->Optional[Dict[str,Any]]:
    if not query:return None
    from urllib.parse import quote
    q=query.strip();toks=q.split();cands=[q]
    parts=[p.strip() for p in re.split(r'[,/]',q) if p.strip()]
    if parts and parts[0]!=q:cands.append(parts[0])
    if len(toks)>=2 and toks[-1].isalpha() and len(toks[-1])==2:cands.append(' '.join(toks[:-1]))
    if toks and toks[0] not in cands:cands.append(toks[0])
    seen=set()
    for c in cands:
        cl=c.lower().strip()
        if not cl or cl in seen:continue
        seen.add(cl)
        j=_safe_get(f'https://geocoding-api.open-meteo.com/v1/search?name={quote(c)}&count=1&language=en&format=json',timeout=4.0)
        if not j or j.get('_error'):continue
        results=j.get('results') or []
        if not results:continue
        r=results[0]
        return {'lat':r.get('latitude'),'lon':r.get('longitude'),'name':r.get('name'),'country':r.get('country'),'admin1':r.get('admin1'),'cc':r.get('country_code'),'tz':r.get('timezone')}
    return None
_US_ST={'alabama':'AL','alaska':'AK','arizona':'AZ','arkansas':'AR','california':'CA','colorado':'CO','connecticut':'CT','delaware':'DE','district of columbia':'DC','florida':'FL','georgia':'GA','hawaii':'HI','idaho':'ID','illinois':'IL','indiana':'IN','iowa':'IA','kansas':'KS','kentucky':'KY','louisiana':'LA','maine':'ME','maryland':'MD','massachusetts':'MA','michigan':'MI','minnesota':'MN','mississippi':'MS','missouri':'MO','montana':'MT','nebraska':'NE','nevada':'NV','new hampshire':'NH','new jersey':'NJ','new mexico':'NM','new york':'NY','north carolina':'NC','north dakota':'ND','ohio':'OH','oklahoma':'OK','oregon':'OR','pennsylvania':'PA','rhode island':'RI','south carolina':'SC','south dakota':'SD','tennessee':'TN','texas':'TX','utah':'UT','vermont':'VT','virginia':'VA','washington':'WA','west virginia':'WV','wisconsin':'WI','wyoming':'WY'}
def _us_st(n:Optional[str])->str:return _US_ST.get((n or '').strip().lower(),(n or '').strip())
_WEATHER_CODE_DESC={0:'clear sky',1:'mainly clear',2:'partly cloudy',3:'overcast',45:'foggy',48:'rime fog',51:'light drizzle',53:'moderate drizzle',55:'dense drizzle',61:'light rain',63:'moderate rain',65:'heavy rain',71:'light snow',73:'moderate snow',75:'heavy snow',77:'snow grains',80:'rain showers',81:'heavy showers',82:'violent showers',85:'snow showers',86:'heavy snow showers',95:'thunderstorm',96:'thunderstorm with hail',99:'severe thunderstorm with hail'}
def _ip_geo()->Optional[Dict[str,Any]]:
    j=_safe_get('http://ip-api.com/json/?fields=status,lat,lon,city,regionName,country,countryCode',timeout=4.0)
    if not j or j.get('_error') or j.get('status')!='success' or j.get('lat') is None:return None
    return {'lat':j.get('lat'),'lon':j.get('lon'),'city':j.get('city'),'region':j.get('regionName'),'cc':j.get('countryCode'),'name':', '.join([x for x in (j.get('city'),j.get('regionName')) if x]) or j.get('country') or 'your area','tz':'auto'}
def fetch_weather(location:str='',lat:Optional[float]=None,lon:Optional[float]=None)->Dict[str,Any]:
    cc='';loc_name=''
    if (lat is None or lon is None) and (not location or location=='__need_geolocation__'):
        g=_ip_geo()
        if g:lat=g['lat'];lon=g['lon'];cc=(g.get('cc') or '').upper();loc_name=f"{g.get('city')}, {_us_st(g.get('region'))}" if cc=='US' and g.get('city') and g.get('region') else (g.get('name') or '')
    if lat is None or lon is None:
        if not location or location=='__need_geolocation__':return {'_error':'could not determine your location — allow location access in the PERMISSIONS panel, or ask "weather in <city>"'}
        g=_geocode(location)
        if not g:return {'_error':f'location "{location}" not found'}
        lat=g['lat'];lon=g['lon'];cc=(g.get('cc') or '').upper();loc_name=f"{g.get('name','?')}, {_us_st(g.get('admin1'))}" if cc=='US' and g.get('admin1') else f"{g.get('name','?')}, {g.get('country','')}".strip(', ');tz=g.get('tz','auto')
    else:loc_name=loc_name or location or f'{lat:.2f},{lon:.2f}';tz='auto'
    url=f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone={tz}&forecast_days=1'
    j=_safe_get(url,timeout=5.0)
    if not j or j.get('_error'):return {'_error':j.get('_error','weather fetch failed') if j else 'weather fetch failed'}
    cur=j.get('current') or {};daily=j.get('daily') or {}
    code=int(cur.get('weather_code') or 0);desc=_WEATHER_CODE_DESC.get(code,f'code {code}')
    hi_list=daily.get('temperature_2m_max') or [];lo_list=daily.get('temperature_2m_min') or []
    units='imperial' if (cc in ('US','LR','MM') or (not cc and (os.environ.get('AMNI_UNITS') or 'imperial').lower()!='metric')) else 'metric'
    _t=cur.get('temperature_2m');_hi=hi_list[0] if hi_list else None;_lo=lo_list[0] if lo_list else None;_w=cur.get('wind_speed_10m')
    _f=lambda c:None if c is None else round(c*1.8+32,1)
    return {'location':loc_name,'temp_c':_t,'humidity_pct':cur.get('relative_humidity_2m'),'wind_kmh':_w,'description':desc,'weather_code':code,'high_c':_hi,'low_c':_lo,'temp_f':_f(_t),'high_f':_f(_hi),'low_f':_f(_lo),'wind_mph':None if _w is None else round(_w/1.609344,1),'units':units,'cc':cc,'tz':j.get('timezone'),'ts':time.time()}
def _wx_icon(dr,x:int,y:int,s:int,code:int)->None:
    sun=code in (0,1);cloudy=code in (2,3,45,48) or code>=51
    if sun:
        cx,cy,r=x+s//2,y+s//2,s//3
        for a in range(8):
            dx,dy=math.cos(a*math.pi/4),math.sin(a*math.pi/4)
            dr.line([(cx+dx*(r+9),cy+dy*(r+9)),(cx+dx*(r+22),cy+dy*(r+22))],fill=(255,205,60),width=6)
        dr.ellipse([cx-r,cy-r,cx+r,cy+r],fill=(255,205,60))
    if code==2:dr.ellipse([x+s*0.05,y+s*0.05,x+s*0.55,y+s*0.55],fill=(255,205,60))
    if cloudy:
        dr.ellipse([x,y+s*0.34,x+s*0.60,y+s*0.74],fill=(150,160,175))
        dr.ellipse([x+s*0.28,y+s*0.20,x+s*0.95,y+s*0.64],fill=(172,182,197))
        dr.rectangle([x+s*0.10,y+s*0.48,x+s*0.86,y+s*0.68],fill=(165,175,190))
    if 51<=code<=67 or 80<=code<=82:
        for i in range(4):dr.line([(x+s*0.20+i*s*0.18,y+s*0.76),(x+s*0.13+i*s*0.18,y+s*0.95)],fill=(90,170,255),width=5)
    if 71<=code<=77 or 85<=code<=86:
        for i in range(4):dr.ellipse([x+s*0.16+i*s*0.18,y+s*0.82,x+s*0.16+i*s*0.18+9,y+s*0.91],fill=(240,245,250))
    if code>=95:dr.polygon([(x+s*0.50,y+s*0.62),(x+s*0.32,y+s*0.90),(x+s*0.46,y+s*0.90),(x+s*0.36,y+s*1.08),(x+s*0.64,y+s*0.80),(x+s*0.50,y+s*0.80),(x+s*0.60,y+s*0.62)],fill=(255,205,60))
    if code in (45,48):
        for i in range(3):dr.line([(x+s*0.05,y+s*0.82+i*10),(x+s*0.90,y+s*0.82+i*10)],fill=(140,150,160),width=4)
def render_weather_card(d:Dict[str,Any])->Optional[str]:
    try:
        from PIL import Image,ImageDraw,ImageFont
        W,H=760,400;img=Image.new('RGB',(W,H),(13,17,23));dr=ImageDraw.Draw(img)
        for i in range(H):dr.line([(0,i),(W,i)],fill=(13+int(9*i/H),17+int(13*i/H),23+int(20*i/H)))
        try:f_big=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',104);f_h=ImageFont.truetype('C:/Windows/Fonts/seguisb.ttf',34);f_m=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',25);f_s=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',19)
        except Exception:f_big=f_h=f_m=f_s=ImageFont.load_default()
        imp=d.get('units')=='imperial';code=int(d.get('weather_code') or 0)
        t=d.get('temp_f') if imp else d.get('temp_c');hi=d.get('high_f') if imp else d.get('high_c');lo=d.get('low_f') if imp else d.get('low_c');wv=d.get('wind_mph') if imp else d.get('wind_kmh')
        rnd=lambda v:'?' if not isinstance(v,(int,float)) else str(int(round(float(v))))
        dr.text((40,32),str(d.get('location','—')),font=f_h,fill=(235,240,245))
        dr.text((42,82),time.strftime('%A · %b %d · %I:%M %p').replace(' 0',' '),font=f_s,fill=(140,150,160))
        dr.text((34,118),f"{rnd(t)}°{'F' if imp else 'C'}",font=f_big,fill=(0,255,157))
        alt=f"{rnd(d.get('temp_c') if imp else d.get('temp_f'))}°{'C' if imp else 'F'}"
        dr.text((44,252),f"{str(d.get('description','—')).capitalize()}  ·  {alt}",font=f_m,fill=(200,210,220))
        _wx_icon(dr,W-206,52,132,code)
        for x,(lab,val) in zip([40,222,408,588],[('HIGH',f"{rnd(hi)}°"),('LOW',f"{rnd(lo)}°"),('HUMIDITY',f"{d.get('humidity_pct','?')}%"),('WIND',f"{'?' if wv is None else wv} {'mph' if imp else 'km/h'}")]):
            dr.text((x,318),lab,font=f_s,fill=(115,125,135));dr.text((x,344),val,font=f_h,fill=(235,240,245))
        outd=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'json','charts');os.makedirs(outd,exist_ok=True)
        p=os.path.join(outd,f'wx_{int(time.time())}.png');img.save(p);return p
    except Exception:return None
def fetch_system_stats()->Dict[str,Any]:
    out={'ts':time.time(),'hostname':socket.gethostname(),'platform':os.name}
    try:
        import psutil as _ps
        out['cpu_pct']=_ps.cpu_percent(interval=0.2)
        out['cpu_count']=_ps.cpu_count(logical=True)
        vm=_ps.virtual_memory()
        out['mem_total_gb']=round(vm.total/(1024**3),2);out['mem_used_gb']=round(vm.used/(1024**3),2);out['mem_pct']=vm.percent
        du=_ps.disk_usage('/')
        out['disk_total_gb']=round(du.total/(1024**3),2);out['disk_used_gb']=round(du.used/(1024**3),2);out['disk_pct']=du.percent
        bt=_ps.boot_time()
        out['uptime_s']=int(time.time()-bt)
    except Exception as e:out['_psutil_error']=str(e)[:120]
    try:
        import torch as _t
        if _t.cuda.is_available():
            i=_t.cuda.current_device()
            out['gpu_name']=_t.cuda.get_device_name(i)
            out['gpu_vram_total_gb']=round(_t.cuda.get_device_properties(i).total_memory/(1024**3),2)
            try:out['gpu_vram_used_gb']=round(_t.cuda.memory_allocated(i)/(1024**3),2)
            except Exception:pass
        else:out['gpu_name']='CPU-only'
    except Exception:pass
    return out
def fetch_time_card(tz_name:Optional[str]=None)->Dict[str,Any]:
    import datetime as _dt
    try:
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                dt=_dt.datetime.now(ZoneInfo(tz_name))
            except Exception:dt=_dt.datetime.now().astimezone()
        else:dt=_dt.datetime.now().astimezone()
    except Exception:dt=_dt.datetime.now()
    return {'iso':dt.isoformat(timespec='seconds'),'tz':str(dt.tzinfo) if dt.tzinfo else 'local','weekday':dt.strftime('%A'),'date_human':dt.strftime('%b %d, %Y'),'time_human':dt.strftime('%I:%M %p'),'epoch':int(time.time())}
_DDG_NEWS_URL='https://duckduckgo.com/html/?q={q}&iar=news'
def fetch_news(query:str='',n:int=6,atlas=None)->Dict[str,Any]:
    q=(query or 'top news').strip()
    try:
        from amni.serve.pii_egress import scrub as _scrub
        q=_scrub(q,atlas=atlas,source='news') or 'top news'
    except Exception:pass
    j=_safe_get(_DDG_NEWS_URL.format(q=__import__('urllib.parse',fromlist=['quote']).quote(q)),timeout=6.0,headers={'User-Agent':'Mozilla/5.0 Amni-Ai/6.10.5'})
    if not j or j.get('_error'):return {'_error':j.get('_error','news fetch failed') if j else 'news fetch failed'}
    html=j.get('_raw','')
    items=[]
    for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',html,re.DOTALL):
        url=m.group(1)
        title=re.sub(r'<[^>]+>','',m.group(2)).strip()
        if not url.startswith('http') or len(title)<6:continue
        src_m=re.search(r'<span[^>]+class="[^"]*result__url[^"]*"[^>]*>(.*?)</span>',html[m.end():m.end()+800])
        source=re.sub(r'<[^>]+>','',(src_m.group(1) if src_m else '')).strip()
        items.append({'title':title[:200],'url':url,'source':source[:80]})
        if len(items)>=int(n):break
    if not items:return {'_error':'no news items parsed (DDG layout change?)','query':q}
    return {'query':q,'items':items,'ts':time.time(),'source':'duckduckgo'}
_YF_QUOTE_URL='https://query1.finance.yahoo.com/v7/finance/quote?symbols={syms}'
def fetch_stock(symbols:str)->Dict[str,Any]:
    syms=(symbols or '').strip().upper().replace(' ','')
    if not syms:return {'_error':'need symbols'}
    j=_safe_get(_YF_QUOTE_URL.format(syms=syms),timeout=6.0,headers={'User-Agent':'Mozilla/5.0 Amni-Ai/6.10.5','Accept':'application/json'})
    if not j or j.get('_error'):return {'_error':j.get('_error','stock fetch failed') if j else 'stock fetch failed','symbols':syms}
    quotes=((j.get('quoteResponse') or {}).get('result') or [])
    if not quotes:return {'_error':'no quotes returned','symbols':syms,'raw':str(j)[:200]}
    out=[]
    for q in quotes:
        out.append({'symbol':q.get('symbol'),'name':q.get('longName') or q.get('shortName') or q.get('symbol'),'price':q.get('regularMarketPrice'),'change':q.get('regularMarketChange'),'change_pct':q.get('regularMarketChangePercent'),'currency':q.get('currency','USD'),'market_state':q.get('marketState','?'),'day_high':q.get('regularMarketDayHigh'),'day_low':q.get('regularMarketDayLow'),'volume':q.get('regularMarketVolume')})
    return {'symbols':syms,'quotes':out,'ts':time.time(),'source':'yahoo_finance'}
def fetch_file_preview(path:str,max_lines:int=40,max_chars:int=2400)->Dict[str,Any]:
    from pathlib import Path as _P
    p=_P(path)
    if not p.exists():return {'_error':f'file not found: {path}'}
    if p.is_dir():return {'_error':f'path is a directory: {path}'}
    try:size=p.stat().st_size
    except Exception as e:return {'_error':f'stat failed: {e}'}
    try:text=p.read_text(encoding='utf-8',errors='replace')[:max_chars]
    except Exception as e:return {'_error':f'read failed: {e}'}
    lines=text.splitlines()[:int(max_lines)]
    return {'path':str(p),'size_bytes':size,'lines_shown':len(lines),'preview':'\n'.join(lines),'ts':time.time(),'ext':p.suffix[1:] if p.suffix else ''}
def fetch_disk()->Dict[str,Any]:
    out={'ts':time.time(),'partitions':[]}
    try:
        import psutil as _ps
        for part in _ps.disk_partitions(all=False):
            try:u=_ps.disk_usage(part.mountpoint)
            except Exception:continue
            out['partitions'].append({'mount':part.mountpoint,'fs':part.fstype,'total_gb':round(u.total/(1024**3),2),'used_gb':round(u.used/(1024**3),2),'free_gb':round(u.free/(1024**3),2),'used_pct':u.percent})
    except Exception as e:return {'_error':f'psutil: {e}'}
    out['partitions']=sorted(out['partitions'],key=lambda x:-x.get('used_pct',0))[:6]
    return out
def fetch_git_status(workdir:Optional[str]=None)->Dict[str,Any]:
    cwd=workdir or os.getcwd()
    try:
        from pathlib import Path as _P
        if not (_P(cwd)/'.git').exists():
            cur=_P(cwd).resolve()
            while cur!=cur.parent:
                if (cur/'.git').exists():cwd=str(cur);break
                cur=cur.parent
    except Exception:pass
    def _run(args):
        try:r=subprocess.run(['git']+args,cwd=cwd,capture_output=True,text=True,timeout=4)
        except Exception:return None
        return r.stdout.strip() if r.returncode==0 else None
    branch=_run(['rev-parse','--abbrev-ref','HEAD']);
    if branch is None:return {'_error':f'not a git repo (or git missing): {cwd}'}
    status=_run(['status','--porcelain']) or ''
    log=_run(['log','--oneline','-n','5']) or ''
    dirty=[ln for ln in status.splitlines() if ln.strip()]
    remote=_run(['remote','get-url','origin']) or ''
    ahead_behind=_run(['rev-list','--left-right','--count','HEAD...@{u}']) or ''
    ahead=behind=0
    try:
        a,b=ahead_behind.split('\t')
        ahead=int(a);behind=int(b)
    except Exception:pass
    return {'workdir':cwd,'branch':branch,'dirty_n':len(dirty),'dirty_sample':[ln[:60] for ln in dirty[:6]],'recent_commits':log.splitlines()[:5],'remote':remote[:120],'ahead':ahead,'behind':behind,'ts':time.time()}
def make_widget_envelope(widget_type:str,data:Dict[str,Any],title:str='',icon:str='')->Dict[str,Any]:
    return {'type':widget_type,'data':({k:v for k,v in data.items() if k!='widget'} if isinstance(data,dict) else data),'title':title or widget_type.title(),'icon':icon or ''}
def make_widget_fence(widget_type:str,data:Dict[str,Any],title:str='',icon:str='')->str:
    env=make_widget_envelope(widget_type,data,title,icon)
    return '```widget\n'+json.dumps(env,default=str)+'\n```'

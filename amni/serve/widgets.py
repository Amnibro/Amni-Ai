"""Widget data fetchers — pure data, no rendering. Each fn returns a dict ready for widget_protocol envelope.
Skills wrap these and emit fenced ```widget JSON blocks for the frontend to render."""
import time,os,socket,subprocess,json,re
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
    j=_safe_get(f'https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&language=en&format=json',timeout=4.0)
    if not j or j.get('_error'):return None
    results=j.get('results') or []
    if not results:return None
    r=results[0]
    return {'lat':r.get('latitude'),'lon':r.get('longitude'),'name':r.get('name'),'country':r.get('country'),'tz':r.get('timezone')}
_WEATHER_CODE_DESC={0:'clear sky',1:'mainly clear',2:'partly cloudy',3:'overcast',45:'foggy',48:'rime fog',51:'light drizzle',53:'moderate drizzle',55:'dense drizzle',61:'light rain',63:'moderate rain',65:'heavy rain',71:'light snow',73:'moderate snow',75:'heavy snow',77:'snow grains',80:'rain showers',81:'heavy showers',82:'violent showers',85:'snow showers',86:'heavy snow showers',95:'thunderstorm',96:'thunderstorm with hail',99:'severe thunderstorm with hail'}
def fetch_weather(location:str='',lat:Optional[float]=None,lon:Optional[float]=None)->Dict[str,Any]:
    if lat is None or lon is None:
        if not location:return {'_error':'need location or lat/lon'}
        g=_geocode(location)
        if not g:return {'_error':f'location "{location}" not found'}
        lat=g['lat'];lon=g['lon'];loc_name=f"{g.get('name','?')}, {g.get('country','')}".strip(', ');tz=g.get('tz','auto')
    else:loc_name=location or f'{lat:.2f},{lon:.2f}';tz='auto'
    url=f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&daily=weather_code,temperature_2m_max,temperature_2m_min&timezone={tz}&forecast_days=1'
    j=_safe_get(url,timeout=5.0)
    if not j or j.get('_error'):return {'_error':j.get('_error','weather fetch failed') if j else 'weather fetch failed'}
    cur=j.get('current') or {};daily=j.get('daily') or {}
    code=int(cur.get('weather_code') or 0);desc=_WEATHER_CODE_DESC.get(code,f'code {code}')
    hi_list=daily.get('temperature_2m_max') or [];lo_list=daily.get('temperature_2m_min') or []
    return {'location':loc_name,'temp_c':cur.get('temperature_2m'),'humidity_pct':cur.get('relative_humidity_2m'),'wind_kmh':cur.get('wind_speed_10m'),'description':desc,'weather_code':code,'high_c':hi_list[0] if hi_list else None,'low_c':lo_list[0] if lo_list else None,'tz':j.get('timezone'),'ts':time.time()}
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
def make_widget_envelope(widget_type:str,data:Dict[str,Any],title:str='',icon:str='')->Dict[str,Any]:
    return {'type':widget_type,'data':data,'title':title or widget_type.title(),'icon':icon or ''}
def make_widget_fence(widget_type:str,data:Dict[str,Any],title:str='',icon:str='')->str:
    env=make_widget_envelope(widget_type,data,title,icon)
    return '```widget\n'+json.dumps(env,default=str)+'\n```'

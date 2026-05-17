import sys; sys.path.insert(0,'.')
from amni.inference.web_crawler import WebCrawler
c=WebCrawler(rate_limit_sec=0.5,timeout=15,max_chars_per_page=2000)
print('=== 1. DDGS raw search (no allow_list filter) ===',flush=True)
from ddgs import DDGS
with DDGS() as ddg:
    rs=list(ddg.text('numpy array indexing',max_results=8))
    print(f'  raw count: {len(rs)}',flush=True)
    for r in rs[:5]:
        u=r.get('href') or r.get('url') or '?'
        d=c._domain(u)
        print(f'    allowed={c._is_allowed(u)}  domain={d}  url={u[:80]}',flush=True)
print()
print('=== 2. fetch wikipedia direct ===',flush=True)
import trafilatura
url='https://en.wikipedia.org/wiki/Python_(programming_language)'
print(f'  allowed? {c._is_allowed(url)}',flush=True)
print(f'  can_fetch (robots.txt)? {c._can_fetch(url)}',flush=True)
import time
t0=time.time()
html=trafilatura.fetch_url(url,no_ssl=False)
print(f'  trafilatura.fetch_url returned: type={type(html).__name__} len={len(html) if html else 0} time={time.time()-t0:.2f}s',flush=True)
if html:
    text=trafilatura.extract(html,include_comments=False,include_tables=False,no_fallback=False)
    print(f'  extracted: {len(text) if text else 0} chars',flush=True)
    if text:print(f'  first 250: {text[:250]!r}',flush=True)
print()
print('=== 3. fetch with requests directly (sanity check) ===',flush=True)
import requests
t0=time.time()
r=requests.get(url,timeout=10,headers={'User-Agent':'Mozilla/5.0 AmniAdam/1.0'})
print(f'  status {r.status_code}, {len(r.text)} chars, {time.time()-t0:.2f}s',flush=True)

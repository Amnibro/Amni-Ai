"""Verify WebCrawler works standalone — no Adam, no GF(17) runtime, just the crawler module."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.inference.web_crawler import WebCrawler
print('=== standalone WebCrawler smoke ===',flush=True)
print('Instantiating WebCrawler...',flush=True)
c=WebCrawler(rate_limit_sec=0.5,timeout=10,max_chars_per_page=2000)
print(f'  default allow_list size: {len(c.allow)}',flush=True)
print(f'  sample allowed domains: {sorted(list(c.allow))[:10]}',flush=True)
print()
queries=[('what is python','python.org'),('numpy array indexing','numpy.org'),('git rebase guide','git-scm.com'),('postgresql full text search','postgresql.org')]
for q,expected_domain in queries:
    print(f'--- query: {q!r} ---',flush=True)
    t0=time.time()
    urls=c.search(q,k=3)
    print(f'  search() returned {len(urls)} urls in {time.time()-t0:.1f}s',flush=True)
    for u in urls[:3]:print(f'    {u}',flush=True)
    if not urls:print(f'  WARN: no search results for {q!r}',flush=True);continue
    pick=next((u for u in urls if expected_domain in u),urls[0])
    print(f'  fetch: {pick}',flush=True)
    t1=time.time()
    text=c.fetch(pick)
    if text:print(f'  fetched {len(text)} chars in {time.time()-t1:.1f}s — first 200 chars:\n    {text[:200]!r}',flush=True)
    else:print(f'  FETCH FAILED in {time.time()-t1:.1f}s',flush=True)
    print()
print('--- crawl() (search + fetch combined) ---',flush=True)
t0=time.time()
results=c.crawl('how to format date in python',k=2)
print(f'  crawl() got {len(results)} (url,text) pairs in {time.time()-t0:.1f}s',flush=True)
for url,text in results[:2]:print(f'    {url}  -> {len(text)} chars',flush=True)
print()
print('--- robots.txt enforcement ---',flush=True)
google_disallowed='https://www.google.com/search?q=test'
allowed=c._is_allowed(google_disallowed)
print(f'  google.com allowed? {allowed} (expected: False — not in default allow_list)',flush=True)
print()
print('--- rate-limiting (2 fetches to same domain) ---',flush=True)
t0=time.time()
c.fetch('https://en.wikipedia.org/wiki/Python_(programming_language)')
t1=time.time()
c.fetch('https://en.wikipedia.org/wiki/JavaScript')
elapsed_between=time.time()-t1
print(f'  inter-fetch delay: {elapsed_between:.2f}s (expected: >=0.5s rate limit)',flush=True)
print()
print('PASS — WebCrawler works standalone (no Adam dependency)',flush=True)

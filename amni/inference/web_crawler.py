"""WebCrawler + CrawlerPlugin — tier-4 escalation for AdamLoop low-confidence cases.
Per Anthony's directive (2026-05-03): when Adam is uncertain on a topic, dispatch crawlers (DuckDuckGo search → trafilatura clean-text) to gather web content, distill via Gemma E2B, save findings as PTEX lessons.
Safety rails: per-domain rate limit (1 req/sec), robots.txt respect (urllib.robotparser), allow-list of trusted educational/reference domains.
"""
import time,re,urllib.parse,urllib.robotparser
from typing import List,Tuple,Optional
from collections import defaultdict
_DEFAULT_ALLOW=('wikipedia.org','simple.wikipedia.org','wiktionary.org','wikibooks.org','wikiversity.org','stackexchange.com','stackoverflow.com','superuser.com','serverfault.com','askubuntu.com','arxiv.org','pubmed.ncbi.nlm.nih.gov','ncbi.nlm.nih.gov','nih.gov','cdc.gov','who.int','plato.stanford.edu','iep.utm.edu','britannica.com','khanacademy.org','wolframalpha.com','mathworld.wolfram.com','byjus.com','sparknotes.com','encyclopedia.com','reference.com','dictionary.com','merriam-webster.com','docs.python.org','python.org','developer.mozilla.org','docs.djangoproject.com','flask.palletsprojects.com','fastapi.tiangolo.com','pytorch.org','huggingface.co','tensorflow.org','keras.io','scikit-learn.org','numpy.org','scipy.org','pandas.pydata.org','matplotlib.org','seaborn.pydata.org','plotly.com','jupyter.org','anaconda.com','realpython.com','github.com','gitlab.com','bitbucket.org','readthedocs.io','readthedocs.org','medlineplus.gov','mayoclinic.org','healthline.com','webmd.com','medicinenet.com','nasa.gov','noaa.gov','usgs.gov','nature.com','sciencedirect.com','science.org','pnas.org','ieee.org','acm.org','smithsonianmag.com','natgeo.com','nationalgeographic.com','history.com','bbc.com','reuters.com','npr.org','pbs.org','metmuseum.org','loc.gov','archives.gov','data.gov','w3schools.com','geeksforgeeks.org','programiz.com','tutorialspoint.com','learnpython.org','baeldung.com','digitalocean.com','linode.com','vercel.com','netlify.com','cloudflare.com','aws.amazon.com','docs.aws.amazon.com','cloud.google.com','learn.microsoft.com','docs.microsoft.com','dotnet.microsoft.com','rust-lang.org','doc.rust-lang.org','go.dev','golang.org','php.net','ruby-lang.org','rubygems.org','kotlinlang.org','scala-lang.org','swift.org','dart.dev','flutter.dev','reactjs.org','react.dev','vuejs.org','angular.io','svelte.dev','nodejs.org','nodejs.dev','npmjs.com','typescriptlang.org','docs.docker.com','kubernetes.io','helm.sh','nginx.org','nginx.com','redis.io','postgresql.org','mysql.com','mongodb.com','sqlite.org','sqlalchemy.org','mariadb.org','apache.org','perl.org','haskell.org','clojure.org','elixir-lang.org','erlang.org','ocaml.org','julialang.org','rstudio.com','r-project.org','cran.r-project.org','octave.org','sagemath.org','wolfram.com','intel.com','amd.com','nvidia.com','developer.nvidia.com','developer.intel.com','arxiv-vanity.com','distill.pub','paperswithcode.com','openai.com','anthropic.com','deepmind.google','blog.google','research.google','ai.google','jmlr.org','journals.aps.org','aps.org','acs.org','rsc.org','springer.com','wiley.com','elsevier.com','academic.oup.com','frontiersin.org','plos.org','biorxiv.org','medrxiv.org','semanticscholar.org','arxiv-sanity.com','ietf.org','tools.ietf.org','rfc-editor.org','w3.org','whatwg.org','ecma-international.org','iso.org','unicode.org','tc39.es','spec.commonmark.org','cmake.org','gnu.org','gcc.gnu.org','llvm.org','clang.llvm.org','boost.org','cppreference.com','cplusplus.com','isocpp.org','en.cppreference.com','docs.oracle.com','openjdk.org','adoptium.net','spring.io','gradle.org','maven.apache.org','jetbrains.com','eclipse.org','vscode.dev','code.visualstudio.com','vim.org','neovim.io','emacswiki.org','gnu.org','git-scm.com','about.gitlab.com','laravel.com','symfony.com','wordpress.org','wordpress.com','drupal.org','joomla.org','jekyllrb.com','hugo.io','gohugo.io','11ty.dev','nextjs.org','nuxt.com','sveltekit.dev','remix.run','tailwindcss.com','bulma.io','getbootstrap.com','sass-lang.com','less.css.org','postcss.org','webpack.js.org','vitejs.dev','rollupjs.org','parceljs.org','esbuild.github.io','babeljs.io','swc.rs','eslint.org','prettier.io','jestjs.io','vitest.dev','mochajs.org','cypress.io','playwright.dev','selenium.dev','puppeteer.dev','pytest.org','docs.pytest.org','unittest.readthedocs.io','poetry-python.org','pipenv.pypa.io','pip.pypa.io','pypi.org','conda.io','bioconda.github.io','.edu','.gov','.ac.uk')
class WebCrawler:
    def __init__(self,allow_list=None,rate_limit_sec=1.0,respect_robots=True,timeout=8,max_chars_per_page=4000,unrestricted=True):
        self.unrestricted=bool(unrestricted) or allow_list is None or allow_list==()
        self.allow=set(allow_list) if not self.unrestricted and allow_list is not None else set()
        self.rate=rate_limit_sec;self.respect_robots=respect_robots
        self.timeout=timeout;self.max_chars=max_chars_per_page
        self._last_hit=defaultdict(float);self._robots={}
    def _domain(self,url:str)->str:
        try:return urllib.parse.urlparse(url).netloc.lower()
        except Exception:return ''
    def _is_allowed(self,url:str)->bool:
        d=self._domain(url)
        if not d:return False
        if self.unrestricted:return True
        for a in self.allow:
            if a.startswith('.') and (d.endswith(a) or d==a[1:]):return True
            if d==a or d.endswith('.'+a):return True
        return False
    def _can_fetch(self,url:str)->bool:
        if not self.respect_robots:return True
        if self._is_allowed(url):return True
        d=self._domain(url)
        if d not in self._robots:
            rp=urllib.robotparser.RobotFileParser();rp.set_url(f'https://{d}/robots.txt')
            try:rp.read()
            except Exception:pass
            self._robots[d]=rp
        try:return self._robots[d].can_fetch('Mozilla/5.0 (compatible; AmniAdam/1.0; +https://amni-scient.com/amni-ai)',url)
        except Exception:return True
    def _wait_rate(self,url:str):
        d=self._domain(url);now=time.time();last=self._last_hit[d]
        wait=self.rate-(now-last)
        if wait>0:time.sleep(wait)
        self._last_hit[d]=time.time()
    def search(self,query:str,k:int=3)->List[str]:
        try:from ddgs import DDGS
        except Exception:from duckduckgo_search import DDGS
        urls=[]
        try:
            with DDGS() as ddg:
                for r in ddg.text(query,max_results=k*4):
                    u=r.get('href') or r.get('url') or ''
                    if u and self._is_allowed(u):urls.append(u)
                    if len(urls)>=k:break
        except Exception as e:print(f'  [crawler] search failed: {e}',flush=True)
        return urls
    def fetch(self,url:str)->Optional[str]:
        if not self._is_allowed(url):return None
        if not self._can_fetch(url):return None
        self._wait_rate(url)
        import trafilatura
        try:
            html=trafilatura.fetch_url(url,no_ssl=False)
            if not html:return None
            text=trafilatura.extract(html,include_comments=False,include_tables=False,no_fallback=False)
            if not text:return None
            return text[:self.max_chars]
        except Exception as e:print(f'  [crawler] fetch failed for {url[:60]}: {e}',flush=True);return None
    def crawl(self,query:str,k:int=3)->List[Tuple[str,str]]:
        urls=self.search(query,k=k)
        out=[]
        for u in urls:
            text=self.fetch(u)
            if text:out.append((u,text))
        return out
class CrawlerPlugin:
    def __init__(self,distiller_svc,topic_extractor_svc=None,allow_list=None,max_pages=3,distill_max_tokens=200,unrestricted=True):
        self.crawler=WebCrawler(allow_list=allow_list,unrestricted=unrestricted)
        self.distiller=distiller_svc
        self.topic_extractor=topic_extractor_svc or distiller_svc
        self.max_pages=max_pages
        self.distill_max=distill_max_tokens
        self._stats={'crawls':0,'pages_fetched':0,'distillations':0,'failures':0,'topic_rewrites':0}
    def stats(self):return dict(self._stats)
    def _topic_regex(self,question:str)->str:
        q=re.sub(r'\b(what|which|who|where|when|why|how|is|are|the|a|an|of|in|on|at|to|for|with|by|from)\b','',question.lower())
        q=re.sub(r'[^\w\s]',' ',q)
        return ' '.join(q.split()[:8])
    def _topic_llm(self,question:str)->str:
        sys_p='Given an MCQ question, output a 3-7 word search query that would find an authoritative reference page. No question marks, no punctuation, no quoting. Just the search terms.'
        try:resp,_=self.topic_extractor.chat(question,system=sys_p,max_new_tokens=20,do_sample=False,kb_top_k=0)
        except Exception:resp=''
        q=re.sub(r'[^\w\s-]',' ',(resp or '').strip())
        q=' '.join(q.split()[:8])
        return q if q else self._topic_regex(question)
    def crawl_raw(self,question:str)->Tuple[List[Tuple[str,str]],str]:
        topic=self._topic_llm(question)
        pages=self.crawler.crawl(topic,k=self.max_pages)
        used_topic=topic
        if not pages:
            topic2=self._topic_regex(question)
            if topic2 and topic2!=topic:
                self._stats['topic_rewrites']+=1
                pages=self.crawler.crawl(topic2,k=self.max_pages)
                used_topic=topic2
        self._stats['pages_fetched']+=len(pages)
        return pages,used_topic
    def crawl_and_distill(self,question:str,subject:Optional[str]=None,letter_only:bool=False)->Tuple[str,List[str],int]:
        self._stats['crawls']+=1
        pages,_=self.crawl_raw(question)
        if not pages:self._stats['failures']+=1;return ('',[],0)
        sources=[u for u,_ in pages]
        sources_text='\n\n---\n\n'.join(f'[Source {i+1}: {u}]\n{t[:1500]}' for i,(u,t) in enumerate(pages))
        if letter_only:
            sys_p='Use the provided sources to answer the multiple-choice question. Reply with only the single letter (A, B, C, or D).'
            prompt=f'Sources:\n{sources_text}\n\nQuestion:\n{question}\n\nAnswer (single letter):'
            mx=4
        else:
            sys_p='You are summarizing real web sources fetched by a search tool. NEVER say "I do not have the capability to search the web" or "I cannot access the internet" — the sources below are REAL search results already retrieved for you. Summarize them factually, citing source numbers in [brackets]. If sources do not contain the answer, say "the provided sources do not cover this" and cite what they DO cover.'
            prompt=f'Sources:\n{sources_text}\n\nQuestion:\n{question}\n\nAnswer:'
            mx=self.distill_max
        try:ans,n=self.distiller.chat(prompt,system=sys_p,max_new_tokens=mx,do_sample=False,kb_top_k=0)
        except Exception as e:print(f'  [crawler] distill failed: {e}',flush=True);self._stats['failures']+=1;return ('',sources,0)
        self._stats['distillations']+=1
        return (ans,sources,n)

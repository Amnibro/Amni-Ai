"""Resident routines: source + asserts that live on python-band pages.

The ray/mmap retrieves them. The sandbox is what makes a result accurate.
"""
import ast,hashlib
from amni.compute.ptex_1t_store import CODE_SUBRANGES_1T

ROUTINES=[
 {"name":"knapsack","tags":("knapsack","dynamic","programming","backpack"),
  "code":(
   "def knapsack(weights, values, capacity):\n"
   "    n = len(weights)\n"
   "    dp = [0] * (capacity + 1)\n"
   "    for i in range(n):\n"
   "        w, v = weights[i], values[i]\n"
   "        for c in range(capacity, w - 1, -1):\n"
   "            dp[c] = max(dp[c], dp[c - w] + v)\n"
   "    return dp[capacity]\n"),
  "tests":[
   "assert knapsack([1,2,3],[1,4,5],5)==9",
   "assert knapsack([2],[10],1)==0",
   "assert knapsack([],[],5)==0",
  ]},
 {"name":"dijkstra","tags":("dijkstra","shortest","path","graph","priority"),
  "code":(
   "def dijkstra(graph, start):\n"
   "    import heapq\n"
   "    dist = {start: 0}\n"
   "    pq = [(0, start)]\n"
   "    while pq:\n"
   "        d, u = heapq.heappop(pq)\n"
   "        if d > dist.get(u, float('inf')): continue\n"
   "        for v, w in graph.get(u, []):\n"
   "            nd = d + w\n"
   "            if nd < dist.get(v, float('inf')):\n"
   "                dist[v] = nd\n"
   "                heapq.heappush(pq, (nd, v))\n"
   "    return dist\n"),
  "tests":[
   "assert dijkstra({'a':[('b',2),('c',5)],'b':[('c',1)]},'a')=={'a':0,'b':2,'c':3}",
   "assert dijkstra({'a':[]},'a')=={'a':0}",
  ]},
 {"name":"longest_increasing_subsequence","tags":("longest","increasing","subsequence","lis"),
  "code":(
   "def longest_increasing_subsequence(nums):\n"
   "    if not nums: return []\n"
   "    n = len(nums)\n"
   "    dp, prev = [1] * n, [-1] * n\n"
   "    for i in range(1, n):\n"
   "        for j in range(i):\n"
   "            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:\n"
   "                dp[i], prev[i] = dp[j] + 1, j\n"
   "    i = max(range(n), key=lambda k: dp[k])\n"
   "    out = []\n"
   "    while i != -1:\n"
   "        out.append(nums[i]); i = prev[i]\n"
   "    return out[::-1]\n"),
  "tests":[
   "assert longest_increasing_subsequence([3,1,2,4])==[1,2,4]",
   "assert longest_increasing_subsequence([5])==[5]",
   "assert longest_increasing_subsequence([])==[]",
  ]},
 {"name":"gcd","tags":("gcd","greatest","common","divisor"),
  "code":(
   "def gcd(a, b):\n"
   "    a, b = abs(a), abs(b)\n"
   "    while b:\n"
   "        a, b = b, a % b\n"
   "    return a\n"),
  "tests":[
   "assert gcd(12,8)==4",
   "assert gcd(7,3)==1",
   "assert gcd(0,5)==5",
  ]},
 {"name":"tree_rotations","tags":("tree","rotate","rotation","bst","node","struct"),
  "code":(
   "class Node:\n"
   "    def __init__(self, key, color='R'):\n"
   "        self.key = key\n"
   "        self.color = color\n"
   "        self.left = None\n"
   "        self.right = None\n"
   "        self.parent = None\n"
   "def left_rotate(root, x):\n"
   "    y = x.right\n"
   "    if y is None: return root\n"
   "    x.right = y.left\n"
   "    if y.left is not None: y.left.parent = x\n"
   "    y.parent = x.parent\n"
   "    if x.parent is None: root = y\n"
   "    elif x is x.parent.left: x.parent.left = y\n"
   "    else: x.parent.right = y\n"
   "    y.left = x\n"
   "    x.parent = y\n"
   "    return root\n"),
  "tests":[
   "assert Node(3).key==3",
   "a=Node(1); b=Node(2); a.right=b; b.parent=a; root=left_rotate(a,a); assert root is b and b.left is a",
  ]},
]

def python_page_for_key(key:str)->int:
 lo,hi=CODE_SUBRANGES_1T["python"]
 span=max(1,hi-lo)
 h=int(hashlib.sha256(key.encode("utf-8")).hexdigest(),16)
 return lo+(h%span)

def format_page(rec:dict)->bytes:
 lines=["#tags "+",".join(rec["tags"]),rec["code"].rstrip()]
 for t in rec["tests"]:
  lines.append("#test "+t)
 blob=("\n".join(lines)+"\n").encode("utf-8")
 if len(blob)>4096:raise ValueError("routine %s exceeds page"%rec["name"])
 return blob.ljust(4096,b" ")

def extract_tests(text:str)->str:
 tests=[]
 for ln in text.splitlines():
  s=ln.strip()
  if s.startswith("#test "):
   tests.append(s[6:].strip())
  elif s.startswith("assert "):
   tests.append(s)
 return "\n".join(tests)

def pack_routines(store,routines=None)->list:
 routines=routines or ROUTINES
 written=[]
 used=set()
 for rec in routines:
  p=python_page_for_key(rec["name"])
  if p in used:
   p=python_page_for_key(rec["name"]+"#2")
  used.add(p)
  store.write_page_data(p,format_page(rec))
  written.append({"name":rec["name"],"page":p})
 return written

def verify_code(code:str,harness:str)->dict:
 from amni.serve.self_debug import run_in_sandbox
 if not harness:
  try:
   ast_ok=True
   import ast as _a;_a.parse(code)
  except Exception:
   ast_ok=False
  return {"ok":False,"reason":"no_tests","parse_ok":ast_ok}
 r=run_in_sandbox(code,harness,timeout=3)
 ok=bool(r.get("ok")) or (r.get("ran") and r.get("returncode")==0 and not r.get("killed"))
 return {"ok":ok,"reason":"sandbox","stderr":(r.get("stderr") or "")[:240],"stdout":(r.get("stdout") or "")[:240]}

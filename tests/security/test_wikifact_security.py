import os,re,sys,ast,time
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from amni.compute import wikifact as WF
SRC=(ROOT/"amni"/"compute"/"wikifact.py").read_text()
LIB=(ROOT/"crates"/"adam-wikifact"/"src"/"lib.rs").read_text()
def test_no_network_or_exec_in_runtime():
 mods={a.name.split(".")[0] for n in ast.walk(ast.parse(SRC)) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in (n.names if isinstance(n,ast.Import) else [ast.alias(n.module or "")])}
 assert not mods&{"socket","urllib","requests","http","subprocess","sqlite3","pickle","shelve","ctypes"}
 assert not re.search(r"\b(eval|exec|os\.system|__import__)\s*\(",SRC)
def test_contact_properties_not_kept():
 keep={int(x) for x in re.search(r"KEEP: \[u16; \d+\] = \[([^\]]+)\]",LIB).group(1).split(",")}
 assert not keep&{968,1329,2002,2003,2013,6634,4264,2397,856}
def test_default_store_is_repo_local():
 assert WF.DIR.resolve().is_relative_to(ROOT) or "AMNI_WIKIFACT_DIR" in os.environ
 assert not re.search("/"+"work"+"space|/home/[a-z]",SRC)
@pytest.mark.parametrize("q",["","?"*5000,"who plays "+"a "*3000+"in b","who wrote \x00\x1b[2J","when was (.*)+ born","who directed '; DROP TABLE facts; --","capital of "+"‮"*50,"who is "+"😀"*200])
def test_hostile_questions_are_bounded(q):
 t=time.time();r=WF.answer(q);assert r is None or (isinstance(r,tuple) and len(r)==3);assert time.time()-t<10

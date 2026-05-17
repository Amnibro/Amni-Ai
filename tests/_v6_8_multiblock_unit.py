"""Iter24 unit: multi-block stitching — setup block + test block must share state in sandbox."""
import sys,subprocess,tempfile,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _extract_python_blocks
SAMPLE_MULTIBLOCK='''**APPROACH:** Two blocks — define helper, then test.

```python
def helper(n):
    return n * 2
```

Now we test it:

```python
result = helper(21)
print(f"Answer: {result}")
```

**TESTS:**
`assert helper(0) == 0`
`assert helper(5) == 10`
'''
def t_extract_multiple():
    blocks=_extract_python_blocks(SAMPLE_MULTIBLOCK)
    assert len(blocks)==2,f'expected 2 blocks, got {len(blocks)}'
    assert 'def helper' in blocks[0]
    assert 'result = helper(21)' in blocks[1]
    print(f'  extract: PASS ({len(blocks)} blocks)')
def t_runnable_last_only_fails():
    blocks=_extract_python_blocks(SAMPLE_MULTIBLOCK)
    runnable=[b for b in blocks if ('print(' in b or 'if __name__' in b)]
    last=runnable[-1]
    with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False) as f:
        f.write(last);p=f.name
    try:
        r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=5)
        assert r.returncode!=0,'OLD behavior: last block alone should fail (helper undefined)'
        assert 'helper' in r.stderr.lower() or 'NameError' in r.stderr
        print(f'  runnable[-1] alone FAILS as expected: stderr has NameError on `helper`')
    finally:os.unlink(p)
def t_stitched_works():
    blocks=_extract_python_blocks(SAMPLE_MULTIBLOCK)
    stitched='\n\n'.join(blocks) if len(blocks)>1 else blocks[-1]
    with tempfile.NamedTemporaryFile('w',suffix='.py',delete=False) as f:
        f.write(stitched);p=f.name
    try:
        r=subprocess.run([sys.executable,p],capture_output=True,text=True,timeout=5)
        assert r.returncode==0,f'stitched should pass: stderr={r.stderr}'
        assert 'Answer: 42' in r.stdout,f'expected "Answer: 42" in stdout: {r.stdout!r}'
        print(f'  stitched ALL blocks PASSES: stdout={r.stdout.strip()!r}')
    finally:os.unlink(p)
def t_single_block_unchanged():
    sample='```python\nprint(2+2)\n```'
    blocks=_extract_python_blocks(sample)
    assert len(blocks)==1
    stitched='\n\n'.join(blocks) if len(blocks)>1 else blocks[-1]
    assert stitched==blocks[0],'single-block path should be unchanged'
    print('  single-block path unchanged: PASS')
print('=== iter24 multi-block stitching unit ===')
t_extract_multiple()
t_runnable_last_only_fails()
t_stitched_works()
t_single_block_unchanged()
print('ALL PASS — multi-block code outputs now share state in sandbox')

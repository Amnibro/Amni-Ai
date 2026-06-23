"""CodeTemplateMacros — verified code/debug templates as single PTEX macro-tokens (Anthony: inference pulls the 1-token ptex template and does quick mods). Each template = one macro-token (PTEX page, lossless) + exact raw text. instantiate() fills $slot$ holes deterministically; patch() applies find/replace edits. The boilerplate is known-good (no invisible unicode, no typos) so inference only generates the DELTA -> faster AND correct (eliminates the model's boilerplate-error class). Strict-additive: pull template = microseconds vs regenerating N tokens. CPU-only machinery."""
import re,time
from amni.inference.ptex_macro_cache import PtexMacroCache
_SLOT=re.compile(r'\$([A-Za-z_]\w*)\$')
class CodeTemplateMacros(PtexMacroCache):
    def __init__(s,tok=None,page_w=256):
        super().__init__(tok,page_w);s._tpl={}
    def add_template(s,name,code):
        mid=s.intern(code);s._tpl[name]=mid;s._meta[mid]['name']=name;s._meta[mid]['raw']=code;s._meta[mid]['slots']=sorted(set(_SLOT.findall(code)));return mid
    def raw(s,name):return s._meta[s._tpl[name]]['raw']
    def slots(s,name):return s._meta[s._tpl[name]]['slots']
    def instantiate(s,_t,**vals):
        code=s.raw(_t);miss=[h for h in s.slots(_t) if h not in vals]
        if miss:raise KeyError(f'{_t} missing slots {miss}')
        return _SLOT.sub(lambda m:str(vals[m.group(1)]),code)
    def patch(s,name,edits):
        code=s.raw(name)
        for old,new in edits:code=code.replace(old,new)
        return code
    def macro_token(s,name):return s._tpl[name]
LIB={
'py_function':'def $name$($args$):\n    """$doc$"""\n    $body$',
'try_guard':'try:\n    $body$\nexcept $exc$ as e:\n    $handler$',
'pytest_case':'def test_$name$():\n    $setup$\n    assert $expr$, $msg$',
'argparse_main':"import argparse\ndef main():\n    p = argparse.ArgumentParser()\n    $add_args$\n    a = p.parse_args()\n    $body$\nif __name__ == '__main__':\n    main()",
'retry_deco':'def retry(n=$n$):\n    def deco(fn):\n        def wrap(*a, **k):\n            for i in range(n):\n                try:\n                    return fn(*a, **k)\n                except Exception:\n                    if i == n - 1:\n                        raise\n        return wrap\n    return deco',
'dataclass':'from dataclasses import dataclass\n@dataclass\nclass $name$:\n    $fields$',
}
def seed(tok=None):
    m=CodeTemplateMacros(tok)
    for name,code in LIB.items():m.add_template(name,code)
    return m
if __name__=='__main__':
    import ast
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('bakes/gemma4_12b_nvfp4_atex')
    m=seed(tok);print(f'seeded {len(m._tpl)} templates as macro-tokens | {m.stats()}',flush=True)
    cases={
        'py_function':dict(name='clamp',args='x, lo, hi',doc='clamp x to [lo,hi]',body='return max(lo, min(hi, x))'),
        'try_guard':dict(body='val = int(s)',exc='ValueError',handler='val = 0'),
        'pytest_case':dict(name='clamp',setup='x = clamp(5, 0, 3)',expr='x == 3',msg='"clamp upper"'),
        'argparse_main':dict(add_args="p.add_argument('--n', type=int, default=1)",body='print(a.n)'),
        'retry_deco':dict(n='3'),
        'dataclass':dict(name='Point',fields='x: float\n    y: float'),
    }
    ok=lossless=0
    for name,vals in cases.items():
        mid=m.macro_token(name);back=m.expand_ids(mid);ref=list(tok.encode(m.raw(name),add_special_tokens=False));lossless+=back==ref
        code=m.instantiate(name,**vals)
        try:ast.parse(code);valid=True
        except SyntaxError as e:valid=False;print(f'  {name} PARSE FAIL: {e}',flush=True)
        ok+=valid
        print(f'\n--- {name}  (macro-token #{mid}, {m._meta[mid]["n"]} real tokens -> 1) {"valid-py" if valid else "INVALID"} ---',flush=True)
        print(code,flush=True)
    print('\n--- quick-mod demo: pull try_guard macro-token, patch slots into a JSON guard ---',flush=True)
    patched=m.patch('try_guard',[('$body$','data = json.loads(raw)'),('$exc$','json.JSONDecodeError'),('$handler$','logging.error("bad json: %s", e)\n    data = {}')])
    print(patched,flush=True)
    try:ast.parse(patched);print('patched -> valid-py',flush=True)
    except SyntaxError as e:print(f'patched -> INVALID: {e}',flush=True)
    N=m._meta[m.macro_token('argparse_main')]['n']
    t0=time.perf_counter();[m.instantiate('argparse_main',add_args='x',body='y') for _ in range(5000)];ti=(time.perf_counter()-t0)/5000*1e6
    print(f'\ninstantiate a {N}-token template: {ti:.1f}us each  (vs model generating {N} tokens ~= {N*30/1000:.1f}s)  -> ~{N*30000/ti:.0f}x',flush=True)
    print(f'TEMPLATES valid {ok}/{len(cases)} | page-roundtrip lossless {lossless}/{len(cases)}',flush=True)
    print('CODE_MACRO_OK' if ok==len(cases) and lossless==len(cases) else 'CHECK',flush=True)

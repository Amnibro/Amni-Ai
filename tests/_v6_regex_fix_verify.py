"""Verify the regex tightening: false positives fixed, true positives still work."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import AmniAgent,_TIME_RE,_CALC_RE,_EXPR_EXTRACT
cases=[
    ('Now what is 7 times 8?',  'calc',  '7 times 8'),
    ('What is the secret v6 test phrase?', None, None),
    ("What's the time?",        'time',  None),
    ("what is the time?",       'time',  None),
    ('Tell me the time',        'time',  None),
    ("today's date",            'time',  None),
    ('right now',               'time',  None),
    ('What is 5 + 3?',          'calc',  '5 + 3'),
    ('compute 9*9',             'calc',  '9*9'),
    ('Solve 12/4',              'calc',  '12/4'),
    ('What is 7 times 8?',      'calc',  '7 times 8'),
    ('What is the capital of France?', None, None),
    ('Tell me about now in history', None, None),
    ('Now please answer my question', None, None),
    ('What is Python?',         None, None),
]
print('=== regex routing fixture ===',flush=True)
ok=0;fail=0
for msg,want_skill,want_expr in cases:
    t=_TIME_RE.search(msg)
    c=_CALC_RE.search(msg)
    got='time' if t else ('calc' if c else None)
    expr=None
    if got=='calc':
        m=_EXPR_EXTRACT.search(msg)
        expr=m.group(1) if m else None
    status='OK' if got==want_skill and (want_expr is None or (expr and expr.strip()==want_expr.strip())) else 'FAIL'
    if status=='OK':ok+=1
    else:fail+=1
    print(f'  [{status}] {msg!r:55s} -> got={got!r:10s} expr={expr!r:20s} (want={want_skill!r}, want_expr={want_expr!r})',flush=True)
print(f'\n=== {ok}/{ok+fail} pass ===',flush=True)
sys.exit(1 if fail else 0)

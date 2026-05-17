"""Unit smoke for _assert_diversity scorer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _assert_diversity
def t_low_diversity():
    s,info=_assert_diversity(['assert f(5)==6','assert f(5)==6'])
    print(f'  trivial dup: score={s:.2f}  {info}')
    assert s<0.4
def t_happy_only():
    s,info=_assert_diversity(['assert f(5)==6','assert f(3)==4','assert f(7)==8'])
    print(f'  happy-only:  score={s:.2f}  {info}')
    assert s<0.6
def t_full_coverage():
    s,info=_assert_diversity(['assert f(0)==1','assert f(-3)==-2','assert f(10**6)==10**6+1','assert f(5)==6'])
    print(f'  diverse:     score={s:.2f}  {info}')
    assert s>0.75,f'expected >0.75 got {s}'
def t_empty():
    s,info=_assert_diversity([])
    print(f'  empty:       score={s:.2f}')
    assert s==0.0
print('=== _assert_diversity unit ===')
t_low_diversity()
t_happy_only()
t_full_coverage()
t_empty()
print('PASS')

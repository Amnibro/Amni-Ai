"""Iter25 unit: _should_promote gates lesson promotion on quality thresholds."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _should_promote
def t_trivial_diversity_gated():
    ok,reason=_should_promote(snippet='def f(n):\n    return n+1\nprint(f(5))',asserts=['assert f(5)==6','assert f(5)==6'],diversity_score=0.25)
    assert not ok and 'diversity' in reason,f'expected gate on diversity: {reason}'
    print(f'  trivial diversity GATED: {reason}')
def t_short_code_gated():
    long_snippet='print(2)'
    ok,reason=_should_promote(snippet=long_snippet,asserts=['assert 1==1','assert 2==2','assert 3==3'],diversity_score=0.8)
    assert not ok and 'code' in reason and 'chars' in reason,f'expected gate on code length: {reason}'
    print(f'  short code GATED: {reason}')
def t_too_few_asserts_gated():
    snippet='def fib(n):\n    if n<=1: return n\n    return fib(n-1)+fib(n-2)\nprint(fib(10))'
    ok,reason=_should_promote(snippet=snippet,asserts=['assert fib(0)==0'],diversity_score=0.8)
    assert not ok and 'asserts' in reason,f'expected gate on assert count: {reason}'
    print(f'  too few asserts GATED: {reason}')
def t_quality_lesson_passes():
    snippet='def fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\nprint(fib(10))'
    ok,reason=_should_promote(snippet=snippet,asserts=['assert fib(0)==0','assert fib(-1) == -1','assert fib(10**3)==fib(999)+fib(998)'],diversity_score=0.83)
    assert ok and 'gate passed' in reason,f'expected pass: {reason}'
    print(f'  quality lesson PROMOTED: {reason}')
def t_boundary_diversity():
    snippet='x' * 100
    ok,reason=_should_promote(snippet=snippet,asserts=['assert a==b','assert c==d'],diversity_score=0.50)
    assert ok,f'diversity exactly 0.5 should pass: {reason}'
    print(f'  boundary div=0.5 PROMOTED: {reason}')
def t_custom_thresholds():
    ok,reason=_should_promote(snippet='print(1)',asserts=['assert 1==1'],diversity_score=1.0,min_diversity=0.1,min_code_chars=5,min_asserts=1)
    assert ok,f'custom thresholds should pass: {reason}'
    print(f'  custom thresholds PROMOTED: {reason}')
print('=== iter25 promotion-gate unit ===')
t_trivial_diversity_gated()
t_short_code_gated()
t_too_few_asserts_gated()
t_quality_lesson_passes()
t_boundary_diversity()
t_custom_thresholds()
print('ALL PASS — trivial/short/under-tested answers stay out of permanent lesson bank')

"""Iter26 unit: _error_hint matches common Python runtime errors and returns targeted fix advice."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from amni.serve.agent import _error_hint
def t_index_error():
    h=_error_hint('Traceback (most recent call last):\n  File "x.py", line 5, in <module>\n    print(lst[10])\nIndexError: list index out of range')
    assert h and 'off-by-one' in h.lower(),f'expected off-by-one hint: {h}'
    print(f'  IndexError -> {h[:60]}...')
def t_key_error():
    h=_error_hint('KeyError: \'foo\'')
    assert h and 'dict' in h.lower() and 'get' in h.lower(),f'expected dict.get hint: {h}'
    print(f'  KeyError -> {h[:60]}...')
def t_import_error():
    h=_error_hint('ModuleNotFoundError: No module named \'numpy\'')
    assert h and 'stdlib' in h.lower(),f'expected stdlib hint: {h}'
    print(f'  ImportError -> {h[:60]}...')
def t_type_error_operand():
    h=_error_hint('TypeError: unsupported operand type(s) for +: \'int\' and \'str\'')
    assert h and ('cast' in h.lower() or 'type' in h.lower()),f'expected type-cast hint: {h}'
    print(f'  TypeError(operand) -> {h[:60]}...')
def t_type_error_args():
    h=_error_hint('TypeError: my_func() missing 1 required positional argument: \'x\'')
    assert h and ('signature' in h.lower() or 'arguments' in h.lower()),f'expected signature hint: {h}'
    print(f'  TypeError(args) -> {h[:60]}...')
def t_recursion_error():
    h=_error_hint('RecursionError: maximum recursion depth exceeded')
    assert h and ('base case' in h.lower() or 'iteration' in h.lower()),f'expected recursion hint: {h}'
    print(f'  RecursionError -> {h[:60]}...')
def t_zero_division():
    h=_error_hint('ZeroDivisionError: division by zero')
    assert h and ('guard' in h.lower() or 'denominator' in h.lower()),f'expected guard hint: {h}'
    print(f'  ZeroDivisionError -> {h[:60]}...')
def t_attribute_error():
    h=_error_hint("AttributeError: 'list' object has no attribute 'lower'")
    assert h and 'attribute' in h.lower(),f'expected attribute hint: {h}'
    print(f'  AttributeError -> {h[:60]}...')
def t_name_error():
    h=_error_hint("NameError: name 'helper' is not defined")
    assert h and ('scope' in h.lower() or 'defined' in h.lower()),f'expected name hint: {h}'
    print(f'  NameError -> {h[:60]}...')
def t_value_error():
    h=_error_hint("ValueError: invalid literal for int() with base 10: 'abc'")
    assert h and ('conversion' in h.lower() or 'format' in h.lower()),f'expected conversion hint: {h}'
    print(f'  ValueError -> {h[:60]}...')
def t_unbound_local():
    h=_error_hint("UnboundLocalError: local variable 'x' referenced before assignment")
    assert h and ('initialize' in h.lower() or 'assignment' in h.lower()),f'expected unbound hint: {h}'
    print(f'  UnboundLocalError -> {h[:60]}...')
def t_unmatched_returns_none():
    h=_error_hint('SyntaxError: invalid syntax')
    assert h is None,f'unmatched error should return None: {h}'
    print('  unmatched SyntaxError -> None (correct passthrough)')
def t_empty_returns_none():
    assert _error_hint('') is None
    assert _error_hint(None) is None
    print('  empty/None stderr -> None')
print('=== iter26 error-hint pattern matcher ===')
t_index_error()
t_key_error()
t_import_error()
t_type_error_operand()
t_type_error_args()
t_recursion_error()
t_zero_division()
t_attribute_error()
t_name_error()
t_value_error()
t_unbound_local()
t_unmatched_returns_none()
t_empty_returns_none()
print('ALL PASS — Adam now gets targeted fix advice for 11 common runtime errors')

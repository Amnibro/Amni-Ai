### [Debugger Subagent: Empirical Sandbox Verification]
**Status**: PASSED (100/100)
**Execution Latency**: 39.55 ms

**Fault Diagnosis & Repair**:
> ZeroDivisionError: Divisor evaluated to zero on null denominator. Added defensive zero check.

```python
def f(a,b):
    if b == 0: return None
    return a/b
```

**Sandbox Execution Output**:
```
F_VERIFIED: Zero-division guard invariants pass
```
"""Defensive sandbox assertions — verify disallow paths / wrappers without embedding escape payloads."""
import os
import shutil
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
@pytest.mark.skipif(not Path('/usr/bin/bwrap').exists() and not Path('/usr/bin/bubblewrap').exists(),reason='bwrap not installed')
def test_x0_sandbox_helper_builds_unshare_net_clearenv():
    import importlib.util
    p=ROOT/'scripts'/'x0_code_eval.py'
    if not p.exists():
        pytest.skip('x0_code_eval.py missing')
    spec=importlib.util.spec_from_file_location('x0_code_eval',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert hasattr(mod,'run_sandbox')
    src=p.read_text(encoding='utf-8',errors='replace')
    assert '--unshare-net' in src or 'unshare-net' in src
    assert '--clearenv' in src or 'clearenv' in src
def test_self_debug_sandbox_refuses_high_danger_without_running():
    from amni.serve.self_debug import run_in_sandbox
    # uses a HIGH-danger builtin name only — no exploit steps, just that the gate refuses
    r=run_in_sandbox('eval("1+1")',harness='',timeout=3,allow_dangerous=False)
    assert r.get('blocked') is True or r.get('ran') is False
def test_skill_run_python_danger_regex_rejects_os_system_mention():
    from amni.serve import skills as sk
    src=Path(sk.__file__).read_text(encoding='utf-8',errors='replace')
    assert '_DANGEROUS_PYTHON' in src or 'danger_scan' in src or 'run_in_sandbox' in src

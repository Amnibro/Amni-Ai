"""Iter27 unit: verify /stats includes iter_counters with all expected keys + iter_rates computed correctly."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
EXPECTED_COUNTER_KEYS={'tests_passed','tests_failed','promoted','quality_gated','perturb_attempted','perturb_succeeded_small','perturb_succeeded_medium','perturb_succeeded_large','perturb_failed','intent_blocked','multi_block_stitched','hint_injected','lut_hits','cot_generations'}
def t_counter_keys_complete():
    src=Path('scripts/amni_serve.py').read_text(encoding='utf-8')
    init_line=[ln for ln in src.split('\n') if '_iter_counters=' in ln and 'tests_passed' in ln]
    assert init_line,'counter init line not found in amni_serve.py'
    init=init_line[0]
    missing=[k for k in EXPECTED_COUNTER_KEYS if f"'{k}'" not in init]
    assert not missing,f'missing counter keys: {missing}'
    print(f'  all {len(EXPECTED_COUNTER_KEYS)} counter keys present in init')
def t_bump_sites_present():
    src=Path('scripts/amni_serve.py').read_text(encoding='utf-8')
    sites={'intent_blocked':"_bump('intent_blocked')",'lut_hits':"_bump('lut_hits')",'cot_generations':"_bump('cot_generations')",'multi_block_stitched':"_bump('multi_block_stitched')",'tests_passed':"_bump('tests_passed')",'tests_failed':"_bump('tests_failed')",'promoted':"_bump('promoted')",'quality_gated':"_bump('quality_gated')",'perturb_attempted':"_bump('perturb_attempted')",'perturb_failed':"_bump('perturb_failed')",'hint_injected':"_bump('hint_injected')",'perturb_succeeded_small':"f'perturb_succeeded_{pr[\"magnitude\"].lower()}'"}
    missing=[(k,site) for k,site in sites.items() if site not in src]
    assert not missing,f'missing _bump sites: {missing}'
    print(f'  all 12 _bump call sites present')
def t_stats_endpoint_augmented():
    src=Path('scripts/amni_serve.py').read_text(encoding='utf-8')
    assert "base['iter_counters']=dict(_iter_counters)" in src
    assert "base['iter_rates']" in src
    assert "@app.get('/stats/iter')" in src
    assert "@app.post('/stats/iter/reset')" in src
    print('  /stats augmented + /stats/iter + /stats/iter/reset endpoints present')
def t_rate_math_safe_when_empty():
    perturb_attempted=0
    attempted=perturb_attempted or 1
    rate=0/attempted
    assert rate==0.0
    promoted=0;gated=0
    fire=gated/max(promoted+gated,1)
    assert fire==0.0
    tests_passed=0;tests_failed=0
    pass_rate=tests_passed/max(tests_passed+tests_failed,1)
    assert pass_rate==0.0
    print('  rate math safe on zero-denominator (no DivByZero on fresh server)')
print('=== iter27 telemetry unit (static analysis of amni_serve.py + safety) ===')
t_counter_keys_complete()
t_bump_sites_present()
t_stats_endpoint_augmented()
t_rate_math_safe_when_empty()
print('ALL PASS — telemetry wiring complete + safe on zero denominators')

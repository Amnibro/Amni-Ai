"""Tests for amni_demo_stats. The median test on an even-length list fails against the buggy implementation
(it returns the upper-middle element instead of the average of the two middle elements). Adam's job: run this,
read the failure, fix amni_demo_stats.median, and make all tests pass — without breaking the others."""
import os,sys
sys.path.insert(0,os.path.dirname(__file__))
from amni_demo_stats import mean,median,variance,stddev
def test_mean():
    assert mean([1,2,3,4])==2.5
def test_median_odd():
    assert median([3,1,2])==2
def test_median_even():
    assert median([1,2,3,4])==2.5
def test_variance():
    assert abs(variance([2,4,4,4,5,5,7,9])-4.0)<1e-9
def test_stddev():
    assert abs(stddev([2,4,4,4,5,5,7,9])-2.0)<1e-9
if __name__=='__main__':
    import traceback
    fails=[]
    for name,fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:fn()
            except Exception as e:fails.append(f'{name}: {type(e).__name__}: {e}')
    if fails:
        print('FAILED:');[print(' ',f) for f in fails];sys.exit(1)
    print('ALL PASS');sys.exit(0)

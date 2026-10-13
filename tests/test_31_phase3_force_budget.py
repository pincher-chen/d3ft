"""Near-zero force checks must retain a bounded, explicit absolute error budget."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation/workflows/phase3'))
import numpy as np
from force_acceptance import check_force_sum


def reject(f, floor=0.):
    try:
        check_force_sum(f, floor)
    except ValueError:
        return
    raise AssertionError('Invalid force sum accepted')


# A tiny uniform force fails the historical relative test but lies well below
# the new explicit absolute budget. The force array itself must not change.
f=np.array([[4e-9,0,0],[5e-9,0,0]])
original=f.copy()
reject(f)
r=check_force_sum(f,1e-6)
assert r['passes_net_force'] and r['passes_ionic_force']
assert r['net_force_limit']==5e-8
np.testing.assert_array_equal(f,original)
# Individually small forces can still have an unacceptable sum.
reject([[2e-7,0,0],[2e-7,0,0]],1e-6)
# The absolute floor must not excuse a large-force relative violation.
reject([[1e-3,0,0],[-.9e-3,0,0]],1e-6)
assert not check_force_sum([[1e-3,0,0],[-1e-3,0,0]],1e-6)['passes_ionic_force']
reject([[0,0,0],[0,0,0]],1.01e-6)
reject([[float('nan'),0,0],[0,0,0]],1e-6)
check_force_sum(np.zeros((2,3)))
print('Explicit force budget, raw-array preservation and failure guards passed')

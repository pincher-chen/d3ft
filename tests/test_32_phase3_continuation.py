"""Continuation must not recycle an unconverged state or change the Miller basis."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'validation/workflows/phase3'))
from continuation import SameBasisContinuation

cfg=SimpleNamespace(epoch=20)
mask=np.ones((3,3,3),bool)
calls=[]

def calculate(config,freq_mask=None,init_params=None):
    calls.append((config.epoch,init_params is not None))
    return SimpleNamespace(params_pw=np.ones((1,2,3,1)),params_occ={'x':np.ones(1)},
        freq_mask=mask.copy() if freq_mask is None else freq_mask.copy(),
        energy_history=np.arange(config.epoch),residual=1e-3 if init_params is not None else 1e-14)

report=lambda out:dict(variational_residual=out.residual)
with TemporaryDirectory() as tmp:
    work=Path(tmp);cache=SameBasisContinuation(5)
    cache.calculate(cfg,None,'first',work,calculate,report)
    out,r,info=cache.calculate(cfg,mask,'second',work,calculate,report)
    assert calls==[(20,False),(5,True),(20,False)]
    assert info['method']=='cold_fallback_after_continuation'
    assert (work/'second_continuation_attempt.json').exists()
    assert r['variational_residual']==1e-14
    changed=mask.copy();changed[0,0,0]=False
    _,_,info=cache.calculate(cfg,changed,'changed',work,calculate,report)
    assert info['method']=='cold' and calls[-1]==(20,False)
    # Even a state passing the broad production 1e-6 gate must not seed this
    # stricter continuation path when it fails the declared 1e-10 gate.
    cache=SameBasisContinuation(5)
    cache.calculate(cfg,None,'insufficient',work,calculate,
                    lambda out:dict(variational_residual=1e-8))
    assert cache.state is None
print('Cold fallback, changed-mask and strict source-stationarity guards passed')

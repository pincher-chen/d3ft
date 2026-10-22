"""Regression for the single-device fast path's omitted probe callback."""
import argparse,json,os
from pathlib import Path
os.environ['D3FT_RESPONSE_SINGLE_DEVICE']='1'
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import projected_solver as solver
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(exist_ok=False)
if jax.default_backend()!='cpu':raise ValueError('CPU software regression only')
entries=[];iteration=[1];counts=[]
def probe(aa,bb,value):
 ref=float(np.real(np.vdot(aa,bb)));err=abs(float(value)-ref)
 entries.append(dict(iteration=iteration[0],error=err))
 if err>1e-12:raise ValueError('Inner probe mismatch')
def callback(row):
 count=sum(e['iteration']==row['iteration'] for e in entries);counts.append(count)
 if count<2:raise ValueError('Single-device fast path omitted probe records')
 iteration[0]=row['iteration']+1
solver.inner_probe=probe
try:
 b=jnp.asarray([1+2j,2-1j,3+.2j,-.1+1j]);diag=jnp.asarray([1.,2.,1.,2.])
 x,record=solver.pcg(jax.jit(lambda v:diag*v),b,lambda v:v,lambda v:v,callback=callback)
finally:solver.inner_probe=None
error=float(jnp.max(abs(x-b/diag)));passed=len(counts)==2 and min(counts)>=2 and error<1e-12
(a.output/'acceptance.json').write_text(json.dumps(dict(passed=passed,per_iteration_counts=counts,entries=entries,solution_error=error,scope='CPU software regression of single-device probe hook; no DFT calculation repeated'),indent=2)+'\n')
print(json.dumps(dict(passed=passed,counts=counts,solution_error=error)),flush=True)
if not passed:raise ValueError('Single-device hook regression failed')

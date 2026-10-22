"""Certified finite-band k-mesh target and its spectral Hellmann--Feynman weights.

Hard extrema require isolated band AND k winners. At a degeneracy hard scalar
gap has no unique full gradient; reject rather than silently select an index.
Soft log-sum-exp over a declared occupied/conduction window is differentiable
at eigenvalue degeneracies (equal weights give a basis-independent trace).
"""
import numpy as np
def select_mesh_target(values,nocc,objective='hard',tau_Ha=None,margin_Ha=1e-5):
 values=np.asarray(values,float)
 if values.ndim!=2 or not np.isfinite(values).all() or not 1<=nocc<=values.shape[1]-2:
  raise ValueError('Finite ordered occupied/edge/buffer spectrum required')
 if np.min(np.diff(values,axis=1)) < -1e-10:raise ValueError('Spectrum not sorted')
 nk,nb=values.shape;vk=int(np.argmax(values[:,nocc-1]));ck=int(np.argmin(values[:,nocc]))
 hard=float(values[ck,nocc]-values[vk,nocc-1])
 if hard<=1e-4:raise ValueError('Positive insulating mesh gap required; no metal/fractional occupation support')
 def runner_up(vec,index,which):
  if len(vec)==1:return None
  other=np.delete(vec,index)
  return float(vec[index]-max(other) if which=='max' else min(other)-vec[index])
 margins=dict(vbm_band_Ha=None if nocc==1 else float(values[vk,nocc-1]-values[vk,nocc-2]),
  cbm_band_Ha=float(values[ck,nocc+1]-values[ck,nocc]),
  vbm_k_Ha=runner_up(values[:,nocc-1],vk,'max'),cbm_k_Ha=runner_up(values[:,nocc],ck,'min'))
 isolated=all(x is None or x>=margin_Ha for x in margins.values());weights=np.zeros_like(values)
 if objective=='hard':
  if not isolated:raise ValueError('Hard mesh gap has unresolved band/k degeneracy: '+str(margins))
  weights[vk,nocc-1]=-1.;weights[ck,nocc]=1.;value=hard;bias=0.
 elif objective=='soft':
  if tau_Ha is None or not np.isfinite(tau_Ha) or tau_Ha<=0:raise ValueError('Preregistered positive soft tau in Hartree required')
  def lse(z):
   mx=float(max(z.reshape(-1)));shift=np.exp(z-mx);return mx+np.log(shift.sum()),shift/shift.sum()
  vm,pv=lse(values[:,:nocc]/tau_Ha);cm,pc=lse(-values[:,nocc:]/tau_Ha)
  weights[:,:nocc]=-pv;weights[:,nocc:]=pc;value=float(-tau_Ha*(cm+vm))
  bias=float(tau_Ha*np.log((nk*nocc)*(nk*(nb-nocc))))
 else:raise ValueError('Choose explicit hard or finite-window soft objective')
 info=dict(objective=objective,target_Ha=value,hard_gap_Ha=hard,hard_gap_isolated=isolated,
  vbm_k_index=vk,cbm_k_index=ck,vbm_band=nocc-1,cbm_band=nocc,margins=margins,
  tau_Ha=tau_Ha,softening_bias_bound_Ha=bias,finite_band_window=[nb,nocc],
  linearized_HF_value_Ha=float(np.sum(weights*values)),
  degeneracy_policy='hard rejects unresolved extrema; soft uses full declared-window spectral trace, not a hard-gap gradient at crossings')
 return weights,info

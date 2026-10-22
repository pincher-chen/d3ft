"""Fresh local quadrature/translation/AD checks; no imported physical results."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from d3ft.pseudopotential import nlcc
from d3ft.pseudopotential.quadrature import qe_simpson_weights
from d3ft.occupation.fd_optional import fermi_dirac_occupation
from d3ft.optim.particle_hole_metric_optional import rotation_weight

P=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);args=ap.parse_args()
spec=importlib.util.spec_from_file_location('legacy_nlcc',args.evidence/'code_before/d3ft/pseudopotential/nlcc.py')
legacy=importlib.util.module_from_spec(spec);legacy.__package__='d3ft.pseudopotential';spec.loader.exec_module(legacy)
records={}
# Test QE's odd/even endpoint weights on independently integrated polynomials.
errors=[]
for size in [5,6,51,52]:
    h=.02;x=np.arange(size)*h;weights=qe_simpson_weights(np.full(size,h))
    for power in [0,1,2]:
        computed=float(weights@(x**power));exact=x[-1]**(power+1)/(power+1)
        errors.append(abs(computed-exact));assert abs(computed-exact)<1e-12
records['simpson_polynomial_max_error']=max(errors)
shape=(7,7,7);length=8.;grid=np.stack(np.meshgrid(*[np.fft.fftfreq(n)*n for n in shape],indexing='ij'),axis=-1)
g=jnp.asarray(grid*2*np.pi/length);position=jnp.array([[.2,.3,.4]])
r=np.arange(0.,6.0001,.01);alpha=1.2;core=np.exp(-alpha*r*r);rab=np.full_like(r,.01)
mask=r>0;rr=[r[mask]];cc=[core[mask]];ww=[rab[mask]];volume=length**3;kmax=float(np.max(np.linalg.norm(np.asarray(g),axis=-1)))+1.
# The no-RAB caller is PP_RHOATOM initialization, which must retain legacy AD.
old=legacy.core_charge_real(position,g,rr,cc,volume,kmax=kmax)
new=nlcc.core_charge_real(position,g,rr,cc,volume,kmax=kmax)
assert np.max(abs(np.asarray(old-new)))<1e-14
def initial_energy(module,pos):
    rho=module.core_charge_real(pos,g,rr,cc,volume,kmax=kmax)
    return jnp.mean(rho*rho)*volume
gold=jax.grad(lambda pos:initial_energy(legacy,pos))(position)
gnew=jax.grad(lambda pos:initial_energy(nlcc,pos))(position)
assert np.max(abs(np.asarray(gold-gnew)))<1e-12
records['legacy_atomic_density_max_difference']=float(jnp.max(abs(old-new)))
records['legacy_atomic_density_gradient_max_difference']=float(jnp.max(abs(gold-gnew)))
cutoff=.64
rec=nlcc.core_charge_reciprocal(position,g,rr,cc,volume,kmax=kmax,r_ab=ww,density_cutoff_Ha=cutoff)
norm2=np.sum(np.asarray(g)**2,axis=-1)
assert np.max(abs(np.asarray(rec)[norm2>2*cutoff+1e-10]))==0.
Q=(np.pi/alpha)**1.5
expected=Q*np.exp(-norm2/(4*alpha))*np.exp(-1j*np.einsum('...i,i->...',np.asarray(g),np.asarray(position[0])))*np.prod(shape)/volume
expected=np.where(norm2<=2*cutoff+1e-10,expected,0.)
error=float(np.max(abs(np.asarray(rec)-expected)))
assert error<1e-8,error
rho=nlcc.core_charge_real(position,g,rr,cc,volume,kmax=kmax,r_ab=ww,density_cutoff_Ha=cutoff)
assert abs(float(rho.mean()*volume)-Q)<1e-10
records.update(analytic_gaussian_reciprocal_max_error=error,core_charge_integral_error=abs(float(rho.mean()*volume)-Q),density_sphere_exact_zero=True)
shift=jnp.array([[.031,-.019,.014]])
shifted=nlcc.core_charge_reciprocal(position+shift,g,rr,cc,volume,kmax=kmax,r_ab=ww,density_cutoff_Ha=cutoff)
phase=jnp.exp(-1j*jnp.einsum('...i,i->...',g,shift[0]))
assert float(jnp.max(abs(shifted-rec*phase)))<1e-12
records['translation_phase_error']=float(jnp.max(abs(shifted-rec*phase)))
direction=jnp.array([[.4,-.3,.2]])
weight=jnp.asarray(np.sin(np.indices(shape).sum(axis=0)))
def objective(pos):
    density=nlcc.core_charge_real(pos,g,rr,cc,volume,kmax=kmax,r_ab=ww,density_cutoff_Ha=cutoff)
    return jnp.sum(density*weight)*volume/np.prod(shape)
vjp=jnp.sum(jax.grad(objective)(position)*direction)
jvp=jax.jvp(objective,(position,),(direction,))[1]
h=1e-5;fd=(objective(position+h*direction)-objective(position-h*direction))/(2*h)
assert abs(float(vjp-jvp))<1e-12
assert abs(float(vjp-fd))<1e-8
records.update(core_position_AD_mode_error=abs(float(vjp-jvp)),core_position_AD_FD_error=abs(float(vjp-fd)))
# Reciprocal/volume deformation at a fixed tabulation bound, away from sphere crossings.
def deformed(t):
    scale=1+t;gg=g/scale;pos=position*scale;vol=volume*scale**3
    density=nlcc.core_charge_real(pos,gg,rr,cc,vol,kmax=kmax,r_ab=ww,density_cutoff_Ha=cutoff)
    return jnp.sum(density*weight)*vol/np.prod(shape)
ad=float(jax.grad(deformed)(jnp.array(0.)));fd=float((deformed(h)-deformed(-h))/(2*h))
assert abs(ad-fd)<1e-8
records['core_cell_AD_FD_error']=abs(ad-fd)
try:
    nlcc.core_charge_real(position,g,rr,cc,volume,kmax=kmax,r_ab=ww)
except ValueError:
    records['missing_density_cutoff_rejected']=True
else:
    raise AssertionError('Missing density cutoff accepted')
# Optional finite-temperature occupation: exact charge and zero-sum tangent.
levels=jnp.array([[[-.2,-.03,.02,.11],[.01,.03,.06,.2]]]);ne=2.3;T=.02
occupation=fermi_dirac_occupation(levels,ne,T)
assert abs(float(occupation.sum())-ne)<1e-12
dz=jnp.arange(levels.size,dtype=float).reshape(levels.shape)/levels.size
_,df=jax.jvp(lambda x:fermi_dirac_occupation(x,ne,T),(levels,),(dz,))
assert abs(float(df.sum()))<1e-12
records['optional_FD_charge_error']=abs(float(occupation.sum())-ne)
records['optional_FD_tangent_charge_error']=abs(float(df.sum()))
xg=jnp.zeros((1,2,4,4),dtype=jnp.complex128)
metric=rotation_weight(occupation,xg)
assert np.isfinite(np.asarray(metric)).all() and float(metric.min())>0
records['optional_PH_metric_finite_positive']=True
report=dict(passed=True,checks=records,platform=str(jax.devices()[0]),
    scope='Fresh CPU formula/AD/interface checks. Not a QE SCF energy certificate, ionic relaxation acceptance, or full physical gap adjoint/forward comparison.')
output=args.evidence/'checks/nlcc_cpu_checks.json';output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

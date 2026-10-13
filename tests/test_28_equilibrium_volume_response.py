"""Volume response from deformed-cell pressures vs independent volume FD."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
path=Path(__file__).resolve().parents[1]/'validation/workflows/phase3/equilibrium_jacobian.py'
sys.path.insert(0,str(path.parent))
spec=importlib.util.spec_from_file_location('eqjac',path)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

# Smooth energy includes a pressure-producing bulk part and volume-dependent
# shape energy. Use the deformed volume J*V, including shear's nonunit J.
V0=20.; B=80.; h=.01
def energy(V,J,n,p):
    W=J*V; Veq=V0*np.exp(.2*p)
    return B*(W*np.log(W/Veq)-W+Veq)+n/2*W*(50+3*p+7*np.log(W/V0))*h*h
def pressure(V,J,n,p):
    W=J*V; Veq=V0*np.exp(.2*p)
    return -B*np.log(W/Veq)-n/2*(57+3*p+7*np.log(W/V0))*h*h
def bulk(V,p):
    Veq=V0*np.exp(.2*p)
    return B*(V*np.log(V/Veq)-V+Veq)
for branch,n in [('tet',6),('shear',4)]:
    s=np.diag([h,h,(1+h)**-2-1]) if branch=='tet' else np.array([[0,0,0],[0,0,h],[0,h,0]])
    J=np.linalg.det(np.eye(3)+s)
    def Q(V,p): return (2*energy(V,J,n,p)-2*bulk(V,p))/(n*V*h*h)
    q=Q(V0,0.)
    rec=dict(strain=s,pressure_GPa=pressure(V0,J,n,0.))
    qv=module.volume_derivative(q,rec,rec,dict(pressure_GPa=0.),n,h)
    d=1e-4
    fdv=(Q(V0*np.exp(d),0)-Q(V0*np.exp(-d),0))/(2*d)
    np.testing.assert_allclose(qv,fdv,atol=2e-6)
    fixed=(Q(V0,d)-Q(V0,-d))/(2*d)
    equilibrium=(Q(V0*np.exp(.2*d),d)-Q(V0*np.exp(-.2*d),-d))/(2*d)
    np.testing.assert_allclose(fixed+.2*qv,equilibrium,atol=2e-6)
print('test_28: equilibrium volume response agrees with independently re-equilibrated FD')

# Isotropic strain changes volume differently at +h and -h. Check the bulk
# estimator's equilibrium derivative directly, including its pressure term.
def eb(V,p):
    Veq=V0*np.exp(.2*p); bp=B+5*p
    return bp*(V*np.log(V/Veq)-V+Veq)
def pb(V,p): return -(B+5*p)*np.log(V/(V0*np.exp(.2*p)))
def qb(V,p): return (eb(V*(1+h)**3,p)+eb(V*(1-h)**3,p)-2*eb(V,p))/(9*V*h*h)
plus=dict(strain=np.eye(3)*h,pressure_GPa=pb(V0*(1+h)**3,0))
minus=dict(strain=-np.eye(3)*h,pressure_GPa=pb(V0*(1-h)**3,0))
qv=module.volume_derivative(qb(V0,0),plus,minus,dict(pressure_GPa=0),9,h)
d=1e-4
fixed=(qb(V0,d)-qb(V0,-d))/(2*d)
actual=(qb(V0*np.exp(.2*d),d)-qb(V0*np.exp(-.2*d),-d))/(2*d)
np.testing.assert_allclose(fixed+.2*qv,actual,atol=2e-6)
print('test_28: bulk finite-strain estimator volume response agrees with full equilibrium FD')

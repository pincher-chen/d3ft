"""Independent finite nonlinear electronic model and complex-real PCG checks."""
import json
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from scipy.optimize import minimize
from projected_solver import pcg

# Complex unknown with a real-linear SPD operator (not complex-linear).
b=jnp.array([1+2j,-.3+.7j]);action=lambda x:3*x+.5*jnp.conj(x)
x,rec=pcg(action,b,lambda x:x,lambda x:x/3)
assert rec['converged'] and np.max(abs(np.asarray(action(x)-b)))<1e-10
try:
 pcg(lambda x:-x,b,lambda x:x,lambda x:x)
 raise AssertionError('Negative curvature accepted')
except RuntimeError: pass
# Occupation-two SCF density model, solved independently at +/-theta.
h0=np.array([[-1.,.12,.08],[.12,.4,.05],[.08,.05,1.1]])
v=np.array([[.2,.15,-.07],[.15,-.3,.04],[-.07,.04,.1]])
k=np.array([[.15,.03,.01],[.03,.12,.02],[.01,.02,.1]])
def energy(y,t):
 q=y/jnp.linalg.norm(y);rho=2*q*q
 return 2*q@(jnp.asarray(h0)+t*jnp.asarray(v))@q+.5*rho@jnp.asarray(k)@rho
energy_and_grad=jax.jit(jax.value_and_grad(energy))
def solve(t):
 r=minimize(lambda y:tuple(np.asarray(z) for z in energy_and_grad(jnp.asarray(y),t)),
            np.array([1.,-.1,-.05]),jac=True,method='BFGS',options={'gtol':1e-12,'maxiter':1000})
 q=r.x/np.linalg.norm(r.x);rho=2*q*q;h=h0+t*v+np.diag(k@rho)
 values,vecs=np.linalg.eigh(h)
 assert np.linalg.norm(h@q-q*values[0])<1e-8
 return q,h,values,vecs
q,h,values,vecs=solve(0.)
project=lambda z:z-jnp.asarray(q)*jnp.dot(jnp.asarray(q),z)
B=h-values[0]*np.eye(3)+4*np.diag(q)@k@np.diag(q)
b=project(jnp.asarray(q*(k@(vecs[:,1]**2-vecs[:,0]**2))))
lam,rec=pcg(lambda z:project(jnp.asarray(B)@project(z)),b,project,lambda z:project(z))
bare=vecs[:,1]@v@vecs[:,1]-vecs[:,0]@v@vecs[:,0]
adjoint=bare-4*np.asarray(lam)@(v@q)
hstep=1e-5
fd=((solve(hstep)[2][1]-solve(hstep)[2][0])-(solve(-hstep)[2][1]-solve(-hstep)[2][0]))/(2*hstep)
assert abs(adjoint-fd)<1e-6,(adjoint,fd)
# Same nonlinear model tangent is independently contracted against the output RHS.
u,tangent=pcg(lambda z:project(jnp.asarray(B)@project(z)),-project(jnp.asarray(v@q)),project,lambda z:project(z))
forward=bare+4*np.asarray(b)@np.asarray(u)
assert abs(adjoint-forward)<1e-10
# The elastic scalar needs the stationarity-response contribution, not just
# the frozen-orbital second derivative. Check its factor against independent
# nonlinear ground-state reoptimization in this model.
he=1e-4
qe0=solve(0.)[0];qep=solve(he)[0];qem=solve(-he)[0]
elastic_fd=float((energy(qep,he)-2*energy(qe0,0.)+energy(qem,-he))/he**2)
elastic_response=float(4*np.asarray(u)@(v@q))
assert abs(elastic_response-elastic_fd)<1e-5,(elastic_response,elastic_fd)
print(json.dumps({'complex_real_linear_PCG':True,'reject_negative_curvature':True,
 'nonlinear_SCF_adjoint':float(adjoint),'independent_reoptimized_FD':float(fd),
 'absolute_AD_FD_error':float(abs(adjoint-fd)),'forward_reverse_error':float(abs(adjoint-forward)),
 'elastic_stationarity_response':elastic_response,'elastic_reoptimized_FD':elastic_fd,
 'elastic_absolute_error':abs(elastic_response-elastic_fd),
 'scope':'independent finite-model algorithm test; not main DFT response acceptance'},indent=2))

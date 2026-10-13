"""Block KS eigensolver: preserve eigenvalue multiplicities in cubic crystals."""
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, lobpcg


def lowest_block(op,dim,count,seed,tolerance=1e-10):
    block_count=count+4
    if dim<=5*block_count:
        raise ValueError('Too few plane waves for block eigensolver')
    action=jax.jit(op)
    block_action=jax.jit(jax.vmap(op,in_axes=1,out_axes=1))
    matrix=LinearOperator((dim,dim),dtype=np.complex128,
        matvec=lambda x:np.asarray(action(jnp.asarray(x.reshape(-1)))),
        matmat=lambda x:np.asarray(block_action(jnp.asarray(x))))
    kinetic=getattr(op,'kinetic_diagonal',None)
    preconditioner=None
    if kinetic is not None:
        diagonal=1/(1+np.asarray(kinetic))
        if diagonal.shape!=(dim,) or not np.isfinite(diagonal).all() or np.min(diagonal)<=0:
            raise ValueError('Invalid kinetic preconditioner')
        preconditioner=LinearOperator((dim,dim),dtype=np.complex128,
            matvec=lambda x:diagonal*x.reshape(-1),matmat=lambda x:diagonal[:,None]*x)
    rng=np.random.default_rng(seed)
    start=rng.normal(size=(dim,block_count))+1j*rng.normal(size=(dim,block_count))
    start=np.linalg.qr(start)[0]
    # Independent block directions can recover an exactly degenerate subspace;
    # a single-vector Krylov sequence may miss its multiplicity despite tiny residuals.
    with warnings.catch_warnings(record=True) as captured:
        vals,vecs=lobpcg(matrix,start,M=preconditioner,largest=False,tol=tolerance,maxiter=1000)
    order=np.argsort(vals)[:count];vals=vals[order];vecs=vecs[:,order]
    errors=np.linalg.norm(matrix.matmat(vecs)-vecs*vals[None,:],axis=0)
    orthogonality=np.max(np.abs(vecs.conj().T@vecs-np.eye(count)))
    if (not np.isfinite(vals).all() or not np.isfinite(errors).all()
            or errors.max()>1e-7 or orthogonality>1e-8):
        messages='; '.join(str(w.message) for w in captured)
        raise ValueError(f'Unconverged block eigensystem: residual={errors.max()}, '
                         f'orthogonality={orthogonality}; {messages}')
    return vals,errors

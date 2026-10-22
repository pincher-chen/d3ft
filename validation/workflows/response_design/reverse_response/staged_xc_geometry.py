"""Exact mixed XC/geometry AD with a staged Fourier/local chain rule.

Refinement is linear and independent of cell deformation. Cache its density
and three complex Cartesian spatial derivatives, then refine each density
direction separately. Apply mixed AD to local XC in bounded chunks and pull
back only a scalar volume ratio and a 3x3 reciprocal deformation. No finite
differences, change of quadrature, or approximate XC kernel is introduced.
"""
import gc
import jax
import jax.numpy as jnp
import numpy as np


def strain_matrix(v):
    return jnp.array([[v[0],v[5]/2,v[4]/2],
                      [v[5]/2,v[1],v[3]/2],
                      [v[4]/2,v[3]/2,v[2]]])


class StagedXCGeometry:
    def __init__(self,rho,g,volume,inverse0,kind,params,factor,chunk_size=16384):
        from d3ft.calc import xc_tunable as xc
        from jax_xc.utils import get_p
        self.volume=volume;self.inverse0=inverse0;self.shape=rho.shape
        self.factor=factor;self.g=g
        self.refine=jax.jit(lambda r:xc._refine_density_grid(r[None,...],g,factor)[0][0])
        self.rho=self.refine(rho);jax.block_until_ready(self.rho)
        self.n=self.rho.size;self.chunk=min(chunk_size,self.n)
        frequencies=tuple(jnp.fft.fftfreq(s)*s for s in self.rho.shape)
        basis=jnp.stack([g[1,0,0],g[0,1,0],g[0,0,1]])
        def component(rec,axis):
            weights=jax.lax.dynamic_index_in_dim(basis,axis,axis=1,keepdims=False)
            vector=(frequencies[0][:,None,None]*weights[0]
                    +frequencies[1][None,:,None]*weights[1]
                    +frequencies[2][None,None,:]*weights[2])
            return jnp.fft.ifftn(1j*vector*rec)
        self.component=jax.jit(component)
        self.fft=jax.jit(jnp.fft.fftn)
        self.u=self.spatial(self.rho)
        functions=[]
        for name in kind.split('+'):
            p=get_p(name,xc.XC_UNPOLARIZED)
            override=(params or {}).get(name)
            if override:p=p._replace(params=p.params._replace(**override))
            functions.append((xc.get_xc_functional(name,polarized=False),p))
        def local_energy(r,s):
            return r*sum(f(p,r,s) for f,p in functions)
        def action(scale,transform,r,u,w,du):
            transformed=scale*(u@transform)
            tangent=scale*(du@transform)
            sigma=jnp.real(jnp.vdot(transformed,transformed))
            dsigma=2*jnp.real(jnp.vdot(transformed,tangent))
            density_direction=jax.jvp(local_energy,(r*scale,sigma),
                                      (w*scale,dsigma))[1]
            return volume/scale*density_direction
        point_gradient=jax.vmap(jax.grad(action,argnums=(0,1)),
                                 in_axes=(None,None,0,0,0,0))
        def contract(scale,transform,r,w,u0,u1,u2,d0,d1,d2):
            arrays=tuple(x.reshape(-1) for x in (r,w,u0,u1,u2,d0,d1,d2))
            def one(i,acc):
                raw=i*self.chunk
                start=jnp.minimum(raw,self.n-self.chunk)
                chunks=[jax.lax.dynamic_slice(x,(start,),(self.chunk,)) for x in arrays]
                rr,ww,a,b,c,da,db,dc=chunks
                us=jnp.stack([a,b,c],axis=-1);dus=jnp.stack([da,db,dc],axis=-1)
                ps,pt=point_gradient(scale,transform,rr,us,ww,dus)
                # dynamic_slice clamps the last start: mask overlapping prefix.
                valid=jnp.arange(self.chunk)+start>=raw
                return (acc[0]+jnp.sum(jnp.where(valid,ps,0.)),
                        acc[1]+jnp.sum(jnp.where(valid[:,None,None],pt,0.),axis=0))
            total=jax.lax.fori_loop(0,(self.n+self.chunk-1)//self.chunk,one,
                                   (jnp.asarray(0.,dtype=r.dtype),jnp.zeros((3,3),dtype=r.dtype)))
            return total[0]/self.n,total[1]/self.n
        self.contract=jax.jit(contract)

    def spatial(self,r):
        rec=self.fft(r);jax.block_until_ready(rec)
        result=[]
        for axis in range(3):
            value=self.component(rec,jnp.asarray(axis));jax.block_until_ready(value)
            result.append(value)
        del rec
        return tuple(result)

    def gradient(self,delta,weight_r):
        # Direction is the conserved-coordinate weight divided by reference V.
        w=self.refine(weight_r/self.volume);jax.block_until_ready(w)
        du=self.spatial(w)
        def geometry(d):
            deformation=jnp.eye(3)+strain_matrix(d)@self.inverse0
            return 1/jnp.linalg.det(deformation),jnp.linalg.inv(deformation)
        (scale,transform),pullback=jax.vjp(geometry,delta)
        partial=self.contract(scale,transform,self.rho,w,*self.u,*du)
        jax.block_until_ready(partial)
        result=pullback(partial)[0];jax.block_until_ready(result)
        del w,du,partial
        gc.collect()
        return result


def parity_check():
    """Independent whole-program mixed AD, with skew cells and Nyquist modes."""
    from jrystal._src.grid import g_vectors
    from d3ft.calc.xc_tunable import xc_energy_tunable,default_xc_params
    from bounded_sigma_fft import bounded_sigma_fft
    from bounded_xc import memory_bounded_xc
    from compact_reciprocal_grid import compact_reciprocal_grid
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(76);kind='gga_x_pbe+gga_c_pbe'
    params=default_xc_params(kind);params['gga_x_pbe']['kappa']-=.025
    cell=jnp.array([[7.,.2,.3],[.1,8.,.2],[.2,.1,9.]])
    rho=jnp.asarray(.16+.025*rng.random((6,8,10)))
    weight=jnp.asarray(.02*rng.normal(size=rho.shape))
    volume=jnp.abs(jnp.linalg.det(cell));g=g_vectors(cell,np.asarray(rho.shape))
    theta=jnp.array([.003,-.002,.001,.002,-.001,.0015])
    inverse0=jnp.linalg.inv(jnp.eye(3)+strain_matrix(theta))
    records=[]
    with bounded_sigma_fft(),memory_bounded_xc(chunk_size=128):
        for factor in [2,5]:
            def reference(d,wr):
                deformed=cell@(jnp.eye(3)+strain_matrix(d)@inverse0).T
                vol=jnp.abs(jnp.linalg.det(deformed))
                gv=g_vectors(deformed,np.asarray(rho.shape))
                density=rho*volume/vol
                def energy(r):return xc_energy_tunable(r[None,...],gv,vol,kind,
                                                       xc_params=params,grid_factor=factor)
                return jax.jvp(energy,(density,),(wr/vol,))[1]
            exact=jax.jit(jax.grad(reference,argnums=0))
            with compact_reciprocal_grid():
                staged=StagedXCGeometry(rho,g,volume,inverse0,kind,params,factor,chunk_size=127)
                for delta,wr in [(jnp.zeros(6),weight),(theta,weight*.31),(-theta*.4,weight*-.27)]:
                    old=np.asarray(exact(delta,wr));new=np.asarray(staged.gradient(delta,wr))
                    error=float(np.max(np.abs(old-new)))
                    if error>1e-10:raise AssertionError((old,new,error))
                    records.append(dict(factor=factor,max_absolute_error=error,passed=True,
                                        reference_gradient=old.tolist(),staged_gradient=new.tolist()))
    return dict(passed=True,cases=records,scope='All six strain derivatives of the original whole-program PBE mixed density/geometry AD versus staged Fourier/local AD; factor2 and5, changed XC parameter, skew cell, Nyquist content, zero and nonzero geometry/weights; nondivisible chunks.')


if __name__=='__main__':
    import json
    print(json.dumps(parity_check(),indent=2))

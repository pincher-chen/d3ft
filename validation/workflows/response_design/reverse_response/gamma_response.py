"""Self-consistent Gamma band-gap tangent and adjoint from saved D3FT orbitals.

Spin restricted, fixed integer occupations, fixed Miller basis. Prescribed
strain/Cartesian displacements reoptimize electrons, without ionic relaxation.
Uses the physical Grassmann horizontal space Q^dagger dQ=0. The Omega+K
operator includes the exact discrete Hartree/XC kernel. No optimizer history,
frozen-density shortcut, dense Hessian or empty-space truncation is used.
"""
import argparse,datetime,hashlib,json,os,sys,time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import numpy as np
from geometry_layout import GeometryLayout
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from ml_collections import ConfigDict
from projected_solver import pcg,real_inner
from bounded_xc import memory_bounded_xc

HA_EV=27.211386245988

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def strain_matrix(v):
    return jnp.array([[v[0],v[5]/2,v[4]/2],[v[5]/2,v[1],v[3]/2],[v[4]/2,v[3]/2,v[2]]])

@contextmanager
def validated_radial_splines():
    # calc() has already checked the same radial nodes outside JIT.
    # Only disable repeated Python boolean checks inside JAX tracing.
    from d3ft.pseudopotential import local
    original=local.CubicSpline
    local.CubicSpline=partial(original,check=False)
    try:yield
    finally:local.CubicSpline=original

class GammaResponse:
    def __init__(self,out,orbitals,values,theta0,block_size=4,*,include_coordinates=False,displacement_unit='bohr',active_axes=None):
        from jrystal._src.grid import g_vectors
        from jrystal._src import potential
        from d3ft import pseudopotential as pp
        from d3ft.calc.xc_tunable import xc_energy_tunable
        self.out=out;self.shape=tuple(out.freq_mask.shape);self.n=int(np.prod(self.shape))
        self.idx=jnp.asarray(np.where(np.asarray(out.freq_mask).reshape(-1))[0])
        electrons=float(np.sum(out.pseudopot.valence_charges))
        self.n_occupied=int(round(electrons/2))
        if self.n_occupied<1 or abs(2*self.n_occupied-electrons)>1e-10:
            raise ValueError('Restricted closed-shell integer electron count required')
        if orbitals.shape[0]!=len(self.idx) or orbitals.shape[1]<self.n_occupied+2 or len(values)<orbitals.shape[1]:
            raise ValueError('Insufficient or mismatched occupied/edge/buffer eigenspace')
        self.q=jnp.asarray(orbitals[:,:self.n_occupied])
        self.edges=jnp.asarray(orbitals[:,[self.n_occupied-1,self.n_occupied]])
        self.values=jnp.asarray(values[:self.n_occupied]);self.block=int(np.gcd(block_size,self.n_occupied))
        if self.block<1:raise ValueError('Positive orbital FFT block size required')
        if (len(out.k_frac)!=1 or np.max(abs(np.asarray(out.k_frac)))>1e-14 or
            out.config.spin!=0 or not out.config.spin_restricted):raise ValueError('Restricted Gamma-only implementation')
        occupations=np.asarray(out.get_occupation(out.params_occ,out.final_eps))
        if (out.temperature!=0. or out.config.d3ft.occupation.enabled or
            occupations.shape[-1]!=self.n_occupied or np.max(abs(occupations-2.))>1e-12):
            raise ValueError('Zero-temperature fixed closed-shell occupations required')
        if out.core_density_grid is not None:raise ValueError('NLCC geometry response is not implemented')
        if os.environ.get('HNL_CONJ_FIX','1')!='1':raise ValueError('Require validated KB conjugation')
        self.cell=jnp.asarray(out.crystal.cell_vectors);self.positions=jnp.asarray(out.crystal.positions)
        self.layout=GeometryLayout(len(out.crystal.positions),theta0,include_coordinates=include_coordinates,displacement_unit=displacement_unit,active_axes=active_axes)
        self.n_variables=self.layout.nvariables
        self.zero=jnp.zeros(self.n_variables)
        self.volume=jnp.abs(jnp.linalg.det(self.cell));self.g=g_vectors(self.cell,np.asarray(self.shape))
        self.xc_params=out.config.d3ft.get('xc_params',None)
        self.xc_factor=out.config.d3ft.get('xc_grid_factor',1)
        def hxc(rho,g,volume):
            rec=jnp.fft.fftn(rho)
            # The stock kohn_sham=True helper detaches rho. Response needs dV_H/drho.
            squared=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
            vhrec=(4*jnp.pi*rec/squared).at[0,0,0].set(0.)
            vh=jnp.real(jnp.fft.ifftn(vhrec))
            def ex(density):
                return xc_energy_tunable(density[None,...],g,volume,out.config.xc,
                    xc_params=self.xc_params,grid_factor=self.xc_factor)
            vx=jax.grad(ex)(rho)*self.n/volume
            return vh+vx
        self.hxc=hxc
        self.rho=jax.jit(lambda:self.density(self.q,self.volume))()
        jax.block_until_ready(self.rho)
        def geometry_base(delta):
            cell,positions=self.layout.deform(self.cell,self.positions,delta)
            volume=jnp.abs(jnp.linalg.det(cell));g=g_vectors(cell,np.asarray(self.shape))
            local=pp.potential_local_reciprocal(positions,g,out.pseudopot.r_grid,
                out.pseudopot.local_potential_grid,out.pseudopot.local_potential_charge,
                volume,kmax=out.grid_kmax)
            gk=pp.gk_vectors(g,jnp.zeros((1,3)),out.freq_mask)
            beta=pp.beta_grid(out.pseudopot.r_grid,out.pseudopot.nonlocal_beta_grid,
                out.pseudopot.nonlocal_angular_momentum,gk,out.beta_kmax,tables=out.beta_tables)
            projectors=pp.projectors_reciprocal(positions,gk,out.pseudopot.nonlocal_angular_momentum,beta)[0]
            diagonal=jnp.sum(gk[0]**2,axis=-1)/2
            return diagonal,projectors,jnp.real(jnp.fft.ifftn(local)),volume,g
        self.geometry_base=geometry_base
        def geometry(delta):
            diagonal,projectors,local,volume,g=geometry_base(delta)
            rho=self.rho*self.volume/volume
            return diagonal,projectors,local+hxc(rho,g,volume),volume
        self.geometry=geometry
        self.nl=jnp.asarray(out.nl_d_matrix)
        self.project=jax.jit(lambda x:x-self.q@(jnp.conj(self.q).T@x))
        self.apply_local=jax.jit(self.local_action)
        zero=self.zero
        print('RESPONSE_STAGE geometry_potential',flush=True)
        self.diagonal,self.projectors,self.veff,self.volume0=jax.jit(geometry)(zero)
        jax.block_until_ready(self.veff)
        self.ham=jax.jit(lambda x:self.ham_action(x,self.diagonal,self.projectors,self.veff,self.volume0))
        self.kernel=jax.jit(lambda dr:jax.jvp(lambda r:hxc(r,self.g,self.volume),(self.rho,),(dr,))[1])
        self.delta_rho=jax.jit(self.density_tangent)
        self.hessian=jax.jit(lambda x:self.project(self.ham(x)-x*self.values[None,:]
                      +self.local_action(self.q,self.kernel(self.density_tangent(x)))))
        self.precondition=jax.jit(lambda x:self.project(self.project(x)/
                                      (.5+self.diagonal[:,None]-self.values[None,:])))
        self.edge_density=jax.jit(lambda:self.density(self.edges[:,1:2],self.volume,occupation=1.)
                        -self.density(self.edges[:,0:1],self.volume,occupation=1.))()
        self.explicit_gap=jax.jit(lambda d:real_inner(self.edges[:,1],self.ham_geometry(d,self.edges[:,1:2])[:,0])
                         -real_inner(self.edges[:,0],self.ham_geometry(d,self.edges[:,0:1])[:,0]))
        self.external=jax.jit(lambda d:jax.jvp(lambda z:self.ham_geometry(z,self.q),(zero,),(d,))[1])
        # Pull back the geometry through a scalar density-direction energy.
        # Forming the full v_xc geometry VJP first needs >130 GiB at 576^3.
        # <W,v_xc>/N = D_rho E_xc[W/V] gives the identical mixed derivative
        # while eliminating the full potential's reverse tape.
        self.edge_g=jnp.abs(self.edges[:,1])**2-jnp.abs(self.edges[:,0])**2
        self.edge_r=self.edge_density*self.volume
        def scalar_geometry(delta,weight_g,weight_r,lam):
            diagonal,projectors,local,volume,g=geometry_base(delta)
            kinetic=jnp.sum(weight_g*diagonal)
            rho=self.rho*self.volume/volume
            rec=jnp.fft.fftn(rho);g2=jnp.sum(g*g,axis=-1).at[0,0,0].set(1.)
            vh=jnp.real(jnp.fft.ifftn((4*jnp.pi*rec/g2).at[0,0,0].set(0.)))
            locality=jnp.sum(weight_r*(local+vh))/self.n
            edge=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),self.edges)
            occ=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),self.q)
            adj=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),lam)
            def coupling(x,y):return jnp.real(jnp.einsum('ami,ab,bmi->',jnp.conj(x),self.nl,y))
            nonlocal_term=(coupling(edge[:,:,1:2],edge[:,:,1:2])-coupling(edge[:,:,:1],edge[:,:,:1])
                           -4*coupling(adj,occ))/volume
            def ex(r):return xc_energy_tunable(r[None,...],g,volume,out.config.xc,
                    xc_params=self.xc_params,grid_factor=self.xc_factor)
            xc_direction=jax.jvp(ex,(rho,),(weight_r/volume,))[1]
            return kinetic+locality+nonlocal_term+xc_direction
        self.scalar_geometry=jax.jit(scalar_geometry)
        self.scalar_gradient=jax.jit(jax.grad(scalar_geometry,argnums=0))
        self.bare=lambda d:self.scalar_gradient(d,self.edge_g,self.edge_r,jnp.zeros_like(self.q))


    def wave(self,x):
        full=jnp.zeros((self.n,x.shape[1]),dtype=x.dtype).at[self.idx,:].set(x)
        return jnp.fft.ifftn(full.reshape(self.shape+(x.shape[1],)),axes=(0,1,2))

    def chunks(self,x):return jnp.swapaxes(x.reshape(x.shape[0],-1,self.block),0,1)

    def density(self,x,volume,occupation=2.):
        # Accumulate short FFT blocks to avoid an all-orbital grid tensor.
        size=self.block if x.shape[1]%self.block==0 else x.shape[1]
        blocks=jnp.swapaxes(x.reshape(x.shape[0],-1,size),0,1)
        def one(r,b):return r+jnp.sum(jnp.abs(self.wave(b))**2,axis=-1),None
        total,_=jax.lax.scan(one,jnp.zeros(self.shape),blocks)
        return total*occupation*self.n**2/volume

    def density_tangent(self,x):
        def one(r,pair):
            q,dx=pair
            return r+jnp.sum(jnp.real(jnp.conj(self.wave(q))*self.wave(dx)),axis=-1),None
        total,_=jax.lax.scan(one,jnp.zeros(self.shape),(self.chunks(self.q),self.chunks(x)))
        return total*4*self.n**2/self.volume

    def local_action(self,x,potential):
        size=self.block if x.shape[1]%self.block==0 else x.shape[1]
        blocks=jnp.swapaxes(x.reshape(x.shape[0],-1,size),0,1)
        def one(b):
            data=jnp.fft.fftn(potential[...,None]*self.wave(b),axes=(0,1,2))
            return data.reshape(self.n,size)[self.idx,:]
        result=jax.lax.map(one,blocks)
        return jnp.swapaxes(result,0,1).reshape(x.shape)

    def ham_action(self,x,diagonal,projectors,potential,volume):
        inner=jnp.einsum('bmg,gi->bmi',jnp.conj(projectors),x)
        coupled=jnp.einsum('ab,bmi->ami',self.nl,inner)
        nl=jnp.einsum('bmg,bmi->gi',projectors,coupled)/volume
        return diagonal[:,None]*x+nl+self.local_action(x,potential)

    def ham_geometry(self,d,x):return self.ham_action(x,*self.geometry(d))

    def reverse_gradient(self,lam):
        weight_g=self.edge_g-4*jnp.real(jnp.sum(jnp.conj(lam)*self.q,axis=1))
        weight_r=self.edge_r-self.delta_rho(lam)*self.volume
        return self.scalar_gradient(self.zero,weight_g,weight_r,lam)

    def solve(self,b,callback):
        return pcg(self.hessian,b,self.project,self.precondition,callback=callback)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('protocol',type=Path);ap.add_argument('--mode',choices=['reverse','forward'],required=True)
    ap.add_argument('--axes',type=int,nargs='*');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.protocol.read_text());a.output.mkdir(exist_ok=False)
    memory_sampler=None
    if cfg.get('development_memory_probe',cfg.get('include_coordinates',False)):
        from memory_probe import MemoryProbe
        memory_sampler=MemoryProbe(a.output).start()
    def stage(name,**extra):
        d=dict(stage=name,time=datetime.datetime.now().isoformat(),pid=os.getpid(),mode=a.mode,**extra)
        dump(a.output/'status.json',d);print('RESPONSE_STAGE',json.dumps(d),flush=True)
    source=Path(cfg['source']);orb=Path(cfg['orbitals']);meta=json.loads((source/'metadata.json').read_text())
    om=json.loads((orb/'metadata.json').read_text());summary=json.loads((orb/'summary.json').read_text())
    try:
        stage('source_validation')
        for name,h in om['source_hashes'].items():
            if sha(source/name)!=h:raise ValueError(f'Orbital source changed: {name}')
        if sha(orb/'orbitals.npy')!=summary['saved_orbitals_sha256']:raise ValueError('Orbital hash mismatch')
        point=json.loads((source/'point.json').read_text())
        if point['residual']>1e-6 or point['acoustic_sum_rule_Ha_Bohr']>5e-8:raise ValueError('Unqualified electronic parent')
        if min(summary['vbm_adjacent_separation_Ha'],summary['cbm_adjacent_separation_Ha'])<1e-5:
            raise ValueError('Band-edge degeneracy requires branch treatment')
        sys.path.insert(0,str(Path(cfg['runtime'])/'validation/workflows/response_design'))
        from radial_quadrature import local_simpson_patch
        from d3ft.calc.ground_state import calc
        from d3ft.parallel import plain_fft_patch
        c=ConfigDict(meta['config']);c.epoch=3;c.crystal_file_path_path=str(source/'input.xyz');c.d3ft.parallel.k_pool=False
        dump(a.output/'metadata.json',dict(protocol_sha256=sha(a.protocol),source=str(source),
            orbital_source=str(orb),mode=a.mode,axes=a.axes,job=os.environ.get('SLURM_JOB_ID'),
            physical_gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),software=cfg['source_hashes'],
            scope='New physical full-basis response pilot; accepted only after residual/FD/native gates'))
        with local_simpson_patch(),plain_fft_patch(k_pool=False):
            stage('reconstruct_geometry')
            with memory_bounded_xc():
                out=calc(c,freq_mask=np.load(source/'mask.npy'))
            with validated_radial_splines(), memory_bounded_xc():
                values=np.asarray(summary['eigenvalues_Ha']);vectors=np.load(orb/'orbitals.npy')
                stage('compile_physical_response')
                response=GammaResponse(out,vectors,values,cfg['theta0'],include_coordinates=cfg.get('include_coordinates',False),displacement_unit=cfg.get('displacement_unit','bohr'))
                charge=float(response.rho.mean()*response.volume)
                if abs(charge-2*response.n_occupied)>1e-8*2*response.n_occupied:raise ValueError('Charge mismatch')
                hq=response.ham(response.q)
                primal_res=float(jnp.max(jnp.linalg.norm(hq-response.q*response.values[None,:],axis=0)))
                bare_gap=float(response.explicit_gap(response.zero))*HA_EV
                if primal_res>1e-7 or abs(bare_gap-summary['gap_eV'])>3e-6:
                    raise ValueError(f'Rebuilt occupied density/Hamiltonian differs from certified primal: residual {primal_res}, gap {bare_gap}')
                stage('primal_gate_passed',residual_Ha=primal_res,gap_eV=bare_gap,charge=charge)
                compile_start=time.monotonic()
                b=response.project(response.apply_local(response.q,response.kernel(response.edge_density)))
                jax.block_until_ready(b)
                # First application is compilation; subsequent solves report warm timings separately.
                probe=response.hessian(b);jax.block_until_ready(probe)
                x=b/jnp.linalg.norm(b);y=response.project(b/(1+response.diagonal[:,None]));y=y/jnp.linalg.norm(y)
                lhs=float(real_inner(x,response.hessian(y)));rhs=float(real_inner(response.hessian(x),y))
                symmetry=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
                if symmetry>1e-8:raise ValueError(f'Real-inner-product symmetry gate failed: {symmetry}')
                compile_seconds=time.monotonic()-compile_start
                bare=response.bare(response.zero);jax.block_until_ready(bare)
                results={};solve_start=time.monotonic()
                def callback(label):return lambda row:stage('response_iteration',response=label,**row)
                if a.mode=='reverse':
                    lam,record=response.solve(b,callback('adjoint'))
                    # VJP of the external Hamiltonian action, contracted against the one adjoint.
                    gradient=response.reverse_gradient(lam)*HA_EV;jax.block_until_ready(gradient)
                    results=dict(gap_gradient_eV=np.asarray(gradient).tolist(),
                        frozen_state_gradient_eV=np.asarray(bare*HA_EV).tolist(),response_record=record)
                    np.save(a.output/'adjoint.npy',np.asarray(lam))
                else:
                    for axis in (a.axes if a.axes is not None else range(response.n_variables)):
                        if not 0<=axis<response.n_variables:raise ValueError('Invalid response axis')
                        direction=jnp.eye(response.n_variables)[axis];external=response.external(direction)
                        u,record=response.solve(-response.project(external),callback(f'tangent_{axis}'))
                        correction=4*real_inner(b,u)
                        results[str(axis)]=dict(gap_derivative_eV=float((bare[axis]+correction)*HA_EV),response_record=record)
                        dump(a.output/'partial_results.json',dict(completed_axes=list(results),results=results))
                dump(a.output/'result.json',dict(mode=a.mode,results=results,n_variables=response.n_variables,variable_names=response.layout.names,gradient_units=response.layout.gradient_units,primal_residual_Ha=primal_res,
                    primal_gap_eV=bare_gap,kernel_symmetry_relative_error=symmetry,
                    kernel_compile_seconds=compile_seconds,solve_and_contract_seconds=time.monotonic()-solve_start,
                    accepted=False,scope='Response solve completed; independent merged FD/native/mode acceptance still required'))
                stage('completed')
    except Exception as exc:stage('failed',error=str(exc));raise
    finally:
        if memory_sampler is not None:memory_sampler.finish()
if __name__=='__main__':main()

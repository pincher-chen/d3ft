"""One complete mode in a fresh process, solely for memory (not timing)."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


def dump(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--N', type=int, choices=[1,6,24,54], required=True)
    ap.add_argument('--mode', choices=['forward','reverse'], required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    c = json.loads(a.protocol.read_text())
    if os.environ.get('SLURMD_NODENAME') != 'hn35' or os.environ.get('CUDA_VISIBLE_DEVICES') != '7':
        raise ValueError('Only hn35 physical GPU7 is authorized')
    for path, expected in c['source_SHA256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen memory/source input changed: ' + path)
    a.output.mkdir(exist_ok=False)
    from device_memory_sampling import Sampler
    sampler = Sampler(a.output,[7],interval=.1).start()
    started = time.monotonic()
    allocator = []
    success = False

    def stage(name, **extra):
        sampler.set_stage(name)
        row = dict(time=datetime.datetime.now().astimezone().isoformat(), N=a.N, mode=a.mode,
                   stage=name, seconds_since_start=time.monotonic()-started, **extra)
        dump(a.output/'status.json',row)
        print('E3_MODE_MEMORY',json.dumps(row),flush=True)

    try:
        runtime = Path(c['runtime'])
        sys.path[:0] = [str(runtime),str(runtime/'validation/workflows/response_design'),
                       str(runtime/'validation/workflows/response_design/reverse_response')]
        import numpy as np
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jnp
        from ml_collections import ConfigDict
        from gamma_response import GammaResponse,validated_radial_splines,HA_EV
        from bounded_xc import memory_bounded_xc
        from projected_solver import real_inner
        from radial_quadrature import local_simpson_patch
        from d3ft.calc.ground_state import calc
        from d3ft.parallel import plain_fft_patch
        if len(jax.devices()) != 1:
            raise ValueError('Fresh single-device process required')

        def allocation(label):
            jax.effects_barrier()
            allocator.append(dict(stage=label, statistics=jax.devices()[0].memory_stats(),
                scope='Raw allocator fields, cumulative since this fresh process; not reset per stage'))
            dump(a.output/'allocator_statistics.json',allocator)

        E = Path(c['E3_root'])
        for name in ['si_acceptance_refined.json','mg_mode_acceptance.json','mg_fd_acceptance.json']:
            if not json.loads((E/name).read_text())['passed']:
                raise ValueError('Existing derivative acceptance failed: '+name)
        cfg = json.loads((E/'mg_response_protocol.json').read_text())
        source, orbitals = Path(cfg['source']), Path(cfg['orbitals'])
        meta=json.loads((source/'metadata.json').read_text())
        spec=json.loads((orbitals/'summary.json').read_text())
        om=json.loads((orbitals/'metadata.json').read_text())
        for name,h in om['source_hashes'].items():
            if sha(source/name)!=h:
                raise ValueError('Accepted source geometry changed: '+name)
        if sha(source/'state.pkl')!=meta['state_sha256'] or sha(orbitals/'orbitals.npy')!=spec['saved_orbitals_sha256']:
            raise ValueError('Accepted owned state/orbitals changed')
        stage('preparation')
        allocation('before_state_reconstruction')
        with (source/'state.pkl').open('rb') as stream:
            state=pickle.load(stream)
        config=ConfigDict(meta['config'])
        config.epoch=1
        config.optimizer_args.learning_rate=0.
        config.d3ft.optim.lr_schedule='constant'
        config.d3ft.parallel.k_pool=False
        config.crystal_file_path_path=str(source/'input.xyz')
        with plain_fft_patch(k_pool=False),local_simpson_patch(),memory_bounded_xc():
            out=calc(config,init_params=state,freq_mask=np.load(source/'mask.npy'))
            with validated_radial_splines():
                axes=list(range(a.N))
                stage('compilation_setup')
                r=GammaResponse(out,np.load(orbitals/'orbitals.npy'),np.asarray(spec['eigenvalues_Ha']),
                                cfg['theta0'],include_coordinates=True,active_axes=axes)
                if r.zero.shape!=(a.N,) or r.n_variables!=a.N:
                    raise ValueError('Not the requested genuine input graph')
                primal=float(jnp.max(jnp.linalg.norm(r.ham(r.q)-r.q*r.values[None,:],axis=0)))
                if primal>1e-7:
                    raise ValueError('Certified primal no longer matches')
                bfun=jax.jit(lambda:r.project(r.apply_local(r.q,r.kernel(r.edge_density))))
                b=bfun()
                jax.block_until_ready(b)
                directional=None
                if a.mode=='forward':
                    directional=jax.jit(lambda d:jax.jvp(r.explicit_gap,(r.zero,),(d,))[1])
                    jax.block_until_ready(directional(jnp.eye(a.N)[0]))
                allocation('after_graph_and_rhs_setup')
                ref=np.asarray(json.loads((E/'mg_mode_acceptance.json').read_text())['reverse_gradient_eV'])[axes]
                rows=[]
                values=[]
                stage('response',note='First-use PCG JIT may occur here; no hidden response warm-up')
                allocation('response_start')
                def callback(row):
                    if row['iteration']==1 or row['iteration']%10==0:
                        dump(a.output/'pcg_progress.json',dict(time=datetime.datetime.now().astimezone().isoformat(),
                            N=a.N,mode=a.mode,completed_axes=len(rows),**row))
                if a.mode=='reverse':
                    lam,rec=r.solve(b,callback)
                    if not rec['converged']:
                        raise ValueError('Reverse response did not converge')
                    g=r.reverse_gradient(lam)*HA_EV
                    jax.block_until_ready(g)
                    values=np.asarray(g).tolist()
                    if np.max(abs(np.asarray(values)-ref))>5e-6:
                        raise ValueError('Reverse gradient differs from certified full54')
                    rows.append(dict(response_record=rec,response_solves=1))
                else:
                    for axis in axes:
                        d=jnp.eye(a.N)[axis]
                        explicit=directional(d)
                        external=r.external(d)
                        u,rec=r.solve(-r.project(external),callback)
                        value=(explicit+4*real_inner(b,u))*HA_EV
                        jax.block_until_ready(value)
                        value=float(value)
                        if not rec['converged'] or abs(value-ref[axis])>5e-6+1e-6*abs(ref[axis]):
                            raise ValueError('Forward tangent did not meet certified tolerance')
                        values.append(value)
                        rows.append(dict(axis=axis,response_record=rec))
                        dump(a.output/'forward_partial.json',dict(completed_axes=len(rows),
                            expected_axes=a.N,gradient_eV=values,responses=rows))
                        stage('response',completed_axes=len(rows),expected_axes=a.N)
                allocation('response_complete')
                memory=sampler.finish()
                sampler=None
                if memory['errors'] or memory['peaks']['7']['present_samples']==0:
                    raise ValueError('PID memory sampling incomplete')
                result=dict(passed=True,N=a.N,mode=a.mode,physical_gpu=7,node='hn35',jobid=250285,
                    response_solves=(1 if a.mode=='reverse' else a.N),gradient_eV=values,
                    gradient_max_abs_error_eV=float(np.max(abs(np.asarray(values)-ref))),
                    primal_residual_Ha=primal,responses=rows,memory=memory,
                    allocator_statistics=allocator,variable_names=r.layout.names,
                    gradient_units=r.layout.gradient_units,total_GPU_hours=(time.monotonic()-started)/3600,
                    protocol_SHA256=sha(a.protocol),source_SHA256=c['source_SHA256'],
                    scope='Memory measurement only; shared node with E9/GaAs, no exclusive timing claim',
                    limitations='Sampled peaks are lower bounds; setup includes compilation, PCG may compile on first use; forward sequential, not N resident responses')
                dump(a.output/'result.json',result)
                success=True
                print('E3_MODE_MEMORY_COMPLETE',a.N,a.mode,flush=True)
    except BaseException as exc:
        dump(a.output/'failure.json',dict(error=repr(exc),complete=False,partial_results_retained=True))
        raise
    finally:
        if sampler is not None:
            sampler.finish()
        dump(a.output/'exit_state.json',dict(completed=success,GPU_hours=(time.monotonic()-started)/3600))


if __name__=='__main__':
    main()

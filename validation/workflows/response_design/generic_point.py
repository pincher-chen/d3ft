"""Immutable arbitrary-cell D3FT point pilot with auditable restart state.

Does not perform structural relaxation, elastic response or a gap adjoint.
An executable pilot is not automatically an accepted physical calculation.
"""
import argparse
from contextlib import ExitStack
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)

from d3ft.config import load_config
from d3ft.calc.ground_state import calc
from d3ft.calc.elastic import _write_xyz
from d3ft.calc.forces import force_report
from d3ft.calc.xc_tunable import default_xc_params

ROOT = Path(__file__).resolve().parents[3]
BOHR = 0.529177210903


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False,
        default=lambda x: np.asarray(x).tolist())+'\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--structure', type=Path, required=True)
    ap.add_argument('--pp-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cutoff', type=float, default=20)
    ap.add_argument('--grid', type=int, nargs='+', default=[32])
    ap.add_argument('--k', type=int, nargs='+', default=[2])
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--empty', type=int, default=0)
    ap.add_argument('--convergence-condition', type=float, default=0)
    ap.add_argument('--strain', default='[[0,0,0],[0,0,0],[0,0,0]]')
    ap.add_argument('--displacements', help='Cartesian Angstrom array, one row per atom')
    ap.add_argument('--resume-from', type=Path)
    ap.add_argument('--resume-quadrature-refinement', action='store_true',
                    help='Allow FFT/XC/local radial quadrature changes with identical ordered Miller basis')
    ap.add_argument('--resume-geometry-continuation', action='store_true',
                    help='Allow cell/positions changes only, holding the parent Miller mask fixed')
    ap.add_argument('--xc-grid-factor', type=int, default=1, choices=range(1,9))
    ap.add_argument('--local-radial-quadrature', choices=['trapezoid','simpson'],
                    default='trapezoid', help='Opt-in diagnostic local-potential quadrature')
    ap.add_argument('--require-stationary', action='store_true')
    ap.add_argument('--skip-forces', action='store_true')
    args = ap.parse_args()
    if args.resume_quadrature_refinement and args.resume_geometry_continuation:
        ap.error('Choose quadrature refinement or fixed-basis geometry continuation')
    if (args.resume_quadrature_refinement or args.resume_geometry_continuation) and not args.resume_from:
        ap.error('Continuation requires --resume-from')
    for key in ('grid','k'):
        values = getattr(args,key)
        if len(values) not in (1,3) or min(values)<1:
            ap.error(f'{key} requires one or three positive dimensions')
    if args.epochs<1 or args.cutoff<=0 or args.empty<0:
        ap.error('invalid numerical settings')
    spec = json.loads(args.structure.read_text())
    cell = np.asarray(spec['cell_angstrom'],float)
    frac = np.asarray(spec['fractional_positions'],float)
    symbols = spec['symbols']
    strain = np.asarray(json.loads(args.strain),float)
    disp = np.zeros_like(frac) if not args.displacements else np.asarray(json.loads(args.displacements),float)
    if cell.shape!=(3,3) or strain.shape!=(3,3) or frac.shape!=(len(symbols),3) or disp.shape!=frac.shape:
        raise ValueError('invalid structure, strain or displacement shape')
    if not all(np.isfinite(x).all() for x in (cell,frac,strain,disp)) or np.linalg.det(cell)<=0:
        raise ValueError('nonfinite or incorrectly oriented cell')
    matrix = np.eye(3)+strain
    positions = (frac @ cell) @ matrix.T + disp
    cell = cell @ matrix.T
    if np.linalg.det(cell)<=0:
        raise ValueError('strain makes a nonpositive cell')
    args.output.mkdir(parents=True,exist_ok=False)
    stage_file = args.output/'status.json'
    def status(stage, **extra):
        dump(stage_file,dict(stage=stage,time=datetime.datetime.now().isoformat(),
             pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),
             step=os.environ.get('SLURM_STEP_ID'),**extra))
        print('response_design_STAGE',stage,extra,flush=True)
    status('initializing')
    try:
        xyz=args.output/'input.xyz'
        _write_xyz(xyz,cell/BOHR,positions/BOHR,symbols)
        shutil.copyfile(args.structure,args.output/'structure_source.json')
        shutil.copyfile(__file__,args.output/'driver_snapshot.py')
        devices=jax.devices()
        if not all(d.platform=='gpu' for d in devices):
            raise RuntimeError('GPU pilot must run inside its allocated GPU step')
        k=np.repeat(args.k,3) if len(args.k)==1 else np.asarray(args.k)
        kpool=len(devices)>1
        if int(np.prod(k))%len(devices):
            raise ValueError('k mesh must divide the visible GPU count')
        config=load_config(ROOT/'validation/config/config_base.yaml')
        config.crystal=None
        config.crystal_file_path_path=str(xyz.resolve())
        config.spin=spec.get('spin',0)
        config.xc='gga_x_pbe+gga_c_pbe'
        # The old YAML registers scalar integer fields. Noncubic meshes and
        # a general floating cutoff legitimately require different types.
        with config.ignore_type():
            config.cutoff_energy=args.cutoff
            config.grid_sizes=args.grid[0] if len(args.grid)==1 else args.grid
            config.k_grid_sizes=args.k[0] if len(args.k)==1 else args.k
        config.empty_bands=args.empty
        config.epoch=args.epochs
        config.seed=args.seed
        config.verbose=False
        config.convergence_condition=args.convergence_condition
        config.pseudopotential_file_dir=str(args.pp_dir.resolve())
        config.optimizer_args.learning_rate=.01
        config.d3ft.optim.init='random'
        config.d3ft.optim.lr_schedule='cosine'
        config.d3ft.optim.lr_final=1e-5
        config.d3ft.occupation.enabled=False
        config.d3ft.kpoints.mesh='gamma'
        config.d3ft.parallel.k_pool=kpool
        config.d3ft.xc_grid_factor=args.xc_grid_factor
        config.d3ft.theta_nonlocal_l=[0.,0.,0.]
        config.d3ft.xc_params=default_xc_params(config.xc)
        pps={s:dict(path=str((args.pp_dir/f'{s}.upf').resolve()),
                   sha256=sha(args.pp_dir/f'{s}.upf')) for s in sorted(set(symbols))}
        physical=dict(cell_angstrom=cell.tolist(),positions_angstrom=positions.tolist(),
                      symbols=symbols,spin=config.spin,cutoff=args.cutoff,grid=args.grid,k=args.k,
                      empty=args.empty,xc=config.xc,pp=pps,xc_grid_factor=args.xc_grid_factor,
                      local_radial_quadrature=args.local_radial_quadrature,
                      radial_patch_sha256=sha(Path(__file__).with_name('radial_quadrature.py'))
                          if args.local_radial_quadrature=='simpson' else None)
        if args.local_radial_quadrature=='simpson':
            shutil.copyfile(Path(__file__).with_name('radial_quadrature.py'),
                            args.output/'radial_quadrature_snapshot.py')
        fingerprint=hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()
        packages={name:importlib.metadata.version(name) for name in ['jax','jaxlib','numpy','ase','optax']}
        metadata=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              source_sha256=sha(args.structure),physical=physical,physical_sha256=fingerprint,
              config=config.to_dict(),packages=packages,python=sys.version,
              devices=[str(d) for d in devices],cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
              code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'d3ft').rglob('*.py')},
              pilot_only=True,job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'))
        from jrystal._src.const import ANGSTROM2BOHR
        metadata['geometry_angstrom_to_bohr']=float(ANGSTROM2BOHR)
        dump(args.output/'metadata.json',metadata)
        # A coarse grid can complete but cannot be admitted as a density
        # representation. Reject aliasing before spending optimizer iterations.
        from jrystal.calc.opt_utils import create_freq_mask, create_grids
        from d3ft.calc.ground_state import _inscribed_g_radius
        grid_vectors, _, _ = create_grids(config)
        candidate_mask = create_freq_mask(config)
        g_cut = float(np.linalg.norm(np.asarray(grid_vectors), axis=-1)[candidate_mask].max())
        radius = float(_inscribed_g_radius(np.asarray(grid_vectors)))
        metadata['density_grid_radius_ratio'] = radius / g_cut
        dump(args.output/'metadata.json',metadata)
        if radius / g_cut < 2.0:
            raise ValueError('density grid aliases the wavefunction products; refine the FFT grid')
        state=mask=None
        if args.resume_from:
            prior=json.loads((args.resume_from/'metadata.json').read_text())
            previous_physical=dict(prior['physical'])
            previous_physical.setdefault('xc_grid_factor',1)
            previous_physical.setdefault('local_radial_quadrature','trapezoid')
            previous_physical.setdefault('radial_patch_sha256',None)
            differences=[k for k in set(previous_physical)|set(physical)
                         if previous_physical.get(k)!=physical.get(k)]
            if prior['code_sha256']!=metadata['code_sha256']:
                raise ValueError('restart requires identical D3FT source')
            permitted = ({'grid','xc_grid_factor','local_radial_quadrature','radial_patch_sha256'}
                         if args.resume_quadrature_refinement
                         else {'cell_angstrom','positions_angstrom'}
                         if args.resume_geometry_continuation else set())
            if differences and not set(differences) <= permitted:
                raise ValueError('restart changed physical inputs beyond the explicitly permitted continuation axes')
            state_path=args.resume_from/'state.pkl'
            if prior.get('state_sha256')!=sha(state_path):
                raise ValueError('checkpoint hash mismatch')
            # Only checkpoints produced by this driver and owned by this workflow.
            with state_path.open('rb') as f: state=pickle.load(f)
            previous_mask=np.load(args.resume_from/'mask.npy')
            mask=previous_mask
            if differences and args.resume_quadrature_refinement:
                def millers(selection):
                    indices=np.argwhere(np.asarray(selection))
                    for axis,n in enumerate(selection.shape):
                        indices[:,axis]=np.where(indices[:,axis]>=(n+1)//2,
                                                 indices[:,axis]-n,indices[:,axis])
                    return indices
                if not np.array_equal(millers(previous_mask),millers(candidate_mask)):
                    raise ValueError('quadrature continuation requires the identical ordered Miller basis')
                mask=np.asarray(candidate_mask)
            metadata['restart']=dict(parent=str(args.resume_from.resolve()),
                                      parent_physical_sha256=prior['physical_sha256'],
                                      changed_numerical_axes=sorted(differences),
                                      fixed_parent_miller_mask=args.resume_geometry_continuation,
                                      ordered_miller_basis_verified=True if differences and args.resume_quadrature_refinement else None)
            dump(args.output/'metadata.json',metadata)
        with ExitStack() as stack:
            if args.local_radial_quadrature=='simpson':
                from radial_quadrature import local_simpson_patch
                stack.enter_context(local_simpson_patch())
            if kpool:
                from d3ft.parallel import plain_fft_patch
                stack.enter_context(plain_fft_patch(k_pool=True))
            start=time.monotonic()
            status('electronic_solve')
            out=calc(config,init_params=state,freq_mask=mask)
            jax.block_until_ready(out.params_pw)
            solve_seconds=time.monotonic()-start
            np.save(args.output/'energy_history.npy',np.asarray(out.energy_history))
            np.save(args.output/'mask.npy',np.asarray(out.freq_mask))
            host_state=jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ))
            with (args.output/'state.pkl').open('xb') as f:pickle.dump(host_state,f,protocol=5)
            metadata['state_sha256']=sha(args.output/'state.pkl')
            dump(args.output/'metadata.json',metadata)
            status('stress_and_stationarity')
            report_start=time.monotonic()
            if kpool:
                sys.path.insert(0,str(ROOT/'validation/workflows/phase3'))
                from residual_report import fused_report
                rep=fused_report(out)
            else:
                from d3ft.calc.stress import stress_report
                rep=stress_report(out)
            stress=np.asarray(rep['stress'])
            residual=float(rep['variational_residual'])
            result=dict(energy_Ha=float(out.total_energy),pressure_GPa=float(rep['pressure_GPa']),
                        stress_Ha_Bohr3=stress.tolist(),residual=residual,
                        epochs=len(out.energy_history),solve_seconds=solve_seconds,
                        actual_fft_shape=list(out.freq_mask.shape),plane_waves=int(np.asarray(out.freq_mask).sum()),
                        natoms=len(symbols),volume_bohr3=float(out.crystal.vol),
                        cell_bohr=np.asarray(out.crystal.cell_vectors).tolist(),
                        positions_bohr=np.asarray(out.crystal.positions).tolist(),
                        k_fractional=np.asarray(out.k_frac).tolist(),
                        basis_convention='common fixed |G| sphere, all k points',
                        stationarity_pass=bool(np.isfinite(residual) and residual<=1e-6),
                        structural_relaxation_performed=False,pilot_only=True)
            if not args.skip_forces:
                status('forces')
                fr=force_report(out)
                result.update(forces_Ha_Bohr=np.asarray(fr['forces_Ha_Bohr']).tolist(),
                              acoustic_sum_rule_Ha_Bohr=float(fr['acoustic_sum_rule']),
                              max_force_Ha_Bohr=float(fr['max_force']),relative_asr=float(fr['relative_asr']))
            density=np.asarray(out.density_grid)
            np.save(args.output/'density.npy',density)
            result['integrated_valence_density']=float(density.sum(axis=0).mean()*result['volume_bohr3'])
            result['report_seconds']=time.monotonic()-report_start
            result['total_seconds']=time.monotonic()-start
            result['physical_acceptance']='not_evaluated_pilot'
            dump(args.output/'point.json',result)
            print('response_design_POINT',json.dumps(result),flush=True)
            if args.require_stationary and not result['stationarity_pass']:
                raise RuntimeError('electronic stationarity gate failed; pilot records retained')
            status('completed_pilot',stationarity_pass=result['stationarity_pass'],total_seconds=result['total_seconds'])
    except Exception as exc:
        status('failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    main()

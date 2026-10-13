"""Restartable current-version Phase 3 point and equilibrium calculations.

Results are immutable per run tag. Each completed point stores its actual cell,
forces, stationarity, parameters, and derivatives. The equilibrium search uses
analytic stress; elastic points use one fixed frequency mask at that equilibrium.
"""
import argparse
import atexit
from contextlib import ExitStack
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from d3ft.config import load_config
from d3ft.calc.ground_state import calc
from d3ft.calc.stress import stress_report
from d3ft.calc.forces import force_report
from d3ft.calc.screening import sensitivity
from d3ft.calc.xc_tunable import default_xc_params
from d3ft.calc.elastic import _HA_BOHR3_TO_GPA, _write_xyz
from force_acceptance import check_force_sum

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
MATERIALS=json.loads((HERE/'materials.json').read_text())
PARAMS=json.loads((HERE/'parameters.json').read_text())
BOHR=0.529177210903

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda v: str(v) if isinstance(v,Path) else np.asarray(v).tolist()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--material',required=True,choices=MATERIALS)
    ap.add_argument('--parameters',default='base')
    ap.add_argument('--vector',help='JSON vector overriding the named parameter set')
    ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['sample','evaluate'],default='sample')
    ap.add_argument('--grid',type=int)
    ap.add_argument('--cutoff',type=int,default=50)
    ap.add_argument('--k',type=int,default=6)
    ap.add_argument('--epochs',type=int,default=3000)
    ap.add_argument('--convergence-condition',type=float,default=0.,help='Energy-window standard-deviation stopping threshold in Ha; zero disables early stopping')
    ap.add_argument('--continuation-epochs',type=int,default=0,help='Optional shorter same-basis schedule; residual must be <=1e-10 or the full cold schedule is rerun')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--empty',type=int)
    ap.add_argument('--displacement',type=float,default=0.,help='atom-1 x displacement in Bohr')
    ap.add_argument('--a0',type=float)
    ap.add_argument('--shear',action='store_true')
    ap.add_argument('--reuse-from',type=Path,help='Reuse matching completed points in a new run directory')
    ap.add_argument('--jit-residual',action='store_true')
    ap.add_argument('--compare-residual',action='store_true')
    ap.add_argument('--fused-report',action='store_true',help='Fuse stress and stationarity in one reverse pass')
    ap.add_argument('--compare-report',action='store_true',help='Compare fused reporting with the eager-stress implementation')
    ap.add_argument('--compare-sensitivity',action='store_true',help='Compare forward/reverse stationary energy parameter AD at each derivative point')
    ap.add_argument('--sample-strain',help='JSON 3x3 strain tensor for a diagnostic sample')
    ap.add_argument('--sample-displacement',help='JSON three-component relative displacement in Bohr')
    ap.add_argument('--balanced-sample',action='store_true',help='Split sample displacement between both atoms')
    ap.add_argument('--xc-grid-factor',type=int,default=1,choices=range(1,9),help='Refine only the valence-density XC quadrature')
    ap.add_argument('--asr-force-floor',type=float,default=0.,help='Explicit raw-force ASR scale floor, at most ionic tolerance 1e-6 Ha/Bohr')
    ap.add_argument('--kpool',action='store_true',help='Explicitly distribute k points over all visible GPUs')
    ap.add_argument('--strain-step',type=float,default=.01)
    ap.add_argument('--pressure-tol',type=float,default=1e-5,help='Relative lattice tolerance |P|/(3B)')
    ap.add_argument('--fixed-mask',type=Path,help='Hold an existing Miller basis throughout parameter FD')
    ap.add_argument('--sample-observables',choices=['full','pressure'],default='full')
    args=ap.parse_args()
    if args.compare_report and not args.fused_report: ap.error('--compare-report requires --fused-report')
    if args.compare_sensitivity and args.mode=='sample' and args.sample_observables!='full':
        ap.error('--compare-sensitivity requires full sample observables')
    if args.mode!='sample' and (args.sample_strain or args.sample_displacement or args.balanced_sample):
        ap.error('Sample geometry flags require --mode sample')
    assert all(d.platform=='gpu' for d in jax.devices())
    if args.kpool:
        assert len(jax.devices())>1 and args.k**3%len(jax.devices())==0
        from d3ft.parallel import plain_fft_patch
        stack=ExitStack(); stack.enter_context(plain_fft_patch(k_pool=True)); atexit.register(stack.close)
    else: assert len(jax.devices())==1
    if not 1e-4<=args.strain_step<=.02: raise ValueError('strain-step must lie in [0.0001,0.02]')
    if not np.isfinite(args.convergence_condition) or args.convergence_condition<0:
        raise ValueError('convergence-condition must be finite and nonnegative')
    if not 0<args.pressure_tol<=1e-5: raise ValueError('pressure-tol must be positive and at most 1e-5')
    m=MATERIALS[args.material]
    if args.continuation_epochs<0 or args.continuation_epochs>=args.epochs:
        raise ValueError('Continuation epochs must be zero or shorter than full epochs')
    if not np.isfinite(args.asr_force_floor) or not 0<=args.asr_force_floor<=1e-6:
        raise ValueError('ASR force floor must be in [0,1e-6] Ha/Bohr')
    vector=np.asarray(json.loads(args.vector) if args.vector else PARAMS[args.parameters],float)
    if vector.shape!=(7,): raise ValueError('seven parameters required')
    # Production arrays can be redirected to a separate persistent filesystem
    # when the project quota is tight.  The default remains byte-for-byte
    # compatible with existing Phase 3 runs.
    results_root=Path(os.environ.get(
        'D3FT_PHASE3_RESULTS_ROOT', ROOT/'validation/results/phase3_20260908'))
    results_root.mkdir(parents=True,exist_ok=True)
    work=results_root/args.tag
    work.mkdir(parents=True,exist_ok=False)
    pp=Path(os.environ['D3FT_REFERENCE_ROOT'])/'pseudopotentials/sg15_pbe'/m['pp']
    c=load_config(ROOT/'validation/config/config_base.yaml')
    c.crystal=None; c.spin=0; c.xc='gga_x_pbe+gga_c_pbe'
    c.cutoff_energy=args.cutoff; c.grid_sizes=args.grid or m['grid']; c.k_grid_sizes=args.k
    c.empty_bands=m['empty'] if args.empty is None else args.empty
    c.epoch=args.epochs; c.convergence_condition=args.convergence_condition; c.seed=args.seed; c.verbose=False
    c.optimizer_args.learning_rate=0.01
    c.pseudopotential_file_dir=str(pp)
    c.d3ft.optim.init='random'; c.d3ft.optim.lr_schedule='cosine'; c.d3ft.optim.lr_final=1e-5
    c.d3ft.occupation.enabled=False; c.d3ft.kpoints.mesh='gamma'
    c.d3ft.parallel.k_pool=args.kpool
    if args.xc_grid_factor!=1: c.d3ft.xc_grid_factor=args.xc_grid_factor
    c.d3ft.theta_nonlocal_l=list(vector[:3])
    xp=default_xc_params(c.xc)
    for val,(f,key) in zip(vector[3:],[(f,k) for f in sorted(xp) for k in sorted(xp[f])]):
        xp[f][key]+=float(val)
    c.d3ft.xc_params=xp
    meta=dict(arguments=vars(args),config=c.to_dict(),parameter_vector=vector.tolist(),
              slurm_job=os.environ.get('SLURM_JOB_ID'),slurm_step=os.environ.get('SLURM_STEP_ID'),
              gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),project=str(ROOT),
              projector_mask=os.environ.get('PROJ_MASK','1'),xla_flags=os.environ.get('XLA_FLAGS'),
              allocator_environment={key:os.environ.get(key) for key in ['XLA_PYTHON_CLIENT_ALLOCATOR',
                  'XLA_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_MEM_FRACTION','XLA_PYTHON_CLIENT_PREALLOCATE']},
              pseudopotentials={str((pp/f'{s}.upf').resolve()):hashlib.sha256((pp/f'{s}.upf').read_bytes()).hexdigest() for s in set(m['symbols'])},
              code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'d3ft').rglob('*.py')})
    dump(work/'metadata.json',meta)
    shutil.copyfile(__file__,work/'worker_snapshot.py')
    snapshot=work/'workflow_snapshot'; snapshot.mkdir()
    files=list(HERE.glob('*.py'))+list(HERE.glob('*.json'))
    for source in files: shutil.copyfile(source,snapshot/source.name)
    meta['workflow_sha256']={source.name:hashlib.sha256(source.read_bytes()).hexdigest() for source in files}
    dump(work/'metadata.json',meta)
    if args.reuse_from:
        previous=json.loads((args.reuse_from/'metadata.json').read_text())
        for key in ['config','parameter_vector','pseudopotentials']:
            if previous[key]!=meta[key]: raise ValueError(f'Reuse requires matching {key}')
        allowed=json.loads((HERE/'reuse_compatibility.json').read_text())['entries'] if (HERE/'reuse_compatibility.json').exists() else []
        changes=[]
        both_xc_one=c.d3ft.get('xc_grid_factor',1)==1 and previous['config']['d3ft'].get('xc_grid_factor',1)==1
        both_k_false=not c.d3ft.parallel.k_pool and not previous['config']['d3ft']['parallel']['k_pool']
        conditions={'both_k_pool_false':both_k_false,'both_xc_grid_one':both_xc_one,
                    'both_k_pool_false_and_xc_grid_one':both_k_false and both_xc_one}
        for source in set(previous['code_sha256'])|set(meta['code_sha256']):
            old,new=previous['code_sha256'].get(source),meta['code_sha256'].get(source)
            if old==new: continue
            match=[e for e in allowed if e['path']==source and e['old']==old and e['new']==new
                   and conditions.get(e['condition'],False)]
            if not match: raise ValueError(f'Reuse source mismatch: {source}')
            changes.append(match[0])
        if changes: meta['reuse_compatible_source_changes']=changes
        meta['reused_from']=str(args.reuse_from.resolve())
        dump(work/'metadata.json',meta)
    print('START',args.tag,'GPU',meta['gpu'],'step',meta['slurm_step'],flush=True)
    records=[]
    continuation=None
    if args.continuation_epochs:
        from continuation import SameBasisContinuation
        continuation=SameBasisContinuation(args.continuation_epochs)
    def report_state(out):
        if args.fused_report:
            from residual_report import fused_report
            return fused_report(out,compare=args.compare_report)
        if args.jit_residual or args.compare_residual:
            from residual_report import report
            return report(out,compare=args.compare_residual)
        return stress_report(out)
    root_mask=np.load(args.fixed_mask) if args.fixed_mask else None
    if root_mask is not None:
        from jrystal._src.grid import proper_grid_size
        actual_grid=tuple(int(n) for n in proper_grid_size(c.grid_sizes))
        if root_mask.shape!=actual_grid: raise ValueError('Fixed mask/actual FFT grid shape mismatch')
        meta['fixed_mask_sha256']=hashlib.sha256(args.fixed_mask.read_bytes()).hexdigest()
        dump(work/'metadata.json',meta)

    def point(label,a,strain=None,u=0.,mask=None,derivatives=False,forces=False):
        t0=time.time()
        strain=np.zeros((3,3)) if strain is None else np.asarray(strain)
        displacement=np.array([u,0.,0.]) if np.ndim(u)==0 else np.asarray(u)
        balanced=label.startswith('shear_') or (args.mode=='sample' and args.balanced_sample)
        if args.reuse_from and (args.reuse_from/f'{label}.json').exists():
            rec=json.loads((args.reuse_from/f'{label}.json').read_text())
            old_u=rec['displacement_Bohr']
            old_u=np.array([old_u,0.,0.]) if np.ndim(old_u)==0 else np.asarray(old_u)
            matches=(abs(rec['a0_A']-a)<1e-12 and np.allclose(rec['strain'],strain,rtol=0,atol=1e-12)
                     and np.allclose(old_u,displacement,rtol=0,atol=1e-12))
            if np.linalg.norm(displacement)>1e-12 and bool(rec.get('balanced_displacement',False))!=balanced:
                matches=False
            if args.compare_sensitivity:
                matches=False  # Recompute both transforms on the same live wavefunction.
            if matches and (not derivatives or 'dE_dp' in rec) and (not forces or 'forces' in rec):
                if args.mode=='evaluate' and rec['residual']>1e-6: raise RuntimeError('Cannot reuse a nonstationary point')
                if mask is None: mask=np.load(args.reuse_from/'mask.npy')
                for suffix in ['.json','.xyz','_energy_history.npy']:
                    shutil.copyfile(args.reuse_from/f'{label}{suffix}',work/f'{label}{suffix}')
                if (args.reuse_from/f'{label}_density.npy').exists():
                    shutil.copyfile(args.reuse_from/f'{label}_density.npy',work/f'{label}_density.npy')
                records.append(rec); dump(work/'points.json',records)
                print('REUSED',label,flush=True)
                return rec,mask
        cell=np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]])/BOHR
        positions=np.array([[0,0,0],[a/4,a/4,a/4]])/BOHR
        cell=cell@(np.eye(3)+strain).T; positions=positions@(np.eye(3)+strain).T
        if balanced:
            positions[0]-=displacement/2
            positions[1]+=displacement/2
        else: positions[1]+=displacement
        xyz=work/f'{label}.xyz'; _write_xyz(str(xyz),cell,positions,m['symbols'])
        cfg=copy.deepcopy(c); cfg.crystal_file_path_path=str(xyz)
        if continuation is None:
            out=calc(cfg,freq_mask=mask)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                # Diagnostic checkpoints are deliberately separate from accepted
                # point files: no stationarity/force validation has run yet.
                np.save(work/f'{label}_unvalidated_energy_history.npy',np.asarray(out.energy_history))
                print('D3FT_STAGE report_begin',label,flush=True)
            rep=report_state(out)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE report_complete',label,flush=True)
            continuation_info=None
        else:
            out,rep,continuation_info=continuation.calculate(cfg,mask,label,work,calc,report_state)
        rec=dict(label=label,a0_A=a,strain=strain.tolist(),displacement_Bohr=displacement.tolist(),
                 balanced_displacement=balanced,
                 energy_Ha=float(out.total_energy),pressure_GPa=float(rep['pressure_GPa']),
                 stress=np.asarray(rep['stress']).tolist(),residual=float(rep['variational_residual']),
                 epochs=len(out.energy_history),volume_bohr3=float(abs(np.linalg.det(cell))))
        rec['wavefunction_sharding']=str(getattr(out.params_pw,'sharding',None))
        rec['fft_grid_shape']=list(out.freq_mask.shape)
        if continuation_info is not None: rec['continuation']=continuation_info
        rec['wavefunction_local_shapes']=[list(s.data.shape) for s in getattr(out.params_pw,'addressable_shards',[])]
        if 'eager_residual' in rep: rec['eager_residual']=rep['eager_residual']
        if 'comparison' in rep: rec['report_comparison']=rep['comparison']
        if forces:
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE force_begin',label,flush=True)
            fr=force_report(out)
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE force_complete',label,flush=True)
            rec['forces']=np.asarray(fr['forces_Ha_Bohr']).tolist(); rec['relative_asr']=fr['relative_asr']
        if derivatives:
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE sensitivity_begin',label,flush=True)
            rec['dE_dp']=sensitivity(out,xp,3).tolist()
            if args.compare_sensitivity:
                if not np.isfinite(rec['residual']) or rec['residual']>1e-6:
                    raise ValueError('Sensitivity comparison requires a stationary source')
                from compare_sensitivity import compare
                comparison=compare(out,xp,rec['dE_dp'],3)
                dump(work/f'{label}_sensitivity_comparison.json',comparison)
                if not comparison['passed']:
                    raise ValueError('Forward/reverse parameter AD comparison failed')
                rec['sensitivity_comparison']=comparison
            if os.environ.get('D3FT_TRACE_STAGES')=='1':
                print('D3FT_STAGE sensitivity_complete',label,flush=True)
        active=np.asarray(out.freq_mask)
        if label.startswith('eq_') or args.mode=='sample':
            density=np.asarray(out.density_grid)
            density_path=work/f'{label}_density.npy'
            np.save(density_path,density)
            rec['density_file']=density_path.name
            rec['density_sha256']=hashlib.sha256(density_path.read_bytes()).hexdigest()
            total_density=density.sum(axis=0) if density.ndim==4 else density
            rec['integrated_valence_density']=float(total_density.mean()*rec['volume_bohr3'])
        np.save(work/f'{label}_energy_history.npy',np.asarray(out.energy_history))
        rec['seconds']=time.time()-t0
        dump(work/f'{label}.json',rec); records.append(rec)
        dump(work/'points.json',records)
        print('POINT',json.dumps(rec),flush=True)
        if args.mode=='evaluate' and rec['residual']>1e-6:
            raise RuntimeError(f'{label}: electronic stationarity failed ({rec["residual"]:.3g})')
        del out
        jax.clear_caches(); gc.collect()
        return rec,active

    a=args.a0 or m['a0']
    if args.mode=='sample':
        full=args.sample_observables=='full'
        strain=np.asarray(json.loads(args.sample_strain),float) if args.sample_strain else None
        displacement=np.asarray(json.loads(args.sample_displacement),float) if args.sample_displacement else args.displacement
        if strain is not None and strain.shape!=(3,3): raise ValueError('Sample strain must be 3x3')
        if np.ndim(displacement)>0 and displacement.shape!=(3,): raise ValueError('Sample displacement must have three components')
        rec,mask=point('sample',a,strain=strain,u=displacement,mask=root_mask,derivatives=full,forces=full)
        np.save(work/'mask.npy',mask)
        dump(work/'summary.json',rec)
        return

    # Newton then secant in lattice constant, with a material-scaled tolerance.
    bg=m['Bguess']; prev=None; eq=None
    for it in range(7):
        eq,mask=point(f'eq_{it}',a,mask=root_mask,derivatives=True,forces=args.shear and m['C44'] is not None)
        p=eq['pressure_GPa']
        if eq['residual']>1e-6: raise RuntimeError('Electronic state is not stationary; inspect point before fitting')
        if abs(p)/(3*bg)<args.pressure_tol: break
        if prev is not None and abs(a-prev[0])>1e-9:
            slope=(p-prev[1])/(a-prev[0])
            if slope<0: bg=-slope*a/3
        change=np.clip(p/(3*bg),-.02,.02)
        prev=(a,p); a*=1+change
    else: raise RuntimeError('Analytic pressure root did not converge')
    np.save(work/'mask.npy',mask)
    center=eq; volume=center['volume_bohr3']
    result=dict(material=args.material,parameter_vector=vector.tolist(),a0_A=a,
                equilibrium_pressure_GPa=p,equilibrium_residual=center['residual'])
    eps=args.strain_step
    strain_label=lambda d:f'{d:+.3f}' if eps==.01 else f'{d:+.6f}'
    iso={0:center}
    for d in [-2*eps,-eps,eps,2*eps]:
        iso[d],_=point('iso_'+strain_label(d),a,np.eye(3)*d,mask=mask,derivatives=True)
    def curvature(data,h,denominator):
        e=(data[h]['energy_Ha']-2*data[0]['energy_Ha']+data[-h]['energy_Ha'])/h**2
        g=(np.asarray(data[h]['dE_dp'])-2*np.asarray(data[0]['dE_dp'])+np.asarray(data[-h]['dE_dp']))/h**2
        return e*_HA_BOHR3_TO_GPA/(denominator*volume),g*_HA_BOHR3_TO_GPA/(denominator*volume)
    b1,g1=curvature(iso,eps,9); b2,g2=curvature(iso,2*eps,9)
    B=(4*b1-b2)/3+2*p/3
    p1,p0=iso[eps]['pressure_GPa'],iso[-eps]['pressure_GPa']
    p2,p3=iso[2*eps]['pressure_GPa'],iso[-2*eps]['pressure_GPa']
    pressure_slope=(8*(p1-p0)-(p2-p3))/(12*eps)
    Bstress=-pressure_slope/3
    # EOS B' inferred from pressure as a function of logarithmic volume:
    # P(d)=-3 B d + (3/2) B(3 B'+1) d² + ... .
    psecond=(p1-2*p+p0)/eps**2
    Bprime=(psecond/(3*B)-1)/3
    gradp=-(np.asarray(iso[eps]['dE_dp'])-np.asarray(iso[-eps]['dE_dp']))*_HA_BOHR3_TO_GPA/(6*eps*volume)
    gradB=(4*g1-g2)/3+(2/3-Bprime)*gradp
    grada=a*gradp/(3*B)
    result.update(B_GPa=B,B_stress_GPa=Bstress,B_h_GPa=b1,B_2h_GPa=b2,strain_step=eps,Bprime=Bprime,
                  da0_dp=grada.tolist(),dB_equilibrium_dp=gradB.tolist())
    if eps==.01: result.update(B_delta01_GPa=b1,B_delta02_GPa=b2)
    if m['C11'] is None:
        dump(work/'summary.json',result); print('DONE',args.tag,flush=True); return
    tet={0:center}
    for d in [-eps,eps]:
        tet[d],_=point('tet_'+strain_label(d),a,np.diag([d,d,(1+d)**-2-1]),mask=mask,derivatives=True)
    D,gradD=curvature(tet,eps,6)
    result.update(C11_minus_C12_GPa=D,C11_GPa=B+2*D/3,C12_GPa=B-D/3,
                  dD_fixed_volume_dp=gradD.tolist())
    # The tetragonal gradient is explicitly labelled fixed-volume. An equilibrium
    # derivative needs a volume-response term; never silently use it in LM.
    if args.shear and m['C44'] is not None:
        shear={0:center}
        for d in [-eps,eps]:
            strain=np.zeros((3,3)); strain[1,2]=strain[2,1]=d
            old=None; u=np.zeros(3); invH=np.eye(3)*3.
            for it in range(15):
                rec,_=point('shear_'+strain_label(d)+f'_{it}',a,strain,u=u,mask=mask,derivatives=True,forces=True)
                f=np.asarray(rec['forces'])
                if rec['residual']>1e-6: raise RuntimeError('Nonstationary shear state')
                fmax=np.abs(f).max()
                acceptance=check_force_sum(f,floor=args.asr_force_floor)
                dump(work/(rec['label']+'_force_acceptance.json'),acceptance)
                if fmax<1e-6: break
                grad=(f[0]-f[1])/2
                if old is not None:
                    step_u=u-old[0]; y=grad-old[1]; sy=step_u@y
                    if sy>1e-14:
                        rho=1/sy; T=np.eye(3)-rho*np.outer(step_u,y)
                        invH=T@invH@T.T+rho*np.outer(step_u,step_u)
                step=-invH@grad
                step*=min(1.,.05/max(np.linalg.norm(step),1e-30))
                old=(u.copy(),grad.copy()); u+=step
            else: raise RuntimeError('Shear internal coordinate did not converge')
            shear[d]=rec
        C44,g44=curvature(shear,eps,4)
        result.update(C44_GPa=C44,dC44_fixed_volume_dp=g44.tolist())
    dump(work/'summary.json',result)
    print('DONE',args.tag,json.dumps(result),flush=True)

if __name__=='__main__': main()

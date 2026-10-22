"""Preregistered k4->k6 checks at the original and actual reverse geometries.

The frozen runtime uses exact compact k-pool terms after independent formula
and physical-checkpoint parity gates. All 216 k6 spectrum points are retained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

P = Path('${D3FT_ROOT}')
R = P/'validation/results/response_design_20261005_numerical_verification'
F = P/'validation/results/response_design_20261002_resource_fill/comparison_runtime'
K = R/'kmesh_v3'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def replace_once(text, old, new):
    if text.count(old)!=1: raise ValueError('Frozen source anchor changed: '+old[:60])
    return text.replace(old,new,1)

def prepare():
    K.mkdir(exist_ok=False)
    for name in ['runs','logs','commands','spectra','drivers']: (K/name).mkdir()
    runtime=K/'runtime';shutil.copytree(F,runtime)
    kernel=P/'validation/workflows/response_design/compact_k_terms.py'
    shutil.copy2(kernel,runtime/'d3ft/calc/compact_k_terms.py')
    gs=runtime/'d3ft/calc/ground_state.py';s=gs.read_text()
    anchor='    if config.use_pseudopotential:\n\n        # pot_nl is threaded in'
    insertion='''    compact = os.environ.get('D3FT_COMPACT_K_TERMS') == '1'
    if compact:
        if not config.d3ft.optim.enabled or not config.spin_restricted or config.d3ft.occupation.enabled:
            raise ValueError('Compact pilot requires restricted Stiefel/integer occupations')
        if not config.use_pseudopotential:
            raise ValueError('Compact pilot requires the pseudopotential energy')
        from .compact_k_terms import make_terms
        compact_terms = make_terms(freq_mask, kpool_mesh_obj, block=16)
        def total_energy(params_pw, params_occ, eps, pot_nl):
            occ = get_occupation(params_occ, eps)
            density, e_kin, e_nloc = compact_terms(params_pw, occ, g_vec, k_vec, pot_nl, crystal.vol, nl_d_matrix)
            density_rec = jnp.fft.fftn(density, axes=range(-3,0))
            return (e_kin + e_nloc + energy.hartree(density_rec,g_vec,crystal.vol)
                    + d3pp.energy_local(density_rec,potential_loc,vol=crystal.vol)
                    + _xc(density,g_vec,crystal.vol,core_density_grid))
    elif config.use_pseudopotential:

        # pot_nl is threaded in'''
    s=replace_once(s,anchor,insertion)
    # Avoid constructing the dense coefficient tensor even at final readout.
    anchor='    if kpool_mesh_obj is not None and os.environ.get("D3FT_FUSE_FINAL_DENSITY") == "1":\n'
    insertion='''    if compact:
        from jax.sharding import NamedSharding, PartitionSpec
        rep = NamedSharding(kpool_mesh_obj, PartitionSpec())
        def compact_density(p,o,nl):
            return compact_terms(p,get_occupation(o,eps_t),g_vec,k_vec,nl,crystal.vol,nl_d_matrix)[0]
        density_grid = jax.jit(compact_density,
            in_shardings=(params['pw'].sharding,params_layout['occ'],pot_layout),out_shardings=rep)(params['pw'],params['occ'],pot_nl_arg)
    elif kpool_mesh_obj is not None and os.environ.get("D3FT_FUSE_FINAL_DENSITY") == "1":
'''
    s=replace_once(s,anchor,insertion)
    s=replace_once(s,'    return GroundStateOutput(\n','    result = GroundStateOutput(\n')
    s+='''
    if compact:
        from ..optim.stiefel import project_tangent
        def checkpoint_report(p,o,nl):
            value,(gp,go)=jax.value_and_grad(lambda x,y:total_energy(x,y,eps_t,nl),argnums=(0,1))(p,o)
            residual=jnp.max(jnp.abs(project_tangent(p,jnp.conj(gp))))
            leaves=jax.tree_util.tree_leaves(go)
            if leaves:
                residual=jnp.maximum(residual,jnp.max(jnp.stack([jnp.max(jnp.abs(x)) for x in leaves])))
            return value+ew,residual
        compiled=jax.jit(checkpoint_report,
            in_shardings=(params['pw'].sharding,params_layout['occ'],pot_layout),out_shardings=(rep,rep))
        result.compact_report=lambda:compiled(params['pw'],params['occ'],pot_nl_arg)
    return result
'''
    gs.write_text(s)
    driver=runtime/'validation/workflows/response_design/generic_point.py';s=driver.read_text()
    s=replace_once(s,"        config.d3ft.optim.lr_schedule='cosine'\n",
                  "        config.d3ft.optim.lr_schedule='constant' if args.learning_rate==0 else 'cosine'\n")
    s=replace_once(s,'    args = ap.parse_args()\n',"    ap.add_argument('--initialize-kmesh-from',type=Path)\n    args = ap.parse_args()\n")
    anchor='        state=mask=None\n'
    insertion='''        state=mask=None
        if args.initialize_kmesh_from:
            if args.resume_from: raise ValueError('Use initial mesh mapping or exact restart, not both')
            parent=args.initialize_kmesh_from
            prior=json.loads((parent/'metadata.json').read_text())
            manifest=json.loads((ROOT.parent/'runtime_patch.json').read_text())
            for path,h in prior['code_sha256'].items():
                expected=manifest['original_core'][path]
                if h!=expected: raise ValueError('Mesh initialization source has different original core: '+path)
            differences=[name for name in physical if physical[name]!=prior['physical'][name]]
            if not set(differences)<={'k'}: raise ValueError('Mesh mapping changed more than k sampling: '+str(differences))
            if prior['state_sha256']!=sha(parent/'state.pkl'): raise ValueError('Initial state hash mismatch')
            mask=np.load(parent/'mask.npy')
            if mask.shape!=candidate_mask.shape: raise ValueError('Initial mask FFT differs')
            from d3ft.grid.kpoints import k_mesh_fractional
            target=k_mesh_fractional(args.k,'gamma')
            old=np.asarray(json.loads((parent/'point.json').read_text())['k_fractional'])
            # Use actual representatives, not periodic shifts: fixed-|G| basis
            # energies are not invariant under k->k+G at finite cutoff.
            nearest=np.argmin(np.linalg.norm(target[:,None,:]-old[None,:,:],axis=-1),axis=1)
            with (parent/'state.pkl').open('rb') as handle: previous=pickle.load(handle)
            def map_k(x):
                x=np.asarray(x)
                return x[:,nearest,...] if x.ndim>1 and x.shape[1]==len(old) else x
            if prior['config']['occupation']!='uniform':
                raise ValueError('Mesh initialization supports fixed uniform occupations only')
            parent_occ=np.asarray(previous['occ'])
            if parent_occ.shape!=(1,len(old),48) or not np.allclose(parent_occ,2./len(old),rtol=0,atol=1e-14):
                raise ValueError('Parent is not the prescribed 96-electron uniform state')
            # Uniform occupation parameters already include the k-point weight.
            # Repeating 2/64 at 216 points gives 324 electrons. Rebuild at 2/216;
            # only the orbitals are nearest-neighbour mapped from the k4 state.
            state=dict(pw=map_k(previous['pw']),
                       occ=np.full((1,len(target),48),2./len(target),dtype=parent_occ.dtype))
            if abs(float(np.sum(state['occ']))-96)>1e-10:
                raise ValueError('Mapped occupation charge gate failed before optimization')
            metadata['mesh_initialization']=dict(parent=str(parent),parent_state_sha256=prior['state_sha256'],
                nearest_parent_indices=nearest.tolist(),fixed_mask_sha256=sha(parent/'mask.npy'),
                parent_occupation_sum=float(parent_occ.sum()),
                target_occupation_sum=float(state['occ'].sum()),
                occupation_convention='Uniform restricted occupations include 1/Nk; rebuild 2/Nk on target mesh',
                scope='Approximate initial orbitals only; k6 is independently reoptimized and gated')
            dump(args.output/'metadata.json',metadata)
'''
    s=replace_once(s,anchor,insertion)
    anchor="            if args.local_radial_quadrature=='simpson':\n                from radial_quadrature import local_simpson_patch\n"
    s=replace_once(s,anchor,"            from bounded_xc import memory_bounded_xc\n            stack.enter_context(memory_bounded_xc())\n"+anchor)
    anchor='            if kpool or args.fused_report_all_devices:\n'
    s=replace_once(s,anchor,"            compact = os.environ.get('D3FT_COMPACT_K_TERMS')=='1'\n            if compact:\n                checkpoint_energy,r=out.compact_report()\n                rep=dict(variational_residual=float(r),pressure_GPa=None,stress=None)\n            elif kpool or args.fused_report_all_devices:\n")
    s=replace_once(s,"            stress=np.asarray(rep['stress'])\n","            stress=None if compact else np.asarray(rep['stress'])\n")
    s=replace_once(s,"pressure_GPa=float(rep['pressure_GPa']),","pressure_GPa=None if compact else float(rep['pressure_GPa']),")
    s=replace_once(s,'stress_Ha_Bohr3=stress.tolist(),','stress_Ha_Bohr3=None if compact else stress.tolist(),')
    anchor='            if not args.skip_forces:\n'
    s=replace_once(s,anchor,"            if compact:\n                result['checkpoint_energy_Ha']=float(checkpoint_energy)\n                result['stationarity_convention']='Maximum absolute conjugated Stiefel-projected gradient of identical discrete energy'\n                result['stress_and_forces_evaluated']=False\n                overlaps=np.einsum('skgi,skgj->skij',np.conj(host_state['pw']),host_state['pw'])\n                result['orthogonality_max_error']=float(np.max(np.abs(overlaps-np.eye(overlaps.shape[-1]))))\n                if result['orthogonality_max_error']>1e-8: raise ValueError('Orthogonality gate failed')\n"+anchor)
    driver.write_text(s)
    # Imports remain adjacent to the immutable drivers.
    shutil.copy2(P/'validation/workflows/response_design/reverse_response/bounded_xc.py',driver.parent/'bounded_xc.py')
    shutil.copy2(__file__,K/'controller.py')
    shutil.copy2(P/'validation/workflows/response_design/numerical_spectrum_worker.py',K/'drivers/spectrum_worker.py')
    original={str(f.relative_to(F)):sha(f) for f in (F/'d3ft').rglob('*.py')}
    dump(K/'runtime_patch.json',dict(original_core=original,original_ground_state_sha256=sha(F/'d3ft/calc/ground_state.py'),
         new_ground_state_sha256=sha(gs),compact_kernel_sha256=sha(kernel),
         scope='Exact bounded FFT accumulation; original live and frozen production cores untouched. No stresses/forces claimed for new k6 rows.'))
    base=read(P/'validation/results/response_design_20261004_reverse_endpoint_k4/ground_command.json')
    b,e=base.index('env')+1,base.index('/bin/bash');settings=dict(x.split('=',1) for x in base[b:e])
    query=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=32']
    allowed=json.loads(subprocess.check_output(query+['/usr/bin/python3','-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True))
    production=read(P/'validation/results/response_design_20261005_band_structure/path_protocol.json')
    states={key:production['states'][key] for key in ['start','reverse']}
    cfg=dict(runtime=str(runtime),python=base[e+2],settings=settings,query=query,allowed_cpus=allowed,
      physical_cpus=[x for x in allowed if x<64],states=states,epochs=3000,
      gap_resolution_budget_eV=.005,gain_resolution_budget_eV=.005,
      original_gamma_target_eV=read(P/'validation/results/response_design_20261004_response_design/protocol.json')['target_gamma_eV'],
      original_target_tolerance_eV=.001,jobid=250285,
      scope='Same accepted geometries and 55913-vector Miller basis: own k6 stationary densities, all216 spectrum points. Pairwise robustness, not continuum convergence or a multi-k reverse update.')
    dump(K/'protocol.json',cfg)
    dump(K/'status.json',dict(stage='prepared',time=datetime.datetime.now().isoformat()))
    dump(R/'kmesh_active.json',dict(root=str(K),prior_failure='v2 repeated k4 occupations 2/64 on216 points, yielding324 electrons. v3 rebuilds uniform weights2/216 and verifies charge before optimization; wrong-charge state is not reused.'))
    print(K)

def execute():
    cfg=read(K/'protocol.json');runtime=Path(cfg['runtime'])
    def stage(label,**kw):
        d=dict(stage=label,time=datetime.datetime.now().isoformat(),pid=os.getpid(),**kw)
        dump(K/'status.json',d);print('NUMERICAL_KMESH_STAGE',json.dumps(d),flush=True)
    def prefix(tag,gpus,compact=True):
        tmp='/dev/shm/d3ft_response_design_numcheck_'+tag
        env=dict(cfg['settings']);env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),TMPDIR=tmp,
          PYTHONPYCACHEPREFIX=tmp+'/python_cache',JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',
          PYTHONPATH=str(runtime)+':'+env['D3FT_REFERENCE_ROOT']+'/software/jrystal',
          JAX_ENABLE_COMPILATION_CACHE='false',OMP_NUM_THREADS='2')
        if compact: env['D3FT_COMPACT_K_TERMS']='1'
        cpus=cfg['physical_cpus'] if len(gpus)>1 else cfg['physical_cpus'][gpus[0]::8]
        affinity=','.join(str(x) for c in cpus for x in [c,c+64] if x in cfg['allowed_cpus'])
        return cfg['query']+['--gres=gpu:8','--gpu-bind=none','taskset','-c',affinity,
          'timeout','--signal=TERM','--kill-after=30s','57600s','env']+[
          k+'='+v for k,v in env.items()]+['/bin/bash',str(R/'private_tmp.sh'),cfg['python'],'-u']
    def run(command,tag):
        dump(K/'commands'/(tag+'.json'),command)
        with (K/'logs'/(tag+'.log')).open('xb') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        dump(K/'commands'/(tag+'_exit.json'),dict(returncode=code))
        if code: raise RuntimeError(f'{tag}: RC{code}; retained log')
    def point(name,parent,k,epochs,lr,initial=True,require=True):
        m=read(Path(parent)/'metadata.json');a=m['arguments'];out=K/'runs'/name
        command=prefix(name,list(range(8)))+[str(runtime/'validation/workflows/response_design/generic_point.py'),
           '--structure',a['structure'],'--pp-dir',a['pp_dir'],'--strain',a['strain'],
           '--cutoff','100','--grid','144','--k',str(k),'--epochs',str(epochs),
           '--xc-grid-factor','4','--local-radial-quadrature','simpson','--skip-forces',
           '--output',str(out),'--learning-rate',str(lr),'--final-learning-rate',str(min(lr,1e-6)),
           '--initialize-kmesh-from' if initial else '--resume-from',str(parent)]
        if require: command.append('--require-stationary')
        run(command,name)
        p=read(out/'point.json')
        if sha(out/'mask.npy')!=sha(Path(parent)/'mask.npy'):raise ValueError('Miller mask changed')
        if abs(p['integrated_valence_density']-96)>.00096:raise ValueError('Density charge gate failed')
        return out,p
    try:
        stage('formula_parity')
        env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        code="import sys,json;sys.path.insert(0,"+repr(str(runtime/'d3ft/calc'))+");from compact_k_terms import parity_check;print(json.dumps(parity_check()))"
        with (K/'logs/formula_parity.log').open('xb') as log:
            proc=subprocess.run([cfg['python'],'-c',code],env=env,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError('Compact formula parity failed')
        text=(K/'logs/formula_parity.log').read_text();report=json.loads(text[text.rfind('\n{')+1:] if '\n{' in text else text)
        dump(K/'formula_parity.json',report)
        stage('physical_k4_checkpoint_parity')
        old=Path(cfg['states']['start']['source'])
        out,p=point('k4_checkpoint_parity',old,4,1,0.,require=True)
        oldpoint=read(old/'point.json')
        density_error=float(np.max(np.abs(np.load(out/'density.npy')-np.load(old/'density.npy'))))
        energy_error=abs(p['checkpoint_energy_Ha']-oldpoint['energy_Ha'])
        parity=dict(density_max_error=density_error,energy_difference_Ha=energy_error,
          original_residual=oldpoint['residual'],compact_residual=p['residual'],
          density_tolerance=1e-10,energy_tolerance_Ha=1e-7,
          passed=density_error<=1e-10 and energy_error<=1e-7 and p['stationarity_pass'])
        dump(K/'physical_parity.json',parity)
        if not parity['passed']: raise RuntimeError('Physical production checkpoint parity failed')
        stage('k6_initial_charge_preflight')
        out,p=point('k6_initial_charge_preflight',old,6,1,0.,require=False)
        preflight=dict(passed=abs(p['integrated_valence_density']-96)<=.00096,
            integrated_valence_density=p['integrated_valence_density'],
            orthogonality_max_error=p['orthogonality_max_error'],
            scope='Zero-step mapped k4 orbitals with new uniform k6 weights; not stationary or an admitted scientific result')
        dump(K/'k6_initial_charge_preflight.json',preflight)
        if not preflight['passed']:raise RuntimeError('New k6 density charge failed before long optimization')
        for name in ['start','reverse']:
            state=cfg['states'][name];stage('k6_electronic_state',state=name,total_k_points=216)
            out,p=point(name+'_k6',state['source'],6,cfg['epochs'],.001,require=False)
            if not p['stationarity_pass']:
                stage('k6_stationarity_extension',state=name,previous_residual=p['residual'])
                out,p=point(name+'_k6_extension',out,6,cfg['epochs'],.001,initial=False,require=True)
            dump(K/(name+'_accepted_source.json'),dict(source=str(out),point_sha256=sha(out/'point.json')))
            stage('k6_spectra',state=name,total=216)
            protocol=dict(states={name:dict(source=str(out),source_sha256={n:sha(out/n) for n in
                ['point.json','metadata.json','mask.npy','density.npy','input.xyz']})},k_fractional=p['k_fractional'],occupied_bands=48,empty_bands=2)
            protocol_path=K/(name+'_spectrum_protocol.json');dump(protocol_path,protocol)
            def spectrum(gpu):
                indices=list(range(gpu,216,8));output=K/'spectra'/(name+'_gpu'+str(gpu))
                command=prefix(name+'_spectrum_gpu'+str(gpu),[gpu],False)+[
                    str(K/'drivers/spectrum_worker.py'),str(protocol_path),name,str(output),json.dumps(indices)]
                run(command,name+'_spectrum_gpu'+str(gpu));return output
            with ThreadPoolExecutor(max_workers=8) as pool: outputs=list(pool.map(spectrum,range(8)))
            rows=[read(f) for output in outputs for f in output.glob('k_*.json')]
            if sorted(x['index'] for x in rows)!=list(range(216)):raise ValueError('Incomplete k6 spectra')
            for row in rows:
                if max(row['residuals_Ha'])>1e-7:raise ValueError('Spectrum residual gate failed')
                if not np.allclose(row['k_frac'],p['k_fractional'][row['index']],rtol=0,atol=1e-10):raise ValueError('Spectrum mesh mismatch')
            v=max(rows,key=lambda x:x['vbm_Ha']);c=min(rows,key=lambda x:x['cbm_Ha'])
            gamma=next(x for x in rows if max(abs(y) for y in x['k_frac'])<1e-10)
            spectrum_result=dict(source=str(out),points=sorted(rows,key=lambda x:x['index']),mesh=[6]*3,
                gap_eV=(c['cbm_Ha']-v['vbm_Ha'])*27.211386245988,
                gamma_gap_eV=(gamma['cbm_Ha']-gamma['vbm_Ha'])*27.211386245988,
                VBM_fractional=v['k_frac'],CBM_fractional=c['k_frac'],full_registered_mesh=True,
                max_eigen_residual_Ha=max(max(r['residuals_Ha']) for r in rows),
                source_hashes=protocol['states'][name]['source_sha256'])
            dump(K/(name+'_spectrum_complete.json'),spectrum_result)
        old={n:read(s['spectrum']) for n,s in cfg['states'].items()}
        new={n:read(K/(n+'_spectrum_complete.json')) for n in ['start','reverse']}
        oldgain=old['reverse']['gap_eV']-old['start']['gap_eV'];newgain=new['reverse']['gap_eV']-new['start']['gap_eV']
        checks=dict(gain_positive=newgain>0,gain_pairwise_stable=abs(newgain-oldgain)<=cfg['gain_resolution_budget_eV'],
            start_gap_pairwise_stable=abs(new['start']['gap_eV']-old['start']['gap_eV'])<=cfg['gap_resolution_budget_eV'],
            reverse_gap_pairwise_stable=abs(new['reverse']['gap_eV']-old['reverse']['gap_eV'])<=cfg['gap_resolution_budget_eV'])
        target_error=new['reverse']['gamma_gap_eV']-cfg['original_gamma_target_eV']
        report=dict(completed=True,passed=all(checks.values()),checks=checks,k4=old,k6=new,
            k4_gain_eV=oldgain,k6_gain_eV=newgain,gain_setting_difference_eV=newgain-oldgain,
            original_target_error_eV=target_error,original_target_pass=abs(target_error)<=cfg['original_target_tolerance_eV'],
            original_target_tolerance_eV=cfg['original_target_tolerance_eV'],
            scope=cfg['scope'],time=datetime.datetime.now().isoformat())
        dump(K/'result.json',report);stage('completed',passed=report['passed'],original_target_pass=report['original_target_pass'])
    except Exception as exc:stage('failed',error=str(exc));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    prepare() if a.prepare else execute()

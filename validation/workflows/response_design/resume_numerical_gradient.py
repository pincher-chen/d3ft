"""Resume only the XC720 adjoint after the five accepted primal points.

Completed points/orbitals remain immutable; no electronic optimization is rerun.
The pointwise XC memory context also covers geometry reconstruction, matching
the already successful orbital reconstruction. Numerical budgets are unchanged.
"""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

P = Path('${D3FT_ROOT}')
BASE = P/'validation/results/response_design_20261005_numerical_verification'
PREVIOUS = BASE/'gradient_v3'
DEST = BASE/'gradient_v4'

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, data):
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def replace_once(text,old,new):
    if text.count(old)!=1: raise RuntimeError('Recovery source anchor changed: '+old[:70])
    return text.replace(old,new,1)

def prepare():
    if read(PREVIOUS/'gradient_status.json')['stage']!='failed':
        raise RuntimeError('Previous campaign is not terminal')
    oldcfg=read(PREVIOUS/'gradient_protocol.json')
    names=['fine_reference','plus_h','minus_h','plus_half','minus_half']
    hashes={}
    checks=[]
    for path,h in {**oldcfg['parent_hashes'],**oldcfg['source_hashes'],**oldcfg['recovery_source_hashes']}.items():
        if sha(Path(path))!=h: raise RuntimeError('Prior input changed: '+path)
    production=read(Path(oldcfg['source'])/'metadata.json')['code_sha256']
    parent_mask=sha(Path(oldcfg['source'])/'mask.npy')
    for name in names:
        src=PREVIOUS/'runs'/name;orb=PREVIOUS/'orbitals'/name
        point=read(src/'point.json');spectrum=read(orb/'summary.json')
        if not point['stationarity_pass'] or point['residual']>1e-6:
            raise RuntimeError('Stationarity failure: '+name)
        if point['acoustic_sum_rule_Ha_Bohr']>5e-8 or abs(point['integrated_valence_density']-96)>.00096:
            raise RuntimeError('ASR/charge failure: '+name)
        if read(src/'metadata.json')['code_sha256']!=production or sha(src/'mask.npy')!=parent_mask:
            raise RuntimeError('Primal core/mask changed: '+name)
        if max(spectrum['residuals_Ha'])>1e-7 or spectrum['orthogonality_error']>1e-8:
            raise RuntimeError('Eigenpair failure: '+name)
        if sha(orb/'orbitals.npy')!=spectrum['saved_orbitals_sha256']:
            raise RuntimeError('Orbital hash failure: '+name)
        for filename,h in read(orb/'metadata.json')['source_hashes'].items():
            if sha(src/filename)!=h:raise RuntimeError('Orbital source binding changed')
        for folder in [src,orb]:
            for f in folder.iterdir():
                if f.is_file():hashes[str(f)]=sha(f)
        checks.append(dict(name=name,residual=point['residual'],ASR_Ha_Bohr=point['acoustic_sum_rule_Ha_Bohr'],
                           electrons=point['integrated_valence_density'],gap_eV=spectrum['gap_eV'],passed=True))
    DEST.mkdir(exist_ok=False)
    for name in ['runs','orbitals','response','logs','commands']: (DEST/name).mkdir()
    shutil.copytree(PREVIOUS/'drivers',DEST/'drivers')
    shutil.copy2(PREVIOUS/'private_tmp.sh',DEST/'private_tmp.sh')
    for name in names:
        for group in ['runs','orbitals']:(DEST/group/name).symlink_to(PREVIOUS/group/name,target_is_directory=True)
    driver=DEST/'drivers/gamma_response.py';text=driver.read_text()
    text=replace_once(text,'import argparse,datetime,hashlib,json,os,sys,time',
                      'import argparse,datetime,hashlib,json,os,sys,time,gc')
    text=replace_once(text,'with local_simpson_patch(),plain_fft_patch(k_pool=False):',
                      'with local_simpson_patch(),plain_fft_patch(k_pool=False),memory_bounded_xc():')
    text=replace_once(text,'            with validated_radial_splines(), memory_bounded_xc():',
                      '            jax.block_until_ready(out.density_grid)\n            jax.clear_caches();gc.collect()\n            with validated_radial_splines():')
    ast.parse(text);driver.write_text(text)
    source=(P/'validation/workflows/response_design/numerical_gradient_verification.py').read_text()
    controller=source[:source.index('def prepare():')]+source[source.index('def execute():'):source.index("if __name__ == '__main__':")]
    controller=replace_once(controller,"R = P / 'validation/results/response_design_20261005_numerical_verification/gradient_v3'",
                            "R = P / 'validation/results/response_design_20261005_numerical_verification/gradient_v4'")
    start=controller.index('        theta = np.asarray(cfg[\'theta0\'])')
    end=controller.index("        stage('fine_reverse_response'",start)
    replacement='''        theta=np.asarray(cfg['theta0']);direction=np.asarray(cfg['direction'])
        fine=R/'runs/fine_reference';orb=R/'orbitals/fine_reference'
        fd_orbitals={name:R/'orbitals'/name for name in ['plus_h','minus_h','plus_half','minus_half']}
        response_cfg=dict(cfg,source=str(fine),orbitals=str(orb))
        dump(R/'fine_response_protocol.json',response_cfg)
        stage('reused_qualified_primals',points=5,electronic_steps_repeated=0)
'''
    controller=controller[:start]+replacement+controller[end:]
    controller=controller.replace("'fine_reverse')","'fine_reverse_v4')")
    controller=controller.replace("prefix(0, 'fine_reverse')","prefix(0, 'fine_reverse_v4')")
    controller+="if __name__=='__main__':execute()\n"
    ast.parse(controller);(DEST/'gradient_controller.py').write_text(controller)
    cfg=dict(oldcfg,root=str(DEST),
             source_hashes={**oldcfg['source_hashes'],**{str(f):sha(f) for f in (DEST/'drivers').glob('*.py')}},
             recovery_source_hashes={**oldcfg['recovery_source_hashes'],**hashes},
             recovery=dict(previous=str(PREVIOUS),reason='Response geometry reconstruction omitted pointwise XC chunking;93.31GiB request. Apply same context as successful orbital reconstruction; release compiled update before response.',
                           electronic_steps_repeated=0,qualified_primal_count=5))
    cfg['source_hashes'][str(DEST/'gradient_controller.py')]=sha(DEST/'gradient_controller.py')
    cfg['source_hashes'][str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    dump(DEST/'gradient_protocol.json',cfg)
    dump(DEST/'reused_primal_audit.json',dict(passed=True,checks=checks,hashes=hashes,time=datetime.datetime.now().isoformat()))
    dump(DEST/'gradient_status.json',dict(stage='prepared',time=datetime.datetime.now().isoformat()))
    dump(BASE/'gradient_active.json',dict(root=str(DEST),prior_failure='v3 completed all5 primal points/orbitals; response reconstruction omitted bounded pointwise XC context and requested93.31GiB. v4 reuses accepted data; no optimization repeated.'))
    print(DEST)

if __name__=='__main__':prepare()

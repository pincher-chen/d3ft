"""Recover only the XC720 adjoint using a staged exact Fourier/local AD chain."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

P=Path('${D3FT_ROOT}')
BASE=P/'validation/results/response_design_20261005_numerical_verification'
PREVIOUS=BASE/'gradient_v6'
DEST=BASE/'gradient_v7'
HELPERS=P/'validation/workflows/response_design/reverse_response'

def read(p):return json.loads(p.read_text())
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def dump(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def replace(source,old,new):
    if source.count(old)!=1:raise RuntimeError('Frozen response source anchor changed: '+old[:70])
    return source.replace(old,new,1)

def prepare():
    if read(PREVIOUS/'gradient_status.json')['stage']!='failed':raise RuntimeError('Previous campaign not terminal')
    parity=read(BASE/'staged_xc_geometry_parity.json')
    if not parity['passed']:raise RuntimeError('Independent staged XC parity not passed')
    if parity['helper_sha256']!=sha(HELPERS/'staged_xc_geometry.py'):raise RuntimeError('Parity helper changed')
    cfg=read(PREVIOUS/'gradient_protocol.json')
    for path,h in {**cfg['parent_hashes'],**cfg['source_hashes'],**cfg['recovery_source_hashes']}.items():
        if sha(path)!=h:raise RuntimeError('Immutable input changed: '+path)
    audit=read(PREVIOUS/'reused_primal_audit.json')
    if not audit['passed']:raise RuntimeError('Primal reuse audit failed')
    for path,h in audit['hashes'].items():
        if sha(path)!=h:raise RuntimeError('Qualified primal input changed: '+path)
    DEST.mkdir(exist_ok=False)
    for name in ['runs','orbitals','response','logs','commands']:(DEST/name).mkdir()
    shutil.copytree(PREVIOUS/'drivers',DEST/'drivers')
    shutil.copy2(PREVIOUS/'private_tmp.sh',DEST/'private_tmp.sh')
    for name in ['staged_xc_geometry.py','staged_scalar_gradient.py']:
        shutil.copy2(HELPERS/name,DEST/'drivers'/name)
    for name in ['fine_reference','plus_h','minus_h','plus_half','minus_half']:
        for group in ['runs','orbitals']:(DEST/group/name).symlink_to(PREVIOUS/group/name,target_is_directory=True)
    driver=DEST/'drivers/gamma_response.py';source=driver.read_text()
    source=replace(source,'        inverse0=jnp.linalg.inv(jnp.eye(3)+strain_matrix(jnp.asarray(theta0)))',
        '        inverse0=jnp.linalg.inv(jnp.eye(3)+strain_matrix(jnp.asarray(theta0)))\n        self.geometry_inverse0=inverse0')
    old='''            def ex(r):return xc_energy_tunable(r[None,...],g,volume,out.config.xc,
                    xc_params=self.xc_params,grid_factor=self.xc_factor)
            xc_direction=jax.jvp(ex,(rho,),(weight_r/volume,))[1]
            return kinetic+locality+nonlocal_term+xc_direction'''
    source=replace(source,old,'            return kinetic+locality+nonlocal_term')
    source=replace(source,'        self.scalar_gradient=jax.jit(jax.grad(scalar_geometry,argnums=0))',
        '        from staged_scalar_gradient import staged_scalar_gradient\n        self.scalar_gradient=staged_scalar_gradient(self,scalar_geometry)')
    source=replace(source,'original scalar mixed-derivative reverse pullback; compressed reciprocal basis; one electronic adjoint',
        'staged exact mixed XC AD through Fourier/local chain; unchanged non-XC pullback; one electronic adjoint')
    ast.parse(source);driver.write_text(source)
    controller=(PREVIOUS/'gradient_controller.py').read_text().replace('numerical_verification/gradient_v6','numerical_verification/gradient_v7').replace('fine_reverse_v6','fine_reverse_v7')
    ast.parse(controller);(DEST/'gradient_controller.py').write_text(controller)
    cfg.update(root=str(DEST),source_hashes={**cfg['source_hashes'],
        **{str(f):sha(f) for f in (DEST/'drivers').glob('*.py')},
        **{str(HELPERS/n):sha(HELPERS/n) for n in ['staged_xc_geometry.py','staged_scalar_gradient.py']},
        str(Path(__file__).resolve()):sha(__file__),str(DEST/'gradient_controller.py'):sha(DEST/'gradient_controller.py')},
        recovery=dict(previous=str(PREVIOUS),electronic_steps_repeated=0,qualified_primal_count=5,
            reason='XC720 whole-grid mixed geometry AD requested80.65GiB in v6. Stage the exact linear Fourier interpolation/spatial derivatives, accumulate local mixed AD in bounded chunks, and pull back only volume ratio and reciprocal3x3 deformation. One electronic adjoint; unchanged PBE/grid/basis/budgets.',
            staged_xc_geometry_parity=parity))
    dump(DEST/'gradient_protocol.json',cfg);dump(DEST/'reused_primal_audit.json',audit)
    dump(DEST/'staged_xc_geometry_parity.json',parity)
    dump(DEST/'gradient_status.json',dict(stage='prepared',time=datetime.datetime.now().isoformat()))
    dump(BASE/'gradient_active.json',dict(root=str(DEST),prior_failure='v6 XC720 geometry derivative OOM80.65GiB; v7 staged exact Fourier/local AD; five qualified primal/orbital states reused.'))
    print(DEST)

if __name__=='__main__':prepare()

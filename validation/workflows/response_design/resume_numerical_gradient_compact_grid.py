"""Resume XC720 with a compressed reciprocal grid and the original pullback."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil

P=Path('${D3FT_ROOT}')
BASE=P/'validation/results/response_design_20261005_numerical_verification'
PREVIOUS=BASE/'gradient_v5'
DEST=BASE/'gradient_v6'
HELPER=P/'validation/workflows/response_design/reverse_response/compact_reciprocal_grid.py'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def prepare():
    if read(PREVIOUS/'gradient_status.json')['stage']!='failed':raise RuntimeError('Prior campaign still active')
    cfg=read(PREVIOUS/'gradient_protocol.json')
    for path,h in {**cfg['parent_hashes'],**cfg['source_hashes'],**cfg['recovery_source_hashes']}.items():
        if sha(Path(path))!=h:raise RuntimeError('Recovery input changed: '+path)
    text=(BASE/'compact_reciprocal_grid_parity.log').read_text();offset=text.rfind('\n{')
    parity=json.loads(text[offset+1:] if offset>=0 else text)
    if not parity['passed']:raise RuntimeError('Compact reciprocal-grid parity failed')
    audit=read(PREVIOUS/'reused_primal_audit.json')
    if not audit['passed']:raise RuntimeError('Primal reuse audit failed')
    for path,h in audit['hashes'].items():
        if sha(Path(path))!=h:raise RuntimeError('Certified primal changed')
    DEST.mkdir(exist_ok=False)
    for name in ['runs','orbitals','response','logs','commands']:(DEST/name).mkdir()
    shutil.copytree(PREVIOUS/'drivers',DEST/'drivers')
    shutil.copy2(PREVIOUS/'private_tmp.sh',DEST/'private_tmp.sh');shutil.copy2(HELPER,DEST/'drivers'/HELPER.name)
    for name in ['fine_reference','plus_h','minus_h','plus_half','minus_half']:
        for group in ['runs','orbitals']:(DEST/group/name).symlink_to(PREVIOUS/group/name,target_is_directory=True)
    driver=DEST/'drivers/gamma_response.py';source=driver.read_text()
    anchor='        from sequential_scalar_gradient import sequential_scalar_gradient\n        self.scalar_gradient=sequential_scalar_gradient(scalar_geometry)'
    if source.count(anchor)!=1:raise RuntimeError('Geometry gradient anchor changed')
    source=source.replace(anchor,'        self.scalar_gradient=jax.jit(jax.grad(scalar_geometry,argnums=0))',1)
    source=source.replace('six sequential exact AD JVPs of the same scalar mixed energy; one electronic adjoint',
                          'original scalar mixed-derivative reverse pullback; compressed reciprocal basis; one electronic adjoint')
    ast.parse(source);driver.write_text(source)
    runner=DEST/'drivers/sigma_runner.py';source=runner.read_text()
    source=source.replace('from bounded_sigma_fft import bounded_sigma_fft',
                         'from bounded_sigma_fft import bounded_sigma_fft\nfrom compact_reciprocal_grid import compact_reciprocal_grid')
    source=source.replace('with bounded_sigma_fft():','with bounded_sigma_fft(),compact_reciprocal_grid():')
    ast.parse(source);runner.write_text(source)
    controller=(PREVIOUS/'gradient_controller.py').read_text().replace('numerical_verification/gradient_v5','numerical_verification/gradient_v6').replace('fine_reverse_v5','fine_reverse_v6')
    ast.parse(controller);(DEST/'gradient_controller.py').write_text(controller)
    cfg.update(root=str(DEST),source_hashes={**cfg['source_hashes'],
        **{str(f):sha(f) for f in (DEST/'drivers').glob('*.py')},
        str(HELPER):sha(HELPER),str(Path(__file__).resolve()):sha(Path(__file__).resolve()),
        str(DEST/'gradient_controller.py'):sha(DEST/'gradient_controller.py')},
        recovery=dict(previous=str(PREVIOUS),electronic_steps_repeated=0,qualified_primal_count=5,
        reason='v4 geometry VJP81.56GiB and v5 geometry JVP88.99GiB. Keep original VJP and represent fine G with1D frequencies+3x3 reciprocal basis, constructing one Cartesian component inside FFT scan. No grid/functional/budget changes.',
        compact_reciprocal_grid_parity=parity))
    dump(DEST/'gradient_protocol.json',cfg);dump(DEST/'reused_primal_audit.json',audit)
    dump(DEST/'compact_reciprocal_grid_parity.json',parity)
    dump(DEST/'gradient_status.json',dict(stage='prepared',time=datetime.datetime.now().isoformat()))
    dump(BASE/'gradient_active.json',dict(root=str(DEST),prior_failure='Geometry derivatives exceeded80GB in v4/v5; v6 compresses exact fine-grid reciprocal vectors and restores original scalar VJP. Five accepted primal states/orbitals reused.'))
    print(DEST)

if __name__=='__main__':prepare()

"""Compare an unmodified-PBE displacement sample with an identical QE input.

Reports raw force/energy/pressure differences, not automatic cutoff convergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from force_acceptance import check_force_sum


def matched_inputs(dft,qe):
    dm=json.loads((dft/'metadata.json').read_text())
    qm=json.loads((qe/'metadata.json').read_text())
    if dm['arguments']['mode']!='sample' or qm['arguments']['mode']!='force':
        raise ValueError('Expected D3FT sample and QE force modes')
    if dm['arguments']['material']!=qm['arguments']['material']:
        raise ValueError('Material mismatch')
    if np.any(np.asarray(dm['parameter_vector'])!=0):
        raise ValueError('Requires unmodified PBE parameters')
    if dm['config']['xc']!='gga_x_pbe+gga_c_pbe':raise ValueError('Requires PBE')
    # Worker keys are Symbol.upf paths; QE metadata keys are element symbols.
    dpp={Path(k).stem:v for k,v in dm['pseudopotentials'].items()}
    if dpp!=qm['pseudopotentials']:raise ValueError('Element-specific UPF bytes differ')
    qinput=qe/'displaced/scf.in';text=qinput.read_text()
    if not re.search(r"input_dft\s*=\s*['\"]pbe['\"]",text,re.I):
        raise ValueError('QE input must explicitly select PBE')
    if dm['config']['spin']!=0 or re.search(r'nspin\s*=\s*[24]',text,re.I):
        raise ValueError('Requires restricted spin in both calculations')
    cutoff=float(re.search(r'ecutwfc\s*=\s*([\d.]+)',text,re.I)[1])
    mesh=list(map(int,re.search(r'K_POINTS automatic\s*\n([^\n]+)',text)[1].split()))
    if cutoff!=2*dm['config']['cutoff_energy'] or mesh!=[dm['config']['k_grid_sizes']]*3+[0]*3:
        raise ValueError('Cutoff or k mesh mismatch')
    lines=(dft/'sample.xyz').read_text().splitlines();nat=int(lines[0])
    cell=np.array(list(map(float,re.search(r'Lattice="([^"]+)"',lines[1])[1].split()))).reshape(3,3)
    atoms=[line.split() for line in lines[2:2+nat]]
    qlines=text.splitlines();ci=qlines.index('CELL_PARAMETERS angstrom');ai=qlines.index('ATOMIC_POSITIONS angstrom')
    qcell=np.array([list(map(float,x.split())) for x in qlines[ci+1:ci+4]])
    qat=[x.split() for x in qlines[ai+1:ai+1+nat]]
    if [a[0] for a in atoms]!=[a[0] for a in qat]:raise ValueError('Atomic species/order mismatch')
    positions=lambda aa:np.array([[float(v) for v in x[1:4]] for x in aa])
    if not np.allclose(cell,qcell,rtol=0,atol=1e-10) or not np.allclose(positions(atoms),positions(qat),rtol=0,atol=1e-10):
        raise ValueError('Actual geometry mismatch')
    return dm,qm


def compare(dft,qe):
    dm,qm=matched_inputs(dft,qe)
    d=json.loads((dft/'summary.json').read_text());q=json.loads((qe/'summary.json').read_text())
    if not np.isfinite(d['residual']) or d['residual']>1e-6:raise ValueError('Nonstationary sample')
    if not all(np.isfinite(x[key]) for x in [d,q] for key in ['energy_Ha','pressure_GPa']):
        raise ValueError('Nonfinite energy or pressure')
    if q.get('precision_source')!='QE XML in explicit Hartree atomic units':raise ValueError('Requires full-precision QE XML')
    xml=Path(q['xml_source'])
    if hashlib.sha256(xml.read_bytes()).hexdigest()!=q['xml_sha256']:raise ValueError('QE XML changed')
    f=np.asarray(d['forces']);fq=np.asarray(q['forces'])
    if f.shape!=fq.shape or not np.isfinite(fq).all():raise ValueError('Invalid force comparison')
    conservation=check_force_sum(f,floor=dm['arguments'].get('asr_force_floor',0.))
    # This fixed displacement is intentionally not an ionic equilibrium.
    conservation={k:v for k,v in conservation.items() if k not in ['passes_ionic_force','ionic_force_tolerance']}
    error=float(np.abs(f-fq).max());scale=float(np.abs(fq).max())
    return dict(material=dm['arguments']['material'],matched_inputs=True,
        d3ft_forces_Ha_Bohr=f.tolist(),qe_forces_Ha_Bohr=fq.tolist(),
        force_difference_Ha_Bohr=(f-fq).tolist(),max_force_error_Ha_Bohr=error,
        force_error_relative_to_largest_reference=error/scale if scale else None,
        energy_difference_Ha=d['energy_Ha']-q['energy_Ha'],pressure_difference_GPa=d['pressure_GPa']-q['pressure_GPa'],
        electronic_residual=d['residual'],raw_force_conservation=conservation,
        source_hashes={str(p):{n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['metadata.json','summary.json']} for p in [dft,qe]},
        scope='Identical geometry, UPF, PBE, cutoff and k mesh; raw implementation differences, not a numerical-convergence or global-ground-state certificate')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dft',type=Path);ap.add_argument('qe',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();result=compare(a.dft,a.qe)
    result['checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

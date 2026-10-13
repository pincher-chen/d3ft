"""Compare completed baseline elastic results, retaining numerical-setting differences."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from equilibrium_jacobian import derive

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('d3ft',type=Path)
ap.add_argument('qe_elastic',type=Path)
ap.add_argument('--qe-bulk',type=Path)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
derive(args.d3ft)
d=json.loads((args.d3ft/'summary.json').read_text())
dm=json.loads((args.d3ft/'metadata.json').read_text())
if np.any(np.asarray(d['parameter_vector'])!=0): raise ValueError('This comparison requires an unmodified PBE baseline')
h=dm['arguments'].get('strain_step',.01)
material=json.loads(Path(__file__).with_name('materials.json').read_text())[d['material']]
elastic_keys=[('a0_A','a0_A')]
if material['C11'] is not None: elastic_keys.append(('C11_minus_C12_GPa',f'D_eps{h:g}'))
if material['C44'] is not None: elastic_keys.append(('C44_GPa',f'C44_eps{h:g}'))
rows=[]; settings=[]
for folder,keys in [(args.qe_elastic,elastic_keys),
                    (args.qe_bulk,[('B_GPa','B_energy_GPa'),('B_stress_GPa','B_stress_GPa')])]:
    if folder is None: continue
    q=json.loads((folder/'summary.json').read_text()); qm=json.loads((folder/'metadata.json').read_text())
    if q['material']!=d['material']: raise ValueError('Different materials')
    if sorted(qm['pseudopotentials'].values())!=sorted(dm['pseudopotentials'].values()):
        raise ValueError('Different UPF bytes')
    inputs=list(folder.glob('*/scf.in'))
    if not inputs: raise ValueError('Missing QE inputs')
    cutoffs=set(); meshes=set()
    for file in inputs:
        text=file.read_text()
        cutoffs.add(float(re.search(r'ecutwfc\s*=\s*([\d.]+)',text,re.I)[1]))
        meshes.add(tuple(map(int,re.search(r'K_POINTS\s+automatic\s*\n([^\n]+)',text,re.I)[1].split())))
    if len(cutoffs)!=1 or len(meshes)!=1: raise ValueError('QE settings changed within run')
    cutoff=cutoffs.pop();mesh=meshes.pop()
    if mesh!=(dm['config']['k_grid_sizes'],)*3+(0,0,0): raise ValueError('Different k meshes')
    settings.append(dict(source=str(folder),qe_cutoff_Ry=cutoff,d3ft_cutoff_Ry=2*dm['config']['cutoff_energy'],
                         same_cutoff=cutoff==2*dm['config']['cutoff_energy'],
                         d3ft_requested_grid=dm['config']['grid_sizes'],
                         d3ft_actual_fft_grid=list(np.load(args.d3ft/'mask.npy').shape),
                         d3ft_xc_grid_factor=dm['config']['d3ft'].get('xc_grid_factor',1),
                         d3ft_asr_force_floor=dm['arguments'].get('asr_force_floor',0.),
                         qe_a0_A=q['a0_A'],d3ft_a0_A=d['a0_A']))
    for dk,qk in keys:
        if dk not in d or qk not in q: raise ValueError(f'Missing observable {dk}/{qk}')
        rows.append(dict(observable=dk,QE_observable=qk,d3ft=d[dk],QE=q[qk],difference=d[dk]-q[qk],
                         difference_percent=100*(d[dk]/q[qk]-1),QE_source=str(folder)))
result=dict(material=d['material'],settings=settings,rows=rows,
            scope='At respective equilibria; report setting differences, not an automatic convergence certificate')
with args.out.open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))

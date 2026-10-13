"""Rebuild the small-case serial/k-pool observable parity check."""
import argparse
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('serial',type=Path)
ap.add_argument('parallel',type=Path)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
records=[json.loads((p/'summary.json').read_text()) for p in [args.serial,args.parallel]]
meta=[json.loads((p/'metadata.json').read_text()) for p in [args.serial,args.parallel]]
for key in ['parameter_vector','pseudopotentials']:
    if meta[0][key]!=meta[1][key]: raise ValueError(f'Mismatched {key}')
for key in ['grid_sizes','cutoff_energy','k_grid_sizes','empty_bands','epoch','seed','xc']:
    if meta[0]['config'][key]!=meta[1]['config'][key]: raise ValueError(f'Mismatched {key}')
for key in ['a0_A','strain','displacement_Bohr']:
    np.testing.assert_allclose(records[0][key],records[1][key],atol=0,rtol=0)
diff={key:float(np.max(np.abs(np.asarray(records[0][key])-np.asarray(records[1][key]))))
      for key in ['energy_Ha','pressure_GPa','stress','forces','dE_dp','residual']}
shapes=records[1]['wavefunction_local_shapes']
if len(shapes)<2 or sum(s[1] for s in shapes)!=meta[1]['config']['k_grid_sizes']**3:
    raise ValueError('Wavefunction k-point distribution is inconsistent')
result=dict(serial=str(args.serial),parallel=str(args.parallel),differences=diff,
            wavefunction_local_shapes=shapes,passed=bool(max(diff.values())<1e-10),
            scope='Small-case numerical and compact-state sharding parity; no large-grid memory certificate')
with args.out.open('x') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
if not result['passed']: raise SystemExit(1)

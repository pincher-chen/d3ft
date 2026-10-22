#!/usr/bin/env python3
"""Portable, explicit opt-in entry for saved-state Gamma response and design."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'validation/workflows/response_design'
SEED=ROOT/'reproducibility/response_design/seeds/MgSiN2_gamma'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def expand(value,seed):
    if isinstance(value,str):return value.replace('${D3FT_ROOT}',str(ROOT)).replace('${SEED_ROOT}',str(seed))
    if isinstance(value,list):return [expand(v,seed) for v in value]
    if isinstance(value,dict):return {k:expand(v,seed) for k,v in value.items()}
    return value


def materialize_seed(output):
    # Own copies: changing an experiment directory cannot alter release inputs.
    source=output/'seed';shutil.copytree(SEED,source)
    meta=expand(read(source/'metadata.json'),source)
    for item in meta['physical']['pp'].values():
        if sha(item['path'])!=item['sha256']:raise ValueError('Pseudopotential hash mismatch')
    if sha(source/'state.pkl')!=meta['state_sha256']:raise ValueError('State hash mismatch')
    (source/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    orbital=expand(read(source/'orbitals/metadata.json'),source)
    orbital['source_hashes']={n:sha(source/n) for n in orbital['source_hashes']}
    (source/'orbitals/metadata.json').write_text(json.dumps(orbital,indent=2)+'\n')
    p=read(source/'point.json');summary=read(source/'orbitals/summary.json')
    if p['residual']>1e-8 or p['acoustic_sum_rule_Ha_Bohr']>5e-8:
        raise ValueError('Unqualified seed')
    if sha(source/'orbitals/orbitals.npy')!=summary['saved_orbitals_sha256']:
        raise ValueError('Independent eigenspace changed')
    return source


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--list',action='store_true')
    ap.add_argument('--stage',choices=['inspect','reverse','forward','spectrum','design54','design6','design48'],default='inspect')
    ap.add_argument('--execute',action='store_true',help='Explicit opt-in to a real calculation; never requests an allocation')
    ap.add_argument('--output',type=Path,help='New experiment directory; existing paths are refused')
    ap.add_argument('--axes',nargs='*',type=int,help='Forward axes0..53; default all54')
    args=ap.parse_args()
    if args.list:
        print(json.dumps(read(ROOT/'examples/response_design/cases.json'),indent=2));return
    plan={'stage':args.stage,'seed':str(SEED),'inputs':54,'strain_inputs':6,'Cartesian_inputs':48,
          'seed_residual_Ha':read(SEED/'point.json')['residual'],
          'seed_gap_eV':read(SEED/'orbitals/summary.json')['gap_eV'],
          'units':'engineering strain then Cartesian displacement in Bohr',
          'execution_requested':args.execute,'GPU_allocation_submitted':False,
          'note':'Inspection only unless --execute is supplied. Real stages can require hours and substantial device memory.'}
    print(json.dumps(plan,indent=2),flush=True)
    if args.stage=='inspect' or not args.execute:return
    if args.output is None:ap.error('--execute requires a new --output directory')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=materialize_seed(output);calculation=output/'calculation'
    for p in [ROOT,ROOT/'third_party/jrystal_dependency',WORK,WORK/'reverse_response',ROOT/'validation/workflows/phase3']:
        sys.path.insert(0,str(p))
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,ROOT/'third_party/jrystal_dependency',WORK,WORK/'reverse_response',ROOT/'validation/workflows/phase3']))
    if args.stage in ('reverse','forward'):
        cfg={'source':str(source),'orbitals':str(source/'orbitals'),'runtime':str(ROOT),
             'theta0':read(ROOT/'examples/response_design/MgSiN2_gamma/design.json')['theta0'],
             'include_coordinates':True,'displacement_unit':'bohr',
             'source_hashes':{'release_manifest':sha(ROOT/'provenance/RESPONSE_DESIGN_MANIFEST.json')}}
        protocol=output/'response.json';protocol.write_text(json.dumps(cfg,indent=2)+'\n')
        command=[sys.executable,str(WORK/'reverse_response/gamma_response.py'),str(protocol),
                 '--mode',args.stage,'--output',str(calculation)]
        if args.axes is not None:command+=['--axes',*map(str,args.axes)]
        subprocess.run(command,env=env,check=True)
    elif args.stage=='spectrum':
        subprocess.run([sys.executable,str(WORK/'reverse_response/e3_primal_spectrum.py'),
                        '--source',str(source),'--output',str(calculation)],env=env,check=True)
    else:
        import numpy as np
        from joint_design_dft import GammaDFTBackend
        from joint_design import run_design
        protocol=expand(read(ROOT/'examples/response_design/MgSiN2_gamma/design.json'),source)
        protocol['backend_status_file']=str(output/'backend_status.json')
        class MaskedBackend(GammaDFTBackend):
            def geometry(self,theta):
                if args.stage=='design6' and not np.array_equal(theta[6:],self.theta0[6:]):raise ValueError('Coordinates changed in strain-only control')
                if args.stage=='design48' and not np.array_equal(theta[:6],self.theta0[:6]):raise ValueError('Strain changed in fixed-cell control')
                return super().geometry(theta)
            def gradient(self,theta,current,folder):
                g=np.asarray(super().gradient(theta,current,folder)).copy()
                if args.stage=='design6':g[6:]=0
                if args.stage=='design48':g[:6]=0
                return g
        backend=MaskedBackend(protocol)
        result=run_design(calculation,protocol['theta0'],protocol['target_eV'],protocol['limits'],
                          backend,max_updates=5 if args.stage=='design6' else 3,max_backtracks=3)
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

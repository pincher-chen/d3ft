"""Independent PBE/SG15 cubic elastic checks; preserve every QE input/output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import numpy as np
from qe_precision import read_state

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BOHR=0.529177210903
GPA=29421.02648438959

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--material',required=True)
    ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['elastic','force','bulk'],default='elastic')
    ap.add_argument('--displacement',type=float,default=.01,help='Bohr, force mode only')
    ap.add_argument('--a0',type=float)
    ap.add_argument('--cutoff',type=int,default=100,help='Ry')
    ap.add_argument('--bands',type=int)
    ap.add_argument('--k',type=int,default=6)
    ap.add_argument('--strain-steps',default='[0.01,0.005]',help='JSON list of positive elastic strain amplitudes')
    ap.add_argument('--bulk-strain-step',type=float,default=.005)
    ap.add_argument('--reuse-from',type=Path,action='append',default=[],help='Reuse complete points only when QE input bytes, executable and UPF hashes match')
    args=ap.parse_args()
    strains=json.loads(args.strain_steps)
    if (not 1e-4<=args.bulk_strain_step<=.01 or args.k<1 or not strains or len(set(strains))!=len(strains)
            or any(not np.isfinite(h) or not 1e-4<=h<=.02 for h in strains)):
        raise ValueError('Invalid k mesh or strain amplitudes')
    m=json.loads((HERE/'materials.json').read_text())[args.material]
    work=ROOT/'validation/results/phase3_20260908'/args.tag
    work.mkdir(parents=True,exist_ok=False)
    pp=Path(os.environ['D3FT_REFERENCE_ROOT'])/'pseudopotentials/sg15_pbe'/m['pp']
    exe=Path(os.environ['QE_EXE'])
    env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_')) and k!='I_MPI_PMI_LIBRARY'}
    env.update(I_MPI_FABRICS='shm',OMP_NUM_THREADS='8')
    metadata=dict(arguments=vars(args).copy(),executable=str(exe),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  precision_parser_sha256=hashlib.sha256(Path(__file__).with_name('qe_precision.py').read_bytes()).hexdigest(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                  pseudopotentials={s:hashlib.sha256((pp/f'{s}.upf').read_bytes()).hexdigest() for s in set(m['symbols'])})
    metadata['arguments']['reuse_from']=[str(p.resolve()) for p in args.reuse_from]
    (work/'metadata.json').write_text(json.dumps(metadata,indent=2))
    (work/'qe_driver_snapshot.py').write_text(Path(__file__).read_text())
    shutil.copyfile(Path(__file__).with_name('qe_precision.py'),work/'qe_precision_snapshot.py')
    reusable=[]
    for source in args.reuse_from:
        old=json.loads((source/'metadata.json').read_text())
        for key in ['executable_sha256','pseudopotentials']:
            if old[key]!=metadata[key]:raise ValueError(f'QE reuse changed {key}')
        reusable.extend(sorted(source.glob('*/scf.in')))
    def point(label,a,strain=None,relax=False):
        dest=work/label; dest.mkdir()
        strain=np.zeros((3,3)) if strain is None else np.asarray(strain)
        cell=np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]])@(np.eye(3)+strain).T
        positions=np.array([[0,0,0],[a/4,a/4,a/4]])@(np.eye(3)+strain).T
        if args.mode=='force': positions[1,0]+=args.displacement*BOHR
        species=list(dict.fromkeys(m['symbols']))
        inp=f"""&control
 calculation='{'relax' if relax else 'scf'}', prefix='x', outdir='./tmp', pseudo_dir='{pp}',
 tprnfor=.true., tstress=.true., etot_conv_thr=1d-10, forc_conv_thr=1d-6, nstep=80
/
&system
 ibrav=0, nat=2, ntyp={len(species)}, ecutwfc={args.cutoff}, ecutrho={4*args.cutoff},
 {('nbnd='+str(args.bands)+',') if args.bands else ''}
 input_dft='pbe', occupations='fixed'
/
&electrons
 conv_thr=1d-12, electron_maxstep=200, mixing_beta=0.7
/
"""
        if relax: inp+="&ions\n ion_dynamics='bfgs'\n/\n"
        inp+='CELL_PARAMETERS angstrom\n'+'\n'.join(' '.join(f'{x:.12f}' for x in r) for r in cell)+'\n'
        inp+='ATOMIC_SPECIES\n'+'\n'.join(f'{s} 1.0 {s}.upf' for s in species)+'\n'
        inp+=f'K_POINTS automatic\n{args.k} {args.k} {args.k} 0 0 0\nATOMIC_POSITIONS angstrom\n'
        inp+='\n'.join(s+' '+' '.join(f'{x:.12f}' for x in r) for s,r in zip(m['symbols'],positions))+'\n'
        (dest/'scf.in').write_text(inp)
        for old_input in reusable:
            old=old_input.parent
            if not (old/'result.json').exists() or old_input.read_text()!=inp:continue
            old_output=(old/'scf.out').read_text()
            if 'JOB DONE.' not in old_output or 'convergence NOT achieved' in old_output:continue
            if relax and 'End of BFGS Geometry Optimization' not in old_output:continue
            rec=json.loads((old/'result.json').read_text())
            if not all(np.isfinite(rec[k]) for k in ['energy_Ha','pressure_GPa']):continue
            precise,xml=read_state(old)
            rec.update(precise)
            rec.update(label=label,source_seconds=rec['seconds'],seconds=0.,reused_from=str(old.resolve()),
                       source_output_sha256=hashlib.sha256((old/'scf.out').read_bytes()).hexdigest())
            shutil.copyfile(old/'scf.out',dest/'scf.out')
            shutil.copyfile(xml,dest/'state.xml')
            (dest/'result.json').write_text(json.dumps(rec,indent=2))
            print('REUSED',json.dumps(rec),flush=True)
            return rec
        start=time.time()
        with (dest/'scf.out').open('w') as f:
            subprocess.run([str(exe),'-inp','scf.in'],cwd=dest,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        text=(dest/'scf.out').read_text()
        if 'JOB DONE.' not in text or 'convergence NOT achieved' in text: raise RuntimeError(label+' SCF failed')
        if relax and 'End of BFGS Geometry Optimization' not in text: raise RuntimeError(label+' relaxation failed')
        energy=float(re.findall(r'!\s+total energy\s+=\s+([-\d.Ee+]+)',text)[-1])/2
        pressure=float(re.findall(r'total\s+stress[^\n]*P=\s*([-\d.Ee+]+)',text)[-1])/10
        rec=dict(label=label,a0_A=a,energy_Ha=energy,pressure_GPa=pressure,seconds=time.time()-start,relax=relax)
        blocks=re.findall(r'total\s+stress[^\n]*\n([^\n]+)\n([^\n]+)\n([^\n]+)',text)
        tensor=np.array([[float(x) for x in line.split()[:3]] for line in blocks[-1]])
        rec['pressure_printed_GPa']=pressure
        rec['pressure_GPa']=float(np.trace(tensor)*GPA/6)
        rec['stress_QE_Ry_bohr3']=tensor.tolist()
        values=re.findall(r'atom\s+\d+\s+type\s+\d+\s+force\s+=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)',text)
        if len(values)<2: raise RuntimeError('Missing atomic forces')
        rec['forces']=(np.array(values[-2:],float)/2).tolist()
        precise,xml=read_state(dest)
        rec.update(precise)
        shutil.copyfile(xml,dest/'state.xml')
        (dest/'result.json').write_text(json.dumps(rec,indent=2))
        print('POINT',json.dumps(rec),flush=True)
        return rec
    a=args.a0 or m['a0']
    if args.mode=='force':
        result=point('displaced',a)
        result['displacement_Bohr']=args.displacement
        (work/'summary.json').write_text(json.dumps(result,indent=2))
        print('DONE',json.dumps(result),flush=True)
        return
    for i in range(5):
        center=point(f'eq_{i}',a)
        p=center['pressure_GPa']
        if abs(p)/(3*m['Bguess'])<1e-5: break
        a*=1+p/(3*m['Bguess'])
    else: raise RuntimeError('QE pressure root failed')
    vol=a**3/(4*BOHR**3)
    results=dict(material=args.material,a0_A=a,pressure_GPa=p,points=[center])
    if args.mode=='bulk':
        h=args.bulk_strain_step
        iso={0:center}
        for d in [-2*h,-h,h,2*h]:
            label=f'{d:+.3f}' if abs(d) in [.01,.005] else f'{d:+.6f}'
            iso[d]=point('iso_'+label,a,np.eye(3)*d)
            results['points'].append(iso[d])
        curv=lambda h:(iso[h]['energy_Ha']-2*center['energy_Ha']+iso[-h]['energy_Ha'])*GPA/(9*vol*h*h)
        results['bulk_strain_step']=h
        results['B_energy_GPa']=(4*curv(h)-curv(2*h))/3+2*p/3
        results['B_stress_GPa']=-(8*(iso[h]['pressure_GPa']-iso[-h]['pressure_GPa'])-(iso[2*h]['pressure_GPa']-iso[-2*h]['pressure_GPa']))/(36*h)
        (work/'summary.json').write_text(json.dumps(results,indent=2))
        print('DONE',json.dumps(results),flush=True)
        return
    for eps in strains:
        tet=[]; shear=[]
        for sign in [-1,1]:
            d=eps*sign
            label=f'{d:+.3f}' if abs(d) in [.01,.005] else f'{d:+.6f}'
            tet.append(point('tet_'+label,a,np.diag([d,d,(1+d)**-2-1])))
            if m['C44'] is not None:
                s=np.zeros((3,3)); s[1,2]=s[2,1]=d
                shear.append(point('shear_'+label,a,s,relax=True))
        results['points']+=tet+shear
        results[f'D_eps{eps}']=(sum(r['energy_Ha'] for r in tet)-2*center['energy_Ha'])*GPA/(6*vol*eps**2)
        if shear: results[f'C44_eps{eps}']=(sum(r['energy_Ha'] for r in shear)-2*center['energy_Ha'])*GPA/(4*vol*eps**2)
        (work/'summary.json').write_text(json.dumps(results,indent=2))
    print('DONE',json.dumps(results),flush=True)

if __name__=='__main__': main()

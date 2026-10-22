"""Fresh AlP NLCC old/new/QE controls; never reads a earlier solver study result."""
import pathlib,json,shutil,hashlib,os,subprocess,datetime
P=pathlib.Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261014_e0_migration'
I=E/'inputs/AlP_NLCC';I.mkdir(parents=True,exist_ok=True)
pp=pathlib.Path('/XYFS03/SSD_POOL/nscc-gz_pinchen/nscc-gz_pinchen_4/chenp/d3ft/pseudopotentials/alp')
for s in ['Al','P']:shutil.copyfile(pp/f'{s}.upf',I/f'{s}.upf')
a=5.48;cell=[[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]];frac=[[0,0,0],[.252,.249,.251]]
structure=I/'structure.json';structure.write_text(json.dumps(dict(name='AlP_NLCC_displaced_control',cell_angstrom=cell,fractional_positions=frac,symbols=['Al','P'],spin=0),indent=2)+'\n')
past=json.loads((P/'validation/results/response_design_20261006_six_strain_design_v2/protocol.json').read_text());JR='${D3FT_ROOT}/third_party/jrystal_dependency'
# Driver derives ROOT from its path: freeze a driver/config in each E0 runtime.
for mode,gpu in [('old',4),('new',5)]:
 r=E/f'runtime_{mode}';w=r/'validation/workflows/response_design';shutil.copyfile(P/'validation/workflows/response_design/generic_point.py',w/'generic_point.py');shutil.copytree(P/'validation/config',r/'validation/config',dirs_exist_ok=True)
 env=past['environment'].copy();tmp=f'/dev/shm/response_design_e0_alp_{mode}';env.update(PYTHONPATH=f'{r}:{JR}',CUDA_VISIBLE_DEVICES=str(gpu),TMPDIR=tmp,JAX_COMPILATION_CACHE_DIR=tmp+'/jax_cache',PYTHONPYCACHEPREFIX=tmp+'/pycache')
 cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash','-c','mkdir -p "$TMPDIR"; exec "$@"','e0','timeout','1800',past['python'],'-u',str(w/'generic_point.py'),'--structure',str(structure),'--pp-dir',str(I),'--output',str(E/f'alp_{mode}'),'--cutoff','50','--grid','64','--k','1','--epochs','4000','--local-radial-quadrature','simpson']
 log=E/'logs'/f'alp_{mode}.log'
 with log.open('xb') as f:proc=subprocess.Popen(cmd,env={**os.environ,**env},stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=P)
 (E/'commands'/f'alp_{mode}.json').write_text(json.dumps(dict(jobid=250285,gpu=gpu,pid=proc.pid,command=cmd,environment=env,log=str(log),time=datetime.datetime.now().isoformat()),indent=2)+'\n');print(mode,proc.pid)
# Geometry is written by the same _write_xyz and read by JR with its historical unit constant.
# QE explicitly receives that exact Bohr cell; no modern-Angstrom constant mismatch.
import numpy as np
conv=1.8897259886;cb=np.asarray(cell)*conv;pos=np.asarray(frac)@cb
qe=E/'alp_qe';qe.mkdir(exist_ok=False);outdir=qe/'scratch';outdir.mkdir()
text=f"""&CONTROL
 calculation='scf', prefix='AlP_e0', pseudo_dir='{I}', outdir='{outdir}', tprnfor=.true., tstress=.true.
/
&SYSTEM
 ibrav=0, nat=2, ntyp=2, ecutwfc=100, ecutrho=400, occupations='fixed', input_dft='PBE', nr1=64, nr2=64, nr3=64, nosym=.true., noinv=.true.
/
&ELECTRONS
 conv_thr=1.0d-12, electron_maxstep=200, mixing_beta=0.3, diagonalization='david'
/
ATOMIC_SPECIES
Al 26.9815385 Al.upf
P 30.973762 P.upf
CELL_PARAMETERS bohr
"""+'\n'.join(' '.join(f'{x:.16f}' for x in row) for row in cb)+"\nATOMIC_POSITIONS bohr\n"+'\n'.join(s+' '+' '.join(f'{x:.16f}' for x in row) for s,row in zip(['Al','P'],pos))+"\nK_POINTS gamma\n"
(qe/'scf.in').write_text(text)
exe='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/bin_cpu_intel/bin/pw.x'
wrapper=qe/'run.sh';wrapper.write_text("#!/bin/bash\nsource /APP/u22/ai_x86/intel/oneapi2023.2/setvars.sh --force > /dev/null 2>&1\nexport OMP_NUM_THREADS=8 MKL_NUM_THREADS=8\nexec timeout 1800 "+shutil.which('bash')+" -c 'exec \"$1\" -in \"$2\"' e0 "+exe+' '+str(qe/'scf.in')+'\n')
cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=8','bash',str(wrapper)];log=E/'logs/alp_qe.log'
with log.open('xb') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,cwd=qe)
(E/'commands/alp_qe.json').write_text(json.dumps(dict(jobid=250285,gpu=None,pid=proc.pid,command=cmd,log=str(log),time=datetime.datetime.now().isoformat()),indent=2)+'\n');print('QE',proc.pid)
(E/'inputs/AlP_NLCC/protocol.json').write_text(json.dumps(dict(scope='fresh same-UPF fixed Gamma basis control',ecutwfc_Ry=100,ecutrho_Ry=400,grid=[64]*3,kpoints=[[0,0,0]],cell_bohr=cb.tolist(),positions_bohr=pos.tolist(),valence_electrons=8,pp={s:hashlib.sha256((I/f'{s}.upf').read_bytes()).hexdigest() for s in ['Al','P']},acceptance=dict(energy_Ha_cell=2e-5,forces_Ha_Bohr=2e-5,stress_Ha_Bohr3=2e-6,stationarity=1e-6),energy_shift_allowed=False),indent=2)+'\n')

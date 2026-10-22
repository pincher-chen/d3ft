"""Matched QE NSCF path on a private copy of the existing reference save."""
import hashlib,json,os,subprocess,time,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parents[1]
GLOBAL=json.loads((R/'metadata/protocol.json').read_text())
material=sys.argv[1];M=GLOBAL['systems'][material]
P=json.loads((R/'inputs'/material/'path.json').read_text())
assert os.environ['SLURM_JOB_ID']=='250285' and os.uname().nodename=='hn35'
env={k:v for k,v in os.environ.items() if not k.startswith(('SLURM_','PMI_')) and k!='I_MPI_PMI_LIBRARY'}
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
           I_MPI_FABRICS='shm',I_MPI_HYDRA_BOOTSTRAP='fork',I_MPI_PIN='1',
           I_MPI_PIN_PROCESSOR_LIST=','.join(str(i) for i in range(32,56)))
exe=Path(os.environ['QE_EXE']);folder=R/'raw'/material/'QE'
command=['mpirun','-np','24',str(exe),'-inp','bands.in','-npool','6']
start=time.time()
with (folder/'bands.out').open('x') as out:
 code=subprocess.run(command,cwd=folder,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
text=(folder/'bands.out').read_text()
assert code==0 and 'JOB DONE.' in text and 'convergence NOT achieved' not in text, 'QE NSCF failed'
xml=folder/'tmp/calc.save/data-file-schema.xml';tree=ET.parse(xml).getroot()
assert tree.findtext('output/convergence_info/nscf_conv/convergence_achieved')!='false'
band=tree.find('output/band_structure');st=tree.find('output/atomic_structure')
alat=float(st.attrib['alat']);cell=np.array([np.fromstring(st.findtext('cell/'+s),sep=' ') for s in ['a1','a2','a3']])
reciprocal=2*np.pi*np.linalg.inv(cell).T
nodes=band.findall('ks_energies')
values=np.array([np.fromstring(n.findtext('eigenvalues'),sep=' ') for n in nodes])
k=np.array([np.fromstring(n.findtext('k_point'),sep=' ')*2*np.pi/alat@np.linalg.inv(reciprocal) for n in nodes])
assert values.shape==(P['points'],40)
assert np.max(abs(k-np.asarray(P['fractional_calculation'])))<1e-8
assert abs(float(band.findtext('nelec'))-M['charge_e'])<1e-12
np.savez_compressed(folder/'bands.npz',energies_Ha=values,k_fractional=k,cell_bohr=cell)
record=dict(complete=True,command=command,exit_code=code,seconds=time.time()-start,
            source_QE_reference=M['QE_reference'],material=material,points=P['points'],bands_solved=40,
            input_sha256=hashlib.sha256((folder/'bands.in').read_bytes()).hexdigest(),
            output_xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
            pw_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
            starting_density_sha256=M['SCF_density_hash'],
            scope='Original converged SCF density, private copy, NSCF only; no density reoptimization')
(folder/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
print('GAAS_QE_PATH_COMPLETE',json.dumps(record),flush=True)

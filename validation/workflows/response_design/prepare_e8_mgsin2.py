"""Fresh parent-reference path and same-model QE SCF/NSCF inputs, CPU only."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from ase.cell import Cell
from ase.data import atomic_numbers,atomic_masses
import spglib

P=Path(__file__).resolve().parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=False)
source=P/'validation/results/response_design_20261002_resource_fill/runs/mgsin2_gamma_strain_ref_fs_r3'
m=json.loads((source/'metadata.json').read_text());parent_path=P/'validation/inputs/response_design_20260930/MgSiN2.json';parent=json.loads(parent_path.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cell=np.asarray(m['physical']['cell_angstrom']);pos=np.asarray(m['physical']['positions_angstrom']);symbols=m['physical']['symbols'];numbers=[atomic_numbers[s] for s in symbols]
ref=np.asarray(parent['cell_angstrom']);frac=np.asarray(parent['fractional_positions'])
bp=Cell(ref).bandpath(npoints=61)
path_frac=np.asarray(bp.kpts)
group=spglib.get_symmetry_dataset((ref,frac,numbers),symprec=1e-3)
actual=spglib.get_symmetry_dataset((cell,pos@np.linalg.inv(cell),numbers),symprec=1e-3)
dist,ticks,labels=bp.get_linear_kpoint_axis()
record=dict(time=datetime.datetime.now().isoformat(),material='MgSiN2',source=str(source),source_metadata_sha256=sha(source/'metadata.json'),parent_input_sha256=sha(parent_path),
 parent_spacegroup=dict(number=int(group.number),international=str(group.international)),actual_spacegroup=dict(number=int(actual.number),international=str(actual.international)),
 path=bp.path,special_points={k:np.asarray(v).tolist() for k,v in bp.special_points.items()},k_fractional=path_frac.tolist(),
 parent_distance_inverse_A=dist.tolist(),parent_ticks_inverse_A=ticks.tolist(),labels=labels,
 actual_distance_inverse_A=None,scope='Independently generated ASE parent high-symmetry path transported to owned prestrained geometry; labels are parent-reference labels, not actual strained-cell symmetries',
 original_state_SCF_kmesh=[1,1,1],QE_SCF_kmesh=[1,1,1],new_response_results=True,no_JCTC_data=True)
# Breaks in the ASE path are preserved; the parent ticks, not an artificial
# line joining disconnected segments, are the plotting metadata.
(a.output/'path.json').write_text(json.dumps(record,indent=2)+'\n')
pp=m['physical']['pp']
for s,item in pp.items():
 if sha(item['path'])!=item['sha256']:raise ValueError('PP changed '+s)
bohr=1.8897259886;cellb=cell*bohr;posb=pos*bohr;unique=list(dict.fromkeys(symbols))
ecut=2*float(m['physical']['cutoff']);grid=np.array(m['config']['grid_sizes']);grid=np.repeat(grid,3) if grid.ndim==0 else grid
factor=int(m['physical']['xc_grid_factor']);fine=grid*factor
# QE activates dense -> smooth interpolation only when dual > 4+1e-8.
# Different forced FFT dimensions with exactly dual=4 silently bypass it.
# Activate that path without changing even one selected density G vector.
rho_cut=4*ecut;rho_basis_check=None
if not np.array_equal(fine,grid):
 candidate=rho_cut+1e-5;b=2*np.pi*np.linalg.inv(cellb).T
 bound=np.ceil(np.linalg.norm(cellb,axis=1)*np.sqrt(candidate)/(2*np.pi)).astype(int)+1
 j,k=np.meshgrid(np.arange(-bound[1],bound[1]+1),np.arange(-bound[2],bound[2]+1),indexing='ij')
 jk=np.stack([j.ravel(),k.ravel()],axis=1);counts=[0,0]
 for n in range(-bound[0],bound[0]+1):
  g=np.column_stack([np.full(len(jk),n),jk])@b;g2=np.sum(g*g,axis=1)
  counts[0]+=int(np.sum(g2<=rho_cut));counts[1]+=int(np.sum(g2<=candidate))
 if counts[0]!=counts[1]:raise ValueError('QE dual activation changes density G set; explicit review required')
 if not candidate/ecut>4+1e-8:raise ValueError('QE double-grid activation threshold not crossed')
 rho_basis_check=dict(original_ecutrho_Ry=rho_cut,corrected_ecutrho_Ry=candidate,original_G=counts[0],corrected_G=counts[1],same_G_set=True)
 rho_cut=candidate
scratch=a.output/'scratch';scratch.mkdir()
def make(kind):
 control=f"&CONTROL\n calculation='{kind}', prefix='MgSiN2_e8', pseudo_dir='{Path(pp[unique[0]]['path']).parent}', outdir='{scratch}', tprnfor=.true., tstress=.true.\n/\n"
 system=f"&SYSTEM\n ibrav=0, nat=16, ntyp=3, ecutwfc={ecut}, ecutrho={rho_cut}, nbnd=56, occupations='fixed', input_dft='PBE', nosym=.true., noinv=.true.,\n nr1={fine[0]}, nr2={fine[1]}, nr3={fine[2]}, nr1s={grid[0]}, nr2s={grid[1]}, nr3s={grid[2]}\n/\n"
 electrons="&ELECTRONS\n conv_thr=1.0d-10, electron_maxstep=150, mixing_beta=0.3, diagonalization='david', diago_thr_init=1.0d-10\n/\n"
 species='ATOMIC_SPECIES\n'+'\n'.join(f'{s} {atomic_masses[atomic_numbers[s]]:.10f} {Path(pp[s]["path"]).name}' for s in unique)+'\n'
 geom='CELL_PARAMETERS bohr\n'+'\n'.join(' '.join(f'{x:.16f}' for x in r) for r in cellb)+'\nATOMIC_POSITIONS bohr\n'+'\n'.join(s+' '+' '.join(f'{x:.16f}' for x in r) for s,r in zip(symbols,posb))+'\n'
 ks='K_POINTS gamma\n' if kind=='scf' else f'K_POINTS crystal\n{len(path_frac)}\n'+'\n'.join(' '.join(f'{x:.16f}' for x in r)+' 1.0' for r in path_frac)+'\n'
 return control+system+electrons+species+geom+ks
for kind in ['scf','nscf']:(a.output/(kind+'.in')).write_text(make(kind))
(a.output/'protocol.json').write_text(json.dumps(dict(**record,cutoff_Ha=ecut/2,ecutwfc_Ry=ecut,ecutrho_Ry=rho_cut,coarse_grid=grid.tolist(),XC_fine_grid=fine.tolist(),pp=pp,QE_doublegrid_G_set_check=rho_basis_check,
 QE_total_bands=56,occupied_bands=48,SCF_density_recomputed=True,identical_Miller_basis=False,
 basis_difference='D3FT fixed |G| sphere, standard QE |G+k| sphere; same cutoff does not give identical basis away from Gamma',
 independent_alignment='Each code VBM; no rigid shift used to claim absolute-energy agreement',
 budget_GPU_hours=0,CPU_timeout_per_stage_seconds=21600),indent=2)+'\n')
exe='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/bin_cpu_intel/bin/pw.x'
(a.output/'run_stage.sh').write_text('#!/bin/bash\nsource /APP/u22/ai_x86/intel/oneapi2023.2/setvars.sh --force >'+str(a.output/'environment_setup.log')+' 2>&1\nsetup_code=$?\nif [ "$setup_code" -ne 0 ]; then tail -n 15 '+str(a.output/'environment_setup.log')+'; exit "$setup_code"; fi\nexport OMP_NUM_THREADS=16 MKL_NUM_THREADS=16\nexec timeout 21600 '+exe+' -in "$1"\n')
print('E8 MgSiN2路径与QE输入已独立生成。\n母相/实际空间群、相同UPF与100Ha截止已记录。\nSCF→NSCF仅CPU，旧密度和旧图不复用。',flush=True)

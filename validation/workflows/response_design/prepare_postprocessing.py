"""Prepare actual-cell symmetry, parent reference path, and immutable state bindings."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[3];R=P/'validation/results';O=R/'response_design_20261005_band_structure'
sys.path.insert(0,str(R/'response_design_20261005_postprocessing_plan/analysis_packages'))
import seekpath,spglib
from ase.data import atomic_numbers
from ase.geometry import cell_to_cellpar
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read(P/'validation/inputs/response_design_20260930/MgSiN2.json');nums=[atomic_numbers[x] for x in s['symbols']]
parent=(np.array(s['cell_angstrom']),np.array(s['fractional_positions']),nums)
path=seekpath.get_path_orig_cell(parent,symprec=1e-5)
points=path['point_coords'];segments=path['path']
# Include the observed mesh VBM explicitly even if convention maps its label.
vbm=np.array([0,.5,.5]);lab=next((k for k,v in points.items() if np.allclose(v,vbm,rtol=0,atol=1e-10)),None)
if lab is None:lab='Q';points[lab]=vbm.tolist();segments.append(('GAMMA',lab))
recip=2*np.pi*np.linalg.inv(parent[0]).T
lengths=[np.linalg.norm((np.array(points[b])-points[a])@recip) for a,b in segments]
step=sum(lengths)/112
unique=[];lookup={};segrows=[];x=0.
for (a,b),length in zip(segments,lengths):
    n=max(3,int(np.ceil(length/step)));frac=np.linspace(points[a],points[b],n+1);indices=[]
    for f in frac:
        key=tuple(np.round(f,12))
        if key not in lookup:lookup[key]=len(unique);unique.append(f.tolist())
        indices.append(lookup[key])
    segrows.append(dict(start=a,end=b,indices=indices,x=np.linspace(x,x+length,n+1).tolist()))
    x+=length
states={
 'start':('response_design_20261003_design_k4_check/runs/design_start_k4','response_design_20261003_design_k4_check/spectra/design_start_k4_complete.json'),
 'reverse':('response_design_20261004_reverse_endpoint_k4/runs/endpoint','response_design_20261004_reverse_endpoint_k4/result.json'),
 'fd_candidate':('response_design_20261004_corrected_k4_check/runs/corrected_endpoint_k4','response_design_20261004_corrected_k4_spectrum_recovery/corrected_endpoint_k4_complete.json')}
report=dict(packages={'seekpath':seekpath.__version__,'spglib':spglib.__version__},parent_path=path,labels=points,segments=segrows,k_fractional=unique,coordinate='parent reciprocal arc length; common fractional path mapped to each strained cell',edge_indices={'Gamma':lookup[(0.,0.,0.)],'mesh_VBM':lookup[tuple(vbm)]},states={})
for name,(source,spectrum) in states.items():
    folder=R/source;p=read(folder/'point.json');cell=np.array(p['cell_bohr'])*.529177210903;f=np.array(p['positions_bohr'])@np.linalg.inv(np.array(p['cell_bohr']))
    syms=[]
    for tol in [1e-5,1e-4,1e-3]:
        d=spglib.get_symmetry_dataset((cell,f,nums),symprec=tol)
        syms.append(dict(symprec_angstrom=tol,number=int(d.number),international=d.international))
    report['states'][name]=dict(source=str(folder.resolve()),spectrum=str((R/spectrum).resolve()),source_sha256={n:sha(folder/n) for n in ['metadata.json','point.json','density.npy','mask.npy','input.xyz']},cell_angstrom=cell.tolist(),fractional_positions=f.tolist(),symbols=s['symbols'],cell_parameters=cell_to_cellpar(cell).tolist(),symmetry=syms,frac_parent_error=float(np.max(abs(f-parent[1]))))
report['parent_symmetry']={'number':path['spacegroup_number'],'international':path['spacegroup_international']}
# JSON accepts numpy values in SeeK-path metadata only through explicit conversion.
(O/'path_protocol.json').write_text(json.dumps(report,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item())+'\n')
print(json.dumps({'unique_points':len(unique),'segments':segments,'parent':report['parent_symmetry'],'states':{n:v['symmetry'] for n,v in report['states'].items()},'edges':report['edge_indices']},indent=2))

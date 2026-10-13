"""Read full-precision QE XML; verify units and agreement with printed output."""
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import numpy as np

GPA=29421.02648438959

def read_state(folder,seen=None):
    folder=Path(folder).resolve()
    seen=set() if seen is None else seen
    if folder in seen:raise ValueError('Cyclic QE reuse provenance')
    seen.add(folder)
    xml=next((p for p in [folder/'state.xml',folder/'tmp/x.save/data-file-schema.xml'] if p.exists()),None)
    if xml is None:
        rec=json.loads((folder/'result.json').read_text())
        source=Path(rec['reused_from'])
        if hashlib.sha256((source/'scf.out').read_bytes()).hexdigest()!=rec['source_output_sha256']:
            raise ValueError('Reused QE output hash mismatch')
        values,xml=read_state(source,seen)
        return values,xml
    root=ET.parse(xml).getroot()
    if root.attrib.get('Units')!='Hartree atomic units':raise ValueError('Unrecognized QE XML units')
    energy=float(root.find('.//output/total_energy/etot').text)
    stress=np.fromstring(root.find('.//output/stress').text,sep=' ').reshape(3,3)
    forces=np.fromstring(root.find('.//output/forces').text,sep=' ').reshape(-1,3)
    if not np.isfinite(energy) or not np.isfinite(stress).all() or not np.isfinite(forces).all():
        raise ValueError('Nonfinite QE XML values')
    text=(folder/'scf.out').read_text()
    if 'JOB DONE.' not in text or 'convergence NOT achieved' in text:raise ValueError('Incomplete QE calculation')
    printed=float(re.findall(r'!\s+total energy\s+=\s+([-\d.Ee+]+)',text)[-1])/2
    if abs(energy-printed)>2.6e-9:raise ValueError('XML energy disagrees with final printed energy')
    blocks=re.findall(r'total\s+stress[^\n]*\n([^\n]+)\n([^\n]+)\n([^\n]+)',text)
    printed_stress=np.array([[float(x) for x in line.split()[:3]] for line in blocks[-1]])/2
    # This executable prints both energy and Ry stress with eight decimals.
    if np.max(np.abs(stress-printed_stress))>2.6e-9:raise ValueError('XML stress disagrees with printed Hartree-converted stress')
    result=dict(energy_Ha=energy,energy_stdout_Ha=printed,pressure_GPa=float(np.trace(stress)*GPA/3),
        stress_QE_Ry_bohr3=(2*stress).tolist(),forces=forces.tolist(),
        precision_source='QE XML in explicit Hartree atomic units',
        xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),xml_source=str(xml))
    return result,xml

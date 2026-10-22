"""Gate-only observer, then fresh same-mesh QE and seven re-equilibrated FDs.

All FD directions are registered before running. Existing physical gates are
unchanged. Only AlP's defined observables (a0, B) can be certified, not absent
C11/C44. Old SG15 and K4/K10 attempt remain separate historical records.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import e1_extend_kmesh as runner
from fd_acceptance import component

P, E = runner.P, runner.E
EXT = E / 'extension_20261015'
A = EXT / 'formal_validation'
dump, now, sha = runner.dump, runner.now, runner.sha


def note(text):
    runner.note(text)


def qe_job(source, tag, k):
    points = json.loads((source/'points.json').read_text())
    point = [r for r in points if r['label'].startswith('eq_')][-1]
    meta = json.loads((source/'metadata.json').read_text())
    xyz = (source/(point['label']+'.xyz')).read_text().splitlines()
    cell = np.fromstring(re.search(r'Lattice="([^"]+)"',xyz[1]).group(1), sep=' ').reshape(3,3)*1.8897259886
    positions = np.array([[float(x) for x in l.split()[1:4]] for l in xyz[2:4]])*1.8897259886
    root = A/tag
    root.mkdir()
    (root/'scratch').mkdir()
    pps = meta['pseudopotentials']
    for path,h in pps.items():
        if sha(path)!=h:
            raise RuntimeError('PP changed')
    # QE 7.5 PW/src/kpoint_grid.f90 lines 73-75 explicitly construct
    # (i-1)/Nk + shift/(2Nk). Offsets 0,0,0 are exactly our Gamma mesh.
    src = Path('/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/PW/src/kpoint_grid.f90')
    text = f"""&CONTROL
 calculation='scf', prefix='AlP_e1', pseudo_dir='{Path(next(iter(pps))).parent}', outdir='{root/'scratch'}', tprnfor=.true., tstress=.true.
/
&SYSTEM
 ibrav=0, nat=2, ntyp=2, ecutwfc=100, ecutrho=400, occupations='fixed', input_dft='PBE', nr1=64, nr2=64, nr3=64
/
&ELECTRONS
 conv_thr=1.0d-12, electron_maxstep=200, mixing_beta=0.3, diagonalization='david'
/
ATOMIC_SPECIES
Al 26.9815385 Al.upf
P 30.973762 P.upf
CELL_PARAMETERS bohr
"""+'\n'.join(' '.join(f'{x:.16f}' for x in r) for r in cell)+'\nATOMIC_POSITIONS bohr\n'+'\n'.join(s+' '+' '.join(f'{x:.16f}' for x in r) for s,r in zip(['Al','P'],positions))+f'\nK_POINTS automatic\n{k} {k} {k} 0 0 0\n'
    (root/'scf.in').write_text(text)
    exe='/HOME/nscc-gz_pinchen/nscc-gz_pinchen_4/HDD_POOL/sf_box/qe-7.5/bin_cpu_intel/bin/pw.x'
    sh=root/'run.sh'
    sh.write_text('#!/bin/bash\nsource /APP/u22/ai_x86/intel/oneapi2023.2/setvars.sh --force >/dev/null 2>&1\nexport OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\nexec timeout 10800 '+exe+' -in '+str(root/'scf.in')+'\n')
    cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','--cpus-per-task=4','bash',str(sh)]
    rec=dict(start=now(),jobid=250285,gpus=[],GPU_hours=0,command=cmd,source=str(source),k=k,
             pp=pps,input_sha256=sha(root/'scf.in'),executable_sha256=sha(exe),
             QE_kpoint_grid_source_sha256=sha(src),symmetry_reduction=True,offsets=[0,0,0],
             exact_common_Miller_basis=False,energy_shift_allowed=False)
    with (root/'scf.log').open('xb') as f:
        proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rec['pid']=proc.pid;dump(root/'launch.json',rec)
        code=proc.wait()
    rec.update(end=now(),returncode=code);dump(root/'launch.json',rec)
    if code:
        raise RuntimeError('QE '+tag+' failed; log preserved')
    xml=root/'scratch/AlP_e1.save/data-file-schema.xml'
    out=ET.parse(xml).getroot().find('output')
    result=dict(k=k,source=str(source),energy_Ha=float(out.findtext('total_energy/etot')),
                stress_Ha_Bohr3=np.fromstring(out.findtext('stress'),sep=' ').reshape(3,3).tolist(),
                forces_Ha_Bohr=np.fromstring(out.findtext('forces'),sep=' ').reshape(2,3).tolist(),
                SCF_converged=(out.findtext('convergence_info/scf_conv/convergence_achieved') or '').strip().lower()=='true',
                xml_sha256=sha(xml),full_mesh_k_points=k**3)
    dump(root/'result.json',result)
    return result


def main():
    A.mkdir(exist_ok=True)
    fd=os.open(A/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd)
    dump(A/'status.json',dict(time=now(),stage='waiting_converged_mesh',GPU_tasks_submitted=0))
    while True:
        f=EXT/'status.json'
        if f.exists():
            s=json.loads(f.read_text())
            if s['stage'] in ['failed','K14_still_not_converged']:
                dump(A/'status.json',dict(time=now(),stage='not_submitted_parent_failed',parent=s));return
            if s['stage']=='adopted_mesh_native_complete_pending_QE_parameter_FD':
                break
        time.sleep(20)
    if not s['force_passed']:
        raise RuntimeError('Adopted-mesh force gate failed')
    k=s['adopted_k'];row=json.loads(Path(s['response_row']).read_text());source=Path(row['source'])
    summary=json.loads((source/'summary.json').read_text())
    # Validate QE symmetry reduction against the completed, full explicit K4
    # control before relying on it at the adopted mesh. No density averaging
    # or hand-weighted irreducible mesh with nosym is used.
    q4=qe_job(E/'runs/AlP_NLCC_k4','qe_k4_symmetry_control',4)
    old_xml=E/'acceptance_20261015/qe_k4/scratch/AlP_e1.save/data-file-schema.xml'
    old=ET.parse(old_xml).getroot().find('output')
    checks=dict(SCF=q4['SCF_converged'],
                energy_reduction=abs(q4['energy_Ha']-float(old.findtext('total_energy/etot')))<1e-8,
                stress_reduction=float(np.max(abs(np.asarray(q4['stress_Ha_Bohr3'])-np.fromstring(old.findtext('stress'),sep=' ').reshape(3,3))))<1e-8)
    dump(A/'QE_symmetry_control.json',dict(checks=checks,passed=all(checks.values()),reference_xml_sha256=sha(old_xml)))
    if not all(checks.values()):
        raise RuntimeError('QE symmetry-control equivalence failed; no adopted-mesh shortcut accepted')
    q=qe_job(source,f'qe_k{k}_adopted',k)
    points=json.loads((source/'points.json').read_text());eq=[p for p in points if p['label'].startswith('eq_')][-1]
    force=json.loads((EXT/f'runs/AlP_NLCC_k{k}_force/summary.json').read_text())
    limits=json.loads((P/'validation/results/response_design_20261014_e0_migration/inputs/AlP_NLCC/protocol.json').read_text())['acceptance']
    errors=dict(energy_Ha_cell=abs(q['energy_Ha']-eq['energy_Ha']),
                stress_Ha_Bohr3=float(np.max(abs(np.asarray(q['stress_Ha_Bohr3'])+np.asarray(eq['stress'])))),
                forces_Ha_Bohr=float(np.max(abs(np.asarray(q['forces_Ha_Bohr'])-np.asarray(force['forces'])))))
    pressure=float(np.trace(np.asarray(q['stress_Ha_Bohr3']))/3*29421.02648438959)
    qchecks={name:err<=limits[name] for name,err in errors.items()}
    qchecks.update(SCF=q['SCF_converged'],QE_at_root=abs(pressure)/(3*summary['B_GPa'])<1e-5)
    dump(A/'QE_acceptance.json',dict(checks=qchecks,passed=all(qchecks.values()),errors=errors,pressure_GPa=pressure,energy_shift_allowed=False))
    if not all(qchecks.values()):
        raise RuntimeError('Adopted-mesh QE physical gate failed; retain row unaccepted')
    # Resolve each axis at the ORIGINAL a0/B numerical budgets. No validation
    # set and no fit parameter is used to choose these numerical FD steps.
    da=np.array(row['da0_dp']);db=np.array(row['dB_equilibrium_dp'])
    h=np.maximum(.01,np.maximum(.75e-4*summary['a0_A']/np.maximum(abs(da),1e-12),.075/np.maximum(abs(db),1e-12)))
    if np.max(h)>.1:
        raise RuntimeError('Seven-axis FD needs an unsafe step; budget/plan must be revised')
    protocol=dict(time=now(),adopted_k=k,source=str(source),steps=h.tolist(),parameter_order=row['parameter_order'],
                  lattice_relative_gate=1e-4,bulk_absolute_gate_GPa=.1,derivative_relative_gate=.02,
                  pressure_tolerance=1e-8,FD_scheme='seven independent re-equilibrated central pairs',
                  applicable_observables=['a0_A','B_GPa'],not_certified=['C11_minus_C12_GPa','C44_GPa'],
                  optional_modules_enabled=False,E1_total_GPU_hours_cap=250)
    dump(A/'FD_protocol.json',protocol)
    runner.A=A;runner.runtime=Path(json.loads((E/'protocol.json').read_text())['runtime']);runner.mask=source/'mask.npy'
    launch=json.loads((EXT/f'AlP_NLCC_k{k}_launch.json').read_text())
    runner.env=launch['environment'];runner.env['D3FT_PHASE3_RESULTS_ROOT']=str(A/'runs');(A/'runs').mkdir()
    runner.spent_before=s['total_GPU_hours_upper_bound'];runner.launches=[]
    fd_results=[]
    for axis,step in enumerate(h):
        pairs=[]
        for sign,name in [(-1,'minus'),(1,'plus')]:
            vector=np.zeros(7);vector[axis]=sign*step
            initial=summary['a0_A']+da[axis]*sign*step
            folder=runner.run_stage(f'k{k}_axis{axis}_{name}',k,initial,
                                   extra_args=['--vector',json.dumps(vector.tolist()),'--pressure-tol','1e-8'])
            pairs.append(json.loads((folder/'summary.json').read_text()))
            runner.derive(folder)
        rows=[component(key,float(gradient*2*step),pairs[1][key]-pairs[0][key],summary['a0_A'],.02)
              for key,gradient in [('a0_A',da[axis]),('B_GPa',db[axis])]]
        root_pass=all(abs(r['equilibrium_pressure_GPa'])/(3*r['B_GPa'])<1e-7 for r in pairs)
        cert=dict(axis=axis,parameter=row['parameter_order'][axis],step=float(step),rows=rows,
                  root_passed=root_pass,passed=root_pass and all(r['joint_gate_passed'] for r in rows),
                  scope='AlP defined a0/B observables only; not a four-elastic-observable certificate')
        fd_results.append(cert);dump(A/f'FD_axis{axis}.json',cert)
        note(f'E1 K{k}独立参数{row["parameter_order"][axis]}重平衡FD '+('PASS' if cert['passed'] else 'FAIL')+'；原2%及信号数值门保持。')
        if not cert['passed']:
            raise RuntimeError(f'Independent parameter FD axis{axis} failed; preserve and do not loosen')
    row.update(accepted_for_calibration=True,independently_reoptimized_parameter_FD=True,
               parameter_FD_certificate=str(A/'acceptance.json'),QE_certificate=str(A/'QE_acceptance.json'))
    dump(A/f'accepted_response_row_k{k}.json',row)
    result=dict(time=now(),passed=True,status='完成',adopted_k=k,response_row=str(A/f'accepted_response_row_k{k}.json'),
                QE_checks=qchecks,seven_independent_reoptimized_parameter_FD=fd_results,
                k_convergence_passed=True,force_passed=True,old_SG15_rows_replaced=False,
                total_GPU_hours_upper_bound=runner.spent_before+sum(r['GPU_hours'] for r in runner.launches))
    dump(A/'acceptance.json',result);dump(A/'status.json',dict(stage='complete',**result))
    dump(E/'latest_acceptance.json',result)
    doc=P/'paper/response_design_EXPERIMENTS_20261014.md';lines=doc.read_text().splitlines()
    for i,line in enumerate(lines):
        if line.startswith('| E1 |'):
            cols=line.split('|');cols[-2]=f' **完成**：新NLCC K{k}满足相邻k收敛、QE/力/数值及七参数独立重平衡FD；旧SG15行独立保留 ';lines[i]='|'.join(cols)
    doc.write_text('\n'.join(lines)+'\n')
    note(f'E1正式完成：采用新NLCC K{k}，证据 extension_20261015/formal_validation/acceptance.json；新七参数行已独立重算/验收，旧SG15行未覆盖，粗网格失败记录保留。')
    print(f'E1 正式完成：采用K{k}。\n七参数重平衡FD、QE、力及原k门全部PASS。\n新行独立保存，旧SG15与失败记录保留。',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        dump(A/'status.json',dict(time=now(),stage='failed',passed=False,error=str(exc)))
        note('E1新网格正式验收尚未完成：'+str(exc)+'；不放宽门、不覆盖旧行。')
        raise

"""CPU-only corrected QE SCF -> Gamma acceptance -> unchanged D3FT path comparison."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
from e1_three_parameter_queue import read,dump


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return dt.datetime.now().astimezone().isoformat()


def main(T):
    os.close(os.open(T/'controller.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY))
    c=read(T/'protocol.json');P=Path(c['project']);child=None
    def state(stage,**kw):dump(T/'status.json',dict(time=now(),stage=stage,controller_pid=os.getpid(),
        jobid=250285,gpus=[],GPU_hours=0,E8_complete=False,**kw))
    def note(s):
        for f in [P/'paper/response_design_EXPERIMENTS_20261014.md',P/'response_design_to_do_list.md',P/'paper/response_design_E8_QE_REPAIR_20261016.md']:
            with f.open('a') as h:h.write('\n- '+now()+' '+s+'\n')
    def run(stage):
        nonlocal child
        cmd=['srun','--jobid=250285','--overlap','--cpu-bind=none','-N1','-n1','-c8',
            'bash',str(T/'run_CPU_stage.sh'),str(T/(stage+'.in'))]
        rec=dict(time=now(),command=cmd,gpus=[],GPU_hours=0,jobid=250285,status='running',
            input_sha256=sha(T/(stage+'.in')),timeout_seconds=43200)
        with (T/(stage+'.log')).open('xb') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rec['pid']=child.pid;dump(T/(stage+'_launch.json'),rec)
            while child.poll() is None:state('CPU_QE_'+stage+'_running',pid=child.pid);time.sleep(15)
        rec.update(end=now(),returncode=child.returncode,status='completed' if child.returncode==0 else 'failed')
        dump(T/(stage+'_launch.json'),rec)
        if child.returncode:raise RuntimeError('CPU QE '+stage+' failed '+str(child.returncode))
        xml=T/'scratch/MgSiN2_e8_repair.save/data-file-schema.xml'
        data=xml.read_bytes();(T/(stage+'_data-file-schema.xml')).write_bytes(data)
        out=ET.fromstring(data).find('output')
        if 'JOB DONE.' not in (T/(stage+'.log')).read_text():raise ValueError('QE completion marker absent')
        gate=out.findtext('convergence_info/'+('scf_conv' if stage=='scf' else 'nscf_conv')+'/convergence_achieved')
        if gate and gate.strip().lower()=='false':raise ValueError('QE not converged')
        if abs(float(out.findtext('band_structure/nelec'))-96)>1e-8:raise ValueError('QE charge changed')
        return out
    try:
        for path,h in c['input_bindings'].items():
            if sha(path)!=h:raise ValueError('Changed frozen input/model '+path)
        scf=run('scf');energy=float(scf.findtext('total_energy/etot'))
        bands=np.fromstring(scf.findtext('band_structure/ks_energies/eigenvalues'),sep=' ')
        reference=read(c['D3FT_Gamma_spectrum']);d=np.asarray(reference['eigenvalues_Ha'])
        if bands.shape!=(56,):raise ValueError('Wrong Gamma band count')
        HA=27.211386245988;delta=(bands-d)*HA;aligned=delta-delta[47]
        gap=(bands[48]-bands[47])*HA;gap_d=(d[48]-d[47])*HA
        energy_error=energy-c['D3FT_energy_Ha']
        gamma=dict(time=now(),QE_energy_Ha=energy,D3FT_energy_Ha=c['D3FT_energy_Ha'],
            energy_error_Ha=energy_error,energy_error_mHa=energy_error*1000,
            QE_gap_eV=gap,D3FT_gap_eV=gap_d,gap_error_eV=gap-gap_d,
            maximum_absolute_eigenvalue_difference_eV=float(abs(delta).max()),
            maximum_VBM_aligned_eigenvalue_difference_eV=float(abs(aligned).max()),
            thresholds=c['Gamma_gates'],QE_eigenvalues_Ha=bands.tolist(),D3FT_eigenvalues_Ha=d.tolist(),
            passed=bool(abs(energy_error)<=c['Gamma_gates']['total_energy_Ha']
                and abs(gap-gap_d)<=c['Gamma_gates']['gap_eV']
                and abs(aligned).max()<=c['Gamma_gates']['VBM_aligned_max_eV']))
        dump(T/'Gamma_acceptance.json',gamma)
        np.savetxt(T/'Gamma_eigenvalue_comparison.csv',np.column_stack([np.arange(1,57),d,bands,delta,aligned]),
            delimiter=',',header='band,D3FT_Ha,QE_Ha,absolute_difference_eV,VBM_aligned_difference_eV',comments='')
        if not gamma['passed']:raise ValueError('Corrected QE Gamma numerical gate failed; no path submitted')
        note('E8修正QE Γ能量/谱门PASS：ΔE='+str(energy_error*1000)+'mHa，Δgap='+str((gap-gap_d)*1000)+'meV；仅CPU自动接61点路径，D3FT已存谱不重算。')
        out=run('nscf');st=out.find('atomic_structure');alat=float(st.attrib['alat'])
        cell=np.asarray([np.fromstring(st.findtext('cell/'+s),sep=' ') for s in ['a1','a2','a3']])
        reciprocal=2*np.pi*np.linalg.inv(cell).T;nodes=out.findall('band_structure/ks_energies')
        frac=np.asarray([np.fromstring(n.findtext('k_point'),sep=' ')*2*np.pi/alat@np.linalg.inv(reciprocal) for n in nodes])
        values=np.asarray([np.fromstring(n.findtext('eigenvalues'),sep=' ') for n in nodes])
        if values.shape!=(61,56) or np.max(abs(frac-np.asarray(c['k_fractional'])))>1e-8:
            raise ValueError('QE full path shape/order mismatch')
        dft=np.asarray([read(Path(c['D3FT_accepted'])/f'k_{i:04d}.json')['eigenvalues_Ha'] for i in range(61)])
        np.savez_compressed(T/'corrected_QE_bands.npz',energies_Ha=values,k_fractional=frac,cell_bohr=cell)
        left=(dft-dft[:,47].max())*HA;right=(values-values[:,47].max())*HA;delta=left-right
        result=dict(time=now(),QE_rerun_completed=True,D3FT_61_points_reused=True,GPU_hours=0,
            Gamma_acceptance=gamma,path_gap_D3FT_eV=float((dft[:,48].min()-dft[:,47].max())*HA),
            path_gap_QE_eV=float((values[:,48].min()-values[:,47].max())*HA),
            VBM_aligned_all_bands_MAE_eV=float(abs(delta).mean()),
            VBM_aligned_all_bands_max_abs_eV=float(abs(delta).max()),
            VBM_aligned_frontier44_55_MAE_eV=float(abs(delta[:,44:56]).mean()),
            E8_complete=False,cross_code_full_path_acceptance_pending=True,
            caveat='Gamma gates preregistered; full-path comparison requires review. No k convergence or experimental validation.')
        dump(T/'corrected_comparison.json',result)
        np.savez_compressed(T/'corrected_aligned_bands.npz',D3FT_eV=left,QE_eV=right,k_fractional=frac,difference_eV=delta)
        state('CPU_QE_repair_complete_full_path_review_pending',Gamma_passed=True,comparison=result)
        dump(Path(c['E8_validation_status']),dict(time=now(),stage='corrected_QE_comparison_review_pending',
            E8_complete=False,Gamma_passed=True,repair_root=str(T),old_comparison_rejected=True))
        note('E8 CPU修正Γ及61点路径已生成，新比较待审；原错误比较保留，不自动把E8标完成，不改论文/图。')
    except Exception as exc:
        state('failed',passed=False,error=repr(exc));note('E8 CPU QE修复停止：'+repr(exc)+'；原错误/新失败保留，E8仍未完成。');raise
    finally:
        if child is not None and child.poll() is None:child.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    main(ap.parse_args().root)

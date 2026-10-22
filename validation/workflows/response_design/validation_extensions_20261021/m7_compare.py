"""Common observable units for M7/E4/M4; never refit or mutate baselines."""
import csv,json
from pathlib import Path
import numpy as np
HA_EV=27.211386245988
BOHR_A=.529177210903

def compare(protocol,output,assigned_gpu_hours,destination):
    p=protocol;d=Path(destination);read=lambda f:json.loads(Path(f).read_text())
    e4=Path(p['baseline_E4_protocol']).parent
    m4=Path(p['baseline_M4_acceptance']).parent
    m7=Path(output)
    sources=[('E4_joint54',e4/'design/trajectory.json',read(e4/'status.json').get('GPU_hours')),
             ('M4_coordinates48',m4/'trajectory.json',read(p['baseline_M4_summary'])['GPU_hours']),
             ('M7_strains6',m7/'trajectory.json',assigned_gpu_hours)]
    rows=[];summaries=[]
    for label,file,gpuh in sources:
        if not file.exists():
            summaries.append(dict(experiment=label,trajectory_available=False));continue
        tr=read(file);ref=tr[0]['readout'];origin=np.array(tr[0]['theta']);prev=origin
        for t in tr:
            r=t['readout'];theta=np.array(t['theta']);force=np.asarray(r['forces_Ha_Bohr'])
            rows.append(dict(experiment=label,step=t['step'],gap_eV=r['gap_eV'],
                 target_error_meV=(r['gap_eV']-p['target_eV'])*1000,
                 energy_change_meV_atom=(r['energy_Ha']-ref['energy_Ha'])*HA_EV*1000/16,
                 max_force_norm_eV_A=float(np.linalg.norm(force,axis=1).max())*HA_EV/BOHR_A,
                 max_force_component_Ha_Bohr=float(abs(force).max()),
                 strain6=theta[:6].tolist(),strain_step_L2=float(np.linalg.norm(theta[:6]-prev[:6])),
                 strain_total_L2=float(np.linalg.norm(theta[:6]-origin[:6])),
                 trajectory_source=str(file)))
            prev=theta
        third=next((t for t in tr if t['step']==3),tr[-1])
        summaries.append(dict(experiment=label,accepted_updates=len(tr)-1,
             initial_gap_eV=ref['gap_eV'],final_gap_eV=tr[-1]['readout']['gap_eV'],
             error_at_step3_or_earlier_meV=(third['readout']['gap_eV']-p['target_eV'])*1000,
             step_used_for_three_step_comparison=third['step'],
             final_target_error_meV=(tr[-1]['readout']['gap_eV']-p['target_eV'])*1000,
             target_reached=abs(tr[-1]['readout']['gap_eV']-p['target_eV'])<=p['limits']['target_tolerance_eV'],
             assigned_GPU_hours=gpuh,scope='Archived per-experiment assigned GPU walltime, not an exclusive-node speed benchmark'))
    result=dict(target_eV=p['target_eV'],target_tolerance_eV=p['limits']['target_tolerance_eV'],
        summaries=summaries,rows=rows,comparison_policy='Same initial state/target/observable units; compare up to3 accepted steps first, additional M7 steps4/5 separately; fresh-state numerical differences retained',
        interpretation='Reaching target with strains alone rules out any claim that internal coordinates are necessary for this target; a miss is limited to the registered bounds/step count.')
    (d/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    with (d/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result

"""Training-only sensitivity selection after excluding failed AlAs."""
from pathlib import Path
import json,datetime,sys
import numpy as np
from prepare_e7_training_only import select,NAMES,SCALES
P=Path(__file__).resolve().parents[3];E=P/'validation/results/response_design_20261015_e7_ct_pilot';A=E/'policy_20261015_1030'
reference=json.loads((P/'validation/results/response_design_20261015_e7_preparation/frozen_protocol.json').read_text())['references']
data={}
for material,observables in [('Si',['a0_A']),('BN',['B_GPa']),('diamond',['a0_A','B_GPa'])]:
    work=E/'runs'/(material+'_cheap');s=json.loads((work/'summary.json').read_text());j=json.loads((work/'equilibrium_jacobian.json').read_text())
    data[material]=dict(source=str(work),observables=observables,targets=reference[material]['primary'],
        gradients=dict(a0_A=(np.asarray(j['dlogV_equilibrium_dp'])*s['a0_A']/3).tolist(),B_GPa=j['dB_equilibrium_dp']))
rowfile=P/'validation/results/response_design_20261014_e1_alp/extension_20261015/response_row_k12.json';row=json.loads(rowfile.read_text())
data['AlP']=dict(source=str(rowfile),observables=['a0_A'],targets={'a0_A':reference['AlP']['review_recommended_a300_A']},
    gradients={'a0_A':row['da0_dp']},conditional_only=True,reference_gate_pending=True,E1_QE_parameter_FD_pending=True)

def identifiable(materials):
    n=sum(len(data[m]['observables']) for m in materials)
    try:return select(data,materials)
    except ValueError as exc:
        # A two-observable fold cannot independently identify three parameters.
        # Freeze the third at its untrained value; never use validation rows.
        rows=[]
        for mat in materials:
            d=data[mat]
            for obs in d['observables']:
                scale=(.005 if obs=='a0_A' else .05)*d['targets'][obs]*np.sqrt(len(d['observables']))
                rows.append(np.asarray(d['gradients'][obs])*SCALES/scale)
        j=np.asarray(rows);rank=int(np.linalg.matrix_rank(j[:,[5,6]]))
        if rank<2:raise
        return dict(materials=materials,observables_count=n,parameter_names=['x:kappa','x:mu'],
            third_parameter_frozen_at_zero=True,reason=str(exc),kappa_mu_rank=rank,
            validation_used_for_selection=False,heldout_results_read=False)

def plan(materials):
    return dict(materials=materials,full_training_selection=identifiable(materials),
        folds=[dict(validation=v,training=[m for m in materials if m!=v],selection=identifiable([m for m in materials if m!=v])) for v in materials],
        max_candidate_attempts=3,max_accepted_updates=3)

result=dict(time=datetime.datetime.now().isoformat(),excluded=['AlAs'],failure_gate_unchanged=True,
    three_material_fallback=plan(['Si','BN','diamond']),conditional_four_material=plan(['Si','BN','diamond','AlP']),
    AlP_required_gates=['E1 adopted-mesh QE','E1 independent parameter FD','frozen experimental reference consistency'],
    primary_target_masks_unchanged=True,heldout_result_files_opened=[],
    source_scope='Fresh E7 PBE core Jacobians; new NLCC E1 K12 AlP row conditional, no old SG15 AlP substitution')
(A/'reduced_parameter_selection.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
for name in ('three_material_fallback','conditional_four_material'):
    d=result[name];print(name,d['full_training_selection']['parameter_names'],
        [(f['validation'],f['selection']['parameter_names']) for f in d['folds']])

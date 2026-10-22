"""Repeat the registered strict reference-row screen without holdout access."""
from pathlib import Path
import datetime,json,copy
from prepare_e7_training_only import P,A,CORE,select
r=json.loads((A/'training_only_selection.json').read_text());data=copy.deepcopy(r['training_data'])
data['Si']['observables']=['a0_A'];data['BN']['observables']=['B_GPa'];data['BN']['targets']['B_GPa']=381.1
out=dict(time=datetime.datetime.now().isoformat(),policy='directly verified primary numerical rows only',
         core_only=select(data,CORE),heldout_results_read=False,
         rows_excluded_from_primary=['Si B: precise original pure-sample row not reverified','BN a0: numerical original row not reverified','AlP a0: source discrepancy; conditional only'])
out['conditional_LOO_plan']=[]
for validation in CORE+['AlP']:
 training=[x for x in CORE+['AlP'] if x!=validation]
 out['conditional_LOO_plan'].append(dict(fold_validation_material=validation,training_materials=training,
    selection=select(data,training),AlP_reference_gate_required=True))
out['candidate_five']=select(data,CORE+['AlP'])
(A/'verified_reference_row_selection_v2.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('E7严格实验主靶筛选：'+out['core_only']['chosen']['name']+'。\n五折仅以各自训练行选择通道，候选AlP参考门待关。\nSiC/GaP结果访问0，GPU提交0。')

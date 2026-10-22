"""Inventory owned response checkpoints; missing orbitals cannot be treated as passed."""
from pathlib import Path
import json, datetime
P=Path(__file__).resolve().parents[3];root=P/'validation/results/response_design_20261014_e2_audit';root.mkdir(exist_ok=True)
rows=[]
def add(label,source,orbitals=None):
 s=Path(source);o=Path(orbitals) if orbitals else None
 rows.append(dict(label=label,source=str(s),orbitals=str(o) if o else None,exists=s.exists(),checkpoint_exists=(s/'state.pkl').exists(),density_exists=(s/'density.npy').exists(),metadata_exists=(s/'metadata.json').exists(),independent_low_orbitals_exist=bool(o and (o/'orbitals.npy').exists()),status='待做',passed=False))
add('MgSiN2_start',P/'validation/results/response_design_20261002_resource_fill/runs/mgsin2_gamma_strain_ref_fs_r3',P/'validation/results/response_design_20261002_next_phase/orbitals/gamma_orbitals_strain_ref')
for label,base,name in [('scalar_reverse', 'response_design_20261004_response_design','reverse_history.json'),('scalar_forward','response_design_20261004_response_design','forward_history.json'),('six_reverse','response_design_20261006_six_strain_design_v2','reverse_history.json'),('six_forward','response_design_20261006_six_strain_design_v2','forward_history.json')]:
 f=P/'validation/results'/base/name
 if f.exists():
  for r in json.loads(f.read_text()):
   if r.get('iteration',0)>0 and 'source' in r:add(f'{label}_iter{r["iteration"]}',r['source'],r.get('orbitals'))
b=P/'validation/results/response_design_20261004_compact_training_v2/training/runs'
for label in ['trial0_Si','trial3_Si','trial0_BN','trial3_BN','holdout_PBE','holdout_learned']:
 add(label,b/label)
 for r in [rows[-1]]:
  r['stored_density_files']=[f.name for f in (b/label).glob('*density.npy')]
  if not r['checkpoint_exists']:r['risk']='Only density/point records saved; optimizer-orbital commutator cannot be measured retroactively. Independent reconstruction must be distinguished from the original state.'
(root/'inventory.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),scope='Preparation only; no physical acceptance inferred',states=rows),indent=2)+'\n')
(root/'README.md').write_text('# E2 本仓库关键态审计准备\n\n输入清单见 inventory.json。已存数据不覆盖；缺失轨道/低谱标为未验，不通过原压力残差推断全谱正确。校准态只有 density/mask 的情况，后续以该真实 density 构造完全相同功能的 KS 算符进行独立谱重占据，原优化器轨道 commutator 仍记为缺失。E0 完成前不启动正式重算。\n')
print(f'E2 输入清单：{len(rows)} 态；{sum(r["checkpoint_exists"] for r in rows)} 个轨道检查点；{sum(r["independent_low_orbitals_exist"] for r in rows)} 个已存低谱。')

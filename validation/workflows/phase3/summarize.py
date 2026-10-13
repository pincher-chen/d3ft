"""Rebuild tables from immutable point/run outputs; never infer missing success."""
import csv
from datetime import datetime
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RESULT=ROOT/'validation/results/phase3_20260908'

def missing_qe_elastic_rows(meta, summary):
    """QE writes a partial summary after each amplitude, before the run ends."""
    args=meta['arguments']
    if 'config' in meta or args.get('mode','elastic')!='elastic':
        return []
    material=json.loads(Path(__file__).with_name('materials.json').read_text())[args['material']]
    strains=json.loads(args.get('strain_steps','[0.01,0.005]'))
    branches=['D']+(['C44'] if material['C44'] is not None else [])
    return [f'{branch}_eps{h}' for h in strains for branch in branches
            if f'{branch}_eps{h}' not in summary]

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def main():
    complete=[]; progress=[]; forces=[]; baseline={}
    for folder in sorted(RESULT.iterdir()):
        if not folder.is_dir() or not (folder/'metadata.json').exists(): continue
        meta=json.loads((folder/'metadata.json').read_text()); args=meta['arguments']
        summary=folder/'summary.json'
        status=dict(tag=folder.name,material=args['material'],mode=args.get('mode','elastic'),
                    has_summary=summary.exists())
        if (folder/'points.json').exists(): status['completed_points']=len(json.loads((folder/'points.json').read_text()))
        progress.append(status)
        if not summary.exists(): continue
        d=json.loads(summary.read_text())
        missing=missing_qe_elastic_rows(meta,d)
        status['summary_complete']=not missing
        if missing:
            status['missing_observables']=missing
            continue
        if args.get('mode')=='evaluate' and 'parameter_vector' in d and np.all(np.asarray(d['parameter_vector'])==0):
            required=['a0_A','B_GPa']
            material=json.loads((Path(__file__).with_name('materials.json')).read_text())[args['material']]
            if material['C11'] is not None: required.append('C11_minus_C12_GPa')
            if material['C44'] is not None: required.append('C44_GPa')
            if all(key in d for key in required): baseline[args['material']]=folder.name
        row=dict(tag=folder.name,material=args['material'],mode=args.get('mode','elastic'),
                 grid=args.get('grid') or meta.get('config',{}).get('grid_sizes'),cutoff=args.get('cutoff'),
                 cutoff_unit='Ha' if 'config' in meta else 'Ry',
                 strain_step=args.get('strain_step',.01) if 'config' in meta else None,
                 strain_steps=args.get('strain_steps','[0.01,0.005]') if 'config' not in meta and args.get('mode','elastic')=='elastic' else None,
                 seed=args.get('seed'),epochs=d.get('epochs'),
                 a0_A=d.get('a0_A'),residual=d.get('residual',d.get('equilibrium_residual')),
                 pressure_GPa=d.get('pressure_GPa',d.get('equilibrium_pressure_GPa')))
        for key in ['energy_Ha','B_GPa','B_energy_GPa','B_stress_GPa','Bprime','C11_minus_C12_GPa','C44_GPa',
                    'D_eps0.01','D_eps0.005','C44_eps0.01','C44_eps0.005']:
            if key in d: row[key]=d[key]
        row.update({key:value for key,value in d.items() if key.startswith(('D_eps','C44_eps'))})
        complete.append(row)
        if not folder.name.startswith('stationarity_') or 'forces' not in d: continue
        f=np.asarray(d['forces']); maxf=np.abs(f).max()
        entry=dict(tag=folder.name,material=args['material'],grid=args.get('grid'),seed=args.get('seed'),
                   epochs=d['epochs'],residual=d['residual'],relative_asr=np.abs(f.sum(0)).max()/maxf,
                   atom0_fx=f[0,0],max_transverse=np.abs(f[:,1:]).max())
        refdir=RESULT/f"qe_force_{args['material']}"
        if (refdir/'summary.json').exists():
            q=json.loads((refdir/'summary.json').read_text())
            u=d['displacement_Bohr']; u=u[0] if isinstance(u,list) else u
            if abs(d['a0_A']-q['a0_A'])<1e-10 and abs(u-q['displacement_Bohr'])<1e-10:
                entry.update(QE_atom0_fx=q['forces'][0][0],force_error_pct=100*(f[0,0]/q['forces'][0][0]-1),
                             pressure_difference_GPa=d['pressure_GPa']-q['pressure_GPa'])
        forces.append(entry)
    write_csv(RESULT/'completed_runs.csv',complete)
    write_csv(RESULT/'stationarity_force_comparison.csv',forces)
    seed_checks=[]
    for material,grid,epochs in [('GaAs',52,3000),('GaP',48,9000)]:
        selected=[r for r in forces if r['material']==material and r['grid']==grid and r['epochs']==epochs]
        by_seed={r['seed']:r for r in selected}
        if not all(s in by_seed for s in [0,1,2]): continue
        selected=[by_seed[s] for s in [0,1,2]]
        raw=[json.loads((RESULT/r['tag']/'summary.json').read_text()) for r in selected]
        values=np.asarray([r['forces'] for r in raw])
        seed_checks.append(dict(material=material,grid=grid,epochs=epochs,tags=[r['tag'] for r in selected],
            max_residual=max(r['residual'] for r in selected),
            max_force_component_span=float(np.ptp(values,axis=0).max()),
            max_relative_asr=max(r['relative_asr'] for r in selected),
            energy_span_Ha=float(np.ptp([r['energy_Ha'] for r in raw])),
            pressure_span_GPa=float(np.ptp([r['pressure_GPa'] for r in raw])),
            mean_atom0_fx=float(values[:,0,0].mean()),
            mean_QE_force_error_pct=float(np.mean([r['force_error_pct'] for r in selected])),
            scope='Stationarity/seed diagnostic at this setting; cutoff/grid not certified'))
    (RESULT/'seed_force_checks.json').write_text(json.dumps(seed_checks,indent=2))
    (RESULT/'progress.json').write_text(json.dumps(progress,indent=2))
    queues=[]
    for path in sorted((RESULT/'queues').glob('*/status.json')):
        queues.append(dict(queue=path.parent.name,**json.loads(path.read_text())))
    lines=['# 当前 Phase 3 进度','',
           '更新时间：'+datetime.now().isoformat(timespec='seconds')+'（本地时间）。','',
           'Phase 3 尚未全部验收；summary 只说明某次工作流已写出结果，不代表数值收敛或迁移性已成立。',
           '完整要求和运行方式见 `validation/workflows/phase3/README.md`。','',
           '已有完整零参数基线（仍需独立收敛验收）：'+('、'.join(sorted(baseline)) or '无')+'。',
           '实际参数差分、完整非线性拟合和整材料留出检验分别验收，不由基线完成数推断。','',
           f'有 summary 的运行：{len(complete)}；元数据运行数：{len(progress)}。',
           '缺少 summary 的运行不得算作成功。QE 分步 summary 还需检查两个应变幅度是否齐全。','']
    if queues:
        lines += ['## 自动队列','', '| 队列 | 状态 | 等待 step / 退出码 |','|---|---|---|']
        lines += [f"| {q['queue']} | {q['state']} | {q.get('step',q.get('dependency',q.get('returncode','—')))} |" for q in queues]
        lines += ['', '队列状态不是科学验收；工作流退出后仍需检查完整结果和数值判据。','']
    lines += [
           '| 运行 | 材料 | summary | 已记录 D3FT 点 |','|---|---|---|---|']
    lines += [f"| {r['tag']} | {r['material']} | {'有' if r['has_summary'] else '无'} | {r.get('completed_points','—')} |" for r in progress]
    lines += ['', '数值汇总：`completed_runs.csv`；含 d 体系力对照：`stationarity_force_comparison.csv`。',
              '原始失败日志保留在 `logs/`。`evaluate_diamond_base` 是本次主动结束的单轴弛豫尝试，',
              '`evaluate_diamond_base_vector` 因原始净力不合格未完成；',
              '`evaluate_diamond_base_balanced` 保持几何中心，已完成金刚石基线。',
              '种子检查见 `seed_force_checks.json`；能量/压力截断检查见 `diamond_cutoff_certificate.json`。']
    (RESULT/'PROGRESS.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(complete)} summaries, {len(forces)} force diagnostics; progress tables regenerated')

if __name__=='__main__': main()

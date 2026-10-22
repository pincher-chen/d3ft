"""Fresh unique Si -> BN -> diamond controls; AlAs explicitly excluded."""
import e7_ct_pilot as p
p.priority_pending=lambda: None
p.note('E7新唯一名复核链启动：Si→BN→diamond；GPU1用于前两项，diamond等0/1/5/6空闲后四卡；GPU7专属E6，AlAs排除。')
for material in ('Si','BN','diamond'):
    gpus=[1] if material!='diamond' else [0,1,5,6]
    coarse=p.E/'runs'/p.selected_tag(material,'cheap')
    if not p.load(coarse/'native_acceptance.json').get('passed'):
        p.note('E7 '+material+'廉价原生门未通过，未提交加密复核。');continue
    work=p.run(material,'fine',gpus,21600 if len(gpus)==1 else 28800,
               p.load(coarse/'summary.json')['a0_A'],run_tag=material+'_fine')
    if work is not None and p.load(work/'native_acceptance.json').get('passed'):
        p.compare(material)
    p.report()
p.note('E7 Si/BN/diamond唯一名加密复核链结束，逐材料验收记录另查；AlAs未提交。')

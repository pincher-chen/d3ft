"""Audit completed six-strain controls, export source data and update both drafts.

Numerical failure stops publication and retains logs. No partial trajectory or
linear prediction is written as a successful design demonstration.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

P = Path(__file__).resolve()
while not (P / 'validation').is_dir():
    P = P.parent
D = P / 'paper_rewriting_output/final_paper'
M = P.parents[1] / 'd3ft2-nmi/papar/sn-article-template'


def read(path):
    return json.loads(path.read_text())


def dump(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def scientific_tex(value):
    if value==0:
        return '0'
    coefficient,exponent=f'{value:.2e}'.split('e')
    return coefficient+r'\times10^{'+str(int(exponent))+'}'


def update_marker(text, marker, content, anchor):
    begin, end = '% BEGIN '+marker, '% END '+marker
    block = begin+'\n'+content+'\n'+end
    if begin in text:
        start = text.index(begin)
        finish = text.index(end,start)+len(end)
        return text[:start]+block+text[finish:]
    if text.count(anchor)!=1:
        raise ValueError('Manuscript insertion anchor changed: '+anchor)
    return text.replace(anchor,block+'\n\n'+anchor,1)


def export(root):
    spec = importlib.util.spec_from_file_location('frozen_six_strain_design',root/'controller.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.audit(root)
    if not result['passed']:
        raise ValueError('Completed numerical/design comparison did not pass; original results retained')
    cfg=read(root/'protocol.json')
    histories={mode:read(root/(mode+'_history.json')) for mode in ['reverse','forward']}
    # Recompute gap from independently saved band eigenvalues, rather than only
    # trusting a controller-recorded gap or an expected target.
    evidence={}
    rows=[]
    for mode,history in histories.items():
        for row in history:
            source=Path(row['source']);orbital=Path(row['orbitals'])
            summary=read(orbital/'summary.json')
            energies=summary['eigenvalues_Ha']
            gap=(energies[48]-energies[47])*27.211386245988
            if abs(gap-row['gap_eV'])>1e-12:
                raise ValueError('Saved eigenvalues disagree with trajectory objective')
            if row['iteration']==len(history)-1 and abs(gap-cfg['target_gamma_eV'])>cfg['design_tolerance_eV']:
                raise ValueError('Terminal target gate failed')
            item=dict(mode=mode,iteration=row['iteration'],gap_eV=gap,error_meV=1000*(gap-cfg['target_gamma_eV']),
                step_norm=row.get('step_norm',0.),total_displacement_norm=row.get('total_displacement_norm',0.))
            item.update({name:value for name,value in zip(['xx','yy','zz','yz','xz','xy'],row['theta'])})
            rows.append(item)
            for path in [source/'point.json',source/'metadata.json',source/'density.npy',source/'mask.npy',source/'input.xyz',
                         orbital/'summary.json',orbital/'metadata.json',orbital/'orbitals.npy']:
                evidence[str(path)]=sha(path)
            if 'response' in row:
                response=Path(row['response'])
                for name in ['result.json','metadata.json']:
                    evidence[str(response/name)]=sha(response/name)
    evidence[str(root/'protocol.json')]=sha(root/'protocol.json')
    with (root/'six_strain_trajectory.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=result['summaries']
    reverse=histories['reverse'][-1];forward=histories['forward'][-1]
    rt=summaries['reverse']['elapsed_seconds'];ft=summaries['forward']['elapsed_seconds']
    ratio=ft/rt
    paragraph=(
        'An additional design frees all six engineering-strain components rather\n'
        'than restricting updates to the original scalar path. The same bounded\n'
        'minimum-norm update, starting state and target are used for independent\n'
        'reverse and same-program forward trajectories. '
        f'They reach {reverse["gap_eV"]:.9f} and {forward["gap_eV"]:.9f} eV in '
        f'{reverse["iteration"]} and {forward["iteration"]} accepted updates, respectively.\n'
        f'The largest componentwise trajectory difference is ${scientific_tex(result["maximum_component_trajectory_difference"])}$. '
        'Each complete gradient uses one electronic adjoint or six forward\n'
        'responses. Serial execution on the same physical H100 takes '
        f'{rt/3600:.3f} and {ft/3600:.3f} h from the shared qualified starting\n'
        f'state (forward-to-reverse time ratio {ratio:.3f}). '
        'These cold workflow times include compilation, electronic reoptimization,\n'
        'readout, launch overhead, source checks and I/O, and exclude construction\n'
        'of the shared initial state. They are one instance, separate from the\n'
        'warmed response-only comparison, and do not establish a universal\n'
        'design speedup or dense-mesh target accuracy (Supplementary Information).')
    lines=[r'\subsection{Six-component strain design control}',
        'Both controls start from the same qualified electronic state and saved',
        'band-edge orbitals; each recomputes its initial complete gradient.',
        'The scalar target remains the original $\\Gamma$ transition plus 0.1 eV.',
        'All six engineering-strain variables are free. With gradient $g$, the',
        'unconstrained proposal is $(E_{\\rm target}-E_g)g/(g^Tg)$, clipped to an',
        'L2 step radius of 0.01 and total displacement radius of 0.03 around the',
        'initial strain. There are at most three accepted updates and two',
        'successive halvings per proposal. Acceptance requires a decrease of',
        'the actually reoptimized target error, and stopping requires the',
        'original 1-meV budget. Fractional coordinates, occupations, Miller set,',
        'PBE, cutoff and quadratures remain fixed. Eigenpair, stationary-state,',
        'charge, raw force-sum, response residual and band-edge separation gates',
        'are checked at each applicable step. The single-target problem does not',
        'uniquely determine a six-strain endpoint; the minimum-norm rule selects',
        'the local update rather than claiming a unique structural optimum.',
        '',r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Actual six-strain design histories. Errors refer to the unchanged $\Gamma$ target; no full-zone band-gap claim is made.}',
        r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Mode & Update & $E_g^\Gamma$ (eV) & Error (meV) & Step norm & Total norm \\',r'\midrule']
    for row in rows:
        lines.append(f'{row["mode"].capitalize()} & {row["iteration"]} & {row["gap_eV"]:.9f} & {row["error_meV"]:.6f} & {row["step_norm"]:.7f} & {row["total_displacement_norm"]:.7f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Final engineering strains from independent reverse and forward controls. Shear entries are twice the corresponding symmetric-matrix off-diagonal entries.}',
        r'\begin{tabular}{lrr}',r'\toprule',r'Component & Reverse & Forward \\',r'\midrule']
    for index,label in enumerate(['xx','yy','zz','yz','xz','xy']):
        lines.append(f'${label}$ & {reverse["theta"][index]:.10f} & {forward["theta"][index]:.10f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Serial same-GPU cold workflow times, in seconds, excluding the shared initial-state construction. Response subprocess times include reconstruction and first-use compilation. Total time also includes controller checks and I/O.}',
        r'\begin{tabular}{lrrrr}',r'\toprule',r'Mode & Response & Ground state & Readout & Total \\',r'\midrule']
    for mode,summary in summaries.items():
        lines.append(f'{mode.capitalize()} & {summary["response_seconds"]:.2f} & {summary["ground_seconds"]:.2f} & {summary["readout_seconds"]:.2f} & {summary["elapsed_seconds"]:.2f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}','',
        'This comparison uses the same-program forward baseline, with the existing',
        'native AD--DFPT gradient and scalar-path controls retained separately.',
        'No CPU/GPU cross-code cost is attributed to AD direction. There is one',
        'trajectory per mode; timing uncertainty, mode-specific peak memory and',
        'continuous-zone target transfer are not established.','']
    fragment='\n'.join(lines)
    (root/'six_strain_design.tex').write_text(fragment)
    snapshot=root/'paper_before_six_strain'
    snapshot.mkdir(exist_ok=False)
    for directory,stem in [(D,'main'),(M,'sn-article')]:
        backup=snapshot/('canonical' if directory==D else 'mirror');backup.mkdir()
        for name in [stem+'.tex','supplementary.tex']:
            shutil.copy2(directory/name,backup/name)
        text=(directory/(stem+'.tex')).read_text()
        text=update_marker(text,'SIX STRAIN DESIGN',paragraph,'% BEGIN MAIN BAND EVIDENCE')
        (directory/(stem+'.tex')).write_text(text)
        shutil.copy2(root/'six_strain_design.tex',directory/'six_strain_design.tex')
        shutil.copy2(root/'six_strain_trajectory.csv',directory/'six_strain_trajectory.csv')
        si=(directory/'supplementary.tex').read_text()
        si=update_marker(si,'SIX STRAIN SUPPLEMENT',r'\input{six_strain_design.tex}',
            r'\subsection{Projected response implementation and cost scope}')
        (directory/'supplementary.tex').write_text(si)
    builds=[]
    for directory,stem in [(D,'main'),(D,'supplementary'),(M,'sn-article'),(M,'supplementary')]:
        command=[sys.executable,str(D/'build_response_paper.py'),'--directory',str(directory),'--stem',stem]
        tag=('canonical' if directory==D else 'mirror')+'_'+stem
        with (root/('publication_build_'+tag+'.log')).open('xb') as log:
            rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        builds.append(dict(directory=str(directory),stem=stem,returncode=rc))
        if rc:
            raise RuntimeError('Document build failed; original snapshot retained: '+tag)
    report=dict(time=datetime.datetime.now().isoformat(),passed=True,original_target_failures_preserved=True,
        submission_ready=False,source_hashes=evidence,builds=builds,result=result,
        new_artifacts={str(root/n):sha(root/n) for n in ['six_strain_trajectory.csv','six_strain_design.tex']})
    dump(root/'publication_acceptance.json',report)
    for directory in [D,M]:
        dump(directory/'SIX_STRAIN_DESIGN_ACCEPTANCE_20261006.json',report)
    todo=(P/'response_design_to_do_list.md').read_text()
    todo=todo.replace('- [ ] **P2b｜六维自由应变设计对照。**','- [x] **P2b｜六维自由应变设计对照。** 已完成真实双轨迹、源审核及正文/SI同步；原协议：')
    todo+='\n\n## '+report['time']+'｜六维设计完成并入文\n\n'
    todo+=f'两模式达到同一原目标，reverse/forward分别{reverse["iteration"]}/{forward["iteration"]}个接受更新；串行同物理GPU冷流程时间分别{rt/3600:.3f}/{ft/3600:.3f}h。独立源/实际谱/梯度/轨迹审核通过；原单变量密网格FAIL保持，六维结果不声明密网格精确命中。正文及SI表已更新并完成四份构建。submission_ready仍false。\n'
    for path in [P/'response_design_to_do_list.md',D/'response_design_to_do_list.md',M/'response_design_to_do_list.md',P.parents[1]/'d3ft2-nmi/response_design_to_do_list.md']:
        path.write_text(todo)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    try:
        while args.wait:
            status=read(root/'status.json')
            if status['stage'] in ['completed','failed']:
                break
            dump(root/'publication_status.json',dict(stage='waiting_for_both_trajectories',
                time=datetime.datetime.now().isoformat(),current_task=status))
            time.sleep(30)
        if read(root/'status.json')['stage']!='completed':
            raise RuntimeError('Numerical campaign failed; publication stopped')
        dump(root/'publication_status.json',dict(stage='auditing',time=datetime.datetime.now().isoformat()))
        report=export(root)
        dump(root/'publication_status.json',dict(stage='completed',time=report['time'],passed=True,submission_ready=False))
    except Exception as exc:
        dump(root/'publication_status.json',dict(stage='failed',time=datetime.datetime.now().isoformat(),error=str(exc),submission_ready=False))
        raise

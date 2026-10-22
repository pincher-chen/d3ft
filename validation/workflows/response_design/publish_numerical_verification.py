"""Wait for both bounded campaigns, archive results, and update the SI honestly."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261005_numerical_verification'
D=P/'paper_rewriting_output/final_paper'
N=P.parents[1]/'d3ft2-nmi'
M=N/'papar/sn-article-template'
sys.path.insert(0,str(P/'validation/workflows/response_design'))
from completed_numerical_evidence import collect,gradient_tex,kmesh_tex,partial_rows
_completed_cache=None

def completed_evidence():
    global _completed_cache
    if _completed_cache is None:
        _completed_cache=collect()
        dump(R/'completed_evidence_snapshot.json',_completed_cache)
    return _completed_cache

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)

def summary():
    active=read(R/'gradient_active.json');groot=Path(active['root'])
    kroot=Path(read(R/'kmesh_active.json')['root'])
    statuses={'gradient':read(groot/'gradient_status.json'),'kmesh':read(kroot/'status.json')}
    result={name:read(path) if path.exists() else None for name,path in {
        'gradient':groot/'gradient_result.json','kmesh':kroot/'result.json'}.items()}
    return groot,statuses,result

def audit_bindings(groot, results):
    partial=completed_evidence()
    checks=[dict(path=path,passed=True,source='completed-point collector source gates')
            for path in partial['source_hashes']]
    if results['gradient']:
        cfg=read(groot/'gradient_protocol.json')
        for path,h in {**cfg['parent_hashes'],**cfg['source_hashes'],**cfg.get('recovery_source_hashes',{})}.items():
            checks.append(dict(path=path,passed=sha(path)==h))
        production=read(Path(cfg['source'])/'metadata.json')['code_sha256']
        for name in ['fine_reference','plus_h','minus_h','plus_half','minus_half']:
            source=groot/'runs'/name;orb=groot/'orbitals'/name
            binding=read(orb/'metadata.json')['source_hashes']
            for filename,h in binding.items():checks.append(dict(path=str(source/filename),passed=sha(source/filename)==h))
            checks.append(dict(path=str(source/'metadata.json'),passed=read(source/'metadata.json')['code_sha256']==production))
            checks.append(dict(path=str(source/'mask.npy'),passed=sha(source/'mask.npy')==sha(Path(cfg['source'])/'mask.npy')))
    if results['kmesh']:
        kroot=Path(read(R/'kmesh_active.json')['root']);cfg=read(kroot/'protocol.json')
        patch=read(kroot/'runtime_patch.json');expected=dict(patch['original_core'])
        expected['d3ft/calc/ground_state.py']=patch['new_ground_state_sha256']
        expected['d3ft/calc/compact_k_terms.py']=patch['compact_kernel_sha256']
        for state,s in cfg['states'].items():
            parent=Path(s['source'])
            for filename,h in s['source_sha256'].items():checks.append(dict(path=str(parent/filename),passed=sha(parent/filename)==h))
            source=Path(results['kmesh']['k6'][state]['source'])
            for filename,h in results['kmesh']['k6'][state]['source_hashes'].items():
                checks.append(dict(path=str(source/filename),passed=sha(source/filename)==h))
            checks.append(dict(path=str(source/'metadata.json'),passed=read(source/'metadata.json')['code_sha256']==expected))
            checks.append(dict(path=str(source/'mask.npy'),passed=sha(source/'mask.npy')==sha(parent/'mask.npy')))
    report=dict(passed=all(c['passed'] for c in checks),checks=checks,
                time=datetime.datetime.now().isoformat())
    dump(R/'source_binding_audit.json',report)
    if not report['passed']:raise RuntimeError('Final source/core binding audit failed; no TeX integration')

def make_report(statuses,results):
    rows=[]
    k=results['kmesh'];g=results['gradient']
    if k:
        for name in ['start','reverse']:
            for quantity in ['gap_eV','gamma_gap_eV']:
                rows.append(dict(group='kmesh',metric=name+'_'+quantity,
                    production=k['k4'][name][quantity],refined=k['k6'][name][quantity],
                    difference=k['k6'][name][quantity]-k['k4'][name][quantity],unit='eV'))
        rows.append(dict(group='kmesh',metric='gap_gain',production=k['k4_gain_eV'],
            refined=k['k6_gain_eV'],difference=k['gain_setting_difference_eV'],unit='eV'))
    if g:
        for metric,coarse,fine in [('directional_gradient','production_directional_gradient_eV','fine_directional_gradient_eV'),
                                   ('Gamma_gap','production_gamma_gap_eV','fine_gamma_gap_eV')]:
            rows.append(dict(group='XC_quadrature',metric=metric,production=g[coarse],
                 refined=g[fine],difference=g[fine]-g[coarse],unit='eV/strain' if metric=='directional_gradient' else 'eV'))
        rows.append(dict(group='derivative_validation',metric='fine_ad_vs_half_step_FD',
             production=g['fine_directional_gradient_eV'],refined=g['fine_fd_half_eV'],
             difference=g['fine_fd_half_eV']-g['fine_directional_gradient_eV'],unit='eV/strain'))
    partial=completed_evidence()
    rows+=partial_rows(partial,gradient=not bool(g),kmesh=not bool(k))
    with (R/'numerical_verification.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['group','metric','production','refined','difference','unit'])
        writer.writeheader();writer.writerows(rows)
    lines=['# MgSiN2 representative numerical verification','',
           'This supplement does not add a multi-k adjoint or change the original ±1-meV objective.',
           'Pairwise gap/gain budget: 5 meV; directional-gradient budget: 1% + 1e-4 eV/strain.',
           'These were fixed before runs. They are numerical robustness criteria, not journal standards.','',
           '| Group | Terminal state | Scientific checks |','|---|---|---|']
    for name in ['gradient','kmesh']:
        result=results[name]
        label='PASS' if result and result['passed'] else 'FAIL' if result else 'PARTIAL DATA; GROUP INCOMPLETE' if partial[name] else 'NO ADMITTED RESULT'
        lines.append(f"| {name} | {statuses[name]['stage']} | {label} |")
    lines+=['','| Metric | Production | Refined | Difference | Unit |',
            '|---|---:|---:|---:|---|']
    lines += [f"| {r['metric']} | {r['production']:.9g} | {r['refined']:.9g} | {r['difference']:.9g} | {r['unit']} |" for r in rows]
    if k:
        lines+=['',f"Original design-target transfer at actual reverse k6 endpoint: error {k['original_target_error_eV']*1000:.6f} meV; "+
                 ('PASS' if k['original_target_pass'] else 'FAIL')+' under unchanged ±1-meV criterion.']
    lines+=['','Fixed Miller masks; fixed ions; the cutoff is not refined in this group.',
            'Completed XC720 FD and k6 starting-state evidence is retained even if the full campaign fails. Production-AD/refined-FD comparison is not a completed refined adjoint.',
            'The compact kernel is admitted only after independent complex-gradient formula parity and physical k4 checkpoint parity.',
            'No new k6 force/stress/ASR claim; stationarity, orthogonality, charge and spectrum eigen-residual gates are retained.']
    text='\n'.join(lines)+'\n';(R/'README_results.md').write_text(text)
    (D/'response_NUMERICAL_VERIFICATION_20261005.md').write_text(text)
    return rows

def tex_fragment(results):
    lines=[r'\subsection{Representative numerical refinement checks}',
           r'\label{sec:representative-refinement}',
           'This group isolates Brillouin-zone sampling at the start and actual reverse-designed',
           'MgSiN$_2$ cells, and XC quadrature at one representative $\\Gamma$ response point.',
           'The pairwise gap and gap-gain budgets are 5 meV; the directional-gradient budget',
           'is 1\\% plus $10^{-4}$ eV per unit strain. These budgets were fixed before execution',
           'and are separate from the unchanged $\\pm1$-meV absolute design-target criterion.']
    k=results['kmesh'];g=results['gradient']
    if k:
        lines += ['The $4^3$ and $6^3$ calculations use identical accepted geometries,',
          'the common 55,913-vector Miller basis, FFT $144^3$, XC quadrature $576^3$,',
          'and independently reoptimized densities. All 216 $6^3$ spectrum points are evaluated.',
          'To bound memory, compact occupied coefficients are expanded in 16-band FFT blocks.',
          'The compact formulas pass an independent full-FFT energy, density and complex-gradient',
          'comparison and a physical production-checkpoint comparison before the $6^3$ runs.',
          'Electronic stationarity, orthogonality, charge and eigen-residual gates are required;',
          'forces, stress and their acoustic-sum-rule check are not evaluated for these new rows.', '',
          r'\begin{table}[htbp]\centering',
          r'\caption{Pairwise sampling check at fixed start and actual reverse geometries. Gaps are in eV.}',
          r'\begin{tabular}{lrr}\toprule State or quantity & $4^3$ & $6^3$ \\ \midrule']
        for name,label in [('start','Start'),('reverse','Actual reverse')]:
            lines.append(f"{label}, mesh gap & {k['k4'][name]['gap_eV']:.8f} & {k['k6'][name]['gap_eV']:.8f} \\\\")
            lines.append(f"{label}, $\\Gamma$ gap & {k['k4'][name]['gamma_gap_eV']:.8f} & {k['k6'][name]['gamma_gap_eV']:.8f} \\\\")
        lines += [f"Mesh-gap gain & {k['k4_gain_eV']:.8f} & {k['k6_gain_eV']:.8f} \\\\",
                  r'\bottomrule\end{tabular}\end{table}', '']
        outcome='passes' if k['passed'] else 'does not pass'
        lines.append(f"The sampling comparison {outcome} all registered pairwise robustness gates.")
        target='passes' if k['original_target_pass'] else 'fails'
        lines.append(f"The actual reverse endpoint has a $\\Gamma$ target error of {k['original_target_error_eV']*1000:.6f} meV at $6^3$ and {target} the unchanged absolute target gate.")
        lines += ['This is a check of two discrete sampling settings, not a continuum error bound;',
                  '$4^3$ and $6^3$ are not nested meshes. No multi-$k$ reverse update is inferred.', '']
    if g:
        gradient_root=Path(read(R/'gradient_active.json')['root'])
        recovery=read(gradient_root/'gradient_protocol.json').get('recovery',{})
        if recovery.get('staged_xc_geometry_parity'):
            lines += ['At refined XC quadrature, the mixed geometry derivative is evaluated',
                'with a staged Fourier/local AD chain rule. Linear density interpolation and',
                'the original complex Cartesian spatial derivatives are computed separately;',
                'local mixed AD contractions are accumulated in bounded chunks, followed by',
                'the pullback of the volume ratio and reciprocal deformation matrix.',
                'Independent six-component comparisons with the original whole-program mixed AD',
                'verify equivalence at two refinement factors and skew, strained cells.',
                'The full-basis electronic adjoint is still solved once. No finite differences,',
                'quadrature change or response truncation is used. Refined execution timings',
                'do not replace the production forward/reverse timing comparison.', '']
        if recovery.get('geometry_derivative_parity'):
            lines += ['At the refined setting, the final scalar geometry derivative is evaluated',
                'by six sequential exact directional AD contractions to bound FFT working storage.',
                'The same full-basis electronic adjoint is solved once; these local contractions',
                'do not introduce additional electronic-response solves or finite differences.',
                'Independent mixed PBE, Hartree and complex-projector comparisons verify equivalence',
                'to the original scalar reverse derivative. The production timing comparison',
                'is unchanged and is not a timing claim for this refined execution layout.']
        if recovery.get('compact_reciprocal_grid_parity'):
            lines += ['For the refined response, reciprocal vectors are represented by the three',
                'one-dimensional frequency arrays and the reciprocal basis, and each Cartesian',
                'component is formed inside the corresponding density-gradient FFT.',
                'This avoids a dense three-component fine-grid tensor and its geometry pullback.',
                'The original scalar reverse derivative and the single electronic adjoint are retained.',
                'Independent energy, density/geometry gradient and mixed Hessian-vector comparisons',
                'verify equivalence without changing Nyquist splitting, PBE or quadrature.']
        lines += ['For the response check, the ordered Miller basis, FFT $144^3$, pseudopotentials,',
          'cell and input direction are unchanged while XC quadrature is refined from $576^3$',
          'to $720^3$. The refined state and the four displaced states are independently reoptimized.',
          'The refined adjoint is compared with central finite differences at $h=10^{-3}$',
          'and $h/2$, with the Richardson value retained as an additional diagnostic.',
          'The three Cartesian density-gradient FFTs are evaluated sequentially with their',
          'original complex magnitudes retained. Independent discrete energy, first-derivative',
          'and Hessian-vector comparisons verify this execution-layout change.', '',
          r'\begin{table}[htbp]\centering',
          r'\caption{One directional response check. Derivatives are in eV per unit strain.}',
          r'\begin{tabular}{lr}\toprule Method or setting & Derivative \\ \midrule',
          f"Production adjoint & {g['production_directional_gradient_eV']:.9f} \\\\",
          f"Refined adjoint & {g['fine_directional_gradient_eV']:.9f} \\\\",
          f"Refined FD, $h$ & {g['fine_fd_h_eV']:.9f} \\\\",
          f"Refined FD, $h/2$ & {g['fine_fd_half_eV']:.9f} \\\\",
          f"Refined Richardson FD & {g['fine_fd_richardson_eV']:.9f} \\\\",
          r'\bottomrule\end{tabular}\end{table}', '',
          f"This comparison {'passes' if g['passed'] else 'does not pass'} all registered quadrature/derivative gates.",
          'It validates one $\\Gamma$ input direction; it does not certify all derivative components,',
          'cutoff convergence or an analytic response on a dense mesh.']
    partial=completed_evidence()
    if not g and partial['gradient']:lines.append(gradient_tex(partial['gradient']))
    if not k and partial['kmesh']:lines.append(kmesh_tex(partial['kmesh']))
    return '\n'.join(lines)+'\n'

def refresh_main_completed_blocks(text,results):
    def replace(start,end,body):
        nonlocal text
        if start not in text:return
        a=text.index(start);b=text.index(end,a)+len(end)
        text=text[:a]+start+'\n'+body+'\n'+end+text[b:]
    g=results['gradient'];k=results['kmesh']
    if g:
        difference=abs(g['fine_directional_gradient_eV']-g['fine_fd_half_eV'])
        body=(f"At refined $720^3$ XC quadrature, the self-consistent adjoint projection was\n"
              f"${g['fine_directional_gradient_eV']:.9f}$ eV per unit strain, compared with\n"
              f"${g['fine_fd_half_eV']:.9f}$ from the independently reoptimized finer-step FD.\n"
              f"Their absolute difference was {difference:.9f} eV per unit strain.\n"
              f"The registered refinement group {'passed' if g['passed'] else 'did not pass'} all numerical gates\n"
              "(Supplementary Information). This remains a one-direction refinement check.\n")
        replace('% BEGIN COMPLETED XC REFINEMENT','% END COMPLETED XC REFINEMENT',body)
    if k:
        body=(f"Complete $6^3$ spectra gave sampled gaps of {k['k6']['start']['gap_eV']:.9f}\n"
              f"and {k['k6']['reverse']['gap_eV']:.9f} eV at the start and actual reverse endpoint,\n"
              f"respectively, a gain of {k['k6_gain_eV']:.9f} eV.\n"
              f"The reverse endpoint's $\\Gamma$ target error was {k['original_target_error_eV']*1000:.6f} meV;\n"
              f"it {'passed' if k['original_target_pass'] else 'failed'} the unchanged 1-meV target budget.\n"
              f"The registered pairwise sampling checks {'passed' if k['passed'] else 'did not all pass'}\n"
              "(Supplementary Information). These two discrete settings do not establish continuum convergence.\n")
        replace('% BEGIN COMPLETED K6 START','% END COMPLETED K6 START',body)
    return text

def publish():
    started=time.monotonic()
    while True:
        groot,statuses,results=summary()
        dump(R/'publication_status.json',dict(stage='waiting_campaigns',time=datetime.datetime.now().isoformat(),
            pid=os.getpid(),campaigns=statuses))
        terminal={'completed','failed','dependency_failed'}
        if all(v['stage'] in terminal for v in statuses.values()):break
        if time.monotonic()-started>60*3600:raise RuntimeError('Registered publication wait exceeded60 hours')
        time.sleep(30)
    audit_bindings(groot,results)
    rows=make_report(statuses,results)
    scientific_complete=all(results.values())
    partial=completed_evidence()
    if any(results.values()) or any(partial[name] for name in ['gradient','kmesh']):
        fragment=tex_fragment(results)
        for directory in [D,M]:
            if not directory.exists():continue
            backup=R/'paper_before_integration'/directory.name;backup.mkdir(parents=True,exist_ok=True)
            for name in ['main.tex','sn-article.tex','supplementary.tex']:
                f=directory/name
                if f.exists() and not (backup/name).exists():shutil.copy2(f,backup/name)
            (directory/'numerical_verification.tex').write_text(fragment)
            si=directory/'supplementary.tex';text=si.read_text()
            if '\\input{numerical_verification}' not in text:
                needle='\\bibliography{'
                if text.count(needle)!=1:raise RuntimeError('SI bibliography insertion anchor changed')
                position=text.index(needle)
                text=text[:position]+'\\input{numerical_verification}\n\n'+text[position:]
                si.write_text(text)
            main=directory/('main.tex' if directory==D else 'sn-article.tex');text=main.read_text()
            text=refresh_main_completed_blocks(text,results)
            main.write_text(text)
            marker='% Representative numerical verification 20261005'
            if marker not in text:
                needle='\\subsection{Calibration and evidence preservation}'
                if text.count(needle)!=1:raise RuntimeError('Methods insertion anchor changed')
                paragraph=marker+'\nAdditional representative sampling and XC-quadrature checks, including their\nregistered pairwise budgets and all pass/fail outcomes, are reported in the\nSupplementary Information. These comparisons retain the original design target\nand do not constitute a continuum-convergence or multi-$k$ adjoint certificate.\n\n'
                main.write_text(text.replace(needle,paragraph+needle,1))
        build=D/'build_response_paper.py';builds=[]
        for directory,stem in [(D,'main'),(D,'supplementary'),(M,'sn-article'),(M,'supplementary')]:
            if not directory.exists():continue
            with (R/f'publication_build_{groot.name}_{directory.name}_{stem}.log').open('xb') as log:
                code=subprocess.run([sys.executable,str(build),'--directory',str(directory),'--stem',stem],stdout=log,stderr=subprocess.STDOUT).returncode
            builds.append(dict(directory=str(directory),stem=stem,returncode=code))
        dump(R/'publication_builds.json',builds)
        if any(b['returncode'] for b in builds):raise RuntimeError('Numerical supplement build failed; data retained')
    appendix='\n## 2026-10-05｜代表性数值验证终态\n\n'
    for name,label in [('gradient','XC网格/单方向梯度'),('kmesh','k4→k6起点与实际反向终点')]:
        r=results[name]
        status='完成且通过预注册数值门' if r and r['passed'] else '完成但存在未通过的数值门' if r else '计算组未完成；已验收的部分结果保留于TeX' if partial[name] else '执行失败，未产生可验收结果'
        appendix+=f'- {label}：{status}。\n'
    appendix+='\n详见 `paper_rewriting_output/final_paper/response_NUMERICAL_VERIFICATION_20261005.md`。结果与失败如实保留；本组不自动关闭P2纯反向密网格目标，也不关闭全部截断能/导数矩阵覆盖。\n'
    todo=P/'response_design_to_do_list.md';text=todo.read_text()
    if '## 2026-10-05｜代表性数值验证终态' not in text:todo.write_text(text+appendix)
    for dst in [D/'response_design_to_do_list.md',N/'response_design_to_do_list.md',M/'response_design_to_do_list.md']:
        if dst.parent.exists():shutil.copy2(todo,dst)
    kroot=Path(read(R/'kmesh_active.json')['root'])
    hashes={str(f):sha(f) for f in [R/'numerical_verification.csv',R/'README_results.md',groot/'gradient_protocol.json',kroot/'protocol.json']}
    dump(R/'publication_status.json',dict(stage='completed',time=datetime.datetime.now().isoformat(),
        scientific_campaigns_complete=scientific_complete,
        all_numerical_checks_passed=scientific_complete and all(r['passed'] for r in results.values()),
        original_target_failures_retained=True,results_present={k:bool(v) for k,v in results.items()},
        completed_evidence_present={k:bool(partial[k]) for k in ['gradient','kmesh']},
        hashes=hashes,submission_ready=False))

if __name__=='__main__':
    try:publish()
    except Exception as exc:
        dump(R/'publication_status.json',dict(stage='failed',error=str(exc),time=datetime.datetime.now().isoformat()))
        raise

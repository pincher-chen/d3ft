"""Admit completed numerical evidence separately from campaign completion."""
import datetime
import hashlib
import json
import math
from pathlib import Path

P=Path('${D3FT_ROOT}')
R=P/'validation/results/response_design_20261005_numerical_verification'
HA_EV=27.211386245988

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as handle:
        for block in iter(lambda:handle.read(16*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def sci(value):
    mantissa,exponent=format(value,'.3e').split('e')
    return f"{mantissa}{chr(92)}times10^{{{int(exponent)}}}"

def collect():
    hashes={str(Path(__file__).resolve()):sha(__file__)}
    def bind(path,expected=None):
        path=Path(path);actual=sha(path)
        if expected is not None and actual!=expected:raise ValueError('Evidence binding changed: '+str(path))
        hashes[str(path)]=actual
    def require(condition,message):
        if not condition:raise ValueError(message)
    data=dict(time=datetime.datetime.now().isoformat(),gradient=None,kmesh=None,source_hashes=hashes,
              scope='Completed qualified points only; campaign completion is tracked separately.')
    groot=Path(read(R/'gradient_active.json')['root']);cfg=read(groot/'gradient_protocol.json')
    bind(groot/'gradient_protocol.json')
    for path,h in {**cfg['parent_hashes'],**cfg['source_hashes'],**cfg.get('recovery_source_hashes',{})}.items():bind(path,h)
    production=Path(cfg['source']);core=read(production/'metadata.json')['code_sha256'];mask=sha(production/'mask.npy')
    names=['fine_reference','plus_h','minus_h','plus_half','minus_half']
    if all((groot/'runs'/n/'point.json').exists() and (groot/'orbitals'/n/'summary.json').exists() for n in names):
        gaps={};checks=[]
        for name in names:
            src=groot/'runs'/name;orb=groot/'orbitals'/name;p=read(src/'point.json');m=read(src/'metadata.json');s=read(orb/'summary.json')
            require(p['stationarity_pass'] and p['residual']<=1e-6,'Stationarity failed: '+name)
            require(p['acoustic_sum_rule_Ha_Bohr']<=5e-8 and abs(p['integrated_valence_density']-96)<=.00096,'ASR/charge failed: '+name)
            require(m['code_sha256']==core and sha(src/'mask.npy')==mask,'Core/basis failed: '+name)
            require(max(s['residuals_Ha'])<=1e-7 and s['orthogonality_error']<=1e-8,'Eigenpair failed: '+name)
            require(all(math.isfinite(v) for v in s['eigenvalues_Ha']+s['residuals_Ha']),'Nonfinite spectrum: '+name)
            bind(src/'state.pkl',m['state_sha256'])
            for filename,h in read(orb/'metadata.json')['source_hashes'].items():bind(src/filename,h)
            bind(orb/'orbitals.npy',s['saved_orbitals_sha256']);bind(orb/'metadata.json');bind(orb/'summary.json')
            gaps[name]=s['gap_eV'];checks.append(dict(name=name,residual=p['residual'],ASR=p['acoustic_sum_rule_Ha_Bohr'],charge=p['integrated_valence_density'],eigen_residual=max(s['residuals_Ha'])))
        old=P/'validation/results/response_design_20261004_reverse_response_v5/reverse/result.json';bind(old)
        olddata=read(old);require(olddata['results']['response_record']['converged'],'Production adjoint not converged')
        gradient=olddata['results']['gap_gradient_eV'];production_ad=sum(x*y for x,y in zip(gradient,cfg['direction']))
        coarse=Path(cfg['orbitals'])/'summary.json';bind(coarse)
        fd_h=(gaps['plus_h']-gaps['minus_h'])/.002;fd_half=(gaps['plus_half']-gaps['minus_half'])/.001
        data['gradient']=dict(production_directional_gradient_eV=production_ad,fine_fd_h_eV=fd_h,
            fine_fd_half_eV=fd_half,fine_fd_richardson_eV=(4*fd_half-fd_h)/3,
            production_gamma_gap_eV=read(coarse)['gap_eV'],fine_gamma_gap_eV=gaps['fine_reference'],
            step_difference_eV=fd_half-fd_h,step_relative_difference=abs(fd_half-fd_h)/abs(fd_half),
            production_vs_fine_fd_difference_eV=fd_half-production_ad,checks=checks,
            refined_adjoint_available=(groot/'gradient_result.json').exists(),
            scope='Production adjoint at XC576 versus reoptimized FD at XC720; one input direction, not a completed XC720 adjoint.')
    kroot=Path(read(R/'kmesh_active.json')['root']);accepted=kroot/'start_accepted_source.json'
    if accepted.exists():
        bind(accepted);cfg=read(kroot/'protocol.json');bind(kroot/'protocol.json');a=read(accepted);src=Path(a['source'])
        bind(src/'point.json',a['point_sha256']);p=read(src/'point.json');m=read(src/'metadata.json')
        require(p['stationarity_pass'] and p['residual']<=1e-6 and p['orthogonality_max_error']<=1e-8,'k6 primal gate failed')
        require(abs(p['integrated_valence_density']-96)<=.00096,'k6 charge failed')
        bind(src/'state.pkl',m['state_sha256']);parent=Path(cfg['states']['start']['source'])
        for filename,h in cfg['states']['start']['source_sha256'].items():bind(parent/filename,h)
        require(sha(src/'mask.npy')==sha(parent/'mask.npy'),'k4/k6 masks differ')
        patch=read(kroot/'runtime_patch.json');expected=dict(patch['original_core'])
        expected['d3ft/calc/ground_state.py']=patch['new_ground_state_sha256'];expected['d3ft/calc/compact_k_terms.py']=patch['compact_kernel_sha256']
        require(m['code_sha256']==expected,'k6 core manifest differs')
        gamma_rows=[]
        for folder in sorted((kroot/'spectra').glob('start_gpu*')):
            for f in folder.glob('k_*.json'):
                row=read(f)
                if max(abs(x) for x in row['k_frac'])<1e-10:gamma_rows.append((folder,f,row))
        if gamma_rows:
            require(len(gamma_rows)==1,'Nonunique Gamma row');folder,f,row=gamma_rows[0]
            require(max(row['residuals_Ha'])<=1e-7 and all(math.isfinite(x) for x in row['eigenvalues_Ha']),'Gamma eigenpair failed')
            bind(folder/'metadata.json');binding=read(folder/'metadata.json')
            for filename,h in binding['source']['source_sha256'].items():bind(src/filename,h)
            bind(f);oldfile=Path(cfg['states']['start']['spectrum']);bind(oldfile);old=read(oldfile)
            require(old['all_registered_gates_passed'] and old['full_mesh'],'Original k4 spectrum unqualified')
            bind(parent/'density.npy',old['density_sha256']);bind(parent/'mask.npy',old['mask_sha256'])
            oldpoint=read(parent/'point.json');gap=(row['cbm_Ha']-row['vbm_Ha'])*HA_EV
            data['kmesh']=dict(k4_start_gamma_gap_eV=old['gamma_gap_eV'],k6_start_gamma_gap_eV=gap,
                gamma_gap_difference_meV=(gap-old['gamma_gap_eV'])*1000,
                k4_energy_Ha=oldpoint['energy_Ha'],k6_energy_Ha=p['energy_Ha'],
                residual=p['residual'],orthogonality=p['orthogonality_max_error'],charge=p['integrated_valence_density'],
                gamma_eigen_residual=max(row['residuals_Ha']),
                scope='Gamma transition of independently stationary k6 starting density only; full-mesh extrema and endpoint comparison remain separate.')
    data['passed_source_gates']=True
    return data

def gradient_tex(g):
    maxres=max(x['residual'] for x in g['checks']);maxasr=max(x['ASR'] for x in g['checks']);maxeig=max(x['eigen_residual'] for x in g['checks'])
    return rf'''At the registered MgSiN$_2$ response cell and predeclared strain direction,
the XC quadrature was refined from $576^3$ to $720^3$ while retaining the
56,087-vector Miller basis, $144^3$ FFT mesh, 100-Ha cutoff, fixed ions and PBE.
The refined reference state and all four displaced states completed electronic
reoptimization and band-edge eigensolution. Their maximum stationary residual
was ${sci(maxres)}$, maximum raw acoustic-sum-rule residual was
${sci(maxasr)}$ Ha/Bohr, and maximum eigenpair residual was ${sci(maxeig)}$ Ha;
the integrated charge was 96 electrons to numerical precision.
The finite-difference steps were $h=10^{{-3}}$ and $h/2$ in the same strain coordinate.

\begin{{table}}[htbp]\centering
\caption{{Completed directional-response evidence under XC refinement.
The production adjoint uses $576^3$ quadrature; the reoptimized finite
differences use $720^3$. These completed entries do not constitute an
adjoint--FD comparison at the refined setting. Derivatives are in eV per unit strain.}}
\label{{tab:completed-xc-response}}
\begin{{tabular}}{{lrr}}\toprule
Estimator & XC grid & Derivative \\ \midrule
Production adjoint & $576^3$ & {g['production_directional_gradient_eV']:.9f} \\
Refined FD, $h$ & $720^3$ & {g['fine_fd_h_eV']:.9f} \\
Refined FD, $h/2$ & $720^3$ & {g['fine_fd_half_eV']:.9f} \\
Refined Richardson FD & $720^3$ & {g['fine_fd_richardson_eV']:.9f} \\
\bottomrule\end{{tabular}}\end{{table}}

The step-halving change was ${sci(abs(g['step_difference_eV']))}$ eV per unit strain,
or {g['step_relative_difference']*100:.6f}\% relative to the finer-step slope.
The production adjoint projection differed from the refined $h/2$ slope by
${sci(abs(g['production_vs_fine_fd_difference_eV']))}$ eV per unit strain.
The reference $\Gamma$ transition was {g['production_gamma_gap_eV']:.11f} eV
at $576^3$ and {g['fine_gamma_gap_eV']:.11f} eV at $720^3$.
These results support the stability of this directional finite difference
and its consistency with the production response under XC refinement.
\textbf{{Incomplete validation:}} the $720^3$ self-consistent adjoint has not
completed because its geometry-derivative evaluation exceeded the GPU memory
capacity. No refined-adjoint value or complete six-component refinement
certificate is asserted.
'''

def kmesh_tex(k):
    return rf'''At the starting geometry, the independently reoptimized $6^3$ state
passed stationary, orthogonality and charge gates, with residual
${sci(k['residual'])}$, overlap error ${sci(k['orthogonality'])}$ and 96 electrons.
The $4^3$ and $6^3$ states shared the same 55,913-vector Miller basis,
$144^3$ FFT mesh, $576^3$ XC quadrature and 100-Ha cutoff.
The completed $\Gamma$ eigensolution at the $6^3$ density had maximum
residual ${sci(k['gamma_eigen_residual'])}$ Ha.

\begin{{table}}[htbp]\centering
\caption{{Completed starting-state checks at fixed geometry and basis.
The transition is evaluated at $\Gamma$ using each setting's own stationary
density; it is not the minimum gap over the full $6^3$ mesh.}}
\label{{tab:completed-k6-start}}
\begin{{tabular}}{{lrr}}\toprule
Quantity & $4^3$ density & $6^3$ density \\ \midrule
Total energy (Ha) & {k['k4_energy_Ha']:.9f} & {k['k6_energy_Ha']:.9f} \\
$\Gamma$ transition (eV) & {k['k4_start_gamma_gap_eV']:.9f} & {k['k6_start_gamma_gap_eV']:.9f} \\
\bottomrule\end{{tabular}}\end{{table}}

The starting $\Gamma$ transition changed by {k['gamma_gap_difference_meV']:.8f} meV.
\textbf{{Incomplete validation:}} the complete $6^3$ sampled gaps and the
actual reverse-endpoint comparison are not yet available. This completed
starting-point result does not establish full-zone gap convergence, gap-gain
stability or target transfer at the endpoint.
'''

def partial_rows(data,gradient=True,kmesh=True):
    rows=[]
    def add(group,metric,a,b,unit):rows.append(dict(group=group,metric=metric,production=a,refined=b,difference=b-a,unit=unit))
    g=data['gradient'];k=data['kmesh']
    if gradient and g:
        add('completed_XC_FD','production_adjoint_vs_refined_half_FD',g['production_directional_gradient_eV'],g['fine_fd_half_eV'],'eV/strain')
        add('completed_XC_FD','refined_FD_step_halving',g['fine_fd_h_eV'],g['fine_fd_half_eV'],'eV/strain')
        add('completed_XC_FD','Gamma_gap_quadrature',g['production_gamma_gap_eV'],g['fine_gamma_gap_eV'],'eV')
    if kmesh and k:
        add('completed_k6_start','Gamma_transition',k['k4_start_gamma_gap_eV'],k['k6_start_gamma_gap_eV'],'eV')
        add('completed_k6_start','total_energy',k['k4_energy_Ha'],k['k6_energy_Ha'],'Ha')
    return rows

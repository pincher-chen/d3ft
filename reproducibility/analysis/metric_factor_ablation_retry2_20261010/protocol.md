# Controlled metric-factor ablation (2026-10-10)
18 developer runs: Al/SiC/GaAs, seeds201/202, three arms. Validation seeds excluded.
All arms use the identical frozen PH annealing prefix (1000 steps; original roundoff path), then the stable corrected stage. The only stage-dependent intervention is rotation_weight after the switch.
Common settings: final_method.json, orbital rate0.1, post-anneal occupation scale10, corrected Rayleigh estimator, delta0.001Ha, trial-rate recycling,20k budget, strict8+3 audits every4k, stop at first complete pass. Initialization is hash-paired across all three arms. No mechanism observer. No timing claims.
Arm1 W=max(PH*sqrt(dE^2+delta^2),1e-14), frozen implementation unchanged.
Arm2 W=max(PH,1e-14); same estimator remains available, gap factor removed.
Arm3 W=max(max(abs(fi-fj),1e-14)*sqrt(dE^2+delta^2),1e-14). The occupation-difference floor and final floor reuse the existing1e-14 numerical boundary floor; no tuned parameter.
ETDM source checked against Ivanov et al., arXiv2101.12597, Sec3.3 Eq31–33: diagonal curvature is -2(epsilon_i-epsilon_j)(f_i-f_j)+beta; Eq33 mixes curvature and a quasi-Newton beta using gamma. Arm3 isolates its occupation-difference factor only, retaining our positive gap regularization, stable algebra, FD coordinates and Armijo. It is NOT a full ETDM or Eq33 reproduction; factors2, mixing beta/gamma, L-BFGS and exponential representation are not introduced.
Source: https://arxiv.org/html/2101.12597 . Hardware: hn48 fourH10080GB under250212, one sequential lane per GPU. These are non-timing experiments. Drain all lanes before any future hn48 isolated timing. Preserve failures/censored without retuning.

Operational retry only: GPU0 acquired an external process after initial launch. All15 startup/guard failures retained in the original campaign. Same18 scientific cases remain, three continue there and15 retry here. Wait at each occupied-card handoff; never terminate another process. Pre-existing paired initial hashes retained. No timing claims.

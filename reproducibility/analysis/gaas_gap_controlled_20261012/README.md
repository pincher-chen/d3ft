# Controlled GaAs gap-factor extension (2026-10-12)

This campaign evaluates the fixed method on previous validation seeds301--305 and1001--1003, without parameter selection. It does not alter draft_v5 or draft_v6. Only allocation250285 (hn35, eight H10080GB cards) is allowed. hn48 is reserved for timing; no allocation is canceled and no old result or failed record is deleted.

## Controlled comparison and reuse

The approved `metric_factor_ablation_retry2_20261010` scientific executor, metric hook, endpoint wrapper, independent spectral audit and all acceptance scripts are copied byte-for-byte. Both arms have the same gradient, positive-occupation energy estimator, stable algebra, occupation scale10, recycling, anneal-then-switch schedule, source trees and physical input. The sole metric difference in the corrected phase is PH factor alone versus PH factor times the regularized gap. The1000-step PH annealing prefix is unchanged in both arms.

`diagnostics/reuse_diff.json` records full per-field comparisons of all eight previous PH×gap executions against the retry2 GaAs PH×gap specification, independently checked against the completed canonical GaAs PH×gap control. All match: full realized ConfigDicts except PRNGseed/path identifiers; actual phase arguments; solver/dependency/driver hashes; input and pseudopotential hashes; audit code/thresholds; and the same QE reference hashes. The PH×gap endpoint hook returns without modifying the frozen rotation-weight function and delegates to exactly the validation timing_driver.py. Production/gather candidates are not installed. Therefore the existing PH×gap executions are reused, each with its original task/certificate SHA256, and eight PH-only cold starts are queued. Any reused-arm initial hash also seeds the pairing guard for its new PH-only partner. This is reuse of fixed validation evidence, not eight additional independent PH×gap runs.

## Physical and algorithm settings

Two-atom zinc-blende GaAs, conventional lattice parameter5.749232952198Angstrom,27points on a3×3×3Gamma mesh,100Ha cutoff,90³FFT,SG15/PBE including Ga3d semicore,18electrons,17representedbands,25independent auditbands, noSOC. TargetT=.01Ha; geometrically anneal .1→.01Ha during the1000-step original PH prefix. At1000 switch to stable_exact001 with orbitalLR=.1 and occupationLR10 (prefix1), implicit FD coordinate temperature .1Ha,200bisections, corrected positive-occupation Rayleigh estimator, delta=.001Ha, metricfloor1e-14, Armijo/recycling and stage resets exactly as retry2. Full8+3 audits at4000,8000,12000,16000,20000; no retuning.

| Claim | Dataset | Task | Metric | Baseline | Ablation | Evidence | Risk |
|---|---|---|---|---|---|---|---|
| Gap scaling affects fixed-setting convergence | Eight previous GaAs validation seeds | Random cold free-energy minimization | First passing original11-test audit; budget censoring and physical residuals | Identical PH×gap | PH only | Same-seed controlled results through20k | Reused starts are not new PH×gap samples; finite audit resolution and fixed-system scope |

No timing/efficiency or AI novelty claim is made. Physics fidelity is assessed only by the original eight internal plus three matched-QE tests. PH-only passing as often/early as PH×gap would falsify a strong fixed-setting necessity claim; all outcomes are retained and reported.

## Outputs and terminal-point convention

- `processed/summary.csv`:16rows, seed,arm,first_accepted_audit (or`n.a.`),status,reuseflag,actualendpointstep,all11endpointresiduals,andall11`step20000_*`residuals.
- `processed/audits.csv`: all available audits, not just passing points.
- `processed/raw/`: endpoint/audit/configuration/provenance summaries; original state/audit paths retained in records.
- `processed/status.json`: counts, strict thresholds and status; execution/infrastructure failures remain distinct from scientific20k censoring.
- `processed/conclusion_3lines.txt`: generated only when every required task is terminal.

The identical retry2 executor stops at the first full passing audit. Thus reused PH×gap arms stopped at4000 and have no20000-step state. Their `step20000_*` entries are explicitly`n.a.`; actual4000residuals are in`endpoint_*`. Scientific censored arms must have a real20000audit, and their20000values are included. No4000state is relabeled as20000, and no unapproved continuation is added. All reported passing steps are audit-observed upperbounds.

## Scheduling and preservation

Idle-card admission is perGPU, with five-second clear sampling. Busy cards are skipped and released cards automatically claim the next task. Same-card external arrivals are recorded without interrupting non-timing calculations. Resource-only orchestration adds status-transition CSV summaries and an automatic final summary; it changes no scientific routine or stopping rule. Existing Slurm allocations, other sessions and all old artifacts remain untouched. The three-line conclusion is printed in the controller terminal log and saved when complete; no manuscript update is triggered.

The legacy task `role=metric_development_ablation` is retained solely for the unmodified executor's same-material initial-state pairing dispatch; `seed_role=previous_validation_fixed_method_no_tuning` states the actual study role. No validation seed is reclassified as tuning data.

## Disk-quota recovery1
Eight original PH-only executions ended when the allocated step lost writable quota at approximately14:41; none is a scientificcensored endpoint. Original metadata/scripts/loghashes are retained under /XYFS03/SSD_POOL/nscc-gz_pinchen/nscc-gz_pinchen_4/work/DFT/paper/jctc/gaas_gap_controlled_20261012/history/quota_failure_20261012_162232. All restart choices are audited in diagnostics/recovery_checkpoint_validation.json. Resume each verified4000 checkpoint at the original8000/12000/16000/20000block boundaries, with the same source, initial parameters, temperatures,LRs and resets. The scientificexecutor on disk is unchanged; a reversible resource-only runtime adapter restores previous audit records and skips already completed blocks. Status/CSV refresh every30seconds and detect exited workers rather than trusting stale runningflags.

Recovery submitted and confirmed on all eight hn35 GPUs at 2026-10-12 16:46:26. See metadata/recovery1_GPU_start_confirmed.json. Allocation retained; no old log or partial result removed.

## Automatically updated outcome

```text
PH×gap：8/8通过；配置逐项一致，复用原严格验收记录。
PH only：0/8通过，8/8在20000步内censored，基础设施/执行失败0项。
结果：/XYFS03/SSD_POOL/nscc-gz_pinchen/nscc-gz_pinchen_4/work/DFT/paper/jctc/gaas_gap_controlled_20261012/processed/summary.csv；仅支持此固定设置下的受控结论，未改论文、未占hn48、allocation保留。
```

# Figure 5 — GaAs: Rayleigh estimates, switch timing, independent seeds

Main text, Section 4.5. GaAs, 3³ mesh, 100 Ha, Ga 3d semicore included.

| Panel | Content |
|---|---|
| (a) | Largest error of the Rayleigh energies on development seeds 201/202: floored divisor versus exact positive occupation. |
| (b) | Step at which the energy factor is activated (open) and first accepted audit (filled), for four activation rules, seeds 201 (○) and 202 (□). |
| (c) | Eight independent seeds: first accepted audit of this work, and the 20000-step endpoint (not accepted) of the PH-only reference configuration. This is a configuration comparison, not the controlled ablation (Table 4). |

## data/
- `GaAs_estimator.csv` — panel (a).
- `GaAs_switch.csv` — panel (b): `rule`, `seed`, `switch_after_steps`, `first_qualified_audit_steps`.
- `GaAs_validation_final.csv` — panel (c), derived from `source_main_results.csv` (final run ledger): this-work runs `validation_GaAs_s30x` / `GaAs_ruleA_new_s100x`, reference runs `PH_control_GaAs_s30x` / `GaAs_PH_new_s100x`, budget 20000 steps.
- `source_main_results.csv` — copy of the complete run ledger used for panel (c).
- `SOURCES.json`.

Note: before 2026-10-12 panel (c) was drawn from values typed into the script; it now reads `GaAs_validation_final.csv`, with identical output.

## script/
`python script/plot_figure_4.py` → `Figure_5.pdf/png`.

# Optional implementation optimization

The exact validated production transform and helper code are retained here. They clone the corrected update's trial objective with zero-padded gather; gradients, direction, Armijo, recycling, callbacks and PH-prefix execution are unchanged. This code is separate from the main validation solver. See `../analysis/implementation_chain_completed_20261010/` for the three-layer equivalence evidence and candidate steady-state timing results.

Use `python scripts/runtime.py paper_frozen scripts/optimized_driver.py reproducibility/scripts/timing_driver.py [driver arguments]` for an opt-in corrected-phase experiment. The original PH prefix must still be run first using `ph_prefix`; the complete default case runner intentionally uses the frozen implementation. This wrapper does not add a new scientific algorithm or relax acceptance.

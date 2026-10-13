"""Optional same-basis continuation, guarded by a tighter stationarity check."""
import copy
import gc
import json
import numpy as np
import jax


class SameBasisContinuation:
    def __init__(self, short_epochs, residual_limit=1e-10):
        if short_epochs <= 0 or not 0 < residual_limit <= 1e-10:
            raise ValueError('Need positive continuation epochs and residual limit <= 1e-10')
        self.short_epochs = short_epochs
        self.residual_limit = residual_limit
        self.state = None
        self.mask = None
        self.label = None

    def calculate(self, config, mask, label, work, calculate, report):
        if self.short_epochs >= config.epoch:
            raise ValueError('Continuation schedule must be shorter than the full schedule')
        can_reuse = self.state is not None and mask is not None and np.array_equal(mask, self.mask)
        details = dict(method='cold', residual_limit=self.residual_limit)
        if can_reuse:
            short = copy.deepcopy(config)
            short.epoch = self.short_epochs
            out = calculate(short, freq_mask=mask, init_params=self.state)
            result = report(out)
            details.update(method='same_basis_continuation', from_label=self.label,
                           short_epochs=len(out.energy_history),
                           short_residual=float(result['variational_residual']))
            if not np.isfinite(details['short_residual']) or details['short_residual'] > self.residual_limit:
                # Preserve the failed short attempt, then reproduce the complete
                # cold schedule. Do not let a failed warm state seed the fallback.
                with (work/(label+'_continuation_attempt.json')).open('x') as f:
                    json.dump(details, f, indent=2)
                np.save(work/(label+'_continuation_attempt_energy.npy'),np.asarray(out.energy_history))
                details['method'] = 'cold_fallback_after_continuation'
                del out, result
                jax.clear_caches()
                gc.collect()
                out = calculate(config, freq_mask=mask)
                result = report(out)
        else:
            out = calculate(config, freq_mask=mask)
            result = report(out)
        residual = float(result['variational_residual'])
        details['accepted_schedule_epochs'] = len(out.energy_history)
        if np.isfinite(residual) and residual <= self.residual_limit:
            # Host storage releases the previous GPU arrays between points.
            self.state = jax.tree_util.tree_map(np.asarray,dict(pw=out.params_pw,occ=out.params_occ))
            self.mask = np.array(out.freq_mask,copy=True)
            self.label = label
        else:
            self.state = self.mask = self.label = None
        return out, result, details

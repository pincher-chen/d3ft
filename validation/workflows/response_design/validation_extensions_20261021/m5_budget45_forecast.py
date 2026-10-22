"""CPU-only accounting; no response/PCG operations or convergence decisions."""
def forecast(elapsed_seconds, iteration, steady_times, profile):
    if elapsed_seconds < 0 or iteration < 1:
        raise ValueError('Invalid accounting observation')
    per = max([float(profile['steady_iteration_seconds_for_forecast'])] +
              [float(t) for t in steady_times[-3:]])
    target = max(32, iteration + 2)  # Original planning rule, NOT PCG maxiter.
    remaining = (target - iteration) * per + 1440.
    startup = float(profile['startup_reconstruction_and_first_compilation_overhead_seconds'])
    modeled = startup + target * per + 1440.
    total_seconds = max(elapsed_seconds + remaining, modeled)
    return dict(predicted_new_GPUh=8 * total_seconds / 3600.,
                steady_iteration_seconds_used=per,
                modeled_total_iterations=target,
                historical_startup_compilation_seconds=startup,
                measured_paid_seconds=elapsed_seconds,
                pullback_reserve_seconds=1440.)

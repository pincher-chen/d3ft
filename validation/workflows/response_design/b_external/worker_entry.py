"""Opt-in execution wrapper; unmodified E7 a0 worker and physical gates."""
import json
import os
from pathlib import Path
import shutil
import sys
import time

from policy import read, dump, forecast, sha, pause_after_first_gaas_point


def main():
    root = Path(os.environ['D3FT_B_ROOT'])
    protocol = read(root / 'protocol.json')
    runtime = Path(protocol['runtime'])
    output = Path(os.environ['D3FT_B_OUTPUT'])
    tag = sys.argv[sys.argv.index('--tag') + 1]
    worker_destination = Path(os.environ['D3FT_PHASE3_RESULTS_ROOT']) / tag
    if worker_destination != output or output.exists():
        raise ValueError('Worker output must match an uncreated destination, separate from wrapper logs')
    sys.path.insert(0, str(runtime / 'validation/workflows/phase3'))
    sys.path.insert(0, str(runtime))
    import worker
    material = os.environ['D3FT_B_MATERIAL']
    output = Path(os.environ['D3FT_B_OUTPUT'])
    started = float(os.environ['D3FT_B_START_EPOCH'])
    source = os.environ.get('D3FT_RESUME_SOURCE')
    for path, expected in protocol['pseudopotentials'][material].items():
        if sha(path) != expected:
            raise ValueError('Frozen SG15 pseudopotential changed')
    if source:
        from scf_checkpoint_resume import install
        install(worker, source, os.environ['D3FT_RESUME_REPORT'])
    original_calc = worker.calc
    memory = None
    memory_passed = False
    if material == 'GaAs':
        # Instrument only the module's progress print. Its first update is
        # already synchronized by ground_state; no update/physics is changed.
        import builtins
        from device_memory_sampling import Sampler, check_peak
        indices = [int(x) for x in os.environ['D3FT_B_PHYSICAL_GPUS'].split(',')]
        if indices != [1,2,3,4,5,6] or os.environ['CUDA_VISIBLE_DEVICES'] != '1,2,3,4,5,6':
            raise ValueError('GaAs six-device execution layout mismatch')
        memory = Sampler(output.parent.parent, indices, interval=.1).start()
        import importlib
        ground = importlib.import_module('d3ft.calc.ground_state')
        def progress_print(*args, **kwargs):
            nonlocal memory_passed
            builtins.print(*args, **kwargs)
            if (not memory_passed and args and str(args[0]).startswith('D3FT_STAGE update_complete 1/')):
                memory.set_stage('first_update_memory_gate')
                report = memory.report()
                import jax
                stats = [dict(logical_device=i, raw_statistics=d.memory_stats())
                         for i,d in enumerate(jax.local_devices())]
                passed = check_peak(report, indices)
                result = dict(time=__import__('datetime').datetime.now().astimezone().isoformat(),
                    passed=passed, physical_gpus=indices, limit_GiB=78,
                    measurement=report, allocator_statistics=stats,
                    same_process_and_optimizer_trajectory=True,
                    steps_or_schedule_changed=False,
                    scope='Initialization + compile + one synchronized native update; not a final peak guarantee')
                dump(root/'GaAs_six_card_memory_acceptance.json',result)
                dump(output.parent.parent/'smoke_memory_acceptance.json',result)
                if not passed:
                    builtins.print('GaAs six-card memory test did not pass; no more updates authorized',flush=True)
                    raise SystemExit(78)
                memory_passed = True
                memory.set_stage('authorized_first_pressure_point')
                builtins.print('GAAS_SIX_CARD_MEMORY_PASS: continue same first pressure point',flush=True)
        ground.print = progress_print

    def budget_check(label=None, checkpoint=False):
        budget = read(root / 'budget.json')
        live = max(0., time.time() - started) * int(os.environ.get('D3FT_B_GPU_COUNT','8')) / 3600
        result = forecast(protocol, budget, material, live, label, output)
        if material == 'GaAs' and not read(root/'GaAs_first_point_policy.json').get('automatic_continue_within_cap'):
            # The only user authorization is to finish eq_0, including any
            # genuine exit124 recovery; 280 is not a continuation permission.
            if not (root / 'GaAs_first_point_policy.json').exists():
                raise ValueError('Missing first-point-only authorization')
            result.update(allowed=label in (None, 'eq_0'),
                          original_280_cap_not_used_for_release=True)
            dump(root / 'live_budget_forecast.json', result)
            if not result['allowed']:
                raise SystemExit(76)
            return
        # At checkpoint, only enforce remaining wall allowance, while future
        # POINT forecasts are tested before calc. Original checkpoint is saved
        # before any stop, and a budget pause never autoruns as exit124.
        if checkpoint:
            allowance = protocol['settings'][material]['upper_GPU_hours'] - result['used_GPU_hours']
            next_chunk = result['rate_GPU_hours_per_point'] * 500 / protocol['settings'][material]['epochs']
            result['allowed'] = allowance > max(0.25, 1.5 * next_chunk)
            result['next_500_step_reserve_GPU_hours'] = max(0.25, 1.5 * next_chunk)
        dump(root / 'live_budget_forecast.json', result)
        if not result['allowed']:
            result['message'] = '需用户批准：B-' + material + '机时预测/剩余额度超过草案上限；保持现场，不启动后续点'
            dump(root / 'NEEDS_USER_APPROVAL.json', result)
            print(result['message'], flush=True)
            raise SystemExit(75)

    def guarded_calc(config, init_params=None, freq_mask=None):
        xyz = Path(config.crystal_file_path_path)
        if source and (Path(source) / 'mask.npy').exists() and not (xyz.parent / 'mask.npy').exists():
            shutil.copyfile(Path(source) / 'mask.npy', xyz.parent / 'mask.npy')
        budget_check(xyz.stem)
        out = original_calc(config, init_params=init_params, freq_mask=freq_mask)
        import numpy as np
        np.save(xyz.parent / 'mask.npy', np.asarray(out.freq_mask))
        return out

    worker.calc = guarded_calc
    from d3ft.calc import checkpoint
    original_write = checkpoint.SCFCheckpoint.write

    def guarded_write(self, *args, **kwargs):
        original_write(self, *args, **kwargs)
        budget_check(checkpoint=True)

    checkpoint.SCFCheckpoint.write = guarded_write
    if material == 'GaAs':
        import point_yield
        point_yield.yield_after_point = pause_after_first_gaas_point
    # Validate the immutable full/PBE vector before any calculation. The E7
    # worker additionally enforces its own certificate/environment contract.
    vector = json.loads(sys.argv[sys.argv.index('--vector') + 1])
    expected = protocol['fitted_vector'] if os.environ['D3FT_B_MODEL'] == 'fitted' else [0.] * 7
    if vector != expected:
        raise ValueError('Evaluation vector differs from frozen model')
    try:
        worker.main()
    finally:
        if memory is not None:
            report = memory.finish()
            dump(root/'GaAs_memory_final_attempt.json', report)


if __name__ == '__main__':
    main()

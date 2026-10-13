"""Compilation/parity diagnostic; never produces a scientific summary."""
import argparse
import json
from pathlib import Path
import jax
from ml_collections import ConfigDict
from d3ft.calc.ground_state import calc
from d3ft.parallel import plain_fft_patch

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('source',type=Path)
ap.add_argument('--epochs',type=int,default=1)
ap.add_argument('--observables',action='store_true')
ap.add_argument('--out',type=Path)
ap.add_argument('--compile-only',action='store_true')
ap.add_argument('--fused-report',action='store_true')
ap.add_argument('--compare-report',action='store_true',
                help='Also run the eager report; unsuitable for full-size memory probes')
args=ap.parse_args()
jax.config.update('jax_enable_x64',True)
meta=json.loads((args.source/'metadata.json').read_text())
cfg=ConfigDict(meta['config'])
cfg.epoch=args.epochs
cfg.crystal_file_path_path=str((args.source/'sample.xyz').resolve())
cfg.d3ft.parallel.k_pool=True
class CompilationFinished(Exception):
    pass

if args.compile_only:
    original_jit=jax.jit

    def diagnostic_jit(fun,*pos,**kwargs):
        compiled_fun=original_jit(fun,*pos,**kwargs)
        if fun.__name__!='update': return compiled_fun

        def capture(*inputs):
            executable=compiled_fun.lower(*inputs).compile()
            stats=executable.memory_analysis()
            record={key:getattr(stats,key) for key in [
                'argument_size_in_bytes','output_size_in_bytes','alias_size_in_bytes',
                'temp_size_in_bytes','generated_code_size_in_bytes']}
            record['scope']='Compiled per-device estimate; update was not executed'
            record['devices']=[str(d) for d in jax.devices()]
            if args.out:
                with args.out.open('x') as f: json.dump(record,f,indent=2)
            print(json.dumps(record),flush=True)
            raise CompilationFinished
        return capture

    jax.jit=diagnostic_jit
    try:
        with plain_fft_patch(k_pool=True): calc(cfg)
    except CompilationFinished:
        raise SystemExit(0)
    finally:
        jax.jit=original_jit
    raise RuntimeError('No optimizer update was intercepted')

with plain_fft_patch(k_pool=True):
    result=calc(cfg)
    record=dict(energy_Ha=float(result.total_energy),sharding=str(result.params_pw.sharding),
                local_shapes=[list(s.data.shape) for s in result.params_pw.addressable_shards],
                epochs=args.epochs,scope='Compilation/parity diagnostic only')
    if args.observables:
        from residual_report import report,fused_report
        from d3ft.calc.forces import force_report
        from d3ft.calc.screening import sensitivity
        import numpy as np
        r=fused_report(result,compare=args.compare_report) if args.fused_report else report(result)
        if 'comparison' in r: record['report_comparison']=r['comparison']
        record.update(pressure_GPa=float(r['pressure_GPa']),residual=float(r['variational_residual']))
        record['forces']=np.asarray(force_report(result)['forces_Ha_Bohr']).tolist()
        record['dE_dp']=np.asarray(sensitivity(result,cfg.d3ft.xc_params.to_dict(),3)).tolist()
if args.out:
    with args.out.open('x') as f: json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)

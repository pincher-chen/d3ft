#!/usr/bin/env python3
"""Selected CPU checks only; never discover/execute campaign controllers."""
import argparse,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'validation/workflows/response_design'
p=argparse.ArgumentParser();p.add_argument('--report',type=Path);args=p.parse_args()
env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT,ROOT/'third_party/jrystal_dependency',W,W/'reverse_response',W/'e9_e10']))
checks=[]
with tempfile.TemporaryDirectory(prefix='response_cpu_') as d:
    cases=[('real_linear_adjoint','reverse_response/check_response.py',[]),
           ('geometry_units_phases','reverse_response/check_geometry_layout.py',['--output',str(Path(d)/'geometry.json')]),
           ('sampled_band_edges','reverse_response/check_mesh_target.py',['--output',str(Path(d)/'mesh.json')]),
           ('compact_k_blocks','reverse_response/test_compact_k_stream.py',[]),
           ('relaxed_design_controller','e9_e10/test_relaxed_cpu.py',[])]
    for name,file,extra in cases:
        r=subprocess.run([sys.executable,str(W/file),*extra],env=env,capture_output=True,text=True,timeout=180)
        checks.append({'name':name,'passed':r.returncode==0,'stdout':r.stdout[-4000:],'stderr':r.stderr[-3000:]})
        print(name,'OK' if r.returncode==0 else 'FAILED',flush=True)
report={'passed':all(c['passed'] for c in checks),'scope':'CPU mathematics/interfaces only, not fresh production DFT validation','checks':checks}
if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
if not report['passed']:raise SystemExit(1)

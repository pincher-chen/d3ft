"""Execute the real worker main on CPU up to a mocked SCF entry, never JAX/GPU.

Only hardware enumeration, config loading and calc are stubs. Argument parsing,
parameter certificate, immutable mkdir, metadata, snapshots and geometry setup
are the actual unmodified worker function extracted with Python AST.
"""
import argparse,ast,atexit,copy,hashlib,json,os,shutil,sys,tempfile,time,types
from contextlib import ExitStack,nullcontext
from pathlib import Path
from unittest.mock import patch
import numpy as np

class EnteredSCF(Exception):pass
class C(dict):
    def __getattr__(self,k):
        try:return self[k]
        except KeyError:raise AttributeError(k) from None
    def __setattr__(self,k,v):self[k]=v
    def to_dict(self):
        def plain(x):return {k:plain(v) for k,v in x.items()} if isinstance(x,dict) else x
        return plain(self)

def check(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());runtime=Path(p['runtime'])
    worker_file=runtime/'validation/workflows/phase3/worker.py'
    tree=ast.parse(worker_file.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    code=compile(ast.Module(body=[main],type_ignores=[]),str(worker_file),'exec')
    materials=json.loads((worker_file.parent/'materials.json').read_text())
    def config(_):return C(optimizer_args=C(),d3ft=C(optim=C(),occupation=C(),kpoints=C(),parallel=C()))
    def dump(path,value):Path(path).write_text(json.dumps(value,default=str))
    results=[]
    with tempfile.TemporaryDirectory(prefix='B_worker_CPU_directory_') as td:
        td=Path(td)
        for material in ['AlP','GaAs']:
            for model in ['fitted','PBE']:
                tag=material+'_'+model+'_CPU';run=td/tag;run.mkdir()
                (run/'launch_intent.json').write_text('{}');(run/'stdout.log').touch()
                output=run/'worker_results'/tag
                vector=p['fitted_vector'] if model=='fitted' else [0.]*7
                cert=root/('full_parameter_freeze.json' if model=='fitted' else 'PBE_parameter_freeze.json')
                g=dict(argparse=argparse,atexit=atexit,ExitStack=ExitStack,copy=copy,gc=__import__('gc'),hashlib=hashlib,json=json,os=os,Path=Path,shutil=shutil,time=time,np=np,
                       jax=types.SimpleNamespace(devices=lambda:[types.SimpleNamespace(platform='gpu')]*8),
                       MATERIALS=materials,PARAMS={},ROOT=runtime,HERE=worker_file.parent,BOHR=.529177210903,
                       __file__=str(worker_file),load_config=config,dump=dump,
                       default_xc_params=lambda _: {'gga_c_pbe':{'beta':.0667,'gamma':.031},'gga_x_pbe':{'kappa':.804,'mu':.2195}},
                       _write_xyz=lambda f,*_:Path(f).write_text('CPU geometry fixture\n'))
                def calc(cfg,**kw):
                    assert output.is_dir() and Path(cfg.crystal_file_path_path).is_file()
                    assert (output/'metadata.json').exists()
                    assert cfg.cutoff_energy==p['settings'][material]['cutoff_Ha']
                    raise EnteredSCF()
                g['calc']=calc
                s=p['settings'][material]
                argv=['worker.py','--material',material,'--vector',json.dumps(vector),'--tag',tag,
                      '--mode','evaluate','--bulk-only','--equilibrium-only','--grid',str(s['grid']),
                      '--cutoff',str(s['cutoff_Ha']),'--k',str(s['k']),'--epochs',str(s['epochs']),
                      '--empty','0','--xc-grid-factor',str(s['xc_grid_factor']),'--a0',str(s['a0_A']),
                      '--fused-report','--kpool']
                parallel=types.ModuleType('d3ft.parallel');parallel.plain_fft_patch=lambda **_:nullcontext()
                frozen=types.ModuleType('frozen_parameters');frozen.canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
                env={'D3FT_PHASE3_RESULTS_ROOT':str(output.parent),'D3FT_REFERENCE_ROOT':p['environment']['D3FT_REFERENCE_ROOT'],
                     'D3FT_EVALUATION_FREEZE_CERT':str(cert),'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'}
                with patch.dict(sys.modules,{'d3ft.parallel':parallel,'frozen_parameters':frozen}),patch.dict(os.environ,env),patch.object(sys,'argv',argv):
                    exec(code,g)
                    try:g['main']()
                    except EnteredSCF:pass
                    else:raise AssertionError('Did not reach SCF')
                    try:g['main']()
                    except FileExistsError:pass
                    else:raise AssertionError('Immutable worker mkdir protection was weakened')
                results.append(dict(material=material,model=model,entered_SCF=True,metadata_written=True,
                                    duplicate_output_rejected=True,actual_GPU_calculation=False))
    return dict(passed=True,cases=results,scope='Real worker main with mocked calc/hardware/config; filesystem and argument setup only, not physical SCF validation')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(check(args.root),indent=2))

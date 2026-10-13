"""A changed numerical model must not count as a successful parameter trial."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'validation/workflows/phase3'
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('phase3_fit_step',HERE/'fit_step.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)


def test_trial_model_identity():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); manifests=[]
        for name in ['baseline','candidate']:
            folder=root/name; folder.mkdir()
            meta=dict(pseudopotentials={'C.upf':'same-bytes'},arguments={'strain_step':.01},
                      config=dict(grid_sizes=2,cutoff_energy=50,k_grid_sizes=6,empty_bands=0,xc='PBE',spin=0))
            (folder/'metadata.json').write_text(json.dumps(meta))
            np.save(folder/'mask.npy',np.ones((2,2,2),dtype=bool))
            manifest=root/f'{name}.json';manifest.write_text(json.dumps({'diamond':str(folder)}))
            manifests.append(manifest)
        excluded=[m for m in fit.MATERIALS if m!='diamond']
        fit.validate_trial(*manifests,excluded)
        mask=np.ones((2,2,2),dtype=bool);mask[0,0,0]=False
        np.save(root/'candidate/mask.npy',mask)
        try:
            fit.validate_trial(*manifests,excluded)
        except ValueError as error:
            assert 'Miller basis' in str(error)
        else:
            raise AssertionError('A changed basis was accepted as a parameter-only trial')


if __name__=='__main__':
    test_trial_model_identity()
    print('trial basis guard passed')

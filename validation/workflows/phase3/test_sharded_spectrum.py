import json
from pathlib import Path
import tempfile
import unittest
from sharded_spectrum import merge, sha


class MergeTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source';source.mkdir();out=root/'out';out.mkdir()
        def put(p,d):p.write_text(json.dumps(d))
        put(source/'metadata.json',{'config':{'k_grid_sizes':[2,1,1]}})
        (source/'density.npy').write_bytes(b'density')
        put(source/'points.json',[dict(label='eq_1',density_file='density.npy',density_sha256=sha(source/'density.npy'))])
        shards=[];mesh=[[0.,0.,0.],[.5,0.,0.]]
        for i in range(2):
            p=root/f'shard{i}';p.mkdir();shards.append(p)
            put(p/'metadata.json',dict(source=str(source),source_metadata_sha256=sha(source/'metadata.json'),
                density_sha256=sha(source/'density.npy'),solver='block',density_electrons=2.,electrons=2.,
                charge_budget=2e-5,mesh=mesh,indices=[i],occupied_bands=1,workflow_sha256={},lightweight_reconstruction=True))
            vals=[0.,2.,3.] if i==0 else [3.,4.,5.]
            put(p/f'k_{i:04d}.json',dict(index=i,k_frac=mesh[i],eigenvalues_Ha=vals,
                residuals_Ha=[1e-10]*3,vbm_Ha=vals[0],cbm_Ha=vals[1],seconds=1.))
        return source,out,shards

    def test_indirect_gap_uses_all_shards(self):
        with tempfile.TemporaryDirectory() as t:
            s,o,p=self.fixture(Path(t));r=merge(s,o,p,[0,1])
            self.assertTrue(r['full_mesh'])
            self.assertFalse(r['mesh_fixed_occupations_passed'])
            self.assertLess(r['gap_eV'],0)
            self.assertTrue((o/'k_0001.json').exists())

    def test_missing_duplicate_and_wrong_density_rejected(self):
        for failure in ('missing','duplicate','density'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as t:
                s,o,p=self.fixture(Path(t))
                if failure=='missing':p=p[:1]
                if failure=='duplicate':p=[p[0],p[0]]
                if failure=='density':(s/'density.npy').write_bytes(b'changed')
                with self.assertRaises(ValueError):merge(s,o,p,[0,1])
                self.assertFalse((o/'summary.json').exists())

    def test_complete_positive_mesh_and_bad_residual(self):
        for invalid in (False,True):
            with self.subTest(invalid=invalid),tempfile.TemporaryDirectory() as t:
                s,o,p=self.fixture(Path(t));path=p[1]/'k_0001.json'
                row=json.loads(path.read_text());row.update(eigenvalues_Ha=[.5,2.,3.],vbm_Ha=.5,cbm_Ha=2.)
                if invalid:row['residuals_Ha'][0]=1e-4
                path.write_text(json.dumps(row))
                if invalid:
                    with self.assertRaises(ValueError):merge(s,o,p,[0,1])
                else:
                    result=merge(s,o,p,[0,1])
                    self.assertTrue(result['mesh_fixed_occupations_passed'])
                    self.assertAlmostEqual(result['gap_eV'],1.5*27.211386245988)


if __name__=='__main__':unittest.main()

"""CPU-verifiable adoption of one saved failed FD point, without source mutation."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def load_saved_point(root,direction,h,sign,delta,mask,wanted_cell,wanted_pos):
 root=Path(root);m=json.loads((root/'metadata.json').read_text());point=json.loads((root/'point.json').read_text())
 if (m['direction'],m['h'],m['sign'])!=(direction,h,sign):raise ValueError('Different registered FD point')
 if not np.array_equal(np.array(m['delta']),np.array(delta)):raise ValueError('FD direction/delta changed')
 for key in ['fixed_parent_miller_basis','fixed_parent_beta_tables','fixed_parent_local_radial_nodes']:
  if not m[key]:raise ValueError('Not original frozen discrete program')
 if not np.array_equal(np.load(root/'mask.npy'),np.asarray(mask)):raise ValueError('Miller mask changed')
 if np.max(abs(np.array(point['cell_bohr'])-wanted_cell))>2e-11 or np.max(abs(np.array(point['positions_bohr'])-wanted_pos))>2e-11:raise ValueError('Saved point geometry mismatch')
 if sha(root/'state.pkl')!=m['state_sha256']:raise ValueError('Saved checkpoint hash changed')
 with (root/'state.pkl').open('rb') as f:state=pickle.load(f)
 return state,{'resume_from':str(root),'resume_state_SHA256':m['state_sha256'],'previous_epochs':point['epochs'],'previous_residual_Ha':point['residual'],'geometry_and_mask_unchanged':True,'optimizer_moments_saved':False,'optimizer_reinitialized':True}

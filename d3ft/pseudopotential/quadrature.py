"""Independent NumPy quadrature reference, including QE's even-size endpoint rule."""
import numpy as np
from scipy.integrate import simpson

def qe_simpson_weights(rab):
 rab=np.asarray(rab,dtype=float);n=len(rab)
 if n<3:raise ValueError('At least three radial points required')
 c=np.zeros(n);c[1:-1]=np.where(np.arange(2,n)%2==0,4.,2.);c[0]=1.
 if n%2:c[-1]=1.
 else:c[-3]-=.25;c[-2]+=1.;c[-1]=1.25
 return c*rab/3

def restored_mesh(r,b,rab):
 r=np.asarray(r);b=np.asarray(b);rab=np.asarray(rab)
 # The frozen UPF loader masks r=0. Restore its zero integrand on uniform meshes.
 if r[0]>0 and np.allclose(np.diff(r),rab[0],rtol=1e-9,atol=1e-12) and abs(r[0]-rab[0])<1e-12:
  return np.r_[0.,r],np.pad(b,((0,0),(1,0))),np.r_[rab[0],rab]
 return r,b,rab

def attach_upf_metadata(pseudopot, directory):
 """Retain QE's maximum projector cutoff index without changing the dependency."""
 import xml.etree.ElementTree as ET
 from jrystal.pseudopotential.load import find_upf
 sizes=[]
 for symbol in pseudopot.atomic_symbols:
  tree=ET.parse(find_upf(directory,symbol)).getroot()
  entries=[b for b in tree.find('PP_NONLOCAL') if b.tag.startswith('PP_BETA')]
  sizes.append(max(int(b.attrib['cutoff_radius_index']) for b in entries) if entries else None)
 pseudopot.beta_cutoff_sizes=sizes
 return pseudopot


def qe_radial_size(r, cutoff=10.0):
 """QE read_pseudo auxiliary msh: first r>10, rounded down to odd length."""
 r=np.asarray(r);i=int(np.searchsorted(r,cutoff,side='right'))
 n=i+1 if i<len(r) else len(r)
 return 2*((n+1)//2)-1

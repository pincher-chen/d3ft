"""Compare effective radial projector kernels and local potentials to DFTK.

D3FT beta and DFTK beta have different unit scalings: compare beta D beta,
not raw beta. Uses the production radial transforms, with no physics mutation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import CubicSpline
from d3ft.pseudopotential.beta import beta_sbt_table
from jrystal.sbt import sbt_numerical
from d3ft.pseudopotential.local import _trapezoid_weights


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference',type=Path);ap.add_argument('pp_dir',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--local-quadrature',choices=['trapezoid','simpson'],default='trapezoid')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.reference.read_text());q=np.asarray(d['q_Bohr_inverse'])
    rows=[];details={};hashes={}
    for symbol,ref in d['elements'].items():
        path=a.pp_dir/(symbol+'.upf');tree=ET.parse(path).getroot()
        def array(tag):return np.fromstring(tree.find(tag).text,sep=' ')
        r=array('PP_MESH/PP_R');v=array('PP_LOCAL')/2
        z=float(tree.find('PP_HEADER').attrib['z_valence']);sel=r>0
        elements=[el for el in tree.find('PP_NONLOCAL') if el.tag.startswith('PP_BETA.')]
        ls=np.array([int(el.attrib['angular_momentum']) for el in elements])
        beta=np.array([np.fromstring(el.text,sep=' ') for el in elements])[:,sel]/r[sel]
        dij=array('PP_NONLOCAL/PP_DIJ').reshape(len(elements),len(elements))/2
        k,b=beta_sbt_table(r[sel],beta,ls,14.)
        b=4*np.pi*CubicSpline(k,b,axis=1)(q)
        local_r=np.concatenate(([0.],r[sel]));vp=np.concatenate(([0.],v[sel]+z/r[sel]))
        if a.local_quadrature=='simpson':
            from radial_quadrature import simpson_weights
            weights=simpson_weights(local_r)
        else:weights=_trapezoid_weights(local_r)
        kl,vl=sbt_numerical(local_r,vp[None,:],l=0,kmax=14.,delta_r=weights)
        local=4*np.pi*CubicSpline(kl,vl[0])(q)-4*np.pi*z/q**2
        le=float(np.abs(local-ref['local_fourier']).max())
        alpha=4*np.pi*float(CubicSpline(kl,vl[0])(0.))
        de=[]
        for channel in ref['channels']:
            l=channel['l'];ix=np.where(ls==l)[0]
            ours=b[ix].T@dij[np.ix_(ix,ix)]@b[ix]
            theirs=np.asarray(channel['beta_fourier']).T@np.asarray(channel['h'])@np.asarray(channel['beta_fourier'])
            absolute=float(np.abs(ours-theirs).max());scale=float(np.abs(theirs).max())
            de.append(dict(l=l,kernel_absolute_error=absolute,kernel_relative_error=absolute/max(scale,1e-30)))
        details[symbol]=dict(local_max_absolute_error=le,alpha_difference=alpha-ref['alpha'],projector_channels=de)
        for i,x in enumerate(q):rows.append(dict(element=symbol,q=float(x),local_d3ft=float(local[i]),local_dftk=ref['local_fourier'][i],difference=float(local[i]-ref['local_fourier'][i])))
        hashes[symbol]=hashlib.sha256(path.read_bytes()).hexdigest()
    with (a.output/'local_form_factors.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (a.output/'diagnostic.json').write_text(json.dumps(dict(elements=details,pp_hashes=hashes,
        local_quadrature=a.local_quadrature,
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        scope='Radial discretization diagnostic using tabulation kmax14 and SciPy cubic interpolation; not a full SCF acceptance or a differentiability test.'),indent=2)+'\n')
    print(json.dumps(details,indent=2))


if __name__=='__main__':main()

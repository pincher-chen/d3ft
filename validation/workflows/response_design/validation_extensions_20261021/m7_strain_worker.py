"""M7: exact E4 design controller/backend, six active strains, fixed coordinates."""
import time
START=time.monotonic()
import argparse,csv,datetime,json,sys
from pathlib import Path
import numpy as np


def backend_class(GammaDFTBackend):
    class StrainOnlyBackend(GammaDFTBackend):
        def __init__(self,protocol):
            self.events=[]
            super().__init__(protocol)
            self.fractional_reference=self.basepos@np.linalg.inv(self.basecell)

        def _event(self,kind,folder,t0,**kw):
            row=dict(kind=kind,folder=str(folder),start_seconds=t0-START,
                     end_seconds=time.monotonic()-START,seconds=time.monotonic()-t0,**kw)
            self.events.append(row)
            event_file=self.p.get('event_file',str(Path(self.p['backend_status_file']).with_name('backend_events.jsonl')))
            with Path(event_file).open('a') as f:f.write(json.dumps(row)+'\n')

        def geometry(self,theta):
            if not np.array_equal(np.asarray(theta)[6:],self.theta0[6:]):
                raise ValueError('M7 requires all48 nonaffine coordinate inputs frozen')
            cell,pos=super().geometry(theta)
            error=float(np.max(abs(pos@np.linalg.inv(cell)-self.fractional_reference)))
            if error>2e-12:raise ValueError('Affine fractional coordinates changed')
            return cell,pos

        def evaluate(self,theta,folder,parent):
            begin=time.monotonic();ok=False
            try:
                result=super().evaluate(theta,folder,parent);ok=True;return result
            finally:self._event('evaluate',folder,begin,completed=ok)

        def gradient(self,theta,current,folder):
            begin=time.monotonic();ok=False
            try:
                full=np.asarray(super().gradient(theta,current,folder))
                if full.shape!=(54,) or not np.isfinite(full).all():raise ValueError('Invalid original54 gradient')
                zero_mode=float(np.max(abs(full[6:].reshape(16,3).sum(axis=0))))
                if zero_mode>1e-6:raise ValueError('Original full-gradient translation check failed')
                active=full.copy();active[6:]=0.
                dump(Path(folder)/'active6_gradient.json',dict(full54_gradient_eV=full.tolist(),
                     active6_masked_gradient_eV=active.tolist(),coordinate_frozen=True,
                     original_full_translation_max=zero_mode))
                ok=True;return active
            finally:self._event('gradient',folder,begin,completed=ok)
    return StrainOnlyBackend


def report_trajectory(trajectory,backend,p):
    rows=[];previous=np.asarray(p['theta0']);previous_end=0.
    for row in trajectory:
        theta=np.asarray(row['theta']);r=row['readout'];force=np.asarray(r['forces_Ha_Bohr'])
        assert np.array_equal(theta[6:],np.asarray(p['theta0'])[6:])
        step_norm=float(np.linalg.norm(theta[:6]-previous[:6]))
        total_norm=float(np.linalg.norm(theta[:6]-np.asarray(p['theta0'])[:6]))
        assert step_norm<=p['limits']['strain_step']+2e-14
        assert total_norm<=p['limits']['strain_total']+2e-14
        cell,pos=backend.geometry(theta)
        event=next(e for e in backend.events if e['kind']=='evaluate' and
                   Path(e['folder'])==Path(r['state_directory']) and e['completed'])
        end=event['end_seconds']
        rows.append(dict(step=row['step'],gap_eV=r['gap_eV'],target_error_meV=1000*(r['gap_eV']-p['target_eV']),
             strain6=theta[:6].tolist(),strain_delta6=(theta[:6]-np.asarray(p['theta0'])[:6]).tolist(),
             strain_step_L2=step_norm,strain_total_L2=total_norm,
             energy_change_meV_atom=(r['energy_Ha']-trajectory[0]['readout']['energy_Ha'])*27.211386245988*1000/16,
             max_force_norm_Ha_Bohr=float(np.linalg.norm(force,axis=1).max()),
             max_force_norm_eV_A=float(np.linalg.norm(force,axis=1).max())*27.211386245988/.529177210903,
             max_force_component_Ha_Bohr=float(abs(force).max()),
             fractional_max_error=float(np.max(abs(pos@np.linalg.inv(cell)-backend.fractional_reference))),
             step_wall_seconds=end-previous_end,readout_wall_seconds=event['seconds'],
             state_directory=r['state_directory']))
        previous=theta;previous_end=end
    return rows


def main():
    global dump
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();p=json.loads(a.protocol.read_text());rt=Path(p['runtime']['gamma'])
    sys.path[:0]=list(map(str,[rt,rt/'validation/workflows/response_design',rt/'validation/workflows/response_design/reverse_response',rt/'validation/workflows/phase3']))
    from joint_design_dft import GammaDFTBackend
    from joint_design import run_design,dump as frozen_dump
    dump=frozen_dump
    backend=backend_class(GammaDFTBackend)(p['backend_protocol'])
    try:
        result=run_design(a.output,p['theta0'],p['target_eV'],p['limits'],backend,
                          max_updates=p['max_updates'],max_backtracks=p['max_backtracks'])
        path=a.output/'trajectory.json'
        trajectory=json.loads(path.read_text()) if path.exists() else [dict(step=0,theta=p['theta0'],readout=json.loads((a.output/'initial/readout.json').read_text()))]
        rows=report_trajectory(trajectory,backend,p)
        result.update(passed=bool(result['target_reached']),completed=True,active_components=list(range(6)),
             coordinates_frozen=True,trajectory_summary=rows,per_step_timing_scope='Initial row includes worker initialization; later rows include adjoint and all attempted trials since previous accepted readout; total assigned GPUh independently in controller ledger',
             worker_wall_seconds=time.monotonic()-START)
        dump(a.output/'acceptance.json',result)
        with (a.output/'trajectory_summary.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        dump(Path(p['root'])/'worker_status.json',dict(stage='completed',target_reached=result['target_reached'],completed_updates=result['completed_updates']))
        print('M7_COMPLETE',json.dumps(result),flush=True)
    except Exception as exc:
        dump(Path(p['root'])/'worker_status.json',dict(stage='failed',error=repr(exc)))
        raise


if __name__=='__main__':main()

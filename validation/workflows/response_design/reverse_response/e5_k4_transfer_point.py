"""Two-state E5 transfer: ordinary four/eight-device k-pool SCF, never an adjoint.

The reference supplies the actual geometry and frozen numerical model. The
64-k parent supplies orbitals only; its density/occupation are never reused.
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
from ml_collections import ConfigDict
from gamma_response import dump, sha, validated_radial_splines
from fixed_radial_program import source_radial_program
from bounded_xc import memory_bounded_xc
from memory_probe import MemoryProbe
from mesh_basis_restart import project_miller_coefficients

P = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(P/'validation/workflows/response_design'))
sys.path.insert(0, str(P/'validation/workflows/phase3'))
from radial_quadrature import local_simpson_patch
from residual_report import fused_report
from d3ft.calc.ground_state import calc
from d3ft.calc.forces import force_report
from d3ft.parallel import plain_fft_patch
from jrystal._src.const import ANGSTROM2BOHR


def main(a):
    resume_latest=json.loads((a.resume_checkpoint/'latest.json').read_text()) if a.resume_checkpoint else None
    expected_epochs=3000-int(resume_latest['iteration']) if resume_latest else (2 if a.smoke else 1 if a.read_only else 3000)
    if len(jax.devices()) != a.devices or a.epochs != expected_epochs or expected_epochs<1:
        raise ValueError('Registered four/eight-device k4 protocol only')
    a.output.mkdir(exist_ok=False)
    probe = MemoryProbe(a.output).start()

    def stage(s, **kw):
        row = dict(stage=s, jobid=os.environ.get('SLURM_JOB_ID'),
                   step=os.environ.get('SLURM_STEP_ID'), time=time.time(), **kw)
        dump(a.output/'status.json', row)
        print('E5_K4_TRANSFER', json.dumps(row), flush=True)

    try:
        m = json.loads((a.reference/'metadata.json').read_text())
        rp = json.loads((a.reference/'point.json').read_text())
        if not rp.get('stationarity_pass') or sha(a.reference/'state.pkl') != m['state_sha256']:
            raise ValueError('Unaccepted or changed geometry source')
        for symbol, pp in m['physical']['pp'].items():
            if sha(pp['path']) != pp['sha256']:
                raise ValueError('Changed UPF '+symbol)
        if m['physical']['xc_grid_factor'] != 4 or m['physical']['cutoff'] != 100:
            raise ValueError('Unexpected frozen physical settings')
        mask = np.load(a.reference/'mask.npy')
        sm = json.loads((a.parent/'metadata.json').read_text())
        if sha(a.parent/'state.pkl') != sm['state_sha256']:
            raise ValueError('Changed 64-k orbital seed')
        checkpoint_state=a.resume_checkpoint/resume_latest['state_file'] if resume_latest else a.parent/'state.pkl'
        if resume_latest and (not resume_latest['complete'] or sha(checkpoint_state)!=resume_latest['state_sha256']):raise ValueError('Incomplete/changed resume checkpoint')
        with checkpoint_state.open('rb') as f:
            parent_state = pickle.load(f)
            pw = np.asarray(parent_state['pw'])
        if pw.shape[:2] != (1, 64) or pw.shape[-1] != 48:
            raise ValueError('Expected 64-k, 48-occupied orbital-only warm start')
        oldmask = np.load((a.resume_checkpoint if resume_latest else a.parent)/'mask.npy')
        if resume_latest and not np.array_equal(oldmask,mask):raise ValueError('Checkpoint restart must preserve Miller basis exactly')
        if a.read_only and not np.array_equal(oldmask,mask):
            raise ValueError('Read-only state must have the identical Miller basis')
        stage('CPU_orbital_initialization_no_density_reuse')
        projection = dict(projected=False, density_reused=False, occupation_reused=False)
        if not np.array_equal(oldmask, mask):
            pw, projection = project_miller_coefficients(pw, oldmask, mask)
            # QR stays on host. A global JAX QR would needlessly put all
            # 64 compact k blocks on GPU0 before the actual k-pool scatter.
            for k in range(64):
                pw[0, k] = np.linalg.qr(pw[0, k], mode='reduced')[0]
            projection.update(projected=True, orthonormalized=True,
                              occupation_reused=False)
        dump(a.output/'initialization.json', dict(projection, source=str(a.parent),
             initialization_only=True, source_state_sha256=sm['state_sha256']))
        (a.output/'input.xyz').write_bytes((a.reference/'input.xyz').read_bytes())
        cell = np.asarray(m['physical']['cell_angstrom'])*float(ANGSTROM2BOHR)
        positions = np.asarray(m['physical']['positions_angstrom'])*float(ANGSTROM2BOHR)
        resume_model=json.loads((a.resume_checkpoint/'model.json').read_text()) if resume_latest else None
        if resume_model and (np.max(abs(np.asarray(resume_model['cell_bohr'])-cell))>2e-11 or np.max(abs(np.asarray(resume_model['positions_bohr'])-positions))>2e-11):raise ValueError('Checkpoint geometry mismatch')
        c = ConfigDict(resume_model['config'] if resume_model else m['config'])
        c.k_grid_sizes = 4
        c.d3ft.parallel.k_pool = True
        c.crystal_file_path_path = str(a.output/'input.xyz')
        c.epoch = a.epochs
        c.optimizer_args.learning_rate = .001
        c.d3ft.optim.lr_schedule = 'cosine'
        c.d3ft.optim.lr_final = 1e-6
        c.d3ft.checkpoint=dict(root=str(a.output/'checkpoints'),interval=500)
        if a.safe_scf_wall_seconds is not None:
            c.d3ft.checkpoint.max_wall_seconds=a.safe_scf_wall_seconds
        if a.read_only:
            c.optimizer_args.learning_rate=0.
            c.d3ft.optim.lr_schedule='constant'
        c.d3ft.charge_gate = dict(enabled=True, absolute_tolerance=1e-6,
                                report_path=str(a.output/'charge_gate.jsonl'))
        with plain_fft_patch(k_pool=True), local_simpson_patch(), memory_bounded_xc(), \
                source_radial_program(m, a.reference) as info, validated_radial_splines():
            if info is None:
                raise ValueError('Frozen radial program is required')
            stage('kpool_SCF', epochs=a.epochs, devices=a.devices, k_per_device=64//a.devices,
                  fresh_uniform_occupation='2/64, no k2 occupation reuse')
            start = time.monotonic()
            init = parent_state if (a.read_only or resume_latest) else {'pw':pw}
            if resume_latest:
                import copy,importlib
                ground=importlib.import_module('d3ft.calc.ground_state')
                original_schedule=ground._learning_rate_schedule
                offset=int(resume_latest['iteration'])
                def tail_schedule(config,base=None):
                    full=copy.deepcopy(config);full.epoch=3000
                    schedule=original_schedule(full,base=base)
                    return (lambda count:schedule(count+offset)) if callable(schedule) else schedule
                ground._learning_rate_schedule=tail_schedule
                c.d3ft.checkpoint_resume=dict(completed_steps=offset,remaining_steps=a.epochs,
                    optimizer_moments_restored=False,learning_rate_schedule_total_steps=3000,
                    restart_mode='physical checkpoint warm start; moments reset; original cosine tail')
                dump(a.output/'checkpoint_resume.json',dict(source=str(a.resume_checkpoint),
                    state_sha256=resume_latest['state_sha256'],source_completed_steps=offset,
                    new_updates=a.epochs,recomputed_old_updates=0,optimizer_moments_restored=False,
                    learning_rate_schedule='original3000stepcosine tail with index offset',
                    source_model_sha256=sha(a.resume_checkpoint/'model.json')))
                try:out = calc(c, init_params=init, freq_mask=mask)
                finally:ground._learning_rate_schedule=original_schedule
            else:
                out = calc(c, init_params=init, freq_mask=mask)
            jax.block_until_ready(out.params_pw)
            out.grid_kmax = info['grid_kmax']
            seconds = time.monotonic()-start
            if len(out.k_frac) != 64 or out.core_density_grid is not None:
                raise ValueError('Expected non-NLCC full 64-k state')
            if max(np.max(abs(np.asarray(out.crystal.cell_vectors)-cell)),
                   np.max(abs(np.asarray(out.crystal.positions)-positions))) > 2e-11:
                raise ValueError('Actual E5 geometry was not preserved')
            if a.read_only:
                density_error=float(np.max(abs(np.asarray(out.density_grid)-np.load(a.parent/'density.npy')))) if (a.parent/'density.npy').exists() else None
                dump(a.output/'read_only_reconstruction.json',dict(density_max_abs_error=density_error,
                    source=str(a.parent),SCF_3000_steps_reused=True,optimizer_update_learning_rate=0))
                if density_error is not None and density_error>1e-9:raise ValueError('Read-only reconstruction changed saved density')
            if a.smoke:
                stage('memory_probe_native_force_stress_and_stationarity')
                report = fused_report(out)
                force = force_report(out)
                charge = float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell)))
                energy = float(out.total_energy)
                numerical_ok = bool(np.isfinite([charge,energy,
                    float(report['variational_residual']),float(force['acoustic_sum_rule'])]).all()
                    and abs(charge-96)<=1e-6)
                stats=[]
                for device in jax.devices():
                    try:
                        raw=device.memory_stats() or {}
                        stats.append(dict(device_id=device.id,statistics={k:int(v) for k,v in raw.items()
                            if isinstance(v,(int,np.integer))}))
                    except Exception as exc:
                        stats.append(dict(device_id=device.id,error=repr(exc)))
                dump(a.output/'smoke_numerics.json',dict(completed=True,numerical_ok=numerical_ok,
                    energy_Ha=energy,charge=charge,epochs=a.epochs,devices=a.devices,
                    k_per_device=64//a.devices,solve_seconds=seconds,
                    native_force_stress_exercised=True,device_memory_statistics=stats,
                    SCF_stationarity_certified=False,production_result=False))
                stage('memory_probe_numerics_complete',passed=numerical_ok)
                if not numerical_ok:
                    raise ValueError('Probe charge or finite-energy gate failed')
                return
            np.save(a.output/'mask.npy', mask)
            from d3ft.calc.checkpoint import _HashingWriter
            temp=a.output/'state.pkl.tmp'
            with temp.open('xb') as f:
                writer=_HashingWriter(f)
                pickle.dump(jax.tree_util.tree_map(np.asarray,
                    dict(pw=out.params_pw, occ=out.params_occ)), writer, protocol=5)
                f.flush();os.fsync(f.fileno());state_digest=writer.digest.hexdigest()
            temp.replace(a.output/'state.pkl')
            np.save(a.output/'density.npy', np.asarray(out.density_grid))
            prior_history=resume_latest.get('energy_history_Ha',[]) if resume_latest else []
            np.save(a.output/'energy_history.npy', np.asarray(prior_history+list(np.asarray(out.energy_history))))
            (a.output/'beta_tables.pkl').write_bytes((a.reference/info['tables_file']).read_bytes())
            # Save a restart even if native force/stress acceptance times out.
            saved = c.to_dict()
            saved['d3ft']['charge_gate'].pop('report_path', None)
            meta = dict(config=saved, physical=dict(m['physical'], k=[4]),
                state_sha256=state_digest, source=str(a.reference),
                source_hashes={n:sha(a.reference/n) for n in
                               ['metadata.json', 'point.json', 'mask.npy', 'input.xyz']},
                frozen_radial_program=dict(info, tables_file='beta_tables.pkl',
                    tables_sha256=sha(a.output/'beta_tables.pkl')),
                initialization_only=str(a.parent), theta=m.get('theta'),
                saved_final_state_read_only=a.read_only,
                resumed_from_checkpoint=str(a.resume_checkpoint) if resume_latest else None,
                source_completed_steps=int(resume_latest['iteration']) if resume_latest else 0,
                optimizer_moments_restored=False if resume_latest else None,
                new_response_state=True, checkpoint_accepted=False,
                jobid=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID'),
                code_sha256={str(f.relative_to(P)):sha(f) for f in (P/'d3ft').rglob('*.py')})
            dump(a.output/'metadata.json', meta)
            del pw
            gc.collect()
            stage('native_force_stress_and_stationarity')
            report = fused_report(out)
            force = force_report(out)
            residual = float(report['variational_residual'])
            asr = float(force['acoustic_sum_rule'])
            charge = float(np.asarray(out.density_grid).mean()*abs(np.linalg.det(cell)))
            passed = bool(np.isfinite([residual, asr, charge]).all()
                and residual <= 1e-8 and asr <= 5e-8 and abs(charge-96) <= 1e-6)
            point = dict(energy_Ha=float(out.total_energy), residual=residual,
                acoustic_sum_rule_Ha_Bohr=asr, charge=charge, electrons=96.,
                forces_Ha_Bohr=np.asarray(force['forces_Ha_Bohr']).tolist(),
                stress_Ha_Bohr3=np.asarray(report['stress']).tolist(),
                cell_bohr=cell.tolist(), positions_bohr=positions.tolist(),
                k_fractional=np.asarray(out.k_frac).tolist(), epochs=a.epochs,
                solve_seconds=seconds, total_completed_epochs=(int(resume_latest['iteration']) if resume_latest else 0)+a.epochs, stationarity_pass=residual<=1e-8,
                physical_passed=passed, structural_relaxation_performed=False,
                ionic_condition='Exact accepted k2 design geometry; no ionic relaxation')
            dump(a.output/'point.json', point)
            meta['checkpoint_accepted'] = passed
            dump(a.output/'metadata.json', meta)
            stage('accepted' if passed else 'failed_physical_gate', passed=passed,
                  residual_Ha=residual, ASR_Ha_Bohr=asr, charge=charge)
            if not passed:
                raise ValueError('Original k4 physical gate failed; no spectral production')
    except Exception as exc:
        stage('failed', error=repr(exc))
        raise
    finally:
        probe.finish()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=3000)
    ap.add_argument('--devices',type=int,choices=[4,8],default=8)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--read-only',action='store_true')
    ap.add_argument('--safe-scf-wall-seconds',type=float)
    ap.add_argument('--resume-checkpoint',type=Path)
    main(ap.parse_args())

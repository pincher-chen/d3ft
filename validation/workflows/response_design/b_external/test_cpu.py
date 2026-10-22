"""No JAX/GPU imports or srun: exercise dependency, budget and restart gates."""
import contextlib
import os
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from policy import dependency_ready, dump, forecast, sha, pause_after_first_gaas_point
from controller import main, prepare_reuse, validate_result


class StopWait(Exception):
    pass


class CPUChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def protocol(self):
        setting = dict(upper_GPU_hours=12., point_rate_GPU_hours=.9, reference_a0_A=5.4635,
                       reference_scope='fixed test reference')
        return dict(node='hn35', jobid=250285, material_order=['AlP','GaAs'],
                    model_order=['fitted','PBE'], E10_root=str(self.root/'E10'),
                    settings={'AlP':setting,'GaAs':dict(setting, upper_GPU_hours=280., point_rate_GPU_hours=30.)},
                    electronic_residual_tolerance=1e-6, root_relative_tolerance=1e-5,
                    fitted_vector=[0.,-.017,0.,0.,0.,-.035,-.014],
                    pseudopotentials={'AlP':{'Al.upf':'abc'}})

    def test_E10_requires_all_states_and_batches(self):
        r=self.root/'E10'
        self.assertFalse(dependency_ready(r))
        dump(r/'path.json', {'k_fractional':[[0,0,0]]*130})
        dump(r/'protocol.json',{'E10':{'path_source':str(r/'path.json')}})
        states={name:dict(passed=True,points=130,bands=56,maximum_ritz_Ha=1e-9)
                for name in ['E4_initial','E4_joint54_endpoint']}
        for name in states:dump(r/'SCF'/name/'accepted.json',dict(passed=True))
        dump(r/'E10_acceptance.json',dict(completed=True,passed=True,states=states))
        dump(r/'path_status.json',dict(stage='path_spectra_running',active={},completed_batches=8))
        self.assertFalse(dependency_ready(r))
        dump(r/'path_status.json',dict(stage='complete',active={},completed_batches=16))
        self.assertTrue(dependency_ready(r))
        states['E4_joint54_endpoint']['passed']=False
        dump(r/'E10_acceptance.json',dict(completed=True,passed=True,states=states))
        self.assertFalse(dependency_ready(r))

    def test_material_caps_and_model_costs_not_cross_subsidized(self):
        p=self.protocol();b=dict(attempts=[],logical_results={})
        self.assertTrue(forecast(p,b,'AlP')['allowed'])
        self.assertTrue(forecast(p,b,'GaAs')['allowed'])
        b['attempts']=[dict(material='AlP',model='fitted',GPU_hours=11.8,output=str(self.root/'absent'))]
        self.assertFalse(forecast(p,b,'AlP')['allowed'])
        self.assertEqual(forecast(p,b,'GaAs')['used_GPU_hours'],0.)
        self.assertFalse(forecast(p,b,'GaAs',live_GPU_hours=120.)['allowed'])

    def test_repeated_attempt_points_counted_once(self):
        p=self.protocol();out=self.root/'output'
        dump(out/'points.json',[dict(label='eq_0',residual=1e-10,seconds=300)])
        a=dict(material='AlP',model='fitted',GPU_hours=.8,output=str(out))
        b=dict(attempts=[a,dict(a,GPU_hours=.1)],logical_results={})
        f=forecast(p,b,'AlP')
        self.assertEqual(f['complete_points']['fitted'],1)
        self.assertAlmostEqual(f['used_GPU_hours'],.9)

    def test_recover_3000_and_9000_without_changing_old_files(self):
        phase3=Path(__file__).resolve().parents[2]/'phase3'
        for total in [3000,9000]:
            source=self.root/str(total);cp=source/'checkpoints/cp';cp.mkdir(parents=True)
            (cp/'state.pkl').write_bytes(b'fake CPU fixture, never unpickled')
            np.save(source/'mask.npy',np.array([True,False]))
            dump(source/'points.json',[])
            dump(cp/'model.json',{'config':{'epoch':total,'crystal_file_path_path':'eq_0.xyz'}})
            dump(cp/'latest.json',dict(complete=True,iteration=500,state_file='state.pkl',
                 state_sha256=sha(cp/'state.pkl'),energy_history_Ha=[0.]*500))
            before={str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()}
            result=prepare_reuse(source,self.root/f'stage_{total}',total,phase3)
            self.assertEqual(json.loads((result/'staging_provenance.json').read_text())['original_total_steps'],total)
            self.assertEqual(before,{str(f.relative_to(source)):sha(f) for f in source.rglob('*') if f.is_file()})
            with self.assertRaises(ValueError):
                prepare_reuse(source,self.root/f'wrong_{total}',total+1,phase3)

    def test_timeout_without_complete_checkpoint_is_not_cold_restarted(self):
        source=self.root/'empty';source.mkdir()
        phase3=Path(__file__).resolve().parents[2]/'phase3'
        with self.assertRaises(ValueError):prepare_reuse(source,self.root/'dest',3000,phase3)

    def test_completed_POINT_only_recovery_without_partial_checkpoint(self):
        source=self.root/'complete_point';source.mkdir()
        dump(source/'points.json',[dict(label='eq_0',residual=1e-10)])
        np.save(source/'mask.npy',np.array([True]))
        phase3=Path(__file__).resolve().parents[2]/'phase3'
        staged=prepare_reuse(source,self.root/'staged',3000,phase3)
        self.assertEqual(json.loads((staged/'points.json').read_text())[0]['label'],'eq_0')

    def test_GaAs_first_point_stops_with_76_and_preserves_POINT(self):
        work=self.root/'GaAs'
        dump(work/'eq_0.json',dict(label='eq_0',residual=1e-10))
        dump(work/'points.json',[dict(label='eq_0',residual=1e-10)])
        before=sha(work/'points.json')
        with self.assertRaises(SystemExit) as exit:
            pause_after_first_gaas_point(work,'eq_0',np.array([True]))
        self.assertEqual(exit.exception.code,76)
        self.assertEqual(sha(work/'points.json'),before)
        self.assertTrue((work/'mask.npy').exists())
        self.assertTrue(json.loads((work/'first_point_pause.json').read_text())['next_point_not_authorized'])
        dump(work/'eq_1.json',dict(residual=1e-10))
        with self.assertRaises(ValueError):pause_after_first_gaas_point(work,'eq_1',np.array([True]))

    def test_original_pressure_and_residual_gates(self):
        p=self.protocol();out=self.root/'out'
        summary=dict(parameter_vector=[0.]*7,a0_A=5.5,equilibrium_residual=1e-9,root_relative=1e-6)
        dump(out/'metadata.json',dict(parameter_vector=[0.]*7,pseudopotentials={'Al.upf':'abc'}))
        dump(out/'summary.json',summary)
        self.assertTrue(validate_result(out,p,'AlP','PBE')['passed'])
        summary['root_relative']=1.01e-5;dump(out/'summary.json',summary)
        self.assertFalse(validate_result(out,p,'AlP','PBE')['passed'])
        summary['parameter_vector'][0]=.01;dump(out/'summary.json',summary)
        with self.assertRaises(ValueError):validate_result(out,p,'AlP','PBE')

    def test_GaAs_200_cap_exact_boundary_and_overforecast(self):
        p=self.protocol();p['settings']['GaAs']['upper_GPU_hours']=200.
        out=self.root/'gaas';points=[dict(label='eq_'+str(i),residual=1e-10,seconds=18000) for i in range(2)]
        dump(out/'points.json',points);dump(out/'eq_1.json',points[-1])
        dump(self.root/'protocol.json',p)
        dump(self.root/'GaAs_first_point_policy.json',dict(automatic_continue_within_cap=True))
        dump(self.root/'GaAs_first_point_measurement.json',dict(reported=True))
        b=dict(attempts=[dict(material='GaAs',model='fitted',GPU_hours=79.5,gpus=list(range(1,7)),output=str(out))],logical_results={},active_model='fitted')
        dump(self.root/'budget.json',b)
        env=dict(D3FT_B_ROOT=str(self.root),D3FT_B_START_EPOCH='1000',D3FT_B_GPU_COUNT='6')
        with patch.dict(os.environ,env),patch('time.time',return_value=1000):
            pause_after_first_gaas_point(out,'eq_1',np.array([True]))
            self.assertEqual(json.loads((out/'eq_1_continuation_budget.json').read_text())['forecast_GPU_hours'],200.)
            b['attempts'][0]['GPU_hours']=79.6;dump(self.root/'budget.json',b)
            with self.assertRaises(SystemExit) as exc:pause_after_first_gaas_point(out,'eq_1',np.array([True]))
            self.assertEqual(exc.exception.code,75)

    def test_GaAs_continuation_requires_first_point_report(self):
        dump(self.root/'GaAs_first_point_policy.json',dict(automatic_continue_within_cap=True))
        out=self.root/'gaas';dump(out/'eq_0.json',dict(residual=1e-10))
        with patch.dict(os.environ,{'D3FT_B_ROOT':str(self.root)}):
            with self.assertRaises(ValueError):pause_after_first_gaas_point(out,'eq_0',np.array([True]))

    def test_adopted_first_point_76_reports_then_queues_reuse_without_marking_failure(self):
        p=self.protocol();p.update(project=str(self.root),runtime=str(self.root/'runtime'))
        p['settings']['GaAs']['upper_GPU_hours']=200.
        dump(self.root/'protocol.json',p);dump(self.root/'execution_SHA.json',{})
        dump(self.root/'GaAs_first_point_policy.json',dict(automatic_continue_within_cap=True))
        out=self.root/'point';dump(out/'points.json',[dict(label='eq_0',residual=1e-10,seconds=18000,a0_A=5.75,pressure_GPa=-1.)])
        run=self.root/'run';run.mkdir()
        a=dict(material='GaAs',model='fitted',GPU_hours=None,gpus=list(range(1,7)),output=str(out),run=str(run),pid=123,start_epoch=1000)
        dump(self.root/'budget.json',dict(attempts=[a],logical_results={k:dict(passed=True) for k in ['AlP_fitted','AlP_PBE']}))
        dump(self.root/'CPU_handover_authorization.json',dict(child_pid=123))
        with patch('controller.AdoptedChild.poll',return_value=76),patch('controller.time.time',return_value=19000),\
             patch('controller.note'),patch('controller.dependency_ready',return_value=True),\
             patch('controller.idle_eight_cards',side_effect=StopWait),patch('controller.subprocess.Popen') as submit:
            with self.assertRaises(StopWait):main(self.root)
            submit.assert_not_called()
        report=json.loads((self.root/'GaAs_first_point_measurement.json').read_text())
        self.assertTrue(report['continuation_authorized'])
        self.assertEqual(report['actual_allocation_GPU_hours'],30.)
        self.assertNotIn('GaAs_fitted',json.loads((self.root/'budget.json').read_text())['logical_results'])

    def test_no_probe_or_submit_before_E10_acceptance(self):
        p=self.protocol();dump(self.root/'protocol.json',p)
        dump(self.root/'budget.json',dict(attempts=[],logical_results={}))
        dump(self.root/'execution_SHA.json',{})
        with patch('controller.idle_eight_cards') as probe, patch('controller.subprocess.Popen') as launch, \
                patch('controller.time.sleep',side_effect=StopWait):
            with self.assertRaises(StopWait):main(self.root)
            probe.assert_not_called();launch.assert_not_called()

    def test_failed_AlP_blocks_GaAs(self):
        p=self.protocol();p['project']=str(self.root/'paper_fixture')
        (Path(p['project'])/'paper').mkdir(parents=True)
        dump(self.root/'protocol.json',p)
        dump(self.root/'budget.json',dict(attempts=[],logical_results={
            'AlP_fitted':dict(passed=True),'AlP_PBE':dict(passed=False)}))
        dump(self.root/'execution_SHA.json',{})
        with patch('controller.idle_eight_cards') as probe, patch('controller.subprocess.Popen') as launch:
            main(self.root);probe.assert_not_called();launch.assert_not_called()
        self.assertEqual(json.loads((self.root/'status.json').read_text())['stage'],
                         'stopped_AlP_not_passed_GaAs_not_submitted')


if __name__=='__main__':
    unittest.main(verbosity=2)

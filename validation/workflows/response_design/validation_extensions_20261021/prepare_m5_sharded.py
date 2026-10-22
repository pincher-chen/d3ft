"""New layout-only runtime, never modifies original response math snapshots."""
import datetime,hashlib,json,shutil
from pathlib import Path
W=Path(__file__).resolve().parent;P=W.parents[3];R=Path(json.loads((P/'validation/results/response_design_20261021_internal_review/current.json').read_text())['root']);old=json.loads((R/'M5-A/protocol.json').read_text());stamp=datetime.datetime.now().astimezone();Q=R/'M5-A'/('sharded_approved_'+stamp.strftime('%Y%m%dT%H%M%S'));Q.mkdir(exist_ok=False);rt=Q/'runtime';rt.mkdir()
for name in ['d3ft','validation/workflows']:shutil.copytree(Path(old['runtime']['mesh'])/name,rt/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
D=rt/'validation/workflows/response_design/reverse_response';original=(D/'kmesh_response.py').read_text();(D/'kmesh_response_original.py').write_text(original)
changes=[('self.rho=jax.jit(', 'self.rho=self._jit_constant('),('self.project=jax.jit(project)','self.project=self._jit_vector(project)'),('self.apply_local=jax.jit(self.local_action)','self.apply_local=self._jit_local(self.local_action)'),('self.ham=jax.jit(', 'self.ham=self._jit_vector('),('self.kernel=jax.jit(', 'self.kernel=self._jit_replicated('),('self.delta_rho=jax.jit(self.density_tangent)','self.delta_rho=self._jit_density_tangent(self.density_tangent)'),('self.hessian=jax.jit(', 'self.hessian=self._jit_vector('),('self.precondition=jax.jit(', 'self.precondition=self._jit_vector('),('self.edge_density=jax.jit(', 'self.edge_density=self._jit_constant('),('self.explicit_gap=jax.jit(', 'self.explicit_gap=self._jit_replicated('),('self.external=jax.jit(', 'self.external=self._jit_external('),('self.scalar_geometry=jax.jit(scalar_geometry)','self.scalar_geometry=self._jit_geometry_scalar(scalar_geometry)'),('self.scalar_gradient=jax.jit(jax.grad(scalar_geometry,argnums=0))','self.scalar_gradient=self._jit_geometry_scalar(jax.grad(scalar_geometry,argnums=0))')]
s=original
for before,after in changes:
 assert s.count(before)==1,(before,s.count(before));s=s.replace(before,after)
helper=""" def _jit_constant(self,f):return jax.jit(f)
 def _jit_vector(self,f):return jax.jit(f)
 def _jit_local(self,f):return jax.jit(f)
 def _jit_replicated(self,f):return jax.jit(f)
 def _jit_density_tangent(self,f):return jax.jit(f)
 def _jit_external(self,f):return jax.jit(f)
 def _jit_geometry_scalar(self,f):return jax.jit(f)
"""
s=s.replace(' def put_k(self,x):',helper+' def put_k(self,x):');(D/'kmesh_response.py').write_text(s)
# Reverse all edits proves that numerical expressions are exactly preserved.
back=s.replace(helper,'')
for before,after in reversed(changes):back=back.replace(after,before)
assert back==original
prior=json.loads((R/'M5-A/summary.json').read_text())['GPU_hours'];proto={**old,'root':str(Q),'original_root':str(R),'runtime_path':str(rt),'node':'hn35','jobid':250285,'gpus':list(range(8)),'authority':'Claude/User2026-10-21 approval: M2 first, M5after; actualGPUlayoutgate required','registered_at':stamp.isoformat(),'original_protocol_preserved':str(R/'M5-A/protocol.json'),'historical_GPU_hours':prior,'GPU_hours_cap':35,'remaining_GPU_hours':35-prior,'queue_conditions':['M2-A terminal summary exists and its GPUstep exited','all8hn35cards memory0/util0 snapshot','allnewCPUtests pass'],'first_GPU_stage':'64k synthetic low-memory layout/Hessian/PCG probe versus independent CPU arrays, all8GPU; fixed threshold2e-10, inner threshold1e-9+1e-11absreference','probe_timeout_seconds':600,'production_policy':'reuse64certifiedspectra/eigenvectors; kpoolTrue read-only64k rebuild on8cards; hard64k full54 adjoint only, no3extra fullforward solves','first_two_PCG_checks':'exact runtime vectors vsNumPydot; allmemory<78GiB; source_density≤1e-9; charge/primal/projectedPCG/translation original gates unchanged','prediction_policy':'At first2PCGiterations and each lateriteration predict at least32totaliterations (or current+2 ifgreater), reserve1440s geometricpullback; originalspent+newspent+8*predicted_remaining/3600>35 =>stop; no loosenediterations/tolerances','stopping':'AnyGPUlayout/NumPyinner/charge/primal/memory/responsegate fails ->retain failureandstop,noautomaticretry. Allcompile/failedtimecount35. No newM5beforeM2; nohn48M5','M2_code_and_jobs_untouched':True}
(Q/'protocol.json').write_text(json.dumps(proto,indent=2)+'\n');(Q/'layout_changes.json').write_text(json.dumps({'all_numerical_expressions_identical_after_reversing_wrappers':True,'changes':changes,'original_source_SHA256':hashlib.sha256(original.encode()).hexdigest()},indent=2)+'\n');(Q/'protocol_SHA256.json').write_text(json.dumps({'protocol.json':hashlib.sha256((Q/'protocol.json').read_bytes()).hexdigest()},indent=2)+'\n');(R/'M5-A/current_sharded.json').write_text(json.dumps({'root':str(Q),'registered_at':stamp.isoformat()},indent=2)+'\n')
line=f"\n- {stamp.isoformat()} 用户/Claude批准M5八卡布局方案，预登记新目录`{Q.relative_to(P)}`：只改分片布局及jit输入输出约束，原数学表达式可逐字还原；先八卡真实GPU小规模Hessian/PCG与CPU数组对照通过，再64k硬目标54伴随；M2终态且hn35全空才能提交。旧5.34246计入35GPUh；所有探针/编译/失败计账，首次2次PCG及后续按至少32迭代+0.4h反传余量预测，超35即停，不放宽原门、不自动重试。64谱复用，未改正式稿、M2进程不动。\n"
for file in ['paper/response_design_EXPERIMENTS_20261014.md','response_design_to_do_list.md']:
 with (P/file).open('a') as f:f.write(line)
print(Q)

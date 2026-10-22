"""Record natural completion of already launched legacy audit shards; submit nothing."""
from pathlib import Path
import json,datetime,time
P=Path(__file__).resolve().parents[3];A=P/'validation/results/response_design_20261014_e2_audit';decision=json.loads((A/'scope_decision_20261014.json').read_text());pending=list(decision['active_shards']);done=[]
while pending:
 for row in pending[:]:
  stat=Path(f'/proc/{row["pid"]}/stat')
  if stat.exists() and stat.read_text().split()[2]!='Z':continue
  f=Path(row['record']);rec=json.loads(f.read_text());out=Path(row['output']);partial=(out/'partial.json').exists();end=datetime.datetime.now();hours=(end-datetime.datetime.fromisoformat(row['start'])).total_seconds()/3600
  rec.update(end=end.isoformat(),GPU_hours=hours,status='按用户决定保留的在途分片已自然结束',partial_artifact_exists=partial,full_state_acceptance=False,further_submission=False);f.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n');done.append(dict(pid=row['pid'],gpu=row['gpu'],GPU_hours=hours,partial_artifact_exists=partial,full_state_acceptance=False));pending.remove(row);print('LEGACY_DRAIN',row['pid'],'partial retained',partial,flush=True)
 (A/'legacy_drain_status.json').write_text(json.dumps(dict(time=datetime.datetime.now().isoformat(),pending=pending,done=done,status='收尾中' if pending else '在途分片自然结束，无后续任务'),ensure_ascii=False,indent=2)+'\n')
 if pending:time.sleep(10)
s=json.loads((A/'density_queue_status.json').read_text());used=sum(r['GPU_hours'] for r in done);s.update(time=datetime.datetime.now().isoformat(),stage='旧校准队列已禁用，在途分片自然结束',active_shards=0,reserved_upper_bound_GPU_hours=0.,gpus=[],legacy_shard_GPU_hours=used,E2_remaining_conservative_GPU_hours=max(0.,100.-5.-used),full_legacy_state_acceptance=False);(A/'density_queue_status.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (P/'response_design_to_do_list.md').open('a') as f:f.write(f'\n- {datetime.datetime.now().isoformat()} 六个旧校准态按用户决定保持未验；先前4个在途审计分片已自然收尾，约{used:.3f}GPUh记录保留，无后续旧校准任务。E2预算留给MgSiN2。\n')
print('Legacy audit drain completed; no new computation submitted.',flush=True)

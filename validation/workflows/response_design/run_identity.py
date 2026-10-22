"""Atomic immutable run names shared by E3/E5/E6/E7 controllers."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,re,time

def reserve_name(root,logical,*,output_parent=None,log_parent=None):
    root=Path(root)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*',logical):
        raise ValueError('Run logical name must be a safe basename')
    registry=root/'.run_registry';registry.mkdir(parents=True,exist_ok=True)
    for attempt in range(1000):
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        tag=f'{logical}__{stamp}_{os.getpid()}_{attempt}'
        if output_parent is not None and (Path(output_parent)/tag).exists():continue
        if log_parent is not None and (Path(log_parent)/(tag+'.log')).exists():continue
        entry=registry/tag
        try:entry.mkdir()
        except FileExistsError:continue
        (entry/'identity.json').write_text(json.dumps(dict(logical_name=logical,run_name=tag,
            reserved_at_UTC=datetime.now(timezone.utc).isoformat(),controller_pid=os.getpid(),
            output_parent=str(output_parent) if output_parent is not None else None,
            log_parent=str(log_parent) if log_parent is not None else None),indent=2)+'\n')
        return tag
    raise RuntimeError('Unable to reserve a unique run identity')

def unique_output_path(path):
    path=Path(path)
    return path.parent/reserve_name(path.parent,path.name,output_parent=path.parent)

def await_detached_record(path):
    """Observe an existing srun after CPU-only handover; never signal its PID.

    Scientific completion must be checked by the caller. The inherited exit
    code is unavailable; GPU-hours include a conservative one-second tail.
    """
    path=Path(path);row=json.loads(path.read_text());pid=row['pid']
    while True:
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]=='Z':break
        time.sleep(1)
    end=datetime.now();start=datetime.fromisoformat(row.get('start',row.get('time')))
    row.update(end=end.isoformat(),returncode=None,inherited_waitpid_exit_code_unavailable=True,
        GPU_hours=max(0,(end-start).total_seconds())*len(row.get('gpus',[row.get('gpu',0)]))/3600,
        GPU_hours_scope='conservative observed end upper bound after CPU-only handover')
    return row

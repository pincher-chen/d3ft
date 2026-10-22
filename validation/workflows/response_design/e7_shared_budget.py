"""Atomic GPU-hour reservations shared by independent E7 fit controllers."""
import contextlib
import fcntl
import json
from pathlib import Path


class SharedBudget:
    def __init__(self, path):
        self.path = Path(path)

    @contextlib.contextmanager
    def locked(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text())
            yield data
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=2) + '\n')
            tmp.replace(self.path)

    def reserve(self, key, hours):
        with self.locked() as d:
            if key in d['jobs']:
                raise ValueError('Duplicate GPU-hour reservation: ' + key)
            total = d['initial_GPU_hours'] + sum(
                j.get('GPU_hours', j['reserved_GPU_hours']) for j in d['jobs'].values())
            record_only = d.get('budget_enforcement') == 'record_only'
            if record_only and not any(key.startswith(prefix) for prefix in
                                      d.get('record_only_run_prefixes', [])):
                raise RuntimeError('Record-only budget authorization does not cover this worker')
            if not record_only and (d['cap_GPU_hours'] is None or total + hours > d['cap_GPU_hours']):
                raise RuntimeError('Shared E7 budget cannot reserve next DFT worker')
            d['jobs'][key] = dict(status='reserved', reserved_GPU_hours=hours)
            if record_only:
                d['jobs'][key].update(budget_enforcement='record_only',
                    projected_with_reservations_GPU_hours=total+hours,
                    authorization_scope='existing GaP fitted/PBE evaluation and exit124 continuations only')

    def finish(self, key, hours, status):
        with self.locked() as d:
            row = d['jobs'][key]
            if 'GPU_hours' in row:
                raise ValueError('GPU-hour record already committed: ' + key)
            row.update(GPU_hours=hours, status=status)

    def summary(self):
        d = json.loads(self.path.read_text())
        committed = d['initial_GPU_hours'] + sum(j.get('GPU_hours', 0) for j in d['jobs'].values())
        reserved = sum(j['reserved_GPU_hours'] for j in d['jobs'].values() if 'GPU_hours' not in j)
        return dict(spent_GPU_hours=committed, reserved_GPU_hours=reserved,
                    cap_GPU_hours=d['cap_GPU_hours'],
                    budget_enforcement=d.get('budget_enforcement','enforce'),
                    reference_cap_GPU_hours=d.get('reference_cap_GPU_hours',d['cap_GPU_hours']))

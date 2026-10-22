"""Low-overhead, PID-specific NVML sampling; never initializes a JAX device.

Uses the NVML v2 process ABI (not the incompatible v3 struct). Sampled peaks
are lower bounds. Device usage and process usage are recorded separately.
"""
import csv
import ctypes as C
import json
import os
from pathlib import Path
import threading
import time


class MemoryInfo(C.Structure):
    _fields_ = [('total', C.c_ulonglong), ('free', C.c_ulonglong),
                ('used', C.c_ulonglong)]


class ProcessInfo(C.Structure):
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
                ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]


class NVML:
    def __init__(self, indices):
        self.lib = C.CDLL('libnvidia-ml.so.1')
        self.check(self.lib.nvmlInit_v2())
        self.handles = {}
        for index in indices:
            handle = C.c_void_p()
            self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(C.c_uint(index), C.byref(handle)))
            uuid = C.create_string_buffer(128)
            self.check(self.lib.nvmlDeviceGetUUID(handle, uuid, len(uuid)))
            self.handles[index] = (handle, uuid.value.decode())

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML error ' + str(code))

    def snapshot(self, pid):
        rows = []
        for index, (handle, uuid) in self.handles.items():
            info = MemoryInfo()
            self.check(self.lib.nvmlDeviceGetMemoryInfo(handle, C.byref(info)))
            count = C.c_uint(128)
            processes = (ProcessInfo * count.value)()
            self.check(self.lib.nvmlDeviceGetComputeRunningProcesses_v2(
                handle, C.byref(count), processes))
            matches = [int(p.usedGpuMemory) for p in processes[:count.value] if p.pid == pid]
            if any(x >= (1 << 63) for x in matches):
                raise RuntimeError('NVML process memory unavailable (not zero)')
            rows.append(dict(index=index, uuid=uuid, pid=pid,
                             process_bytes=sum(matches) if matches else 0,
                             process_present=bool(matches), device_bytes=int(info.used),
                             total_bytes=int(info.total)))
        return rows


class Sampler:
    def __init__(self, output, indices, interval=.1):
        self.output = Path(output)
        self.indices = list(indices)
        self.interval = interval
        self.pid = os.getpid()
        self.stage = 'preparation'
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.peaks = {}
        self.stages = {}
        self.errors = []
        self.count = 0
        self.started = time.monotonic()

    def start(self):
        self.nvml = NVML(self.indices)
        self.sample()  # fail closed before launching physical work
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def set_stage(self, stage):
        with self.lock:
            self.stage = stage
        return self.sample()

    def sample(self):
        rows = self.nvml.snapshot(self.pid)
        with self.lock:
            stage = self.stage
            self.count += 1
            for row in rows:
                key = str(row['index'])
                peak = self.peaks.setdefault(key, dict(process_bytes=0, device_bytes=0,
                                                       present_samples=0, uuid=row['uuid']))
                peak['process_bytes'] = max(peak['process_bytes'], row['process_bytes'])
                peak['device_bytes'] = max(peak['device_bytes'], row['device_bytes'])
                peak['present_samples'] += int(row['process_present'])
                st = self.stages.setdefault(stage, {}).setdefault(key, dict(
                    baseline_process_bytes=row['process_bytes'], peak_process_bytes=0,
                    baseline_device_bytes=row['device_bytes'], peak_device_bytes=0))
                st['peak_process_bytes'] = max(st['peak_process_bytes'], row['process_bytes'])
                st['peak_device_bytes'] = max(st['peak_device_bytes'], row['device_bytes'])
            # CSV is written by the sampling thread; synchronous snapshots still
            # contribute to summary peaks, including stage boundaries.
        return stage, rows

    def loop(self):
        with (self.output / 'memory_samples.csv').open('x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seconds', 'stage', 'physical_gpu', 'gpu_uuid', 'pid',
                             'process_bytes', 'process_present', 'device_bytes', 'total_bytes'])
            while not self.stop.is_set():
                try:
                    stage, rows = self.sample()
                    for r in rows:
                        writer.writerow([time.monotonic()-self.started, stage, r['index'],
                                         r['uuid'], r['pid'], r['process_bytes'],
                                         r['process_present'], r['device_bytes'], r['total_bytes']])
                    f.flush()
                except Exception as exc:
                    with self.lock:
                        if len(self.errors) < 20:
                            self.errors.append(repr(exc))
                self.stop.wait(self.interval)

    def report(self):
        with self.lock:
            d = dict(pid=self.pid, physical_gpus=self.indices, interval_seconds=self.interval,
                     samples=self.count, peaks=self.peaks, stages=self.stages, errors=self.errors,
                     sampled_peak_is_lower_bound=True,
                     source='NVML v2 exact PID; process bytes and device usage are distinct',
                     jobid=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID'))
            return json.loads(json.dumps(d))

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=3)
        self.sample()
        d = self.report()
        for devices in d['stages'].values():
            for x in devices.values():
                x['peak_increment_process_bytes'] = x['peak_process_bytes'] - x['baseline_process_bytes']
                x['peak_increment_device_bytes'] = x['peak_device_bytes'] - x['baseline_device_bytes']
        (self.output / 'memory_peak.json').write_text(json.dumps(d, indent=2)+'\n')
        return d


def check_peak(report, indices, limit_bytes=78 * 2**30):
    peaks = report['peaks']
    return (not report['errors'] and all(
        str(i) in peaks and peaks[str(i)]['present_samples'] > 0
        and peaks[str(i)]['process_bytes'] < limit_bytes
        and peaks[str(i)]['device_bytes'] < limit_bytes for i in indices))

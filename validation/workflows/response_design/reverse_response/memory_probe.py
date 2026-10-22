"""PID-specific nvidia-smi sampling for development, not benchmark timing."""
import os,subprocess,threading,time,json,csv
class MemoryProbe:
 def __init__(self,output,interval=1.):
  self.output=output;self.interval=interval;self.stop_event=threading.Event();self.peak=0;self.samples=0;self.errors=[];self.devices=set()
 def start(self):
  self.start_time=time.monotonic();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
 def loop(self):
  with (self.output/'memory_samples.csv').open('x',newline='') as f:
   writer=csv.writer(f);writer.writerow(['seconds','pid','gpu_uuid','used_MiB'])
   while not self.stop_event.is_set():
    try:
     r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
     if r.returncode:raise RuntimeError(r.stderr.strip())
     for line in r.stdout.splitlines():
      pid,uuid,mem=[v.strip() for v in line.split(',')]
      if int(pid)==os.getpid():
       value=float(mem);writer.writerow([time.monotonic()-self.start_time,pid,uuid,value]);f.flush()
       self.peak=max(self.peak,value);self.samples+=1;self.devices.add(uuid)
    except Exception as exc:
     if len(self.errors)<10:self.errors.append(str(exc))
    self.stop_event.wait(self.interval)
 def finish(self):
  self.stop_event.set();self.thread.join(timeout=6)
  d=dict(peak_used_MiB=self.peak if self.samples else None,samples=self.samples,gpu_UUIDs=sorted(self.devices),
   pid=os.getpid(),jobid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
   source='nvidia-smi exact process PID, one-second sampling; sampled peak is a lower bound',
   scope='development memory short test; sampling overhead excluded from future formal timing',errors=self.errors)
  (self.output/'memory_peak.json').write_text(json.dumps(d,indent=2)+'\n')

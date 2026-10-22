"""Observe allocation memory during XC720 recovery without changing its run."""

from gpu_probe_retry import install_gpu_probe_retry
install_gpu_probe_retry()
import datetime
import json
from pathlib import Path
import subprocess
import time

P=Path('${D3FT_ROOT}')
G=P/'validation/results/response_design_20261005_numerical_verification/gradient_v7'
command=['srun','--jobid=250212','--overlap','-N1','-n1','--cpus-per-task=1',
         'nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits']

def main():
    maximum=0;count=0;begun=time.monotonic()
    with (G/'memory_observations.jsonl').open('x') as handle:
        while time.monotonic()-begun<8*3600:
            state=json.loads((G/'gradient_status.json').read_text())
            sampled=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
            rows=[]
            if sampled.returncode==0:
                for line in sampled.stdout.strip().splitlines():
                    index,memory,util=map(int,line.split(','))
                    rows.append(dict(gpu=index,memory_MiB=memory,utilization_percent=util))
                    if index==0:maximum=max(maximum,memory)
            record=dict(time=datetime.datetime.now().isoformat(),stage=state['stage'],gpus=rows,
                        gpu0_observed_peak_MiB=maximum,returncode=sampled.returncode)
            log=G/'logs/fine_reverse_v7.log'
            if log.exists():
                with log.open('rb') as f:
                    f.seek(max(0,log.stat().st_size-4000));record['last_log_line']=f.read().decode(errors='replace').splitlines()[-1:]
            handle.write(json.dumps(record)+'\n');handle.flush();count+=1
            (G/'memory_observation_summary.json').write_text(json.dumps(dict(time=record['time'],samples=count,
                gpu0_observed_peak_MiB=maximum,latest=record,
                scope='10-second device memory samples; an observed lower bound on peak allocation, not an exact allocator peak or a mode-comparison benchmark'),indent=2)+'\n')
            if state['stage'] in ['completed','failed']:break
            time.sleep(10)

if __name__=='__main__':main()

"""Retry observational GPU probes; never turn an unknown GPU into an idle one.

Install in CPU controllers only. Calculation subprocesses retain their original
physical timeouts. Three attempts per batch, 60 s between failures; exhausted
batches remain in the waiting state, with a durable JSONL diagnostic.
"""
import datetime,json,os,subprocess,threading,time
from pathlib import Path

def is_gpu_probe(command):
    if isinstance(command,str):return 'nvidia-smi' in command
    return any('nvidia-smi' in str(v) for v in command or [])

def retry_run(command,*args,runner=None,sleeper=time.sleep,delay=60,attempts=3,journal=None,**kwargs):
    runner=runner or _original_run
    if not is_gpu_probe(command):return runner(command,*args,**kwargs)
    attempt=0
    while True:
        attempt+=1
        try:
            # check=True must not bypass retries on transient nonzero exits.
            options=dict(kwargs);options['check']=False
            if options.get('timeout') is None:options['timeout']=45
            result=runner(command,*args,**options)
            if result.returncode==0 and (not any('--query-gpu' in str(v) for v in command) or str(result.stdout).strip()):return result
            error='probe returncode '+str(result.returncode)+': '+str(result.stderr)[-1000:]
        except (subprocess.TimeoutExpired,OSError) as exc:error=repr(exc)
        row=dict(time=datetime.datetime.now().astimezone().isoformat(),pid=os.getpid(),
                 batch=(attempt-1)//attempts+1,attempt_in_batch=(attempt-1)%attempts+1,
                 error=error,retry_after_seconds=delay,unknown_is_not_idle=True)
        print('GPU_PROBE_RETRY '+json.dumps(row),flush=True)
        target=journal or os.environ.get('response_GPU_PROBE_JOURNAL')
        if target:
            with Path(target).open('a') as f:f.write(json.dumps(row)+'\n')
        sleeper(delay)

_original_run=subprocess.run
_lock=threading.Lock()
def install_gpu_probe_retry():
    with _lock:
        if getattr(subprocess.run,'_response_gpu_retry',False):return
        original=subprocess.run
        def wrapped(command,*args,**kwargs):
            return retry_run(command,*args,runner=original,**kwargs)
        wrapped._response_gpu_retry=True
        subprocess.run=wrapped

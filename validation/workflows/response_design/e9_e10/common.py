"""CPU-only immutable records and resume validation for E9/E10."""
import datetime,hashlib,json,os
from pathlib import Path

def now():return datetime.datetime.now().astimezone().isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def read(p):return json.loads(Path(p).read_text())
def unique(base,label):
 base=Path(base);name=label+'__'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+str(os.getpid());p=base/name;p.mkdir(parents=True,exist_ok=False);return p

def latest_checkpoint(root,total):
 candidates=[]
 for f in Path(root).glob('**/checkpoints/**/latest.json'):
  try:
   q=read(f);k=f.parent/q['state_file'];model=f.parent/'model.json'
   if q.get('complete') and 0<int(q['iteration'])<=total and k.exists() and model.exists() and sha(k)==q['state_sha256']:candidates.append((int(q['iteration']),f.parent))
  except (ValueError,KeyError,OSError):continue
 return max(candidates,key=lambda x:x[0]) if candidates else None

"""Wait for material accuracy acceptance, then run one same-GPU cost task."""
import argparse,json,subprocess,time,datetime
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();root=a.root
for _ in range(2160):
 try:
  result=json.loads((root/'acceptance.json').read_text());break
 except (FileNotFoundError,json.JSONDecodeError):time.sleep(20)
else:raise SystemExit('Response acceptance never arrived')
if not result['passed']:
 (root/'warmed_cost_not_run.json').write_text(json.dumps({'reason':'upstream material accuracy gate failed','time':datetime.datetime.now().isoformat()},indent=2)+'\n')
 raise SystemExit(1)
command=json.loads((root/'warmed_cost_command.json').read_text())
(root/'warmed_cost_launch.json').write_text(json.dumps({'time':datetime.datetime.now().isoformat(),'command':command},indent=2)+'\n')
with (root/'warmed_cost.log').open('x') as out:r=subprocess.run(command,stdout=out,stderr=subprocess.STDOUT)
raise SystemExit(r.returncode)

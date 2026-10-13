from pathlib import Path
import json,sys
from compare_reference import compare
s,a,reference,output=sys.argv[1:]
d=compare(Path(s),Path(a),reference)
Path(output).write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))

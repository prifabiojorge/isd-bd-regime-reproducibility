"""Check the scientific source/config hashes in the lightweight repository."""
from pathlib import Path
import hashlib,json
base=Path(__file__).resolve().parents[1]
m=json.loads((base/'MANIFESTO_CODIGO.json').read_text())
bad=[k for k,v in m.items() if not (base/k).is_file() or hashlib.sha256((base/k).read_bytes()).hexdigest()!=v]
if bad:raise SystemExit('FAILED: '+', '.join(bad))
print(json.dumps({'status':'PASS','scientific_source_files_verified':len(m),'scientific_campaign_reexecuted':False}))

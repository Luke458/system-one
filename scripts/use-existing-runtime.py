"""Create a local overlay without changing an existing working ROCm environment.
Usage: python scripts/use-existing-runtime.py /absolute/path/to/runtime/bin/python
"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
source = Path(sys.argv[1]).absolute()
target = root / '.venv'
if target.exists():
    raise SystemExit('.venv already exists; refusing to overwrite it')
paths = json.loads(subprocess.check_output([str(source), '-c', 'import json,site; print(json.dumps(site.getsitepackages()))'], text=True))
subprocess.run(['uv', 'venv', '--python', str(source), str(target)], check=True)
python = target / 'bin/python'
local = json.loads(subprocess.check_output([str(python), '-c', 'import json,site; print(json.dumps(site.getsitepackages()))'], text=True))[0]
(Path(local) / 'host_rocm_runtime.pth').write_text('\n'.join(paths) + '\n')
subprocess.run(['uv', 'pip', 'install', '--python', str(python), '--no-deps', '-e', str(root)], check=True)
print('Overlay created. Dependencies are inherited read-only; upstream runtime changes can affect this project.')

"""Check the exported inventory and syntax without importing or executing research code."""
from pathlib import Path
import ast, hashlib

root=Path(__file__).resolve().parents[2]
manifest=root/'research-code/source-manifest.sha256'
listed=set()
for line in manifest.read_text().splitlines():
    expected,name=line.split('  ',1)
    path=root/name
    assert path.is_file(),name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    listed.add(name)
actual={p.relative_to(root).as_posix() for p in (root/'research-code/overlay').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert listed==actual,sorted(actual^listed)
python_files=list((root/'research-code/overlay').rglob('*.py'))+list((root/'course-analysis/scripts').glob('*.py'))
for path in python_files:ast.parse(path.read_text(),filename=str(path))
print(f'PASS: {len(listed)} overlay files and {len(python_files)} Python syntax checks; no model execution.')

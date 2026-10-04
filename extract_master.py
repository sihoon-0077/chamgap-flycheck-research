"""Extract the self-contained master document into a NEW directory, without execution."""
from __future__ import annotations
import argparse
import hashlib
import re
from pathlib import Path, PurePosixPath

PATTERN = re.compile(
    r'^<!-- CHAMGAP_FILE_BEGIN path=([^\n ]+) sha256=([0-9a-f]{64}) -->\n'
    r'~~~~[^\n]*\n(.*?)^~~~~\n<!-- CHAMGAP_FILE_END -->\s*$',
    re.MULTILINE | re.DOTALL,
)

def extract(master: Path, output: Path) -> int:
    text=master.read_text(encoding='utf-8')
    items=PATTERN.findall(text)
    if not items:
        raise ValueError('No embedded source files found')
    root=output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Use a new or empty output directory')
    checked=[]; seen=set()
    for rel, expected, body in items:
        p=PurePosixPath(rel)
        if p.is_absolute() or any(part in ('..','.') for part in p.parts) or ':' in rel or '\\' in rel:
            raise ValueError(f'Unsafe path: {rel}')
        if rel in seen: raise ValueError(f'Duplicate path: {rel}')
        seen.add(rel)
        target=(root / Path(*p.parts)).resolve()
        if root not in target.parents: raise ValueError('Path escaped output directory')
        raw=body.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError(f'Content checksum mismatch: {rel}')
        checked.append((target,raw))
    root.mkdir(parents=True,exist_ok=True)
    for target,raw in checked:
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f: f.write(raw)
    return len(checked)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('master',type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    print(f'Extracted {extract(a.master,a.out)} files. No code was executed.')

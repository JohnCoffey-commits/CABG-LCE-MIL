"""Read only named provenance files and directory metadata; never open images.

Run with the approved SSH ControlMaster, passing this file to python3 stdin.
The JSON stdout is captured locally. This does not send keys to tmux.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/Medic-AD')
DATA = Path('/home/data/medic-ad')
TREE = ROOT / 'worktrees/stage2aa-late-only-timing'
AA = ROOT / 'outputs/cabg-mil-v1.2/late-only-timing-v1'
CONTROL = AA.with_name('late-only-timing-control-v1')
FILES = {
    'pool': DATA / 'cabg-mil-v1.1-gate-d2-final-v2/training-development-manifest.jsonl',
    'pool_audit': DATA / 'cabg-mil-v1.1-gate-d2-final-v2/dataset-audit.json',
    'd3r': DATA / 'cabg-lce-mil-v1.2-d3r-v1/manifest.jsonl',
    'd3r_exclusion': DATA / 'cabg-lce-mil-v1.2-d3r-v1/exclusion-audit.json',
    'pilot': DATA / 'cabg-lce-mil-v1.2-d4-pilot-v2/manifest.jsonl',
    'train': DATA / 'cabg-lce-mil-v1.2-d4-scout-v1/train-manifest.jsonl',
    'development': DATA / 'cabg-lce-mil-v1.2-d4-scout-v1/eval-manifest.jsonl',
    'split': DATA / 'cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json',
    'aa_verification': AA / 'verification.json',
    'aa_inventory': CONTROL / 'artifact-inventory-v1.json',
    'aa_source': AA / 'source.json',
}
opened = []


def guard(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes)):
        return
    p = os.fsdecode(args[0])
    if p.startswith(str(DATA)):
        if Path(p) not in FILES.values():
            raise PermissionError('Unregistered data-body open: ' + p)
        opened.append(p)


def command(*args):
    p = subprocess.run(args, capture_output=True, text=True, check=True)
    return p.stdout.strip()


def main():
    sys.addaudithook(guard)
    result = {'epoch': time.time(), 'files': {}, 'images_opened': 0}
    for name, path in FILES.items():
        raw = path.read_bytes()
        result['files'][name] = {
            'path': str(path), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'text': raw.decode(),
        }
    sources = json.loads(result['files']['aa_source']['text'])
    result['source_checks'] = {
        n: hashlib.sha256((TREE / n).read_bytes()).hexdigest() == h
        for n, h in sources.items()
    }
    result['head'] = command('git', '-C', str(TREE), 'rev-parse', 'HEAD')
    result['git_status'] = command('git', '-C', str(TREE), 'status', '--short')
    result['panes'] = command('tmux', 'list-panes', '-s', '-t', 'medic-ad', '-F',
                             '#{pane_id}|#{pane_pid}|#{pane_tty}|#{pane_current_command}|#{pane_current_path}')
    result['capture'] = command('tmux', 'capture-pane', '-p', '-t', 'medic-ad', '-S', '-50')
    result['processes'] = command('ps', '-eo', 'pid,ppid,stat,etime,comm')
    result['gpu'] = command('nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu,driver_version', '--format=csv,noheader')
    result['space'] = command('df', '-B1', '/home')
    result['data_root_entries'] = sorted(p.name for p in DATA.iterdir())
    official = DATA / 'official/med_anomaly'
    result['official_datasets'] = sorted(p.name for p in official.iterdir())
    # Br35H is the other mounted MRI dataset. Directory entries and file sizes
    # alone are collected: no hashing, decoding, or opening image bodies.
    result['br35h_entries'] = []
    for folder, dirs, files in os.walk(official / 'br35h', followlinks=False):
        for name in sorted(files):
            path = Path(folder) / name
            result['br35h_entries'].append({
                'path': str(path.relative_to(official / 'br35h')),
                'bytes': path.lstat().st_size,
                'symlink': path.is_symlink(),
            })
    result['data_body_open_log'] = opened
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()

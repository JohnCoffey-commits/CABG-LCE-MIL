"""Immutable full24 execution contracts, buffered evidence IO, cumulative limits."""
import fcntl
import gzip
import io
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
import torch
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2m.constants import SOURCE_FILES
from reproduction.stage2p import TRAIN_SHA256, EVAL_SHA256, CONFIG_SHA256
from reproduction.stage2q.run import save_raw as _save_raw
from reproduction.stage2s.common import write

BOUNDARIES = (4, 8, 12, 16, 20, 23, 24)
LIMITS = {'update': 60, 'block': 64, 'eval': 192}
WALL_SECONDS = 14400
FLOOR = 8_000_000_000

def disk_guard(path):
    if shutil.disk_usage(path).free < FLOOR:
        raise RuntimeError('Full24 8GB free-space floor reached')

def save_raw(path, value):
    disk_guard(Path(path).parent)
    return _save_raw(path, value)

def load_raw(path):
    # Read the gzip stream once: PyTorch ZIP seeking on gzip is quadratic IO.
    with gzip.open(path, 'rb') as f:
        return torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)

class Budget:
    def __init__(self, root):
        self.path = Path(root)/'budget.json'
        if not self.path.exists():
            write(self.path, {'started_epoch': time.time(), 'limits': LIMITS,
                             'counts': dict.fromkeys(LIMITS, 0), 'events': []})
        state = json.loads(self.path.read_text())
        remaining = WALL_SECONDS - (time.time()-state['started_epoch'])
        if remaining <= 0: raise RuntimeError('Full24 time budget exhausted')
        def expired(*_): raise TimeoutError('Full24 cumulative 4-hour ceiling')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(max(1, int(remaining)))

    def reserve(self, kind, context):
        if kind not in LIMITS: raise ValueError(kind)
        disk_guard(self.path.parent)
        with self.path.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            s = json.load(f)
            if s['counts'][kind] >= LIMITS[kind] or time.time()-s['started_epoch'] >= WALL_SECONDS:
                raise RuntimeError('Full24 hard budget would be exceeded')
            s['counts'][kind] += 1
            s['events'].append({'kind': kind, 'context': context, 'epoch': time.time()})
            f.seek(0); json.dump(s, f, indent=2); f.truncate(); f.flush(); os.fsync(f.fileno())


def source_identity(commit):
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if head != commit or subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=repo, text=True):
        raise RuntimeError('Execution commit/clean tracked source contract')
    paths = set(SOURCE_FILES) | {str(p.relative_to(repo)) for d in ('stage2q','stage2s','stage2t')
                               for p in (repo/'reproduction'/d).rglob('*.py')}
    paths |= {str(p.relative_to(repo)) for p in (repo/'reproduction/stage2t').glob('*.sh')}
    if any(subprocess.check_output(['git','ls-files','--error-unmatch',p],cwd=repo,stderr=subprocess.DEVNULL).strip()==b'' for p in paths):
        raise RuntimeError('Execution source must be committed')
    return {p: sha256_file(repo/p) for p in sorted(paths)}


def provenance(commit, source, run_id, campaign):
    pre = json.loads((Path(campaign)/'preflight.json').read_text())
    if pre['code_commit'] != commit or pre['source_sha256'] != canonical_json_sha256(source):
        raise RuntimeError('Preflight/source closure')
    return {'head': commit, 'source_sha256': canonical_json_sha256(source), 'run_id': run_id,
            'train_manifest_sha256': TRAIN_SHA256, 'eval_manifest_sha256': EVAL_SHA256,
            'configuration_sha256': CONFIG_SHA256, 'preregistration_sha256': pre['preregistration_sha256'],
            'setting': 'stage2s_deterministic_bilinear_and_flash', 'schedule_horizon': 24}


def validate_range(mode, start, end, resume, run_id):
    if run_id not in ('R0', 'R1') or not 0 <= start < end <= 24:
        raise ValueError('Run/range contract')
    if bool(resume) != bool(start): raise ValueError('Nonzero start requires own checkpoint')
    if mode == 'replay' and (start,end) not in ((12,13),(23,24)):
        raise ValueError('Only preregistered no-update replays')
    if mode not in ('train','replay'): raise ValueError(mode)

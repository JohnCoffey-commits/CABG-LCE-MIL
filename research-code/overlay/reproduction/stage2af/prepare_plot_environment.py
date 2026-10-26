"""Create an isolated CPU plotting environment without changing model packages."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from reproduction.stage2af.budget import Budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root
    assert platform.system() == 'Linux'
    budget = Budget(root)
    budget.check(space=1_000_000_000)
    venv = root / 'environment/plot-venv-v1'
    wheels = root / 'environment/plot-wheels-v1'
    assert not venv.exists() and not wheels.exists()
    wheels.mkdir()
    # Charge a conservative dependency-download allowance once, without reducing
    # previous accounting. Exact retained wheel sizes/hashes are recorded below;
    # the allowance also covers index and dependency-resolution metadata.
    allowance = 200_000_000
    with budget.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert budget.snapshot()['download_bytes'] + allowance <= 50_000_000_000
        entry = dict(epoch=time.time(), source='isolated CPU plot dependencies',
                     bytes=allowance, accounting='conservative charged allowance; measured wheel payload retained separately',
                     index='https://pypi.org/simple', package='matplotlib==3.10.5')
        with (root / 'control/downloads.jsonl').open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(root / 'environment/tmp'),
               MPLCONFIGDIR=str(root / 'environment/plot-mpl-cache-v1'))
    env.pop('PYTHONPATH', None)
    subprocess.run(['/home/envs/medic-ad-train/bin/python', '-B', '-m', 'venv', str(venv)], env=env, check=True)
    python = str(venv / 'bin/python')
    pip = [python, '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check',
           '--retries', '0', '--timeout', '30']
    subprocess.run(pip + ['download', '--no-cache-dir', '--only-binary=:all:',
                         '--index-url', 'https://pypi.org/simple', '--dest', str(wheels),
                         'matplotlib==3.10.5'], env=env, check=True)
    identities = [{'name': p.name, 'bytes': p.stat().st_size,
                   'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(wheels.glob('*.whl'))]
    payload = sum(row['bytes'] for row in identities)
    assert payload < allowance, 'Payload exceeds conservative download accounting'
    subprocess.run(pip + ['install', '--no-index', '--find-links', str(wheels),
                         '--report', str(root / 'environment/plot-install-report-v1.json'),
                         'matplotlib==3.10.5'], env=env, check=True)
    result = subprocess.check_output([python, '-B', '-c',
        "import sys,json,matplotlib,numpy,PIL; matplotlib.use('Agg'); import matplotlib.pyplot; "
        "print(json.dumps({'python':sys.version,'matplotlib':matplotlib.__version__,'numpy':numpy.__version__,'Pillow':PIL.__version__}))"],
        env=env, text=True)
    report = dict(status='PASS', platform=platform.platform(), python=python,
                  versions=json.loads(result), wheels=identities,
                  measured_wheel_payload_bytes=payload,
                  download_budget_charged_bytes=allowance,
                  scope='CPU plot rendering only; model/evaluation environment not modified',
                  initial_plot_preflight='Local bundled and default Python, and model environment, lacked matplotlib; no model execution failed.')
    with (root / 'environment/plot-environment-v1.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'wheels'}), flush=True)


if __name__ == '__main__':
    main()

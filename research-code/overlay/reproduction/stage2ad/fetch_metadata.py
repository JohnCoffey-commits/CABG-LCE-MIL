"""Explicit-URL text metadata capture. No crawling, assets, credentials or images."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit


def capture(url, output):
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix + '.receipt.json').exists():
        raise FileExistsError(output)
    u = urlsplit(url)
    if u.scheme != 'https' or u.username or u.password or any(
        u.path.lower().endswith(x) for x in ('.gz', '.zip', '.jpg', '.jpeg', '.png', '.nii', '.h5', '.pdf')
    ):
        raise ValueError('Only explicitly selected HTTPS text/metadata endpoints')
    start = time.time()
    # Verified system TLS; curl does not load linked assets. Retain failed bodies.
    r = subprocess.run(['curl', '--silent', '--show-error', '--location', '--fail-with-body',
                        '--proto', '=https', '--proto-redir', '=https', '--max-time', '45',
                        '--max-filesize', '6000000', url], capture_output=True)
    with output.open('xb') as f:
        f.write(r.stdout)
    receipt = dict(url=url, started_epoch=start, ended_epoch=time.time(),
                   returncode=r.returncode, stderr=r.stderr.decode(errors='replace'),
                   bytes=len(r.stdout), sha256=hashlib.sha256(r.stdout).hexdigest(),
                   assets_requested=False)
    with output.with_suffix(output.suffix + '.receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('url'); p.add_argument('output', type=Path)
    a = p.parse_args(); print(json.dumps(capture(a.url, a.output)))

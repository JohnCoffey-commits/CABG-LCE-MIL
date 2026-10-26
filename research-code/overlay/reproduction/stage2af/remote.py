"""Use only the user-established SSH multiplex connection; never fall back."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

SSH = ["ssh", "-F", "/dev/null", "-S",
       "/path/to/your/existing-ssh-control.sock",
       "-o", "ControlMaster=no", "-o", "BatchMode=yes",
       "-o", "ProxyCommand=/usr/bin/false", "-o", "ConnectTimeout=8",
       "-T", "USER@YOUR_EXECUTION_HOST"]

def run_script(script: str, timeout: float = 45) -> subprocess.CompletedProcess:
    return subprocess.run(SSH + ["python3 -B -"], input=script, text=True,
                          capture_output=True, timeout=timeout)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("script", help="Local Python source, or - for stdin")
    p.add_argument("--timeout", type=float, default=45)
    args = p.parse_args()
    script = sys.stdin.read() if args.script == "-" else Path(args.script).read_text()
    result = run_script(script, args.timeout)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()

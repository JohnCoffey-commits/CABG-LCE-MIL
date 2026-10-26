"""Close the cumulative stage ledger after all owned research jobs terminate."""
import argparse
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def same_process_live(process):
    pid = process.get('child_pid', process.get('pid'))
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return False
    data = path.read_text().rsplit(')', 1)[1].split()
    # Fields after comm begin at field3; starttime is field22.
    return data[19] == str(process['proc_start_ticks']) and data[0] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    now = time.time()
    base = read(root / 'resource-ledger.json')
    events = records(root / 'control/events.jsonl')
    starts = {event['id']: event for event in events if event['kind'] == 'gpu_start'}
    ends = {event['id']: event for event in events if event['kind'] == 'gpu_end'}
    assert len(starts) == sum(e['kind'] == 'gpu_start' for e in events)
    assert len(ends) == sum(e['kind'] == 'gpu_end' for e in events)
    assert set(starts) == set(ends), ('unclosed GPU accounting', set(starts) ^ set(ends))
    jobs = []
    intervals = []
    for folder in sorted((root / 'control').iterdir()):
        process_file = folder / 'process.json'
        if not process_file.is_file():
            continue
        process = read(process_file)
        assert not same_process_live(process), ('owned job still running', folder.name)
        assert (folder / 'exit.json').is_file(), ('missing terminal evidence', folder.name)
        end = read(folder / 'exit.json')
        name = process['name']
        if process.get('gpu'):
            start_event, end_event = starts[name], ends[name]
            assert process['child_pid'] == start_event['pid'] == end_event['pid']
            assert process['proc_start_ticks'] == start_event['proc_start_ticks']
            assert end['exit_code'] == end_event['exit_code']
            assert abs(end['wall_seconds'] - (end_event['epoch'] - start_event['epoch'])) < 2
            intervals.append((start_event['epoch'], end_event['epoch']))
        monitor = folder / 'gpu-monitor.jsonl'
        observed = records(monitor) if monitor.is_file() else []
        sampled_peak = max((r['memory_MiB'] for r in observed), default=0)
        if 'peak_memory_MiB' in end:
            assert sampled_peak == end['peak_memory_MiB']
        jobs.append(dict(name=name, gpu=bool(process.get('gpu')), start_epoch=process['start_epoch'],
                         end_epoch=end['end_epoch'], wall_seconds=end['wall_seconds'],
                         exit_code=end['exit_code'], sampled_peak_memory_MiB=sampled_peak))
    intervals.sort()
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), 'GPU jobs overlap'
    reserved = [e for e in events if e['kind'] == 'update_reserved']
    completed = [e for e in events if e['kind'] == 'update_completed']
    def key(e):
        return e.get('trajectory'), e.get('attempt'), e['step'], e['category']
    reserved_counts = Counter(key(e) for e in reserved)
    completed_counts = Counter(key(e) for e in completed)
    assert not (completed_counts - reserved_counts), 'Completed updates without reservation'
    formal_counts = Counter(e.get('trajectory') for e in reserved if e['category'] == 'formal')
    assert len(reserved) <= 6100 and all(v <= 1000 for v in formal_counts.values())
    extra = sum(e['category'] != 'formal' for e in reserved)
    assert extra <= 100
    gpu_seconds = sum(end - start for start, end in intervals)
    assert gpu_seconds <= 216000
    download_bytes = sum(r.get('bytes', 0) for r in records(root / 'control/downloads.jsonl'))
    assert download_bytes <= 50_000_000_000
    plot_environment = root / 'environment/plot-environment-v1.json'
    plot_download = read(plot_environment) if plot_environment.is_file() else None
    disk = shutil.disk_usage(root)
    assert disk.free >= 50_000_000_000
    # Stat this stage only; do not hash checkpoint bodies again or traverse
    # symlinked historical dependencies and data outside this new output root.
    storage = defaultdict(lambda: dict(files=0, bytes=0))
    seen = set()
    for directory, folders, files in os.walk(root, followlinks=False):
        for filename in files:
            path = Path(directory) / filename
            if path.is_symlink():
                continue
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            group = path.relative_to(root).parts[0]
            storage[group]['files'] += 1
            storage[group]['bytes'] += stat.st_size
    elapsed = now - base['start_epoch']
    hours = math.ceil(elapsed / 60) / 60
    cost = hours * base['pricing']['estimated_total_usd_per_hour']
    assert cost <= 50
    phases = defaultdict(lambda: dict(jobs=0, process_seconds=0.0))
    for job in jobs:
        if not job['gpu']:
            continue
        name = job['name']
        phase = 'formal_training' if name.startswith('formal-') else 'evaluation' if name.startswith('eval-') else 'linear_head' if name.startswith('linear-') else 'preflight_calibration_debug'
        phases[phase]['jobs'] += 1
        phases[phase]['process_seconds'] += job['wall_seconds']
    result = dict(status='PASS', observation_epoch=now, ledger_start_epoch=base['start_epoch'],
                  instance_seconds=elapsed, billed_hours_rounded_up_to_minute=hours,
                  estimated_cost_usd=cost, pricing=base['pricing'], gpu_process_seconds=gpu_seconds,
                  gpu_jobs=len(starts), phases=dict(phases), updates_reserved=len(reserved),
                  updates_completed=len(completed), extra_updates_reserved=extra,
                  reservations_without_completed_event=sum((reserved_counts-completed_counts).values()),
                  formal_updates_reserved_by_trajectory=dict(formal_counts), download_bytes=download_bytes,
                  download_accounting='Cumulative budget charge; includes conservative allowances where explicitly marked in downloads.jsonl, not an exact network-traffic counter.',
                  plot_dependency_download=None if plot_download is None else {
                      'charged_bytes': plot_download['download_budget_charged_bytes'],
                      'measured_retained_wheel_payload_bytes': plot_download['measured_wheel_payload_bytes'],
                      'environment': str(plot_environment)},
                  sampled_peak_memory_MiB=max((j['sampled_peak_memory_MiB'] for j in jobs), default=0),
                  sampling_caveat='nvidia-smi samples every15seconds; observed maximum, not exact allocation peak',
                  disk=dict(total_bytes=disk.total, used_bytes=disk.used, free_bytes=disk.free),
                  stage_storage_by_top_level=dict(storage), stage_file_bytes=sum(v['bytes'] for v in storage.values()),
                  all_recorded_owned_processes_terminated=True, jobs=jobs,
                  billing_caveat='Public-rate estimate includes preparation, idle and retries from goal creation. Not an invoice. Instance remains on; later retention and running charges continue.')
    with (root / 'resource-closeout-v1.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('jobs', 'pricing')}), flush=True)


if __name__ == '__main__':
    main()

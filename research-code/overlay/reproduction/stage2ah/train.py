"""Frozen four-track study, with prospectively reusable 24-step trial prefixes."""
import argparse
import gc
import hashlib
import json
import os
import platform
import signal
import time
import traceback
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    for arg in ('root', 'old-root'):
        p.add_argument('--' + arg, type=Path, required=True)
    p.add_argument('--seed', type=int, choices=[43, 44], required=True)
    p.add_argument('--arm', choices=['aligned', 'shuffled'], required=True)
    p.add_argument('--stop', type=int, choices=[24, 1000], default=1000)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    assert platform.system() == 'Linux' and Path('/home/Medic-AD').is_dir(), 'SERVER ONLY'
    import torch
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    from reproduction.stage2af.runtime import Runtime, write, sha_file, save_gradients
    from reproduction.stage2af.core import compose, apply_update
    from reproduction.stage2af.guard import InputGuard
    from reproduction.stage2af.inputs import read_rows, load_prepared
    from reproduction.stage2af.execution_amendment import load_execution_amendment, expected_frozen_parameters
    from reproduction.stage2s.common import state_digest, restore_runtime
    from reproduction.stage2i.rng_state import snapshot_rng_state, restore_rng_state
    from reproduction.stage2ah.transport import Transport
    from reproduction.stage2ah.budget import Budget
    plan = json.loads((a.root/'protocol.json').read_text())
    protocol = sha_file(a.root/'protocol.json')
    assert json.loads((a.root/'protocol-seal.json').read_text())['sha256'] == protocol
    assert sha_file(a.root/'plan/protocol.md') == plan['protocol_document_sha256']
    baseline, oldprotocol, amendment, amendment_hash = load_execution_amendment(a.old_root)
    assert oldprotocol == plan['baseline_protocol_sha256']
    trajectory = f's{a.seed}-{a.arm}'
    out = a.root/'formal'/trajectory/'attempt-v1'
    if not a.resume:
        out.mkdir(parents=True, exist_ok=False)
    else:
        assert a.resume == out/'checkpoint-step0024.pt' and a.stop == 1000
        assert (out/'prefix-summary.json').is_file() and not (out/'summary.json').exists()
    budget = Budget(a.root)
    budget.check(storage=True)
    rows = [x for x in read_rows(a.old_root/'data/prepared-v1/manifest.jsonl') if x['split'] == 'train']
    guard = InputGuard(a.old_root, rows).install()
    coefficient = plan['fixed_coefficients'][str(a.seed)]
    r = hook = None
    last = 0
    started = time.time()
    def stop(*args):
        raise RuntimeError('Supervisor requested stop')
    signal.signal(signal.SIGTERM, stop)
    try:
        r = Runtime(a.old_root, a.seed)
        assert r.initial_identity == baseline['initial_identities'][str(a.seed)]
        params = {x['name']: x for x in r.initial_full['parameters']}
        for name, expected in expected_frozen_parameters(baseline, amendment, a.seed).items():
            assert params[name] == expected, name
        assert r.initial_full['buffers'] == baseline['initial_buffers']
        if not a.resume:
            write(out/'initial-state.json', r.initial_full)
            write(out/'sequence.json', r.train_sequence)
            # Risk test: replacement disabled must be bitwise identical, including
            # all language logits. Restore RNG/runtime so trial initialization is unchanged.
            rng = snapshot_rng_state()
            inputs = load_prepared(rows[0], r.wrapper.device)
            with torch.no_grad():
                native = r.model(**inputs, tune_mode='default', return_anomaly_evidence=True,
                                 use_cache=False, diff_mode=False).logits.detach().cpu()
            restore_runtime(r.model, r.clean_runtime)
            restore_rng_state(rng)
            hook = Transport(r.model, a.arm).install()
            hook.enabled = False
            with torch.no_grad():
                disabled = r.model(**inputs, tune_mode='default', return_anomaly_evidence=True,
                                   use_cache=False, diff_mode=False).logits.detach().cpu()
            assert torch.equal(native, disabled), 'Disabled replacement changed logits'
            hook.enabled = True
            r.restore_initial()
            write(out/'engineering-start.json', dict(disabled_logits_exact=True,
                  baseline_initial_exact=True, additional_parameters=0, device=torch.cuda.get_device_name(0),
                  pid=os.getpid(), host=platform.node(), protocol_sha256=protocol,
                  baseline_amendment_sha256=amendment_hash))
            del inputs, native, disabled
        else:
            hook = Transport(r.model, a.arm).install()
            last = r.load(a.resume, protocol, a.arm, coefficient)
            assert last == 24
            write(out/'resume.json', dict(checkpoint_sha256=sha_file(a.resume), step=last, epoch=time.time()))
        for step in range(last + 1, a.stop + 1):
            t = time.time()
            budget.check()
            lm, lce, images, inputs, rng = r.collect(step)
            assert hook.calls > 0 and hook.nonzero_gradient_events > 0, 'Transport LM gradient missing'
            applied, details = compose(lm, lce, [x['label'] for x in images], 'fixed_lce', r.controller, coefficient)
            raw = save_gradients(out/f'gradients-step{step:04}.pt.gz', lm, lce, images) if step in plan['sparse_gradient_steps'] else None
            budget.event('update_reserved', trajectory=trajectory, step=step, category='formal')
            lr = apply_update(r.masters, r.optimizer, r.scheduler, applied, r.model)
            budget.event('update_completed', trajectory=trajectory, step=step, category='formal')
            last = step
            rec = dict(trajectory=trajectory, step=step, category='formal', images=images, rng=rng, lr=lr,
                       lm_rms=details['lm_rms'], lce_rms=details['lce_rms'], controller=details['budget'],
                       applied_coefficient=coefficient, applied_dynamic_cap=False, gradient=details['applied_gradient'],
                       raw=raw, applied_digest=state_digest(applied) if raw else None,
                       transport=hook.report(), seconds_before_checkpoint=time.time()-t)
            write(out/f'step{step:04}.json', rec)
            if step in plan['checkpoint_steps']:
                r.save(out/f'checkpoint-step{step:04}.pt', step, a.arm, coefficient, protocol)
            print(json.dumps(dict(trajectory=trajectory, step=step,
                     lm_mean=sum(x['lm_loss'] for x in images)/3, seconds=time.time()-t)), flush=True)
            del lm, lce, images, inputs, applied, details
            gc.collect()
        if a.stop == 24:
            # A no-update next-block replay tests the actual saved adapter, masters,
            # optimizer/scheduler, runtime and RNG against uninterrupted execution.
            before_masters = state_digest(r.masters)
            before_scheduler = r.scheduler.state_dict()
            lm1, aux1, images1, inputs1, rng1 = r.collect(25)
            fingerprints = ([state_digest(x) for x in lm1], [state_digest(x) for x in aux1], images1, rng1)
            del lm1, aux1, inputs1
            assert r.load(out/'checkpoint-step0024.pt', protocol, a.arm, coefficient) == 24
            assert before_masters == state_digest(r.masters) and before_scheduler == r.scheduler.state_dict()
            lm2, aux2, images2, inputs2, rng2 = r.collect(25)
            assert fingerprints == ([state_digest(x) for x in lm2], [state_digest(x) for x in aux2], images2, rng2)
            del lm2, aux2, inputs2
            r.load(out/'checkpoint-step0024.pt', protocol, a.arm, coefficient)
            write(out/'resume-test.json', dict(status='PASS', next_block_lm_lce_and_rng_exact=True,
                 masters_scheduler_exact=True, optimizer_updates_in_test=0, checkpoint_sha256=sha_file(out/'checkpoint-step0024.pt')))
        frozen = r.verify_frozen()
        assert not guard.violations
        suffix = 'prefix' if a.stop == 24 else 'final'
        write(out/f'file-open-audit-{suffix}.json', guard.record())
        write(out/('prefix-summary.json' if a.stop == 24 else 'summary.json'), dict(
              status='PREFIX_COMPLETE' if a.stop == 24 else 'COMPLETE', trajectory=trajectory, last_step=last,
              wall_seconds=time.time()-started, protocol_sha256=protocol, frozen=frozen,
              transport=hook.report(), resources=budget.snapshot()))
    except BaseException as exc:
        path = out/f'failure-{int(time.time())}.json'
        write(path, dict(status='FAILED', last_step=last, error=repr(exc), traceback=traceback.format_exc(), resources=budget.snapshot()))
        raise
    finally:
        if hook: hook.close()
        if r: r.close()


if __name__ == '__main__':
    main()

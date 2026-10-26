"""Frozen generated-answer endpoints; no parameter updates or forced answers."""
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'old-root'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--seed', type=int, choices=[43, 44], required=True)
    parser.add_argument('--arm', choices=['aligned', 'shuffled'], required=True)
    parser.add_argument('--split', choices=['development', 'final'], required=True)
    a = parser.parse_args()
    assert platform.system() == 'Linux' and Path('/home/Medic-AD').is_dir()
    import numpy as np
    import torch
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0)
    from reproduction.stage2af.runtime import Runtime, write, sha_file
    from reproduction.stage2af.inputs import read_rows, load_prepared, DECODE
    from reproduction.stage2af.evaluate import load_adapter
    from reproduction.stage2af.metrics import score_maps, parse_answer, summary
    from reproduction.stage2af.guard import InputGuard
    from reproduction.stage2s.common import restore_runtime
    from reproduction.stage2q.run import full_state
    from reproduction.stage2ah.transport import Transport
    from reproduction.stage2ah.budget import Budget
    protocol = sha_file(a.root/'protocol.json')
    plan = json.loads((a.root/'protocol.json').read_text())
    assert protocol == json.loads((a.root/'protocol-seal.json').read_text())['sha256']
    if a.split == 'final':
        release = json.loads((a.root/'final-evaluation-release.json').read_text())
        assert release['protocol_sha256'] == protocol
        assert len(release['complete_trajectories']) == 4
    rows = [x for x in read_rows(a.old_root/'data/prepared-v1/manifest.jsonl') if x['split'] == a.split]
    assert len(rows) == (128 if a.split == 'development' else 542)
    guard = InputGuard(a.old_root, rows).install()
    budget = Budget(a.root)
    budget.check(storage=True)
    r = Runtime(a.old_root, a.seed, training=False)
    hook = Transport(r.model, a.arm).install()
    def post_ffn(module, inputs, output):
        hook.record('post_ffn', output)
    def visual(module, inputs, output):
        if hook.capture:
            combined = output[2]
            assert combined.shape == (80, 3584)
            hook.record('direct', combined[:-16])
            hook.record('fed_ano', combined[-16:])
    handles = [hook.module.post_ffn.register_forward_hook(post_ffn),
               r.model.visual.register_forward_hook(visual)]
    try:
        nodes = plan['development_steps'] if a.split == 'development' else [1000]
        for step in nodes:
            checkpoint = a.root/'formal'/f's{a.seed}-{a.arm}'/'attempt-v1'/f'checkpoint-step{step:04}.pt'
            meta = load_adapter(r, checkpoint, protocol, a.arm, plan['fixed_coefficients'][str(a.seed)])
            assert meta['step'] == step
            conditions = ['trained']
            if a.split == 'development' and a.arm == 'aligned' and step == 1000:
                conditions.append('inference_shuffle')
            for condition in conditions:
                hook.arm = 'shuffled' if condition == 'inference_shuffle' else a.arm
                name = f's{a.seed}-{a.arm}-step{step:04}-{a.split}'
                if condition != 'trained': name += '-'+condition
                out = a.root/'evaluation'/name/'v1'
                out.mkdir(parents=True, exist_ok=False)
                write(out/'checkpoint.json', meta)
                restore_runtime(r.model, r.clean_runtime)
                before = full_state(r.model)
                write(out/'parameters-before.json', before)
                started = time.time()
                records = []
                for i, row in enumerate(rows):
                    budget.check()
                    rec = {k: row[k] for k in ['sample_id','target','sha256','group_id','split',
                           'prepared_sha256','pixel_sha256_prepared','image_grid_thw','input_ids']}
                    inputs = load_prepared(row, r.wrapper.device)
                    assert 'labels' not in inputs and 'anomaly_labels' not in inputs
                    restore_runtime(r.model, r.clean_runtime)
                    hook.capture = True
                    hook.trace = {}
                    t = time.time()
                    with torch.inference_mode():
                        result = r.model(**inputs, tune_mode='default', return_anomaly_evidence=True,
                                         use_cache=False, diff_mode=False)
                    trace = hook.trace
                    scores = score_maps(torch.from_numpy(trace['abnormal']), torch.from_numpy(trace['normal']))
                    yes = r.contract['answer_suffix_tokens']['Yes'][0]
                    no = r.contract['answer_suffix_tokens']['No'][0]
                    logits = result.logits[0, -1, [yes, no]].float().cpu().numpy()
                    weight = trace['weight16'].ravel()
                    rec.update(**scores, yes_no_logit_gap=float(logits[0]-logits[1]),
                               transport_arm=hook.arm, condition=condition,
                               transported_mass=float(weight.sum()),
                               transported_support=float(np.exp(-np.sum(weight*np.log(np.maximum(weight,1e-30))))))
                    rec.update(fed_ano_sha256=hashlib.sha256(trace['fed_ano'].tobytes()).hexdigest(),
                               fed_direct_sha256=hashlib.sha256(trace['direct'].tobytes()).hexdigest())
                    raw = dict(abnormal=trace['abnormal'], normal=trace['normal'], yes_no_logits=logits,
                               weight16=trace['weight16'], weight32=trace['weight32'])
                    if i < 8:
                        raw.update({key: trace[key] for key in ['values','native','pooled','output','attention_logits',
                                                               'post_ffn','direct','fed_ano']})
                    path = out/f'raw-{i:04}.npz'
                    with path.open('xb') as f: np.savez_compressed(f, **raw)
                    rec.update(raw_file=path.name, raw_sha256=sha_file(path), score_seconds=time.time()-t,
                               full_transport_trace=i<8)
                    del result, raw, trace
                    hook.capture = False
                    hook.trace = {}
                    restore_runtime(r.model, r.clean_runtime)
                    t = time.time()
                    with torch.inference_mode(): generated = r.model.generate(**inputs, **DECODE)
                    tokens = generated[0, inputs['input_ids'].shape[1]:].cpu().tolist()
                    text = r.wrapper.processor.tokenizer.decode(tokens, skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False)
                    rec.update(output_ids=tokens, text=text, parsed=parse_answer(text), generation_seconds=time.time()-t)
                    write(out/f'record-{i:04}.json', rec)
                    records.append(rec)
                    del generated, inputs
                    gc.collect()
                    if (i+1)%64 == 0:
                        print(json.dumps(dict(endpoint=name, completed=i+1, total=len(rows))), flush=True)
                restore_runtime(r.model, r.clean_runtime)
                after = full_state(r.model)
                write(out/'parameters-after.json', after)
                assert before == after and all(p.grad is None for p in r.model.parameters())
                assert not guard.violations
                write(out/'file-open-audit.json', guard.record())
                write(out/'observations.json', records)
                write(out/'summary.json', dict(status='PASS', endpoint=name, metrics=summary(records),
                      protocol_sha256=protocol, checkpoint_sha256=meta['sha256'],
                      model_wall_seconds=time.time()-started, parameter_and_buffer_identity_preserved=True,
                      transport=hook.report()))
                print(json.dumps(dict(endpoint=name, metrics=summary(records))), flush=True)
    finally:
        for handle in handles: handle.remove()
        hook.close()
        r.close()


if __name__ == '__main__': main()

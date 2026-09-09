"""Real-model training-only Route A runner; never imports an effect evaluator."""
import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import torch

from reproduction.stage2h.state_audit import trainable_state_record
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2i.rng_state import snapshot_rng_state, restore_rng_state, rng_state_fingerprint
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit, NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l.controller import DynamicController, compute_block_gradients, exact_s9_from_full
from reproduction.stage2m.constants import TRAINABLE_NAMES, S9_NAMES, SOURCE_FILES
from reproduction.stage2p import INITIAL_STATE_SHA256, INITIAL_RNG_SHA256, BASE_SHA256
from reproduction.stage2q.execution import AttentionExecution, cpu_tree
from reproduction.stage2q.run import canonicalize_dormant, full_state, data_contract, save_raw
from reproduction.stage2s.common import (BASELINE, Budget, write, runtime_state, restore_runtime,
                                         state_digest, setup_flags, first_step_prediction)
from reproduction.stage2s.resize import ResizeProbe


def collect(wrapper, dataset, collator, manifest, block, attention, probe):
    from reproduction.stage2h.runtime import move_batch_to_device
    model = wrapper.llm; named = dict(model.named_parameters())
    params = [named[n] for n in TRAINABLE_NAMES]
    lm, lce, images, inputs = [], [], [], []
    rng_before = rng_state_fingerprint(snapshot_rng_state())
    for index in range(block*3, block*3+3):
        row = manifest[index]
        batch = move_batch_to_device(collator([dataset[index]]), wrapper.device)
        inputs.append(cpu_tree(batch)); label = batch.pop("anomaly_labels")
        if int(label.item()) != int(row["scout_label"] == "abnormal"):
            raise RuntimeError("Label alignment changed")
        batch.update(tune_mode="default", return_anomaly_evidence=True)
        probe.capture = index == 0; attention.scope = probe.scope = "forward"
        with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
            result = model(**batch)
        maps = observer.finalize(result.anomaly_evidence)
        evidence = compute_lce(maps["abnormal_raw"], maps["normal_raw"], [row["scout_label"]])
        attention.scope = probe.scope = "lm"
        gl = torch.autograd.grad(result.loss, params, retain_graph=True, allow_unused=True)
        attention.scope = probe.scope = "lce"
        ga = torch.autograd.grad(evidence["per_image_loss"][0], params, allow_unused=True)
        if any(g is None or not torch.isfinite(g).all() for g in gl):
            raise RuntimeError("LM gradient support/numerics")
        lm.append({n: g.detach().float().cpu() for n, g in zip(TRAINABLE_NAMES, gl)})
        lce.append(exact_s9_from_full(dict(zip(TRAINABLE_NAMES, ga)), S9_NAMES))
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Unexpected model .grad")
        images.append({"sample_id": row["sample_id"], "sha256": row["sha256"], "label": row["scout_label"],
                       "exposure_id": row["scout_exposure_id"], "lm_loss": float(result.loss.detach()),
                       "lce_loss": float(evidence["per_image_loss"][0].detach()),
                       "lce_score": float(evidence["score"][0].detach()),
                       "effective_support": float(evidence["effective_support"][0].detach()),
                       "top11_mass": float(evidence["top11_mass"][0].detach())})
        del result, maps, evidence, gl, ga, batch
        gc.collect(); torch.cuda.synchronize()
    return lm, lce, images, inputs, {"before_block": rng_before, "after_block": rng_state_fingerprint(snapshot_rng_state())}


def run(args):
    if args.repeats not in (1, 2): raise RuntimeError("Repeat budget")
    args.output.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.campaign); setup_flags()
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != BASELINE or subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True):
        raise RuntimeError("Historical tracked source identity changed")
    source = {p: sha256_file(repo/p) for p in SOURCE_FILES}
    source.update({str(p.relative_to(repo)): sha256_file(p) for p in (repo/"reproduction/stage2s").glob("*.py")})
    write(args.output/"source.json", {"head": head, "files": source})
    split, manifest = data_contract(args.split_audit, 4)
    preflight = json.loads(args.preflight.read_text())
    if canonical_json_sha256(preflight["base_checkpoint"]) != BASE_SHA256:
        raise RuntimeError("Base checkpoint contract")
    attention = AttentionExecution("canonical_flash_deterministic").install()
    if args.setting not in ('native','deterministic_bilinear'): raise RuntimeError('Unknown setting')
    probe = ResizeProbe(deterministic=args.setting == 'deterministic_bilinear').install()
    import flash_attn
    write(args.output/'environment.json', {'torch':torch.__version__, 'cuda':torch.version.cuda,
          'flash_attn':flash_attn.__version__, 'global_deterministic':torch.are_deterministic_algorithms_enabled(),
          'cudnn_deterministic':torch.backends.cudnn.deterministic,'tf32':torch.backends.cuda.matmul.allow_tf32,
          'cublas':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'torch_num_threads':torch.get_num_threads(),
          'setting':args.setting,'precision':'BF16 model and gradients; FP32 masters/controller unchanged'})
    monitor = NvidiaSmiMonitor(args.output/"nvidia-smi.csv"); monitor.start()
    audit = MedicalImageOpenAudit(Path(split["image_root"]), [str((Path(split["image_root"])/r["relative_path"]).resolve()) for r in manifest[:3]])
    audit.install()
    try:
        os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = split["train_annotation"]
        os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = split["image_root"]
        from reproduction.stage2h.runtime import load_stage2h_runtime, make_stage2h_dataset
        wrapper, loaded = load_stage2h_runtime(args.base_model, seed=42, training=True, gradient_checkpointing=True)
        model = wrapper.llm
        if canonical_json_sha256(loaded["step0_state"]) != INITIAL_STATE_SHA256: raise RuntimeError("T21 initialization")
        write(args.output/"dormant.json", canonicalize_dormant(model))
        if canonical_json_sha256(trainable_state_record(model)) != INITIAL_STATE_SHA256:
            raise RuntimeError('Dormant canonicalization changed T21')
        full = full_state(model); write(args.output/"full-initial-state.json", full)
        initial = cpu_tree({n: dict(model.named_parameters())[n] for n in TRAINABLE_NAMES})
        initial_runtime = runtime_state(model)
        _, dataset, collator = make_stage2h_dataset(args.base_model, "medic_ad_stage2h_calibration", shuffle=False)
        if [(r.get("cabg_sample_id"), r.get("scout_exposure_id")) for r in dataset.list_data_dict] != [(r["sample_id"], r["scout_exposure_id"]) for r in manifest]:
            raise RuntimeError("Data order")
        initial_rng = snapshot_rng_state()
        if rng_state_fingerprint(initial_rng) != INITIAL_RNG_SHA256: raise RuntimeError("Initial RNG")
        save_raw(args.output/"start.pt.gz", {"trainable": initial, "runtime": initial_runtime, "rng": initial_rng})
        for repeat in range(args.repeats):
            restore_runtime(model, initial_runtime); restore_rng_state(initial_rng)
            budget.reserve("block", f"{args.output.name}:repeat{repeat}")
            started = time.monotonic()
            lm, lce, images, inputs, rng = collect(wrapper, dataset, collator, manifest, 0, attention, probe)
            controller = DynamicController()
            applied, details = compute_block_gradients(lm, lce, [r["label"] for r in images], trainable_names=TRAINABLE_NAMES, s9_names=S9_NAMES, controller=controller)
            raw = {"inputs": inputs, "lm": [{n: v.to(torch.bfloat16) for n,v in g.items()} for g in lm],
                   "lce": [{n: v.to(torch.bfloat16) for n,v in g.items()} for g in lce],
                   "aggregate_lm": details["aggregate_lm"], "aggregate_lce": details["aggregate_lce"],
                   "applied": applied, "updated_master": first_step_prediction(initial, applied)}
            if any(not torch.equal(g[n], raw[key][i][n].float()) for key,gs in (('lm',lm),('lce',lce)) for i,g in enumerate(gs) for n in g):
                raise RuntimeError('Gradient archival cast is not lossless')
            saved = save_raw(args.output/f"repeat{repeat}.pt.gz", raw)
            write(args.output/f"repeat{repeat}.json", {"images": images, "rng": rng, "controller": details["budget"],
                  "gradient": details["gradient"], "raw": saved, "seconds": time.monotonic()-started,
                  "update_kind": "analytic_first_step_prediction_no_optimizer_step", "real_updates": 0})
            del raw, lm, lce, applied, details; gc.collect(); torch.cuda.empty_cache()
        save_raw(args.output/"resize-captures.pt.gz", probe.captures)
        restore_runtime(model, initial_runtime)
        if full_state(model) != full: raise RuntimeError("No-step state mutation")
        write(args.output/"attention.json", attention.report())
        write(args.output/"resize-calls.json", probe.calls)
        result = audit.result()
        if result["status"] != "SUCCESS": raise RuntimeError("Image allowlist")
        write(args.output/"file-open-audit.json", result)
        write(args.output/"result.json", {"status": "SUCCESS", "setting": args.setting, "real_updates": 0,
                                          "complete_state_unchanged": True, "repeats": args.repeats})
    finally:
        probe.close(); attention.close()
        result = monitor.stop(); write(args.output/"monitor-summary.json", result)
        if result["peak_memory_used_mib"] > 22500: raise RuntimeError("VRAM ceiling")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--setting", default="native")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--base-model", type=Path, default=Path("/home/checkpoints/Lingshu-7B"))
    p.add_argument("--split-audit", type=Path, default=Path("/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json"))
    p.add_argument("--preflight", type=Path, default=Path("/home/Medic-AD/outputs/cabg-mil-v1.2/d4-scout-reproducibility-v1/preflight.json"))
    run(p.parse_args())

if __name__ == "__main__": main()

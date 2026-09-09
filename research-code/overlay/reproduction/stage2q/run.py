"""Small first-block and short-training producer using original Scout primitives."""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch

from reproduction.stage2h.state_audit import tensor_sha256, trainable_state_record
from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file, git_identity, implementation_source_record
from reproduction.stage2i.rng_state import snapshot_rng_state, restore_rng_state, rng_state_fingerprint
from reproduction.stage2i.run_gate_d3 import MedicalImageOpenAudit, NvidiaSmiMonitor
from reproduction.stage2k.evidence import RawAttentionObserver
from reproduction.stage2k.producer_math import compute_lce
from reproduction.stage2l.controller import DynamicController, compute_block_gradients, exact_s9_from_full
from reproduction.stage2l.checkpoint import save_atomic, load_strict, tensor_inventory
from reproduction.stage2m.constants import TRAINABLE_NAMES, S9_NAMES, SOURCE_FILES
from reproduction.stage2m.train_scout import _checkpoint_payload, _move_optimizer_state, cosine_multiplier, matched_configuration
from reproduction.stage2p import INITIAL_STATE_SHA256, INITIAL_RNG_SHA256, TRAIN_SHA256, CONFIG_SHA256, BASE_SHA256
from reproduction.stage2q.execution import AttentionExecution, cpu_tree

DORMANT_NAMES = tuple(f"segmentation_head.blocks.{i}.{suffix}" for i in range(4) for suffix in ("dwconv.bias","dwconv.weight","dwconv2.bias","dwconv2.weight","dwconv3.bias","dwconv3.weight","norm.bias","norm.weight")) + tuple("segmentation_head."+suffix for suffix in ("proj.bias","proj.weight","upsampler.0.norm.bias","upsampler.0.norm.weight","upsampler.0.upsample.bias","upsampler.1.norm.bias","upsampler.1.norm.weight","upsampler.1.upsample.bias"))
CACHE_ATTRIBUTES=("last_visual_tokens","last_visual_grid_thw","_last_segmentation_preview")
DORMANT_NAMES += ("segmentation_head.upsampler.0.upsample.weight", "segmentation_head.upsampler.1.upsample.weight")


def canonicalize_dormant(model):
    named=dict(model.named_parameters())
    if any(n not in named or named[n].requires_grad or n in TRAINABLE_NAMES for n in DORMANT_NAMES):
        raise RuntimeError("Dormant parameter contract changed")
    before=inventory({n:named[n] for n in DORMANT_NAMES})
    for n in DORMANT_NAMES:
        with torch.no_grad():named[n].zero_()
    def forbidden(*args):
        raise RuntimeError("Canonicalized dormant segmentation head was invoked")
    model.segmentation_head.register_forward_pre_hook(forbidden)
    return {"names":list(DORMANT_NAMES),"before":before,"after":inventory({n:named[n] for n in DORMANT_NAMES}),"guard":"raise_on_any_segmentation_head_forward","active_t21_unchanged":True}


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def save_raw(path, value):
    if Path(path).exists():
        raise FileExistsError(path)
    if shutil.disk_usage(Path(path).parent).free < 700 * 1024**2:
        raise RuntimeError("Diagnostic disk floor reached")
    with Path(path).open("xb") as f, gzip.GzipFile(fileobj=f, mode="wb", compresslevel=1, mtime=0) as z:
        torch.save(value, z)
    return {"file": str(path), "sha256": sha256_file(path)}


def load_raw(path):
    with gzip.open(path, "rb") as f:
        return torch.load(f, map_location="cpu", weights_only=False)


def inventory(mapping):
    return tensor_inventory(mapping)


def full_state(model):
    # Includes frozen newly initialized tensors and all registered buffers.
    return {"parameters": inventory(dict(model.named_parameters())), "buffers": inventory(dict(model.named_buffers()))}


def runtime_state(model):
    return {"buffers": cpu_tree(dict(model.named_buffers())), "rope_deltas": {name: cpu_tree(module.rope_deltas) for name,module in model.named_modules() if hasattr(module, "rope_deltas")},"visual_caches":{name:{attr:cpu_tree(getattr(module,attr)) for attr in CACHE_ATTRIBUTES if hasattr(module,attr)} for name,module in model.named_modules() if any(hasattr(module,attr) for attr in CACHE_ATTRIBUTES)}}


def restore_runtime(model, state):
    buffers = dict(model.named_buffers())
    if set(buffers) != set(state["buffers"]):
        raise RuntimeError("Buffer identity changed")
    for name,value in state["buffers"].items():
        buffers[name].copy_(value.to(buffers[name]))
    modules = dict(model.named_modules())
    for name,value in state["rope_deltas"].items():
        modules[name].rope_deltas = None if value is None else value.to(next(model.parameters()).device)
    for name,module in modules.items():
        for attr in CACHE_ATTRIBUTES:
            stored=state.get("visual_caches",{}).get(name,{})
            if attr in stored:
                value=stored[attr]
                setattr(module,attr,value.to(next(model.parameters()).device) if isinstance(value,torch.Tensor) else value)
            elif hasattr(module,attr):
                delattr(module,attr)


def data_contract(split_path, block_limit):
    split = json.loads(split_path.read_text())
    train = [json.loads(l) for l in Path(split["train_manifest"]).read_text().splitlines()]
    if sha256_file(Path(split["train_manifest"])) != TRAIN_SHA256 or len(train) != 72:
        raise RuntimeError("Original training manifest mismatch")
    if sha256_file(Path(split["train_annotation"])) != split["train_annotation_sha256"]:
        raise RuntimeError("Training annotation mismatch")
    # Held-out exclusion metadata only; never read its images or prediction files.
    excluded = [json.loads(l) for l in Path(split["eval_manifest"]).read_text().splitlines()]
    for key,path in (("d3r","/home/data/medic-ad/cabg-lce-mil-v1.2-d3r-v1/manifest.jsonl"),("d4_pilot","/home/data/medic-ad/cabg-lce-mil-v1.2-d4-pilot-v2/manifest.jsonl")):
        if sha256_file(Path(path)) != split["source_hashes"][key]:
            raise RuntimeError("Exclusion metadata changed")
        excluded.extend(json.loads(l) for l in Path(path).read_text().splitlines())
    for key in ("sample_id","sha256","relative_path"):
        if {r[key] for r in train} & {r[key] for r in excluded}:
            raise RuntimeError("Training boundary overlap")
    source = Path("/home/data/medic-ad/cabg-mil-v1.1-gate-d2-final-v2/training-development-manifest.jsonl")
    if sha256_file(source) != split["source_hashes"]["training_development"]:
        raise RuntimeError("Development source changed")
    members = {(r['sample_id'],r['sha256'],r['relative_path']) for r in (json.loads(l) for l in source.read_text().splitlines())}
    for r in train[:block_limit*3]:
        if (r['sample_id'],r['sha256'],r['relative_path']) not in members:
            raise RuntimeError("Not original development data")
        path = (Path(split["image_root"])/r["relative_path"]).resolve()
        if not path.is_relative_to(Path(split["image_root"]).resolve()) or sha256_file(path) != r["sha256"]:
            raise RuntimeError("Training image identity changed")
    return split, train


def collect(wrapper, dataset, collator, manifest, block, attention, raw_inputs=False):
    from reproduction.stage2h.runtime import move_batch_to_device
    model = wrapper.llm
    named = dict(model.named_parameters())
    params = [named[n] for n in TRAINABLE_NAMES]
    lm_images, lce_images, images, inputs = [], [], [], []
    before = snapshot_rng_state()
    for index in range(block*3, block*3+3):
        row = manifest[index]
        batch = move_batch_to_device(collator([dataset[index]]), wrapper.device)
        input_record = cpu_tree(batch)
        labels = batch.pop("anomaly_labels")
        if int(labels.item()) != int(row["scout_label"] == "abnormal"):
            raise RuntimeError("Label alignment")
        batch.update(tune_mode="default", return_anomaly_evidence=True)
        attention.scope = "forward"
        with RawAttentionObserver(model.visual.anomaly_qformer.anomaly_attention) as observer:
            result = model(**batch)
        maps = observer.finalize(result.anomaly_evidence)
        lce = compute_lce(maps["abnormal_raw"], maps["normal_raw"], [row["scout_label"]])
        attention.scope = "lm"
        gl = torch.autograd.grad(result.loss, params, retain_graph=True, allow_unused=True)
        attention.scope = "lce"
        ga = torch.autograd.grad(lce["per_image_loss"][0], params, retain_graph=False, allow_unused=True)
        if any(g is None or not torch.isfinite(g).all() for g in gl):
            raise RuntimeError("Invalid LM T21 gradient")
        lm = {n:g.detach().float().cpu() for n,g in zip(TRAINABLE_NAMES,gl)}
        aux = exact_s9_from_full(dict(zip(TRAINABLE_NAMES,ga)), S9_NAMES)
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Model .grad residue")
        lm_images.append(lm); lce_images.append(aux)
        images.append({"sample_id":row["sample_id"],"exposure_id":row["scout_exposure_id"],"sha256":row["sha256"],"label":row["scout_label"], "lm_loss":float(result.loss.detach()), "lce_loss":float(lce["per_image_loss"][0].detach()), "lce_score":float(lce["score"][0].detach()), "effective_support":float(lce["effective_support"][0].detach()), "top11_mass":float(lce["top11_mass"][0].detach()), "input_inventory":inventory(input_record), "lm_s9_norm":math.sqrt(sum(float(lm[n].double().square().sum()) for n in S9_NAMES)), "lce_s9_norm":math.sqrt(sum(float(aux[n].double().square().sum()) for n in S9_NAMES)), "exact_s9":True,"structural_zero_outside_s9":True})
        if raw_inputs:
            inputs.append(input_record)
        del result, lce, maps, gl, ga, batch
        gc.collect(); torch.cuda.synchronize()
    return lm_images,lce_images,images,inputs,{"before_block":rng_state_fingerprint(before),"after_block":rng_state_fingerprint(snapshot_rng_state())}


def make_optimizer(initial, device):
    masters = {n:torch.nn.Parameter(initial[n].to(device).float().clone(), requires_grad=False) for n in TRAINABLE_NAMES}
    optimizer = torch.optim.AdamW(list(masters.values()), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0.)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_multiplier)
    return masters,optimizer,scheduler


def update(masters,optimizer,scheduler,applied):
    before = {n:p.detach().clone() for n,p in masters.items()}
    optimizer.zero_grad(set_to_none=True)
    for n,p in masters.items():
        p.grad = applied[n].to(p)
    lr = optimizer.param_groups[0]["lr"]
    optimizer.step();scheduler.step()
    if any(not torch.isfinite(p).all() for p in masters.values()):
        raise RuntimeError("Nonfinite master")
    return {"total_l2_change":math.sqrt(sum(float((masters[n]-before[n]).double().square().sum()) for n in masters)), "changed_tensor_count":sum(bool(torch.any(masters[n]!=before[n])) for n in masters),"learning_rate_used":lr,"scheduler_step":scheduler.last_epoch}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.mode == "diagnostic" and args.blocks != 1 or args.mode == "train" and args.blocks not in (4,12) or args.mode == "resume" and args.blocks != 13:
        raise RuntimeError("Outside registered budget")
    if args.mode != "resume" and args.resume is not None:
        raise RuntimeError("Fresh starts cannot accept checkpoint")
    split,manifest = data_contract(args.split_audit,args.blocks)
    preflight = json.loads(args.preflight.read_text())
    if canonical_json_sha256(preflight["base_checkpoint"]) != BASE_SHA256 or canonical_json_sha256(matched_configuration()) != CONFIG_SHA256:
        raise RuntimeError("Base/config identity")
    repo = Path(__file__).resolve().parents[2]
    source_files = (*SOURCE_FILES,*[str(p.relative_to(repo)) for p in sorted((repo/"reproduction/stage2q").rglob("*.py"))])
    identity = git_identity(repo)
    if identity["dirty"]:
        raise RuntimeError("Source must be clean")
    provenance = {"source_commit":identity["commit"],"implementation_fingerprint":implementation_source_record(repo,source_files)["fingerprint"],"base_checkpoint_fingerprint":BASE_SHA256,"train_manifest_sha256":TRAIN_SHA256,"matched_configuration_sha256":CONFIG_SHA256,"execution_setting":args.setting,"scope":"training_only_reproducibility"}
    write(args.output/"provenance.json",provenance)
    attention = AttentionExecution(args.setting,microreplay=args.mode=="diagnostic").install()
    import flash_attn
    environment = {"torch":torch.__version__,"cuda":torch.version.cuda,"flash_attn":flash_attn.__version__,"flash_interface_file":attention.interface.__file__,"flash_interface_sha256":sha256_file(Path(attention.interface.__file__)),"attention":"flash_attention_2","gradient_checkpointing":"nonreentrant","model_dtype":"bfloat16","master_dtype":"float32","torch_deterministic":torch.are_deterministic_algorithms_enabled(),"cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,"matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"environment":{k:os.environ.get(k) for k in ("FLASH_ATTENTION_DETERMINISTIC","CUBLAS_WORKSPACE_CONFIG","PYTHONHASHSEED")}}
    write(args.output/"environment.json",environment)
    write(args.output/"cpu-thread-settings.json",{"torch_num_threads":torch.get_num_threads(),"torch_num_interop_threads":torch.get_num_interop_threads(),"OMP_NUM_THREADS":os.environ.get("OMP_NUM_THREADS"),"MKL_NUM_THREADS":os.environ.get("MKL_NUM_THREADS")})
    monitor = NvidiaSmiMonitor(args.output/"nvidia-smi.csv");monitor.start()
    startblock = 12 if args.mode=="resume" else 0
    allowed = [str((Path(split["image_root"])/r["relative_path"]).resolve()) for r in manifest[startblock*3:args.blocks*3]]
    fileaudit = MedicalImageOpenAudit(Path(split["image_root"]),allowed);fileaudit.install()
    try:
        os.environ["MEDIC_AD_STAGE2H_CALIBRATION_ANNOTATION"] = split["train_annotation"]
        os.environ["MEDIC_AD_STAGE2H_IMAGE_ROOT"] = split["image_root"]
        from reproduction.stage2h.runtime import load_stage2h_runtime,make_stage2h_dataset
        wrapper,audit = load_stage2h_runtime(args.base_model,seed=42,training=True,gradient_checkpointing=True)
        model = wrapper.llm
        if canonical_json_sha256(audit["step0_state"]) != INITIAL_STATE_SHA256:
            raise RuntimeError("Original T21 initialization mismatch")
        write(args.output/"initial-state.json",audit["step0_state"])
        if args.setting.startswith("canonical_"):
            canonical=canonicalize_dormant(model)
            if canonical_json_sha256(trainable_state_record(model))!=INITIAL_STATE_SHA256:
                raise RuntimeError("Dormant canonicalization changed T21")
            write(args.output/"dormant-state-repair.json",canonical)
        write(args.output/"runtime-config.json",{"training":model.training,"text_attention_dropout":getattr(model.config.get_text_config(),"attention_dropout",None),"dropout_modules":[{"name":n,"p":m.p,"training":m.training} for n,m in model.named_modules() if isinstance(m,torch.nn.Dropout)],"text_attention":model.config._attn_implementation,"gradient_checkpointing":True,"use_cache":model.config.use_cache,"dormant_head_guard":args.setting.startswith("canonical_")})
        initial_full = full_state(model);write(args.output/"full-initial-state.json",initial_full)
        initial_runtime = runtime_state(model)
        initial = cpu_tree({n:dict(model.named_parameters())[n] for n in TRAINABLE_NAMES})
        _,dataset,collator = make_stage2h_dataset(args.base_model,"medic_ad_stage2h_calibration",shuffle=False)
        if [(r.get("cabg_sample_id"),r.get("scout_exposure_id")) for r in dataset.list_data_dict] != [(r["sample_id"],r["scout_exposure_id"]) for r in manifest]:
            raise RuntimeError("Exposure order")
        initial_rng = snapshot_rng_state()
        if rng_state_fingerprint(initial_rng) != INITIAL_RNG_SHA256:
            raise RuntimeError("Original RNG identity")
        if args.mode == "diagnostic":
            save_raw(args.output/"start.pt.gz",{"trainable":initial,"runtime":initial_runtime,"rng":initial_rng})
            for repeat in range(2):
                restore_runtime(model,initial_runtime);restore_rng_state(initial_rng)
                if full_state(model) != initial_full:
                    raise RuntimeError("Diagnostic start state drift")
                started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                lm,lce,images,inputs,rng = collect(wrapper,dataset,collator,manifest,0,attention,True)
                controller=DynamicController()
                applied,details=compute_block_gradients(lm,lce,[r["label"] for r in images],trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=controller)
                masters,opt,sched=make_optimizer(initial,wrapper.device)
                delta=update(masters,opt,sched,applied)
                raw={"inputs":inputs,"lm":[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lm],"lce":[{n:v.to(torch.bfloat16) for n,v in g.items()} for g in lce],"aggregate_lm":details["aggregate_lm"],"aggregate_lce":details["aggregate_lce"],"applied":applied,"updated_master":cpu_tree(masters),"optimizer_inventory":inventory(opt.state_dict()),"kernels":attention.kernels if repeat==0 else {}}
                # BF16 storage is lossless for gradients originally produced in BF16.
                if any(not torch.equal(g[n],raw[key][i][n].float()) for key,gs in (("lm",lm),("lce",lce)) for i,g in enumerate(gs) for n in g):
                    raise RuntimeError("Gradient archival cast is not lossless")
                elapsed=time.perf_counter()-started
                saved=save_raw(args.output/f"repeat{repeat}.pt.gz",raw)
                restore_runtime(model,initial_runtime)
                if full_state(model)!=initial_full:
                    raise RuntimeError("No-update diagnostic mutated parameters/buffers")
                write(args.output/f"repeat{repeat}.json",{"status":"SUCCESS","repeat":repeat,"images":images,"rng":rng,"controller":details["budget"],"gradient":details["gradient"],"update":delta,"raw":saved,"block_seconds":elapsed,"peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,"state_unchanged":True,"model_grad_empty":True})
                del raw,lm,lce,applied,details,masters,opt,sched;gc.collect();torch.cuda.empty_cache()
        else:
            masters,opt,sched=make_optimizer(initial,wrapper.device);controller=DynamicController();anchor="0"*64
            resume_audit={"resumed":False}
            if args.mode=="resume":
                if args.resume is None:
                    raise RuntimeError("Resume requires own short12 checkpoint")
                payload,check=load_strict(args.resume,provenance)
                if payload["sampler_state"]["cursor"]!=12:
                    raise RuntimeError("Not block12 state")
                sideinfo=json.loads(args.resume.with_name("runtime-state.json").read_text())
                sidepath=args.resume.with_name("runtime-state.pt.gz")
                if sideinfo["checkpoint_sha256"]!=check["checkpoint_sha256"] or sideinfo["raw"]["sha256"]!=sha256_file(sidepath):
                    raise RuntimeError("Runtime sidecar identity")
                side=load_raw(sidepath)
                named=dict(model.named_parameters())
                for n in TRAINABLE_NAMES:
                    masters[n].data.copy_(payload["master_state"][n].to(wrapper.device));named[n].data.copy_(payload["adapter_state"][n].to(wrapper.device))
                opt.load_state_dict(payload["optimizer_state"]);_move_optimizer_state(opt,wrapper.device)
                sched.load_state_dict(payload["scheduler_state"]);controller.load_state_dict(payload["controller_state"])
                restore_runtime(model,side);restore_rng_state(payload["rng_state"])
                anchor=payload["trace_state"]["last_record_sha256"]
                exact={"adapter":all(torch.equal(named[n].cpu(),payload["adapter_state"][n]) for n in TRAINABLE_NAMES),"master":all(torch.equal(masters[n].cpu(),payload["master_state"][n]) for n in TRAINABLE_NAMES),"optimizer":inventory(cpu_tree(opt.state_dict()))==inventory(payload["optimizer_state"]),"controller":controller.state_dict()==payload["controller_state"],"scheduler":sched.state_dict()==payload["scheduler_state"],"runtime":inventory(runtime_state(model))==inventory(side),"rng":rng_state_fingerprint(snapshot_rng_state())==check["rng_fingerprint"]}
                if not all(exact.values()):
                    raise RuntimeError("Fresh-process restore mismatch")
                resume_audit={"resumed":True,"exact":exact,"checkpoint_sha256":check["checkpoint_sha256"],"sampler":payload["sampler_state"],"provenance":payload["provenance"]}
            write(args.output/"resume-audit.json",resume_audit)
            for block in range(startblock,args.blocks):
                started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
                lm,lce,images,_,rng=collect(wrapper,dataset,collator,manifest,block,attention)
                applied,details=compute_block_gradients(lm,lce,[r["label"] for r in images],trainable_names=TRAINABLE_NAMES,s9_names=S9_NAMES,controller=controller)
                delta=update(masters,opt,sched,applied)
                for n,p in masters.items():
                    dict(model.named_parameters())[n].data.copy_(p.data.to(torch.bfloat16))
                saved=save_raw(args.output/f"applied-{block+1:02d}.pt.gz",applied)
                record={"block_id":block,"images":images,"rng":rng,"controller":details["budget"],"gradient":details["gradient"],"parameter_update":delta,"applied":saved,"block_seconds":time.perf_counter()-started,"previous_record_sha256":anchor,"peak_allocated_mib":torch.cuda.max_memory_allocated()/1024**2}
                anchor=canonical_json_sha256(record);record["record_sha256"]=anchor
                with (args.output/"trace.jsonl").open("a") as f:
                    f.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
                del lm,lce,applied,details;gc.collect();torch.cuda.empty_cache()
            payload=_checkpoint_payload(model,masters,opt,sched,controller,cursor=args.blocks,split=split,run_provenance=provenance,last_hash=anchor)
            check=save_atomic(args.output/"checkpoint.pt",payload)
            saved=save_raw(args.output/"runtime-state.pt.gz",runtime_state(model))
            write(args.output/"runtime-state.json",{"checkpoint_sha256":check["checkpoint_sha256"],"raw":saved})
        write(args.output/"attention-calls.json",attention.report())
        file_result=fileaudit.result();file_result.update(heldout_image_files_opened=0,protected_internal_test_image_files_opened=0,protected_internal_test_outputs_read=0)
        if file_result["status"]!="SUCCESS":
            raise RuntimeError("File allowlist violation")
        write(args.output/"file-open-audit.json",file_result)
        write(args.output/"result.json",{"status":"SUCCESS","mode":args.mode,"setting":args.setting,"blocks":args.blocks,"claim_scope":"training_only_reproducibility"})
    finally:
        report=monitor.stop();write(args.output/"monitor-summary.json",report);attention.close()
        if report["peak_memory_used_mib"]>22500:
            raise RuntimeError("External VRAM budget exceeded")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("diagnostic","train","resume"),required=True)
    p.add_argument("--setting",choices=("original","canonical_dormant","canonical_flash_deterministic"),required=True)
    p.add_argument("--blocks",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base-model",type=Path,default=Path("/home/checkpoints/Lingshu-7B"))
    p.add_argument("--split-audit",type=Path,default=Path("/home/data/medic-ad/cabg-lce-mil-v1.2-d4-scout-v1/split-audit.json"))
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--resume",type=Path)
    run(p.parse_args())


if __name__=="__main__":
    main()

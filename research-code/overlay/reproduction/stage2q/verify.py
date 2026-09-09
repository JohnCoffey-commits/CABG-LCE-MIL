"""Independent comparisons of raw saved gradients/state; no producer math import."""
import argparse
import gzip
import itertools
import json
import math
from pathlib import Path

import torch

from reproduction.stage2i.fingerprint import canonical_json_sha256, sha256_file
from reproduction.stage2l.constants import TRAINABLE_NAMES, S9_NAMES
from reproduction.stage2l.checkpoint import load_strict


def load(path):
    return json.loads(Path(path).read_text())


def raw(path):
    with gzip.open(path,"rb") as f:
        return torch.load(f,map_location="cpu",weights_only=False)


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def compare_tensor(a,b):
    require(a.shape==b.shape and a.dtype==b.dtype,"Tensor identity")
    x,y=a.double(),b.double()
    require(bool(torch.isfinite(x).all() and torch.isfinite(y).all()),"Nonfinite tensor")
    nx,ny=float(x.square().sum()),float(y.square().sum())
    diff=float((x-y).square().sum());dot=float((x*y).sum())
    threshold=max(1e-12,1e-6*max(float(x.abs().max()) if x.numel() else 0,float(y.abs().max()) if y.numel() else 0))
    near=(x.abs()<=threshold)|(y.abs()<=threshold)
    signs=(x.sign()!=y.sign())
    exact=torch.equal(a,b)
    return {"exact":exact,"absolute_l2":math.sqrt(diff),"relative_l2":math.sqrt(diff)/max(math.sqrt(nx),math.sqrt(ny),1e-12),"cosine":1. if exact else dot/math.sqrt(nx*ny) if nx*ny else 0.,"max_abs_difference":float((x-y).abs().max()) if x.numel() else 0.,"meaningful_sign_disagreements":int((signs&~near).sum()),"near_zero_sign_disagreements":int((signs&near).sum()),"near_zero_threshold":threshold,"elements":x.numel(),"a_squared_norm":nx,"b_squared_norm":ny,"dot":dot}


def compare_map(a,b):
    require(set(a)==set(b),"Gradient/tensor support identity")
    per={n:compare_tensor(a[n],b[n]) for n in sorted(a)}
    diff=sum(r["absolute_l2"]**2 for r in per.values());nx=sum(r["a_squared_norm"] for r in per.values());ny=sum(r["b_squared_norm"] for r in per.values());dot=sum(r["dot"] for r in per.values())
    exact=all(r["exact"] for r in per.values())
    return {"exact":exact,"absolute_l2":math.sqrt(diff),"relative_l2":math.sqrt(diff)/max(math.sqrt(nx),math.sqrt(ny),1e-12),"cosine":1. if exact else dot/math.sqrt(nx*ny) if nx*ny else 0.,"meaningful_sign_disagreements":sum(r["meaningful_sign_disagreements"] for r in per.values()),"near_zero_sign_disagreements":sum(r["near_zero_sign_disagreements"] for r in per.values()),"per_tensor":per}


def close(a,b,message,rtol=2e-6,atol=1e-10):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=rtol,abs_tol=atol),message)


def norm32(g):
    total=torch.zeros((),dtype=torch.float32)
    for n in sorted(g):
        total+=g[n].float().square().sum()
    return float(total.sqrt())


def verify_raw_math(data,record,start):
    require(len(data["lm"])==len(data["lce"])==3,"Three images")
    for g in data["lm"]:
        require(set(g)==set(TRAINABLE_NAMES),"LM T21")
    for g in data["lce"]:
        require(set(g)==set(S9_NAMES),"LCE S9")
    labels=[r["label"] for r in record["images"]]
    require(labels.count("normal")==1 and labels.count("abnormal")==2,"Class balance")
    lm={n:torch.stack([g[n].float() for g in data["lm"]]).mean(0) for n in sorted(TRAINABLE_NAMES)}
    lce={n:.5*(torch.stack([data["lce"][i][n].float() for i,l in enumerate(labels) if l=="normal"]).mean(0)+torch.stack([data["lce"][i][n].float() for i,l in enumerate(labels) if l=="abnormal"]).mean(0)) for n in sorted(S9_NAMES)}
    require(all(torch.equal(lm[n],data["aggregate_lm"][n]) for n in lm),"Independent LM aggregation")
    require(all(torch.equal(lce[n],data["aggregate_lce"][n]) for n in lce),"Independent balanced LCE aggregation")
    rms={}
    for key in ("lm","lce"):
        per=[norm32({n:g[n] for n in S9_NAMES})**2 for g in data[key]]
        rms[key]=math.sqrt(.5*(sum(v for v,l in zip(per,labels) if l=="normal")+sum(v for v,l in zip(per,labels) if l=="abnormal")/2)+1e-16)
    c=record["controller"]
    close(c["corrected_lm_rms"],rms["lm"],"LM RMS")
    close(c["corrected_lce_rms"],rms["lce"],"LCE RMS")
    close(c["lm_ema"],.1*rms["lm"],"LM EMA")
    close(c["lce_ema"],.1*rms["lce"],"LCE EMA")
    lmnorm=math.sqrt(sum(float(lm[n].double().square().sum()) for n in S9_NAMES))
    lcenorm=math.sqrt(sum(float(lce[n].double().square().sum()) for n in S9_NAMES))
    close(c["lambda_raw"],.1*rms["lm"]/(rms["lce"]+1e-8),"raw lambda")
    close(c["lambda_cap"],.2*lmnorm/(lcenorm+1e-8),"cap lambda")
    close(c["lambda_final"],min(c["lambda_raw"],c["lambda_cap"]),"final lambda")
    combined={n:v.clone() for n,v in lm.items()}
    for n in lce:
        combined[n].add_(lce[n]*c["lambda_final"])
    scale=min(1.,1./(norm32(combined)+1e-12))
    close(record["gradient"]["clip_scale"],scale,"global clip")
    # CPU reduction thread counts may round the recomputed scalar differently.
    # After independently bounding its error, reconstruct the actual FP32 multiply.
    require(all(torch.equal(combined[n]*record["gradient"]["clip_scale"],data["applied"][n]) for n in lm),"Independent T21 combined/clipped gradient")
    require(c["lambda_final"]*lcenorm<=.2*lmnorm+1e-7,"Trust cap")
    maximum=0.
    for n,g in data["applied"].items():
        # Algebraic first-step AdamW with original epsilon and zero weight decay.
        expected=start["trainable"][n].float().double()-1e-4*g.double()/(g.double().abs()+1e-8)
        maximum=max(maximum,float((expected-data["updated_master"][n].double()).abs().max()))
    require(maximum<=2e-7,"AdamW one-step mathematical mismatch")
    return {"aggregation_and_applied_gradient_bitwise_reconstructed":True,"controller_and_clip_recomputed":True,"adamw_closed_form_max_abs_error":maximum}


def small_enough(c):
    return c["relative_l2"]<=1e-5 and c["cosine"]>=1-1e-8 and c["meaningful_sign_disagreements"]==0


def verify_process(p):
    a=load(p/"file-open-audit.json");require(a["status"]=="SUCCESS" and not a["non_allowlisted_paths"],"File boundary")
    require(a["heldout_image_files_opened"]==a["protected_internal_test_image_files_opened"]==0,"Image boundary")
    require(load(p/"monitor-summary.json")["peak_memory_used_mib"]<=22500,"Resource budget")
    calls=load(p/"attention-calls.json")
    if calls["setting"]=="canonical_flash_deterministic":
        require(not any(k.endswith("False") for k in calls["actual_calls"]),"Deterministic option not effective")
    return calls


def diagnostic(root, state_mismatch_audit=False):
    dirs=[root/"p0",root/"p1"]
    for p in dirs:verify_process(p)
    full_equal=load(dirs[0]/"full-initial-state.json")==load(dirs[1]/"full-initial-state.json")
    require(full_equal or state_mismatch_audit,"Cross-process full initial state mismatch")
    require(load(dirs[0]/"provenance.json")==load(dirs[1]/"provenance.json"),"Cross-process provenance")
    reference=raw(dirs[0]/"repeat0.pt.gz");start=raw(dirs[0]/"start.pt.gz")
    reference_record=load(dirs[0]/"repeat0.json")
    math_checks={};comparisons={};kernels={}
    for process,p in enumerate(dirs):
        current_start=raw(p/"start.pt.gz")
        require(all(torch.equal(start["trainable"][n],current_start["trainable"][n]) for n in TRAINABLE_NAMES),"Initial trainable raw identity")
        for repeat in range(2):
            record=load(p/f"repeat{repeat}.json");path=p/f"repeat{repeat}.pt.gz"
            require(sha256_file(path)==record["raw"]["sha256"],"Raw artifact hash")
            data=raw(path);label=f"p{process}-r{repeat}"
            require(record["rng"]==reference_record["rng"],"Repeat RNG identity")
            require([r["input_inventory"] for r in record["images"]]==[r["input_inventory"] for r in reference_record["images"]],"Processed inputs differ")
            for x,y in zip(reference["inputs"],data["inputs"]):
                require(x.keys()==y.keys(),"Input keys")
                require(all(torch.equal(x[k],y[k]) if isinstance(x[k],torch.Tensor) else x[k]==y[k] for k in x),"Saved input mismatch")
            math_checks[label]=verify_raw_math(data,record,current_start)
            if label!="p0-r0":
                compared={key:compare_map(reference[key],data[key]) for key in ("aggregate_lm","aggregate_lce","applied")}
                compared.update({f"{key}_image{i}":compare_map(reference[key][i],data[key][i]) for key in ("lm","lce") for i in range(3)})
                deltas=[{n:d["updated_master"][n]-s["trainable"][n].float() for n in TRAINABLE_NAMES} for d,s in ((reference,start),(data,current_start))]
                compared["adamw_update"]=compare_map(*deltas)
                compared["bf16_updated_model"]=compare_map({n:v.to(torch.bfloat16) for n,v in reference["updated_master"].items()},{n:v.to(torch.bfloat16) for n,v in data["updated_master"].items()})
                compared["all_forward_observations_exact"]=all(x[k]==y[k] for x,y in zip(reference_record["images"],record["images"]) for k in ("lm_loss","lce_loss","lce_score","effective_support","top11_mass"))
                comparisons[label]=compared
            for kind,kernel in data["kernels"].items():
                kernels[f"{label}-{kind}"]={"deterministic":kernel["deterministic"],"scope":kernel["scope"],"same_input_kernel_replays":[compare_map(kernel["outputs"][0],k) for k in kernel["outputs"][1:]]}
            del data
    ready=full_equal and all(small_enough(c[k]) for c in comparisons.values() for k in ("applied","adamw_update"))
    return {"status":"SUCCESS","single_block_comparable":ready,"all_gradient_and_update_comparisons_exact":all(v["exact"] for c in comparisons.values() for v in c.values() if isinstance(v,dict)),"comparisons":comparisons,"kernel_replays":kernels,"independent_math_checks":math_checks,"full_initial_state_equal":full_equal,"cross_process_comparisons_not_attributable_if_state_mismatch":not full_equal,"inputs_rng_equal":True,"heldout_images_loaded":0,"setting":load(dirs[0]/"provenance.json")["execution_setting"]}


def trace_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def short_pair(root, blocks, resume_root=None):
    dirs=[root/"p0",root/"p1"]
    require(blocks in (4,12),"Registered short horizon")
    traces=[];checkpoints=[];comparisons=[]
    for process,p in enumerate(dirs):
        verify_process(p)
        provenance=load(p/"provenance.json")
        trace=trace_rows(p/"trace.jsonl")
        require(len(trace)==blocks,"Short trace length")
        require(load(p/"resume-audit.json")["resumed"] is False,"Short run not initialized")
        payload,manifest=load_strict(p/"checkpoint.pt",provenance)
        require(payload["sampler_state"]["cursor"]==blocks,"Short checkpoint cursor")
        if resume_root is not None:
            require(blocks==12,"Only12 permits recovery block")
            q=resume_root/f"p{process}"
            verify_process(q)
            require(load(q/"provenance.json")==provenance,"Execution settings changed at resume")
            audit=load(q/"resume-audit.json")
            require(all(audit["exact"].values()) and audit["checkpoint_sha256"]==manifest["checkpoint_sha256"],"Own checkpoint exact restore")
            continuation=trace_rows(q/"trace.jsonl")
            require(len(continuation)==1 and continuation[0]["block_id"]==12,"Bounded resume computation")
            trace+=continuation
            payload,manifest=load_strict(q/"checkpoint.pt",provenance)
        require(payload["trace_state"]["last_record_sha256"]==trace[-1]["record_sha256"],"Checkpoint trace anchor")
        require(payload["controller_state"]["valid_blocks"]==len(trace),"EMA block cursor")
        require(payload["scheduler_state"]["last_epoch"]==len(trace),"Scheduler cursor")
        require(all(float(v['step'])==len(trace) for v in payload['optimizer_state']['state'].values()),"Adam steps")
        previous="0"*64;ema_lm=ema_lce=0.
        for i,r in enumerate(trace):
            require(r["block_id"]==i and r["previous_record_sha256"]==previous,"Trace chain order")
            previous=r['record_sha256'];require(previous==canonical_json_sha256({k:v for k,v in r.items() if k!='record_sha256'}),"Trace hash")
            if i:require(r['rng']['before_block']==trace[i-1]['rng']['after_block'],"Training RNG discontinuity")
            require(r['parameter_update']['scheduler_step']==i+1,"Optimizer/scheduler drift")
            close(r['parameter_update']['learning_rate_used'],1e-4*.5*(1+math.cos(math.pi*i/24)),"Fixed LR schedule")
            c,g=r['controller'],r['gradient'];images=r['images']
            for key in ('lm_loss','lce_loss','lce_score','effective_support','top11_mass','lm_s9_norm','lce_s9_norm'):
                require(all(math.isfinite(float(im[key])) for im in images),'Finite training observations')
            norms={key:math.sqrt(.5*(sum(im[key+'_s9_norm']**2 for im in images if im['label']=='normal')+sum(im[key+'_s9_norm']**2 for im in images if im['label']=='abnormal')/2)+1e-16) for key in ('lm','lce')}
            ema_lm=.9*ema_lm+.1*norms['lm'];ema_lce=.9*ema_lce+.1*norms['lce']
            close(c['lm_ema'],ema_lm,'Short LM EMA');close(c['lce_ema'],ema_lce,'Short LCE EMA')
            close(c['lambda_raw'],.1*ema_lm/(ema_lce+(1-.9**(i+1))*1e-8),'Short dynamic lambda')
            close(c['lambda_cap'],.2*c['lm_s9_norm']/(c['lce_s9_norm']+1e-8),'Short cap')
            close(c['lambda_final'],min(c['lambda_raw'],c['lambda_cap']),'Unmodified CABG')
            close(g['clip_scale'],min(1.,1/(g['preclip_full_norm']+1e-12)),'Short clip')
            require(g['scaled_auxiliary_shared_norm']<=.2*g['lm_shared_norm']+1e-5*g['lm_shared_norm']+1e-8,'Short trust cap')
        checkpoints.append(payload);traces.append(trace)
    require(load(dirs[0]/'full-initial-state.json')==load(dirs[1]/'full-initial-state.json'),'Full model initial identity')
    require(load(dirs[0]/'provenance.json')==load(dirs[1]/'provenance.json'),'Matched settings/source')
    for i,(a,b) in enumerate(zip(*traces)):
        require(a['rng']==b['rng'],'Cross-run RNG identity')
        require([(im['exposure_id'],im['sha256'],im['input_inventory']) for im in a['images']]==[(im['exposure_id'],im['sha256'],im['input_inventory']) for im in b['images']],'Training inputs/order')
        paths=[(root if i<blocks else resume_root)/f'p{p}'/f'applied-{i+1:02d}.pt.gz' for p in range(2)]
        for path,r in zip(paths,(a,b)):require(sha256_file(path)==r['applied']['sha256'],'Applied artifact digest')
        comparisons.append(compare_map(raw(paths[0]),raw(paths[1])))
    def distance(x,y):
        return math.sqrt(sum((a-b)**2 for a,b in zip(x,y)))/max(math.sqrt(sum(a*a for a in x)),math.sqrt(sum(b*b for b in y)),1e-12)
    scalars={key:distance(*[[im[key] for r in t for im in r['images']] for t in traces]) for key in ('lm_loss','lce_loss','effective_support','top11_mass')}
    scalars['lambda']=distance(*[[r['controller']['lambda_final'] for r in t] for t in traces])
    master=compare_map(checkpoints[0]['master_state'],checkpoints[1]['master_state'])
    path=max(sum(r['parameter_update']['total_l2_change'] for r in t) for t in traces)
    relative_update_path=master['absolute_l2']/max(path,1e-12)
    flags=[[sorted(im['effective_support'] for im in r['images'])[1]<128 or sorted(im['top11_mass'] for im in r['images'])[1]>.35 for r in t] for t in traces]
    passed=all(small_enough(c) for c in comparisons) and max(scalars.values())<=1e-5 and relative_update_path<=1e-5 and flags[0]==flags[1]
    return {'status':'SUCCESS','comparable':passed,'blocks_per_trajectory':len(traces[0]),'all_applied_gradients_exact':all(c['exact'] for c in comparisons),'master_state_exact':master['exact'],'controller_exact':checkpoints[0]['controller_state']==checkpoints[1]['controller_state'],'scheduler_exact':checkpoints[0]['scheduler_state']==checkpoints[1]['scheduler_state'],'scalar_series_relative_l2':scalars,'master_relative_update_path':relative_update_path,'master_comparison':master,'applied_comparisons':comparisons,'spatial_flags_equal':flags[0]==flags[1],'fresh_resume_verified':resume_root is not None,'heldout_images_loaded':0}


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument('--short-blocks',type=int);p.add_argument('--resume-root',type=Path)
    p.add_argument('--state-mismatch-audit',action='store_true')
    a=p.parse_args();require(not a.output.exists(),"No evaluator overwrite")
    result=short_pair(a.root,a.short_blocks,a.resume_root) if a.short_blocks else diagnostic(a.root,a.state_mismatch_audit)
    with a.output.open("x") as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons","kernel_replays","applied_comparisons","master_comparison")},indent=2))


if __name__=="__main__":main()

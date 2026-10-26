"""Bind admitted data, methods, execution sources and measured gates before formal use."""
import argparse,json,math,time
from pathlib import Path
from reproduction.stage2af.frozen import sha
from reproduction.stage2af.core import ARMS
from reproduction.stage2af.inputs import QUESTION,DECODE
from reproduction.stage2l.constants import TRAINABLE_NAMES

def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;repo=root/'code/runtime'
    gates=['debug/cpu-tests-v3.json','data/admission-v1/admission.json','data/admission-v1/independent-verification.json','data/prepared-v1/verification.json',
           'debug/calibration-independent-verification-v1.json','debug/preflight-continuous-v1/summary.json','debug/preflight-resume-verification-v2/summary.json']
    for rel in gates:assert read(root/rel)['status'] in ['PASS','PASS_IMAGE_LEVEL_ADMISSION']
    assert read(root/'debug/preflight-resume-verification-v2/summary.json')['resume_exact']
    env=read(root/'environment/identity-v1.json');assert 'NVIDIA L4' in env['gpu'] and env['cublas']==':4096:8'
    coefficients={};initials={}
    for seed in [43,44]:
        d=root/'debug'/f'calibration-s{seed}-v1';s=read(d/'summary.json');assert s['status']=='PASS' and s['actual_updates']==0
        coefficients[str(seed)]=s['coefficient'];initials[str(seed)]=read(d/'initial-identity.json')
    # Bind all archived model/optimizer helpers plus this producer layer, while
    # keeping the independent checker free to be corrected with its own identity.
    helpers=[]
    for directory in ['reproduction/stage2h','reproduction/stage2i','reproduction/stage2k','reproduction/stage2l','reproduction/stage2m','reproduction/stage2p','reproduction/stage2q','reproduction/stage2s','models/Qwen2_5_VL','qwen-vl-finetune/qwenvl','reproduction/stage2f','reproduction/stage2ac']:
        helpers += list((repo/directory).rglob('*.py'))
    producer=['runtime.py','core.py','inputs.py','guard.py','budget.py','frozen.py','train.py','evaluate.py','linear_head.py','metrics.py','driver.py','job.py']
    helpers += [repo/'reproduction/stage2af'/name for name in producer]
    sources={str(p):sha(p) for p in sorted(set(helpers))};sources.update(env['implementation_files'])
    inputs={str(root/rel):sha(root/rel) for rel in gates+['data/prepared-v1/manifest.jsonl','data/prepared-v1/input-contract.json','data/admission-v1/train-manifest.jsonl','data/admission-v1/development-manifest.jsonl','data/admission-v1/final-manifest.jsonl','engineering-protocol-v1.md','environment/identity-v1.json','data-admission-report.md','engineering-amendment-01.md']}
    pre=read(root/'debug/preflight-continuous-v1/summary.json');times=pre['update_seconds']
    state={'status':'FROZEN_FOR_FORMAL_EXECUTION','frozen_epoch':time.time(),'stage':'stage2af','seeds':[43,44],'arms':list(ARMS),'horizon':1000,
        'dataset':{'name':'ChestX-Det','train':1000,'development':128,'final':542,'final_source':'official test after fixed QC exclusions','grouping':'image duplicate components; patient identity unknown','base_source_family_exposure':'NIH-Chest-X-Ray reported in Lingshu pretraining; exact sample overlap unknown'},
        'base_model':'/home/checkpoints/Lingshu-7B',
        'initial_frozen_parameters':{r['name']:r for r in read(root/'debug/preflight-continuous-v1/initial-state.json')['parameters'] if r['name'] not in TRAINABLE_NAMES},
        'initial_buffers':read(root/'debug/preflight-continuous-v1/initial-state.json')['buffers'],'initial_identities':initials,'fixed_coefficients':coefficients,
        'fixed_coefficient_rule':'median of8 initial training-block min(.1 class-RMS LM/LCE ratio,.2 aggregate shared LM/LCE norm ratio), eps1e-8; no dynamic cap after selection',
        'batch':{'size':3,'target_order':[0,1,1],'LM_weights':[1/3]*3,'LCE_weights':[.5,.25,.25],'sampler':'per-seed Python Random shuffled class queues, cyclic without replacement within each queue'},
        'optimizer':{'kind':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'master_dtype':'float32','model_dtype':'bfloat16','gradient_dtype':'bfloat16 before FP32 composition','global_clip':1.,'scheduler':'cosine horizon1000, optimizer then scheduler'},
        'cabg':{'rho':.1,'rho_max':.2,'ema_beta':.9,'epsilon':1e-8,'shared_support':'exact S9','total_trainable_support':'T21'},
        'question':QUESTION,'decode':DECODE,'parser':'strip lowercase entire response yes/no with optional single . or !; anything else invalid and wrong',
        'primary_metric':'actual generated balanced_accuracy','primary_comparisons':['cabg_lce minus lm_only','cabg_lce minus fixed_lce'],
        'secondary_metrics':['accuracy','class_correct_rates','invalid_count','anomaly_AUROC','anomaly_AP','support','top11','yes_no_logit_gap'],
        'evaluation_nodes':[24,100,300,1000],'main_endpoint':1000,'checkpoint_steps':[24]+list(range(50,1001,50)),
        'sparse_gradient_steps':[1,24,100,300,1000],'evaluation_order':'all formal training, all fixed dev endpoints, fixed head training/dev, then final release and all1000 endpoints/head final',
        'head':{'feature':'initial frozen visual[0] mean,3584D, seed43 zero-VPT','recipe':'prior stage2ac train-only standardization; class-balanced logistic C1, penalty1/N; no search','train_n':1000,'decision_threshold':0},
        'uncertainty':{'scope':'image-level descriptive paired class-stratified bootstrap; no patient independence or two-seed significance claim','resamples':10000,'rng_seed':20260918,'interval':[.025,.975],'zero_discordance':'report degenerate bootstrap and exact per-class97.5% upper discordance bound','checker':'independent multinomial100000draws seed20260919; CI tolerance max(.005,2*class-BA lattice resolution)'},
        'raw_tolerances':{'score':2e-6,'effective_support':.002,'top11_mass':2e-6},
        'failure_policy':'Nonfinite/true numerical instability terminates affected trajectory; engineering fixes preserve attempts and same scientific rules; lost/replayed updates count as extra; no best checkpoint selection or step extension',
        'limits':read(root/'resource-ledger.json')['limits'],'sources':sources,'input_files':inputs,'base_file_identities':env['base_model'],
        'preflight':{'continuous_update_seconds_including_sparse_saves':times,'conservative6000update_hours':max(times)*6000/3600,'checkpoint_estimate_bytes':read(root/'debug/preflight-continuous-v1/step0002.json')['bytes']},
        'download_estimate_bytes':1446064671,'outputs_not_evidence_of_success':'all six trajectories and required results remain incomplete at freeze'}
    assert state['preflight']['conservative6000update_hours']<55,'Measured run cannot fit GPU limit with evaluation margin'
    cp=state['preflight']['checkpoint_estimate_bytes'];projected=cp*21*6+100_000_000_000
    import shutil
    assert shutil.disk_usage(root).free-projected>=50_000_000_000
    state['peak_forecast']={'additional_checkpoint_bytes':cp*21*6,'other_new_and_retry_reserve_bytes':100_000_000_000,'projected_free_floor_bytes':shutil.disk_usage(root).free-projected}
    write(root/'protocol.json',state);write(root/'protocol-seal.json',{'sha256':sha(root/'protocol.json'),'frozen_epoch':state['frozen_epoch']})
    print(json.dumps({'status':state['status'],'sha256':sha(root/'protocol.json'),'fixed_coefficients':coefficients,'preflight':state['preflight'],'peak_forecast':state['peak_forecast']}),flush=True)
if __name__=='__main__':main()

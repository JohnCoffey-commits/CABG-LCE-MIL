"""Snapshot a small explicit whitelist; never read images/checkpoints or mutate sources."""
from pathlib import Path
import hashlib, json, shutil, sys, platform
import numpy, pandas, matplotlib, scipy, sklearn, PIL

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
AF = 'scale-study-20260918-v1/'
AG = 'mechanism-study-20260919-v1/'
AH = 'evidence-transport-20260919-v1/'
SOURCES = {
 'af_per_image.csv': AF+'verified-execution-v2/results-v1/per-image.csv',
 'af_endpoints.csv': AF+'verified-execution-v2/results-detail-v1/endpoint-details.csv',
 'af_pairs.csv': AF+'verified-execution-v2/results-detail-v1/paired-comparisons.csv',
 'af_steps.csv': AF+'verified-execution-v2/training-summary-v1/step-summary.csv',
 'af_trajectories.csv': AF+'verified-execution-v2/training-summary-v1/trajectory-summary.csv',
 'af_admission.json': AF+'verified-execution-v2/data/admission-v1/admission.json',
 'af_train_manifest.csv': AF+'verified-execution-v2/data/admission-v1/train-manifest.csv',
 'af_development_manifest.csv': AF+'verified-execution-v2/data/admission-v1/development-manifest.csv',
 'af_final_manifest.csv': AF+'verified-execution-v2/data/admission-v1/final-manifest.csv',
 'af_verification.json': AF+'verified-execution-v2/results-v1/independent-verification.json',
 'af_training_verification.json': AF+'verified-execution-v2/training-independent-verification-v1.json',
 'af_audit.md': AF+'completion-audit.md',
 'ag_endpoints.csv': AG+'tables/endpoints.csv',
 'ag_pairs.csv': AG+'tables/paired-comparisons.csv',
 'ag_a1_summary.json': AG+'verified-execution-v1/results/a1-v1/summary.json',
 'ag_a1_verification.json': AG+'verified-execution-v1/results/a1-v1/independent-verification-v1.json',
 'ag_a2_verification.json': AG+'verified-execution-v1/results/a2-independent-verification-v1.json',
 'ag_audit.md': AG+'completion-audit.md',
 'ah_per_image.csv': AH+'verified-execution-v1/results-v1/per-image.csv',
 'ah_endpoints.csv': AH+'verified-execution-v1/results-v1/endpoint-summary.csv',
 'ah_pairs.csv': AH+'verified-execution-v1/results-v1/paired-comparisons.csv',
 'ah_traces.json': AH+'verified-execution-v1/results-v1/trace-descriptives-posthoc.json',
 'ah_verification.json': AH+'verified-execution-v1/results-v1/independent-verification.json',
 'ah_training_verification.json': AH+'verified-execution-v1/training-independent-verification-v1.json',
 'ah_jobs.csv': AH+'closure-v1/job-costs.csv',
 'ah_audit.md': AH+'completion-audit.md',
}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    dest=ROOT/'data/source'; dest.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,rel in SOURCES.items():
        src=BASE/rel
        assert src.is_file() and src.stat().st_size < 5_000_000, str(src)
        out=dest/name
        if out.exists():
            assert sha(out)==sha(src), f'Existing snapshot changed: {name}'
        else: shutil.copyfile(src,out)
        records.append({'name':name,'source_relative_to_research_root':rel,'source_absolute':str(src),
                        'snapshot':str(out.relative_to(ROOT)),'bytes':src.stat().st_size,'sha256':sha(src)})
    (ROOT/'provenance').mkdir(exist_ok=True)
    (ROOT/'provenance/input-manifest.json').write_text(json.dumps({'scope':'small accepted local records only','files':records},indent=2)+'\n')
    env={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
         'libraries':{m.__name__:m.__version__ for m in (numpy,pandas,matplotlib,scipy,sklearn,PIL)},
         'model_libraries_loaded':False,'model_executions':0,'bootstrap_seed':20260920,'bootstrap_replicates':20000}
    (ROOT/'provenance/environment.json').write_text(json.dumps(env,indent=2)+'\n')
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in env['libraries'].items()).replace('sklearn==','scikit-learn==').replace('PIL==','Pillow==')+'\n')
    print(json.dumps({'snapshots':len(records),'bytes':sum(x['bytes'] for x in records),'root':str(ROOT)}))

if __name__=='__main__': main()

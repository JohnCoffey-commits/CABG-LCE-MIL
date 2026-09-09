"""Prepare a prospective diagnostic plan from the fresh metadata capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from reproduction.stage2ab.readiness import unpack,partition
from reproduction.stage2ac.contracts import write,selected_development,unique_training


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    s=json.loads((a.evidence/'remote-snapshot.stdout').read_text());data=unpack(s)
    parts=partition(data)
    assert all(x['unused']==0 for x in parts.values())
    chosen=selected_development(data['development']);train=unique_training(data['train'])
    assert len(train)==64
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in data['development']}
    features=[{**r,'role':'train'} for r in train]+[{**r,'role':'development'} for r in data['development']]
    env=json.loads((a.evidence/'live-environment.json').read_text())
    instance_start=env['epoch']-int(env['instance_elapsed']['stdout'])
    previous=771.  # Conservatively carry the earlier retained instance-on window.
    sources=data['aa_source'].copy()
    extra=['models/Qwen2_5_VL/Qwen2_5_VL_hf.py','transformers/generation/utils.py',
           'transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py',
           'transformers/models/qwen2_vl/image_processing_qwen2_vl.py']
    for folder in ('stage2ab','stage2ac'):
        extra.extend(str(f.relative_to(repo)) for f in (repo/'reproduction'/folder).iterdir() if f.suffix in ('.py','.md','.sh'))
    for name in extra:sources[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    for name,h in sources.items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==h
        assert hashlib.sha256(subprocess.check_output(['git','show',head+':'+name],cwd=repo)).hexdigest()==h
    phases={f's{s}-{arm}':{'seed':s,'arm':arm,'mode':'answers'} for s in (43,44) for arm in ('none','full','early')}
    phases['features']={'seed':43,'mode':'features'}
    write(a.output,{'code_commit':head,'sources':sources,'split':data['split'],
                    'development_check':chosen,'feature_rows':features,'phases':phases,
                    'instance_start_epoch':instance_start,'carried_instance_seconds':previous,
                    'deadline_epoch':instance_start+36000-previous,'limit_instance_seconds':36000,
                    'independent_cohort_admitted':False,'generation_scope':'four-case development pipeline only',
                    'initial_capture_sha256':hashlib.sha256((a.evidence/'remote-snapshot.stdout').read_bytes()).hexdigest()})
    print(json.dumps({'sources':len(sources),'selected':[r['sample_id'] for r in chosen],'deadline_epoch':instance_start+36000-previous}))


if __name__=='__main__':main()

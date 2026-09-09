"""Independent local stdlib mirror, source and descriptive-result verification."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    mirror=a.evidence/'verified-metadata-v1';root=mirror/'capability-pilot-v1';control=mirror/'capability-pilot-control-v1'
    inventory=read(control/'artifact-inventory-v1.json');verified=0
    for e in inventory['entries']:
        path=mirror/e['root']/e['path']
        if e['mirror']:
            assert path.stat().st_size==e['bytes'] and sha(path)==e['sha256'];verified+=1
        else:assert not path.exists()
    plan=read(root/'plan.json');assert plan==read(a.evidence/'attempt-v1/plan-v2.json')
    repo=Path(__file__).resolve().parents[2];v=read(root/'independent-verification.json')
    for name,h in plan['sources'].items():
        body=subprocess.check_output(['git','show',plan['code_commit']+':'+name],cwd=repo)
        assert hashlib.sha256(body).hexdigest()==h
        if name!='reproduction/stage2ac/verify.py':assert sha(repo/name)==h
    assert v['status']=='PASS' and v['independent_cases']==0
    checker=subprocess.check_output(['git','show',v['checker_commit']+':reproduction/stage2ac/verify.py'],cwd=repo)
    assert hashlib.sha256(checker).hexdigest()==v['checker_sha256']
    answers={};all_paths=set();pixels=0
    for name,cfg in plan['phases'].items():
        phase=root/name;result=read(phase/'result.json');audit=read(phase/'file-open-audit.json')
        assert result['status']=='PASS' and result['optimizer_updates']==0
        assert audit['status']=='SUCCESS' and not audit['internal_test_outputs_read'] and not audit['non_allowlisted_paths']
        expected=plan['feature_rows'] if cfg['mode']=='features' else plan['development_check']
        allowed={str(Path(plan['split']['image_root'])/r['relative_path']) for r in expected}
        assert set(audit['allowed_paths'])==allowed and set(audit['opened_paths'])==allowed
        all_paths.update(audit['opened_paths'])
        if cfg['mode']=='features':continue
        rows=read(phase/'observations.json');assert len(rows)==4
        correct=invalid=0
        for r,e in zip(rows,expected):
            assert r['sample_id']==e['sample_id'] and r['sha256']==e['sha256'] and r['target']==int(e['scout_label']=='abnormal')
            m=re.fullmatch(r'(yes|no)[.!]?',r['text'].strip(),flags=re.I)
            answer=int(m[1].lower()=='yes') if m else None
            assert answer==r['parsed'];invalid+=answer is None;correct+=answer is not None and answer==r['target']
            pixels+=r['generation_pixels_equal_bridge']
        answers[name]={'n':4,'correct':correct,'invalid':invalid,'accuracy':correct/4}
        assert answers[name]==v['answer_pipeline'][name]
    assert len(all_paths)==96
    baseline={}
    for role in ('train','development'):
        rows=[r for r in read(root/'linear-head/predictions.json') if r['role']==role]
        pos=[r['score'] for r in rows if r['target']];neg=[r['score'] for r in rows if not r['target']]
        auc=sum((Fraction(1) if x>y else Fraction(1,2) if x==y else Fraction(0)) for x in pos for y in neg)/(len(pos)*len(neg))
        ap=sum((Fraction(sum(x>=s for x in pos),sum(r['score']>=s for r in rows)) for s in pos),Fraction(0))/len(pos)
        accuracy=Fraction(sum(r['prediction']==r['target'] for r in rows),len(rows))
        assert abs(float(auc)-v['simple_baseline'][role]['auroc'])<1e-12
        assert abs(float(ap)-v['simple_baseline'][role]['average_precision'])<1e-12
        assert float(accuracy)==v['simple_baseline'][role]['accuracy']
        baseline[role]={'n':len(rows),'auroc_fraction':str(auc),'ap_fraction':str(ap),'accuracy_fraction':str(accuracy),
                        'confusion':dict(Counter(f"target{r['target']}_pred{r['prediction']}" for r in rows))}
    budget=read(root/'budget.json');assert dict(Counter(e['kind'] for e in budget['events']))==budget['counts']==v['counts']
    assert budget['model_wall_seconds']<5400 and max(e['epoch'] for e in budget['events'])<plan['deadline_epoch']-5400
    assert read(control/'closure-observation-v1.json')['ledger_instance_seconds']<36000
    result={'status':'PASS_METADATA_SOURCES_AND_DESCRIPTIVE_REDUCTIONS','mirrored_files_verified':verified,
            'remote_binary_bodies_not_copied':inventory['remote_body_files'],'inventory_sha256':sha(control/'artifact-inventory-v1.json'),
            'producer_commit':plan['code_commit'],'checker_commit':v['checker_commit'],'producer_source_files_verified':len(plan['sources']),
            'unique_original_images_opened':len(all_paths),'generation_pixels_equal_old_teacher_forcing':pixels,
            'answer_pipeline':answers,'simple_head_exact_metrics':baseline,'independent_cases':0,
            'scientific_conclusion':'INCONCLUSIVE_TRANSFER_NOT_MEASURED','research_decision':'BLOCKED_INDEPENDENT_COHORT_READINESS',
            'scope':'Local metadata/hash/stdlib arithmetic verification; raw tensor checks were performed separately on remote CPU',
            'script_sha256':sha(Path(__file__))}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'metadata_files':verified,'independent_cases':0}))


if __name__=='__main__':main()

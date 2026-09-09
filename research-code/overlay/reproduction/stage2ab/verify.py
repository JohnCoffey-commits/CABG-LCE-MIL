"""Independent stdlib verification; does not import the readiness producer."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify(workspace, snapshot_path, report_path):
    workspace = Path(workspace); repo = workspace / 'Medic-AD'
    s = json.loads(Path(snapshot_path).read_text())
    r = json.loads(Path(report_path).read_text())
    objects = {}
    for name, item in s['files'].items():
        body = item['text'].encode()
        assert sha(body) == item['sha256'] and len(body) == item['bytes'], name
        objects[name] = ([json.loads(line) for line in body.splitlines() if line]
                         if item['path'].endswith('.jsonl') else json.loads(body))
    assert s['files']['aa_verification']['sha256'] == '3a341d15a2333e1e535c753ece2320dec1206b9dd7c8b4f6cc05937ee8bb50ba'
    assert s['files']['aa_inventory']['sha256'] == '8cc2ec649030aeab58f0790fdf98e2c3a5c41c8e7becc93f835002ee036ed0b1'
    assert objects['aa_verification']['status'] == 'PASS'
    assert s['head'] == '71aaf0ad2f17a456d213b7d40b2ae7f080c6c7c2' and s['git_status'] == ''
    sources = objects['aa_source']; assert len(sources) == 146
    for name, expected in sources.items():
        assert sha((repo / name).read_bytes()) == expected, name
        git_body = subprocess.check_output(['git', 'show', s['head'] + ':' + name], cwd=repo)
        assert sha(git_body) == expected, name
    assert set(s['source_checks']) == set(sources) and all(s['source_checks'].values())
    split = objects['split']
    for name, field in [('pool','training_development'),('d3r','d3r'),('pilot','d4_pilot')]:
        assert s['files'][name]['sha256'] == split['source_hashes'][field]
    assert s['files']['train']['sha256'] == split['train_manifest_sha256']
    assert s['files']['development']['sha256'] == split['eval_manifest_sha256']
    partitions = {}
    for key in ('sample_id', 'relative_path', 'sha256'):
        role_sets = {role: set(item[key] for item in objects[role])
                     for role in ('pool', 'd3r', 'pilot', 'train', 'development')}
        # Count role membership separately for each original pool identity.
        occupancy = Counter()
        for item in role_sets['pool']:
            roles = tuple(role for role in ('d3r', 'pilot', 'train', 'development') if item in role_sets[role])
            occupancy[roles] += 1
        assert occupancy == {('d3r',):24, ('pilot',):12, ('train',):64, ('development',):32}
        assert set.union(*(role_sets[k] for k in ('d3r','pilot','train','development'))) == role_sets['pool']
        expected = dict(pool=132, excluded=36, train=64, development=32, eligible=96, unused=0, used_union_equals_eligible=True)
        assert r['partition'][key] == expected
        partitions[key] = {','.join(k):v for k,v in occupancy.items()}
    assert len(objects['train']) == 72
    assert not any(item.get(k) for item in objects['pool'] for k in ('patient_id','subject_id','study_id','patient_group_id'))
    assert r['pool_group_identifiers_available'] == 0 and split['patient_independence_verified'] is False
    assert r['patient_independence_verified'] is False
    br = s['br35h_entries']; assert len(br) == len({x['path'] for x in br}) == 3000
    assert Counter(Path(x['path']).parent.as_posix() for x in br) == {'test/good':1500,'test/ungood':1500}
    assert all(Path(x['path']).suffix.lower() == '.jpg' and not x['symlink'] for x in br)
    assert r['br35h']['files'] == 3000 and r['br35h']['nonimage_sidecars'] == []
    assert r['br35h']['source_and_group_separation'] == 'NOT_ESTABLISHED'
    assert r['br35h']['prior_use_history'] == 'NOT_CLOSED'
    assert r['br35h']['image_bodies_opened'] == 0
    assert r['br35h']['class_directory_counts'] == {'test/good':1500,'test/ungood':1500}
    assert r['execution_validity'] == 'VALID_METADATA_AUDIT'
    allowed_reads = {v['path'] for v in s['files'].values() if v['path'].startswith('/home/data/medic-ad/')}
    assert set(s['data_body_open_log']) == allowed_reads and len(s['data_body_open_log']) == 8
    identities = {v['sha256']:v for v in objects['development']}
    independently_measured = []
    assert {(x['seed'],x['arm']) for x in r['historical_development_rankings']} == {(seed,arm) for seed in (43,44) for arm in ('none','full','early')}
    for item in r['historical_development_rankings']:
        path = Path(item['source']); body = path.read_bytes(); assert sha(body) == item['sha256']
        # Bind each consumed prediction file to its original closed inventory.
        campaign = path.parents[1]; control = campaign.with_name(campaign.name.replace('-v1','-control-v1'))
        inv = json.loads((control / 'artifact-inventory-v1.json').read_text())
        rel = str(path.relative_to(campaign))
        entry = [x for x in inv['entries'] if x['root'] == campaign.name and x['path'] == rel]
        assert len(entry) == 1 and entry[0]['sha256'] == sha(body) and entry[0]['bytes'] == len(body)
        rows = json.loads(body); assert len(rows) == 32 and len({x['sha256'] for x in rows}) == 32
        assert {x['sha256'] for x in rows} == set(identities)
        assert all(x['label'] == identities[x['sha256']]['scout_label'] for x in rows)
        pos = [x['score'] for x in rows if x['label']=='abnormal']
        neg = [x['score'] for x in rows if x['label']=='normal']
        assert len(pos)==20 and len(neg)==12
        # Pairwise comparisons, rather than producer's grouped rank accumulation.
        auc = sum((Fraction(1) if p>n else Fraction(1,2) if p==n else Fraction(0)) for p in pos for n in neg)/240
        # Precision at each positive's complete threshold, not producer's loop.
        ap = sum((Fraction(sum(q>=p for q in pos), sum(x['score']>=p for x in rows)) for p in pos), Fraction(0))/20
        assert abs(float(auc)-item['auroc']) <= 1e-12
        assert abs(float(ap)-item['average_precision']) <= 1e-12
        prior = objects['aa_verification']['summaries'][str(item['seed'])][item['arm']]
        assert abs(float(auc)-prior['auroc']) <= 1e-12 and abs(float(ap)-prior['average_precision']) <= 1e-12
        independently_measured.append({'seed':item['seed'],'arm':item['arm'],'auroc_fraction':str(auc),'ap_fraction':str(ap)})
    assert r['scientific_decision'] == 'BLOCKED_INDEPENDENT_COHORT_NOT_ESTABLISHED'
    assert r['method_value_decision'] == 'UNDETERMINED'
    assert r['new_answer_effect'] == r['simple_baseline_effect'] == 'NOT_MEASURED'
    assert all(r[k] == 0 for k in ('new_independent_cases','new_development_predictions','model_loads','optimizer_updates','images_opened','protected_images_or_results_opened'))
    return {'verification':'PASS','scientific_decision':r['scientific_decision'],
            'snapshot_sha256':sha(Path(snapshot_path).read_bytes()),'producer_report_sha256':sha(Path(report_path).read_bytes()),
            'historical_sources_verified':146,'partitions_by_membership':partitions,
            'independent_development_reductions':independently_measured,
            'independence_scope':'Separate stdlib numerical implementation/process, not independent patients or an external laboratory.'}


def main():
    p=argparse.ArgumentParser()
    for field in ('workspace','snapshot','report','output'):p.add_argument('--'+field,type=Path,required=True)
    a=p.parse_args(); result=verify(a.workspace,a.snapshot,a.report)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(result['verification'],result['scientific_decision'])


if __name__=='__main__':main()

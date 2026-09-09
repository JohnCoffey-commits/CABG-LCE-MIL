"""Independent stdlib recount and math check; does not import readiness.py.

Checks source evidence supporting decisive blockers, not just producer flags.
This is a separate implementation, not an external expert or clinical review.
"""
import argparse
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re


def verify(root):
    def js(name): return json.loads((root / name).read_text())
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    report=js('readiness-report-v1.json'); registry=js('admission-register-v1.json')
    raw=root/'source-metadata-v1'
    receipts=[]
    for receipt in sorted(raw.glob('*.receipt.json')):
        item=json.loads(receipt.read_text()); body=Path(str(receipt).removesuffix('.receipt.json'))
        assert digest(body)==item['sha256'] and body.stat().st_size==item['bytes']
        assert item['assets_requested'] is False and item['url'].startswith('https://')
        receipts.append({'file':body.name,'returncode':item['returncode']})
    assert len(receipts)>=25
    assert registry['images_opened']==0 and registry['predictions_seen'] is False
    assert len(registry['candidates'])==8
    for candidate in registry['candidates']:
        gates=candidate['gates']
        assert set(gates)=={'rights','grouping','source_isolation','shown_2d_truth','rendering_and_members','precision_and_cost'}
        blocked=[]
        for key,value in gates.items():
            assert value['status'] in {'PASS','FAIL','UNKNOWN'} and value['reason'] and value['sources']
            for source in value['sources']:
                path=(root/source['path']).resolve()
                assert path.is_relative_to(root.resolve()) and digest(path)==source['sha256']
            if value['status']!='PASS':blocked.append(key)
        got=report['candidates'][candidate['id']]
        assert blocked and got['status']=='NOT_ADMITTED' and set(got['unresolved_gates'])==set(blocked)
    # Separate table reader/reduction, and reconstruct exact source inventory sets.
    lines=(raw/'ds001226-participants.tsv').read_text().splitlines()
    columns=lines[0].split('\t'); pos=columns.index('tumor type & grade')
    positives=set(); negatives=set()
    for line in lines[1:]:
        cells=line.split('\t'); (negatives if cells[pos]=='none' else positives).add(cells[0])
    assert len(positives)==25 and len(negatives)==11 and not positives & negatives
    tree=json.loads((raw/'ds001226-tree.json').read_text())
    assert not tree['truncated']
    paths={e['path'] for e in tree['tree']}
    for identity in positives|negatives:
        assert f'{identity}/ses-preop/anat/{identity}_ses-preop_T1w.nii.gz' in paths
    for identity in positives:
        assert f'derivatives/tumor_masks/{identity}/anat/{identity}_space_T1_label-tumor.nii' in paths
    facts=report['facts']; d=facts['ds001226']
    assert (d['participants'],d['positive'],d['negative'],d['t1_cases'],d['native_mask_cases'])==(36,25,11,36,25)
    assert d['tree_sha']==tree['sha']
    ogm=json.loads((raw/'ogm-tree.json').read_text()); assert not ogm['truncated']
    subject_names=[e['path'] for e in ogm['tree'] if e['type']=='tree' and '/' not in e['path'] and e['path'].startswith('sub-olf')]
    groups={re.sub('fu$','',name) for name in subject_names}
    rows=list(csv.DictReader(io.StringIO((raw/'ogm-pinned-participants.tsv').read_text()),delimiter='\t'))
    assert len(subject_names)==60 and len(groups)==37 and sum(r['follow_up']=='yes' for r in rows)==23
    assert facts['ogm']==dict(participant_rows=37,subject_folders=60,independent_groups=37,followup_folders=23)
    dv=json.loads((raw/'motum-dataverse.json').read_text())['data']['latestVersion']
    directories={f.get('directoryLabel','') for f in dv['files'] if '/images/' in f.get('directoryLabel','')}
    assert len(directories)==66 and all(not f['restricted'] for f in dv['files'] if '/images/' in f.get('directoryLabel',''))
    assert facts['motum']['image_groups']==66 and dv['license']['name']=='CC BY 4.0'
    categories={}
    for directory in directories:
        category=re.search(r'\d+-+(.*)-[^-]+$',directory.split('/')[-1])[1]
        categories[category]=categories.get(category,0)+1
    assert categories==facts['motum']['groups_by_category'] and sum(categories.values())==66
    assert facts['motum']['noncanonical_group_names']==['007--LungMeta-AYEY']
    # Decisive semantic evidence; these assertions make contrary source changes visible.
    assert 'no original/patient identifiers retained' in (raw/'bdneuro-pinned-readme.md').read_text()
    assert '[FILL IN' in (raw/'bdneuro-pinned-readme.md').read_text()
    plain=lambda name: re.sub(r'\s+',' ',html.unescape(re.sub('<[^>]+>',' ',(raw/name).read_text())))
    assert 'complete subject-level independence cannot be guaranteed' in plain('brisc2026.html')
    assert 'Br35H' in plain('brisc2026.html')
    assert 'May 2015 and October 2017' in plain('aerts2018.html')
    assert 'Data Sharing' in (raw/'fastmri-access.html').read_text()
    assert 'not be considered clinical ground truth or an exhaustive list' in (root/'inherited-source-evidence-v1/fastmri-plus-README.md').read_text()
    for version in (1,6,7):
        body=(raw/f'bdneuro-v{version}.html').read_text().split('window.INITIAL_STATE = ',1)[1]
        state=json.JSONDecoder().raw_decode(body)[0]['dataset']['snapshot']
        expected={k:state[k] for k in ('version','doi','publish_date','licence')}
        assert facts['bdneuro_versions'][str(version)]==expected
    # Inverse normal via erf bisection, independent of producer NormalDist.
    lo,hi=0.,5.
    for _ in range(100):
        mid=(lo+hi)/2
        if (1+math.erf(mid/math.sqrt(2)))/2 < .9875:lo=mid
        else:hi=mid
    z=(lo+hi)/2; checks=0
    for name,rows in report['precision_design_scenarios'].items():
        expected=(25,11) if name.startswith('ds001226') else (63,146)
        for row in rows:
            a,b=expected; assert (row['n_positive'],row['n_negative'])==expected and not row['observed']
            q=row['assumed_q']; half=z*math.sqrt(q*(1/a+1/b))/2
            bound=((1-.0125**(1/a))+(1-.0125**(1/b)))/2
            assert abs(half-row['approximate_half_width'])<1e-12
            assert abs(bound-row['zero_discordance_simultaneous_absolute_bound'])<1e-12
            assert abs(z-row['simultaneous_z'])<1e-12
            assert abs(4*a*b/(a+b)-row['effective_n'])<1e-12
            assert row['full_none_planning_screen']==(half<=.05)
            assert row['full_head_planning_screen']==(half<=.075)
            for n in expected:
                upper=1-.0125**(1/n); assert abs((1-upper)**n-.0125)<1e-12
            checks+=1
    assert checks==10
    assert report['execution']=='PASS' and report['scientific']=='NOT_MEASURED'
    assert report['decision']=='BLOCKED_INDEPENDENT_COHORT_READINESS' and report['independent_effect'] is None
    assert report['independent_predictions']==report['model_loads']==report['optimizer_updates']==0
    return dict(status='PASS_INDEPENDENT_METADATA_AND_PRECISION_CHECK', source_receipts_checked=len(receipts),
                failed_http_captures=[r for r in receipts if r['returncode']],
                unusable_captcha_body='ogm-paper.html (not evidence for admission; pinned README/TSV used)',
                candidate_decisions=8, independent_design_recomputations=checks,
                scientific='NOT_MEASURED',decision=report['decision'],
                verification_scope='Separate implementation and source check, same research operator; not independent clinical annotation',
                report_sha256=digest(root/'readiness-report-v1.json'), registry_sha256=digest(root/'admission-register-v1.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); result=verify(a.evidence)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

"""Read metadata and reviewed claims; never open MRI or instantiate a model.

Evidence status is a documented human scientific judgment, not machine proof of
permission or independence. This checker prevents missing/unknown gates from
being silently promoted and computes explicitly hypothetical precision screens.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from statistics import NormalDist

GATES = ('rights', 'grouping', 'source_isolation', 'shown_2d_truth',
         'rendering_and_members', 'precision_and_cost')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def gate_decision(candidate, evidence):
    if set(candidate['gates']) != set(GATES):
        raise ValueError('Exact gate set required')
    failed = []
    for name, gate in candidate['gates'].items():
        if gate['status'] not in ('PASS', 'FAIL', 'UNKNOWN') or not gate['reason'].strip():
            raise ValueError('Explicit evidence judgment required')
        if not gate['sources']:
            raise ValueError('Gate must cite source evidence, including unresolved gates')
        for source in gate['sources']:
            path = (evidence / source['path']).resolve()
            if not path.is_relative_to(evidence.resolve()) or sha(path) != source['sha256']:
                raise ValueError('Source escapes evidence root or hash mismatch')
        if gate['status'] != 'PASS':
            failed.append(name)
    if not failed:
        # This is only a reviewed readiness screen. No automatic GPU authorization.
        return 'REVIEWED_GATES_PASS_REQUIRES_EXECUTION_LOCK', failed
    return 'NOT_ADMITTED', failed


def precision(n_positive, n_negative, q):
    if type(n_positive) is not int or type(n_negative) is not int or min(n_positive, n_negative) < 1:
        raise ValueError('Positive independent class counts required')
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('Discordance scenario must be in [0,1]')
    neff = 4 / (1 / n_positive + 1 / n_negative)
    z = NormalDist().inv_cdf(1 - .05 / 4)  # Two 97.5% two-sided comparisons.
    half = z * math.sqrt(q / neff)
    # If every paired case difference is zero: |E[D]|<=P(D!=0).
    # Four one-sided bounds (two classes x two comparisons), union bound <=.05.
    zero = sum(-math.expm1(math.log(.05 / 4) / n) for n in (n_positive, n_negative)) / 2
    return dict(n_positive=n_positive, n_negative=n_negative, effective_n=neff,
                assumed_q=q, simultaneous_z=z, approximate_half_width=half,
                zero_discordance_simultaneous_absolute_bound=zero,
                full_none_planning_screen=half <= .05,
                full_head_planning_screen=half <= .075,
                observed=False)


def source_facts(root):
    rows = list(csv.DictReader(io.StringIO((root / 'ds001226-participants.tsv').read_text()), delimiter='\t'))
    ids = [r['participant_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate participant ID')
    tumor = {r['participant_id'] for r in rows if r['tumor type & grade'] != 'none'}
    tree = read(root / 'ds001226-tree.json')
    if tree.get('truncated') is not False:
        raise ValueError('Incomplete tree')
    paths = [e['path'] for e in tree['tree']]
    t1 = {p.split('/')[0] for p in paths if re.fullmatch(r'sub-[^/]+/ses-preop/anat/[^/]+_T1w.nii.gz', p)}
    masks = {p.split('/')[2] for p in paths if p.startswith('derivatives/tumor_masks/') and p.endswith('_space_T1_label-tumor.nii')}
    if t1 != set(ids) or masks != tumor:
        raise ValueError('T1/mask inventory does not cover participant list')
    ogm = read(root / 'ogm-tree.json')
    if ogm.get('truncated') is not False:
        raise ValueError('Incomplete OGM tree')
    folders = {e['path'] for e in ogm['tree'] if re.fullmatch(r'sub-olf\d+(fu)?', e['path'])}
    groups = {p.removesuffix('fu') for p in folders}
    ogm_rows = list(csv.DictReader(io.StringIO((root / 'ogm-pinned-participants.tsv').read_text()), delimiter='\t'))
    if groups != {r['participant_id'] for r in ogm_rows}:
        raise ValueError('OGM subject/followup mismatch')
    motum = read(root / 'motum-dataverse.json')['data']['latestVersion']
    group_dirs = {f['directoryLabel'].split('/')[-1] for f in motum['files']
                  if re.fullmatch(r'MOTUM-v.2.2/images/[^/]+', f.get('directoryLabel', ''))}
    bd = {}
    for v in (1, 6, 7):
        text = (root / f'bdneuro-v{v}.html').read_text()
        state = re.search(r'window.INITIAL_STATE\s*=\s*({.*?})\s*\n\s*window.initialTime', text, re.S)
        snapshot = json.loads(state[1])['dataset']['snapshot']
        bd[str(v)] = {k: snapshot[k] for k in ('version', 'doi', 'publish_date', 'licence')}
    return dict(ds001226=dict(participants=len(ids), positive=len(tumor), negative=len(ids)-len(tumor),
                              t1_cases=len(t1), native_mask_cases=len(masks), tree_sha=tree['sha']),
                ogm=dict(participant_rows=len(ogm_rows), subject_folders=len(folders),
                         independent_groups=len(groups), followup_folders=len(folders)-len(groups)),
                motum=dict(version=f"{motum['versionNumber']}.{motum['versionMinorNumber']}",
                           license=motum['license']['name'], image_groups=len(group_dirs),
                           groups_by_category=dict(Counter(p.split('-')[-2] for p in group_dirs)),
                           noncanonical_group_names=sorted(p for p in group_dirs if '--' in p)),
                bdneuro_versions=bd)


def audit(evidence):
    registry = read(evidence / 'admission-register-v1.json')
    facts = source_facts(evidence / 'source-metadata-v1')
    ds = facts['ds001226']
    design = {name: [precision(a, b, q) for q in (.05, .10, .20, .50, 1.)]
              for name, a, b in [('ds001226_all_cases_upper_ceiling', ds['positive'], ds['negative']),
                                 ('MMCBT_paper_counts_not_admitted', 63, 146)]}
    decisions = {}
    for c in registry['candidates']:
        if c['id'] in decisions:
            raise ValueError('Duplicate candidate')
        status, failed = gate_decision(c, evidence)
        decisions[c['id']] = dict(status=status, unresolved_gates=failed)
    return dict(execution='PASS', scientific='NOT_MEASURED', independent_effect=None,
                decision='BLOCKED_INDEPENDENT_COHORT_READINESS' if all(
                    v['status']=='NOT_ADMITTED' for v in decisions.values()) else 'ADMISSION_REVIEW_REQUIRED',
                candidates=decisions, facts=facts, precision_design_scenarios=design,
                independent_predictions=0, model_loads=0, optimizer_updates=0,
                evidence_scope='Public text metadata and documented judgments; no MRI volume or model output')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a=p.parse_args()
    result=audit(a.evidence); write(a.output, result)
    print(json.dumps({k: result[k] for k in ('execution','scientific','decision')}))

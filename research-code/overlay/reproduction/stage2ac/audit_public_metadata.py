"""Read-only independent counts/identity check of acquired public metadata, no MRI."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sources=[]
    for folder in ('public-metadata-v2','public-metadata-v3'):
        root=a.evidence/folder
        for r in json.loads((root/'sources.json').read_text()):
            assert r.get('exit',0)==0
            body=(root/r['name']).read_bytes()
            assert len(body)==r['bytes'] and hashlib.sha256(body).hexdigest()==r['sha256']
            sources.append(r)
    root=a.evidence/'public-metadata-v2';facts={}
    for name,column in [('ds001226','tumor type & grade'),('ds004910','group')]:
        rows=list(csv.DictReader((root/(name+'-participants.tsv')).open(),delimiter='\t'))
        ids=[r['participant_id'] for r in rows];assert len(set(ids))==len(ids)
        description=json.loads((root/(name+'-dataset_description.json')).read_text())
        assert description['License']=='CC0'
        facts[name]={'n_group_ids':len(ids),'labels':dict(collections.Counter(r[column] for r in rows)),
                     'license':description['License'],'dataset_doi':description['DatasetDOI'],
                     'metadata_columns':list(rows[0]),'cross_training_patient_separation_verified':False,
                     'two_dimensional_slice_labels_verified':False,'admitted':False}
    root=a.evidence/'public-metadata-v3'
    reviewed=[r[0] for r in csv.reader((root/'fastmri-plus-brain_file_list.csv').open())]
    assert len(reviewed)==len(set(reviewed))==1001  # This file is headerless.
    annotations=list(csv.DictReader((root/'fastmri-plus-brain.csv').open()));files={r['file'] for r in annotations}
    assert files<=set(reviewed) and len(annotations)==8213 and len(files)==997
    facts['fastmri_plus']={'reviewed_volumes':len(reviewed),'annotation_rows':len(annotations),
                          'annotated_volumes':len(files),'reviewed_without_annotation_rows':len(set(reviewed)-files),
                          'label_counts':dict(collections.Counter(r['label'] for r in annotations)),
                          'metadata_columns':list(annotations[0]),'raw_image_access_agreement_verified':False,
                          'cross_training_patient_separation_verified':False,'admitted':False}
    assert facts['ds001226']['n_group_ids']==36 and facts['ds001226']['labels']['none']==11
    assert facts['ds004910']['n_group_ids']==20 and facts['ds004910']['labels']['Health control']==2
    result={'status':'PASS_METADATA_COUNTS_AND_IDENTITIES','source_files':sources,'facts':facts,
            'independent_cohort_admitted':False,'independent_effect':'NOT_MEASURED',
            'scope':'Raw public tabular metadata only; source separation and permissions need their own evidence',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'metadata_files':len(sources),'cohorts_admitted':0}))


if __name__=='__main__':main()

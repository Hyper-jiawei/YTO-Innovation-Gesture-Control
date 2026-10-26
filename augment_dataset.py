"""Export inspectable training-only variants using an existing split manifest."""
import bootstrap
import argparse
import hashlib
import json
from pathlib import Path
from lg.labels import canonical_label,ACTIVE_DATASET
from lg.augmentation import expand_training


def main():
    p=argparse.ArgumentParser(description='Export training-only keypoint augmentation; original dataset unchanged')
    p.add_argument('--copies',type=int,default=4)
    p.add_argument('--data',type=Path,default=bootstrap.ROOT/ACTIVE_DATASET)
    p.add_argument('--report',type=Path,default=bootstrap.ROOT/'outputs/training_report.json')
    p.add_argument('--output',type=Path,default=bootstrap.ROOT/'outputs/augmented_training.jsonl')
    a=p.parse_args()
    if not 0<=a.copies<=20: p.error('--copies must be between 0 and 20')
    report=json.loads(a.report.read_text(encoding='utf-8'))
    source_hash=hashlib.sha256(a.data.read_bytes()).hexdigest()
    if report['data_sha256']!=source_hash:
        p.error('Dataset changed since training. Rerun training with augmentation to create a fresh split.')
    parts=report['indices']
    train=set(parts['train'])
    if train & (set(parts['validation'])|set(parts['test'])|set(parts['excluded'])):
        p.error('Split manifest contains overlapping indices')
    rows=[json.loads(line) for line in a.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows: row['label']=canonical_label(row['label'])
    vectors,_=expand_training(rows,parts['train'],a.copies,42,source_hash,a.output)
    print(f'Original training samples: {len(train)}; added: {len(train)*a.copies}; exported: {len(vectors)}')
    print(f'Written {a.output}; validation/test unchanged; raw dataset unchanged.')


if __name__=='__main__': main()

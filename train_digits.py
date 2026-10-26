"""GPU MLP, operator/session-held-out validation; only train split augmented."""
import bootstrap
import argparse
import json
import random
import hashlib
from pathlib import Path
import numpy as np
from lg.labels import LABELS,DIGIT_LABELS,LOVE_LABEL,canonical_label,ACTIVE_DATASET


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=bootstrap.ROOT/ACTIVE_DATASET)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--group-by',choices=['operator','session'],default='operator')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--allow-partial',action='store_true',help='允许部分数字，但必须含 I Love U')
    p.add_argument('--split',choices=['group','temporal'],default='group',help='temporal 为单人按类别/左右手时间分段的初步验证')
    p.add_argument('--output-dir',type=Path,default=bootstrap.ROOT,help='模型和报告根目录；默认本工程')
    p.add_argument('--augment-copies',type=int,default=0,help='每个训练原样本新增增强数量；不扩增验证/测试集')
    a=p.parse_args()
    if a.epochs < 1: p.error('--epochs must be positive')
    if not 0<=a.augment_copies<=20: p.error('--augment-copies must be between 0 and 20')
    if not a.data.exists(): p.error('No data yet. Run collect_digits.py and save samples first.')
    import torch
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import classification_report,confusion_matrix
    from lg.vision import cuda_device,network
    rows=[json.loads(line) for line in a.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows: row['label']=canonical_label(row['label'])
    labels=sorted({r['label'] for r in rows})
    required=set(LABELS)
    if not set(labels).issubset(required): raise ValueError(f'无效旧标签: {set(labels)-required}；当前只使用 1–9、I Love U、ok。')
    if a.allow_partial:
        if LOVE_LABEL not in labels or len(labels)<2 or not set(labels).issubset(set(LABELS)):
            raise ValueError('初步训练至少需要两个类别并包含 I Love U；支持 1–9、I Love U、ok。')
    elif not required.issubset(labels): raise ValueError(f'缺少类别: {required-set(labels)}')
    groups=np.array([r[a.group_by] for r in rows])
    x=np.asarray([r['features'] for r in rows],np.float32)
    if x.shape != (len(rows),63) or not np.isfinite(x).all():
        raise ValueError('数据必须是有限的 63 维 MediaPipe 关键点特征。')
    y=np.asarray([labels.index(r['label']) for r in rows],dtype=np.int64)
    source_hash=hashlib.sha256(a.data.read_bytes()).hexdigest()
    if a.split=='temporal':
        from lg.training import temporal_split
        train,val,test,excluded=temporal_split(rows)
    else:
        if len(set(groups))<3: raise ValueError('至少采集 3 个独立操作员/会话组；单人不完整轮次可显式用 --split temporal。')
        split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
        train,val=next(split.split(x,y,groups))
        test=None
        excluded=np.array([],dtype=np.int64)
    if set(y[train])!=set(range(len(labels))) or set(y[val])!=set(range(len(labels))):
        raise ValueError('训练/验证组未覆盖所有类别；每个操作员/会话应采齐 1–9、I Love U、ok。')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device='cpu' if a.cpu else cuda_device()
    model=network(torch,len(labels)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    counts=np.bincount(y[train],minlength=len(labels))
    weight=torch.tensor(len(train)/np.maximum(counts,1)/len(labels),dtype=torch.float32,device=device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=weight)
    if a.augment_copies:
        from lg.augmentation import expand_training
        augmented_path=a.output_dir/'outputs/augmented_training.jsonl'
        train_x,parents=expand_training(rows,train,a.augment_copies,42,source_hash,augmented_path)
        train_y=y[parents]
    else:
        train_x,train_y=x[train],y[train]
        augmented_path=None
    tx=torch.tensor(train_x,device=device); ty=torch.tensor(train_y,dtype=torch.long,device=device)
    vx=torch.tensor(x[val],device=device); vy=torch.tensor(y[val],dtype=torch.long,device=device)
    best=float('inf'); patience=0; best_state=None
    for epoch in range(a.epochs):
        model.train()
        order=torch.randperm(len(tx),device=device)
        for indices in order.split(256):
            # Noise augmentation only on training data; real viewpoint diversity must be collected.
            batch=tx[indices]+torch.randn_like(tx[indices])*.008
            loss=loss_fn(model(batch),ty[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss=loss_fn(model(vx),vy).item()
        if val_loss<best:
            best=val_loss; patience=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: patience+=1
        print(f'epoch={epoch+1} device={device} val_loss={val_loss:.4f}')
        if patience>=15: break
    if best_state is None: raise RuntimeError('Training failed')
    model.load_state_dict(best_state)
    evaluation=test if test is not None else val
    ex=torch.tensor(x[evaluation],device=device)
    with torch.inference_mode():
        probabilities=model(ex).softmax(-1)
        confidence,predicted=probabilities.max(-1)
        pred=predicted.cpu().numpy()
        confidence=confidence.cpu().numpy()
    hand_metrics={}
    for hand in ('left','right','unspecified'):
        mask=np.array([rows[i].get('physical_hand','unspecified')==hand for i in evaluation])
        if mask.any():
            hand_metrics[hand]=classification_report(y[evaluation][mask],pred[mask],labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0)
    report=dict(device=device,labels=labels,group_by=a.group_by,
                train_groups=sorted(set(groups[train])),validation_groups=sorted(set(groups[val])),
                report=classification_report(y[evaluation],pred,labels=list(range(len(labels))),target_names=labels,output_dict=True,zero_division=0),
                confusion_matrix=confusion_matrix(y[evaluation],pred,labels=list(range(len(labels)))).tolist(),
                per_physical_hand=hand_metrics,partial_vocabulary=a.allow_partial,
                hand_label_counts={hand:{label:sum(r.get('physical_hand','unspecified')==hand and r['label']==label for r in rows) for label in labels} for hand in ('left','right','unspecified')},
                missing_numeric_labels=sorted(set(DIGIT_LABELS)-set(labels)),
                split=a.split,evaluated_on='held_out_temporal_test' if test is not None else 'early_stopping_validation',
                train_count=len(train),validation_count=len(val),test_count=len(test) if test is not None else 0,excluded_count=len(excluded),
                data_sha256=source_hash,
                augmentation=dict(copies_per_training_sample=a.augment_copies,original_training_count=len(train),
                                  effective_training_count=len(train_x),added_count=len(train_x)-len(train),
                                  seed=42,export_path=str(augmented_path) if augmented_path else None,
                                  validation_augmented=False,test_augmented=False),
                validation_loss=best,
                indices=dict(train=train.tolist(),validation=val.tolist(),test=test.tolist() if test is not None else [],excluded=excluded.tolist()),
                confidence_threshold=.85,accepted_coverage=float((confidence>=.85).mean()),
                accepted_accuracy=float((pred[confidence>=.85]==y[evaluation][confidence>=.85]).mean()) if (confidence>=.85).any() else None,
                note='Temporal split is within the same operator/class/hand; not independent person/session validation. Test unused for early stopping.' if test is not None else 'Validation set used for early stopping; independent acceptance set still required.')
    (a.output_dir/'models').mkdir(exist_ok=True,parents=True)
    (a.output_dir/'outputs').mkdir(exist_ok=True,parents=True)
    torch.save(dict(labels=labels,state_dict=best_state,confidence_threshold=.85),a.output_dir/'models/digits.pt')
    (a.output_dir/'outputs/training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {a.output_dir}/models/digits.pt; restart main.py to load. Independent test required.')
    print(f"Evaluation: accuracy={report['report']['accuracy']:.4f}, macro_F1={report['report']['macro avg']['f1-score']:.4f}, coverage@0.85={report['accepted_coverage']:.4f}")

if __name__=='__main__': main()

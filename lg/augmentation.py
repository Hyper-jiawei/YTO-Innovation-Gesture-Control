"""Bounded landmark augmentation with traceable training-only parents.

Inspired by processdata/addata.py rotations/noise and handpose_x geometric
transforms, with optional mirroring for shared labels. This is not image
augmentation or palm/back rendering.
"""
import json
from pathlib import Path
import numpy as np
from .core import features


def augment_landmarks(vector,rng):
    p=np.asarray(vector,dtype=np.float32).reshape(21,3).copy()
    p-=p[0].copy()
    mirrored=bool(rng.random()<.5)
    if mirrored: p[:,0]*=-1
    z,x,y=np.deg2rad(rng.uniform([-12,-5,-5],[12,5,5]))
    rz=np.array([[np.cos(z),-np.sin(z),0],[np.sin(z),np.cos(z),0],[0,0,1]])
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    p=p@(rz@rx@ry).T
    noise=np.clip(rng.normal(0,.004,p.shape),-.012,.012)
    noise[0]=0
    p+=noise
    return features(p),dict(mirrored=mirrored,z_degrees=float(np.rad2deg(z)),
                            x_degrees=float(np.rad2deg(x)),y_degrees=float(np.rad2deg(y)),noise_std=.004)


def expand_training(rows,indices,copies,seed,source_hash,export_path=None):
    if copies<0: raise ValueError('copies must not be negative')
    rng=np.random.default_rng(seed)
    vectors,parents=[],[]
    path=Path(export_path) if export_path else None
    if path: path.parent.mkdir(parents=True,exist_ok=True)
    out=path.open('w',encoding='utf-8') if path else None
    try:
        for idx in indices:
            idx=int(idx)
            row=rows[idx]
            for variant in range(copies+1):
                if variant:
                    vector,transform=augment_landmarks(row['features'],rng)
                else:
                    vector=np.asarray(row['features'],dtype=np.float32)
                    transform=dict(original=True)
                vectors.append(vector)
                parents.append(idx)
                if out:
                    # Export feature vectors only. Synthetic vectors are NOT camera points.
                    record=dict(label=row['label'],operator=row.get('operator'),session=row.get('session'),
                        source_physical_hand=row.get('physical_hand'),source_index=idx,variant=variant,
                        split='train',source_sha256=source_hash,seed=seed,features=vector.tolist(),transform=transform)
                    out.write(json.dumps(record,ensure_ascii=False)+'\n')
    finally:
        if out: out.close()
    return np.asarray(vectors,dtype=np.float32),np.asarray(parents,dtype=np.int64)

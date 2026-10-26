"""Explicit temporal blocks for a single operator's incomplete collection sessions."""
import numpy as np
from collections import defaultdict


def temporal_split(rows):
    buckets=defaultdict(list)
    for i,row in enumerate(rows):
        buckets[(row['label'],row.get('physical_hand','unspecified'))].append(i)
    train,validation,test,excluded=[],[],[],[]
    for key,indices in sorted(buckets.items()):
        indices=sorted(indices,key=lambda i:rows[i].get('time',i))
        n=len(indices)
        if n<20:
            raise ValueError(f'{key}: temporal split requires at least 20 samples per class/hand')
        a,b=int(n*.6),int(n*.8)
        gap=max(1,int(n*.05))
        train.extend(indices[:a])
        excluded.extend(indices[a:a+gap])
        validation.extend(indices[a+gap:b-gap])
        excluded.extend(indices[b-gap:b])
        test.extend(indices[b:])
    return tuple(np.array(v,dtype=np.int64) for v in (train,validation,test,excluded))

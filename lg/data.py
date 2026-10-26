"""One key press, one labelled MediaPipe observation. Never infer ground truth."""
import json
import time
from collections import Counter
from pathlib import Path
from .core import features
from .labels import LABELS, canonical_label


def sample_counts(path):
    counts = Counter()
    if Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                counts[(row.get('physical_hand','unspecified'),canonical_label(row['label']))] += 1
    return counts


def save_snapshot(path, hands, label, operator, session, physical_hand, side='both',capture='single_keypress'):
    if physical_hand not in ('left','right'):
        raise ValueError('Choose physical hand: L=LEFT or R=RIGHT first')
    selected = [h for h in hands if side == 'both' or
                ('left' if h.points[0,0] < .5 else 'right') == side]
    if len(selected) != 1:
        raise ValueError(f'Need exactly ONE visible selected hand; found {len(selected)}')
    label = canonical_label(label)
    if label not in LABELS:
        raise ValueError('Invalid label')
    h = selected[0]
    row = dict(operator=operator,session=session,label=label,physical_hand=physical_hand,
               side='left' if h.points[0,0] < .5 else 'right',time=time.time(),
               points=h.points.tolist(),features=features(h.points).tolist(),capture=capture)
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as out:
        out.write(json.dumps(row)+'\n')
        out.flush()
    return row

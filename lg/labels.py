"""Shared collected/trained gesture vocabulary."""
LOVE_LABEL = 'I Love U'
DIGIT_LABELS = [str(i) for i in range(1,10)]
LABELS = DIGIT_LABELS + [LOVE_LABEL,'ok']
FIST_DIGIT = 9
ACTIVE_DATASET = 'data/current/digits.jsonl'


def canonical_label(label):
    return LOVE_LABEL if str(label) in ('unknown','unknow',LOVE_LABEL) else str(label)

"""Restore original semantic training families alongside later compositions.

Successive random replays reduced the original 64 positive families from
24,000 rows to about 3,000. This version restores them without reading
held-out prediction errors or replacing either frozen challenge.
"""
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from .data import read_jsonl, write_jsonl
from .semantic_data import TRAIN, NONE_TRAIN, VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST


def generate(directory='data/neural_v12',seed=1961):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        raise ValueError('数据版本已存在，请使用新目录')
    rng=random.Random(seed)
    training_frames=set(TRAIN+NONE_TRAIN)
    rows=[r for r in read_jsonl('data/semantic/train.jsonl') if r.get('template') in training_frames]
    replay=read_jsonl('data/neural_v11/train.jsonl');rng.shuffle(replay)
    selected_ids={r['id'] for r in rows}
    for group,limit in [('incidental',20000),('clean',12000),('noisy',14000),('other_semantic',15000)]:
        def matches(r):
            category=r.get('category','')
            if group=='incidental':return r['id'].startswith('incidental_')
            if group=='clean':return category.startswith('basic')
            if group=='noisy':return not category.startswith(('basic','semantic'))
            return category.startswith('semantic') and not r['id'].startswith('incidental_') and r.get('template') not in training_frames
        addition=[r for r in replay if r['id'] not in selected_ids and matches(r)][:limit]
        rows.extend(addition);selected_ids.update(r['id'] for r in addition)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic')
              for s in ('challenge','validation','calibration') for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    reserved.update(r['text'] for r in read_jsonl('examples/context_role_probe.jsonl'))
    assert all(r['text'] not in reserved and r.get('template') not in reserved_frames for r in rows)
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    for s in ('validation','calibration'):
        write_jsonl(dest/(s+'.jsonl'),read_jsonl(Path('data/neural_v11')/(s+'.jsonl')))
    families=Counter(r.get('template') for r in rows)
    report={'rows':len(rows),'seed':seed,'strategy':'restore all original semantic training families alongside incidental, basic and noisy replay',
            'original_positive_family_rows':sum(families[t] for t in TRAIN),
            'original_positive_family_min_rows':min(families[t] for t in TRAIN),
            'original_negative_family_rows':sum(families[t] for t in NONE_TRAIN),
            'no_argument_rows':sum(not r['spans'] for r in rows),
            'categories':dict(Counter(r.get('category',r.get('operation','unknown')) for r in rows)),
            'test_error_mining':False,'full_reserved_frames_excluded':True,'abstract_linguistic_parts_overlap':True,
            'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

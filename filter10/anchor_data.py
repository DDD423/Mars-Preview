"""Clean short-command replay to prevent regression hidden by noisy tests."""
import hashlib
import json
import random
from pathlib import Path
from .data import FAMILIES,SLOT_RE,QUOTES,NO_VALUE,render,read_jsonl,write_jsonl
from .noisy_data import literal
from .semantic_data import TRAIN,VALIDATION,CALIBRATION,TEST,NONE_VAL,NONE_CAL,NONE_TEST


def generate(directory='data/neural_v4',count=14000,seed=1109):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic') for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    frames=[t for group in FAMILIES.values() for t in group[:8]]+TRAIN
    rows=[]
    for i in range(count):
        template=rng.choice(NO_VALUE if i%10==0 else frames)
        if template in reserved_frames:continue
        vals={m.group(2):literal(rng,m.group(1),'clean_training',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.2 else ('','') for k in vals}
        text,spans=render(template,vals,quotes)
        row={'id':f'clean_{i:06}','text':text,'spans':spans,'template':template,
             'category':'basic_anchor','source':'semantic_augmentation'}
        if len(text)<=256 and text not in reserved:rows.append(row)
    rows += [dict(r,category='basic_anchor',source='semantic_augmentation') for r in read_jsonl('data/train.jsonl')
             if r['text'] not in reserved and r.get('template') not in reserved_frames]
    replay=read_jsonl('data/neural_v3/train.jsonl');rng.shuffle(replay)
    rows += [r for r in replay if not r.get('category','').startswith('semantic')][:24000]
    rows += [r for r in replay if r.get('category','').startswith('semantic')][:40000]
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    training_texts={r['text'] for r in rows}
    challenge_texts={r['text'] for d in ('data/noisy','data/semantic')
                     for r in read_jsonl(Path(d)/'challenge.jsonl')}
    for s in ('validation','calibration'):
        combined=read_jsonl(Path('data/neural_v3')/(s+'.jsonl'))
        core=[dict(r,category='basic_validation') for r in read_jsonl(Path('data')/(s+'.jsonl'))
              if r['text'] not in training_texts and r['text'] not in challenge_texts]
        write_jsonl(dest/(s+'.jsonl'),combined+core)
    report={'rows':len(rows),'seed':seed,'strategy':'short clean commands and original training replay, plus semantic/noisy composition',
        'test_error_mining':False,'handwritten_examples_are_development_only':True,
        'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

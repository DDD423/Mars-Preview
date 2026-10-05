"""Training contexts with role-independent trailing remarks.

Development diagnostics found type scores sensitive to unrelated tails.
The frozen challenges, their errors and inference grammar are not changed.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl
from .noisy_data import literal
from .context_algebra import declaration, role
from .constituent_data import frame as argument_frame
from .semantic_data import TRAIN, NONE_TRAIN, VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST

TAILS = ['', '，先不要执行', '，请原样保留', '，等我确认后再处理',
         '，麻烦先记下来', '，其余部分保持不变', '，先做预览', '，之后再决定操作',
         '，读取即可', '，谢谢', '，暂时不要修改', '，先检查一下',
         '；以上只是这次给出的参数', '，不要和操作的名字混淆']


def frame(rng):
    kind = rng.choice(['NAME', 'PATH', 'TEXT', 'VALUE'])
    branch = rng.randrange(10)
    if branch < 5:
        # A short direct role declaration receives most of the single-value
        # supervision, rather than sparsely sampling a large grammar product.
        word = role(rng, kind, natural=True)
        x = '[[' + kind + ':a]]'
        return rng.choice([x + '是' + word, x + '给的是' + word,
                           word + '是' + x, word + '写作' + x,
                           '用' + x + '表示' + word]) + rng.choice(TAILS)
    if branch < 7:
        return declaration(rng, kind, 'a', True) + '；' + declaration(rng, rng.choice(['NAME','PATH','TEXT','VALUE']), 'b', True) + rng.choice(TAILS)
    if branch < 9:
        return (rng.choice(TRAIN) if rng.random() < .5 else argument_frame(rng)) + rng.choice(TAILS)
    return rng.choice(NONE_TRAIN + ['暂时只说参数类型，没有给出实际内容',
                                   '还缺少实际标识，别自行补充'])


def generate(directory='data/neural_v10', count=36000, seed=1733):
    dest = Path(directory); dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reserved_frames = set(VALIDATION + CALIBRATION + TEST + NONE_VAL + NONE_CAL + NONE_TEST)
    reserved = {r['text'] for d in ('data/noisy','data/semantic')
                for s in ('challenge','validation','calibration')
                for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    reserved.update(r['text'] for r in read_jsonl('examples/context_role_probe.jsonl'))
    rows=[]
    for i in range(count):
        template=frame(rng)
        if template in reserved_frames: continue
        values={m.group(2):literal(rng,m.group(1),'neutral_training',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.25 else ('','') for k in values}
        text,spans=render(template,values,quotes)
        prefix=rng.choice(['','','嗯，','麻烦你，'])
        text=prefix+text
        if not 0<len(text)<=256 or text in reserved: continue
        rows.append({'id':f'neutral_{i:06}','text':text,
                     'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
                     'template':template,'category':'semantic_neutral_tail','source':'semantic_augmentation'})
    replay=read_jsonl('data/neural_v7/train.jsonl');rng.shuffle(replay)
    for group,limit in [('clean',12000),('semantic',24000),('noisy',15000)]:
        def matches(r):
            c=r.get('category','')
            return ('semantic' if c.startswith('semantic') else 'clean' if c.startswith('basic') else 'noisy')==group
        rows.extend([r for r in replay if matches(r)][:limit])
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    for s in ('validation','calibration'):
        write_jsonl(dest/(s+'.jsonl'),read_jsonl(Path('data/neural_v7')/(s+'.jsonl')))
    report={'rows':len(rows),'seed':seed,'strategy':'role-independent tails and concentrated direct declarations; replay',
            'test_error_mining':False,'full_reserved_frames_excluded':True,'abstract_linguistic_parts_overlap':True,
            'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

"""Natural, training-only role relations. No challenge errors are read.

The role words are control text, not a dictionary of possible parameter values.
All literal values are filled afterwards and remain opaque to inference.
"""
import hashlib
import json
import random
from pathlib import Path

from .data import QUOTES, SLOT_RE, read_jsonl, render, write_jsonl
from .noisy_data import literal
from .semantic_data import VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST


def frame(rng):
    a, b = '[[NAME:a]]', '[[NAME:b]]'
    p, q = '[[PATH:a]]', '[[PATH:b]]'
    x, y = '[[VALUE:a]]', '[[VALUE:b]]'
    t = '[[TEXT:b]]'
    noun = rng.choice(['文件', '文档', '目录', '文件夹'])
    old = rng.choice(['旧名称', '原名称', '原名字', '原本的名称', '之前的名字', '目前的名字'])
    new = rng.choice(['新名称', '新的名称', '新名字', '改完后的名字', '接下来使用的名字'])
    op = rng.randrange(9)
    if op == 0:
        return rng.choice([
            f'{noun}的{old}是{a}，{new}选{b}，只改名字',
            f'{a}是{noun}的{old}，把{new}设为{b}',
            f'{noun}现在名叫{a}，改名后请叫{b}',
            f'{new}打算用{b}，要改的{noun}原名叫{a}',
            f'原来的{noun}名称写作{a}；请把名字更新为{b}',
            f'请将{noun}从原名称{a}改成新名称{b}',
        ])
    if op == 1:
        return rng.choice([
            f'用{b}这个新名称取代{noun}现在使用的名称{a}',
            f'请采用{b}这个名字来替换{noun}旧名称{a}',
            f'将{b}选作新名，原来的{noun}叫{a}',
            f'把新名称{b}用于原名为{a}的{noun}',
            f'请用{b}来替代{noun}原本的名字{a}',
            f'原有名称{a}要被新名称{b}替换掉',
        ])
    if op == 2:
        return rng.choice([
            f'{noun}的完整路径是{p}，这里只给地址，不提供名字',
            f'完整地址请采用{p}；文件名未提供，请读取这个位置',
            f'{p}是完整地址，文件名另说，先打开这个位置',
            f'请在完整路径{p}读取文件，不需要另外给文件名',
            f'要用的完整文件地址为{p}，名称还未确定',
            f'先打开文件；它的完整地址写为{p}，名字暂不指定',
        ])
    if op == 3:
        return rng.choice([
            f'源位置的完整地址为{p}，目的位置的完整地址为{q}，复制过去',
            f'{p}是原文件完整路径，{q}是副本完整路径',
            f'复制的来源完整路径用{p}，接收的完整路径用{q}',
            f'原始完整地址{p}要改到新的完整地址{q}',
        ])
    if op == 4:
        return rng.choice([
            f'标识{x}用于来源，标识{y}用于目的地，两者类型都未说明',
            f'{x}是源参数，{y}是接收参数；尚未提供参数类别',
            f'传入的源对象表示为{x}，目标对象表示为{y}，类型仍然未知',
            f'来源标识给定为{x}而目的标识给定为{y}，先不区分名称和地址',
            f'先给出源标识{x}，再给出目标标识{y}，二者的类型待确认',
        ])
    if op == 5:
        return rng.choice([
            f'{noun}名称用{a}，要加进去的原文用{t}',
            f'给名称为{a}的文档加入{t}这段正文',
            f'{t}是需要添加的正文，接收的文档名称是{a}',
            f'要搜索的原文写作{t}，检索范围是叫{a}的文档',
            f'文档名是{a}，搜索需要包含{t}这段文字的地方',
        ])
    if op == 6:
        return rng.choice([
            f'{a}是这个{noun}当前使用的名字，先读取',
            f'打开这个{noun}，它现在的名称写作{a}',
            f'{a}这个名称用于指定要打开的{noun}',
            f'只读取名叫{a}的{noun}，别改名字或正文',
            f'我说的名称是{a}，对应的{noun}只需查看',
        ])
    if op == 7:
        return rng.choice([
            f'这里没有给出{noun}名称，只是在解释原名和新名的区别',
            '原来和现在都是时间说明，未给出实际名称',
            '先不要把动作名称当成文件名，参数还没提供',
            '完整地址和名称是不同的信息，现在两者都没有给出',
            '来源和目的地都未指定，不要自己补参数',
            '正文的类型是文字，但还没提供实际正文',
        ])
    return rng.choice([
        f'参数标识给定为{x}，类型信息还缺少',
        f'{x}用来标识目前的操作对象，名称或路径尚不区分',
        f'给出的目标标识为{x}，对象类型需要之后再确认',
    ])


def generate(directory='data/neural_v7', count=30000, seed=1459):
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # Read held-out input strings only to prevent accidental exact duplicates.
    # Their outputs/errors do not choose any training expression.
    reserved_frames = set(VALIDATION + CALIBRATION + TEST + NONE_VAL + NONE_CAL + NONE_TEST)
    reserved = {r['text'] for d in ('data/noisy', 'data/semantic')
                for s in ('challenge', 'validation', 'calibration')
                for r in read_jsonl(Path(d) / (s + '.jsonl'))}
    rows = []
    for i in range(count):
        template = frame(rng)
        if template in reserved_frames:
            continue
        values = {m.group(2): literal(rng, m.group(1), 'relation_training', i, True)
                  for m in SLOT_RE.finditer(template)}
        quotes = {k: rng.choice(QUOTES[1:]) if rng.random() < .35 else ('', '') for k in values}
        text, spans = render(template, values, quotes)
        prefix = rng.choice(['', '', '麻烦你，', '我想这样：', '嗯，'])
        text = prefix + text
        if len(text) > 256 or text in reserved:
            continue
        rows.append({'id': f'relation_{i:06}', 'text': text,
                     'spans': [dict(s, start=s['start'] + len(prefix), end=s['end'] + len(prefix)) for s in spans],
                     'template': template, 'category': 'semantic_relation', 'source': 'semantic_augmentation'})
    replay = read_jsonl('data/neural_v5/train.jsonl')
    rng.shuffle(replay)
    for group, limit in [('clean', 12000), ('semantic', 18000), ('noisy', 15000)]:
        def matches(row):
            cat = row.get('category', '')
            return ('semantic' if cat.startswith('semantic') else 'clean' if cat.startswith('basic') else 'noisy') == group
        rows.extend([r for r in replay if matches(r)][:limit])
    rng.shuffle(rows)
    write_jsonl(dest / 'train.jsonl', rows)
    for split in ('validation', 'calibration'):
        write_jsonl(dest / (split + '.jsonl'), read_jsonl(Path('data/neural_v5') / (split + '.jsonl')))
    report = {'rows': len(rows), 'seed': seed, 'strategy': 'natural role relations and multi-argument supervision; development-driven',
              'test_error_mining': False, 'full_reserved_frames_excluded': True, 'abstract_linguistic_parts_overlap': True,
              'sha256': {s: hashlib.sha256((dest / (s + '.jsonl')).read_bytes()).hexdigest()
                         for s in ('train', 'validation', 'calibration')}}
    (dest / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

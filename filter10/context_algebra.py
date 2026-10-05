"""Compose grammatical role phrases rather than blindly replacing words.

Training expressions only; there is no corresponding inference grammar.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import QUOTES, SLOT_RE, read_jsonl, render, write_jsonl
from .noisy_data import literal
from .semantic_data import VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST


def role(rng, kind, natural=False):
    if kind == 'NAME':
        owner = rng.choice(['文件', '文档', '目录', '文件夹', '这份文件', '这个文档'] + (['这份文档', '那个目录'] if natural else []))
        temporal = rng.choice(['', '原来', '现在', '之前', '新', '目前', '现有'])
        if natural:
            owned = owner + '的' if not temporal else owner + '的新' if temporal == '新' else owner + temporal + '的'
            qualifier = temporal if temporal in ('', '新') else temporal + '的'
            return rng.choice([owned + rng.choice(['名称', '名字']),
                               qualifier + rng.choice(['文件名', '文档名', '文件夹名称'])])
        if temporal in ('原来', '现在', '之前', '目前', '现有'):
            temporal += '的'
        return rng.choice([owner + '的' + temporal + rng.choice(['名称', '名字']),
                           temporal + rng.choice(['文件名', '文档名', '文件夹名称'])])
    if kind == 'PATH':
        owner = rng.choice(['文件', '文档', '目录', '文件夹', '源文件', '目的文件', '这份文件'])
        return rng.choice([owner + '的完整' + rng.choice(['路径', '地址']),
                           rng.choice(['', '来源', '目标', '接收位置的']) + rng.choice(['完整路径', '完整地址'])])
    if kind == 'TEXT':
        task = rng.choice(['检索', '搜索', '查找', '匹配', '写入', '添加', '补充', '追加'])
        return rng.choice([rng.choice(['要', '需要', '准备']) + task + '的' + rng.choice(['原文', '正文', '文字']),
                           rng.choice(['', '新增的', '本次提供的']) + rng.choice(['正文', '搜索词', '原文'])])
    return rng.choice(['类型未确定的参数', '未知类型的目标标识', '本次传入的对象标识',
                       '待确认类别的标识', '操作目标标识', '尚未分类的输入参数'])


def declaration(rng, kind, key, natural=False):
    r = role(rng, kind, natural)
    x = '[[' + kind + ':' + key + ']]'
    return rng.choice([
        x + rng.choice(['是', '正是', '给出的是', '表示的是', '提供的是', '指的是']) + r,
        r + rng.choice(['是', '写作', '设为', '确定为', '采用', '用']) + x,
        r + rng.choice(['由', '用']) + x + rng.choice(['给出', '表示', '确定']),
        '用' + x + rng.choice(['表示', '指定', '作为']) + r,
        x + '这个参数对应的是' + r,
    ])


def frame(rng, natural=False):
    kinds = ['NAME', 'PATH', 'TEXT', 'VALUE']
    kind = rng.choice(kinds)
    follow = {'NAME': ['只读取它对应的文档', '只查看，不改名', '不是操作名称'],
              'PATH': ['读取这个位置', '先预览该位置的文件', '不另行提供名称'],
              'TEXT': ['保留原文的每个字符', '不作为文件名', '先在当前文档里处理'],
              'VALUE': ['暂时不区分名字和路径', '具体类别还没说明', '之后再补充类型信息']}
    branch = rng.randrange(10)
    if branch < 5:
        return declaration(rng, kind, 'a', natural) + rng.choice(['，', '；', '。']) + rng.choice(follow[kind])
    if branch < 8:
        other = rng.choice(kinds)
        return declaration(rng, kind, 'a', natural) + rng.choice(['，另外，', '；然后，', '，']) + declaration(rng, other, 'b', natural) + '，请原样记录'
    if branch == 8:
        third = rng.choice(kinds)
        return declaration(rng, kind, 'a', natural) + '；' + declaration(rng, rng.choice(kinds), 'b', natural) + '；' + declaration(rng, third, 'c', natural)
    return rng.choice(['目前只在解释' + role(rng, kind, natural) + '的概念，没有给出具体的值',
                       '这里的原来和现在不是名称，实际参数仍然缺少',
                       '不要猜测标识，用户还未说明实际要用哪一个',
                       '名称和正文还没有给出，先保持现状'])


def generate(directory='data/neural_v8', count=40000, seed=1543, natural=False):
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reserved_frames = set(VALIDATION + CALIBRATION + TEST + NONE_VAL + NONE_CAL + NONE_TEST)
    reserved = {r['text'] for d in ('data/noisy', 'data/semantic')
                for s in ('challenge', 'validation', 'calibration')
                for r in read_jsonl(Path(d) / (s + '.jsonl'))}
    reserved.update(r['text'] for r in read_jsonl('examples/context_role_probe.jsonl'))
    rows = []
    for i in range(count):
        template = frame(rng, natural)
        if template in reserved_frames:
            continue
        values = {m.group(2): literal(rng, m.group(1), 'context_training', i, True) for m in SLOT_RE.finditer(template)}
        quotes = {k: rng.choice(QUOTES[1:]) if rng.random() < .35 else ('', '') for k in values}
        text, spans = render(template, values, quotes)
        prefix = rng.choice(['', '', '嗯，', '麻烦你，', '我的意思是：'])
        text = prefix + text
        if not 0 < len(text) <= 256 or text in reserved:
            continue
        rows.append({'id': f'context_{i:06}', 'text': text,
                     'spans': [dict(s, start=s['start'] + len(prefix), end=s['end'] + len(prefix)) for s in spans],
                     'template': template, 'category': 'semantic_context_algebra', 'source': 'semantic_augmentation'})
    replay = read_jsonl('data/neural_v7/train.jsonl')
    rng.shuffle(replay)
    for group, limit in [('clean', 12000), ('semantic', 24000), ('noisy', 15000)]:
        def matches(r):
            category = r.get('category', '')
            return ('semantic' if category.startswith('semantic') else 'clean' if category.startswith('basic') else 'noisy') == group
        rows.extend([r for r in replay if matches(r)][:limit])
    rng.shuffle(rows)
    write_jsonl(dest / 'train.jsonl', rows)
    for split in ('validation', 'calibration'):
        write_jsonl(dest / (split + '.jsonl'), read_jsonl(Path('data/neural_v7') / (split + '.jsonl')))
    report = {'rows': len(rows), 'seed': seed, 'natural_possessive_phrases': natural,
              'strategy': 'grammatical role phrase composition; varied declaration and argument count; replay',
              'test_error_mining': False, 'full_reserved_frames_excluded': True, 'abstract_linguistic_parts_overlap': True,
              'sha256': {s: hashlib.sha256((dest / (s + '.jsonl')).read_bytes()).hexdigest() for s in ('train', 'validation', 'calibration')}}
    (dest / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report

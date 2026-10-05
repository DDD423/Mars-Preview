"""Training-only role declarations with independent missing-field remarks.

All variants are generated before rendering opaque payloads. These templates
are never consulted by inference. Reserved frames/texts are exclusion-only.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from .data import QUOTES, SLOT_RE, read_jsonl, render, write_jsonl
from .noisy_data import literal
from .semantic_data import VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST

ROLES = {
    'NAME': ['文件名称', '文档名字', '文件夹名称', '目录名字', '这份文件的名字', '当前文档的名称'],
    'PATH': ['文档的完整路径', '文件的完整地址', '目标文件路径', '来源文件地址', '该目录的路径', '接收位置的完整地址'],
    'TEXT': ['需要检索的原文', '待搜索的文字', '写进去的正文', '准备追加的原文', '查找时匹配的文本', '需要写入的内容'],
    'VALUE': ['尚未分类的参数', '类型未知的目标标识', '类别待确认的输入值', '类型暂未说明的对象标识'],
}
REMARKS = ['（原文别改）', '（先不用执行）', '（只在当前工作区）', '（这不是另一个参数）',
           '（其他项稍后再说）', '（先记住这一项）', '（别替我猜其他项）']
TAILS = ['', '，先按原文记录', '，下一步再讨论怎么操作', '，先别执行', '，麻烦原样保留']
PREFIXES = ['', '', '嗯，', '我换个说法，', '麻烦先听清楚：', '这次只想说明参数：']


def declaration(rng, kind, key):
    x = '[[' + kind + ':' + key + ']]'
    role = rng.choice(ROLES[kind])
    note = rng.choice(REMARKS)
    return rng.choice([
        role + rng.choice(['用', '选', '指定为', '给定为', '先记为', '填写成']) + x,
        x + rng.choice(['是', '对应的是', '作为', '表示']) + role,
        x + note + rng.choice(['是', '对应的是', '表示']) + role,
        role + rng.choice(['由', '用']) + x + rng.choice(['表示', '给出', '确定']),
        '这一项先填' + x + '，说的是' + role,
        x + '应填写在【' + role + '】这一项',
        '关于' + role + '，我提供的是' + x,
    ])


def missing(rng, kind, other=True):
    role = rng.choice(ROLES[kind])
    prefix = rng.choice(['另一项的', '其余对象的', '另外一份文档的', '']) if other else ''
    return prefix + role + rng.choice(['还未提供', '先不指定', '目前不知道', '稍后再给', '暂时缺少'])


def frame(rng):
    kinds = tuple(ROLES)
    branch = rng.randrange(10)
    if branch < 3:
        kind = rng.choice(kinds)
        other = rng.choice([k for k in kinds if k != kind])
        item = declaration(rng, kind, 'a')
        absent = missing(rng, other)
        return rng.choice([item + '；' + absent, absent + '；但' + item,
                           item + '，至于' + absent, '虽然' + absent + '，不过' + item])
    if branch < 5:
        count = rng.choice([2, 2, 3])
        declared = [declaration(rng, rng.choice(kinds), chr(ord('a') + i)) for i in range(count)]
        separator = rng.choice(['；另外，', '。接着，', '；', '，然后，'])
        return separator.join(declared) + rng.choice(['', '；' + missing(rng, rng.choice(kinds))])
    if branch < 7:
        return declaration(rng, rng.choice(kinds), 'a') + rng.choice(['，其他参数暂缺', '，先不要改动其他项', ''])
    if branch == 7:
        # Two explicit names with comments and word order independent of
        # whether an operation is currently permitted.
        a, b = '[[NAME:a]]', '[[NAME:b]]'
        return rng.choice([
            '新的称呼先选' + b + '，而' + a + rng.choice(REMARKS) + '是现在的文件名字',
            '文档现在叫' + a + rng.choice(REMARKS) + '，想改成的名称是' + b,
            a + '是之前用的文件名，' + b + '是接下来要用的文件名',
            '别马上操作，' + a + '这个文件名将换为' + b + '，其他项还没给',
        ])
    # Missing values and concept explanations are full-sentence negatives;
    # mentioning a role/operator does not supply its literal value.
    return rng.choice([
        missing(rng, rng.choice(kinds), False) + '；先不要猜内容',
        '先解释【' + rng.choice(ROLES[rng.choice(kinds)]) + '】这个字段的含义',
        '我说的是删除这种操作，目前没有说要处理哪个对象',
        '刚才说的那份文档先留着，具体名字我还没写出来',
        '这个参数究竟是名字还是路径还没说，具体值也没给',
        '目前只讨论搜索与写入的区别，没有提供需要处理的原文',
    ])


def generate(directory='data/neural_v13', count=30000, seed=2081):
    dest = Path(directory)
    if dest.exists():
        raise ValueError('Use a new training directory; existing data are never overwritten')
    rng = random.Random(seed)
    reserved_frames = set(VALIDATION + CALIBRATION + TEST + NONE_VAL + NONE_CAL + NONE_TEST)
    reserved = {r['text'] for d in ('data/noisy', 'data/semantic')
                for split in ('challenge', 'validation', 'calibration')
                for r in read_jsonl(Path(d) / (split + '.jsonl'))}
    reserved.update(r['text'] for r in read_jsonl('examples/context_role_probe.jsonl'))
    rows = []
    for i in range(count):
        template = frame(rng)
        if template in reserved_frames:
            continue
        values = {}
        for slot in SLOT_RE.finditer(template):
            kind, key = slot.groups()
            # Payload shape is independent of declared role in 45% of rows;
            # it is augmentation material, not a name/type lookup table.
            payload_kind = rng.choice(tuple(ROLES)) if rng.random() < .45 else kind
            values[key] = literal(rng, payload_kind, 'scope_training', i, True)
        quotes = {key: rng.choice(QUOTES[1:]) if rng.random() < .35 else ('', '') for key in values}
        text, spans = render(template, values, quotes)
        prefix = rng.choice(PREFIXES)
        text = prefix + text + rng.choice(TAILS)
        if not 0 < len(text) <= 256 or text in reserved:
            continue
        rows.append({'id': f'scope_{i:06}', 'text': text,
                     'spans': [dict(s, start=s['start'] + len(prefix), end=s['end'] + len(prefix)) for s in spans],
                     'template': template, 'category': 'semantic_scope' if spans else 'semantic_scope_missing',
                     'source': 'semantic_augmentation'})
    fresh = len(rows)
    # Limit repeats of semantic frames, rather than spending most updates
    # on hundreds of names filled into a small number of fixed sentences.
    replay = read_jsonl('data/neural_v11/train.jsonl') + read_jsonl('data/neural_v6/train.jsonl')
    rng.shuffle(replay)
    groups = defaultdict(list)
    seen = {r['text'] for r in rows}
    frame_counts = Counter()
    for row in replay:
        if row['text'] in seen or row['text'] in reserved or row.get('template') in reserved_frames:
            continue
        category = row.get('category', '')
        group = 'semantic' if category.startswith('semantic') else 'clean' if category.startswith('basic') else 'noisy'
        if group == 'semantic':
            key = row.get('template') or row['id']
            if frame_counts[key] >= 8:
                continue
            frame_counts[key] += 1
        seen.add(row['text'])
        groups[group].append(row)
    for group, limit in [('semantic', 35000), ('clean', 12000), ('noisy', 15000)]:
        rows.extend(groups[group][:limit])
    assert all(r['text'] not in reserved and r.get('template') not in reserved_frames for r in rows)
    rng.shuffle(rows)
    dest.mkdir(parents=True)
    write_jsonl(dest / 'train.jsonl', rows)
    for split in ('validation', 'calibration'):
        write_jsonl(dest / (split + '.jsonl'), read_jsonl(Path('data/neural_v11') / (split + '.jsonl')))
    report = {'rows': len(rows), 'fresh_rows': fresh, 'seed': seed,
              'strategy': 'independent supplied/missing roles; delayed role declarations; comments; frame-limited semantic replay',
              'category_counts': dict(Counter(r.get('category', '') for r in rows)),
              'test_error_mining': False, 'reserved_frames_exclusion_only': True,
              'payload_role_independent_fraction': .45, 'semantic_replay_frame_cap': 8,
              'sha256': {s: hashlib.sha256((dest / (s + '.jsonl')).read_bytes()).hexdigest()
                         for s in ('train', 'validation', 'calibration')}}
    (dest / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report

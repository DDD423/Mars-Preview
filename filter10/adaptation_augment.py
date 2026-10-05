"""Compositional control-phrase augmentation, using development data only.

Names remain opaque. Rewrites operate only on text outside annotated slots.
The original frozen challenge and development evaluation files are preserved.
"""
import hashlib
import json
import random
import re
from pathlib import Path
from .adaptation_data import TRAIN, NONE, AMBIGUOUS, PREFIXES, SUFFIXES
from .data import SLOT_RE, QUOTES, render, random_value, read_jsonl, write_jsonl

REWRITES = {
 '文件夹': ['文件夹', '目录'],
 '这段文字': ['这段文字', '这段原文', '这段文本'],
 '这段原文': ['这段原文', '这段文字', '这段文本'],
 '对应的': ['对应的', '指向的', '所指的'],
 '复制一份到': ['复制一份到', '拷贝一份到', '拷一份到', '复制一份至'],
 '换成': ['换成', '换为', '更换成', '更换为', '改成', '改为'],
 '改称': ['改称', '改叫', '更名为'],
 '名字': ['名字', '名称'],
 '名称': ['名称', '名字'],
 '复制': ['复制', '拷贝', '拷'],
 '拷贝': ['拷贝', '复制'],
 '挪到': ['挪到', '移到', '搬到', '移动到'],
 '移动到': ['移动到', '移到', '搬到', '移动至'],
 '文档': ['文档', '文件'],
 '文件': ['文件', '文档'],
 '目录': ['目录', '文件夹'],
 '打开': ['打开', '打开一下'],
 '读一下': ['读一下', '读取一下', '读一读'],
 '读取': ['读取', '读一下'],
 '搜一下': ['搜一下', '搜索一下', '搜一搜'],
 '搜索': ['搜索', '查找', '搜'],
 '找一下': ['找一下', '查找一下', '找找'],
 '包含': ['包含', '含有'],
 '写进': ['写进', '写入', '写到', '放进'],
 '放进': ['放进', '写进', '写入'],
 '加上': ['加上', '添加', '补上'],
 '追加': ['追加', '补充', '添加'],
 '取名': ['取名', '起名', '命名为'],
 '建个': ['建个', '建一个', '创建一个'],
 '弄一个': ['弄一个', '新建一个', '建一个'],
 '新建': ['新建', '创建', '建立'],
 '删掉': ['删掉', '删除', '删除一下'],
 '删除': ['删除', '删掉', '删除一下'],
}
MATCH = re.compile('|'.join(re.escape(k) for k in sorted(REWRITES, key=len, reverse=True)))


def rewrite(row, rng):
    pieces, spans, cursor, offset = [], [], 0, 0
    for span in sorted(row['spans'], key=lambda s: s['start']):
        plain = MATCH.sub(lambda m: rng.choice(REWRITES[m.group()]), row['text'][cursor:span['start']])
        pieces.append(plain)
        offset += len(plain)
        value = row['text'][span['start']:span['end']]
        pieces.append(value)
        spans.append(dict(span, start=offset, end=offset+len(value)))
        offset += len(value)
        cursor = span['end']
    pieces.append(MATCH.sub(lambda m: rng.choice(REWRITES[m.group()]), row['text'][cursor:]))
    return dict(row, text=''.join(pieces), spans=spans)


def diverse_value(rng, kind, index):
    if rng.random() < .6:
        return random_value(rng, kind, 'train')
    words = ['预算', '最终版', '阅读', '记录', '明天', '不是', '过程', '星球', '属性', '删除', '哪个']
    text = ''.join(rng.choices(words + list('计划報告😀🚀é𠀀 ABx12'), k=rng.randint(1, 8))).strip() or '甲'
    if kind == 'PATH':
        text = rng.choice(['C:/新 工作区/', './资料/', '../新目录/', '/tmp/ζ/', '\\\\nas\\共享\\', '~/工作/']) + text
    if kind == 'TEXT' and rng.random() < .3:
        text += '\n第二行保留空格 ' + str(index)
    return text


def generate(directory='data/adaptive_v2', count=26000, seed=317):
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reserved = {r['text'] for split in ('validation', 'calibration', 'challenge')
                for r in read_jsonl(Path('data/adaptive')/f'{split}.jsonl')}
    rows = []
    for index in range(count):
        frames, category = (TRAIN, 'colloquial') if index % 10 < 8 else (NONE, 'no_literal') if index % 10 == 8 else (AMBIGUOUS, 'ambiguous')
        frame = rng.choice(frames)
        values = {m.group(2): diverse_value(rng, m.group(1), index) for m in SLOT_RE.finditer(frame)}
        quotes = {key: rng.choice(QUOTES[1:]) if rng.random() < .4 else ('', '') for key in values}
        text, spans = render(frame, values, quotes)
        prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
        spans = [dict(s, start=s['start']+len(prefix), end=s['end']+len(prefix)) for s in spans]
        row = {'id': f'v2_{index:06}', 'text': prefix+text+suffix, 'spans': spans,
               'category': category, 'source': 'adaptive_augmentation',
               'expected_status': 'needs_context' if category == 'ambiguous' else None}
        rows.append(rewrite(row, rng))
    # Add existing samples for retention. Do not add validation or test errors.
    replay = read_jsonl('data/adaptive/train.jsonl')
    rng.shuffle(replay)
    rows += replay[:10000]
    rows = [r for r in rows if r['text'] not in reserved and 0 < len(r['text']) <= 256]
    rng.shuffle(rows)
    write_jsonl(dest/'train.jsonl', rows)
    for split in ('validation', 'calibration'):
        write_jsonl(dest/f'{split}.jsonl', read_jsonl(Path('data/adaptive')/f'{split}.jsonl'))
    manifest = {'seed': seed, 'training_rows': len(rows), 'control_only_rewrites': True,
        'challenge_read_for_duplicate_exclusion_only': True,
        'development_splits_are_existing_sets': True,
        'sha256': {split: hashlib.sha256((dest/f'{split}.jsonl').read_bytes()).hexdigest()
                   for split in ('train', 'validation', 'calibration')}}
    (dest/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

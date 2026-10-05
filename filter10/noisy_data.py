"""Frozen noisy benchmark and disjoint synthetic post-training instances.

The test shares a compositional generator/noise distribution with training,
but not sentences or random literal namespaces. It is a robustness benchmark,
not a claim about arbitrary unseen natural language or real-user traffic.
"""
import hashlib
import json
import random
import re
from pathlib import Path
from .adaptation_composition import frame
from .adaptation_data import NONE, AMBIGUOUS
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl

GROUPS = {'complex': 200, 'punctuation': 250, 'colloquial': 250, 'spacing_typo': 200,
          'opaque_literals': 250, 'no_literal': 200, 'reference': 150}
OPS = ['open','delete','rename','transfer','create','search','write','sequence']
PREFIX = ['', '嗯，', '那个，', '呃，麻烦你', '我我想让你', '能不能帮我', '请先',
          '我正在整理资料。', '稍等，还是先', '啊，这样吧，', '麻烦一下，', '可以的话，']
SUFFIX = ['', '吧', '就行', '，谢谢', '，别改其他文件', '，行不行？', '呗', '好了']
TYPO = {'文件':'文 件', '文档':'文檔', '目录':'目錄', '名称':'名 称', '名字':'名子',
        '路径':'路经', '打开':'打 开', '复制':'复 制', '删除':'删 除', '搜索':'搜 索'}
TYPO_MATCH = re.compile('|'.join(TYPO))


def literal(rng, kind, namespace, index, adversarial=False):
    controls = ['删除', '叫做', '重命名为', '名字是', '的文件', '复制至', '属性', '它',
                '路径是', '写入内容', '包含文字', '先不要删除', 'FILE', '{1}']
    if rng.random() < (.55 if adversarial else .25):
        value = rng.choice(controls)
        if adversarial and rng.random() < .45:
            value += ' ' + rng.choice(controls)
    else:
        alphabet = list('报告计划春天ABCxyz019éζ😀🚀𠀀 _-')
        value = ''.join(rng.choices(alphabet, k=rng.randint(2, 30))).strip() or 'X'
        if rng.random() < .55:
            value += '_' + namespace + str(index) + rng.choice(['.txt','.md',' 计划.csv',''])
    if kind == 'PATH':
        value = rng.choice(['C:/工作 区/','D:\\目录\\','./space dir/','../新资料/','/tmp/ζ/',
                            '\\\\host\\shared space\\','~/文件/']) + value
    elif kind == 'TEXT':
        value = rng.choice([value, '第一行\n第二行 '+value, '先不要删除 '+value, '{1} '+value])
    return value


def control_noise(plain, rng, group):
    # Only control text changes. Names, paths, case, spaces and Unicode stay
    # byte-for-byte opaque after annotation; boundaries are rebuilt below.
    if group in ('punctuation', 'complex', 'colloquial', 'spacing_typo'):
        plain = plain.replace('，', rng.choice(['，', ',', '；', '，嗯，', '，然后，', '\n']))
    if group == 'spacing_typo':
        plain = TYPO_MATCH.sub(lambda m: TYPO[m.group()] if rng.random()<.35 else m.group(), plain)
        plain = ''.join(c + (' ' if c not in ' \n' and rng.random()<.08 else '') for c in plain)
    return plain


def one(rng, group, namespace, index):
    if group == 'no_literal':
        pattern = rng.choice(NONE)
    elif group == 'reference':
        pattern = rng.choice(AMBIGUOUS)
    else:
        op = rng.choice(['rename','transfer','sequence','write','search']) if group=='complex' else rng.choice(OPS)
        pattern = frame(rng, op)
        if group=='complex' and rng.random()<.5:
            # Three/four independently typed slots, including a second action.
            pattern += rng.choice(['，再打开名为[[NAME:f]]的文件',
                                   '，然后在路径[[PATH:g]]中搜索原文[[TEXT:h]]'])
    prefix, suffix = rng.choice(PREFIX), rng.choice(SUFFIX)
    pattern = prefix+pattern+suffix
    pieces, spans, cursor, offset = [], [], 0, 0
    for match in SLOT_RE.finditer(pattern):
        plain = control_noise(pattern[cursor:match.start()], rng, group)
        pieces.append(plain); offset += len(plain)
        value = literal(rng, match.group(1), namespace, index, group=='opaque_literals')
        quote = rng.choice(QUOTES[1:]) if rng.random()<(.6 if group=='opaque_literals' else .4) else ('','')
        # Unquoted personal pronouns after a generic noun are not uniquely
        # identifiable names. Preserve difficulty with an explicit quote.
        if value in ('它',) or (len(pattern)>140 and len(value)>20):
            quote = ('“','”')
        pieces.extend([quote[0], value, quote[1]])
        spans.append({'start': offset+len(quote[0]), 'end': offset+len(quote[0])+len(value), 'type': match.group(1)})
        offset += len(quote[0])+len(value)+len(quote[1])
        cursor = match.end()
    pieces.append(control_noise(pattern[cursor:], rng, group))
    return {'id': f'{namespace}_{group}_{index:06}', 'text': ''.join(pieces), 'spans': spans,
            'category': group, 'source': 'noisy_augmentation',
            'expected_status': 'needs_context' if group=='reference' else None}


def make(rng, distribution, namespace, unique=True, forbidden=()):
    rows, seen = [], set(forbidden)
    for group, count in distribution.items():
        accepted = 0
        while accepted<count:
            row = one(rng, group, namespace, len(rows))
            if not 0<len(row['text'])<=256 or (unique and row['text'] in seen):
                continue
            seen.add(row['text']); rows.append(row); accepted += 1
    rng.shuffle(rows)
    return rows


def generate(directory='data/noisy', seed=521, train_count=70000):
    dest = Path(directory); dest.mkdir(parents=True, exist_ok=True)
    test = dest/'challenge.jsonl'
    if not test.exists():
        rows = make(random.Random(seed), GROUPS, 'heldout')
        write_jsonl(test, rows)
        (dest/'challenge_manifest.json').write_text(json.dumps({'rows':len(rows),
            'sha256':hashlib.sha256(test.read_bytes()).hexdigest(), 'frozen_before_training':True,
            'groups':GROUPS, 'seed':seed, 'metric':'raw model sentence exact boundary/type, threshold 0.5',
            'shared_generator_with_training':True, 'not_real_user_traffic':True},indent=2),encoding='utf-8')
    pinned = json.loads((dest/'challenge_manifest.json').read_text())['sha256']
    assert hashlib.sha256(test.read_bytes()).hexdigest()==pinned
    reserved = {r['text'] for r in read_jsonl(test)}
    splits = {}
    for s, offset in [('validation',1),('calibration',2)]:
        splits[s] = make(random.Random(seed+offset), GROUPS, s, forbidden=reserved)
        reserved.update(r['text'] for r in splits[s])
    distribution = {g: round(n*train_count/1500) for g,n in GROUPS.items()}
    rows = make(random.Random(seed+3), distribution, 'training', unique=False)
    replay = read_jsonl('data/adaptive_v3/train.jsonl')
    random.Random(seed+4).shuffle(replay)
    rows += replay[:16000]
    rows = [r for r in rows if r['text'] not in reserved]
    random.Random(seed+5).shuffle(rows)
    splits['train'] = rows
    for s, rows in splits.items():
        write_jsonl(dest/(s+'.jsonl'), rows)
    manifest = {'seed':seed,'train_count_requested':train_count,'test_sha256':pinned,
        'no_test_error_mining':True,'shared_compositional_generator':True,
        'splits': {s:{'rows':len(rows),'sha256':hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest()}
                   for s,rows in splits.items()}}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

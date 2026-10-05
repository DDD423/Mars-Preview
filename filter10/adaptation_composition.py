"""Generate compositional training expressions; never mine challenge errors.

These are training grammars, not inference rules. Semantic slots are inserted
only after sampling control phrases and word order. Exact reserved sentences
are removed. Abstract structures can overlap the development/test families;
this is documented, not presented as a strict unseen-grammar benchmark.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl
from .adaptation_data import PREFIXES, SUFFIXES, NONE, AMBIGUOUS
from .adaptation_augment import diverse_value


def frame(rng, operation):
    pick = rng.choice
    n, b, t, p, q = '[[NAME:a]]', '[[NAME:b]]', '[[TEXT:c]]', '[[PATH:d]]', '[[PATH:e]]'
    obj = pick(['文件'+n, '文档'+n, n+'这个文件', n+'这份文档', n+'那份文件',
                '名为'+n+'的文件', '叫做'+n+'的文档', '路径'+p+'的文件', '路径为'+p+'的文档'])
    dest = pick(['目录'+b, '文件夹'+b, b+'这个目录', '路径'+q, '路径为'+q])
    if operation == 'open':
        act = pick(['打开', '打开一下', '看看', '查看', '读一下', '读取', '先读'])
        return pick([act+obj, '把'+obj+act, obj+pick(['，', '，先', '先', '，给我'])+act,
                     '从'+obj+pick(['里', '里面', '中'])+'提取文字',
                     act+obj+pick(['的内容', '的正文', '里的字', '里面的文字']),
                     obj+pick(['里的字', '的正文', '的内容'])+'给我读出来'])
    if operation == 'delete':
        act = pick(['删除', '删掉', '删了', '删除一下', '去掉'])
        return pick([act+obj, '把'+obj+act, '先别'+act+obj,
                     obj+pick(['不要了，', '不需要了，', '用不着了，', '，先别'])+act])
    if operation == 'rename':
        noun = pick(['名字', '名称', '文件名'])
        act = pick(['改成', '改为', '换成', '换为', '更换成', '更改为', '设成', '设为'])
        return pick(['把'+obj+'的'+noun+act+b,
                     '给'+obj+'换个'+noun+pick(['叫', '，叫', '，改叫', '叫做'])+b,
                     obj+'的'+noun+act+b,
                     obj+pick(['，以后叫', '，以后改叫', '改叫', '改称', '更名为'])+b,
                     '用'+b+'替换'+obj+'的'+noun])
    if operation == 'transfer':
        act = pick(['复制', '复制一份', '拷贝', '拷贝一份', '拷一份', '挪', '移', '搬', '移动'])
        to = pick(['到', '进', '至'])
        return pick(['把'+obj+act+to+dest+pick(['', '里', '中']),
                     obj+act+to+dest+pick(['', '里', '中']),
                     '将'+obj+act+to+dest,
                     obj+'先复制一份到当前目录',
                     obj+'复制一份，放到'+dest,
                     '把'+obj+'搬进选中的文件夹'])
    if operation == 'create':
        act = pick(['建一个', '新建一个', '弄个', '创建一个', '建个', '弄一个'])
        kind = pick(['文件', '空文件', '文档', '空文档', '目录', '文件夹'])
        name = pick(['名字用', '名称设成', '名称定为', '取名', '文件名用', '命名为', '名字就叫'])
        return pick([act+kind+'，'+name+n, act+'叫'+n+'的'+kind,
                     kind+'的'+name+n, name+n+'，'+act+kind])
    if operation == 'search':
        act = pick(['搜索', '搜索一下', '查找', '查找一下', '找一下', '找出', '搜'])
        cue = pick(['原文', '文字', '关键词', '内容'])
        return pick([act+cue+t, act+'正文'+pick(['含有', '包含', '有'])+t+'的文件',
                     act+cue+t+'出现在哪些文件中',
                     obj+pick(['里面', '里', '中'])+act+cue+t,
                     '目录'+b+'中找含有'+t+'的文档',
                     '在目录'+b+'里'+act+'包含'+t+'的文件',
                     '搜索词'+pick(['用', '设为', '就用', '设成'])+t])
    if operation == 'write':
        act = pick(['追加', '添加', '补上', '写入', '放入', '加上'])
        cue = pick(['原文', '文字', '一段原文', '一段文字', '内容'])
        return pick(['给'+obj+act+cue+t,
                     obj+pick(['里面', '里', '末尾', '的末尾', '，'])+act+cue+t,
                     '向'+obj+act+cue+t,
                     '往'+obj+'里面写入内容'+t,
                     '把'+t+pick(['这段文字', '这段原文', '这段文本'])+pick(['写进', '放进', '写入'])+obj,
                     '把文字'+t+'写进'+obj+'里面',
                     obj+pick(['的内容', '的正文'])+pick(['换成', '换为', '改为'])+t])
    if operation == 'sequence':
        act = pick(['读取', '打开', '读一下', '查看'])
        return pick([obj+'先'+act+'，再搜索它里面的原文'+t,
                     '先打开'+obj+'再读取文档'+b,
                     '先别删除'+obj+'，只删文档'+b,
                     '先不要动'+obj+'，读取文件'+b,
                     obj+'打开一下，再在它里面查找原文'+t,
                     '如果有备份，就删掉'+obj,
                     '先打开'+obj+'，然后把它拷到路径'+q,
                     '新建文件，名字用'+n+'，正文设为'+t])
    return pick(NONE if operation == 'none' else AMBIGUOUS)


def generate(directory='data/adaptive_v3', count=42000, seed=419):
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    reserved = {r['text'] for s in ('validation', 'calibration', 'challenge')
                for r in read_jsonl(Path('data/adaptive')/(s+'.jsonl'))}
    operations = ['open', 'delete', 'rename', 'transfer', 'create', 'search', 'write', 'sequence', 'none', 'reference']
    rows = []
    for index in range(count):
        op = operations[index % len(operations)]
        pattern = frame(rng, op)
        values = {m.group(2): diverse_value(rng, m.group(1), index) for m in SLOT_RE.finditer(pattern)}
        quotes = {k: rng.choice(QUOTES[1:]) if rng.random() < .45 else ('', '') for k in values}
        text, spans = render(pattern, values, quotes)
        prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
        spans = [dict(s, start=s['start']+len(prefix), end=s['end']+len(prefix)) for s in spans]
        rows.append({'id': f'v3_{index:06}', 'text': prefix+text+suffix, 'spans': spans,
            'operation': op, 'category': 'ambiguous' if op=='reference' else 'no_literal' if op=='none' else 'colloquial',
            'source': 'adaptive_augmentation', 'expected_status': 'needs_context' if op=='reference' else None})
    replay = read_jsonl('data/adaptive_v2/train.jsonl')
    rng.shuffle(replay)
    rows += replay[:18000]
    rows = [r for r in rows if r['text'] not in reserved and 0<len(r['text'])<=256]
    rng.shuffle(rows)
    write_jsonl(dest/'train.jsonl', rows)
    for split in ('validation', 'calibration'):
        write_jsonl(dest/(split+'.jsonl'), read_jsonl(Path('data/adaptive')/(split+'.jsonl')))
    manifest = {'seed': seed, 'training_rows': len(rows), 'compositional_training_only': True,
        'reserved_sentences_removed': True, 'challenge_errors_inspected': False,
        'abstract_grammar_overlap_possible': True,
        'sha256': {s: hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

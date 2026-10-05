"""Colloquial context-only supervision. No name lexicon is used at inference.

Frames, not just random names, are split before training. The frozen challenge
corpus is not used for mining, checkpoint selection, calibration or routing.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, read_jsonl, render, write_jsonl

TRAIN = [
 '把文件[[NAME:a]]的名称换成[[NAME:b]]', '文件[[NAME:a]]，给它换个名字叫[[NAME:b]]',
 '打开名为[[NAME:a]]的文档，并将文件名改为[[NAME:b]]',
 '把[[NAME:a]]那份文件改称[[NAME:b]]', '[[NAME:a]]这个文档，以后就叫[[NAME:b]]',
 '给文件[[NAME:a]]换个名字，叫[[NAME:b]]', '文件[[NAME:a]]的新名字是[[NAME:b]]',
 '文件[[NAME:a]]的名字换成[[NAME:b]]吧', '文档[[NAME:a]]改叫[[NAME:b]]',
 '将名字为[[NAME:a]]的文件更名为[[NAME:b]]', '用[[NAME:b]]替换文件[[NAME:a]]的名字',
 '名字就用[[NAME:b]]，原文件叫[[NAME:a]]',
 '打开文件[[NAME:a]]吧', '[[NAME:a]]这个文件打开一下', '帮我打开[[NAME:a]]这份文档',
 '我想看看文件[[NAME:a]]', '把[[NAME:a]]那份文件打开', '让我看一下[[NAME:a]]这个文档',
 '能不能打开名字为[[NAME:a]]的文件', '[[NAME:a]]这个文档我想读一下',
 '把文档[[NAME:a]]给我打开一下呗', '文档[[NAME:a]]先打开再说',
 '先把[[NAME:a]]这个文件读一下', '文件[[NAME:a]]读一下内容',
 '我想读文档[[NAME:a]]的正文', '麻烦读取[[NAME:a]]那份文件',
 '把文件[[NAME:a]]里的字提出来', '文档[[NAME:a]]，提取一下文字',
 '[[NAME:a]]这个文本里的字给我提取出来', '能把文件[[NAME:a]]里的内容读出来吗',
 '文件[[NAME:a]]不要了，删掉吧', '把[[NAME:a]]那份文档去掉',
 '[[NAME:a]]这个文件删一下', '帮我删了文件[[NAME:a]]',
 '文档[[NAME:a]]给我删除一下呗', '先别删除文件[[NAME:a]]',
 '[[NAME:a]]这份文件暂时别删', '不要动叫[[NAME:a]]的文件',
 '把文件[[NAME:a]]复制一份到目录[[NAME:b]]', '文件[[NAME:a]]拷到文件夹[[NAME:b]]里',
 '把[[NAME:a]]这个文档拷贝到[[NAME:b]]这个目录',
 '给文件[[NAME:a]]做个副本，放到目录[[NAME:b]]',
 '文件[[NAME:a]]移到目录[[NAME:b]]里吧', '将[[NAME:a]]这个文件挪到[[NAME:b]]这个文件夹',
 '把文件[[NAME:a]]搬到目录[[NAME:b]]', '文件[[NAME:a]]复制到路径[[PATH:b]]',
 '把文件[[NAME:a]]挪去路径[[PATH:b]]', '复制路径[[PATH:a]]的文件到路径[[PATH:b]]',
 '把路径[[PATH:a]]的文档拷一份到路径[[PATH:b]]',
 '文件[[NAME:a]]拷贝一份，放在当前目录', '把文件[[NAME:a]]移动到选中的文件夹',
 '建个文件，名字用[[NAME:a]]', '弄一个文档，取名[[NAME:a]]',
 '帮我建个叫[[NAME:a]]的文件', '我想新建一个文件，命名为[[NAME:a]]',
 '建一个文件夹，叫[[NAME:a]]就行', '目录名字设成[[NAME:a]]',
 '给我新建个目录，名称定为[[NAME:a]]', '新文件就叫[[NAME:a]]吧',
 '文件名用[[NAME:a]]，先创建一个空文件', '创建文档，名字设为[[NAME:a]]',
 '路径[[PATH:a]]那个文件打开一下', '打开路径[[PATH:a]]对应的文件',
 '把路径[[PATH:a]]的文件读出来', '读取路径[[PATH:a]]的正文',
 '路径[[PATH:a]]的文件先别删', '把路径[[PATH:a]]对应的文档删掉',
 '在路径[[PATH:a]]里找包含[[TEXT:b]]的文件',
 '在路径[[PATH:a]]对应的文件里搜索原文[[TEXT:b]]',
 '帮我搜一下关键词[[TEXT:a]]', '找找含有[[TEXT:a]]的文档',
 '我想找正文里有[[TEXT:a]]的文件', '搜索词就用[[TEXT:a]]',
 '找一下原文[[TEXT:a]]出现在哪些文件中', '查一下哪些文档包含文字[[TEXT:a]]',
 '帮我搜索[[TEXT:a]]这段文字', '[[TEXT:a]]这段原文搜一下',
 '在文件[[NAME:a]]中找一下原文[[TEXT:b]]', '文件[[NAME:a]]里搜关键词[[TEXT:b]]',
 '文档[[NAME:a]]加上一段文字[[TEXT:b]]', '给文件[[NAME:a]]补上原文[[TEXT:b]]',
 '往文件[[NAME:a]]里写点内容[[TEXT:b]]', '文件[[NAME:a]]的末尾加上文字[[TEXT:b]]',
 '在文件[[NAME:a]]末尾写入[[TEXT:b]]', '文档[[NAME:a]]，追加内容[[TEXT:b]]',
 '把[[TEXT:a]]这段文字写进文件[[NAME:b]]', '把原文[[TEXT:a]]放进文档[[NAME:b]]里',
 '把文字[[TEXT:a]]添到文件[[NAME:b]]的末尾', '往路径[[PATH:a]]的文件里写入[[TEXT:b]]',
 '把内容[[TEXT:a]]写进路径[[PATH:b]]的文档', '文件[[NAME:a]]的内容换成[[TEXT:b]]',
 '新建文件[[NAME:a]]，内容设为[[TEXT:b]]', '文件[[NAME:a]]不要写入文字[[TEXT:b]]',
 '如果有备份，就删文件[[NAME:a]]', '如果文件存在，打开文件[[NAME:a]]',
 '先打开文件[[NAME:a]]，然后打开文件[[NAME:b]]',
 '读完文件[[NAME:a]]再读文档[[NAME:b]]', '别删文件[[NAME:a]]，只删除文件[[NAME:b]]',
 '文件[[NAME:a]]不要改名，文件[[NAME:b]]才改叫[[NAME:c]]',
 '文件[[NAME:a]]先打开，然后在它里面找原文[[TEXT:b]]',
 '把名为[[NAME:a]]的文件打开，再把它复制到路径[[PATH:b]]',
 '给我打开[[VALUE:a]]', '把[[VALUE:a]]复制到[[VALUE:b]]',
 '把[[VALUE:a]]移到[[VALUE:b]]', '读取一下[[VALUE:a]]',
]
VALIDATION = [
 '把文件[[NAME:a]]的名字更换成[[NAME:b]]', '给文档[[NAME:a]]换个名称叫[[NAME:b]]',
 '[[NAME:a]]这份文档，以后叫[[NAME:b]]吧', '打开一下文件[[NAME:a]]',
 '先读文件[[NAME:a]]的内容', '文档[[NAME:a]]不要了，删除吧',
 '把文档[[NAME:a]]拷一份到目录[[NAME:b]]', '把文件[[NAME:a]]搬进文件夹[[NAME:b]]',
 '新建一个空文档，名字用[[NAME:a]]', '打开路径[[PATH:a]]所指的文档',
 '搜索词设为[[TEXT:a]]', '文件[[NAME:a]]里追加一段原文[[TEXT:b]]',
 '把[[TEXT:a]]这段原文写进文档[[NAME:b]]', '目录[[NAME:a]]中找含有[[TEXT:b]]的文档',
 '先不要动文件[[NAME:a]]，读取文件[[NAME:b]]',
 '文件[[NAME:a]]先读取，再搜索它里面的原文[[TEXT:b]]',
]
CALIBRATION = [
 '将文件[[NAME:a]]的名字换为[[NAME:b]]', '文件[[NAME:a]]换个名字，叫[[NAME:b]]吧',
 '帮我看看文档[[NAME:a]]', '[[NAME:a]]这份文件读取一下',
 '删除一下文件[[NAME:a]]', '文档[[NAME:a]]复制一份，放到目录[[NAME:b]]',
 '把文件[[NAME:a]]移进文件夹[[NAME:b]]', '给我弄个文件，名字用[[NAME:a]]',
 '路径[[PATH:a]]的文档给我打开', '搜索一下关键词[[TEXT:a]]',
 '给文档[[NAME:a]]追加一段文字[[TEXT:b]]', '将文字[[TEXT:a]]放进文件[[NAME:b]]',
 '文件[[NAME:a]]的正文换为[[TEXT:b]]', '把文件[[NAME:a]]复制到选中的目录',
 '如果有备份，删除路径[[PATH:a]]的文件', '文件[[NAME:a]]打开后给它追加原文[[TEXT:b]]',
]
CHALLENGE = [
 '把文件[[NAME:a]]的名称改成[[NAME:b]]就行', '文件[[NAME:a]]的名称换为[[NAME:b]]',
 '给[[NAME:a]]那份文档换个名字叫[[NAME:b]]', '文档[[NAME:a]]，以后改叫[[NAME:b]]',
 '我想看看[[NAME:a]]这份文件', '[[NAME:a]]这个文件，先打开一下',
 '帮我读一下文件[[NAME:a]]里的字', '从文档[[NAME:a]]里面提取文字',
 '文件[[NAME:a]]不需要了，删除一下', '先别删[[NAME:a]]那份文件',
 '文件[[NAME:a]]拷贝到目录[[NAME:b]]中', '将文件[[NAME:a]]复制一份到文件夹[[NAME:b]]',
 '把[[NAME:a]]这个文档挪到目录[[NAME:b]]', '将文件[[NAME:a]]搬进目录[[NAME:b]]',
 '文件[[NAME:a]]先复制一份到当前目录', '建一个空文档，文件名用[[NAME:a]]',
 '弄个叫[[NAME:a]]的目录', '新建文件，名称设成[[NAME:a]]',
 '路径[[PATH:a]]指向的文件，打开一下', '读一下路径[[PATH:a]]的文档内容',
 '帮我搜索文字[[TEXT:a]]', '找出正文含有[[TEXT:a]]的文件',
 '搜索原文[[TEXT:a]]出现在哪份文档', '文档[[NAME:a]]里面查找关键词[[TEXT:b]]',
 '向文件[[NAME:a]]追加一段原文[[TEXT:b]]', '往文档[[NAME:a]]里面写入内容[[TEXT:b]]',
 '把[[TEXT:a]]这段原文放进文件[[NAME:b]]', '把文字[[TEXT:a]]写进文档[[NAME:b]]里面',
 '文件[[NAME:a]]的正文换成[[TEXT:b]]', '别给文件[[NAME:a]]添加原文[[TEXT:b]]',
 '有备份的话，删掉文件[[NAME:a]]', '先打开文档[[NAME:a]]再读取文档[[NAME:b]]',
 '先别删除文件[[NAME:a]]，只删文档[[NAME:b]]',
 '文件[[NAME:a]]打开一下，再在它里面搜索原文[[TEXT:b]]',
 '先打开名为[[NAME:a]]的文件，然后把它拷到路径[[PATH:b]]',
 '新建文件，名字用[[NAME:a]]，正文设为[[TEXT:b]]',
 '把路径[[PATH:a]]的文件移到路径[[PATH:b]]',
 '在目录[[NAME:a]]中搜索含有[[TEXT:b]]的文件',
]
NONE = [
 '只查看文件属性', '查看文件的属性', '显示所有文件', '统计文件数量', '列出目录结构',
 '保留当前选中的文件', '暂时别继续操作', '把当前文件复制到选中的目录', '暂停操作',
 '先等一下', '先不操作了', '别删所有文件', '全部文件都保留', '打开当前选中的文档',
 '给选中的文档改个名字', '把这些文件复制到当前目录', '看看目录里有多少文件',
 '把文件按时间排序', '先别改文件的名字', '删除文件前先备份', '先检查一下文件大小',
 '只看看文件的修改时间', '不用改内容', '别操作了', '保持名称不变',
]
AMBIGUOUS = [
 '把它删掉', '那个文件帮我打开', '把刚才那个挪到这边', '给它换个名字叫[[NAME:a]]',
 '把它复制到路径[[PATH:a]]', '把那个文件里的字提出来', '给那个追加文字[[TEXT:a]]',
 '把前面那个删了', '那份文件改成[[NAME:a]]', '把那个放到目录[[NAME:a]]',
 '它的内容换成[[TEXT:a]]', '把刚才那个文件复制一份',
 '打开文件[[NAME:a]]和文件[[NAME:b]]，然后删掉它',
 '把文件[[NAME:a]]复制到目录[[NAME:b]]，再删除那个',
]
AMBIGUOUS_TEST = [
 '帮我把它删除一下', '刚刚那个文件打开一下', '把前面那份复制到路径[[PATH:a]]',
 '它改叫[[NAME:a]]吧', '给那份文件加上原文[[TEXT:a]]',
 '把那个文件搬到目录[[NAME:a]]',
 '先打开文件[[NAME:a]]和文档[[NAME:b]]，然后把它删除',
 '文件[[NAME:a]]复制到文件夹[[NAME:b]]后，把那个删掉',
]
TEST_NONE = [
 '看看文件的大小', '只查看文档属性', '暂时不要继续了', '先停一下吧', '所有文件先留着',
 '把工作区文件按大小排序', '当前选中的文件先打开', '统计一下文档数量',
 '列出所有文件的名称', '不要修改任何文件',
]
PREFIXES = ['', '嗯，', '麻烦你', '能不能', '我想让你', '那个，', '呃，', '请帮我', '先', '现在']
SUFFIXES = ['', '吧', '就行', '，谢谢', '，别操作其他文件']


def value(rng, kind, split, index):
    reserved = ['删除', '叫做', '重命名为', '属性', '文件', '复制至', '路径是', '名字是', '那个', '它', '写入', '包含文字']
    raw = rng.choice(reserved) if rng.random() < .3 else f'{split}_{index}_{rng.randrange(1000000)}' + rng.choice(['.txt', '🚀.md', ' 计划.csv', 'ζ', '𠀀'])
    if kind == 'PATH':
        return rng.choice(['C:/陌生 工作区/', './新目录/', '../资料/', '/tmp/星球/', '\\\\nas\\共享\\', 'D:\\资料\\', '~/files/']) + raw
    if kind == 'TEXT':
        return rng.choice([raw, '今天完成了 ' + raw, '第一行\n第二行 ' + raw, '先不要删除 ' + raw, '{1} ' + raw])
    return raw


def make_rows(frames, count, split, rng, category='colloquial', modifiers=True):
    rows = []
    for index in range(count):
        frame = frames[index % len(frames)]
        values = {m.group(2): value(rng, m.group(1), split, index) for m in SLOT_RE.finditer(frame)}
        quotes = {key: rng.choice(QUOTES[1:]) if rng.random() < .35 else ('', '') for key in values}
        text, spans = render(frame, values, quotes)
        prefix = rng.choice(PREFIXES) if modifiers else ''
        suffix = rng.choice(SUFFIXES) if modifiers else ''
        text = prefix + text + suffix
        spans = [dict(s, start=s['start']+len(prefix), end=s['end']+len(prefix)) for s in spans]
        rows.append({'id': f'{split}_{category}_{index:05}', 'family': f'{split}_{category}_{index%len(frames):03}',
                     'text': text, 'spans': spans, 'category': category,
                     'expected_status': 'needs_context' if category == 'ambiguous' else None,
                     'source': 'adaptive_augmentation' if split == 'train' else 'assistant_authored_adaptive'})
    return rows


def generate(data_dir='data/adaptive', seed=203):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    challenge = directory / 'challenge.jsonl'
    if not challenge.exists():
        test = make_rows(CHALLENGE, 760, 'challenge', rng)
        test += make_rows(TEST_NONE, 100, 'challenge', rng, 'no_literal')
        test += make_rows(AMBIGUOUS_TEST, 120, 'challenge', rng, 'ambiguous')
        write_jsonl(challenge, test)
        challenge.with_name('challenge_manifest.json').write_text(json.dumps({
            'rows': len(test), 'sha256': hashlib.sha256(challenge.read_bytes()).hexdigest(),
            'frozen_before_training': True, 'used_for_tuning': False,
            'source': 'assistant-authored held-out expression frames; not real-user traffic'}, indent=2), encoding='utf-8')
    else:
        pinned = json.loads(challenge.with_name('challenge_manifest.json').read_text())['sha256']
        if hashlib.sha256(challenge.read_bytes()).hexdigest() != pinned:
            raise ValueError('Frozen challenge has changed')
        # Keep training generation reproducible on repeated invocations.
        rng = random.Random(seed)
        make_rows(CHALLENGE, 760, 'challenge', rng)
        make_rows(TEST_NONE, 100, 'challenge', rng, 'no_literal')
        make_rows(AMBIGUOUS_TEST, 120, 'challenge', rng, 'ambiguous')
    reserved = {r['text'] for r in read_jsonl(challenge)}
    replay = read_jsonl('data/posttrain/train.jsonl')
    rng.shuffle(replay)
    fresh = make_rows(TRAIN, 15000, 'train', rng)
    fresh += make_rows(NONE, 2400, 'train', rng, 'no_literal')
    fresh += make_rows(AMBIGUOUS, 1600, 'train', rng, 'ambiguous')
    splits = {'train': replay[:14000] + fresh}
    for split, frames in [('validation', VALIDATION), ('calibration', CALIBRATION)]:
        rows = make_rows(frames, 1000, split, rng)
        rows += make_rows(NONE, 150, split, rng, 'no_literal')
        rows += make_rows(AMBIGUOUS, 150, split, rng, 'ambiguous')
        rows += read_jsonl(f'data/posttrain/{split}.jsonl')[:400]
        splits[split] = rows
    manifest = {'seed': seed, 'parent': 'artifacts/round4/filter1.0.pt', 'splits': {}}
    for split, rows in splits.items():
        rows = [r for r in rows if r['text'] not in reserved and 0 < len(r['text']) <= 256]
        rng.shuffle(rows)
        path = directory / f'{split}.jsonl'
        write_jsonl(path, rows)
        manifest['splits'][split] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

"""Development-driven training for incidental remarks and complete span sets.

No inference grammar accompanies these templates. Held-out text is read only
for duplicate exclusion, never to select expressions or mine prediction errors.
"""
from collections import Counter
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, read_jsonl, render, write_jsonl
from .noisy_data import literal
from .context_algebra import declaration, role
from .semantic_data import TRAIN, NONE_TRAIN, VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST

REMARKS = ['（正文保留）', '（只换称呼）', '（其他文件先不动）',
           '（先做一次预览）', '（就是这份）', '（仅在工作区内）',
           '（稍后确认）', '（注意不要覆盖）', '（原样保留字符）',
           '，先强调一下别改内容，', '，麻烦先确认一下，']
TAILS = ['', '，我只是先给出参数', '，先别执行', '，谢谢你',
         '；内容和属性保持原样', '，请完整保留原文', '，之后再决定操作']


def frame(rng):
    a, b = '[[NAME:a]]', '[[NAME:b]]'
    noun = rng.choice(['文件', '文档', '目录', '文件夹'])
    remark = rng.choice(REMARKS)
    branch = rng.randrange(10)
    if branch < 3:
        # Remarks belong to the control text, between an opaque parameter
        # and its role explanation. They are independently varied.
        return rng.choice([
            f'请将{a}{remark}这份{noun}改叫{b}',
            f'{a}{remark}是{noun}现在的名称，新名称选{b}',
            f'原名{a}{remark}要换成新的名字{b}',
            f'要改名的{noun}叫{a}{remark}；新名请选{b}',
            f'新的名字用{b}，旧名称{a}{remark}请先记下来',
        ])
    if branch == 3:
        old = rng.choice(['过去的名字', '原有的名称', '原本的名字', '目前使用的名称', '原来的名称'])
        return rng.choice([
            f'拿{b}这个新名字替换{noun}{old}{a}',
            f'选定新名字{b}，替换掉{noun}{old}{a}',
            f'{noun}{old}{a}需要更新；请采用新名字{b}',
            f'{b}用于新的名称，{noun}{old}是{a}',
        ])
    if branch == 4:
        kind = rng.choice(['NAME', 'PATH', 'TEXT', 'VALUE'])
        x = '[[' + kind + ':a]]'
        return x + remark + rng.choice(['是', '给出的是', '表示的是']) + role(rng, kind, True) + rng.choice(TAILS)
    if branch == 5:
        return declaration(rng,rng.choice(['NAME','PATH','TEXT','VALUE']),'a',True) + rng.choice(['；同时，','，另外，','。然后，']) + declaration(rng,rng.choice(['NAME','PATH','TEXT','VALUE']),'b',True) + rng.choice(TAILS)
    if branch == 6:
        return rng.choice([
            '把原文[[TEXT:a]]写入名字为[[NAME:b]]的文档，正文不要再改变',
            '文件的名字用[[NAME:a]]，匹配的文字用[[TEXT:b]]，两项都原样记录',
            '先查看叫做[[NAME:a]]的文件，再到目录[[NAME:b]]，只先做预览',
            '把文件名[[NAME:a]]复制到目录名[[NAME:b]]，其他目录不变',
            '名称是[[NAME:a]]的文件，改用[[NAME:b]]这个名字',
        ])
    if branch == 7:
        return rng.choice(TRAIN) + rng.choice(TAILS)
    missing = rng.choice(['实际名字', '具体文件名称', '完整文件地址', '检索的具体原文', '真正的路径', '写进去的文字'])
    context = rng.choice(['这里仅在说明参数类型', '我只是介绍操作的叫法',
                          '现在是在讨论命令的意思', '这次是在解释名称与操作的区别'])
    return rng.choice([
        f'{context}，{missing}仍未给定',
        f'{missing}没有提供；{context}，先等待',
        f'尚未确定{missing}，不要把操作叫法当成它',
        rng.choice(NONE_TRAIN) + rng.choice(TAILS),
    ])


def generate(directory='data/neural_v11',count=32000,seed=1847):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        raise ValueError('训练目录已生成，请使用新目录保留数据版本')
    rng=random.Random(seed)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic')
              for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    reserved.update(r['text'] for r in read_jsonl('examples/context_role_probe.jsonl'))
    rows=[]
    for i in range(count):
        template=frame(rng)
        if template in reserved_frames:continue
        values={m.group(2):literal(rng,m.group(1),'incidental_training',i,True)
                for m in SLOT_RE.finditer(template)}
        if values and rng.random()<.2:
            key=rng.choice(list(values))
            # Long random payloads and action-like payloads require the same
            # boundaries; lengths are bounded after complete rendering.
            alphabet='资料计划ABCxyz_0123456789删除叫做文件🚀𠀀'
            values[key]=''.join(rng.choice(alphabet) for _ in range(rng.randint(40,100)))
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.35 else ('','') for k in values}
        text,spans=render(template,values,quotes)
        prefix=rng.choice(['','','嗯，','请你帮忙，','我这次想这样：'])
        text=prefix+text
        if not 0<len(text)<=256 or text in reserved:continue
        rows.append({'id':f'incidental_{i:06}','text':text,
                     'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
                     'template':template,'category':'semantic_incidental' if spans else 'semantic_missing_argument',
                     'source':'semantic_augmentation'})
    replay=read_jsonl('data/neural_v10/train.jsonl');rng.shuffle(replay)
    for group,limit in [('clean',12000),('semantic',26000),('noisy',16000)]:
        def matches(r):
            category=r.get('category','')
            return ('semantic' if category.startswith('semantic') else 'clean' if category.startswith('basic') else 'noisy')==group
        rows.extend([r for r in replay if matches(r)][:limit])
    assert all(r['text'] not in reserved and r.get('template') not in reserved_frames for r in rows)
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    for split in ('validation','calibration'):
        write_jsonl(dest/(split+'.jsonl'),read_jsonl(Path('data/neural_v10')/(split+'.jsonl')))
    report={'rows':len(rows),'seed':seed,'strategy':'incidental remarks, argument order, long payloads, missing arguments; development-driven replay',
            'categories':dict(Counter(r.get('category',r.get('operation','unknown')) for r in rows)),
            'test_error_mining':False,'full_reserved_frames_excluded':True,'abstract_linguistic_parts_overlap':True,
            'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

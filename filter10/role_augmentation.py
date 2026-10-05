"""Train role concepts and grammatical connectors, never a name dictionary.

Role phrases are training supervision only. They are not imported by the
model/harness at inference. Full reserved development/test frames are removed.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl
from .noisy_data import literal
from .semantic_data import VALIDATION,CALIBRATION,TEST,NONE_VAL,NONE_CAL,NONE_TEST

ROLES = {
 'NAME':['文件名','文件名称','文档名','文档名称','目录名称','文件夹的名称','原文件名','新文件名',
         '目前的文件名','原来的名字','准备使用的新名称','现有的文件名','这个文档的名字'],
 'PATH':['完整路径','完整文件路径','文件完整地址','完整文件地址','文件的完整地址','文件的完整路径',
         '完整地址','源文件的完整地址','目标完整路径','源路径','目的完整路径'],
 'TEXT':['搜索用的原文','需要查找的原文','需要查询的文字','新增正文','需要写入的正文','正文',
         '原文','检索内容','查找内容','搜索词','准备写进去的文字','新增的正文'],
 'VALUE':['操作目标','目标标识','源标识','目的标识','传入参数','未知类型的参数','暂未分类的目标',
          '未知类别的对象标识','需要操作的目标','尚未分类的参数']}
COPULA = ['是','才是','就是','正是','的确是','其实是','给的是','给出的是','提供的是','代表的是','指的是',
          '表示的是','将作为','用作','作为']
PREFIXES = ['','嗯，','呃，','我想这样：','麻烦你，','请看，','本次操作，']


def pattern(rng,kind):
    role=rng.choice(ROLES[kind]); x='[['+kind+':a]]'; y='[['+kind+':b]]'
    if kind=='NAME' and rng.random()<.2:
        return rng.choice(['以'+x+'命名的文件只需读取','用'+x+'命名的是这份文档，请保留',
            '文档采用'+x+'命名，暂时只做预览','以'+x+'命名的文档，正文用[[TEXT:c]]表示',
            '要查看的是采用'+x+'命名的文档，先不要修改',
            x+'（仅修改文件名）这份文档采用'+y+'这个新名字',
            x+'（请保持正文不变）这份文件需要读取',
            '以'+x+'命名的文档复制给以'+y+'命名的目录，先做预览'])
    return rng.choice([
        x+rng.choice(COPULA)+role+'；这是本次操作的参数',
        '本次提供的'+role+rng.choice(['为','是','用','指定为','采用'])+x+'，参数原样保留',
        role+'具体由'+x+rng.choice(['给出','确定','表示','指定'])+'，操作稍后处理',
        x+'这个'+role+'需要保持原样，稍后处理',
        '请用'+x+'作为本次的'+role+'，动作另行确认',
        x+'表示'+role+'，本次仅采用这个参数',
        '提供的'+role+'由'+x+'给出，先不要执行',
        '先采用'+x+'这个'+role+'作为本次参数',
        x+'和'+y+'分别是两个'+role+'，都作为本次参数',
        x+'是源'+role+'而'+y+'是目标'+role+'，都保持原样',
    ])


def generate(directory='data/neural_v2',count=42000,seed=839):
    p=Path(directory);p.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    reserved_templates=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic') for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    rows=[]
    for i in range(count):
        kind=rng.choice(list(ROLES)); template=pattern(rng,kind)
        if i%9==0:
            role=rng.choice(ROLES[kind])
            template=rng.choice([role+'只是概念说明，没有提供具体的字面参数',
                '这里的'+role+'是类型名称，不是一个具体对象',
                '目前尚未给出'+role+'，不要猜测参数',
                '提到'+role+'只是为了说明类别，尚无具体参数'])
        if template in reserved_templates:
            continue
        vals={m.group(2):literal(rng,m.group(1),'roletrain',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.5 else ('','') for k in vals}
        text,spans=render(template,vals,quotes);prefix=rng.choice(PREFIXES)
        row={'id':f'role_{i:06}','text':prefix+text,
             'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
             'template':template,'category':'semantic_role_augmentation','source':'semantic_augmentation'}
        if len(row['text'])<=256 and row['text'] not in reserved:rows.append(row)
    noise=read_jsonl('data/noisy/train.jsonl');rng.shuffle(noise)
    original=read_jsonl('data/semantic/train.jsonl');rng.shuffle(original)
    rows+=noise[:46000]+original[:16000]
    rows=[r for r in rows if r['text'] not in reserved];rng.shuffle(rows)
    write_jsonl(p/'train.jsonl',rows)
    for s in ('validation','calibration'):
        write_jsonl(p/(s+'.jsonl'),read_jsonl(Path('data/neural')/(s+'.jsonl')))
    manifest={'seed':seed,'rows':len(rows),'role_phrases_training_only':True,
        'held_out_full_frames_removed':True,'test_error_mining':False,
        'abstract_role_structures_overlap':True,
        'sha256':{s:hashlib.sha256((p/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (p/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

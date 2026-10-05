"""Equivalent control-language variants, with literal contents untouched."""
import hashlib,json,random,re
from pathlib import Path
from .data import FAMILIES,SLOT_RE,QUOTES,render,read_jsonl,write_jsonl
from .semantic_data import TRAIN,VALIDATION,CALIBRATION,TEST,NONE_VAL,NONE_CAL,NONE_TEST,NONE_TRAIN
from .noisy_data import literal
from .constituent_data import frame
from .role_composition import frame as role_frame

GROUPS=[
 ['文件完整地址','完整文件地址','文件完整路径','完整文件路径','完整地址','完整路径'],
 ['文件名称','文档名称','文件名字','文档名字','文件名','文档名'],
 ['目录名称','目录名字','文件夹名称','文件夹名字'],
 ['这个名称','这个名字','这一名称','这一名字','该名称','该名字'],
 ['这个新名','这个新名字','这个新名称','新的名称','新的名字'],
 ['名称','名字'],['文件','文档'],['这份','那份','这个','该'],
 ['这段原文','这段文字','这段正文','这段内容','这几个字'],
 ['原文','正文','文字','文本内容'],
 ['需要查询','需要查找','需要搜索','要检索','要匹配'],
 ['查找','搜索','查询','检索'],
 ['写入','追加','添加','补充','补上','加入'],
 ['采用','选用','使用','指定为','具体为','原样是'],
 ['用作','作为','将作为'],
 ['表示','代表','指的是','意味着'],
 ['是','就是','其实是','给的是','给出的是'],
 ['命名成','命名为','取名为','起名为','取名字叫'],
 ['命名','起名','取名'],
 ['改名为','重命名为','改叫','名称改成','取新名'],
 ['原来','原本','之前'],['现在','目前','当前'],
 ['没有说明','未说明','没有交代','尚未明确'],
 ['类型未知','类别未明','具体类型待确认','类别仍未确定'],
]
CHOICES={word:group for group in GROUPS for word in group}
MATCH=re.compile('|'.join(re.escape(w) for w in sorted(CHOICES,key=len,reverse=True)))


def vary(template,rng):
    # Slots remain syntax, never undergo lexical substitution.
    pieces=[];cursor=0
    replace=lambda m:rng.choice(CHOICES[m.group()]) if rng.random()<.45 else m.group()
    for slot in SLOT_RE.finditer(template):
        pieces += [MATCH.sub(replace,template[cursor:slot.start()]),slot.group()]
        cursor=slot.end()
    pieces.append(MATCH.sub(replace,template[cursor:]))
    return ''.join(pieces)


def declaration(rng):
    n='[[NAME:a]]';m='[[NAME:b]]';t='[[TEXT:a]]'
    nameverb=rng.choice(['名字叫','名称是','命名为','起名为','取名为','原名采用'])
    return rng.choice([
        '这份文档'+nameverb+n+'，只需要读取它',
        '文档原来'+nameverb+n+'，现在名称采用'+m,
        n+'这几个字表示文档名称，需要做的是读取',
        '创建一份文档，名称选用'+n+'这几个字',
        '需要查询的是原文，原文具体为'+t,
        '查找的是正文，具体的正文采用'+t,
        t+'这段原文代表检索内容，暂时不要写入',
        '新增的正文给定为'+t+'，追加到当前文档',
    ])


def generate(directory='data/neural_v6',count=44000,seed=1361):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic') for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    basic=[s for group in FAMILIES.values() for s in group[:8]]
    rows=[]
    for i in range(count):
        generator=rng.randrange(5)
        template=rng.choice(TRAIN+basic) if generator==0 else frame(rng) if generator==1 else role_frame(rng) if generator==2 else declaration(rng)
        if i%10==0:template=rng.choice(NONE_TRAIN)
        template=vary(template,rng)
        if template in reserved_frames:continue
        vals={m.group(2):literal(rng,m.group(1),'paraphrase_training',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.4 else ('','') for k in vals}
        text,spans=render(template,vals,quotes);prefix=rng.choice(['','嗯，','麻烦你，','呃，','请先，'])
        row={'id':f'paraphrase_{i:06}','text':prefix+text,
             'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
             'template':template,'category':'semantic_paraphrase','source':'semantic_augmentation'}
        if len(row['text'])<=256 and row['text'] not in reserved:rows.append(row)
    replay=read_jsonl('data/neural_v5/train.jsonl');rng.shuffle(replay)
    for group,n in [('clean',24000),('semantic',32000),('noisy',24000)]:
        select=lambda r: ('semantic' if r.get('category','').startswith('semantic') else 'clean' if r.get('category','').startswith('basic') else 'noisy')==group
        rows += [r for r in replay if select(r)][:n]
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    for split in ('validation','calibration'):
        write_jsonl(dest/(split+'.jsonl'),read_jsonl(Path('data/neural_v5')/(split+'.jsonl')))
    report={'rows':len(rows),'seed':seed,'strategy':'training control-language paraphrases and role declarations; replay',
        'test_error_mining':False,'full_reserved_frames_excluded':True,'abstract_linguistic_parts_overlap':True,
        'paraphrases_may_sound_unusual':True,
        'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

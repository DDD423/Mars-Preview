"""Held-out expression families for semantic role/boundary generalization.

These task-specific labels describe explicit command parameters, not all
nouns. Family lists are separate; test errors never generate training frames.
All names are filled after selecting the expression and remain opaque.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl
from .noisy_data import literal

TRAIN = [
 '[[NAME:a]]是文件名，请打开这个文件', '这个文档的名字是[[NAME:a]]，读取它',
 '[[NAME:a]]用作文件名，不是执行命令', '文件的标识名称采用[[NAME:a]]',
 '[[NAME:a]]这几个字是文档名称，操作是打开', '要读取的文档名称由[[NAME:a]]表示',
 '选取名为[[NAME:a]]的那份，做读取操作', '我指的是[[NAME:a]]这个名称的文档',
 '要删的是[[NAME:a]]这份文档，别理解成别的操作', '[[NAME:a]]对应的文档请删除',
 '文件取[[NAME:a]]这个名字，再创建它', '请创建文档，[[NAME:a]]是它的命名结果',
 '[[NAME:a]]作为目录名称，不是目录路径', '这个文件夹就采用[[NAME:a]]作为名字',
 '[[NAME:a]]是原文件的名字，[[NAME:b]]是新名字',
 '旧文件名[[NAME:a]]，新文件名[[NAME:b]]，执行改名',
 '[[NAME:b]]这个名字替换[[NAME:a]]这个文件的名字',
 '给[[NAME:a]]这份文档采用[[NAME:b]]这个新名字',
 '原来叫[[NAME:a]]，现在想叫[[NAME:b]]，帮我改好',
 '把[[NAME:a]]（只改名称）这份文件改叫[[NAME:b]]',
 '[[NAME:a]]（别动内容）这个文档，新的名称是[[NAME:b]]',
 '文档的新名称选[[NAME:b]]，原名称是[[NAME:a]]',
 '[[NAME:a]]这个名称不要换成别的，保持这个名字',
 '我希望[[NAME:a]]作为文件名保持不变',
 '[[PATH:a]]是完整文件路径，打开这个位置的文件',
 '[[PATH:a]]表示文件的完整地址，不是文件名称',
 '文件的完整路径用[[PATH:a]]表示',
 '[[PATH:a]]这个完整路径对应的文件需要读取',
 '要读取的文件完整地址为[[PATH:a]]',
 '[[PATH:a]]是源路径，[[PATH:b]]是复制后的目标路径',
 '源文件位于完整路径[[PATH:a]]，目标完整路径为[[PATH:b]]',
 '完整地址采用[[PATH:a]]，这个位置下的文件请删除',
 '[[TEXT:a]]是搜索用的原文，不是文件名称',
 '[[TEXT:a]]这段文字需要出现在搜索结果中',
 '需要查询的文字是[[TEXT:a]]，先在所有文档里查',
 '检索目标是一段文字，具体为[[TEXT:a]]',
 '[[TEXT:a]]用作全文查找的词句',
 '[[TEXT:a]]这段原文作为搜索条件',
 '我说的查找内容指[[TEXT:a]]这段文字',
 '[[TEXT:a]]是写入的正文，把它加到当前文档',
 '要写进去的是[[TEXT:a]]这段原文',
 '用[[TEXT:a]]这段文字补充当前文件的正文',
 '[[TEXT:a]]（按原样保留）这段文字追加到文件[[NAME:b]]',
 '[[NAME:a]]这份文档需要添加[[TEXT:b]]这段原文',
 '原文采用[[TEXT:b]]，写入对象是名为[[NAME:a]]的文件',
 '[[TEXT:a]]是正文，[[NAME:b]]是承载它的文档名字',
 '[[NAME:a]]是文档名，要查的是[[TEXT:b]]这段原文',
 '[[TEXT:b]]作为搜索词，查找范围是名为[[NAME:a]]的文件',
 '操作目标表示为[[VALUE:a]]，目前没有说明它是名字还是路径',
 '需要操作的目标是[[VALUE:a]]，具体类型没有交代',
 '[[VALUE:a]]是用户给出的目标标识，暂时不说明类型',
 '请处理这个目标：[[VALUE:a]]，类型待确认',
 '[[VALUE:a]]作为源目标，[[VALUE:b]]作为目的目标，类型尚未说明',
 '参数只给了[[VALUE:a]]，名字或路径暂不区分',
 '文件名称设定为[[NAME:a]]，不要把名称里的动作词当命令',
 '[[NAME:a]]这个文件我只想打开，暂时不要删除',
 '需要移动的文档叫[[NAME:a]]，接收它的目录叫[[NAME:b]]',
 '[[NAME:b]]是接收目录的名称，请把名为[[NAME:a]]的文件复制过去',
 '[[NAME:a]]（这是一份文档）复制到[[NAME:b]]这个目录中',
 '目的文件夹以[[NAME:b]]命名，复制对象以[[NAME:a]]命名',
 '文件位于[[PATH:a]]这个完整路径，只读取内容',
 '[[NAME:a]]是待处理文件名；先读取，再在其中查[[TEXT:b]]这段原文',
 '创建一个文档，以[[NAME:a]]命名，并以[[TEXT:b]]作为正文',
 '[[NAME:a]]和[[NAME:b]]分别是两份文件的名字，都打开',
]
VALIDATION = [
 '[[NAME:a]]才是文件名，需要做的是读取',
 '要打开的是用[[NAME:a]]命名的那份文件',
 '目录名称由[[NAME:a]]确定，不表示目录地址',
 '[[NAME:a]]这个名称对应的文件，给我看看',
 '[[NAME:a]]是目前的文件名，准备采用[[NAME:b]]这个新名',
 '请用[[NAME:b]]这个名称替代文档原来的名字[[NAME:a]]',
 '把[[NAME:a]]（其他不修改）这份文档的名字换为[[NAME:b]]',
 '[[PATH:a]]才是文件完整地址，读取这个地址的文件',
 '完整文件地址指定[[PATH:a]]，名称暂时不提供',
 '从[[PATH:a]]这个完整路径复制到[[PATH:b]]这个完整路径',
 '[[TEXT:a]]才是需要查找的原文，不是名称',
 '搜索需要匹配[[TEXT:a]]这段正文',
 '文档以[[NAME:a]]命名，正文采用[[TEXT:b]]',
 '[[TEXT:a]]这段文字写到用[[NAME:b]]命名的文档里',
 '目前操作目标为[[VALUE:a]]，其类型未知',
 '[[VALUE:a]]是源标识而[[VALUE:b]]是目标标识，类型没有提供',
]
CALIBRATION = [
 '[[NAME:a]]给出了文档名称，打开它',
 '我想用[[NAME:a]]作为文件夹名字',
 '文档原名由[[NAME:a]]给出，新名由[[NAME:b]]给出',
 '要替换的是[[NAME:a]]这个文档的名称，改成[[NAME:b]]',
 '[[PATH:a]]给出了完整路径，打开这条路径下的文件',
 '用[[PATH:a]]表示完整文件路径，请读取该文件',
 '[[TEXT:a]]给出了检索原文，请搜索',
 '[[TEXT:a]]这段正文不要改动，写入当前文档',
 '名为[[NAME:a]]的文档以[[TEXT:b]]作为新增正文',
 '[[VALUE:a]]给出了目标但没说具体类型',
 '目的目录称为[[NAME:b]]，源文档称为[[NAME:a]]，复制它',
 '把[[NAME:a]]（只处理这份）这个文件打开',
]
TEST = [
 '需要打开的文件采用[[NAME:a]]作为名称',
 '[[NAME:a]]（就是那份文档）只读取就好',
 '把[[NAME:a]]这个名字所指的文件打开',
 '[[NAME:a]]这几个字指文件名称，我要的操作是删除',
 '创建一个文档，用[[NAME:a]]这几个字命名',
 '[[NAME:a]]是那份文档现有的名字；想改成[[NAME:b]]',
 '文档原来命名成[[NAME:a]]，现在采用[[NAME:b]]作为名称',
 '使用[[NAME:b]]作为新名字，替换[[NAME:a]]这个文档的原名字',
 '[[PATH:a]]给的是文件完整地址，查看这个地址的内容',
 '打开文件，完整地址由[[PATH:a]]给出',
 '[[PATH:a]]这个完整地址指向的文件，先别删除',
 '完整路径由[[PATH:a]]给出，目标完整路径由[[PATH:b]]给出，复制一份',
 '[[TEXT:a]]这段原文才是要搜索的内容',
 '查询的是正文，正文原样是[[TEXT:a]]',
 '[[TEXT:a]]作为新增的正文，请追加到当前文档',
 '向用[[NAME:a]]命名的文件，补上[[TEXT:b]]这段文字',
 '[[TEXT:b]]这段正文交给名为[[NAME:a]]的文件保存',
 '[[VALUE:a]]仅表示要操作的对象，具体属于名字还是路径没有交代',
 '[[VALUE:a]]作为传入参数，具体类别暂时未知',
 '把以[[NAME:a]]命名的文档，复制到以[[NAME:b]]命名的目录',
]
NONE_TRAIN = [
 '删除是操作名称，先不要执行', '重命名只是命令名称，不是文件名字',
 '我在解释打开操作，没有提供任何文件名', '文件名称这个概念不是具体名称',
 '路径只是参数的类型名称，没有具体地址', '文字只是参数的类别，不是要搜索的原文',
 '没有提供文件名，所以先不操作', '只查看文件属性，没有指定具体文件名',
 '名字里可以有动作词，我现在没有给出名字', '暂时不给原文，先不要查找',
 '只是解释复制命令，不是在指定文件', '操作名称不是文件名称，不要混淆',
 '目录名称还没有想好，不要创建', '完整路径还没有提供，等一下',
 '搜索词尚未确定，暂时不要查询', '创建和删除都是操作，不是要创建的名字',
]
NONE_VAL = ['这里的打开说的是操作类别，未指定名称', '不是让你提取命令名称作为文件名',
            '正文尚未提供，不要随意填充', '需要先确认文件名，当前没有具体参数']
NONE_CAL = ['操作名不能当作实际参数', '现在只谈文件名的格式，没有给出具体名称',
            '目前尚缺完整路径', '未给出要写入的具体文字']
NONE_TEST = ['我说的删除属于操作名称，具体文件名还没告诉你',
             '提到文件名称这几个字是在解释概念，不是在给出名字',
             '目标路径没有指定，先别猜位置',
             '我没有提供要查找的原文，请先等待']


def instances(frames, n, namespace, rng, group='semantic'):
    rows=[]
    for i in range(n):
        pattern=frames[i%len(frames)]
        values={m.group(2):literal(rng,m.group(1),namespace,i,True) for m in SLOT_RE.finditer(pattern)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.5 else ('','') for k in values}
        text, spans=render(pattern,values,quotes)
        prefix=rng.choice(['','嗯，','麻烦你，','呃，','我想这样：'])
        text=prefix+text
        spans=[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans]
        if len(text)>256:
            continue
        rows.append({'id':f'{namespace}_{i:06}', 'text':text, 'spans':spans,'family':f'{namespace}_{i%len(frames):03}',
                     'template':pattern,'category':group,'source':'semantic_augmentation'})
    return rows


def generate(directory='data/semantic',seed=631):
    dest=Path(directory); dest.mkdir(parents=True,exist_ok=True)
    test=dest/'challenge.jsonl'
    if not test.exists():
        rng=random.Random(seed)
        rows=instances(TEST,400,'heldout_semantic',rng)
        # Different wrappers give 100 unique no-parameter sentences while the
        # underlying four families stay held out from training/development.
        neg=[]
        for pattern in NONE_TEST:
            for p in ('','嗯，','麻烦你，','呃，','我想这样：'):
                for suffix in ('','。','，谢谢','，不用操作','，等我补充'):
                    neg.append({'id':f'semantic_none_{len(neg):03}', 'text':p+pattern+suffix,'spans':[],
                        'template':pattern,'family':'heldout_none_'+str(NONE_TEST.index(pattern)),
                        'category':'semantic_no_literal','source':'assistant_authored_semantic'})
        rows+=neg
        write_jsonl(test,rows)
        (dest/'challenge_manifest.json').write_text(json.dumps({'rows':len(rows),
            'sha256':hashlib.sha256(test.read_bytes()).hexdigest(),'frozen_before_semantic_training':True,
            'positive_families':len(TEST),'negative_families':len(NONE_TEST),
            'test_families_disjoint':True,'syntax_sha256':hashlib.sha256(Path('filter10/syntax.py').read_bytes()).hexdigest(),
            'scope':'task-specific explicit parameter roles; not unrestricted reasoning'},indent=2),encoding='utf-8')
    assert hashlib.sha256(test.read_bytes()).hexdigest()==json.loads((dest/'challenge_manifest.json').read_text())['sha256']
    splits={}
    for name,positive,negative,count,offset in [
        ('train',TRAIN,NONE_TRAIN,24000,1),('validation',VALIDATION,NONE_VAL,800,2),
        ('calibration',CALIBRATION,NONE_CAL,600,3)]:
        rng=random.Random(seed+offset)
        rows=instances(positive,count,name+'_semantic',rng)
        rows+=instances(negative,max(150,count//10),name+'_none',rng,'semantic_no_literal')
        splits[name]=rows
    reserved={r['text'] for r in read_jsonl(test)}
    for name in ('validation','calibration'):
        splits[name]=[r for r in splits[name] if r['text'] not in reserved]
        reserved.update(r['text'] for r in splits[name])
    splits['train']=[r for r in splits['train'] if r['text'] not in reserved]
    for name,rows in splits.items():
        write_jsonl(dest/(name+'.jsonl'),rows)
    manifest={'seed':seed,'no_test_error_mining':True,'families_disjoint':True,
        'train_templates':TRAIN+NONE_TRAIN,'validation_templates':VALIDATION+NONE_VAL,
        'calibration_templates':CALIBRATION+NONE_CAL,'test_templates':TEST+NONE_TEST,
        'splits':{name:{'rows':len(rows),'sha256':hashlib.sha256((dest/(name+'.jsonl')).read_bytes()).hexdigest()}
                  for name,rows in splits.items()}}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

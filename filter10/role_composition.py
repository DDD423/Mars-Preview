"""Training-only role/operation composition, motivated by development errors.

Not an inference grammar. Explicit names/values are filled after selecting the
roles; reserved full frames and sentences remain excluded.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, read_jsonl, write_jsonl
from .noisy_data import literal
from .role_augmentation import ROLES, COPULA, PREFIXES
from .semantic_data import VALIDATION, CALIBRATION, TEST, NONE_VAL, NONE_CAL, NONE_TEST

ROLE_WORDS={k:list(v) for k,v in ROLES.items()}
ROLE_WORDS['NAME']+=['名称','名字','新名','原名','旧文件名','新名字','文档原来的名字','现在的名字']
ROLE_WORDS['PATH']+=['文件完整路径','文件地址','完整的文件地址','这份文档的完整地址','接收文件的完整地址']
ROLE_WORDS['TEXT']+=['这段正文','这段原文','要查找的文字','需要匹配的正文','新增文字','追加内容']
ROLE_WORDS['VALUE']+=['尚未确定类别的标识','未知类型的目标','目前的操作目标']

FOLLOWUPS={
 'NAME':['打开这个名字所指的文档','读取这个名称对应的文件','预览这份文档','创建一份文档',
         '采用这个名字','别把名称当成命令','只是名字，不是操作名称','需要做的是打开','需要做的是读取'],
 'PATH':['读取这个地址下的文件','打开这个位置的文档','查看这个地址对应的内容','访问这条路径下的文档',
         '删除这个位置的文件','先不要移动此处的文件','不要把地址当成名称','需要读取该地址对应的文件'],
 'TEXT':['检索这段原文','匹配文档中的内容','将原文写入当前文档','追加到当前文档',
         '别把正文当成名称','需要做的是搜索','需要做的是写入','这段内容原样保留'],
 'VALUE':['其类型还待确认','类别仍然未知','并未明确具体类型','现在不区分名字与路径',
          '还不知道目标属于什么类型','先保留标识，不猜对象','缺少类型说明，等待确认']}


def clause(rng,kind,key):
    slot='[['+kind+':'+key+']]';role=rng.choice(ROLE_WORDS[kind])
    # Whole clauses combine roles with varying surrounding actions. The
    # classification must not be dominated by a distant "file" or "name".
    return rng.choice([
        slot+rng.choice(COPULA)+role,
        rng.choice(['目前','本次','提供的','当前','用户给的','需要采用的',''])+role+
            rng.choice(['为','采用','指定','设为','用','就是','选','给定为'])+slot,
        role+rng.choice(['由','选用','以'])+slot+rng.choice(['给出','确定','表示','指定','来表示']),
        slot+rng.choice(['这个','这一','这份'])+role,
        '请用'+slot+'作为'+role,
        slot+'表示'+role,
    ])


def frame(rng):
    kind=rng.choice(list(ROLE_WORDS));x='[['+kind+':a]]';role=rng.choice(ROLE_WORDS[kind])
    choice=rng.randrange(6)
    if choice<3:
        return clause(rng,kind,'a')+rng.choice(['，','；','。','，嗯，'])+rng.choice(FOLLOWUPS[kind])
    if choice==3:
        other=rng.choice(list(ROLE_WORDS))
        return clause(rng,kind,'a')+rng.choice(['，','；','，另外，','；同时，'])+clause(rng,other,'b')+'，参数均原样保留'
    if choice==4:
        if kind=='NAME':
            return rng.choice(['我想查看','先读取','预览','保留'])+'以'+x+'命名的文档，先保持内容不变'
        return '请处理'+x+'这个'+role+'，'+rng.choice(FOLLOWUPS[kind])
    if kind=='NAME':
        y='[[NAME:b]]';remark=rng.choice(['先不要调整内容','只修改名字','这份是文档','原文不改变','不要修改正文'])
        return (rng.choice(['把','请将','先把',''])+x+'（'+remark+'）'+rng.choice(['这份文件','这个文档','这个文件'])+
                rng.choice(['改叫','将名称改为','取新名','修改名字为'])+y)
    return x+'（这个参数原样保留）'+role+'，'+rng.choice(FOLLOWUPS[kind])


def generate(directory='data/neural_v3',count=42000,seed=947):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic') for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    rows=[]
    for i in range(count):
        template=frame(rng)
        if i%9==0:
            role=rng.choice(ROLE_WORDS[rng.choice(list(ROLE_WORDS))])
            operation=rng.choice(['打开','读取','复制','删除','重命名','创建','搜索','写入'])
            template=rng.choice(['现在只是讨论'+role+'，尚未给出具体参数',
                role+'不是具体对象，请等用户提供原文','操作类别可以解释，但'+role+'还没有提供',
                operation+'是命令的名称；没有给文件提供名字',
                '需要区分'+operation+'这种动作和实际文档名称，暂不指定参数',
                role+'是槽类型，当前并不提供这个槽的值',
                '目前只说明'+role+'的意义，参数还缺少',
                '操作'+operation+'正在介绍，用户未指定具体对象',
                '先解释'+operation+'，不要把动作词作为文件的命名结果'])
        if template in reserved_frames:continue
        vals={m.group(2):literal(rng,m.group(1),'rolecompose',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.5 else ('','') for k in vals}
        text,spans=render(template,vals,quotes);prefix=rng.choice(PREFIXES)
        row={'id':f'rolecompose_{i:06}','text':prefix+text,
             'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
             'template':template,'category':'semantic_role_composition','source':'semantic_augmentation'}
        if len(row['text'])<=256 and row['text'] not in reserved:rows.append(row)
    replay=read_jsonl('data/neural_v2/train.jsonl');rng.shuffle(replay)
    # Replay both noise and original semantic frames rather than replacing them.
    noise=[r for r in replay if not r.get('category','').startswith('semantic')]
    semantic=[r for r in replay if r.get('category','').startswith('semantic')]
    rows+=noise[:32000]+semantic[:32000];rng.shuffle(rows)
    write_jsonl(dest/'train.jsonl',rows)
    for s in ('validation','calibration'):
        write_jsonl(dest/(s+'.jsonl'),read_jsonl(Path('data/neural_v2')/(s+'.jsonl')))
    report={'seed':seed,'rows':len(rows),'name_dictionary':False,'test_error_mining':False,
            'strategy':'development-driven role and operation composition; replay prior data',
            'held_out_full_frames_excluded':True,'abstract_role_structures_overlap':True,
            'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report

"""Training-only composition of argument phrases and operation clauses.

Motivated by development boundary errors (including/excluding function words).
No inference rules or test-error examples are generated here.
"""
import hashlib,json,random
from pathlib import Path
from .data import SLOT_RE,QUOTES,render,read_jsonl,write_jsonl
from .noisy_data import literal
from .semantic_data import VALIDATION,CALIBRATION,TEST,NONE_VAL,NONE_CAL,NONE_TEST,NONE_TRAIN


def argument(rng,kind,key):
    x='[['+kind+':'+key+']]'
    if kind=='NAME':
        noun=rng.choice(['文件','文档','那份文件','这份文档','目录','文件夹'])
        return rng.choice(['名为'+x+'的'+noun,'叫做'+x+'的'+noun,
            '以'+x+'命名的'+noun,'用'+x+'命名的'+noun,
            x+'这个名字对应的'+noun,x+'这个名称所指的'+noun,
            x+'（正文暂时不改）这个'+noun,x+'这个'+noun])
    if kind=='PATH':
        return rng.choice([x+'这个完整路径',x+'这一完整地址',
            x+'这个完整地址对应的文档','路径为'+x+'的文件',
            '完整地址为'+x+'的文档','完整路径'+x+'下的文件'])
    if kind=='TEXT':
        return rng.choice([x+'这段原文',x+'这段正文',x+'这段文字',
                           '原文'+x,'文字'+x,'正文'+x])
    return rng.choice([x+'这个未知类型参数',x+'这个目标标识','类型待确认的目标'+x])


def frame(rng):
    op=rng.randrange(11)
    if op==0:
        return rng.choice(['要读取的是','需要打开的是','实际想预览的是','请打开','我想查看','保留'])+argument(rng,'NAME','a')
    if op==1:
        kind=rng.choice(['NAME','PATH'])
        return rng.choice(['从','先从','请从','现在从'])+argument(rng,kind,'a')+rng.choice(['复制到','移动给','备份到'])+argument(rng,kind,'b')
    if op==2:
        return rng.choice(['向','给','往'])+argument(rng,'NAME','a')+rng.choice(['补充','添加','追加','写入'])+argument(rng,'TEXT','b')
    if op==3:
        return argument(rng,'TEXT','a')+rng.choice(['写入','追加到','保存到','补充到'])+argument(rng,'NAME','b')+rng.choice(['里','中',''])
    if op==4:
        return rng.choice(['搜索要匹配','查询需要找出','检索要包含','查找内容采用'])+argument(rng,'TEXT','a')
    if op==5:
        return rng.choice(['这个文档','文档','文件'])+rng.choice(['用','以'])+'[[NAME:a]]命名，'+rng.choice(['正文用','新增内容为','原文采用','追加的正文指定为'])+'[[TEXT:b]]'
    if op==6:
        return argument(rng,'NAME','a')+rng.choice(['的名称改为','取新名称','改名为','的名字换成','名字换为','名称调整为'])+'[[NAME:b]]'+rng.choice(['这个新名字','这个新名称',''])
    if op==7:
        return rng.choice(['采用','请使用','要用','取'])+'[[NAME:b]]'+rng.choice(['这个名字','这个名称',''])+rng.choice(['替换','代替','作为新名称取代'])+argument(rng,'NAME','a')+'的原名称'
    if op==8:
        return argument(rng,'PATH','a')+'中的文件请读取，先不修改'
    if op==9:
        x='[[VALUE:a]]'
        return rng.choice(['目前的参数给定为'+x+'，该参数尚未分类',
            x+'所表示的是操作目标；尚不知道属于哪类标识',
            '传入标识采用'+x+'，没有明确类型',
            x+'仅给出目标对象的标识，类型信息仍缺少'])
    return '请保留'+argument(rng,'NAME','a')+'，然后在其中搜索'+argument(rng,'TEXT','b')


def generate(directory='data/neural_v5',count=38000,seed=1223):
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=True);rng=random.Random(seed)
    reserved_frames=set(VALIDATION+CALIBRATION+TEST+NONE_VAL+NONE_CAL+NONE_TEST)
    reserved={r['text'] for d in ('data/noisy','data/semantic') for s in ('challenge','validation','calibration')
              for r in read_jsonl(Path(d)/(s+'.jsonl'))}
    rows=[]
    for i in range(count):
        template=frame(rng)
        if i%10==0:
            operation=rng.choice(['打开','删除','复制','重命名','创建','搜索','写入','读取'])
            template=rng.choice(NONE_TRAIN+[
                '不是要把动作的名称作为文档名字，请先说明操作类别',
                '别把命令的名字直接当成文件名称，现在还没有目标',
                '请先解释'+operation+'的含义，先不指定参数',
                '当前解释的是'+operation+'这种动作，尚未说明操作对象',
                '命令名称只用于说明操作方式，不提供实际文件标识'])
        if template in reserved_frames:continue
        vals={m.group(2):literal(rng,m.group(1),'constituent_training',i,True) for m in SLOT_RE.finditer(template)}
        quotes={k:rng.choice(QUOTES[1:]) if rng.random()<.4 else ('','') for k in vals}
        text,spans=render(template,vals,quotes);prefix=rng.choice(['','嗯，','呃，','麻烦一下，','先这样：'])
        row={'id':f'constituent_{i:06}','text':prefix+text,
             'spans':[dict(s,start=s['start']+len(prefix),end=s['end']+len(prefix)) for s in spans],
             'template':template,'category':'semantic_constituent','source':'semantic_augmentation'}
        if len(row['text'])<=256 and row['text'] not in reserved:rows.append(row)
    replay=read_jsonl('data/neural_v4/train.jsonl');rng.shuffle(replay)
    for group,n in [('clean',24000),('semantic',24000),('noisy',24000)]:
        select=lambda r: ('semantic' if r.get('category','').startswith('semantic') else 'clean' if r.get('category','').startswith('basic') else 'noisy')==group
        rows += [r for r in replay if select(r)][:n]
    rng.shuffle(rows);write_jsonl(dest/'train.jsonl',rows)
    for split in ('validation','calibration'):
        write_jsonl(dest/(split+'.jsonl'),read_jsonl(Path('data/neural_v4')/(split+'.jsonl')))
    report={'rows':len(rows),'seed':seed,'strategy':'development-driven argument phrase and operation clause composition; three-group replay',
        'test_error_mining':False,'held_out_full_frames_excluded':True,'abstract_linguistic_parts_overlap':True,
        'sha256':{s:hashlib.sha256((dest/(s+'.jsonl')).read_bytes()).hexdigest() for s in ('train','validation','calibration')}}
    (dest/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

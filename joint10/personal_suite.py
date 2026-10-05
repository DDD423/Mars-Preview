"""Forty separately authored raw prompts with ordinary, unsuffixed names.

Frozen during posttraining, before any predictions on these cases. Report
separately from the primary 580 synthetic-family acceptance set.
"""
import hashlib,json,re
from pathlib import Path
from translator10.data import ROLES,actions,plan,slot,save_rows
from translator10.language import serialize,FILE_TOOLS,nonexecution

CASES=[
 ('read','打开叫做%A%的文件看看全文',{'A':'删除'}),
 ('read','麻烦把文件%A%的文字给我显示一下',{'A':'会议记录😀.txt'}),
 ('trash','将名为%A%的文件放进回收站',{'A':'打开'}),
 ('rename','把叫做%A%的文件改名为%B%',{'A':'删除','B':'请勿删除.txt'}),
 ('rename','旧文件%A%更名为%B%，位置不用改',{'A':'草稿.txt','B':'最终版.txt'}),
 ('copy','文件%A%复制到完整路径%B%，原件留下',{'A':'采购清单.txt','B':'./备份/采购清单.txt'}),
 ('move','把文件%A%移动到完整地址%B%',{'A':'记录.txt','B':'./归档/记录.txt'}),
 ('copy_into','将文件%A%拷贝进目录%B%里',{'A':'读我.txt','B':'我的资料'}),
 ('move_into','把文件%A%搬到目录%B%里面',{'A':'合同.txt','B':'归档目录'}),
 ('mkdir','创建名为%A%的目录',{'A':'我的 文档😀'}),
 ('empty','新建一个名为%A%的空白文本文件',{'A':'打开删除复制.txt'}),
 ('write','新建文件%A%，内容为%B%',{'A':'笔记.txt','B':'不要删除，这是正文。\n第二行😀'}),
 ('write','把文字%B%保存为新文件%A%',{'A':'a {1} b.txt','B':'字面 {1} {2} 应完整保存'}),
 ('overwrite','允许覆盖路径%A%，全文改为%B%',{'A':'./文档/旧稿.txt','B':'全新的文字，保留空格 A B 。'}),
 ('append','向文件%A%末尾追加文字%B%，保留原文',{'A':'日志.txt','B':'\n新的记录😀'}),
 ('read_write','读取文件%A%的原文，再写入新文件%B%',{'A':'内容.txt','B':'./导出/副本.txt'}),
 ('copy_rename','复制文件%A%到完整路径%B%，然后把副本改名为%C%',{'A':'报告.txt','B':'./临时/副本.txt','C':'报告 完整版.txt'}),
 ('conditional','如果路径%A%存在，才读取它',{'A':'./文档/说明.txt'}),
 ('search_name','搜索文件名包含%A%的文件',{'A':'备份'}),
 ('search_text','搜索正文包含%A%的文件',{'A':'不要删除我的内容'}),
]
NEGATIVE=[
 ('不要删除叫做%A%的文件',{'A':'删除'}),('禁止移动文件%A%到完整路径%B%',{'A':'草稿.txt','B':'./文档/草稿.txt'}),
 ('别重命名文件%A%为%B%',{'A':'旧稿.txt','B':'新稿.txt'}),('不允许复制文件%A%到完整地址%B%',{'A':'报告.txt','B':'./导出/报告.txt'}),
 ('不要向文件%A%追加文字%B%',{'A':'日志.txt','B':'删除文件'}),('请勿覆盖文件%A%，内容%B%',{'A':'记录.txt','B':'新的全文'}),
 ('所有文件操作都取消',{}),('这次先不要执行，文件保持原样',{}),('别动任何文件了，取消',{}),('不用处理文件了，停下这次任务',{}),
]
UNCLEAR=[
 ('删除那个文件','UNCLEAR_REFERENCE'),('把上次提到的文件改个名字','UNCLEAR_REFERENCE'),('请把它复制到之前那个目录','UNCLEAR_REFERENCE'),
 ('打开刚才的那个文档','UNCLEAR_REFERENCE'),('创建一个文件，名字我还没想好','MISSING_PARAMETER'),('重命名文件，但是没给新名字','MISSING_PARAMETER'),
 ('帮我在终端运行一个命令','UNSUPPORTED_TASK'),('写一个程序然后运行','UNSUPPORTED_TASK'),('文件要搬走，具体位置还没决定','AMBIGUOUS_TARGET'),('一次连续执行十个操作','TOO_COMPLEX'),
]

def annotated(template,roles,values,target=None,category='refusal'):
    spans=[];bindings={};segments=[];text='';mapping={};cursor=0
    for match in re.finditer(r'%([ABC])%',template):
        literal=template[cursor:match.start()];text+=literal;segments.append({'kind':'literal','text':literal})
        letter=match.group(1);value=values[letter];number=len(spans)+1;placeholder='{'+str(number)+'}';start=len(text);text+=value
        spans.append({'start':start,'end':len(text),'text':value,'type':roles[letter],'placeholder':placeholder})
        bindings[placeholder]={'value':value,'type':roles[letter],'start':start,'end':len(text)}
        segments.append({'kind':'slot','id':placeholder});mapping.setdefault(letter,slot(number));cursor=match.end()
    text+=template[cursor:];segments.append({'kind':'literal','text':template[cursor:]})
    target=target or actions(category,mapping);request={'segments':segments,'slots':[{'slot':i+1,'type':s['type']} for i,s in enumerate(spans)],'enabled_tools':list(FILE_TOOLS)}
    return {'raw':text,'spans':spans,'bindings':bindings,'request':request,'target':target,'tokens':serialize(target),'category':category,'family':'personal_frozen'}

def freeze(out='data/joint-personal-frozen.jsonl'):
    destination=Path(out)
    if destination.exists():raise RuntimeError('Personal suite already frozen')
    rows=[annotated(t,ROLES[c],v,category=c) for c,t,v in CASES]
    for t,v in NEGATIVE:
        roles={key:('PATH' if '完整' in t and key=='B' else 'TEXT' if ('文字' in t or '内容' in t) and key=='B' else 'NAME') for key in v}
        rows.append(annotated(t,roles,v,nonexecution('noop','CANCELLED_REQUEST')))
    rows += [annotated(t,{}, {},nonexecution('clarification',reason)) for t,reason in UNCLEAR]
    info=save_rows(destination,rows);info.update(authored_raw_prompts=True,synthetic_name_suffixes=False,never_training_or_calibration=True)
    destination.with_suffix('.manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');return info

if __name__=='__main__':print(json.dumps(freeze()))

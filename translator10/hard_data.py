"""Validation-driven augmentation. Does not read test or independent suites."""
import copy
import json
import random
import re
from pathlib import Path
from .data import ROLES, PREFIXES, SUFFIXES, example, rejection, compose, plan, step, slot, result, save_rows, add_variations, load_rows
from .language import serialize, canonical, masks, FILE_TOOLS, nonexecution

# New training expressions for validation failure phenomena: colloquial intent,
# scope, reversed arguments and reference chains. No inference parsing rules.
HARD={
"list":["给我目录%A%的列表", "我需要看看目录%A%现在有什么", "把目录%A%展开列清单", "查看目录%A%的一层项目"],
"read":["我需要知道文件%A%里面的文字", "只看文件%A%正文，不要修改", "不删除文件%A%，打开读它", "我想从文件%A%获取内容"],
"search_name":["我需要名称带%A%的文件", "查找文件，按名字关键字%A%筛选", "文件名出现%A%的都列出来", "用名称片段%A%找文件"],
"search_text":["我需要正文带%A%的文件", "查找文件，按内容关键字%A%筛选", "文件内容出现%A%的都列出来", "用文字片段%A%找文件"],
"mkdir":["我想要一个新文件夹%A%", "需要新目录%A%，请建好", "帮我弄一个名称为%A%的目录", "新增一个目录，叫%A%"],
"empty":["我想要一个新空白文本%A%", "需要空文件%A%，请建好", "帮我弄一个名称为%A%的空白文件", "新增一个空文本，叫%A%"],
"write":["我需要新建文本%A%，文字是%B%", "先说内容%B%，再说新文件名%A%，请创建", "帮我保存%B%，需要新文件%A%", "我想要新文件%A%内写上%B%"],
"overwrite":["文件%A%不再保留旧文，写成%B%", "我需要替换文件%A%全部文字为%B%", "内容%B%覆盖到原文件%A%", "将文件%A%整段正文更新为%B%，允许覆盖"],
"append":["文件%A%保留旧文，接着加%B%", "我需要在文件%A%尾部增添%B%", "内容%B%追加到原文件%A%", "将文件%A%正文末尾补写%B%，不覆盖"],
"copy":["我想要复制，来源文件%A%，完整目标路径%B%", "副本完整路径%B%，拷贝来源文件%A%", "我需要从文件%A%拷出一份，完整目的地址%B%", "文件%A%原处保留，复制后完整路径是%B%"],
"copy_into":["我想要复制，来源文件%A%，目标目录%B%", "目的目录%B%，拷贝来源文件%A%并保留名称", "我需要文件%A%的一份副本放在%B%目录", "文件%A%原处保留，副本收纳在目录%B%"],
"move":["我想要移动，来源文件%A%，完整目标路径%B%", "迁移目的完整路径%B%，原文件%A%", "我需要从文件%A%迁过去，完整目的地址%B%", "文件%A%原处不留，移动后完整路径是%B%"],
"move_into":["我想要移动，来源文件%A%，目标目录%B%", "迁入目的目录%B%，原文件%A%并保留名称", "我需要把文件%A%移走放在%B%目录", "文件%A%原处不留，移入目录%B%收纳"],
"rename":["我想要重命名，原文件%A%，新名称%B%", "新名称%B%，改名来源文件%A%", "我需要让文件%A%换一个名字%B%", "文件%A%位置保留，名字换成%B%"],
"trash":["我想要删除，目标文件%A%，使用回收", "需要移除文件%A%到回收目录", "文件%A%删掉但是要能恢复", "把文件%A%收走，放入回收目录"],
"restore":["我想要恢复，记录编号%A%", "需要找回删除记录%A%的文件", "删除记录%A%还原到原来的位置", "撤销回收记录%A%并还原文件"],
"read_write":["我需要先读文件%A%再把它的文字写入新文件%B%", "来源文件%A%的正文读取后，用来创建文件%B%", "先读%A%得到内容，接着原样存为新文件%B%", "新文件%B%使用来源%A%实际读出的文字"],
"copy_rename":["我需要文件%A%的副本，路径%B%，再让副本名称变为%C%", "来源%A%拷贝到%B%之后，把刚才的副本叫%C%", "拷完文件%A%为%B%，再给复制结果换名%C%", "先复制%A%为%B%，下一步只重命名这份副本为%C%"],
"conditional":["找得到路径%A%才读文件%A%，否则跳过", "我需要查看文件%A%，但只在它存在时读取", "判断路径%A%存在后才读它", "若有文件%A%，再读取原文"],
"refusal":["这次文件任务作罢，不用弄了", "不要执行，保留原状就好", "我决定取消所有文件指令", "别处理了，我撤回请求"],
}
REJECT={
"MISSING_PARAMETER":["搜索一下但搜索词没说", "创建文件但名字待补充", "我想读取，但还没给文件", "追加到文件，具体文件和内容待补充"],
"UNCLEAR_REFERENCE":["读取上一次说的那个", "把之前那个文件更名", "它的内容存到那里", "沿用昨天指定的那个文件"],
"AMBIGUOUS_TARGET":["对文件%A%做点事情，怎么操作待定", "我需要整理文件%A%，还没确定方法", "复制文件%A%到%B%，这可能是目录也可能是新文件", "文件%A%改一下，但我没说明要改什么"],
"UNSUPPORTED_TASK":["用终端执行%A%", "我想编程实现%A%", "请识别PDF中的文字%A%", "控制浏览器去访问%A%"],
"TOO_COMPLEX":["任务有五步以上，全部做完", "需要连续执行九个步骤", "请自动反复重试直到成功", "需要执行超过四个顺序操作"],
}

def rewrite_words(template,rng):
    # Curated equivalent lexical variants; no character corruption across action
    # boundaries, which would manufacture ambiguous training labels.
    swaps={"复制":["拷贝","复制","拷一份"],"移动":["移走","移动","搬迁","挪动"],"迁移":["迁移","搬移"],"新目录":["新目录","新文件夹"],"创建":["创建","建立","生成"],"重命名":["重命名","改名","更名"],"删除":["删除","回收移除"],"正文":["正文","文字内容","全文"],"需要":["需要","想要","打算要"],"我想要":["我要","我想要","我打算"]}
    for key,values in swaps.items():
        if key in template:template=template.replace(key,rng.choice(values))
    return template

def build_hard(data_dir,out_path,seed=2718,count=30000):
    root=Path(data_dir);errors=json.loads((Path("artifacts/translator-round1")/"validation_errors.json").read_text(encoding="utf-8"))
    error_counts={}
    for e in errors:error_counts[e["category"]]=error_counts.get(e["category"],0)+1
    # Weight difficult phenomena, retain every tool to avoid forgetting.
    cats=list(HARD);weights=[1+min(5,error_counts.get(c,0)/60) for c in cats]
    rng=random.Random(seed);rows=[]
    for i in range(count):
        family="hard2_"+str(i%240)
        if i%5==0:
            reason=rng.choice(list(REJECT));text=rewrite_words(rng.choice(REJECT[reason]),rng)
            e=rejection(rng.choice(PREFIXES)+text+rng.choice(SUFFIXES),reason,rng,family)
        else:
            cat=rng.choices(cats,weights=weights)[0];text=rewrite_words(rng.choice(HARD[cat]),rng)
            e=example(rng.choice(PREFIXES)+text+rng.choice(SUFFIXES),ROLES[cat],cat,rng,family=family)
            if i%4==0 and cat!="refusal":
                other=rng.choice(cats[:-1]);text2=rewrite_words(rng.choice(HARD[other]),rng)
                # Training never sees held-out pair sum=0 mod4.
                if (cats.index(cat)+cats.index(other))%4!=0:
                    e=compose((text,cat),(text2,other),rng,family)
        rows.append(add_variations(e,rng))
    # Explicit path constructors and result-reference chains, including 4 steps.
    for i in range(3000):
        text=rng.choice(["在目录%A%里新建文件%B%，内容为%C%", "新文件%B%放在目录%A%里面，写入文字%C%", "把文字%C%写到目录%A%内新建的文件%B%"])
        e=example(text,{"A":"NAME","B":"NAME","C":"TEXT"},"write",rng,family="hard2_join_name")
        letters=[p[1] for p in re.findall(r"%[A-H]%",text)];s={k:slot(letters.index(k)+1) for k in letters}
        e["target"]=plan([step("fs.write_text",path={"join":{"directory":s["A"],"name":s["B"]}},text=s["C"])])
        e["tokens"]=serialize(e["target"]);rows.append(e)
    for i in range(2000):
        text="创建目录%A%，把文件%B%复制进去，再把副本改名为%C%，然后读取它"
        e=example(text,{"A":"NAME","B":"NAME","C":"NAME"},"copy_rename",rng,family="hard2_four_step_reference")
        e["target"]=plan([step("fs.mkdir",path=slot(1)),step("fs.copy",source=slot(2),destination={"join":{"directory":result(1),"basename_of":slot(2)}}),step("fs.rename",source=result(2),new_name=slot(3)),step("fs.read_text",path=result(3))]);e["tokens"]=serialize(e["target"]);rows.append(e)
    for i in range(1000):
        e=example("如果路径%A%不存在，就创建空文件%A%",{"A":"PATH"},"empty",rng,family="hard2_negated_condition")
        e["target"]=plan([dict(step("fs.write_text",path=slot(1),text=""),when={"exists":slot(1),"negate":True})]);e["tokens"]=serialize(e["target"]);rows.append(e)
    for e in rows:masks(e["request"],e["tokens"])
    info=save_rows(Path(out_path),rows);info.update(seed=seed,validation_error_categories=error_counts)
    Path(out_path).with_suffix(".manifest.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    return info

def build_real_filter(data_dir,hard_path,out_path,device="cuda",limit=16000):
    """Only train rows, and only when actual filter boundaries match known labels.

    Actual filter types may be VALUE/PATH instead of NAME. No contradictory gold
    action is assigned after a missing/wrong binding. Those rows are not guessed.
    """
    import torch
    torch.set_num_threads(4)
    from filter10 import FilterHarness
    from davework.adapter import planner_input
    filter=FilterHarness(device=device);rng=random.Random(31415)
    source=load_rows(Path(data_dir)/"train.jsonl")+load_rows(hard_path);rng.shuffle(source)
    output=[];rejected=0
    for e in source:
        if e["category"]=="composition":continue # compose raw span metadata is not a filter fixture
        f=filter.extract(e["raw"]).to_dict()
        expected=[(s["start"],s["end"],s["text"]) for s in e["spans"]]
        actual=[(s["start"],s["end"],f["original"][s["start"]:s["end"]]) for s in f["spans"]]
        if expected!=actual or f["status"] in ("uncertain","needs_context","unsupported"):
            rejected+=1
        else:
            r=planner_input({"id":"ctx_eval","filter":f,"settings":{"enabled_tools":e["request"]["enabled_tools"]}})
            try:masks(r,e["tokens"])
            except (ValueError,KeyError):rejected+=1;continue
            output.append(dict(e,request=r,family="real_filter_train"))
        if (len(output)+rejected)%500==0:print(json.dumps({"event":"real_filter_train","accepted":len(output),"rejected":rejected}),flush=True)
        if len(output)>=limit:break
    info=save_rows(Path(out_path),output);info.update(rejected=rejected,source="train and hard2 only; exact spans required")
    Path(out_path).with_suffix(".manifest.json").write_text(json.dumps(info,indent=2),encoding="utf-8");return info

def build_extension(out_path):
    rng=random.Random(1618);rows=[]
    for i in range(2000):
        cat=rng.choice(["read","write","append","search_text","overwrite"])
        text=rng.choice(HARD[cat])+"，编码使用GB18030"
        e=example(text,ROLES[cat],cat,rng,family="hard2_encoding")
        for s in e["target"]["steps"]:s["args"]["encoding"]="gb18030"
        e["tokens"]=serialize(e["target"]);rows.append(e)
    for i in range(1000):
        e=example("回收文件%A%，再根据删除记录恢复它",ROLES["trash"],"trash",rng,family="hard2_trash_restore")
        e["target"]=plan([step("fs.trash",path=slot(1)),step("fs.restore",record_id=result(1,"record_id"))]);e["tokens"]=serialize(e["target"]);rows.append(e)
    for i in range(1000):
        e=example("新建文件%A%写%B%，再新建文件%C%写%D%，然后新建文件%E%写%F%，最后新建文件%G%写%H%",{c:"NAME" if j%2==0 else "TEXT" for j,c in enumerate("ABCDEFGH")},"write",rng,family="hard2_eight_slots")
        e["target"]=plan([step("fs.write_text",path=slot(j),text=slot(j+1)) for j in (1,3,5,7)]);e["tokens"]=serialize(e["target"]);rows.append(e)
    for e in rows:masks(e["request"],e["tokens"])
    return save_rows(Path(out_path),rows)

def build_round3(out_path):
    rng=random.Random(57721);rows=[]
    phrases={
      "rename":["从今天起，文件%A%叫作%B%", "让文件%A%今后使用名字%B%", "往后文件%A%的名字就是%B%", "文件%A%将来称作%B%", "以后文件%A%用名称%B%来叫它", "请使文件%A%改用名字%B%"],
      "copy_into":["我打算在目录%B%存文件%A%的一份副本", "目的文件夹%B%应该有文件%A%的一份拷贝", "目录%B%里放副本，来源文件%A%", "让目录%B%保存文件%A%的一份复制品"],
      "move_into":["我打算在目录%B%存文件%A%，原处移走", "目的文件夹%B%接收文件%A%，源文件搬走", "目录%B%里放入文件%A%，原位置去掉它", "把文件%A%收纳进目录%B%，原处移走"],
      "refusal":["目前不要处理了，保持文件原样", "什么也不用执行了，原样保持", "保持现状，所以不要进行操作", "先别弄，原状不用动", "不需要操作了，保持不变", "不用处理这些文件，取消这次指令"]}
    reject={
      "MISSING_PARAMETER":["我想搜索", "需要你来读取文件", "请帮忙做一次删除", "我打算新建文件", "我需要追加文字", "准备复制文件，但参数稍后提供"],
      "UNCLEAR_REFERENCE":["我需要移动上次谈的文件", "读取前一轮所说的文件", "把昨天提到的文件改名字", "更名先前那个", "上次的那个文件给我读读", "复制之前所说的那一个"],
      "AMBIGUOUS_TARGET":["整理文件%A%，但怎么做还待决定", "调整文件%A%，没有说具体操作", "文件%A%随便处理，操作方式不明确", "搬文件%A%到%B%，目标是文件还是目录暂不清楚"],
      "UNSUPPORTED_TASK":["识别图片里的文字%A%", "对PDF进行文字识别%A%", "替我写代码%A%", "帮忙完成程序%A%"],
      "TOO_COMPLEX":["我有六个顺序步骤都要执行", "总共有九个步骤，连续完成", "任务需要五步，不要拆开", "失败就自动无限重试"]}
    for i in range(15000):
        if i%3==0:
            reason=rng.choice(list(reject));e=rejection(rng.choice(PREFIXES)+rng.choice(reject[reason])+rng.choice(SUFFIXES),reason,rng,"hard3_refusal")
        else:
            cat=rng.choice(list(phrases));text=rng.choice(phrases[cat]);e=example(rng.choice(PREFIXES)+text+rng.choice(SUFFIXES),ROLES[cat],cat,rng,family="hard3_roles")
            if cat!="refusal" and i%2==0:
                other=rng.choice(list(HARD)[:-1]);cats=list(HARD)
                if (cats.index(cat)+cats.index(other))%4!=0:
                    e=compose((text,cat),(rng.choice(HARD[other]),other),rng,"hard3_composition")
        rows.append(add_variations(e,rng))
    return save_rows(Path(out_path),rows)

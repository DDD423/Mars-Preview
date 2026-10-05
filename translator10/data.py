"""Symbolic supervised data. Split by expression family, not random rows.

Raw parameter values exist only for filter evaluation/augmentation; never in the
translator input. Independent suite is frozen before any training invocation.
"""
import hashlib
import json
import random
import re
from pathlib import Path
from .language import FILE_TOOLS, canonical, fingerprint, nonexecution, serialize, masks

# Eight independently phrased families per intent. Last three belong exclusively
# to validation, calibration and test, respectively.
FAMILIES = {
 "list": ["列出目录%A%的文件", "看看目录%A%有哪些东西", "显示路径%A%下的项目", "把目录%A%的清单给我", "查看目录%A%的直接子项", "我想知道目录%A%里面都有啥", "请展示目录%A%的条目列表", "麻烦枚举一下目录%A%的内容"],
 "read": ["读取叫做%A%的文件", "打开名为%A%的文本文件看看", "把文件%A%的正文读出来", "查看路径%A%的文本内容", "显示文件%A%里的文字", "文件%A%写了啥，给我看看", "请取出文件%A%中的文本", "我想瞧瞧文件%A%的全文"],
 "search_name": ["搜索名称包含%A%的文件", "找名字里带%A%的文件", "按文件名搜索%A%", "帮我查一下名中有%A%的文件", "检索文件名称关键词%A%", "哪些文件的名字含有%A%", "请按名称关键字%A%查找", "我要找文件名出现%A%的那些文件"],
 "search_text": ["搜索正文包含%A%的文件", "找内容里带%A%的文本", "按文件内容搜索%A%", "帮我查文字里有%A%的文件", "检索正文关键词%A%", "哪些文件的内容含有%A%", "请按正文关键字%A%查找", "我要找正文出现%A%的那些文件"],
 "mkdir": ["创建叫做%A%的目录", "新建名为%A%的文件夹", "建立目录%A%", "给我建个目录%A%", "生成路径为%A%的文件夹", "我需要一个目录，名字是%A%", "请建立名称为%A%的目录", "麻烦开辟一个文件夹叫%A%"],
 "empty": ["创建叫做%A%的空文件", "新建名为%A%的文本文件，内容留空", "建立空白文件%A%", "给我建个文件%A%，里面不要写东西", "生成路径为%A%的空文本文件", "我需要一个空文件，名字是%A%", "请建立名称为%A%的空白文件", "麻烦弄一个叫%A%的文件，先别填内容"],
 "write": ["把文字%B%写入新文件%A%", "创建名为%A%的文件，内容为%B%", "在路径%A%新建文件并写入%B%", "给我建个文件%A%，里面写%B%", "将%B%保存为新文件%A%", "我要把%B%存到新文件%A%里面", "请创建文件%A%，用%B%作为内容", "麻烦把内容%B%放进新建的%A%文件"],
 "overwrite": ["把文件%A%的全文替换为%B%", "覆盖文件%A%，内容改成%B%", "将%B%写入文件%A%，允许覆盖", "把文件%A%原来的文字全部换成%B%", "使用%B%重写文件%A%", "文件%A%的旧内容不要了，改为%B%", "请以%B%覆盖文件%A%全文", "麻烦让文件%A%只保留内容%B%"],
 "append": ["向文件%A%追加文字%B%", "在名为%A%的文件末尾加上%B%", "把%B%接在文件%A%后面", "给文件%A%再添一段%B%", "追加内容%B%到路径%A%的文件", "文件%A%别动原文，末尾添上%B%", "请把%B%附加到已有文件%A%末尾", "麻烦在文件%A%最后补上%B%，保留前文"],
 "copy": ["复制文件%A%到完整路径%B%", "把文件%A%拷贝为%B%", "从路径%A%复制一份到路径%B%", "给文件%A%做个副本，副本完整路径%B%", "将%B%作为目标完整路径，复制来源%A%", "我要文件%A%的副本，保存在完整路径%B%", "请从%A%拷贝到完整目标路径%B%", "麻烦照着文件%A%另存一份为%B%"],
 "copy_into": ["复制文件%A%到目录%B%里面", "把名为%A%的文件拷贝进文件夹%B%", "在目录%B%里放一份文件%A%的副本", "给文件%A%做个副本放入目录%B%", "将目录%B%作为目的文件夹，复制文件%A%", "我要目录%B%里面有文件%A%的一份拷贝", "请把来源%A%复制进目录%B%，保留文件名", "麻烦拷一份文件%A%放在文件夹%B%内"],
 "move": ["移动文件%A%到完整路径%B%", "把文件%A%迁移为%B%", "从路径%A%移到路径%B%", "给文件%A%换个位置，新完整路径%B%", "将%B%作为目标完整路径，移动来源%A%", "我要文件%A%迁到完整路径%B%", "请从%A%移动到完整目标路径%B%", "麻烦把文件%A%搬过去，新地址是完整路径%B%"],
 "move_into": ["移动文件%A%到目录%B%里面", "把名为%A%的文件搬进文件夹%B%", "把目录%B%当作文件%A%的新位置", "给文件%A%换个位置，放入目录%B%", "将目录%B%作为目的文件夹，移动文件%A%", "我要目录%B%里面收纳文件%A%，原处不留", "请把来源%A%移进目录%B%，保留文件名", "麻烦搬走文件%A%放在文件夹%B%内"],
 "rename": ["把叫做%A%的文件改名为%B%", "将文件%A%重命名成%B%", "文件%A%换个名字叫%B%", "帮我把文件%A%的名称改成%B%", "使用新名字%B%命名原文件%A%", "我要原文件%A%以后叫%B%", "请让文件%A%改用名称%B%", "麻烦替文件%A%更名为%B%"],
 "trash": ["删除叫做%A%的文件", "把文件%A%放进回收站", "移除名为%A%的文件，可以恢复", "回收文件%A%", "将路径%A%的文件移入回收目录", "文件%A%不要了，先收进回收站", "请把文件%A%丢到可恢复的回收目录", "麻烦让文件%A%离开工作区，留个恢复记录"],
 "restore": ["恢复删除记录%A%", "还原回收记录%A%", "把记录%A%对应的文件恢复回来", "从回收站恢复记录%A%", "撤销记录%A%的回收操作", "记录%A%删掉的东西我要找回来", "请恢复记录编号为%A%的文件", "麻烦取回删除记录%A%所指的文件"],
 "read_write": ["读取文件%A%，将读到的内容写入新文件%B%", "把文件%A%的全文复制到新建文本文件%B%中", "先打开文件%A%，再把原文保存成新文件%B%", "取出文件%A%的正文，然后写到新文件%B%", "新文件%B%的内容使用从文件%A%读出的文字", "我要读文件%A%，接着用它的原文建文件%B%", "请读取来源%A%后把结果写入新文件%B%", "麻烦把文件%A%读出来，读到啥就存入新文件%B%"],
 "copy_rename": ["复制文件%A%到完整路径%B%，再把副本改名为%C%", "把文件%A%拷贝为%B%，然后把它改名叫%C%", "从%A%复制一份到%B%，接着给副本命名%C%", "先复制文件%A%为%B%，再把刚复制的文件更名为%C%", "给%A%做个路径为%B%的副本，副本新名字%C%", "我要先拷贝文件%A%到%B%，之后它改叫%C%", "请把%A%复制到%B%以后，将复制结果重命名为%C%", "麻烦弄一份%A%的副本存成%B%，随后把副本名称换成%C%"],
 "conditional": ["如果路径%A%存在，就读取文件%A%", "路径%A%存在时才打开它", "只有文件%A%存在才读它的内容", "检查路径%A%，有这个文件才读取", "读取文件%A%，前提是它存在", "先看看路径%A%有没有，有的话再读", "请在文件%A%存在的条件下读取它", "麻烦读文件%A%，找不到就跳过"],
 "refusal": ["不要删除任何文件，取消请求", "刚才说的都不做了", "别执行文件操作了", "取消所有操作", "停止请求，什么也别改", "先不弄了，保持现在这样", "不要动文件，我撤回请求", "算了，这次不用执行任何操作"],
}

PREFIXES=["", "请", "麻烦你", "帮我", "那个，", "嗯，", "我这会儿想", "能不能帮我", "嘿，"]
SUFFIXES=["", "。", "，谢谢", "，就这样", "，可以吗", "，拜托了", "吧", "一下"]
ROLES={"list":{"A":"PATH"}, "search_name":{"A":"TEXT"},"search_text":{"A":"TEXT"},"mkdir":{"A":"NAME"},"empty":{"A":"NAME"},"write":{"A":"NAME","B":"TEXT"},"overwrite":{"A":"PATH","B":"TEXT"},"append":{"A":"NAME","B":"TEXT"},"copy":{"A":"NAME","B":"PATH"},"copy_into":{"A":"NAME","B":"NAME"},"move":{"A":"NAME","B":"PATH"},"move_into":{"A":"NAME","B":"NAME"},"rename":{"A":"NAME","B":"NAME"},"trash":{"A":"NAME"},"restore":{"A":"VALUE"},"read":{"A":"NAME"},"read_write":{"A":"NAME","B":"PATH"},"copy_rename":{"A":"NAME","B":"PATH","C":"NAME"},"conditional":{"A":"PATH"},"refusal":{}}

def slot(i): return {"slot":i}
def result(i,field="path"): return {"result":{"step":"step_"+str(i),"field":field}}
def step(tool,**args): return {"tool":tool,"args":args}
def plan(steps):
    return {"protocol":"davework/1","kind":"plan","context_id":"ctx_eval","steps":[dict(s,id="step_"+str(i+1)) for i,s in enumerate(steps)]}

def actions(category,s):
    a=s.get("A"); b=s.get("B"); c=s.get("C")
    table={
      "list":[step("fs.list",path=a)],"read":[step("fs.read_text",path=a)],
      "search_name":[step("fs.search",query=a,mode="name")],"search_text":[step("fs.search",query=a,mode="text")],
      "mkdir":[step("fs.mkdir",path=a)],"empty":[step("fs.write_text",path=a,text="")],
      "write":[step("fs.write_text",path=a,text=b)],"overwrite":[step("fs.write_text",path=a,text=b,overwrite=True)],
      "append":[step("fs.append_text",path=a,text=b)],
      "copy":[step("fs.copy",source=a,destination=b)],"move":[step("fs.move",source=a,destination=b)],
      "copy_into":[step("fs.copy",source=a,destination={"join":{"directory":b,"basename_of":a}})],
      "move_into":[step("fs.move",source=a,destination={"join":{"directory":b,"basename_of":a}})],
      "rename":[step("fs.rename",source=a,new_name=b)],"trash":[step("fs.trash",path=a)],
      "restore":[step("fs.restore",record_id=a)],
      "read_write":[step("fs.read_text",path=a),step("fs.write_text",path=b,text=result(1,"text"))],
      "copy_rename":[step("fs.copy",source=a,destination=b),step("fs.rename",source=result(1),new_name=c)],
      "conditional":[dict(step("fs.read_text",path=a),when={"exists":a,"negate":False})],
    }
    return nonexecution("noop","CANCELLED_REQUEST") if category=="refusal" else plan(table[category])

def raw_value(kind,rng):
    atoms=["删除","打开","最终版","草稿","随机🦄","a {1} b","Folder Space","Ω测试","先复制再读取","X.Y.txt"]
    v=rng.choice(atoms)+str(rng.randrange(1000000))
    if kind=="PATH": return "./资料/"+v+".txt"
    if kind=="TEXT": return rng.choice(["这是一段文字\n第二行","abc{1}🔍",v,"不要删除我的内容", " A B "])
    if kind=="VALUE": return "%032x"%rng.getrandbits(128)
    return v

def example(template,roles,category,rng,enabled=None,family="independent"):
    segments=[];bindings={};spans=[];slots=[];mapping={}; raw=""; normalized=""
    for part in re.split(r"(%[A-H]%)",template):
        if re.fullmatch(r"%[A-H]%",part or ""):
            letter=part[1]; typ=roles[letter]; i=len(slots)+1
            # Every appearance is a span, even when the user repeats the same name.
            value=bindings[mapping[letter]["binding"]]["value"] if letter in mapping else raw_value(typ,rng)
            start=len(raw);raw+=value;id="{"+str(i)+"}";normalized+=id
            segments.append({"kind":"slot","id":id}); slots.append({"slot":i,"type":typ})
            bindings[id]={"type":typ,"value":value,"start":start,"end":len(raw)}
            spans.append({"type":typ,"text":value,"start":start,"end":len(raw),"placeholder":id})
            mapping.setdefault(letter,{"slot":i,"binding":id})
        else:
            raw+=part;normalized+=part
            if part:segments.append({"kind":"literal","text":part})
    request={"segments":segments,"slots":slots,"enabled_tools":list(enabled or FILE_TOOLS),"context_id":"ctx_eval"}
    target=actions(category,{k:slot(v["slot"]) for k,v in mapping.items()})
    if target["kind"]=="plan" and any(s["tool"] not in request["enabled_tools"] for s in target["steps"]): target=nonexecution("clarification","UNSUPPORTED_TASK")
    tokens=serialize(target);masks(request,tokens)
    return {"family":family,"category":category,"raw":raw,"normalized":normalized,"bindings":bindings,"spans":spans,"request":request,"target":target,"tokens":tokens}

CLARIFICATIONS={
 "MISSING_PARAMETER":["删除一个文件", "把文件改名", "创建文件但我还没想好名称", "帮我读取", "追加一些内容", "搜索一下", "建个目录", "覆盖文件，内容待定"],
 "UNCLEAR_REFERENCE":["把它删除", "读取刚才那个", "你知道我说的那个文件吧，改名", "把之前的内容追加进去", "还是昨天那个文件", "把上次的文件移动一下", "恢复刚才那条记录", "将它拷贝到那里"],
 "AMBIGUOUS_TARGET":["把文件%A%复制到%B%，不知道这是文件还是目录", "移动文件%A%到%B%，目标类型没确定", "对文件%A%处理一下", "文件%A%有问题，帮我弄好", "随便改一下文件%A%", "整理文件%A%但我还没决定怎么整理", "对文件%A%做点操作", "给文件%A%换一下，具体没想好"],
 "UNSUPPORTED_TASK":["运行终端命令%A%", "写一个程序%A%", "打开浏览器搜索%A%", "识别图片%A%中的文字", "提取PDF文件%A%里的文字", "发送邮件%A%", "在网上下载%A%", "训练模型%A%"],
 "TOO_COMPLEX":["读取、复制、移动、重命名、删除、恢复六个步骤都做一遍", "需要执行十个连续文件操作", "处理九个不同参数，按顺序做五步", "连续做八个步骤再检查结果", "请规划超过四步的操作", "把全部任务自动循环做到成功", "不断自动重试文件操作直到完成", "这次文件任务需要五个以上步骤"],
}

def rejection(template,reason,rng,family):
    roles={k:"VALUE" for k in re.findall(r"%([A-H])%",template)}
    e=example(template,roles,"refusal",rng,family=family)
    e["category"]="clarification";e["target"]=nonexecution("clarification",reason);e["tokens"]=serialize(e["target"]);return e

def add_variations(e,rng):
    # Types are an input feature, not dictionaries of recognizable names.
    for s in e["request"]["slots"]:
        if s["type"] in ("NAME","PATH") and rng.random()<.25: s["type"]="VALUE"
    if rng.random()<.08:
        disabled=rng.choice(FILE_TOOLS);e["request"]["enabled_tools"].remove(disabled)
        if e["target"]["kind"]=="plan" and any(s["tool"]==disabled for s in e["target"]["steps"]):
            e["target"]=nonexecution("clarification","UNSUPPORTED_TASK");e["tokens"]=serialize(e["target"])
    masks(e["request"],e["tokens"]);return e

def compose(left,right,rng,family):
    a=example(left[0],ROLES[left[1]],left[1],rng,family=family)
    b=example(right[0],ROLES[right[1]],right[1],rng,family=family)
    n=len(a["request"]["slots"])
    if n+len(b["request"]["slots"])>8: raise ValueError("Too many slots")
    def remap(v):
        if isinstance(v,dict):
            if "slot" in v:return slot(v["slot"]+n)
            if "result" in v:return result(int(v["result"]["step"].split("_")[-1])+len(a["target"]["steps"]),v["result"]["field"])
            return {k:remap(x) for k,x in v.items()}
        if isinstance(v,list):return [remap(x) for x in v]
        return v
    seg=a["request"]["segments"]+[{"kind":"literal","text":"，然后"}]
    for s in b["request"]["segments"]:
        seg.append(dict(s,id="{"+str(int(s["id"][1:-1])+n)+"}") if s["kind"]=="slot" else s)
    steps=a["target"]["steps"]+[remap(s) for s in b["target"]["steps"]]
    e=dict(a,category="composition",raw=a["raw"]+"，然后"+b["raw"],request={"segments":seg,"slots":a["request"]["slots"]+[dict(s,slot=s["slot"]+n) for s in b["request"]["slots"]],"enabled_tools":list(FILE_TOOLS),"context_id":"ctx_eval"},target=plan(steps))
    # Raw span records are only needed for ordinary raw-filter examples.
    e["tokens"]=serialize(e["target"]);masks(e["request"],e["tokens"]);return e

def save_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    return {"rows":len(rows),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def load_rows(path):
    with Path(path).open(encoding="utf-8") as f:return [json.loads(line) for line in f]

def generate(directory,seed=1729):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists(): raise RuntimeError("Frozen dataset exists; use a new directory instead of overwriting")
    rng=random.Random(seed);manifest={"seed":seed,"schema_fingerprint":fingerprint(),"families":{},"files":{},"independent_authoring":"independent authored expressions; fixed parameter/noise expansion, synthetic, not real user data"}
    categories=list(FAMILIES)
    for split,variants,count in [("train",range(5),40000),("validation",[5],4000),("calibration",[6],4000),("test",[7],6000)]:
        rows=[];families=[(cat,v) for cat in categories for v in variants]
        for i in range(count):
            cat,v=families[i%len(families)];family=cat+"_"+str(v)
            if i%13==0:
                reason=list(CLARIFICATIONS)[(i//13)%5];template=CLARIFICATIONS[reason][v]
                e=rejection(PREFIXES[rng.randrange(len(PREFIXES))]+template+SUFFIXES[rng.randrange(len(SUFFIXES))],reason,rng,family)
            elif i%7==0 and cat!="refusal":
                # Held-out composition pairs never occur in training.
                candidates=[x for x in categories[:-1] if (categories.index(x)+categories.index(cat))%4 != (0 if split=="train" else 1)]
                other=rng.choice(candidates)
                e=compose((FAMILIES[cat][v],cat),(FAMILIES[other][v],other),rng,family)
            else:
                template=rng.choice(PREFIXES)+FAMILIES[cat][v]+rng.choice(SUFFIXES)
                e=example(template,ROLES[cat],cat,rng,family=family)
                # Scoped negation: cancel the forbidden tool, retain the requested read.
                if cat=="read" and rng.random()<.25:
                    e=example("不要删除文件%A%，只"+FAMILIES[cat][v],ROLES[cat],cat,rng,family=family)
                if cat=="refusal" and rng.random()<.2:
                    e=example(["你好", "谢谢你", "今天心情不错", "我只是来聊一聊", "先打个招呼", "早上好呀", "晚上好", "嗨，祝你愉快"][v],{},cat,rng,family=family)
                    e["target"]=nonexecution("noop","NO_ACTION");e["tokens"]=serialize(e["target"])
            rows.append(add_variations(e,rng))
        manifest["families"][split]=[c+"_"+str(v) for c,v in families]
        manifest["files"][split]=save_rows(root/(split+".jsonl"),rows)
    # Frozen suite is constructed before training, entirely separate expressions.
    from .independent import build_independent
    independent=build_independent(rng)
    manifest["files"]["independent"]=save_rows(root/"independent.jsonl",independent)
    combos=[]
    for i in range(400):
        left=categories[i%19];right=categories[(i*7+3)%19]
        if (categories.index(left)+categories.index(right))%4!=0: right=categories[(-categories.index(left))%16]
        combos.append(compose((FAMILIES[left][7],left),(FAMILIES[right][7],right),rng,"heldout_combo"))
    manifest["files"]["combinations"]=save_rows(root/"combinations.jsonl",combos)
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest

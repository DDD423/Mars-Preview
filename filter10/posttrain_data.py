"""Post-training data and a newly frozen, separately authored test set.

Old audit cases are development/regression cases now. New tests are never
used in training, hard-negative mining or confidence calibration.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import SLOT_RE, QUOTES, read_jsonl, render, random_value, write_jsonl

NEW_FRAMES = [
    "创建一个文件，名字叫[[NAME:a]]", "新建一个目录，名称为[[NAME:a]]",
    "名为[[NAME:a]]的文档，帮我读取一下", "叫做[[NAME:a]]的文件，请打开",
    "名称为[[NAME:a]]的文件，先不要删除", "文件名是[[NAME:a]]，帮我创建它",
    "把文件[[NAME:a]]的名字改成[[NAME:b]]", "将文档[[NAME:a]]的名称改为[[NAME:b]]",
    "向名为[[NAME:a]]的文件追加文字[[TEXT:b]]", "把原文[[TEXT:a]]写入名称为[[NAME:b]]的文件",
    "从路径[[PATH:a]]的文件里提取文字", "打开[[VALUE:a]]",
    "读取[[VALUE:a]]", "删除[[VALUE:a]]", "复制[[VALUE:a]]到[[VALUE:b]]",
    "移动[[VALUE:a]]到[[VALUE:b]]", "搜索正文包含[[TEXT:a]]的文档",
    "在这些文件中查找文字[[TEXT:a]]", "找出包含原文[[TEXT:a]]的文件",
    "如果已经备份，就删除名为[[NAME:a]]的文件",
    "读取叫做[[NAME:a]]的文档，如果不存在就跳过",
    "新建名为[[NAME:a]]的目录，暂时不要写入内容",
    "请把叫做[[NAME:a]]的文档重命名为[[NAME:b]]，保留提示{1}",
    "将名称为[[NAME:a]]的文件移动到路径[[PATH:b]]",
    "复制名为[[NAME:a]]的文件到名为[[NAME:b]]的文件夹",
    "打开路径为[[PATH:a]]的文档", "不要删除路径为[[PATH:a]]的文件",
    "读取路径为[[PATH:a]]的文件内容", "创建文件，名字叫[[NAME:a]]",
    "文件名字为[[NAME:a]]，先读取它的内容", "从叫做[[NAME:a]]的文档中提取文本",
    "查找内容中有[[TEXT:a]]的文件", "找出正文里有[[TEXT:a]]的文档",
    "搜索包含文字[[TEXT:a]]的文档", "查找关键词[[TEXT:a]]",
    "给名为[[NAME:a]]的文件添加内容[[TEXT:b]]", "给叫做[[NAME:a]]的文档追加原文[[TEXT:b]]",
    "向路径为[[PATH:a]]的文件追加内容[[TEXT:b]]",
    "把文字[[TEXT:a]]写到名为[[NAME:b]]的文件里",
    "将原文[[TEXT:a]]追加到路径[[PATH:b]]的文件中",
    "把文件[[NAME:a]]复制到当前目录", "将文档[[NAME:a]]移动到选中的文件夹",
    "打开文件[[NAME:a]]，只读取不要修改", "读取文档[[NAME:a]]，然后停止",
    "创建名为[[NAME:a]]的文件，内容为[[TEXT:b]]",
    "新建名称是[[NAME:a]]的文档，写入[[TEXT:b]]",
    "将路径[[PATH:a]]的文件复制至路径[[PATH:b]]",
    "把路径[[PATH:a]]的文件移到名称是[[NAME:b]]的目录",
    "把名叫[[NAME:a]]的文件改名叫[[NAME:b]]",
    "文件名为[[NAME:a]]，新名称为[[NAME:b]]",
    "将文件[[NAME:a]]重命名成[[NAME:b]]", "把[[VALUE:a]]的名字改为[[NAME:b]]",
    "提取文件[[NAME:a]]中的文本", "从路径为[[PATH:a]]的文档中读取文字",
    "只打开叫做[[NAME:a]]的文件，其他文件不要动",
    "把名为[[NAME:a]]的文件备份到路径[[PATH:b]]",
    "在路径为[[PATH:a]]的目录中搜索包含[[TEXT:b]]的文件",
    "在名称是[[NAME:a]]的文件中查找原文[[TEXT:b]]",
    "请删除文件[[NAME:a]]，前提是已经备份", "打开[[VALUE:a]]中的文字",
    # Validation families 60..69: held out from post-training.
    "名称是[[NAME:a]]的文档，请帮我打开它",
    "帮我建立一个文件，文件名为[[NAME:a]]",
    "从名叫[[NAME:a]]的文件里读取正文",
    "搜索内容包含原文[[TEXT:a]]的文件",
    "向名称为[[NAME:a]]的文档添加文字[[TEXT:b]]",
    "将文件[[NAME:a]]复制至目录[[NAME:b]]",
    "把名称为[[NAME:a]]的文件改名成[[NAME:b]]",
    "读取路径[[PATH:a]]的文档内容",
    "把原文[[TEXT:a]]写到路径为[[PATH:b]]的文件里",
    "如果存在，打开叫做[[NAME:a]]的文档",
    # Calibration families 70..79: separate from validation.
    "帮我打开名称是[[NAME:a]]的那份文件",
    "创建一个空文件，名称是[[NAME:a]]",
    "名叫[[NAME:a]]的文件，先提取文字",
    "搜索文字内容含有[[TEXT:a]]的文档",
    "向文件[[NAME:a]]追加原文[[TEXT:b]]",
    "将路径为[[PATH:a]]的文件移动到路径[[PATH:b]]",
    "把名为[[NAME:a]]的文档改名成[[NAME:b]]",
    "在路径[[PATH:a]]的文件中查找文字[[TEXT:b]]",
    "把内容[[TEXT:a]]追加到叫做[[NAME:b]]的文件中",
    "文件名称为[[NAME:a]]，请读取",
]

NEGATIVES = ["请不要删除当前选中的文件", "显示昨天创建的所有文档，只列出文件名",
             "先暂停，等我确认后再继续", "打开当前选中的文件", "只处理已经备份的文件",
             "不要删除任何目录", "显示工作区所有文件", "复制全部文件到当前目录",
             "暂停", "取消操作", "不要写入内容", "等待我选择文件"]

# Separate acceptance scenarios: no frame below is added to NEW_FRAMES.
FRESH_TEST = [
    "请打开那个名为[[NAME:a]]的文件", "删除名称是[[NAME:a]]的这份文档",
    "创建新的文档，名字为[[NAME:a]]", "新建一个空目录，名字叫[[NAME:a]]",
    "名叫[[NAME:a]]的文档，麻烦提取正文", "名称为[[NAME:a]]的文件，读一下内容",
    "文件名称是[[NAME:a]]，请打开它", "文档名为[[NAME:a]]，不要删除",
    "把叫做[[NAME:a]]的文件移动至路径[[PATH:b]]",
    "将名字是[[NAME:a]]的文档复制到名叫[[NAME:b]]的目录",
    "把文档[[NAME:a]]的名字改为[[NAME:b]]", "把名为[[NAME:a]]的文件重命名成[[NAME:b]]",
    "打开路径为[[PATH:a]]的那份文件", "从路径[[PATH:a]]的文档中提取正文",
    "读取路径为[[PATH:a]]的文本", "删除路径[[PATH:a]]的文件，前提是已备份",
    "查找正文包含文字[[TEXT:a]]的文件", "搜索内容里有[[TEXT:a]]的文档",
    "在文档里查找关键词[[TEXT:a]]", "找出包含[[TEXT:a]]的文本",
    "给名字为[[NAME:a]]的文件写入原文[[TEXT:b]]",
    "向路径[[PATH:a]]的文档追加文字[[TEXT:b]]",
    "将内容[[TEXT:a]]写到名称为[[NAME:b]]的文件中",
    "把文字[[TEXT:a]]追加到名为[[NAME:b]]的文档里",
    "打开[[VALUE:a]]，只查看", "读取[[VALUE:a]]，不要编辑",
    "复制[[VALUE:a]]到[[VALUE:b]]，如果不存在就跳过",
    "把文件[[NAME:a]]移动至当前目录", "给文件[[NAME:a]]改名为[[NAME:b]]",
    "文件名叫[[NAME:a]]，新名字叫[[NAME:b]]",
    "如果已经保存，读取名为[[NAME:a]]的文档",
    "只删除叫做[[NAME:a]]的文件，其他文档保留",
    "创建一个文件，名字是[[NAME:a]]，初始内容为[[TEXT:b]]",
    "从名称为[[NAME:a]]的文件中提取所有文字",
    "在路径[[PATH:a]]的目录下查找包含原文[[TEXT:b]]的文档",
    "查看叫做[[NAME:a]]的文件的正文",
    "先不要移动名为[[NAME:a]]的文件到路径[[PATH:b]]",
    "请把路径[[PATH:a]]的文件备份到路径[[PATH:b]]",
    "帮我建立名为[[NAME:a]]的文件夹",
    "在叫做[[NAME:a]]的文件中搜索文字[[TEXT:b]]",
]

TEST_VALUES = ["删除", "叫做", "写入", "文件", "包含", "重命名为", "晚报🚀.md", "Unknown ζ.txt", "名字是", "的文档"]
TEST_PATHS = ["C:/评测空间/叫做.txt", "\\\\archive\\shared\\删除.md", "./另一个目录/名字是.csv",
              "/srv/ζ space/新内容.txt", "D:\\从未见过\\包含.txt", "C:/重命名为/写入.log",
              "./星球🚀/的文档.md", "/home/file/Unknown ζ.txt", "D:/文件/{1}.txt", "C:/次级/叫做.txt"]
TEST_TEXTS = ["不要删除", "包含", "今天叫做明天", "line A\nline B", "🔍未知关键词", "重命名为",
              "hello ζ world", "文件名字是甲", "{1}", "这是一段原文"]


def rows_for(frames, count, split, rng, offset=0):
    rows = []
    while len(rows) < count:
        index = len(rows)
        template = frames[index % len(frames)]
        if rng.random() < 0.1:
            text, spans = rng.choice(NEGATIVES), []
        else:
            values = {m.group(2): random_value(rng, m.group(1), "train" if split == "train" else split)
                      for m in SLOT_RE.finditer(template)}
            quotes = {key: rng.choice(QUOTES) if rng.random() < 0.5 else ("", "") for key in values}
            text, spans = render(template, values, quotes)
            prefix = rng.choice(["", "请", "麻烦", "帮我", "现在", "先", "请先不要"])
            text = prefix + text
            spans = [dict(s, start=s["start"] + len(prefix), end=s["end"] + len(prefix)) for s in spans]
        if len(text) > 256:
            continue
        rows.append({"id": f"post_{split}_{index:05}", "family": f"post_{offset+index%len(frames):03}",
                     "text": text, "spans": spans, "source": "posttrain_structural_augmentation"})
    return rows


def generate_posttrain(data_dir="data/posttrain", seed=117):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fresh = directory / "fresh.jsonl"
    if not fresh.exists():
        rows = []
        for family, template in enumerate(FRESH_TEST):
            for index in range(10):
                values, quotes = {}, {}
                for m in SLOT_RE.finditer(template):
                    kind, key = m.groups()
                    pool = TEST_PATHS if kind == "PATH" else TEST_TEXTS if kind == "TEXT" else TEST_VALUES
                    values[key] = pool[(index + (3 if key == "b" else 0)) % len(pool)]
                    if kind == "VALUE" and index % 2:
                        values[key] = TEST_PATHS[index]
                    quotes[key] = QUOTES[1 + index % 4] if index % 3 == 0 else ("", "")
                text, spans = render(template, values, quotes)
                rows.append({"id": f"fresh_{family:02}_{index:02}", "family": f"fresh_{family:02}",
                             "text": text, "spans": spans, "source": "assistant_authored_fresh"})
        for index, text in enumerate(["请列出全部文档", "查看当前选中的文件", "先不要写入任何内容",
             "如果没有备份就停止", "删除所有空文件夹", "只处理昨天的文件", "显示目录结构", "打开当前文件",
             "暂时不需要复制", "停止操作", "只读取不要修改", "统计文件数量", "请等待下一条指令",
             "把当前文件复制到选中的目录", "保持文件名称不变", "显示文件列表", "不用操作", "暂停所有任务",
             "检查当前目录", "关闭选中的文档"]):
            rows.append({"id": f"fresh_none_{index:02}", "family": "fresh_none", "text": text, "spans": [],
                         "source": "assistant_authored_fresh"})
        write_jsonl(fresh, rows)
        (directory / "fresh_manifest.json").write_text(json.dumps({"rows": len(rows),
            "sha256": hashlib.sha256(fresh.read_bytes()).hexdigest(), "frozen_before_posttraining": True,
            "source": "assistant-authored, not collected real-user traffic"}, indent=2), encoding="utf-8")
    reserved = {r["text"] for r in read_jsonl(fresh)}
    old = read_jsonl("data/train.jsonl") + read_jsonl("data/train_extra.jsonl")
    old = [r for r in old if r["text"] not in reserved]
    rng.shuffle(old)
    new = rows_for(NEW_FRAMES[:60], 6000, "train", rng)
    train = old[:14000] + [r for r in new if r["text"] not in reserved]
    rng.shuffle(train)
    splits = {"train": train, "validation": rows_for(NEW_FRAMES[60:70], 1000, "validation", rng, 60),
              "calibration": rows_for(NEW_FRAMES[70:80], 1000, "calibration", rng, 70)}
    manifest = {"seed": seed, "base_checkpoint": "artifacts/baseline/filter1.0.pt", "splits": {}}
    for name, rows in splits.items():
        rows = [r for r in rows if r["text"] not in reserved]
        path = directory / f"{name}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Old audit is now explicitly regression data, not a fresh generalization claim.
    write_jsonl(directory / "regression.jsonl", read_jsonl("data/audit.jsonl"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

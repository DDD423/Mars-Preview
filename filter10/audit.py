"""Independently written acceptance sentences, frozen BEFORE training.

These are assistant-authored test cases, not collected human/user utterances.
Thirty separately authored scenarios x ten concrete instances = 300 rows.
None of the acceptance rows are used for fitting or calibration.
"""
import hashlib
import json
from pathlib import Path
from .data import render, write_jsonl

SCENARIOS = [
    ("collision", "删除叫做[[NAME:a]]的文件"),
    ("name", "请把名称为[[NAME:a]]的文件打开"),
    ("name", "名为[[NAME:a]]的文档，帮我读取一下"),
    ("name", "创建一个文件，名字叫[[NAME:a]]"),
    ("name", "新建名为[[NAME:a]]的目录，暂时不要写入内容"),
    ("multi", "将名称为[[NAME:a]]的文件移动到路径[[PATH:b]]"),
    ("multi", "复制名为[[NAME:a]]的文件到名为[[NAME:b]]的文件夹"),
    ("multi", "把文件[[NAME:a]]的名字改成[[NAME:b]]"),
    ("path", "打开路径为[[PATH:a]]的文档"),
    ("path", "不要删除路径为[[PATH:a]]的文件"),
    ("path", "从路径[[PATH:a]]的文件里提取文字"),
    ("path", "读取路径为[[PATH:a]]的文件内容"),
    ("text", "搜索正文包含[[TEXT:a]]的文档"),
    ("text", "在这些文件中查找文字[[TEXT:a]]"),
    ("text", "找出包含原文[[TEXT:a]]的文件"),
    ("multi", "向名为[[NAME:a]]的文件追加文字[[TEXT:b]]"),
    ("multi", "把原文[[TEXT:a]]写入名称为[[NAME:b]]的文件"),
    ("negation", "不要打开叫做[[NAME:a]]的文件"),
    ("negation", "只复制名称为[[NAME:a]]的文件，别删除它"),
    ("condition", "如果已经备份，就删除名为[[NAME:a]]的文件"),
    ("condition", "读取叫做[[NAME:a]]的文档，如果不存在就跳过"),
    ("value", "打开[[VALUE:a]]"),
    ("value", "复制[[VALUE:a]]到[[VALUE:b]]"),
    ("value", "读取[[VALUE:a]]，但不要修改它"),
    ("quotes", "删除名为“[[NAME:a]]”的文件"),
    ("quotes", "搜索包含「[[TEXT:a]]」的文档"),
    ("placeholder", "请把叫做[[NAME:a]]的文档重命名为[[NAME:b]]，保留提示{1}"),
    ("no_value", "请不要删除当前选中的文件"),
    ("no_value", "显示昨天创建的所有文档，只列出文件名"),
    ("no_value", "先暂停，等我确认后再继续"),
]


def generate_audit(directory="data"):
    directory = Path(directory)
    path = directory / "audit.jsonl"
    if path.exists():
        return {"rows": 300, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "preserved": True}
    names = ["删除", "打开", "不要删除", "Raven Report.txt", "星港🚀.md", "{1}", "a 的文件 b", "叫做", "η-é-𠀀.txt", "名字不是动作"]
    texts = ["删除", "不要打开", "hello world", "第一行\n第二行", "🔍关键字", "{1}", "用户说：删除文件", "a 的文件 b", "é中𠀀", "复制到路径"]
    paths = ["C:/Audit 2026/删除.txt", "D:\\独立评估\\打开.md", "./新目录/不要删除",
             "/tmp/never_seen_903.txt", "C:/火星/星港🚀.md", "./{1}/资料.txt", "D:/a 的文件 b/log.txt",
             "\\\\host\\share\\叫做.txt", "/home/η-é/𠀀.txt", "C:/独立数据/名字不是动作"]
    rows = []
    for scenario_index, (category, template) in enumerate(SCENARIOS):
        for index in range(10):
            values = {}
            from .data import SLOT_RE
            for match in SLOT_RE.finditer(template):
                kind, key = match.groups()
                pool = paths if kind == "PATH" else texts if kind == "TEXT" else names
                values[key] = pool[(index + (3 if key == "b" else 0)) % 10]
                if kind == "VALUE" and index % 2:
                    values[key] = paths[index]
            text, spans = render(template, values)
            if category == "no_value":
                # Variants remain independent no-literal cases.
                text = ["", "麻烦", "现在", "如果方便，", "请先", "这次", "我希望", "暂时", "注意：", "请帮我"][index] + text
            rows.append({"id": f"audit_{scenario_index:02}_{index:02}", "family": f"audit_{scenario_index:02}",
                         "category": category, "text": text, "spans": spans,
                         "source": "assistant_authored_independent"})
    write_jsonl(path, rows)
    info = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": "assistant-authored; not human-collected; frozen before training"}
    (directory / "audit_manifest.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info

"""Annotated expression families, deterministic augmentation and isolated splits.

Templates declare literal spans. No filename dictionary is used at inference.
The independent audit set is created by audit.py, not from these templates.
"""
import hashlib
import json
import random
import re
from pathlib import Path

SLOT_RE = re.compile(r"\[\[(NAME|PATH|TEXT|VALUE):([a-z]+)\]\]")

# Each row is a distinct expression family. Indices 0..7 train, 8 validation,
# 9 calibration, 10..11 test. Wrappers and literal randomization never move a
# family between splits. Ten operation groups x twelve families = 120.
FAMILIES = {
    "open": [
        "打开叫做[[NAME:a]]的文件", "把名为[[NAME:a]]的文档打开",
        "打开文件[[NAME:a]]", "打开路径为[[PATH:a]]的文件",
        "帮我打开[[VALUE:a]]", "将[[NAME:a]]这个文件打开",
        "找到名称是[[NAME:a]]的文件并打开", "打开名叫[[NAME:a]]的文件夹",
        "把叫做[[NAME:a]]的文件打开看看", "打开那个名字是[[NAME:a]]的文档",
        "请打开名称为[[NAME:a]]的文件", "我想打开名为[[NAME:a]]的那份文件",
    ],
    "delete": [
        "删除叫做[[NAME:a]]的文件", "把名为[[NAME:a]]的文件删除",
        "删除文件[[NAME:a]]", "删除路径为[[PATH:a]]的文件",
        "不要删除[[VALUE:a]]", "将[[NAME:a]]这个文件删除",
        "只删除名称是[[NAME:a]]的文件", "删掉名叫[[NAME:a]]的目录",
        "把叫做[[NAME:a]]的文件删掉", "删除那个名字是[[NAME:a]]的文档",
        "请删除名称为[[NAME:a]]的文件", "我想删掉名为[[NAME:a]]的那份文件",
    ],
    "create": [
        "创建叫做[[NAME:a]]的文件", "新建一个名为[[NAME:a]]的文件夹",
        "创建文件[[NAME:a]]", "在路径[[PATH:a]]下新建文件[[NAME:b]]",
        "新建[[VALUE:a]]", "建立[[NAME:a]]这个目录",
        "创建名称是[[NAME:a]]的空文件", "新建名叫[[NAME:a]]的文档",
        "建一个叫做[[NAME:a]]的文件", "创建那个名字是[[NAME:a]]的目录",
        "请创建名称为[[NAME:a]]的文件", "我想新建名为[[NAME:a]]的那份文档",
    ],
    "move": [
        "把叫做[[NAME:a]]的文件移动到路径[[PATH:b]]",
        "将名为[[NAME:a]]的文件移到名为[[NAME:b]]的目录",
        "移动文件[[NAME:a]]到文件夹[[NAME:b]]",
        "把路径[[PATH:a]]的文件移动到路径[[PATH:b]]",
        "移动[[VALUE:a]]到[[VALUE:b]]",
        "将[[NAME:a]]这个文件移动到[[NAME:b]]这个目录",
        "把名称是[[NAME:a]]的文档移入名称是[[NAME:b]]的文件夹",
        "将名叫[[NAME:a]]的文件挪到路径[[PATH:b]]",
        "把叫做[[NAME:a]]的文件移进叫做[[NAME:b]]的目录",
        "移动那个名字是[[NAME:a]]的文档到路径[[PATH:b]]",
        "请把名称为[[NAME:a]]的文件移动至路径[[PATH:b]]",
        "我想将名为[[NAME:a]]的文件移到名称为[[NAME:b]]的文件夹",
    ],
    "copy": [
        "把叫做[[NAME:a]]的文件复制到路径[[PATH:b]]",
        "将名为[[NAME:a]]的文件复制到名为[[NAME:b]]的目录",
        "复制文件[[NAME:a]]到文件夹[[NAME:b]]",
        "把路径[[PATH:a]]的文件复制到路径[[PATH:b]]",
        "复制[[VALUE:a]]到[[VALUE:b]]",
        "将[[NAME:a]]这个文件复制到[[NAME:b]]这个目录",
        "把名称是[[NAME:a]]的文档复制进名称是[[NAME:b]]的文件夹",
        "将名叫[[NAME:a]]的文件备份到路径[[PATH:b]]",
        "把叫做[[NAME:a]]的文件复制进叫做[[NAME:b]]的目录",
        "复制那个名字是[[NAME:a]]的文档到路径[[PATH:b]]",
        "请把名称为[[NAME:a]]的文件复制至路径[[PATH:b]]",
        "我想将名为[[NAME:a]]的文件复制到名称为[[NAME:b]]的文件夹",
    ],
    "rename": [
        "把叫做[[NAME:a]]的文件重命名为[[NAME:b]]",
        "将名为[[NAME:a]]的文件改名为[[NAME:b]]",
        "将文件[[NAME:a]]重命名为[[NAME:b]]",
        "把路径[[PATH:a]]的文件改名为[[NAME:b]]",
        "把[[VALUE:a]]改名为[[NAME:b]]",
        "将[[NAME:a]]这个文件的名字改为[[NAME:b]]",
        "把名称是[[NAME:a]]的文档名称改成[[NAME:b]]",
        "给名叫[[NAME:a]]的文件改名为[[NAME:b]]",
        "把叫做[[NAME:a]]的文件改名叫[[NAME:b]]",
        "将那个名字是[[NAME:a]]的文档改名为[[NAME:b]]",
        "请把名称为[[NAME:a]]的文件重命名成[[NAME:b]]",
        "我想将名为[[NAME:a]]的文件名字改成[[NAME:b]]",
    ],
    "search": [
        "搜索包含[[TEXT:a]]的文件", "在文件中查找文字[[TEXT:a]]",
        "搜索内容[[TEXT:a]]", "在路径[[PATH:a]]下搜索包含[[TEXT:b]]的文档",
        "搜索[[TEXT:a]]", "找出正文包含[[TEXT:a]]的文件",
        "检索出现文字[[TEXT:a]]的文档", "查找包含原文[[TEXT:a]]的文件",
        "寻找内容含有[[TEXT:a]]的文档", "搜索那个包含文字[[TEXT:a]]的文件",
        "请找出内容中包含[[TEXT:a]]的文件", "我想查找正文里有[[TEXT:a]]的文档",
    ],
    "read": [
        "读取叫做[[NAME:a]]的文件", "读取名为[[NAME:a]]的文档内容",
        "读取文件[[NAME:a]]", "读取路径为[[PATH:a]]的文件",
        "读取[[VALUE:a]]", "读取[[NAME:a]]这个文件的内容",
        "查看名称是[[NAME:a]]的文档正文", "读取名叫[[NAME:a]]的文件内容",
        "把叫做[[NAME:a]]的文件内容读出来", "读取那个名字是[[NAME:a]]的文档",
        "请读取名称为[[NAME:a]]的文件", "我想读取名为[[NAME:a]]的那份文件",
    ],
    "write": [
        "向叫做[[NAME:a]]的文件写入[[TEXT:b]]",
        "在名为[[NAME:a]]的文档里追加文字[[TEXT:b]]",
        "向文件[[NAME:a]]写入内容[[TEXT:b]]",
        "向路径[[PATH:a]]的文件写入[[TEXT:b]]",
        "往[[VALUE:a]]写入[[TEXT:b]]",
        "把文字[[TEXT:a]]写入[[NAME:b]]这个文件",
        "在名称是[[NAME:a]]的文档中添加内容[[TEXT:b]]",
        "给名叫[[NAME:a]]的文件追加原文[[TEXT:b]]",
        "把内容[[TEXT:a]]写到叫做[[NAME:b]]的文件里",
        "向那个名字是[[NAME:a]]的文档追加[[TEXT:b]]",
        "请向名称为[[NAME:a]]的文件写入文字[[TEXT:b]]",
        "我想在名为[[NAME:a]]的文件里添加原文[[TEXT:b]]",
    ],
    "extract": [
        "提取叫做[[NAME:a]]的文件中的文字", "从名为[[NAME:a]]的文档提取文本",
        "提取文件[[NAME:a]]中的文字", "提取路径为[[PATH:a]]的文件中的文字",
        "提取[[VALUE:a]]中的文字", "提取[[NAME:a]]这个文档的正文",
        "从名称是[[NAME:a]]的文件中提取文字", "提取名叫[[NAME:a]]的文件的文本",
        "把叫做[[NAME:a]]的文件中的文字提取出来", "提取那个名字是[[NAME:a]]的文档的文字",
        "请提取名称为[[NAME:a]]的文件中的文字", "我想提取名为[[NAME:a]]的那份文档的文本",
    ],
}

PREFIXES = ["", "请", "帮我", "麻烦", "先", "不要", "只需要", "如果文件存在就", "我希望你", "请先不要"]
SUFFIXES = ["", "。", "，谢谢", "，但不要处理其他文件", "，仅限当前工作区", "，如果不存在就跳过"]
NO_VALUE = ["列出当前工作区的文件", "不要删除任何文件", "先不要执行任何操作",
            "只处理昨天创建的文件", "显示当前目录", "如果没有备份就先别删除",
            "暂停操作", "打开当前选中的文件", "复制所有文件", "删除空文件夹"]
NO_VALUE_HELDOUT = {
    "validation": ["列出所有文档", "保持当前状态", "不要修改文件内容", "先查看文件列表",
                   "如果文件不存在就跳过", "备份全部文件", "显示文件大小", "只处理今天修改的文件",
                   "关闭当前文档", "取消上一步操作"],
    "calibration": ["显示工作区目录结构", "停止删除操作", "不要覆盖已有文档", "只打开选中的文件",
                    "如果没有权限就停止", "先检查是否存在备份", "统计所有文件", "展示文件类型",
                    "保留所有子目录", "现在不需要执行"],
    "test": ["列出最近访问的文档", "禁止删除任何目录", "保留当前选中的文件", "仅处理已经备份的文件",
             "暂时别继续操作", "检查空目录", "显示今天的文件列表", "不要移动这些文档",
             "只查看文件属性", "等待下一条指令"],
}
QUOTES = [("", ""), ("\"", "\""), ("“", "”"), ("「", "」"), ("'", "'")]


def random_value(rng, kind, split):
    if rng.random() < 0.20:
        # Deliberate verb/name collisions shared across splits; the report also
        # isolates test rows with genuinely unseen values.
        value = rng.choice(["删除", "打开", "复制到", "不要删除", "名字", "路径", "叫做", "写入内容", "{1}", "文件"])
    else:
        length = rng.randint(1, 18)
        if rng.random() < 0.02:
            length = rng.randint(40, 90)
        styles = ["ascii", "chinese", "mixed", "phrase"]
        style = rng.choice(styles)
        if style == "ascii":
            value = "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") for _ in range(length))
        elif style == "chinese":
            base = {"train": 0x4e00, "validation": 0x6200, "calibration": 0x7200, "test": 0x8200}[split]
            value = "".join(chr(base + rng.randint(0, 1000)) for _ in range(length))
        elif style == "mixed":
            value = "".join(rng.choice("报告计划资料ABC012_ 😀🚀é中") for _ in range(length)).strip() or "😀"
        else:
            value = rng.choice(["年度报告", "项目资料", "删除的文件", "今天不要删除", "a 的文件 b", "名称是打开", "hello world", "x{1}y"])
            value += "_" + str(rng.randint(0, 999999))
        if kind in ("NAME", "VALUE") and rng.random() < 0.35:
            value += rng.choice([".txt", ".md", ".pdf", ".docx", ".csv"])
    if kind == "PATH" or (kind == "VALUE" and rng.random() < 0.35):
        value = rng.choice(["C:/工作区/", "D:\\资料\\", "./workspace/", "/home/user/", "\\\\server\\共享\\"]) + value
    return value


def render(template, values, quotes=None):
    chunks, spans, offset, last = [], [], 0, 0
    for match in SLOT_RE.finditer(template):
        plain = template[last:match.start()]
        chunks.append(plain)
        offset += len(plain)
        kind, key = match.groups()
        value = values[key]
        before, after = (quotes or {}).get(key, ("", ""))
        chunks.append(before)
        offset += len(before)
        spans.append({"start": offset, "end": offset + len(value), "type": kind})
        chunks.extend([value, after])
        offset += len(value) + len(after)
        last = match.end()
    chunks.append(template[last:])
    return "".join(chunks), spans


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate(directory="data", seed=42):
    directory = Path(directory)
    split_indices = {"train": list(range(8)), "validation": [8], "calibration": [9], "test": [10, 11]}
    sizes = {"train": 12000, "validation": 1000, "calibration": 1000, "test": 2000}
    manifest = {"seed": seed, "generator_version": 1, "splits": {}}
    audit_path = directory / "audit.jsonl"
    reserved = {row["text"] for row in read_jsonl(audit_path)} if audit_path.exists() else set()
    for split_num, (split, indices) in enumerate(split_indices.items()):
        rng = random.Random(seed + split_num * 100003)
        families = [(f"{operation}_{index:02}", operation, templates[index])
                    for operation, templates in FAMILIES.items() for index in indices]
        rows = []
        while len(rows) < sizes[split]:
            index = len(rows)
            family_id, operation, template = families[index % len(families)]
            if rng.random() < 0.1:
                text = rng.choice(NO_VALUE if split == "train" else NO_VALUE_HELDOUT[split])
                spans = []
            else:
                values = {m.group(2): random_value(rng, m.group(1), split) for m in SLOT_RE.finditer(template)}
                quotes = {key: rng.choice(QUOTES) for key in values}
                text, spans = render(template, values, quotes)
                prefix, suffix = rng.choice(PREFIXES), rng.choice(SUFFIXES)
                text = prefix + text + suffix
                spans = [dict(span, start=span["start"] + len(prefix), end=span["end"] + len(prefix)) for span in spans]
            if len(text) > 256:
                raise ValueError("Generated input exceeds the model limit")
            if text in reserved:
                continue
            rows.append({"id": f"{split}_{index:05}", "family": family_id, "operation": operation,
                         "text": text, "spans": spans, "source": "synthetic"})
        rng.shuffle(rows)
        path = directory / f"{split}.jsonl"
        write_jsonl(path, rows)
        manifest["splits"][split] = {"rows": len(rows), "families": [f[0] for f in families],
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest

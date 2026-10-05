"""Refinement based on validation failure categories, never test predictions.

Augment only training-family skeletons. Held-out skeletons and whole sentences
are explicitly excluded. Mutations teach control-language synonyms, not names.
"""
import hashlib
import json
import random
from pathlib import Path
from .data import FAMILIES, SLOT_RE, PREFIXES, SUFFIXES, QUOTES, random_value, read_jsonl, render, write_jsonl


def mutate(template, operation, rng):
    for original in ["名称是", "名为", "名叫", "叫做"]:
        if original in template:
            template = template.replace(original, rng.choice(["名称是", "名称为", "名字是", "名为", "名叫", "叫做", "名字叫"] ))
            break
    if operation == "search":
        for original in ["搜索", "查找", "找出", "检索"]:
            if original in template:
                template = template.replace(original, rng.choice(["搜索", "查找", "寻找", "找出", "检索"] ))
                break
        template = template.replace("包含", rng.choice(["包含", "含有", "含", "包含文字", "包含原文"]))
        template = template.replace("正文", rng.choice(["正文", "内容", "文字内容"]))
    if operation in ("move", "copy"):
        if "移动到" in template:
            template = template.replace("移动到", rng.choice(["移动到", "移到", "移入", "移进", "移动至"]))
        if "复制到" in template:
            template = template.replace("复制到", rng.choice(["复制到", "复制至", "复制进", "备份到"]))
    if operation == "write":
        template = template.replace("写入", rng.choice(["写入", "追加", "添加"]))
        template = template.replace("文字", rng.choice(["文字", "内容", "原文"]))
    if rng.random() < 0.4:
        if rng.random() < 0.5:
            template = template.replace("文档", "文件")
        else:
            template = template.replace("文件夹", "__FOLDER__").replace("文件", "文档").replace("__FOLDER__", "文件夹")
    if rng.random() < 0.3:
        template = template.replace("目录", "文件夹") if rng.random() < 0.5 else template.replace("文件夹", "目录")
    return template


def augment(data_dir="data", count=8000, seed=91):
    directory = Path(data_dir)
    reserved = {row["text"] for split in ["validation", "calibration", "test", "audit"]
                for row in read_jsonl(directory / f"{split}.jsonl")}
    held_templates = {t for group in FAMILIES.values() for t in group[8:]}
    rng = random.Random(seed)
    rows = []
    operations = list(FAMILIES)
    weights = [4 if op == "search" else 2 if op == "write" else 1 for op in operations]
    prefixes = PREFIXES + ["请你", "请帮我", "我想", "能否", "如果已经备份就"]
    while len(rows) < count:
        operation = rng.choices(operations, weights=weights, k=1)[0]
        family_index = rng.randrange(8)
        template = mutate(FAMILIES[operation][family_index], operation, rng)
        if template in held_templates:
            continue
        values = {m.group(2): random_value(rng, m.group(1), "train") for m in SLOT_RE.finditer(template)}
        quotes = {key: (rng.choice(QUOTES[1:]) if rng.random() < 0.5 else ("", "")) for key in values}
        text, spans = render(template, values, quotes)
        prefix, suffix = rng.choice(prefixes), rng.choice(SUFFIXES)
        text = prefix + text + suffix
        spans = [dict(span, start=span["start"] + len(prefix), end=span["end"] + len(prefix)) for span in spans]
        if text in reserved or len(text) > 256:
            continue
        rows.append({"id": f"extra_{len(rows):05}", "family": f"{operation}_{family_index:02}",
                     "operation": operation, "text": text, "spans": spans,
                     "source": "training_family_synonym_augmentation", "template": template})
    path = directory / "train_extra.jsonl"
    write_jsonl(path, rows)
    info = {"seed": seed, "rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "reason": "validation errors on unseen search wording and write boundaries",
            "heldout_skeletons_excluded": True, "test_predictions_consulted": False}
    (directory / "augmentation_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

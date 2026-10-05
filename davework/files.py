"""Workspace file tools. Destructive operations are recoverable."""
import codecs
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from .paths import Workspace
from .registry import WorkError

TEXT_LIMIT = 1024 * 1024
BINARY_EXT = {".pdf", ".doc", ".docx", ".xlsx", ".pptx", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".exe", ".dll", ".zip", ".pt", ".pyc"}


def cancelled(cancel):
    if cancel.is_set():
        raise WorkError("CANCELLED", "任务已停止")


def read_text(path, encoding, limit=TEXT_LIMIT):
    if path.suffix.lower() in BINARY_EXT:
        raise WorkError("UNSUPPORTED_FILE", "首版仅支持文本文件，不提取 PDF、DOCX 或图片")
    with path.open("rb") as file:
        data = file.read(limit + 1)
    truncated = len(data) > limit
    data = data[:limit]
    if b"\x00" in data:
        raise WorkError("BINARY_FILE", "文件包含二进制内容")
    try:
        text = codecs.getincrementaldecoder(encoding)("strict").decode(data, final=not truncated)
    except UnicodeDecodeError:
        raise WorkError("ENCODING_REQUIRED", "解码失败，请明确选择 UTF-8 或 GB18030", str(path))
    return {"path": str(path), "text": text, "truncated": truncated}


def record_paths(ws, record_id):
    if not isinstance(record_id, str) or len(record_id) != 32 or any(c not in "0123456789abcdef" for c in record_id):
        raise WorkError("INVALID_RECORD", "记录 ID 必须是 32 位十六进制字符串")
    root = ws.path(".davework-trash", internal=True)
    record = ws.path(str(root / (record_id + ".json")), internal=True)
    payload = ws.path(str(root / record_id), internal=True)
    return root, record, payload


def recovery_record(ws, record_id):
    _, meta, payload = record_paths(ws, record_id)
    if not meta.is_file() or not payload.exists():
        raise WorkError("RECORD_NOT_FOUND", "删除 / 备份记录不存在或已恢复")
    record = json.loads(meta.read_text(encoding="utf-8"))
    target = ws.path(record["original"], allow_root=False)
    return record, target, payload, meta


def stash(ws, source, copy=False):
    rid = uuid.uuid4().hex
    root, meta, payload = record_paths(ws, rid)
    root.mkdir(exist_ok=True)
    ws.tree(source)
    if copy:
        shutil.copy2(source, payload)
    else:
        os.rename(source, payload)
    try:
        meta.write_text(json.dumps({"record_id": rid, "original": str(source), "kind": "backup" if copy else "trash"}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        if not copy:
            os.rename(payload, source)
        else:
            payload.unlink()
        raise
    return rid


def copy_file(source, destination, cancel):
    with source.open("rb") as src, destination.open("xb") as dst:
        while True:
            cancelled(cancel)
            data = src.read(1024 * 1024)
            if not data:
                break
            dst.write(data)
    shutil.copystat(source, destination)


def execute_file(tool, args, context, cancel):
    ws = Workspace(context["workspace"], context["settings"]["excludes"])
    cancelled(cancel)
    if tool == "fs.list":
        p = ws.path(args["path"])
        entries = []
        for child in sorted(p.iterdir(), key=lambda x: x.name.casefold()):
            if child.name in ws.excludes:
                continue
            try:
                safe = ws.path(str(child))
                entries.append({"name": child.name, "path": str(safe), "directory": safe.is_dir()})
            except WorkError:
                continue
            if len(entries) > 1000:
                break
        return {"path": str(p), "entries": entries[:1000], "truncated": len(entries) > 1000}
    if tool == "fs.read_text":
        return read_text(ws.path(args["path"]), args["encoding"])
    if tool == "fs.search":
        matches, truncated = [], False
        for p in ws.walk():
            cancelled(cancel)
            if args["mode"] == "name":
                hit = args["query"].casefold() in p.name.casefold()
                content = None
            elif p.is_file():
                try:
                    content = read_text(p, args["encoding"])
                    truncated |= content["truncated"]
                    hit = args["query"] in content["text"]
                except WorkError:
                    continue
            else:
                continue
            if hit:
                matches.append({"path": str(p), "name": p.name})
            if len(matches) >= 1000:
                truncated = True
                break
        return {"matches": matches, "truncated": truncated}
    if tool == "fs.restore":
        record, target, payload, meta = recovery_record(ws, args["record_id"])
        if target.exists():
            raise WorkError("TARGET_EXISTS", "恢复目标已存在，请先移动该目标")
        ws.path(str(payload), internal=True)
        os.rename(payload, target)
        meta.unlink()
        return {"path": str(target)}
    if tool == "fs.trash":
        p = ws.path(args["path"], allow_root=False)
        rid = stash(ws, p)
        return {"record_id": rid, "path": str(p)}
    if tool in ("fs.mkdir", "fs.write_text", "fs.append_text"):
        target = ws.path(args["path"], allow_root=False)
        if tool == "fs.mkdir":
            target.mkdir()
            return {"path": str(target)}
        backup = ""
        append = tool == "fs.append_text"
        old = b""
        if append:
            if not target.is_file():
                raise WorkError("NOT_FILE", "追加目标必须是已有文本文件")
            # Validate the explicitly selected encoding without re-encoding old bytes.
            old = target.read_bytes()
            if target.suffix.lower() in BINARY_EXT or b"\x00" in old:
                raise WorkError("BINARY_FILE", "追加仅支持文本文件")
            try:
                old.decode(args["encoding"], errors="strict")
            except UnicodeDecodeError:
                raise WorkError("ENCODING_REQUIRED", "原文件编码不匹配，请明确选择编码")
            args = dict(args, overwrite=True)
        if target.exists():
            if not args["overwrite"]:
                raise WorkError("TARGET_EXISTS", "目标已存在，未选择覆盖")
            if not target.is_file():
                raise WorkError("NOT_FILE", "不能覆盖目录")
            backup = stash(ws, target, copy=True)
        fd, tmp = tempfile.mkstemp(prefix=".dave-write-", dir=str(target.parent))
        temp = ws.path(tmp)
        try:
            with os.fdopen(fd, "wb") as file:
                file.write(old)
                file.write(args["text"].encode(args["encoding"]))
                file.flush()
                os.fsync(file.fileno())
            cancelled(cancel)
            if args["overwrite"]:
                os.replace(temp, target)
            else:
                # Atomic no-clobber publication for file targets.
                os.link(temp, target)
                temp.unlink()
        finally:
            if temp.exists():
                temp.unlink()
        return {"path": str(target), "backup_id": backup}
    source = ws.path(args["source"], allow_root=False)
    target = ws.path(str(source.with_name(args["new_name"])) if tool == "fs.rename" else args["destination"], allow_root=False)
    if target.exists():
        raise WorkError("TARGET_EXISTS", "目标冲突，不覆盖")
    if target == source or source in target.parents:
        raise WorkError("INVALID_TARGET", "不能把目录放入自身内部")
    ws.tree(source)
    if tool in ("fs.move", "fs.rename"):
        cancelled(cancel)
        os.rename(source, target)
        return {"path": str(target)}
    staging = ws.path(str(target.parent / (".dave-copy-" + uuid.uuid4().hex)))
    try:
        if source.is_dir():
            staging.mkdir()
            for base, dirs, files in os.walk(source, followlinks=False):
                cancelled(cancel)
                relative = Path(base).relative_to(source)
                for name in dirs:
                    (staging / relative / name).mkdir()
                for name in files:
                    copy_file(ws.path(str(Path(base) / name)), ws.path(str(staging / relative / name)), cancel)
        else:
            copy_file(source, staging, cancel)
        cancelled(cancel)
        if staging.is_dir():
            os.rename(staging, target)
        else:
            os.link(staging, target)
            staging.unlink()
    finally:
        if staging.exists():
            # The staging path is validated within the captured workspace above.
            ws.path(str(staging), allow_root=False)
            if staging.is_dir():
                shutil.rmtree(staging)
            else:
                staging.unlink()
    return {"path": str(target)}

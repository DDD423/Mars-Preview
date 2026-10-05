"""Single source of truth for forms, validation and model tool descriptions."""
from copy import deepcopy


class WorkError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


def param(kind, label, required=True, default=None, choices=None):
    d = {"type": kind, "label": label, "required": required}
    if not required:
        d["default"] = default
    if choices:
        d["choices"] = choices
    return d


TOOLS = {}


def tool(name, label, access, params, returns):
    TOOLS[name] = {"name": name, "label": label, "access": access,
                   "params": params, "returns": returns}


P = param
tool("fs.list", "列目录", "read", {"path": P("path", "目录", False, ".")}, {"path": "string", "entries": "array", "truncated": "boolean"})
tool("fs.read_text", "读取文本", "read", {"path": P("source", "文件"), "encoding": P("string", "编码", False, "utf-8", ["utf-8", "gb18030"])}, {"path": "string", "text": "string", "truncated": "boolean"})
tool("fs.search", "搜索名称 / 内容", "read", {"query": P("string", "搜索词"), "mode": P("string", "搜索方式", False, "name", ["name", "text"]), "encoding": P("string", "编码", False, "utf-8", ["utf-8", "gb18030"])}, {"matches": "array", "truncated": "boolean"})
tool("fs.mkdir", "创建目录", "write", {"path": P("target", "新目录路径")}, {"path": "string"})
tool("fs.write_text", "创建 / 写入文本", "write", {"path": P("target", "文件路径"), "text": P("string", "内容"), "overwrite": P("boolean", "允许覆盖并备份", False, False), "encoding": P("string", "编码", False, "utf-8", ["utf-8", "gb18030"])}, {"path": "string", "backup_id": "string"})
tool("fs.append_text", "追加文本", "write", {"path": P("source", "已有文件"), "text": P("string", "追加内容"), "encoding": P("string", "原文件编码", False, "utf-8", ["utf-8", "gb18030"])}, {"path": "string", "backup_id": "string"})
tool("fs.copy", "复制", "write", {"source": P("source", "来源"), "destination": P("target", "完整目标路径")}, {"path": "string"})
tool("fs.move", "移动", "write", {"source": P("source", "来源"), "destination": P("target", "完整目标路径")}, {"path": "string"})
tool("fs.rename", "重命名", "write", {"source": P("source", "来源"), "new_name": P("name", "新名称（无目录）")}, {"path": "string"})
tool("fs.trash", "移入回收目录", "write", {"path": P("source", "文件 / 目录")}, {"record_id": "string", "path": "string"})
tool("fs.restore", "恢复删除 / 备份", "write", {"record_id": P("string", "删除 / 备份记录 ID")}, {"path": "string"})
tool("terminal.exec", "运行程序", "terminal", {"program": P("string", "程序"), "argv": P("array", "参数数组", False, []), "timeout": P("integer", "超时秒数", False, None)}, {"exit_code": "integer", "stdout": "string", "stderr": "string", "truncated": "boolean"})
tool("terminal.powershell", "PowerShell 脚本", "terminal", {"script": P("string", "脚本（原样运行）"), "timeout": P("integer", "超时秒数", False, None)}, {"exit_code": "integer", "stdout": "string", "stderr": "string", "truncated": "boolean"})


def catalog():
    return deepcopy(list(TOOLS.values()))


def strict(obj, allowed, required=()):
    if not isinstance(obj, dict):
        raise WorkError("INVALID_OBJECT", "必须是 JSON 对象")
    extra, missing = set(obj) - set(allowed), set(required) - set(obj)
    if extra or missing:
        raise WorkError("INVALID_FIELDS", "存在多余或缺失字段", {"extra": sorted(extra), "missing": sorted(missing)})


def check_value(value, spec):
    kind = spec["type"]
    ok = (isinstance(value, str) if kind in ("string", "source", "target", "path", "name") else
          isinstance(value, bool) if kind == "boolean" else
          isinstance(value, int) and not isinstance(value, bool) if kind == "integer" else
          isinstance(value, list) if kind == "array" else False)
    if value is None and spec.get("default", 1) is None:
        return
    if not ok:
        raise WorkError("INVALID_TYPE", "参数类型错误：" + spec["label"])
    if spec.get("choices") and value not in spec["choices"]:
        raise WorkError("INVALID_CHOICE", "不支持的选项：" + spec["label"])
    if kind in ("source", "target", "path", "name") and (not value or "\x00" in value):
        raise WorkError("INVALID_PATH", "路径 / 名称不能为空或包含 NUL")
    if kind == "integer" and value is not None and not 1 <= value <= 3600:
        raise WorkError("INVALID_TIMEOUT", "超时必须在 1–3600 秒之间")

"""Confined file paths, exact name bindings and preview fingerprints."""
import os
from pathlib import Path
from .registry import WorkError

INTERNAL = {".davework", ".davework-trash"}
DEFAULT_EXCLUDES = [".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".cache", ".davework", ".davework-trash"]


def fingerprint(path):
    try:
        s = path.stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_mode]
    except FileNotFoundError:
        return None


def is_link(path):
    # FILE_ATTRIBUTE_REPARSE_POINT includes Windows junctions.
    try:
        return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)
    except FileNotFoundError:
        return False


class Workspace:
    def __init__(self, root, excludes=None):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise WorkError("INVALID_WORKSPACE", "工作区必须是存在的目录")
        self.excludes = set(DEFAULT_EXCLUDES if excludes is None else excludes) | INTERNAL

    def path(self, value, internal=False, allow_root=True):
        p = Path(value)
        if not p.is_absolute():
            p = self.root / p
        if os.name == "nt" and any(":" in part for part in p.parts[1:]):
            raise WorkError("INVALID_PATH", "文件工具不支持 NTFS 备用数据流")
        if os.name == "nt":
            reserved = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {prefix + str(i) for prefix in ("COM", "LPT") for i in range(1, 10)}
            if any(part.rstrip(" .").split(".")[0].upper() in reserved for part in p.parts[1:]):
                raise WorkError("INVALID_PATH", "文件工具不支持 Windows 设备名称")
        p = p.resolve()
        try:
            relative = p.relative_to(self.root)
        except ValueError:
            raise WorkError("OUTSIDE_WORKSPACE", "解析后的真实路径在工作区之外", str(p))
        if not internal and any(part.lower() in INTERNAL for part in relative.parts):
            raise WorkError("RESERVED_PATH", "Dave Work 内部目录不可作为文件工具目标")
        if not allow_root and p == self.root:
            raise WorkError("WORKSPACE_ROOT", "不能修改或删除工作区根目录")
        return p

    def walk(self):
        for base, dirs, files in os.walk(self.root, followlinks=False):
            base = Path(base)
            dirs[:] = sorted(d for d in dirs if d not in self.excludes and not is_link(base / d))
            for name in dirs + sorted(files):
                p = base / name
                if not is_link(p):
                    yield self.path(str(p))

    def name(self, name, selection=None, directory=False):
        if Path(name).name != name or "/" in name or "\\" in name:
            raise WorkError("INVALID_NAME", "NAME 参数必须是完整名称，不是路径")
        matches = [p for p in self.walk() if os.path.normcase(p.name) == os.path.normcase(name) and (not directory or p.is_dir())]
        if selection is not None:
            chosen = self.path(selection)
            if chosen not in matches:
                raise WorkError("INVALID_SELECTION", "选择不属于当前精确名称匹配结果")
            return chosen
        if len(matches) != 1:
            raise WorkError("AMBIGUOUS_NAME" if matches else "NAME_NOT_FOUND",
                            "名称重名，请选择实际路径" if matches else "找不到完整名称", [str(p) for p in matches])
        return matches[0]

    def tree(self, path):
        """Check whole subtree before a recursive mutation; no junction traversal."""
        paths = [path]
        if is_link(path):
            raise WorkError("LINK_NOT_ALLOWED", "递归操作不接受链接或 junction")
        if path.is_dir():
            for base, dirs, files in os.walk(path, followlinks=False):
                for name in dirs + files:
                    p = Path(base) / name
                    if is_link(p):
                        raise WorkError("LINK_NOT_ALLOWED", "目录内含链接或 junction", str(p))
                    paths.append(self.path(str(p)))
        return paths

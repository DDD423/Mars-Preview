"""Versioned plans, one-shot previews, immutable contexts and sequential tasks."""
import copy
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from . import PROTOCOL
from .adapter import FilterAdapter, NoModelPlanner, TranslatorAdapter, planner_input
from .files import execute_file, recovery_record, record_paths
from .paths import DEFAULT_EXCLUDES, Workspace, fingerprint
from .registry import TOOLS, WorkError, check_value, strict
from .terminal import run_terminal

PROJECT = Path(__file__).resolve().parent.parent


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


class Runtime:
    def __init__(self, state_dir=None):
        self.state = Path(state_dir or PROJECT / ".davework")
        self.state.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.config = {"workspace": str(PROJECT), "recent_workspaces": [],
                       "checkpoint": str(PROJECT / "artifacts" / "filter1.0.pt"), "device": "cpu",
                       "translator_checkpoint": str(PROJECT / "artifacts" / "translator1.0.pt"), "translator_device": "cpu",
                       "enabled_tools": list(TOOLS), "terminal_timeout": 120,
                       "auto_scroll": True, "excludes": DEFAULT_EXCLUDES}
        if (self.state / "config.json").exists():
            try:
                saved = json.loads((self.state / "config.json").read_text(encoding="utf-8"))
                self.config.update({k: v for k, v in saved.items() if k in self.config})
            except (OSError, ValueError):
                pass
        self.filter = FilterAdapter()
        self.planner = NoModelPlanner()
        self.translator = TranslatorAdapter()
        self.contexts, self.previews, self.tasks = {}, {}, {}
        self.active = None
        self.closed = False
        self.worker = None
        self.history = []
        if (self.state / "tasks.json").exists():
            try:
                self.history = json.loads((self.state / "tasks.json").read_text(encoding="utf-8"))[-20:]
                self.tasks = {t["id"]: t for t in self.history}
            except (OSError, ValueError):
                pass

    def settings(self):
        with self.lock:
            status = dict(self.filter.status)
            if status["state"] == "ready" and self.filter.key != (self.config["checkpoint"], self.config["device"]):
                status = {"state": "not_loaded", "message": "配置已更改，等待新请求加载", "device": self.config["device"]}
            translator = dict(self.translator.status)
            if translator["state"] == "ready" and self.translator.key != (self.config["translator_checkpoint"], self.config["translator_device"]):
                translator = {"state": "not_loaded", "message": "配置已更改，等待新请求加载"}
            return {"config": copy.deepcopy(self.config), "filter": status, "translator": translator,
                    "main_model": "translator1.0" if translator["state"] == "ready" else "没有模型", "active_task": self.active, "protocol": PROTOCOL}

    def configure(self, changes):
        strict(changes, set(self.config) - {"recent_workspaces"})
        with self.lock:
            candidate = copy.deepcopy(self.config)
            candidate.update(changes)
            ws = Workspace(candidate["workspace"], candidate["excludes"])
            candidate["workspace"] = str(ws.root)
            if any(candidate[key] not in ("cpu", "cuda") for key in ("device", "translator_device")):
                raise WorkError("INVALID_DEVICE", "设备只能是 cpu 或 cuda")
            if "cuda" in (candidate["device"], candidate["translator_device"]):
                import torch
                try:
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA 不可用")
                    torch.zeros(1, device="cuda").add_(1).cpu()
                except Exception as exc:
                    raise WorkError("CUDA_UNAVAILABLE", "CUDA 配置被拒绝，保留此前配置", str(exc))
            if any(not isinstance(candidate[key], str) for key in ("checkpoint", "translator_checkpoint")):
                raise WorkError("INVALID_CHECKPOINT", "权重路径必须是字符串")
            check_value(candidate["terminal_timeout"], {"type": "integer", "label": "超时"})
            if not isinstance(candidate["auto_scroll"], bool):
                raise WorkError("INVALID_CONFIG", "自动滚动必须是布尔值")
            for key in ("enabled_tools", "excludes"):
                if not isinstance(candidate[key], list) or any(not isinstance(v, str) for v in candidate[key]):
                    raise WorkError("INVALID_CONFIG", key + " 必须是字符串数组")
            if set(candidate["enabled_tools"]) - set(TOOLS):
                raise WorkError("UNKNOWN_TOOL", "工具开关包含未知工具")
            candidate["recent_workspaces"] = ([str(ws.root)] + [p for p in candidate["recent_workspaces"] if p != str(ws.root)])[:8]
            atomic_json(self.state / "config.json", candidate)
            self.config = candidate
            return self.settings()

    def context(self, text="", extract=True):
        if not isinstance(text, str) or len(text) > 16384:
            raise WorkError("INVALID_TEXT", "输入必须是文字且不超过 16384 个字符")
        with self.lock:
            settings = copy.deepcopy(self.config)
        ws = Workspace(settings["workspace"], settings["excludes"])
        result = {"original": text, "normalized": text, "status": "not_requested", "spans": [],
                  "bindings": {}, "segments": [{"kind": "literal", "text": text}], "reason": "手动工具上下文"}
        if extract:
            try:
                result = self.filter.extract(text, settings)
            except WorkError as exc:
                result.update(status="error", reason=exc.message + "：" + str(exc.details))
        context = {"id": "ctx_" + uuid.uuid4().hex, "workspace": str(ws.root),
                   "settings": settings, "filter": copy.deepcopy(result), "created_at": time.time()}
        with self.lock:
            self.contexts[context["id"]] = context
            if len(self.contexts) > 100:
                self.contexts.pop(next(iter(self.contexts)))
        request = planner_input(context)
        if extract and result["status"] not in ("error", "unsupported", "unsupported_length", "uncertain", "needs_context"):
            planned = self.translator.plan(request, settings)
        elif extract:
            planned = {"protocol": PROTOCOL, "kind": "clarification", "context_id": context["id"], "message": "filter 未能可靠处理该请求：" + result.get("reason", "")}
        else:
            planned = self.planner.plan(request)
        return {"context_id": context["id"], "workspace": context["workspace"], "filter": result, "planner": planned}

    def get_context(self, context_id):
        if not isinstance(context_id, str):
            raise WorkError("INVALID_CONTEXT", "context_id 必须是字符串")
        with self.lock:
            if context_id not in self.contexts:
                raise WorkError("CONTEXT_NOT_FOUND", "请求上下文不存在或已过期，请重新提取")
            return copy.deepcopy(self.contexts[context_id])

    def _value(self, value, spec, context, prior, selection=None, deferred=False,
               dependencies=None, selections=None, selection_key=None, joins=None):
        if isinstance(value, dict):
            if set(value) == {"join"}:
                if spec["type"] not in ("target", "path"):
                    raise WorkError("JOIN_TYPE", "路径组合只可用于目标路径")
                join = value["join"]
                if not isinstance(join, dict) or set(join) not in ({"directory", "name"}, {"directory", "basename_of"}):
                    raise WorkError("INVALID_JOIN", "join 必须指定 directory 和 name 或 basename_of")
                ws = Workspace(context["workspace"], context["settings"]["excludes"])
                fixed = {}
                for key, item in join.items():
                    nested_key = (selection_key or "") + ".join." + key
                    kind = "name" if key == "name" else "source"
                    try:
                        leaf = self._value(item, {"type": kind, "label": key, "directory": key == "directory"}, context, prior,
                                           (selections or {}).get(nested_key), deferred)
                    except WorkError as exc:
                        if exc.code == "AMBIGUOUS_NAME":
                            exc.details = {"selection_key": nested_key, "candidates": exc.details}
                        raise
                    fixed[key] = copy.deepcopy(item) if isinstance(item, dict) and "result" in item else leaf
                    if isinstance(leaf, dict):
                        continue
                    if key != "name":
                        leaf = str(ws.path(leaf))
                        fixed[key] = fixed[key] if isinstance(fixed[key], dict) else leaf
                        if dependencies is not None: dependencies.append(leaf)
                        if key == "directory" and not Path(leaf).is_dir() and not (deferred and isinstance(item, dict) and "result" in item):
                            raise WorkError("NOT_DIRECTORY", "组合路径的 directory 必须是目录")
                        if key == "basename_of" and not Path(leaf).exists() and not (deferred and isinstance(item, dict) and "result" in item):
                            raise WorkError("SOURCE_NOT_FOUND", "basename_of 的来源不存在或已失效")
                    elif leaf in (".", "..") or any(c in leaf for c in "\\/:\x00") or leaf != leaf.rstrip(" ."):
                        raise WorkError("INVALID_NEW_NAME", "组合路径的 name 必须是单一完整名称")
                if joins is not None: joins[selection_key] = {"join": fixed}
                # Preserve deferred result references without exposing their content to the model.
                resolved = {}
                for key, leaf in fixed.items():
                    resolved[key] = self._value(leaf, {"type": "name" if key == "name" else "source", "label": key}, context, prior, deferred=deferred)
                if any(isinstance(x, dict) for x in resolved.values()):
                    return {"join": fixed}
                name = resolved.get("name", Path(resolved.get("basename_of", "")).name)
                return str(ws.path(str(Path(resolved["directory"]) / name), allow_root=False))
            if set(value) == {"slot"}:
                slot = value["slot"]
                if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1:
                    raise WorkError("INVALID_SLOT", "slot 必须是正整数")
                binding = context["filter"]["bindings"].get("{" + str(slot) + "}")
                if binding is None:
                    raise WorkError("MISSING_SLOT", "当前上下文没有该占位符", slot)
                if spec["type"] not in ("string", "source", "target", "path", "name"):
                    raise WorkError("SLOT_TYPE", "占位符只可用于文字参数")
                if spec["type"] in ("source", "target", "path", "name") and binding["type"] == "TEXT":
                    raise WorkError("SLOT_TYPE", "TEXT 不能用于文件路径参数")
                text = binding["value"]
                check_value(text, spec)
                # Opaque VALUEs may be exact names or explicit paths. This is
                # parameter binding, not a natural-language action parser.
                bare_value = binding["type"] == "VALUE" and text not in (".", "..") and not any(c in text for c in "\\/:")
                if spec["type"] in ("source", "path") and (binding["type"] == "NAME" or bare_value):
                    ws = Workspace(context["workspace"], context["settings"]["excludes"])
                    return str(ws.name(text, selection, directory=spec.get("directory", False) or spec["type"] == "path"))
                return text
            if set(value) == {"result"}:
                ref = value["result"]
                strict(ref, {"step", "field"}, {"step", "field"})
                if not isinstance(ref["step"], str) or not isinstance(ref["field"], str):
                    raise WorkError("INVALID_REFERENCE", "结果引用 step 和 field 必须是字符串")
                if ref["step"] not in prior:
                    raise WorkError("INVALID_REFERENCE", "结果只能引用当前计划中已完成的前序步骤")
                previous = prior[ref["step"]]
                schema = TOOLS[previous["tool"]]["returns"]
                if ref["field"] not in schema:
                    raise WorkError("INVALID_REFERENCE", "结果字段未声明")
                expected = "string" if spec["type"] in ("source", "target", "path", "name") else spec["type"]
                if schema[ref["field"]] != expected:
                    raise WorkError("REFERENCE_TYPE", "结果字段与参数类型不符")
                if "result" not in previous or ref["field"] not in previous["result"]:
                    if deferred:
                        return copy.deepcopy(value)
                    raise WorkError("RESULT_UNAVAILABLE", "该结果在预览时无法确定，或前序步骤已跳过；请分成两个计划", value)
                text = previous["result"][ref["field"]]
                if ref["field"] == "text" and previous["result"].get("truncated"):
                    raise WorkError("TRUNCATED_REFERENCE", "读取内容已截断，不能把不完整正文作为后续写入内容")
                check_value(text, spec)
                return text
            raise WorkError("INVALID_REFERENCE", "仅支持 {slot:整数} 或 {result:{step,field}}；引用不能携带其他上下文字段")
        if spec["type"] == "array":
            if not isinstance(value, list) or len(value) > 256:
                raise WorkError("INVALID_ARRAY", "argv 必须是最多 256 项的文字数组")
            return [self._value(v, {"type": "string", "label": "argv"}, context, prior, deferred=deferred) for v in value]
        check_value(value, spec)
        return copy.deepcopy(value)

    def _bind(self, step, context, prior, selections, deferred=False, dependencies=None, joins=None):
        tool = step["tool"]
        if tool not in TOOLS:
            raise WorkError("UNKNOWN_TOOL", "未知工具", tool)
        if tool not in context["settings"]["enabled_tools"]:
            raise WorkError("TOOL_DISABLED", "工具在该上下文中已关闭", tool)
        params = TOOLS[tool]["params"]
        strict(step["args"], params, [p for p, spec in params.items() if spec["required"]])
        args = {}
        for key, spec in params.items():
            value = step["args"].get(key, copy.deepcopy(spec.get("default")))
            if tool == "terminal.powershell" and key == "script" and not isinstance(value, str):
                raise WorkError("SCRIPT_LITERAL", "PowerShell 脚本必须是原文字符串，参数通过 DAVE_SLOT 环境变量传递")
            try:
                selection_key = step["id"] + "." + key
                value_spec = spec
                if tool == "fs.write_text" and key == "path" and step["args"].get("overwrite") is True and isinstance(value, dict) and set(value) == {"slot"}:
                    value_spec = dict(spec, type="source")
                args[key] = self._value(value, value_spec, context, prior, selections.get(selection_key), deferred,
                                        dependencies, selections, selection_key, joins)
            except WorkError as exc:
                if exc.code == "AMBIGUOUS_NAME" and not isinstance(exc.details, dict):
                    exc.details = {"selection_key": step["id"] + "." + key, "candidates": exc.details}
                raise
        ws = Workspace(context["workspace"], context["settings"]["excludes"])
        for key, spec in params.items():
            if isinstance(args[key], dict):
                continue
            if spec["type"] in ("source", "target", "path"):
                args[key] = str(ws.path(args[key], allow_root=TOOLS[tool]["access"] == "read"))
            if spec["type"] == "name":
                name = args[key]
                if name in (".", "..") or any(c in name for c in "\\/:\x00") or name != name.rstrip(" ."):
                    raise WorkError("INVALID_NEW_NAME", "新名称必须是单一完整名称，不含目录或尾随空格 / 点")
        return args

    def preview(self, plan, selections=None):
        if not isinstance(plan, dict):
            raise WorkError("INVALID_PLAN", "计划必须是对象")
        kind = plan.get("kind")
        if kind in ("noop", "clarification"):
            strict(plan, {"protocol", "kind", "context_id", "message"}, {"protocol", "kind", "context_id", "message"})
            if not isinstance(plan["message"], str):
                raise WorkError("INVALID_MESSAGE", "message 必须是文字")
        elif kind == "plan":
            strict(plan, {"protocol", "kind", "context_id", "steps"}, {"protocol", "kind", "context_id", "steps"})
        else:
            raise WorkError("INVALID_KIND", "kind 必须是 plan、clarification 或 noop")
        if plan["protocol"] != PROTOCOL:
            raise WorkError("PROTOCOL_VERSION", "不支持的协议版本")
        context = self.get_context(plan["context_id"])
        selections = selections or {}
        if not isinstance(selections, dict) or any(not isinstance(v, str) for v in selections.values()):
            raise WorkError("INVALID_SELECTION", "selections 必须是字段到真实路径的对象")
        ws = Workspace(context["workspace"], context["settings"]["excludes"])
        pins, bound, prior, used_selections = {}, [], {}, set()
        if kind == "plan":
            if not isinstance(plan["steps"], list) or not 1 <= len(plan["steps"]) <= 64:
                raise WorkError("INVALID_STEPS", "步骤数量必须为 1–64")
            for step in plan["steps"]:
                strict(step, {"id", "tool", "args", "when"}, {"id", "tool", "args"})
                if not isinstance(step["id"], str) or not step["id"] or step["id"] in prior or not isinstance(step["tool"], str):
                    raise WorkError("INVALID_STEP", "步骤 id 必须唯一且非空，tool 必须是字符串")
                dependencies, joins = [], {}
                args = self._bind(step, context, prior, selections, deferred=True, dependencies=dependencies, joins=joins)
                used_selections.update(step["id"] + "." + k for k in step["args"])
                used_selections.update(k + ".join." + leaf for k, j in joins.items() for leaf in j["join"])
                when = None
                if "when" in step:
                    strict(step["when"], {"exists", "negate"}, {"exists"})
                    negate = step["when"].get("negate", False)
                    if not isinstance(negate, bool):
                        raise WorkError("INVALID_CONDITION", "negate 必须是布尔值")
                    cond = self._value(step["when"]["exists"], {"type": "source", "label": "存在条件"}, context, prior, deferred=True)
                    when = {"exists": cond if isinstance(cond, dict) else str(ws.path(cond)), "negate": negate}
                touched, predicted = [Path(p) for p in dependencies], {}
                if step["tool"].startswith("fs."):
                    touched += [Path(args[k]) for k, spec in TOOLS[step["tool"]]["params"].items() if spec["type"] in ("path", "source", "target") and isinstance(args[k], str)]
                    if step["tool"] == "fs.rename" and all(isinstance(args[k], str) for k in ("source", "new_name")):
                        touched.append(ws.path(str(Path(args["source"]).with_name(args["new_name"])), allow_root=False))
                    if step["tool"] == "fs.restore" and isinstance(args["record_id"], str):
                        _, target, payload, meta = recovery_record(ws, args["record_id"])
                        touched += [target, payload, meta]
                    if step["tool"] in ("fs.copy", "fs.move", "fs.rename", "fs.trash"):
                        raw_source = args.get("source", args.get("path"))
                        source = Path(raw_source) if isinstance(raw_source, str) else None
                        if source is not None and source.exists():
                            touched += ws.tree(source)
                    path = args.get("destination", args.get("path"))
                    if step["tool"] == "fs.rename" and all(isinstance(args[k], str) for k in ("source", "new_name")):
                        path = str(Path(args["source"]).with_name(args["new_name"]))
                    if step["tool"] == "fs.restore" and isinstance(args["record_id"], str):
                        path = str(target)
                    if isinstance(path, str):
                        predicted["path"] = path
                if when and isinstance(when["exists"], str):
                    touched.append(Path(when["exists"]))
                for p in touched:
                    pins[str(p)] = fingerprint(p)
                    if p != ws.root and p.parent.exists():
                        pins[str(p.parent)] = fingerprint(p.parent)
                checks = set(str(p) for p in touched)
                checks.update(str(p.parent) for p in touched if p != ws.root and p.parent.exists())
                item = {"id": step["id"], "tool": step["tool"], "args": args, "when": when, "checks": sorted(checks),
                        "paths": list(dict.fromkeys(str(p) for p in touched)), "joins": joins}
                bound.append(item)
                prior[step["id"]] = {"tool": step["tool"], "result": predicted}
            if set(selections) - used_selections:
                raise WorkError("INVALID_SELECTION", "选择包含无效字段")
        preview = {"id": "pre_" + uuid.uuid4().hex, "plan": copy.deepcopy(plan), "context": context,
                   "steps": bound, "pins": pins, "created_at": time.time(), "consumed": False}
        preview["digest"] = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            self.previews = {k: v for k, v in self.previews.items() if time.time() - v["created_at"] < 600 and not v["consumed"]}
            self.previews[preview["id"]] = preview
        return {"preview_id": preview["id"], "digest": preview["digest"], "workspace": context["workspace"],
                "kind": kind, "steps": [{k: v for k, v in s.items() if k not in ("paths", "checks", "joins")} for s in bound],
                "message": plan.get("message"), "expires_in": 600}

    def execute(self, preview_id, digest):
        with self.lock:
            if self.closed:
                raise WorkError("CLOSED", "服务正在退出")
            if self.active:
                raise WorkError("BUSY", "首版同时只能执行一个任务")
            preview = self.previews.get(preview_id)
            if not preview or preview["consumed"] or time.time() - preview["created_at"] > 600:
                raise WorkError("PREVIEW_EXPIRED", "预览已过期或使用过，请重新预览")
            if digest != preview["digest"]:
                raise WorkError("PREVIEW_CHANGED", "计划与预览不一致")
            for p, expected in preview["pins"].items():
                if fingerprint(Path(p)) != expected:
                    raise WorkError("TARGET_CHANGED", "预览目标发生变化，请重新预览", p)
            preview["consumed"] = True
            task_id = "task_" + uuid.uuid4().hex
            task = {"id": task_id, "status": "running", "workspace": preview["context"]["workspace"],
                    "created_at": time.time(), "events": [], "context_id": preview["context"]["id"],
                    "plan": preview["plan"], "context": preview["context"], "results": {}}
            task["cancel"] = threading.Event()
            self.tasks[task_id] = task
            self.active = task_id
            self.worker = threading.Thread(target=self._run, args=(task, copy.deepcopy(preview)), daemon=True)
            self.worker.start()
            return {"task_id": task_id}

    def emit(self, task, kind, data):
        with self.lock:
            size = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            task.setdefault("event_bytes", 0)
            if task["event_bytes"] + size > 8 * 1024 * 1024 and kind in ("stdout", "stderr", "step_completed"):
                if not task.get("event_limit"):
                    task["event_limit"] = True
                    task["events"].append({"seq": len(task["events"]) + 1, "time": time.time(), "kind": "history_limit", "data": {"message": "任务事件达到 8 MiB，后续内容不保存，步骤状态仍保留"}})
                return
            if task["event_bytes"] + size > 8 * 1024 * 1024:
                def bounded(value):
                    if isinstance(value, str) and len(value) > 1024:
                        return value[:1024] + " [事件字段已截断]"
                    if isinstance(value, dict):
                        return {k: bounded(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [bounded(v) for v in value[:32]]
                    return value
                data = bounded(data)
                size = len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
            task["event_bytes"] += size
            task["events"].append({"seq": len(task["events"]) + 1, "time": time.time(), "kind": kind, "data": data})

    def _run(self, task, preview):
        context, cancel = preview["context"], task["cancel"]
        pins = preview["pins"]
        prior = {}
        self.emit(task, "task_started", {"workspace": context["workspace"]})
        try:
            if preview["plan"]["kind"] != "plan":
                self.emit(task, preview["plan"]["kind"], {"message": preview["plan"]["message"]})
            for step in preview["steps"]:
                if cancel.is_set():
                    raise WorkError("CANCELLED", "任务已停止")
                sid = step["id"]
                # Stored bound arguments are authoritative: never re-run name lookup.
                for name in step["checks"]:
                    if fingerprint(Path(name)) != pins[name]:
                        raise WorkError("TARGET_CHANGED", "步骤目标与预览不一致", name)
                ws = Workspace(context["workspace"], context["settings"]["excludes"])
                for name in step["paths"]:
                    ws.path(name, internal=step["tool"] == "fs.restore")
                when = step["when"]
                if when and isinstance(when["exists"], dict):
                    when = dict(when, exists=str(ws.path(self._value(when["exists"], {"type": "source", "label": "存在条件"}, context, prior))))
                if when and (Path(when["exists"]).exists() == when["negate"]):
                    self.emit(task, "step_skipped", {"step": sid, "reason": "文件存在性条件不满足"})
                    prior[sid] = {"tool": step["tool"]}
                    continue
                # Resolve refs from actual results; verify against preview's resolved args.
                raw = next(s for s in preview["plan"]["steps"] if s["id"] == sid)
                args = self._bind(dict(step, args=step["args"]), context, prior, {})
                # Predicted refs are checked against actual observations, including argv elements.
                def verify(value, resolved, spec):
                    if isinstance(value, dict) and "result" in value:
                        actual = self._value(value, spec, context, prior)
                        if not isinstance(resolved, dict) and actual != resolved:
                            raise WorkError("RESULT_CHANGED", "实际结果与预览不一致", value)
                    elif isinstance(value, list):
                        for a, b in zip(value, resolved):
                            verify(a, b, {"type": "string", "label": "argv"})
                for key, value in raw["args"].items():
                    verify(value, step["args"][key], TOOLS[step["tool"]]["params"][key])
                for key, fixed_join in step.get("joins", {}).items():
                    arg_key = key.split(".", 1)[1]
                    actual = self._value(fixed_join, TOOLS[step["tool"]]["params"][arg_key], context, prior)
                    if isinstance(step["args"][arg_key], str) and actual != step["args"][arg_key]:
                        raise WorkError("RESULT_CHANGED", "组合路径实际结果与预览不一致")
                if step["tool"] == "fs.rename":
                    ws.path(str(Path(args["source"]).with_name(args["new_name"])), allow_root=False)
                self.emit(task, "step_started", {"step": sid, "tool": step["tool"], "args": args})
                if step["tool"].startswith("terminal."):
                    result = run_terminal(args, step["tool"] == "terminal.powershell", context, cancel,
                                          lambda k, v: self.emit(task, k, dict(v, step=sid)))
                else:
                    result = execute_file(step["tool"], args, context, cancel)
                prior[sid] = {"tool": step["tool"], "result": result}
                task["results"][sid] = result
                self.emit(task, "step_completed", {"step": sid, "result": {k: v for k, v in result.items() if not (step["tool"].startswith("terminal.") and k in ("stdout", "stderr"))}})
                # Refresh only paths affected by our own mutation, including ancestor directory metadata.
                if TOOLS[step["tool"]]["access"] == "write":
                    keys = {"fs.copy": ("destination",), "fs.move": ("source", "destination"),
                            "fs.rename": ("source",), "fs.restore": ()}.get(step["tool"], ("path",))
                    affected = [Path(args[k]) for k in keys if k in args]
                    if step["tool"] == "fs.restore":
                        affected += [Path(p) for p in step["paths"]]
                    if "path" in result:
                        affected.append(Path(result["path"]))
                    # Writes/trash can create a recovery record, changing the
                    # shared trash directory's metadata. Refresh its ancestors
                    # while preserving pins on unrelated existing records.
                    for key in ('backup_id','record_id'):
                        if result.get(key):
                            affected.extend(record_paths(ws,result[key])[1:])
                    for name in pins:
                        p = Path(name)
                        if any(p == a or a in p.parents or (p.is_dir() and p in a.parents) for a in affected):
                            pins[name] = fingerprint(p)
                if cancel.is_set():
                    raise WorkError("CANCELLED", "任务已停止")
            task["status"] = "completed"
        except WorkError as exc:
            task["status"] = "cancelled" if exc.code == "CANCELLED" else "failed"
            self.emit(task, "error", exc.to_dict())
        except Exception as exc:
            task["status"] = "failed"
            self.emit(task, "error", {"code": "EXECUTION_ERROR", "message": str(exc)})
        finally:
            self.emit(task, "task_finished", {"status": task["status"]})
            with self.lock:
                self.active = None
                saved = {k: copy.deepcopy(v) for k, v in task.items() if k != "cancel"}
                # Streams live in event records. Bound duplicated result strings in history.
                for result in saved["results"].values():
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 8192:
                            result[key] = value[:8192] + "\n[历史结果字段已截断；完整输出见事件记录]"
                task["results"] = saved["results"]
                self.history = (self.history + [saved])[-20:]
                try:
                    atomic_json(self.state / "tasks.json", self.history)
                except OSError as exc:
                    self.emit(task, "history_error", {"message": str(exc)})
                allowed = {t["id"] for t in self.history}
                self.tasks = {k: v for k, v in self.tasks.items() if k in allowed}

    def cancel(self, task_id):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise WorkError("TASK_NOT_FOUND", "任务不存在")
            if task["status"] == "running":
                task["cancel"].set()
            return {"status": task["status"]}

    def events(self, task_id, after=0):
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                raise WorkError("TASK_NOT_FOUND", "任务不存在")
            return {"task_id": task_id, "status": task["status"], "events": copy.deepcopy(task["events"][after:])}

    def recent(self):
        with self.lock:
            return [{"id": t["id"], "status": t["status"], "workspace": t["workspace"], "created_at": t["created_at"]} for t in reversed(list(self.tasks.values()))]

    def close(self):
        with self.lock:
            self.closed = True
            if self.active:
                self.tasks[self.active]["cancel"].set()
        if self.worker:
            self.worker.join(timeout=10)

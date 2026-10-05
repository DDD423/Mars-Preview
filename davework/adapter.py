"""Filter adapter and raw-value-free future planner input."""
import threading
from copy import deepcopy
from .registry import WorkError, catalog


class FilterAdapter:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.key = None
        self.status = {"state": "not_loaded", "message": "尚未加载"}

    def extract(self, text, settings):
        with self.lock:
            key = (settings["checkpoint"], settings["device"])
            try:
                if self.key != key:
                    self.status = {"state": "loading", "message": "正在加载 filter1.0"}
                    import torch
                    from filter10 import FilterHarness
                    torch.set_num_threads(4)
                    model = FilterHarness(checkpoint=key[0], device=key[1])
                    self.model, self.key = model, key
                result = self.model.extract(text)
                self.status = {"state": "ready", "message": "filter1.0 已加载", "device": key[1]}
                if result.restore() != text:
                    raise WorkError("RESTORE_FAILED", "filter 结构化还原校验失败")
                return result.to_dict()
            except Exception as exc:
                self.status = {"state": "error", "message": str(exc)}
                raise WorkError("FILTER_ERROR", "filter 加载或提取失败", str(exc))


def planner_input(context, observations=None):
    result = context["filter"]
    # Never send original, spans, proposals, references or binding values.
    return {"protocol": "davework/1", "context_id": context["id"],
            "segments": deepcopy(result["segments"]),
            "slots": [{"slot": int(k[1:-1]), "type": v["type"]} for k, v in result["bindings"].items()],
            "tools": catalog(), "enabled_tools": deepcopy(context.get("settings", {}).get("enabled_tools")),
            "filter_status": result.get("status"),
            "filter_sha256": result.get("checkpoint_sha256"),
            "filter_confidence": min((float(s.get('confidence',1.)) for s in result.get('spans',[])),default=1.),
            "filter_boundary_margin": min((float(s.get('boundary_margin',0.)) for s in result.get('spans',[])),default=0.),
            "filter_start_confidence": min((float(s.get('start_confidence',.5)) for s in result.get('spans',[])),default=.5),
            "filter_end_confidence": min((float(s.get('end_confidence',.5)) for s in result.get('spans',[])),default=.5),
            "filter_count_confidence": min((float(s.get('count_confidence',.5)) for s in result.get('spans',[])),default=.5),
            "filter_count_match": all(s.get('count_match',True) for s in result.get('spans',[])),
            "observations": deepcopy(observations or [])}


class TranslatorAdapter:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.key = None
        self.status = {"state": "not_loaded", "message": "尚未加载"}

    def plan(self, request, settings):
        with self.lock:
            key = (settings["translator_checkpoint"], settings["translator_device"])
            try:
                if self.key != key:
                    self.status = {"state": "loading", "message": "正在加载 translator1.0"}
                    from translator10.inference import TranslatorPlanner
                    model = TranslatorPlanner(key[0], key[1])
                    self.model, self.key = model, key
                expected = (self.model.checkpoint.get('calibration') or {}).get('filter_sha256')
                if expected and request.get('filter_sha256') != expected:
                    raise WorkError('MODEL_PAIR_MISMATCH', 'filter 与 translator 的校准权重不匹配，请选择同一检查点的一对权重')
                result = self.model.plan(request)
                self.status = {"state": "ready", "message": "translator1.0 已加载", "device": key[1]}
                return result
            except Exception as exc:
                self.status = {"state": "error", "message": str(exc)}
                return {"protocol": "davework/1", "kind": "clarification", "context_id": request["context_id"],
                        "message": "translator 加载或规划失败：" + str(exc)}


class NoModelPlanner:
    def plan(self, request):
        return {"protocol": "davework/1", "kind": "noop", "context_id": request["context_id"],
                "message": "没有模型；参数已提取。请使用工具面板或 JSON 调试提交明确指令。"}

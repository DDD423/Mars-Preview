"""Neural span selection, conservative acceptance and lossless rendering.

This module never executes instructions or reads/mutates user files.
"""
import bisect
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from .model import ContextSpanModel, LABELS, ModelConfig
from .boundaries import constrain_probabilities

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts" / "filter1.0.pt"
PATH_PREFIX = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\|/|\.{1,2}[\\/]|~[\\/])")


def looks_like_path(value):
    return bool(PATH_PREFIX.match(value))


def select_spans(pairs, probabilities, threshold=0.5):
    """Maximum weight non-overlapping intervals; deterministic tie breaking."""
    if len(pairs) == 0:
        return []
    labels = probabilities[:, 1:].argmax(axis=1) + 1
    scores = probabilities[np.arange(len(pairs)), labels]
    selected_indices = np.flatnonzero((scores >= threshold) & (scores > probabilities[:, 0]))
    candidates = []
    for index in selected_indices:
        start, end = pairs[int(index)]
        probability = float(scores[index])
        weight = math.log(max(probability, 1e-9) / max(float(probabilities[index, 0]), 1e-9))
        candidates.append((end, start, int(labels[index]), probability, weight))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    ends = [item[0] for item in candidates]
    best, previous, take = [0.0], [], []
    for i, (end, start, label, probability, weight) in enumerate(candidates):
        compatible = bisect.bisect_right(ends, start, 0, i)
        include = best[compatible] + weight
        chosen = include > best[-1] + 1e-12
        best.append(include if chosen else best[-1])
        previous.append(compatible)
        take.append(chosen)
    result, index = [], len(candidates)
    while index:
        if take[index - 1]:
            end, start, label, probability, _ = candidates[index - 1]
            result.append({"start": start, "end": end, "type": LABELS[label], "confidence": probability})
            index = previous[index - 1]
        else:
            index -= 1
    return sorted(result, key=lambda span: (span["start"], span["end"]))


def decode_record(pairs, probabilities, threshold, empty_threshold=0.35):
    proposals = select_spans(pairs, probabilities, 0.5)
    if not proposals:
        maximum = float(probabilities[:, 1:].max()) if len(probabilities) else 0.0
        return ("ok" if maximum < empty_threshold else "uncertain"), [], []
    if any(span["confidence"] < threshold for span in proposals):
        return "uncertain", [], proposals
    return "ok", proposals, proposals


@dataclass
class FilterResult:
    original: str
    status: str
    normalized: str
    spans: List[dict] = field(default_factory=list)
    bindings: dict = field(default_factory=dict)
    segments: List[dict] = field(default_factory=list)
    proposals: List[dict] = field(default_factory=list)
    reason: str = ""
    version: str = "filter1.0"

    def to_dict(self):
        return asdict(self)

    def restore(self):
        """Restore using structured segments, never string.replace('{1}', ...)."""
        return "".join(segment["text"] if segment["kind"] == "literal"
                       else self.bindings[segment["id"]]["value"] for segment in self.segments)


def render_result(text, status, selected, proposals=None, reason=""):
    spans, bindings, segments, rendered, cursor = [], {}, [], [], 0
    for index, prediction in enumerate(selected, 1):
        start, end = prediction["start"], prediction["end"]
        if not 0 <= cursor <= start < end <= len(text):
            raise ValueError("Invalid or overlapping span offsets")
        literal = text[cursor:start]
        segments.append({"kind": "literal", "text": literal})
        rendered.append(literal)
        placeholder = "{" + str(index) + "}"
        value = text[start:end]
        model_type = prediction["type"]
        output_type = "PATH" if model_type == "VALUE" and looks_like_path(value) else model_type
        span = dict(prediction, type=output_type, model_type=model_type, value=value,
                    placeholder=placeholder, type_source="path_prefix" if output_type != model_type else "model")
        spans.append(span)
        bindings[placeholder] = {"value": value, "type": output_type, "start": start, "end": end}
        segments.append({"kind": "slot", "id": placeholder})
        rendered.append(placeholder)
        cursor = end
    segments.append({"kind": "literal", "text": text[cursor:]})
    rendered.append(text[cursor:])
    return FilterResult(text, status, "".join(rendered), spans, bindings, segments,
                        proposals or [], reason)


class FilterHarness:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cpu"):
        self.device = torch.device(device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("version") != "filter1.0" or payload.get("labels") != list(LABELS):
            raise ValueError("Unsupported checkpoint format")
        self.model = ContextSpanModel(ModelConfig(**payload["config"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.training_info = payload.get("training", {})
        calibration = payload.get("calibration", {})
        self.temperature = float(calibration.get("temperature", 1.0))
        self.threshold = float(calibration.get("threshold", 0.95))
        self.empty_threshold = float(calibration.get("empty_threshold", 0.35))
        self.calibration = calibration

    def extract(self, text: str) -> FilterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a Unicode string")
        if len(text) > self.model.config.max_chars:
            return render_result(text, "unsupported", [], reason="超过256个Unicode字符，未截断")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return render_result(text, "unsupported", [], reason="输入包含无效的Unicode代理字符")
        if not text:
            return render_result(text, "ok", [])
        pairs, logits = self.model.all_scores([text], self.device)[0]
        probabilities = torch.softmax(logits / self.temperature, dim=-1).numpy()
        probabilities = constrain_probabilities(text, pairs, probabilities)
        status, selected, proposals = decode_record(pairs, probabilities, self.threshold, self.empty_threshold)
        reason = "候选置信度不足，保留原文；请明确名称或使用引号" if status == "uncertain" else ""
        return render_result(text, status, selected, proposals, reason)

    @staticmethod
    def restore(result: FilterResult):
        return result.restore()

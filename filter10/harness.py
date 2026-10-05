"""Neural span selection, conservative acceptance and lossless rendering.

This module never executes instructions or reads/mutates user files.
"""
import bisect
import hashlib
import io
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from .model import ContextSpanModel, LABELS, ModelConfig

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "artifacts" / "filter1.0.pt"
PATH_PREFIX = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\|/|\.{1,2}[\\/]|~[\\/])")


def looks_like_path(value):
    return bool(PATH_PREFIX.match(value))


def select_spans(pairs, probabilities, threshold=0.5, max_count=None,exact_count=False,min_count=None):
    """Maximum weight non-overlapping intervals; deterministic tie breaking."""
    if len(pairs) == 0:
        return []
    labels = probabilities[:, 1:].argmax(axis=1) + 1
    scores = probabilities[np.arange(len(pairs)), labels]
    if (exact_count or min_count is not None) and max_count:
        evidence=np.log(np.maximum(scores.astype(np.float64),1e-300)/np.maximum(probabilities[:,0].astype(np.float64),1e-300))
        selected_indices=np.argsort(evidence,kind='stable')[-2048:]
    else:selected_indices = np.flatnonzero((scores >= threshold) & (scores > probabilities[:, 0]))
    candidates = []
    for index in selected_indices:
        start, end = pairs[int(index)]
        probability = float(scores[index])
        # Preserve learned margins for very confident overlapping candidates.
        # A 1e-9 floor collapsed distinct NONE probabilities into equal weights
        # and could choose a longer span containing surrounding filler.
        weight = math.log(max(probability, 1e-300) / max(float(probabilities[index, 0]), 1e-300))
        candidates.append((end, start, int(labels[index]), probability, weight))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    if max_count is not None:
        if max_count <= 0:
            return []
        # Cardinality-constrained interval scheduling. The cap comes from
        # the model's sentence-count query, never from text syntax rules.
        ends = [item[0] for item in candidates]
        compatible = [bisect.bisect_right(ends, c[1], 0, i) for i, c in enumerate(candidates)]
        table = [[0.0] * (len(candidates) + 1) for _ in range(max_count + 1)]
        if exact_count or min_count is not None:
            for k in range(1,max_count+1):table[k]=[-float('inf')]*(len(candidates)+1)
        take = [[False] * (len(candidates) + 1) for _ in range(max_count + 1)]
        for k in range(1, max_count + 1):
            for i, c in enumerate(candidates, 1):
                include = table[k-1][compatible[i-1]] + c[4]
                chosen = include > table[k][i-1] + 1e-12
                table[k][i] = include if chosen else table[k][i-1]
                take[k][i] = chosen
        selected_count=max_count
        if min_count is not None:
            selected_count=max(range(min_count,max_count+1),key=lambda k:table[k][-1])
        result, k, i = [], selected_count, len(candidates)
        while k and i:
            if take[k][i]:
                end, start, label, probability, _ = candidates[i-1]
                result.append({'start':start,'end':end,'type':LABELS[label],'confidence':probability})
                i = compatible[i-1]; k -= 1
            else:
                i -= 1
        return sorted(result, key=lambda s:(s['start'],s['end']))
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


def select_model_spans(pairs, logits, count_logits=None, count_policy='cap', count_threshold=None):
    tensor = torch.as_tensor(logits)
    if not tensor.is_floating_point():
        tensor = tensor.float()
    probabilities = torch.softmax(tensor, -1).numpy()
    cap = None;minimum=None
    if count_policy not in ('cap','zero_only','off','exact','range'):
        raise ValueError('Unknown learned-count policy')
    if count_threshold is not None and not 0 <= count_threshold <= 1:
        raise ValueError('Count threshold must be in [0,1]')
    if count_logits is not None and count_policy != 'off':
        tensor = torch.as_tensor(count_logits)
        if not tensor.is_floating_point():
            tensor = tensor.float()
        p = torch.softmax(tensor, -1)
        count = int(p.argmax())
        # Legacy checkpoints predict 4-or-more; newer readouts predict 0..8.
        # Low-confidence counts retain the ordinary interval decoder.
        if count_policy == 'zero_only':
            cap = 0 if float(p[0]) >= .5 else None
        elif float(p[count]) >= (count_threshold if count_threshold is not None else
                                .7 if count_policy in ('exact','range') else .5):
            if len(p)==9 or count<4:cap=count
            elif count_policy=='range':cap,minimum=8,4
    return select_spans(pairs, probabilities, .5, max_count=cap,exact_count=count_policy in ('exact','range') and cap is not None,min_count=minimum)


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
    references: List[dict] = field(default_factory=list)
    decision_source: str = ""
    checkpoint_sha256: str = ""

    def to_dict(self):
        return asdict(self)

    def restore(self):
        """Restore using structured segments, never string.replace('{1}', ...)."""
        return "".join(segment["text"] if segment["kind"] == "literal"
                       else self.bindings[segment["id"]]["value"] for segment in self.segments)


def render_result(text, status, selected, proposals=None, reason="", refine_paths=True):
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
        declared_type = prediction["type"]
        model_type = prediction.get("model_type", declared_type)
        output_type = "PATH" if refine_paths and declared_type == "VALUE" and looks_like_path(value) else declared_type
        span = dict(prediction, type=output_type, model_type=model_type, value=value,
                    placeholder=placeholder, type_source="syntax+path_prefix" if prediction.get("verified_by_syntax") and output_type != declared_type else
                    "syntax" if prediction.get("verified_by_syntax") else
                    "path_prefix" if output_type != declared_type else "model")
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
        if hasattr(checkpoint, 'read'):
            checkpoint.seek(0)
            checkpoint_bytes = checkpoint.read()
        else:
            checkpoint_bytes = Path(checkpoint).read_bytes()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        if payload.get("version") != "filter1.0" or payload.get("labels") != list(LABELS):
            raise ValueError("Unsupported checkpoint format")
        self.model = ContextSpanModel(ModelConfig(**payload["config"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()
        self.training_info = payload.get("training", {})
        self.model.boundary_mix=float(self.training_info.get('boundary_mix',0.))
        self.model.local_window=int(self.training_info.get('local_window',0))
        self.model.local_mix=float(self.training_info.get('local_mix',0.))
        self.posttrained = bool(self.training_info.get("posttrained", False))
        self.adaptive = bool(self.training_info.get('adaptive', False))
        self.neural_only = bool(self.training_info.get('neural_only', False))
        self.learned_count = bool(self.training_info.get('learned_count', False))
        self.count_policy = self.training_info.get('count_policy','cap')
        self.count_threshold = self.training_info.get('count_threshold')
        self.sentence_query_task = self.training_info.get('sentence_query_task','parameter_count')
        calibration = payload.get("calibration", {})
        self.temperature = float(calibration.get("temperature", 1.0))
        self.threshold = float(calibration.get("threshold", 0.95))
        self.empty_threshold = float(calibration.get("empty_threshold", 0.35))
        self.calibration = calibration

    def decode_scores(self, text, pairs, logits, threshold=None, temperature=None, return_info=False, count_logits=None, neural_evidence=None):
        threshold = self.threshold if threshold is None else threshold
        temperature = self.temperature if temperature is None else temperature
        probabilities = torch.softmax(torch.as_tensor(logits) / temperature, dim=-1).numpy()
        if self.neural_only:
            if self.learned_count and self.count_policy != 'off' and count_logits is None:
                raise ValueError('This checkpoint requires its neural sentence-count logits; use extract() or score with_counts=True')
            # Thin harness: no syntax, quote mask, path heuristics or reference
            # resolver. Labels/boundaries come directly from the model at .5.
            probabilities = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
            selected = [dict(s, decision_source='model_only') for s in select_model_spans(pairs, logits, count_logits, self.count_policy, getattr(self, 'count_threshold', None))]
            # Numeric alternative-boundary evidence for joint calibration.
            # It changes neither selected spans nor raw-value-free model input.
            evidence=(torch.as_tensor(logits)[:,1:].amax(-1)-torch.as_tensor(logits)[:,0]).numpy()
            pair_indices={pair:i for i,pair in enumerate(pairs)}
            starts=np.asarray([s for s,e in pairs]);ends=np.asarray([e for s,e in pairs])
            count_probability=torch.as_tensor(count_logits).softmax(-1) if count_logits is not None else None
            predicted_count=int(count_probability.argmax()) if count_probability is not None else None
            count_match=predicted_count==len(selected) or (count_probability is not None and len(count_probability)==5 and predicted_count==4 and len(selected)>=4)
            start_probability=torch.as_tensor(neural_evidence['start']).softmax(-1)[:,1] if neural_evidence is not None else None
            end_probability=torch.as_tensor(neural_evidence['end']).softmax(-1)[:,1] if neural_evidence is not None else None
            for span in selected:
                index=pair_indices[(span['start'],span['end'])]
                competitors=((starts==span['start'])|(ends==span['end']))
                competitors[index]=False
                span['boundary_margin']=float(evidence[index]-evidence[competitors].max()) if competitors.any() else 20.
                if count_probability is not None:
                    span['count_confidence']=float(count_probability.max())
                    span['count_match']=bool(count_match)
                if neural_evidence is not None:
                    span['start_confidence']=float(start_probability[span['start']])
                    span['end_confidence']=float(end_probability[span['end']])
            decoded = ('ok' if selected else 'no_literals', selected, selected,
                       {'decision_source': 'model_only', 'references': []})
            return decoded if return_info else decoded[:3]
        # Compatibility for historical checkpoints only. Neural inference
        # neither imports nor invokes a text-boundary or syntax decoder here.
        from .boundaries import constrain_probabilities
        from .syntax import analyze_syntax
        probabilities = constrain_probabilities(text, pairs, probabilities)
        if self.adaptive or self.neural_only:
            from .adaptive import decode
            decoded = decode(text, pairs, probabilities, dict(self.calibration, threshold=threshold))
            return decoded if return_info else decoded[:3]
        if not self.posttrained:
            return decode_record(pairs, probabilities, threshold, self.empty_threshold)
        evidence = analyze_syntax(text)
        if evidence.no_literals:
            return "no_literals", [], []
        indices = {pair: i for i, pair in enumerate(pairs)}
        proposals = []
        for span in evidence.spans:
            probability = probabilities[indices[(span["start"], span["end"])]]
            proposals.append(dict(span, confidence=float(probability[LABELS.index(span["type"])]),
                                  model_type=LABELS[int(probability.argmax())], decision_source="syntax"))
        if evidence.pending or evidence.ambiguous:
            return "uncertain", [], proposals
        if proposals:
            return "ok", proposals, proposals
        status, selected, raw = decode_record(pairs, probabilities, threshold, self.empty_threshold)
        if not selected:
            return ("uncertain" if raw else "no_match"), [], raw
        if any(a["end"] == b["start"] for a, b in zip(selected, selected[1:])):
            return "uncertain", [], raw
        return status, selected, raw

    def _identify_result(self, result):
        result.checkpoint_sha256 = self.checkpoint_sha256
        return result

    def extract(self, text: str) -> FilterResult:
        if not isinstance(text, str):
            raise TypeError("text must be a Unicode string")
        if len(text) > self.model.config.max_chars:
            return self._identify_result(render_result(text, "unsupported", [], reason="超过256个Unicode字符，未截断"))
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:
            return self._identify_result(render_result(text, "unsupported", [], reason="输入包含无效的Unicode代理字符"))
        if not text:
            return self._identify_result(render_result(text, "ok", []))
        scored = self.model.all_scores([text], self.device, with_counts=self.learned_count,with_evidence=self.neural_only)[0]
        pairs, logits = scored[:2]
        count_logits = scored[2] if self.learned_count else None
        neural_evidence=scored[3 if self.learned_count else 2] if self.neural_only else None
        info = {}
        if self.adaptive or self.neural_only:
            status, selected, proposals, info = self.decode_scores(text, pairs, logits, return_info=True, count_logits=count_logits,neural_evidence=neural_evidence)
        else:
            status, selected, proposals = self.decode_scores(text, pairs, logits)
        reason = {"uncertain": "存在歧义或不完整参数，保留原文；请明确名称或使用引号",
                  "no_match": "未识别到可确认的参数，不能视为成功理解整句",
                  "no_literals": "已识别为不含显式名称、路径或原文参数的操作"}.get(status, "")
        if status == 'needs_context':
            reason = '指代对象不明确；请提供文件/目录名称或路径。已知参数可保留，不猜测指代对象。'
        if self.neural_only and status == 'no_literals':
            reason = '模型未提取显式参数；不代表指代已有对象或已理解执行意图。'
        result = render_result(text, status, selected, proposals, reason, refine_paths=not self.neural_only)
        result.references = info.get('references', [])
        result.decision_source = info.get('decision_source', '')
        return self._identify_result(result)

    @staticmethod
    def restore(result: FilterResult):
        return result.restore()

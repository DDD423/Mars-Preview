"""Calibration uses calibration.jsonl ONLY; test and audit are never fitted."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from .data import read_jsonl
from .harness import FilterHarness, decode_record, looks_like_path, render_result, select_spans, select_model_spans
from .model import LABEL_IDS
from .boundaries import candidate_mask, constrain_probabilities
from .training import span_key


def softmax(logits, temperature=1.0):
    x = logits / temperature
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)


def collect_scores(harness, rows, batch_size=32):
    records = []
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        scored = harness.model.all_scores([row["text"] for row in batch], harness.device, with_counts=harness.learned_count)
        for row, item in zip(batch, scored):
            pairs, logits = item[:2]
            if harness.learned_count:
                row = dict(row, _count_logits=item[2].numpy(), _count_policy=harness.count_policy)
            records.append((row, pairs, logits.numpy()))
    return records


def gold_for_harness(row):
    return [dict(s, type="PATH" if s["type"] == "VALUE" and
                 looks_like_path(row["text"][s["start"]:s["end"]]) else s["type"]) for s in row["spans"]]


def summarize(records, threshold=0.95, temperature=1.0, raw=False, collect_errors=False, harness=None, ignore_counts=False):
    # A thin harness has precisely the raw-model policy, including VALUE.
    # Historical evaluation must not silently refine its types into PATH.
    raw = raw or (harness is not None and harness.neural_only)
    n = len(records)
    tp = pred_total = gold_total = exact = positive_exact = positives = covered = 0
    no_value = false_no_value = uncertain = accepted = accepted_exact = roundtrip = 0
    categories = {}
    errors = []
    for row, pairs, logits in records:
        probabilities = torch.softmax(torch.as_tensor(logits), -1).numpy() if raw else softmax(logits, temperature)
        if raw:
            status, selected = "ok", select_model_spans(pairs, logits, None if ignore_counts else row.get('_count_logits'), row.get('_count_policy','cap'))
            proposals, gold_spans = selected, row["spans"]
            pred = span_key(selected)
        else:
            probabilities = constrain_probabilities(row["text"], pairs, probabilities)
            if harness is None:
                status, selected, proposals = decode_record(pairs, probabilities, threshold)
            else:
                status, selected, proposals = harness.decode_scores(row['text'], pairs, logits, threshold, temperature)
            gold_spans = gold_for_harness(row)
            selected = [dict(s, type="PATH" if s["type"] == "VALUE" and
                            looks_like_path(row["text"][s["start"]:s["end"]]) else s["type"]) for s in selected]
            pred = span_key(selected)
        gold = span_key(gold_spans)
        success = status in ("ok", "no_literals")
        is_exact = success and gold == pred
        exact += is_exact
        tp += len(gold & pred)
        pred_total += len(pred)
        gold_total += len(gold)
        if gold:
            positives += 1
            positive_exact += is_exact
            covered += status == "ok" and bool(pred)
        else:
            no_value += 1
            false_no_value += bool(pred)
        uncertain += status == "uncertain"
        accepted += success
        accepted_exact += is_exact
        result = render_result(row["text"], status, selected, proposals, refine_paths=not raw)
        roundtrip += result.restore() == row["text"]
        category = row.get("category", row.get("operation", "unknown"))
        cat = categories.setdefault(category, {"rows": 0, "exact": 0, "uncertain": 0})
        cat["rows"] += 1
        cat["exact"] += is_exact
        cat["uncertain"] += status == "uncertain"
        if collect_errors and not is_exact:
            errors.append({"id": row["id"], "text": row["text"], "status": status,
                           "gold": gold_spans, "predicted": selected, "proposals": proposals})
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    result = {"rows": n, "exact_accuracy": exact / max(n, 1),
              "positive_exact_accuracy": positive_exact / max(positives, 1),
              "span_precision": precision, "span_recall": recall,
              "span_f1": 2 * precision * recall / max(precision + recall, 1e-9),
              "positive_coverage": covered / max(positives, 1),
              "accepted_sentence_accuracy": accepted_exact / max(accepted, 1),
              "uncertain_rate": uncertain / max(n, 1),
              "no_value_false_replacement_rate": false_no_value / max(no_value, 1),
              "roundtrip_accuracy": roundtrip / max(n, 1),
              "counts": {"correct_sentences": exact, "positive_rows": positives, "covered_positive_rows": covered,
                         "true_positive_spans": tp, "predicted_spans": pred_total, "gold_spans": gold_total,
                         "no_value_rows": no_value, "false_no_value_rows": false_no_value, "accepted_rows": accepted},
              "categories": categories}
    result["targets_met"] = {"exact_90": result["positive_exact_accuracy"] >= 0.90,
                             "precision_98": precision >= 0.98,
                             "coverage_85": result["positive_coverage"] >= 0.85,
                             "no_value_false_1": result["no_value_false_replacement_rate"] <= 0.01,
                             "roundtrip_100": result["roundtrip_accuracy"] == 1.0}
    return result, errors


def calibrate(checkpoint="artifacts/filter1.0.pt", data_dir="data", device="cpu"):
    torch.set_num_threads(4)
    harness = FilterHarness(checkpoint, device)
    if harness.neural_only:
        raise ValueError('纯模型权重使用固定0.5阈值；本阶段不进行语法或路径校准，请使用neurevaluate评测')
    rows = read_jsonl(Path(data_dir) / "calibration.jsonl")
    records = collect_scores(harness, rows)
    # Temperature is chosen on the full candidate population, not on the
    # artificially balanced positive/negative training sample.
    temperatures = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0]
    nlls = []
    for temperature in temperatures:
        loss_sum = count = 0
        for row, pairs, logits in records:
            probs = softmax(logits, temperature)
            eligible = candidate_mask(row["text"], pairs)
            targets = np.zeros(len(pairs), dtype=np.int64)
            index = {pair: i for i, pair in enumerate(pairs)}
            for span in row["spans"]:
                targets[index[(span["start"], span["end"])]] = LABEL_IDS[span["type"]]
            loss_sum += float(-np.log(np.maximum(probs[np.arange(len(pairs)), targets][eligible], 1e-12)).sum())
            count += int(eligible.sum())
        nlls.append(loss_sum / count)
    temperature = temperatures[int(np.argmin(nlls))]
    search = []
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    if harness.posttrained:
        thresholds = [t for t in thresholds if t >= 0.97]
    for threshold in thresholds:
        metrics, _ = summarize(records, threshold, temperature, harness=harness)
        search.append({"threshold": threshold, "metrics": metrics})
    feasible = [entry for entry in search if entry["metrics"]["span_precision"] >= 0.98 and
                entry["metrics"]["accepted_sentence_accuracy"] >= 0.98 and
                entry["metrics"]["no_value_false_replacement_rate"] <= 0.01 and
                entry["metrics"]["counts"]["predicted_spans"] > 0]
    if feasible:
        chosen = max(feasible, key=lambda e: (e["metrics"]["positive_coverage"], e["metrics"]["positive_exact_accuracy"], e['threshold']))
    else:
        # Do not disguise complete abstention as perfect precision. If there
        # is no feasible operating point, expose that fact and favor usable
        # F1 while keeping the achieved precision visible in the report.
        chosen = max(search, key=lambda e: (e["metrics"]["span_f1"], e["metrics"]["positive_exact_accuracy"]))
    calibration = {"temperature": temperature, "threshold": chosen["threshold"], "empty_threshold": 0.35,
                   "rows": len(rows), "precision_constraint_feasible": bool(feasible),
                   "temperature_nll": dict(zip(map(str, temperatures), nlls)), "metrics": chosen["metrics"],
                   "source_sha256": hashlib.sha256((Path(data_dir) / "calibration.jsonl").read_bytes()).hexdigest(),
                   "syntax_decisions_are_not_probability_thresholded": harness.posttrained}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["calibration"] = calibration
    torch.save(payload, checkpoint)
    output = Path(checkpoint).parent / "calibration.json"
    output.write_text(json.dumps({**calibration, "threshold_search": search}, ensure_ascii=False, indent=2), encoding="utf-8")
    return calibration


def evaluate(checkpoint="artifacts/filter1.0.pt", data_dir="data", device="cpu", splits=None):
    torch.set_num_threads(4)
    harness = FilterHarness(checkpoint, device)
    data_dir = Path(data_dir)
    splits = splits or ["test", "audit"]
    report = {"version": "filter1.0", "model": harness.model.metadata(), "calibration": harness.calibration,
              "training": harness.training_info,
              "device": str(harness.device), "offset_units": "Unicode code points; start inclusive, end exclusive",
              "limitations": ["Tests are synthetic or assistant-authored, not real user traffic.",
                              "Context-only typing is ambiguous; VALUE is intentional.",
                              "A softmax score is not a guarantee, including after temperature calibration.",
                              "No arbitrary multi-step or cross-sentence reference resolution."], "splits": {}}
    training_rows = read_jsonl(data_dir / "train.jsonl")
    extra_path = data_dir / "train_extra.jsonl"
    if harness.training_info.get("rows", 0) > len(training_rows) and extra_path.exists():
        training_rows += read_jsonl(extra_path)
    train_values = {row["text"][s["start"]:s["end"]] for row in training_rows for s in row["spans"]}
    for split in splits:
        started = time.time()
        path = data_dir / f"{split}.jsonl"
        if split in ("audit", "fresh"):
            pinned = json.loads((data_dir / f"{split}_manifest.json").read_text(encoding="utf-8"))["sha256"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != pinned:
                raise ValueError("The frozen audit set has changed; refusing to evaluate")
        rows = read_jsonl(path)
        records = collect_scores(harness, rows)
        raw, _ = summarize(records, raw=True)
        combined, errors = summarize(records, harness.threshold, harness.temperature, collect_errors=True, harness=harness)
        unseen = [record for record in records if record[0]["spans"] and
                  all(record[0]["text"][s["start"]:s["end"]] not in train_values for s in record[0]["spans"])]
        unseen_metrics, _ = summarize(unseen, harness.threshold, harness.temperature, harness=harness)
        report["splits"][split] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "model_only": raw,
                                    "model_and_harness": combined, "unseen_values": unseen_metrics,
                                    "seconds": round(time.time() - started, 2)}
        error_path = Path(checkpoint).parent / f"{split}_errors.json"
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"split": split, "model_exact": raw["exact_accuracy"], "harness": combined}, ensure_ascii=False), flush=True)
    report_path = Path(checkpoint).parent / ("evaluation.json" if splits == ["test", "audit"] else "_".join(splits) + "_evaluation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(report, report_path.with_suffix(".md"))
    return report


def write_report(report, path):
    lines = ["# filter1.0 评估报告", "", f"模型参数：{report['model']['parameters']:,}；类型：NONE / NAME / PATH / TEXT / VALUE。",
             "", "测试集为合成数据和助手独立编写的句子，不代表真实用户使用准确率。拒绝处理的正例计入整句失败。",
             "", "| 测试集与方式 | 整句正确 | 有参数整句正确 | 片段精确率 | 正例覆盖率 | 无参数误替换 | 完整还原 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    audit_results = report["splits"].get("audit")
    if audit_results and not all(audit_results["model_and_harness"]["targets_met"].values()):
        lines.insert(2, "**结论：独立审计未达到全部目标，当前适合受约束句式，尚不能声称覆盖大多数自由中文表达。**")
        lines.insert(3, "")
    for split, results in report["splits"].items():
        for key, label in [("model_only", "模型单独"), ("model_and_harness", "模型+harness")]:
            m = results[key]
            vals = [m[k] for k in ["exact_accuracy", "positive_exact_accuracy", "span_precision", "positive_coverage", "no_value_false_replacement_rate", "roundtrip_accuracy"]]
            lines.append(f"| {split}（{m['rows']}条） {label} | " + " | ".join(f"{v:.2%}" for v in vals) + " |")
    lines += ["", "## 验收目标", ""]
    for split, results in report["splits"].items():
        targets = results["model_and_harness"]["targets_met"]
        lines.append(f"- {split}：" + "；".join(f"{key}={'达标' if value else '未达标'}" for key, value in targets.items()))
    lines += ["", "## 解释与限制", "", "- NAME 只表示句中声明的名称，不等于已确认存在或唯一的文件。",
              "- 模型单独结果不应用引号边界约束；模型+harness 应用该约束并按校准阈值保守接受。",
              "- 模型独立评估使用原始槽类型；harness 评估允许 VALUE 按明确路径前缀细化为 PATH。",
              "- 覆盖率只统计有参数的句子，不能通过接受大量无参数句子提高此指标。",
              "- 字符偏移以 Python Unicode 码点计数；JavaScript 调用方需要换算 UTF-16 偏移。",
              "- 未达标项不能解读为已达到生产环境可靠性；可查看同目录的错误案例。",
              "- 跨表达家族划分不能保证所有抽象语法完全独立；同时给出独立审计集结果。", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

"""Calibrate on calibration only; inspect genuine neural gains on held-out frames."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .data import read_jsonl
from .evaluation import collect_scores, gold_for_harness, softmax, summarize
from .harness import FilterHarness, render_result
from .training import span_key
from .boundaries import constrain_probabilities, candidate_mask
from .model import LABEL_IDS
from .posttrain_evaluation import SyntaxOnly


def analyze(records, harness, policy=None, errors=False):
    policy = policy or harness.calibration
    ordinary = [r for r in records if r[0].get('expected_status') != 'needs_context']
    original = harness.calibration
    harness.calibration = policy
    metrics, failures = summarize(ordinary, policy.get('threshold', .95), policy.get('temperature', 1),
                                  harness=harness, collect_errors=errors)
    ambiguous = [r for r in records if r[0].get('expected_status') == 'needs_context']
    correct_status = false_accept = exact_slots = restored = 0
    routes = {}
    for row, pairs, logits in records:
        status, selected, proposals, info = harness.decode_scores(row['text'], pairs, logits,
            policy.get('threshold', .95), policy.get('temperature', 1), return_info=True)
        route = info.get('decision_source', 'legacy')
        routes[route] = routes.get(route, 0) + 1
        restored += render_result(row['text'], status, selected, proposals).restore() == row['text']
        if row.get('expected_status') == 'needs_context':
            correct_status += status == 'needs_context'
            false_accept += status in ('ok', 'no_literals')
            pred = set()
            from .harness import looks_like_path
            for s in selected:
                kind = 'PATH' if s['type'] == 'VALUE' and looks_like_path(row['text'][s['start']:s['end']]) else s['type']
                pred.add((s['start'], s['end'], kind))
            exact_slots += pred == span_key(gold_for_harness(row))
            if errors and (status != 'needs_context' or pred != span_key(gold_for_harness(row))):
                failures.append({'id': row['id'], 'text': row['text'], 'status': status,
                    'gold': gold_for_harness(row), 'predicted': selected, 'references': info.get('references', [])})
    harness.calibration = original
    metrics['ambiguous_reference'] = {'rows': len(ambiguous),
        'clarification_recall': correct_status/max(1, len(ambiguous)),
        'incorrect_autoaccept_rate': false_accept/max(1, len(ambiguous)),
        'explicit_slots_exact': exact_slots/max(1, len(ambiguous))}
    metrics['all_rows_roundtrip'] = restored/max(1, len(records))
    metrics['routes'] = routes
    return metrics, failures


def calibrate(checkpoint='artifacts/adaptive.pt', data_dir='data/adaptive', device='cpu'):
    torch.set_num_threads(4)
    h = FilterHarness(checkpoint, device)
    records = collect_scores(h, read_jsonl(Path(data_dir)/'calibration.jsonl'))
    nlls = {}
    for temp in (.65, .8, 1.0, 1.25, 1.5, 2):
        loss = count = 0
        for row, pairs, logits in records:
            probs = softmax(logits, temp)
            eligible = candidate_mask(row['text'], pairs)
            targets = np.zeros(len(pairs), dtype=np.int64)
            indices = {pair: i for i, pair in enumerate(pairs)}
            for s in row['spans']:
                targets[indices[(s['start'], s['end'])]] = LABEL_IDS[s['type']]
            loss += float(-np.log(np.maximum(probs[np.arange(len(pairs)), targets][eligible], 1e-12)).sum())
            count += int(eligible.sum())
        nlls[temp] = loss/count
    temperature = min(nlls, key=nlls.get)
    search = []
    for threshold in (.55, .6, .65, .7, .75, .8, .85, .9, .925, .95, .97, .98, .99, .995):
        policy = {'temperature': temperature, 'threshold': threshold,
                  'agreement_threshold': max(.5, threshold-.1), 'rule_threshold': threshold}
        m, _ = analyze(records, h, policy)
        search.append({'policy': policy, 'metrics': m})
    feasible = [e for e in search if e['metrics']['span_precision'] >= .98 and
        e['metrics']['accepted_sentence_accuracy'] >= .98 and
        e['metrics']['no_value_false_replacement_rate'] <= .01 and
        e['metrics']['counts']['predicted_spans'] > 0]
    chosen = max(feasible or search, key=lambda e: (
        e['metrics']['positive_exact_accuracy'] if feasible else e['metrics']['span_f1'],
        e['metrics']['span_precision'], e['policy']['threshold']))
    calibration = dict(chosen['policy'], precision_constraint_feasible=bool(feasible), metrics=chosen['metrics'],
        temperature_nll=nlls, rows=len(records),
        source_sha256=hashlib.sha256((Path(data_dir)/'calibration.jsonl').read_bytes()).hexdigest())
    payload = torch.load(checkpoint, weights_only=True, map_location='cpu')
    payload['calibration'] = calibration
    torch.save(payload, checkpoint)
    Path(checkpoint).with_name('adaptive_calibration.json').write_text(json.dumps(
        dict(calibration, search=search), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(calibration, ensure_ascii=False), flush=True)
    return calibration


def evaluate(checkpoint='artifacts/adaptive.pt', data_dir='data/adaptive', device='cpu', splits=None):
    torch.set_num_threads(4)
    new = FilterHarness(checkpoint, device)
    old = FilterHarness('artifacts/round4/filter1.0.pt', device)
    report = {'version': 'filter1.0', 'parameters': new.model.metadata()['parameters'],
        'training': new.training_info, 'calibration': new.calibration, 'splits': {},
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'challenge_used_for_aggregate_stopping': True,
        'challenge_errors_used_for_training_or_routing': False,
        'calibration_source': 'calibration.jsonl only',
        'limitations': ['Assistant-authored/synthetic tests, not real-user traffic.',
                        'Context-only candidate scoring cannot resolve every lexical ambiguity.',
                        'Pronouns are never mapped to objects from unseen history.']}
    for split in splits or ['challenge', 'previous_acceptance', 'original_synthetic']:
        path = Path(data_dir)/f'{split}.jsonl'
        if split == 'previous_acceptance':
            path = Path('data/posttrain/acceptance.jsonl')
        if split == 'original_synthetic':
            path = Path('data/test.jsonl')
        if split == 'challenge':
            pinned = json.loads(path.with_name('challenge_manifest.json').read_text())['sha256']
            if hashlib.sha256(path.read_bytes()).hexdigest() != pinned:
                raise ValueError('Frozen challenge changed')
        rows = read_jsonl(path)
        before, after = collect_scores(old, rows), collect_scores(new, rows)
        normal_before = [r for r in before if r[0].get('expected_status') != 'needs_context']
        normal_after = [r for r in after if r[0].get('expected_status') != 'needs_context']
        old_raw, _ = summarize(normal_before, raw=True)
        new_raw, _ = summarize(normal_after, raw=True)
        old_full_raw, _ = summarize(before, raw=True)
        new_full_raw, _ = summarize(after, raw=True)
        old_pipeline, _ = summarize(normal_before, old.threshold, old.temperature, harness=old)
        new_pipeline, errors = analyze(after, new, errors=True)
        rules, _ = summarize(normal_after, harness=SyntaxOnly())
        gaps = []
        for record in normal_after:
            row, pairs, logits = record
            status, spans, _ = SyntaxOnly().decode_scores(row['text'], pairs, logits, .5, 1)
            from .adaptive import canonical
            if status not in ('ok', 'no_literals') or canonical(row['text'], spans) != span_key(gold_for_harness(row)):
                gaps.append(record)
        gap_pipeline, _ = analyze(gaps, new)
        gap_raw, _ = summarize(gaps, raw=True)
        result = {'rows': len(rows), 'old_model_only': old_raw, 'new_model_only': new_raw,
            'old_model_full_comprehensive': old_full_raw, 'new_model_full_comprehensive': new_full_raw,
            'old_pipeline': old_pipeline, 'new_pipeline': new_pipeline, 'rules_only': rules,
            'rule_failure_subset': {'rows': len(gaps), 'new_model': gap_raw, 'new_pipeline': gap_pipeline}}
        report['splits'][split] = result
        Path(checkpoint).with_name(f'adaptive_{split}_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'split': split, 'old_model': old_raw['exact_accuracy'],
            'new_model': new_raw['exact_accuracy'], 'rules': rules['exact_accuracy'],
            'new_model_full_comprehensive': new_full_raw['exact_accuracy'],
            'pipeline': {k: new_pipeline[k] for k in ('exact_accuracy', 'span_precision', 'positive_coverage', 'ambiguous_reference')},
            'rule_gap_rows': len(gaps), 'rule_gap_exact': gap_pipeline['exact_accuracy']}, ensure_ascii=False), flush=True)
    output = Path(checkpoint).with_name('adaptive_evaluation.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(report, output.with_suffix('.md'))
    return report


def write_report(report, path):
    lines = ['# filter1.0 口语与模型接管评估', '',
        '继续训练现有权重；结构、232,197 个参数与模型名不变。规则结果需要模型支持，模型可以纠正规则或在规则失效时接管。', '',
        'challenge 在训练前冻结，包含 760 条口语/多参数表达、100 条无参数句与 120 条不明确指代句。'
        '均为助手编写/合成数据，不代表真实用户效果。按表达划分仍不能保证抽象语法完全独立。', '',
        '按用户要求反复检查综合测试的汇总成绩，以达到 80% 作为停止条件；'
        '测试因此参与了停止/选优，不能称为完全未查看的一次性独立评测。'
        '测试错误没有用于定向造样本、训练、阈值校准或 harness 调参。', '',
        '| 数据集 | 原模型单独 | 新模型单独 | 纯规则 | 原系统 | 新系统 | 新系统片段精确率 | 覆盖率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, r in report['splits'].items():
        m = r['new_pipeline']
        lines.append(f'| {split} | ' + ' | '.join(f'{v:.2%}' for v in [
            r['old_model_only']['exact_accuracy'], r['new_model_only']['exact_accuracy'], r['rules_only']['exact_accuracy'],
            r['old_pipeline']['exact_accuracy'], m['exact_accuracy'], m['span_precision'], m['positive_coverage']]) + ' |')
    lines += ['', '## 规则失效时与指代处理', '']
    for split, r in report['splits'].items():
        full = r['new_model_full_comprehensive']
        lines.append(f"- {split}：包含指代句的纯模型综合整句正确率 **{full['exact_accuracy']:.2%}**，共 {full['rows']} 句。"
                     '纯模型只评估显式片段，不具备 needs_context 输出；指代澄清另列。')
    for split, r in report['splits'].items():
        g = r['rule_failure_subset']
        ref = r['new_pipeline']['ambiguous_reference']
        lines.append(f"- {split}：规则失败 {g['rows']} 句，新系统整句补救正确率 {g['new_pipeline']['exact_accuracy']:.2%}；"
                     f"指代不明 {ref['rows']} 句，提示澄清率 {ref['clarification_recall']:.2%}，错误自动接受率 {ref['incorrect_autoaccept_rate']:.2%}。")
    lines += ['', '## 解释', '',
        '- 常规提取指标排除必须澄清的指代句；这些句子单列，不能通过猜测名称来提高提取准确率。',
        '- needs_context 可保留已确认的显式参数；references 记录未确定的原文指代，不生成假名字。',
        '- 当前句子里只有一个明确先行对象时，可记录简单指代关联；不查历史消息、桌面状态或文件存在性。',
        '- precision_constraint_feasible 见校准 JSON；未达标时不宣称部署可靠性。',
        '- 拒绝处理计入常规整句失败，覆盖率和自动接受精确率同时报告。',
        '- confidence 是候选类别分数，不是系统正确概率。所有输出都应可无损还原。',
        '- 旧测试仅用于回归比较，独立能力以新 challenge 为准。错误详见同目录 JSON。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')

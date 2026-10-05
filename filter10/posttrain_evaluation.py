"""Frozen acceptance evaluation plus controlled model/harness ablations."""
import hashlib
import json
import time
from pathlib import Path

import torch

from .data import read_jsonl
from .evaluation import collect_scores, summarize
from .harness import FilterHarness
from .syntax import analyze_syntax


class SyntaxOnly:
    def decode_scores(self, text, pairs, logits, threshold, temperature):
        evidence = analyze_syntax(text)
        if evidence.no_literals:
            return 'no_literals', [], []
        proposals = [dict(s, confidence=0.0, decision_source='syntax') for s in evidence.spans]
        if evidence.pending or evidence.ambiguous:
            return 'uncertain', [], proposals
        return ('ok', proposals, proposals) if proposals else ('no_match', [], [])


def evaluate_posttrain(checkpoint='artifacts/posttrained.pt', device='cpu'):
    torch.set_num_threads(4)
    new = FilterHarness(checkpoint, device)
    old = FilterHarness('artifacts/baseline/filter1.0.pt', device)
    old_new = FilterHarness('artifacts/baseline/filter1.0.pt', device)
    old_new.posttrained = True
    out = Path(checkpoint).parent
    report = {'version': 'filter1.0', 'parameters': new.model.metadata()['parameters'],
              'training': new.training_info, 'calibration': new.calibration,
              'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              'syntax_sha256': hashlib.sha256(Path(__file__).with_name('syntax.py').read_bytes()).hexdigest(),
              'stage_one_fresh_used_for_harness_development': True,
              'acceptance_used_for_tuning': False, 'splits': {}}
    seen = {r['text'] for split in ('train', 'validation', 'calibration')
            for r in read_jsonl(Path('data/posttrain') / f'{split}.jsonl')}
    for split, path in [('acceptance', Path('data/posttrain/acceptance.jsonl')),
                        ('fresh_development', Path('data/posttrain/fresh.jsonl')),
                        ('regression', Path('data/posttrain/regression.jsonl')),
                        ('original_synthetic', Path('data/test.jsonl'))]:
        if split in ('acceptance', 'fresh_development'):
            manifest = json.loads(path.with_name(path.stem + '_manifest.json').read_text(encoding='utf-8'))
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
                raise ValueError('Frozen test corpus changed')
        started = time.time()
        rows = read_jsonl(path)
        excluded = []
        if split == 'acceptance':
            excluded = [r['id'] for r in rows if r['text'] in seen]
            rows = [r for r in rows if r['text'] not in seen]
        before = collect_scores(old, rows)
        after = collect_scores(new, rows)
        results = {}
        cases = [
            ('old_model_only', before, None, 1.0, 0.5, True),
            ('posttrained_model_only', after, None, 1.0, 0.5, True),
            ('old_model_old_harness', before, None, old.temperature, old.threshold, False),
            ('posttrained_model_old_harness', after, None, old.temperature, old.threshold, False),
            ('old_model_new_harness', before, old_new, new.temperature, new.threshold, False),
            ('posttrained_model_new_harness', after, new, new.temperature, new.threshold, False),
            ('syntax_only', after, SyntaxOnly(), new.temperature, new.threshold, False),
        ]
        for key, records, decoder, temperature, threshold, raw in cases:
            metrics, errors = summarize(records, threshold, temperature, raw=raw,
                                        harness=decoder, collect_errors=key == 'posttrained_model_new_harness')
            results[key] = metrics
            if key == 'posttrained_model_new_harness':
                (out / f'posttrain_{split}_errors.json').write_text(
                    json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
        results['rows'] = len(rows)
        results['excluded_existing_sentences'] = excluded
        results['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        results['seconds'] = round(time.time() - started, 2)
        report['splits'][split] = results
        print(json.dumps({'split': split, 'comparison': {k: {
            metric: results[k][metric] for metric in ('exact_accuracy', 'span_precision', 'positive_coverage')}
            for k, *_ in cases}}, ensure_ascii=False), flush=True)
    (out / 'posttrain_evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_report(report, out / 'posttrain_evaluation.md')
    return report


def write_report(report, path):
    fresh = report['splits']['acceptance']['posttrained_model_new_harness']
    lines = ['# filter1.0 后训练评估', '',
        '保留原来的双 GRU 结构、标签和模型名，共 232,197 个参数。继续更新原权重，未从零重建。', '',
        'acceptance 是第二阶段修复前冻结的 592 条助手编写测试，去除 1 条与已有训练句重复的无参数句，评估 591 条'
        '（32 种句式各 16 个实例、79 条无参数句）；不是实际用户流量。'
        '第一批 420 条测试暴露错误后已用于 harness 开发，称为 fresh_development。'
        '旧 300 条审计已用于问题分析，列为回归集。后二者不能再称为独立验证。', '',
        '**本次新测试：' + ('全部验收目标达标。**' if all(fresh['targets_met'].values()) else '存在未达标项，见下文。**'), '',
        '| 数据集 / 方式 | 整句正确 | 有参数整句正确 | 片段精确率 | 正例覆盖 | 接受句完整正确 | 无参数误替换 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels = {'old_model_only': '原模型单独', 'posttrained_model_only': '后训练模型单独',
        'old_model_old_harness': '原模型 + 原 harness', 'posttrained_model_old_harness': '后训练模型 + 原 harness',
        'old_model_new_harness': '原模型 + 新 harness', 'posttrained_model_new_harness': '后训练模型 + 新 harness',
        'syntax_only': '仅语法规则（消融）'}
    for split, result in report['splits'].items():
        for key, label in labels.items():
            m = result[key]
            lines.append(f'| {split} / {label} | ' + ' | '.join(f'{m[k]:.2%}' for k in (
                'exact_accuracy', 'positive_exact_accuracy', 'span_precision', 'positive_coverage',
                'accepted_sentence_accuracy', 'no_value_false_replacement_rate')) + ' |')
    lines += ['', '## 怎么解释这些数字', '',
        '- 模型单独逐一枚举候选，用外部上下文评分；不使用引号、路径规则或语法边界规则。',
        '- 新 harness 在明确的操作语法中确认边界和槽类型；不查名字字典，不接触用户文件。'
        '这些语法确认可以覆盖模型的低分，返回 decision_source=syntax；confidence 仍是该类型的真实模型分数，不能当作系统正确概率。',
        '- 神经回退接受阈值来自 1,000 条校准句；规则接受不由此阈值决定。温度及阈值见 JSON。',
        '- 原模型 + 新 harness 和纯规则消融用于检查系统提升来自何处。语法覆盖范围内，规则可能贡献绝大部分提升；'
        '不能把系统正确率宣传为这个小模型本身的自由中文理解能力。',
        '- 拒绝处理计入整句失败；无参数句只有已识别为 no_literals 才算成功。未识别不是成功。',
        '- 完整还原只证明片段保存无损，不证明语义正确。所有本次数据集的还原结果及计数均在 JSON 中。',
        '- 591 条仅有 79 条无参数负例；零次误替换不足以统计证明真实错误率低于 1%。'
        '同一家族的十六个实例相关，不应按 591 个独立真实场景解释。',
        '- 仍只适用于受支持的常见中文工作区命令；需要引号消除复杂名称、标点和连接词的歧义。'
        '不负责文件匹配、执行权限、跨句指代或复杂自然语言理解。', '', '## 新测试验收', '']
    lines += [f'- {key}：' + ('达标' if value else '未达标') for key, value in fresh['targets_met'].items()]
    lines += ['', '新测试错误案例保存在 posttrain_acceptance_errors.json；未针对这些错误再次调参。'
              '第一阶段原始结果保存在 posttrain_stage1/，如实保留失败记录。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')

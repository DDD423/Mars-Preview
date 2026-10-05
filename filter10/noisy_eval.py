"""Fixed raw-model evaluation, with no syntactic or path correction."""
import hashlib
import json
from pathlib import Path
import torch
from .data import read_jsonl
from .evaluation import collect_scores, summarize
from .harness import FilterHarness, select_model_spans
from .training import span_key


def evaluate(checkpoint='artifacts/noisy.pt', directory='data/noisy', device='cpu', split='challenge', final=False):
    torch.set_num_threads(4)
    p = Path(checkpoint); data = Path(directory)/(split+'.jsonl')
    if split == 'challenge':
        pin = json.loads(data.with_name('challenge_manifest.json').read_text())['sha256']
        if hashlib.sha256(data.read_bytes()).hexdigest()!=pin:
            raise ValueError('Frozen noisy challenge changed')
    h = FilterHarness(p, device)
    rows = read_jsonl(data); records = collect_scores(h, rows)
    metrics, errors = summarize(records, raw=True, collect_errors=final)
    matches = restored = 0
    if h.neural_only:
        from .harness import select_spans, render_result
        for row, pairs, logits in records:
            raw = select_model_spans(pairs, logits, row.get('_count_logits'), h.count_policy)
            status, selected, proposals = h.decode_scores(row['text'], pairs, logits, count_logits=row.get('_count_logits'))
            matches += span_key(selected)==span_key(raw)
            restored += render_result(row['text'], status, selected, proposals, refine_paths=False).restore()==row['text']
    report = {'version':'filter1.0','parameters':h.model.metadata()['parameters'],
        'checkpoint':str(p),'checkpoint_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'split':split,
        'training':h.training_info,'metric_policy':'raw softmax, threshold=0.5, weighted non-overlapping selection; no text rules',
        'metrics':metrics,'goal_90_met':metrics['exact_accuracy']>=.9,
        'thin_harness_model_match_rate':matches/len(rows) if h.neural_only else None,
        'thin_harness_roundtrip':restored/len(rows) if h.neural_only else None,
        'source':'synthetic noisy robustness benchmark, shared compositional generator with training',
        'test_aggregate_used_for_stopping':True,'test_error_samples_used_for_training':False,
        'reference_scope':'Only explicit spans are scored; no hidden referent can be inferred.'}
    output = p.with_name('noisy_'+split+'_evaluation.json')
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if final:
        output.with_name('noisy_'+split+'_errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


def comprehensive(checkpoint='artifacts/noisy.pt', device='cpu', final=False):
    """Both frozen suites and their predeclared rule-error subsets must pass."""
    torch.set_num_threads(4)
    frozen=json.loads(Path('artifacts/frozen_rule_gap.json').read_text())
    if hashlib.sha256(Path('filter10/syntax.py').read_bytes()).hexdigest()!=frozen['syntax_sha256']:
        raise ValueError('Reference grammar changed after benchmark freeze')
    h=FilterHarness(checkpoint,device)
    all_records=[]; all_gaps=[]; results={}; matches=restored=0
    for directory in ('data/noisy','data/semantic'):
        p=Path(directory)/'challenge.jsonl'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=frozen['splits'][directory]['data_sha256']:
            raise ValueError('Frozen suite changed')
        records=collect_scores(h,read_jsonl(p)); gap_ids=set(frozen['splits'][directory]['gap_ids'])
        gaps=[r for r in records if r[0]['id'] in gap_ids]
        metrics,errors=summarize(records,raw=True,collect_errors=final)
        span_metrics,_=summarize(records,raw=True,ignore_counts=True)
        gap_metrics,_=summarize(gaps,raw=True)
        results[directory]={'raw_model':metrics,'span_head_only':span_metrics,
                            'fixed_rules_exact':frozen['splits'][directory]['rule_span_exact'],
                            'rule_gap':gap_metrics}
        all_records+=records; all_gaps+=gaps
        if h.neural_only:
            from .harness import select_spans, render_result
            for row,pairs,logits in records:
                raw=select_model_spans(pairs,logits,row.get('_count_logits'),h.count_policy)
                status,selected,proposals=h.decode_scores(row['text'],pairs,logits,count_logits=row.get('_count_logits'))
                matches+=span_key(raw)==span_key(selected)
                restored+=render_result(row['text'],status,selected,proposals,refine_paths=False).restore()==row['text']
        if final:
            Path(checkpoint).with_name(Path(directory).name+'_final_errors.json').write_text(
                json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    overall,_=summarize(all_records,raw=True)
    gap_overall,_=summarize(all_gaps,raw=True)
    passes=all(v['raw_model']['exact_accuracy']>=.9 and v['rule_gap']['exact_accuracy']>=.9 for v in results.values())
    report={'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'training':h.training_info,'parameters':h.model.metadata()['parameters'],
        'learned_parameter_count':h.learned_count,
        'count_policy':h.count_policy if h.learned_count else 'off',
        'sentence_query_task':h.sentence_query_task if h.learned_count else None,
        'metric':'raw model, fixed threshold 0.5, no grammar/quote/path/reference correction',
        'suite_results':results,'combined':overall,'combined_rule_gap':gap_overall,'goal_met':passes,
        'thin_harness_model_match_rate':matches/len(all_records) if h.neural_only else None,
        'thin_harness_roundtrip':restored/len(all_records) if h.neural_only else None,
        'benchmark_limitations':['Noisy suite shares a generator with training.',
            'Semantic suite holds out full sentence templates, not every abstract linguistic structure.',
            'Tests are assistant-authored/synthetic, not real-user traffic.',
            'Aggregate test scores are repeatedly checked for stopping; no test error mining.',
            'Explicit spans only: no hidden pronoun resolution or operating-system execution.']}
    Path(checkpoint).with_name('neural_comprehensive_evaluation.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(report,Path(checkpoint).with_name('neural_evaluation.md'))
    print(json.dumps({'goal_met':passes,'combined_exact':overall['exact_accuracy'],
        'combined_rule_gap_exact':gap_overall['exact_accuracy'],
        'suites':{k:{'model':v['raw_model']['exact_accuracy'],'rule_gap_model':v['rule_gap']['exact_accuracy'],
                     'rules':v['fixed_rules_exact']} for k,v in results.items()}},ensure_ascii=False),flush=True)
    return report


def write_report(report,path):
    training=report['training']
    lines=['# filter1.0 纯模型评估','',
        '**本轮全部停止条件已达到。**' if report['goal_met'] else '**尚未达到全部停止条件，训练仍在继续。**','',
        f"参数 {report['parameters']:,}；实际权重为 run {training.get('run')} / epoch {training.get('epoch')}。",
        '固定阈值 0.5，按神经评分选择互不重叠片段。没有语法、引号、路径或指代修正。',
        '整句只有全部片段边界与类型集合完全正确才算成功；多提、漏提、错类型都算失败。','',
        '| 冻结挑战 | 条目 | 纯模型整句 | 有参数句整句 | 片段精确率 | 片段召回率 | 固定规则整句 | 规则错误子集上的模型整句 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for directory,v in report['suite_results'].items():
        m=v['raw_model'];gap=v['rule_gap']
        lines.append(f"| {directory} | {m['rows']} | {m['exact_accuracy']:.2%} | {m['positive_exact_accuracy']:.2%} | {m['span_precision']:.2%} | {m['span_recall']:.2%} | {v['fixed_rules_exact']:.2%} | {gap['exact_accuracy']:.2%}（{gap['rows']}条） |")
    if report.get('learned_parameter_count'):
        lines += ['', '整句辅助判断由同一双GRU和同一五分类头的另一个神经查询学习；没有新增参数。候选分类仍只读取候选外部上下文。',
                  '数量查询读取整句，因此参数内部变化可能影响数量查询，但不会影响该候选的类型评分。']
        if report.get('sentence_query_task') == 'parameter_presence':
            lines += ['本权重的整句任务为有无显式参数（0=无，1=有），不预测确切数量。']
        lines += [f"本权重的数量使用策略为 `{report.get('count_policy','cap')}`：cap限制数量，zero_only仅用高置信的无参数判断，off不参与推理。"]
        if report.get('count_policy','cap') == 'cap':
            lines += ['数量置信至少0.5时限制输出数量；4或更多与低置信数量不设上限。数量判断可能改变最终提取。']
        elif report.get('count_policy') == 'zero_only':
            lines += ['只有无参数概率至少0.5时不提取片段；其他情况直接采用候选头的数值选择。']
        else:
            lines += ['数量查询仅作为辅助训练任务，完整推理使用候选头，不用整句数量覆盖其判断。']
        for directory,v in report['suite_results'].items():
            lines.append(f"{directory} 仅候选头、不使用神经数量判断的整句正确率：{v['span_head_only']['exact_accuracy']:.2%}。")
    overall=report['combined'];gap=report['combined_rule_gap']
    lines += ['',f"合计 {overall['counts']['correct_sentences']}/{overall['rows']}，整句 {overall['exact_accuracy']:.2%}；固定规则错误子集合计 {gap['exact_accuracy']:.2%}（{gap['rows']}条）。",
        f"无参数句误替换 {overall['counts']['false_no_value_rows']}/{overall['counts']['no_value_rows']}；片段精确率 {overall['span_precision']:.2%}，召回率 {overall['span_recall']:.2%}。",
        f"薄 harness 与纯模型片段一致率：{report.get('thin_harness_model_match_rate')}；完整还原率：{report.get('thin_harness_roundtrip')}。",
        '', '## 训练与验证', '',f"训练目录：{training.get('data_dir','历史权重未记录')}；该轮 {training.get('rows')} 条数据。",
        f"AdamW，初始学习率 {training.get('learning_rate')}，批量 {training.get('batch_size')}；候选排序损失权重 {training.get('ranking_weight')}，正确参数额外分类损失权重 {training.get('positive_loss_weight',0)}。",
        '仍在原有双 GRU 上继续训练；没有新增模型参数或名称词典。训练材料中名称只用于填入参数和构造错误边界。',
        '校验项包括 UTF-8 编码、候选内部不可见、结构化还原和冻结数据完整性。',
        '', '## 如何解释测试', '',
        '噪音挑战与训练共享组合生成方式。语义挑战隔离完整句式，但抽象语法、词语和结构零件会重叠；500条语义实例实际只有498个不同输入、20种有参数完整句式和4种无参数核心表达。',
        '两套挑战及各自固定规则错误子集都达到90%才满足本轮停止条件。规则错误子集在本阶段语义训练前固定；它只说明这版规则失败，不说明任何程序都无法处理。',
        '测试汇总反复查看用于停止，逐条挑战错误不用于训练；因此成绩存在选择偏差，不能当作一次性盲测估计。',
        '数据是助手编写或合成，成绩不代表真实用户自由中文正确率。基础短句与旧审计属于开发/历史回归，另行报告。',
        '只提取显式参数，不确认文件存在、操作授权或隐含指代。不执行文件操作。超过256个Unicode字符明确返回unsupported。',
        '改变候选内部文字不改变该候选神经评分；其他候选竞争和整句结果仍可能改变。',
        'softmax置信分数不是正确概率。no_literals只是没有提取到显式参数，不表示已理解整句。',
        '', '## 校验值', '',f"权重 SHA-256：`{report['checkpoint_sha256']}`",f"父权重 SHA-256：`{training.get('parent_sha256')}`"]
    if training.get('structured_weight'):
        i=lines.index('## 如何解释测试')
        lines[i:i]=[f"整句片段集合损失权重 {training['structured_weight']}；在采样候选上学习正确类型且互不重叠的完整集合，惩罚多提、漏提和竞争边界。无参数句仍由独立的整句神经查询监督。",
                    f"训练长度分组块大小 {training.get('length_bucket_size',0)}；只影响批次计算，不影响推理或参数量。",'']
    if training.get('counterfactual_weight'):
        i=lines.index('## 如何解释测试')
        lines[i:i]=[f"参数内容扰动训练权重 {training['counterfactual_weight']}，每行概率 {training['counterfactual_probability']}：只改已标注参数的UTF-8输入，保留Unicode边界、控制语句和标签。推理没有此扰动；不增加参数或名称分类表。",'']
    if training.get('exclude_context_aliases'):
        info=training.get('context_alias_preparation',{})
        i=lines.index('## 如何解释测试')
        lines[i:i]=[f"仅训练阶段排除前后文相同的矛盾NONE监督：{info.get('alias_intervals')}个候选，影响{info.get('affected_rows')}行。候选仍参与排序与集合损失；正例标注不变。外部上下文匹配表不用于推理、不存入权重。",'']
    for split,pin in training.get('data_sha256',{}).items():lines.append(f"{split} SHA-256：`{pin}`")
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

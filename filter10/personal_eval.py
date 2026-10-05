"""Evaluate user-labelled JSONL with the fixed pure-model policy."""
import hashlib
import json
from pathlib import Path
import torch
from .data import read_jsonl
from .evaluation import collect_scores, summarize
from .harness import FilterHarness
from .model import LABELS


def evaluate(input,checkpoint='artifacts/filter1.0.pt',device='cpu',output='artifacts/personal_evaluation.json'):
    torch.set_num_threads(4)
    source=Path(input);rows=read_jsonl(source)
    if not rows:raise ValueError('测试文件为空')
    for i,row in enumerate(rows,1):
        row.setdefault('id',f'personal_{i}');text=row['text']
        if not isinstance(text,str) or not 0<len(text)<=256:
            raise ValueError(f'第{i}条需包含1到256个Unicode字符')
        text.encode('utf-8');end=0
        for span in sorted(row['spans'],key=lambda s:s['start']):
            if span['type'] not in LABELS[1:] or not end<=span['start']<span['end']<=len(text):
                raise ValueError(f'第{i}条标签类型、边界或重叠不合法')
            end=span['end']
    h=FilterHarness(checkpoint,device)
    metrics,errors=summarize(collect_scores(h,rows),raw=True,collect_errors=True)
    report={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            'policy':'pure model, fixed threshold 0.5; exact boundaries and types',
            'metrics':metrics,'errors':errors}
    dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'report':str(dest),'rows':len(rows),'exact_accuracy':metrics['exact_accuracy'],
                      'span_precision':metrics['span_precision'],'span_recall':metrics['span_recall']},ensure_ascii=False))
    return report

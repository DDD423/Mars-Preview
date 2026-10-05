"""Inspect benchmark structure, without collecting model/test errors."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .data import read_jsonl


def inventory(data_dir='data/neural_v7', output='artifacts/benchmark_inventory.json'):
    training = read_jsonl(Path(data_dir) / 'train.jsonl')
    training_text = {r['text'] for r in training}
    training_frames = {r['template'] for r in training if r.get('template')}
    training_literals = {r['text'][s['start']:s['end']] for r in training for s in r['spans']}
    suites = {}
    for directory in ('data/noisy', 'data/semantic'):
        source = Path(directory) / 'challenge.jsonl'
        rows = read_jsonl(source)
        frames = Counter(r['template'] for r in rows if r.get('template'))
        positive_frames = {r['template'] for r in rows if r['spans'] and r.get('template')}
        negative_frames = {r['template'] for r in rows if not r['spans'] and r.get('template')}
        positives = [r for r in rows if r['spans']]
        suites[directory] = {
            'rows': len(rows), 'unique_sentences': len({r['text'] for r in rows}),
            'positive_rows': len(positives), 'no_argument_rows': len(rows) - len(positives),
            'literal_types': dict(Counter(s['type'] for r in rows for s in r['spans'])),
            'argument_count_distribution': dict(sorted(Counter(len(r['spans']) for r in rows).items())),
            'recorded_positive_full_frames': len(positive_frames), 'recorded_negative_full_frames': len(negative_frames),
            'rows_with_recorded_full_frame': sum(frames.values()),
            'largest_repeated_full_frame_rows': max(frames.values()) if frames else None,
            'categories': dict(Counter(r.get('category', 'unknown') for r in rows)),
            'train_sentence_overlap': len(training_text & {r['text'] for r in rows}),
            'shared_full_templates': len(training_frames & set(frames)) if len(frames) else None,
            'fully_unseen_literal_rows': sum(all(r['text'][s['start']:s['end']] not in training_literals
                                                for s in r['spans']) for r in positives),
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    report = {'training_directory': str(data_dir), 'training_rows': len(training),
              'training_unique_sentences': len(training_text), 'training_full_frames': len(training_frames),
              'suites': suites,
              'interpretation': 'Instance counts are not independent expression-family counts. No predictions/errors read; no test fitting.'}
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report

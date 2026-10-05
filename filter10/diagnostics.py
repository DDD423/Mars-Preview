"""Reproducible local inference timings and inspectable example outputs."""
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from .harness import FilterHarness


def diagnostics(checkpoint='artifacts/filter1.0.pt',data_dir=None):
    torch.set_num_threads(4)
    harness = FilterHarness(checkpoint)
    training_data=Path(data_dir or harness.training_info.get('data_dir','data/neural_v3' if harness.training_info.get('run',0)>=11 else 'data/neural_v2'))
    bench = {'device': 'cpu', 'threads': 4, 'platform': platform.platform(),
             'torch': torch.__version__, 'checkpoint_bytes': Path(checkpoint).stat().st_size,
             'includes': 'GRU encoding, all-candidate scoring, harness and rendering; excludes process/model startup',
             'measurements': []}
    for length in (32, 128, 256):
        text = '删除叫做' + 'N' * (length - 7) + '的文件'
        for _ in range(5):
            harness.extract(text)
        samples = []
        for _ in range(30):
            start = time.perf_counter()
            result = harness.extract(text)
            samples.append((time.perf_counter() - start) * 1000)
        bench['measurements'].append({'characters': len(text), 'candidate_count': length * (length+1)//2,
            'repeats': len(samples), 'median_ms': round(statistics.median(samples), 2),
            'p95_ms': round(sorted(samples)[int(.95*len(samples))-1], 2), 'status': result.status})
    demos = [harness.extract(text).to_dict() for text in [
        '删除叫做删除的文件', '创建一个文件，名字叫Raven Report.txt',
        '从路径\\\\host\\share\\叫做.txt的文件里提取文字',
        '向名为打开的文件追加文字🔍关键字',
        '请把叫做叫做的文档重命名为删除，保留提示{1}',
        '不要删除名为“重命名为”的文件',
        '把原文“line A\nline B”写入名称为“Z.txt”的文件',
        '向名为Z的文件追加文字', '显示工作区所有文件', '这是一句天气闲聊。']]
    for result in demos:
        assert harness.extract(result['original']).restore() == result['original']
    directory = Path(checkpoint).parent
    prefix='neural' if harness.neural_only else 'posttrain'
    (directory / (prefix+'_benchmark.json')).write_text(json.dumps(bench, indent=2), encoding='utf-8')
    (directory / (prefix+'_demos.json')).write_text(json.dumps(demos, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = {'version': 'filter1.0', 'posttrained': True, 'architecture_changed': False,
        'parameters': harness.model.metadata()['parameters'], 'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'parent_sha256': harness.training_info['parent_sha256'],
        'neural_only':harness.neural_only,
        'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            ([Path('filter10/model.py'), Path('filter10/harness.py'), Path('filter10/noisy_eval.py'),
              Path('filter10/training.py'), Path('filter10/scope_data.py'), Path('filter10/__main__.py'),
              Path('requirements.lock.txt'), Path('artifacts/run24_command.ps1'),
              training_data/'train.jsonl',training_data/'validation.jsonl',training_data/'calibration.jsonl',
              Path('data/noisy/challenge.jsonl'),Path('data/semantic/challenge.jsonl'),
              Path('artifacts/frozen_rule_gap.json')] if harness.neural_only else
             [Path('filter10/model.py'), Path('filter10/harness.py'), Path('filter10/syntax.py'),
              Path('data/posttrain/train.jsonl'), Path('data/posttrain/acceptance.jsonl')])},
        'report': 'artifacts/neural_evaluation.md' if harness.neural_only else 'artifacts/posttrain_evaluation.md',
        'scope': 'Chinese workspace-command literal extraction; no file execution'}
    if harness.neural_only:
        manifest['selection'] = json.loads(Path('artifacts/default_selection.json').read_text(encoding='utf-8'))
        manifest['selection_applies_to_checkpoint'] = manifest['selection']['checkpoint_sha256'] == manifest['checkpoint_sha256']
    (directory / (prefix+'_release_manifest.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(bench, ensure_ascii=False, indent=2))
    return bench

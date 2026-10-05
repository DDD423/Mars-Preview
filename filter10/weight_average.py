"""Average aligned checkpoints from one post-training lineage.

There is one model at inference, with unchanged layers and parameter count.
Coefficient selection must use development data only, never challenge errors.
"""
import hashlib
from pathlib import Path
import torch
from .model import ContextSpanModel, ModelConfig
from .training import save_checkpoint


def average(checkpoints, output, coefficients=None):
    if len(checkpoints) < 2:
        raise ValueError('At least two aligned checkpoints are required')
    payloads = [torch.load(p, map_location='cpu', weights_only=True) for p in checkpoints]
    coefficients = list(coefficients or [1.0] * len(checkpoints))
    if len(coefficients) != len(payloads) or any(w < 0 for w in coefficients) or sum(coefficients) <= 0:
        raise ValueError('Coefficients must be nonnegative, aligned with checkpoints, and have positive sum')
    coefficients = [w / sum(coefficients) for w in coefficients]
    base = payloads[-1]
    hashes = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in checkpoints]
    for index in range(1, len(payloads)):
        if payloads[index].get('training', {}).get('parent_sha256') != hashes[index - 1]:
            raise ValueError('Supply consecutive checkpoints from one verified parent lineage')
    for item in payloads:
        if item['config'] != base['config'] or item['labels'] != base['labels'] or item['version'] != base['version']:
            raise ValueError('Incompatible architectures or labels')
        if not item.get('training', {}).get('neural_only'):
            raise ValueError('Only fixed-policy neural checkpoints can be averaged here')
        if list(item['state_dict']) != list(base['state_dict']):
            raise ValueError('State dictionary structures differ')
    model = ContextSpanModel(ModelConfig(**base['config']))
    state = {}
    for key, value in base['state_dict'].items():
        tensors = [p['state_dict'][key] for p in payloads]
        if any(t.shape != value.shape or t.dtype != value.dtype for t in tensors):
            raise ValueError('State tensor shape/dtype mismatch: ' + key)
        if value.is_floating_point():
            state[key] = sum(w * t.double() for w, t in zip(coefficients, tensors)).to(value.dtype)
        elif all(torch.equal(t, value) for t in tensors):
            state[key] = value
        else:
            raise ValueError('Non-floating state differs: ' + key)
    model.load_state_dict(state)
    training = dict(base['training'])
    training.update({'weight_average': [dict(checkpoint=str(p), coefficient=w,
                                            sha256=pin, run=item['training'].get('run'),
                                            epoch=item['training'].get('epoch'),
                                            data_dir=item['training'].get('data_dir'))
                                        for p, w, pin, item in zip(checkpoints, coefficients, hashes, payloads)],
                     'parent_sha256': hashes[-1], 'method': 'aligned_checkpoint_weight_average',
                     'inherited_training_metadata_describes_last_source': True,
                     'validation': None,
                     'validation_policy': 'unvalidated_weight_average; evaluate on development before selection'})
    save_checkpoint(model, output, {'training': training})
    return {'output': str(output), 'parameters': model.metadata()['parameters'],
            'weight_average': training['weight_average']}

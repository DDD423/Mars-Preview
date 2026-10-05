"""Inspect the model, fixed syntax rules and deployed pipeline separately."""
from .harness import select_spans, select_model_spans
from .syntax import analyze_syntax


def compare(harness, text):
    pipeline = harness.extract(text)
    if pipeline.status == 'unsupported' or not text:
        return {'pipeline': pipeline.to_dict(), 'model_only': None, 'rules_only': None}
    item = harness.model.all_scores([text], harness.device, with_counts=harness.learned_count)[0]
    pairs, logits = item[:2]
    counts = item[2] if harness.learned_count else None
    neural = select_model_spans(pairs, logits, counts, harness.count_policy)
    span_head = select_spans(pairs, logits.softmax(-1).numpy())
    syntax = analyze_syntax(text)
    enrich = lambda spans: [dict(s, value=text[s['start']:s['end']]) for s in spans]
    return {'input': text, 'model_only': {'spans': enrich(neural),
                'policy': 'raw model, threshold 0.5; no quote/path/syntax refinement'},
            'rules_only': {'spans': enrich(syntax.spans), 'pending': syntax.pending,
                           'ambiguous': syntax.ambiguous, 'no_literals': syntax.no_literals},
            'span_head_only': {'spans':enrich(span_head)},
            'learned_count_probabilities':counts.softmax(-1).tolist() if counts is not None else None,
            'count_policy':harness.count_policy if harness.learned_count else 'off',
            'sentence_query_task':harness.sentence_query_task if harness.learned_count else None,
            'pipeline': pipeline.to_dict()}

"""Deterministic syntax constraints, without interpreting literal contents.

With strict external-context scoring, a candidate including quotation marks
is indistinguishable from a genuine unquoted value with the same exterior.
This unavoidable ambiguity is resolved by the harness before selection.
"""
import numpy as np

QUOTE_PAIRS = {'"': '"', "'": "'", '“': '”', '「': '」', '‘': '’'}


def quoted_interiors(text):
    result = []
    active = None
    for index, char in enumerate(text):
        # A backslash-escaped quote belongs to the literal contents.
        backslashes, previous = 0, index - 1
        while previous >= 0 and text[previous] == "\\":
            backslashes += 1
            previous -= 1
        if backslashes % 2:
            continue
        if active is None and char in QUOTE_PAIRS:
            active = (index, QUOTE_PAIRS[char])
        elif active is not None and char == active[1]:
            if index > active[0] + 1:
                result.append((active[0] + 1, index))
            active = None
    return result


def eligible_span(start, end, interiors):
    for inner_start, inner_end in interiors:
        quote_start, quote_end = inner_start - 1, inner_end + 1
        if end <= quote_start or start >= quote_end:
            continue
        if (start, end) != (inner_start, inner_end):
            return False
    return True


def candidate_mask(text, pairs):
    interiors = quoted_interiors(text)
    if not interiors:
        return np.ones(len(pairs), dtype=bool)
    return np.fromiter((eligible_span(s, e, interiors) for s, e in pairs), dtype=bool, count=len(pairs))


def constrain_probabilities(text, pairs, probabilities):
    mask = candidate_mask(text, pairs)
    if bool(mask.all()):
        return probabilities
    constrained = probabilities.copy()
    constrained[~mask] = 0
    constrained[~mask, 0] = 1
    return constrained

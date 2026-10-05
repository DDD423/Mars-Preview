"""Model-led decisions with syntax evidence and explicit unresolved references.

No object is invented for an unresolved pronoun. A reference is resolved only
to one unambiguous preceding literal in the current sentence, never to history.
"""
import re
import numpy as np
from .boundaries import quoted_interiors
from .model import LABELS
from .syntax import analyze_syntax, NAME, PATH

REF = re.compile(r'刚(?:才|刚)(?:那个|那份|的(?:文件|文档))|前面(?:那个|那份)|那个(?:文件|文档|目录|东西)?|那份(?:文件|文档)?|它们|它|这边|那边|这里|那里')
META = re.compile(r'(?:只)?(?:查看|看看|显示|检查)(?:一下)?(?:文件|文档)(?:的)?(?:属性|大小|修改时间|创建时间)|(?:把)?(?:工作区)?文件按(?:时间|大小|名称)排序|列出所有文件的名称')
STOP = re.compile(r'(?:暂时)?(?:先)?(?:别|不要|不用)(?:再)?继续(?:操作)?(?:了)?|先(?:停|等)一下(?:吧)?|(?:所有|全部)文件(?:先)?(?:留着|保留)|别操作了')
SCOPE = re.compile(r'打开|删除|删|复制|拷|移|搬|读|提取|搜索|查找|搜|写|追加|添加|补|创建|新建|建个|弄个|命名|改名|名字|名称|列出|统计|检查|查看|看看|排序|属性|暂停|停止|操作|保留|备份|保持|等一下')
POLITE = re.compile(r'^(?:(?:嗯|呃|那个)，|麻烦你|能不能|我想让你|请帮我|帮我|请|先|现在)*')
TAIL = re.compile(r'(?:吧|就行|，谢谢|，别操作其他文件)*$')
DESCRIPTION = re.compile(r'(?:(?:名字|名称)(?:是|为|叫做|叫)|名为|名叫|叫做)')
SWALLOWED_CONTROL = re.compile(r'的(?:名字|名称)(?:换|改|更换|更改|设)|(?:这个|这份|那份)(?:文件|文档)|(?:复制一份|拷贝一份|拷贝|复制|移动|搬|挪|移)(?:到|进|至)|(?:先|再)(?:读取|打开|搜索)|里面(?:查找|搜索|写入)')


def confirmed_syntax(text, spans, evidence, neural=()):
    """Retain the mature grammar for clearly declared/quoted literal slots.

    A generic noun followed by an entire colloquial operation is not a
    confirmed literal. These cases remain with the model. This is a hybrid
    system: grammar decisions are exposed, not counted as learned accuracy.
    """
    if not spans or evidence.pending or evidence.ambiguous or evidence.no_literals or generic_no_literal(text):
        return False
    interiors = set(quoted_interiors(text))
    for candidate in neural:
        if candidate['confidence'] < .8:
            continue
        pair = (candidate['start'], candidate['end'])
        exact = any(pair == (r['start'], r['end']) and candidate['type'] == r['type'] for r in spans)
        if not exact and pair in interiors:
            return False  # a complete quoted operand was missed/misbounded
        outside = not any(r['start'] <= candidate['start'] and candidate['end'] <= r['end'] for r in spans)
        if outside and candidate['end']-candidate['start'] >= 2 and re.search(
                r'(?:名字|名称)(?:换为|换成|改为|改成|叫|设为)|文件夹|目录|关键词|原文|文字|，叫\s*$',
                text[:candidate['start']]):
            return False  # the grammar covered only part of a multi-slot command
    for s in spans:
        if (s['start'], s['end']) in interiors:
            continue
        if s['confidence'] < .05:
            return False  # learned evidence strongly contradicts the boundary
        if s['type'] == 'VALUE':
            return False
        if s['type'] == 'NAME' and not re.search(r'(?:'+NAME+r')\s*$', text[:s['start']]):
            value = text[s['start']:s['end']]
            if s['confidence'] < .2 or SWALLOWED_CONTROL.search(value) or re.search(
                    r'(?:名称|名字|先读取|再搜索|里追加|里搜索|里查找|里写入|读一下|读出来|不需要了|不要了|前先备份)', value):
                return False
            # Generic file/dir arguments are trusted only in the mature
            # command vocabulary; colloquial suffixes need the learned model.
            marker = text[s.get('marker_start', s['start']):s['start']]
            if not re.fullmatch(r'(?:把|将|打开|查看|读取|删除|删掉|给|向|复制|移动|复制到|复制至|移动到|移动至|移到|移入)(?:文件夹|文件|文档|目录)\s*', marker):
                return False
        if s['type'] == 'TEXT' and re.search(r'的(?:正文|内容)(?:换成|换为|改为|设为)', text[:s['start']]):
            return False
        if s['type'] in ('TEXT', 'PATH') and re.search(
                r'出现在哪|(?:所指|指向|对应)的(?:文件|文档)|(?:写进|放进)(?:文件|文档)|(?:吧|就行)$',
                text[s['start']:s['end']]):
            return False
    return True


def references(text, spans):
    quotes = quoted_interiors(text)
    found = []
    for m in REF.finditer(text):
        if any(s <= m.start() < e for s, e in quotes):
            continue
        if any(s['start'] <= m.start() and m.end() <= s['end'] and
               re.search(r'(?:' + NAME + r'|' + PATH + r')\s*$', text[:s['start']]) for s in spans):
            continue
        if m.group() == '那个' and text[m.end():].startswith(('，', ',')) and POLITE.fullmatch(text[:m.start()]):
            continue  # conversational filler, not an object reference
        description = DESCRIPTION.match(text, m.end()) if m.group() == '那个' else None
        described = [s for s in spans if description and
                     description.end() <= s['start'] <= description.end()+1]
        if len(described) == 1:
            s = described[0]
            found.append({'start': m.start(), 'end': m.end(), 'text': m.group(),
                'status': 'resolved_in_sentence', 'resolution': 'explicit_description',
                'antecedent_start': s['start'], 'antecedent_end': s['end']})
            continue
        prior = [s for s in spans if s['type'] in ('NAME', 'PATH', 'VALUE') and s['end'] <= m.start()]
        adjacent = [s for s in prior if s['end'] == m.start()]
        referent = adjacent[0] if len(adjacent) == 1 else prior[0] if len(prior) == 1 else None
        prior_mentions = [a for a in re.finditer(r'文件夹|文件|文档|目录|路径', text[:m.start()])
                          if not any(s <= a.start() < e for s, e in quotes)]
        if not adjacent and len(prior_mentions) > 1:
            referent = None
        if m.group() in ('这里', '那里', '这边', '那边', '它们'):
            referent = None
        item = {'start': m.start(), 'end': m.end(), 'text': m.group(),
                'status': 'resolved_in_sentence' if referent else 'unresolved'}
        if referent:
            item['antecedent_start'] = referent['start']
            item['antecedent_end'] = referent['end']
        found.append(item)
    return found


def generic_no_literal(text):
    body = TAIL.sub('', POLITE.sub('', text)).strip()
    return bool(META.fullmatch(body) or STOP.fullmatch(body))


def canonical(text, spans):
    from .harness import looks_like_path
    return {(s['start'], s['end'], 'PATH' if s['type'] == 'VALUE' and
             looks_like_path(text[s['start']:s['end']]) else s['type']) for s in spans}


def decode(text, pairs, probabilities, policy):
    from .harness import select_spans
    evidence = analyze_syntax(text)
    indices = {pair: i for i, pair in enumerate(pairs)}
    neural = select_spans(pairs, probabilities, .5)
    neural = [dict(s, model_type=s['type'], decision_source='model') for s in neural]
    rules = []
    for span in evidence.spans:
        p = probabilities[indices[(span['start'], span['end'])]]
        rules.append(dict(span, confidence=float(p[LABELS.index(span['type'])]),
                          model_type=LABELS[int(p.argmax())], decision_source='rule'))
    proposals = neural if neural else rules
    threshold = float(policy.get('threshold', .95))
    agreement_threshold = float(policy.get('agreement_threshold', .7))
    rule_threshold = float(policy.get('rule_threshold', .8))
    min_neural = min((s['confidence'] for s in neural), default=0)
    agreement = bool(neural) and canonical(text, neural) == canonical(text, rules)
    selected, route = [], 'abstain'

    if evidence.no_literals:
        selected, route = [], 'syntax_no_literals'
    elif confirmed_syntax(text, rules, evidence, neural):
        selected, route = [dict(s, decision_source='syntax_confirmed') for s in rules], 'syntax_confirmed'
    elif neural and min_neural >= (agreement_threshold if agreement else threshold):
        selected = neural
        route = 'agreement' if agreement else 'model_override' if rules else 'model_fallback'
    elif rules and not evidence.pending and not evidence.ambiguous:
        # Rules no longer accept independently of the model. Their complete
        # span/type set must be supported, and confident competing spans veto.
        supported = all(s['confidence'] >= rule_threshold for s in rules)
        contradict = any(s['confidence'] >= threshold and
                         (s['start'], s['end'], s['type']) not in
                         {(r['start'], r['end'], r['type']) for r in rules} for s in neural)
        if supported and not contradict:
            selected, route = rules, 'model_supported_rule'

    # Determine references from the accepted candidates, or high-confidence
    # suggestions when abstaining. Quoted literals are always opaque.
    reference_spans = selected or [s for s in proposals if s['confidence'] >= .8]
    refs = references(text, reference_spans)
    unresolved = [r for r in refs if r['status'] == 'unresolved']
    info = {'decision_source': route, 'references': refs, 'rule_model_agreement': agreement}
    if unresolved:
        # Confirmed explicit parameters may be retained, but no inferred name
        # is inserted for the pronoun; the result is not eligible for execution.
        info['decision_source'] = 'needs_context'
        selected = [s for s in selected if not any(s['start'] < r['end'] and s['end'] > r['start'] for r in unresolved)]
        return 'needs_context', selected, proposals, info

    # Explicitly missing arguments and unmatched leading quotes must not be
    # completed by a model hallucination. A genuine literal can contain markers.
    incomplete = any((p['end'] == len(text.rstrip()) and not any(
        s['start'] < p['start'] and s['end'] >= p['end'] and
        re.search(r'(?:' + NAME + r'|' + PATH + r')\s*$', text[:s['start']]) for s in selected))
        or (p['end'] < len(text) and text[p['end']] in '\"\'“「‘' and
            not any(a == p['end'] + 1 for a, b in quoted_interiors(text)))
        for p in evidence.pending)
    if incomplete:
        return 'uncertain', [], proposals, dict(info, decision_source='incomplete_parameter')
    if selected:
        return 'ok', selected, proposals, info
    # An empty learned span set is meaningful for in-domain commands such as
    # "统计文件数量". Scope detection is only a gate; it does not find names.
    # A borderline positive candidate or a missing argument still abstains.
    empty_supported = not neural and float(probabilities[:, 1:].max()) < float(policy.get('empty_threshold', .35))
    explicit_declaration = re.search(r'(?:' + NAME + r'|' + PATH + r')', text)
    contextual_operands = all(re.match(r'(?:到|至|进)?(?:当前|选中|所有|全部|这些|任何)', text[p['end']:])
                              for p in evidence.pending)
    if evidence.no_literals or (not neural and (generic_no_literal(text) or
                       (empty_supported and not rules and not explicit_declaration and contextual_operands and SCOPE.search(text)))):
        return 'no_literals', [], proposals, dict(info, decision_source=route if evidence.no_literals else 'model_and_no_literal_evidence')
    return ('uncertain' if neural or rules or evidence.pending else 'no_match'), [], proposals, info

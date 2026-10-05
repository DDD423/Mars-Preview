import json
import random
import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from .data import read_jsonl
from .harness import select_spans, select_model_spans
from .model import ContextSpanModel, ModelConfig, LABEL_IDS, encode_texts
from .boundaries import constrain_probabilities, eligible_span, quoted_interiors


def sample_candidates(row, rng, random_negatives=40, quote_competitors=False, raw_negatives=False):
    n = len(row["text"])
    positives = {(s["start"], s["end"]): LABEL_IDS[s["type"]] for s in row["spans"]}
    negatives = set()
    for start, end in positives:
        for delta_start in (-2, -1, 0, 1, 2):
            for delta_end in (-2, -1, 0, 1, 2):
                s, e = start + delta_start, end + delta_end
                if 0 <= s < e <= n and (s, e) not in positives:
                    negatives.add((s, e))
    for _ in range(random_negatives):
        start = rng.randrange(n)
        end = rng.randrange(start + 1, n + 1)
        if (start, end) not in positives:
            negatives.add((start, end))
    # Short operator spans are frequent sources of false positives.
    for _ in range(12):
        start = rng.randrange(n)
        end = min(n, start + rng.randint(1, 4))
        if (start, end) not in positives:
            negatives.add((start, end))
    interiors = quoted_interiors(row["text"])
    negatives.update(tuple(pair) for pair in row.get("hard_negatives", []) if tuple(pair) not in positives)
    # Pure-model training must include fragments inside quoted names. The
    # earlier sampler excluded them because the harness removed them at test.
    # Complete quote-enclosing spans stay ranking-only: their exterior is
    # identical to an unquoted positive and NONE would be contradictory.
    outer = {(s-1,e+1) for s,e in interiors}
    pairs = list(positives) + sorted(pair for pair in negatives if
        (pair not in outer if raw_negatives else eligible_span(*pair, interiors)))
    aliases = row.get('_ce_context_aliases',set())
    targets = [positives.get(pair, -1 if pair in aliases else 0) for pair in pairs]
    if quote_competitors:
        # Quote-enclosing candidates share exterior context with unquoted
        # positives. Rank them below inner literals without contradictory NONE
        # cross-entropy targets. -1 means ranking-only, not a sixth class.
        for s, e in interiors:
            pair = (s-1, e+1)
            if pair not in pairs:
                pairs.append(pair)
                targets.append(-1)
    return pairs, targets


def span_key(spans):
    return {(s["start"], s["end"], s["type"]) for s in spans}


def literal_candidates(row, max_width=12):
    """Training-only NONE intervals in gold literal gaps; no word vocabulary."""
    cursor=0;gaps=[]
    for span in sorted(row['spans'],key=lambda s:s['start']):
        gaps.append((cursor,span['start']));cursor=span['end']
    gaps.append((cursor,len(row['text'])))
    return {(start,end) for left,right in gaps for start in range(left,right)
            for end in range(start+1,min(right,start+max_width)+1)}


def online_candidates(model,banks,rows,rng,count=12,pool_size=512,literal_width=0):
    """Mine fresh errors using this batch's contexts, never validation/test.

    Selection has no gradient; the chosen candidates are scored again with
    gradients alongside all positives. Candidate interiors remain invisible.
    """
    candidates=[];groups=[]
    for index,row in enumerate(rows):
        start=len(candidates);n=len(row['text']);gold={(s['start'],s['end']) for s in row['spans']}
        if not gold:groups.append((start,start));continue
        aliases=row.get('_ce_context_aliases',set());outer={(s-1,e+1) for s,e in quoted_interiors(row['text'])}
        if n*(n+1)//2<=pool_size:pool={(s,e) for s in range(n) for e in range(s+1,n+1)}
        else:
            pool=set()
            for _ in range(pool_size):
                s=rng.randrange(n);pool.add((s,rng.randrange(s+1,n+1)))
            for s,e in gold:
                for ds in range(-4,5):
                    for de in range(-4,5):
                        if 0<=s+ds<e+de<=n:pool.add((s+ds,e+de))
        if literal_width:
            pool.update(literal_candidates(row,literal_width))
        candidates.extend((index,s,e) for s,e in sorted(pool-gold-aliases-outer));groups.append((start,len(candidates)))
    if not candidates:return []
    scores=[]
    with torch.no_grad():
        detached=tuple(b.detach() for b in banks)
        boundary=model.boundary_logits(detached) if model.boundary_mix else None
        for begin in range(0,len(candidates),8192):
            c=torch.tensor(candidates[begin:begin+8192],device=banks[0].device)
            logits=model.score(detached,c,boundary);scores.append((logits[:,1:].amax(-1)-logits[:,0]).cpu())
    scores=torch.cat(scores);selected=[]
    for start,end in groups:
        if end>start:selected.extend(candidates[start+int(i)] for i in scores[start:end].topk(min(count,end-start)).indices)
    return selected


def prepare_context_aliases(rows):
    """Find training negatives indistinguishable from annotated positives.

    The candidate classifier sees exactly (prefix, suffix). Intern only
    positive contexts, then match each row's endpoints once. The resulting
    interval sets stay in training memory; none is shipped for inference.
    """
    prefixes, suffixes, labels = {}, {}, {}
    for row in rows:
        text = row['text']
        for span in row['spans']:
            left,right = text[:span['start']],text[span['end']:]
            p=prefixes.setdefault(left,len(prefixes))
            q=suffixes.setdefault(right,len(suffixes))
            labels.setdefault((p,q),set()).add(span['type'])
    aliases = affected = 0
    for row in rows:
        text=row['text'];n=len(text)
        starts=[(s,prefixes[text[:s]]) for s in range(n) if text[:s] in prefixes]
        ends=[(e,suffixes[text[e:]]) for e in range(1,n+1) if text[e:] in suffixes]
        gold={(s['start'],s['end']) for s in row['spans']}
        ambiguous={(s,e) for s,p in starts for e,q in ends if s<e and (p,q) in labels and (s,e) not in gold}
        row['_ce_context_aliases']=ambiguous
        aliases+=len(ambiguous);affected+=bool(ambiguous)
    return {'training_positive_contexts':len(labels),'training_type_conflict_contexts':sum(len(v)>1 for v in labels.values()),
            'alias_intervals':aliases,'affected_rows':affected,
            'classification':'alias negatives ignored; retained for ranking and span-set likelihood',
            'training_only':True,'inference_lookup':False}


def counterfactual_tokens(tokens, rows, rng, probability=.5):
    """Change only annotated payloads, preserving every Unicode boundary.

    This is supervised training augmentation, never an inference rule or a
    name dictionary. Every labelled type is kept, including PATH and VALUE;
    their type comes from context, not the random replacement's appearance.
    """
    if not 0 <= probability <= 1:
        raise ValueError('counterfactual probability must be in [0,1]')
    # Include common grammatical characters as opaque payload material too.
    # Otherwise a competing boundary after an unseen particle (e.g. 会/能)
    # can score well even though it incorrectly drops the start of a name.
    # This is a character pool for supervised corruption, not a name lexicon.
    pool = 'abXYZ019甲乙资料删除叫做文档路径🚀𠀀éζ/_-会能在给请把我有是新旧原留保内外容末尾名词文字本表记笔录议与和从到为以所将让想要用的地得份个这那上下来去后前日月年开关读写存取建改移复制回收还复空目目录称呼全完整段句行点计划报告清单草稿备忘说明成最终初始工作任务生活课堂训练设备业务财务活动周报开发归档访谈统计采购手会议'
    replacement, _ = encode_texts([pool], tokens.device)
    replacement = replacement[0]
    altered = tokens.clone()
    changed = torch.zeros(len(rows), dtype=torch.bool, device=tokens.device)
    for index, row in enumerate(rows):
        if not row['spans'] or rng.random() >= probability:
            continue
        changed[index] = True
        for span in row['spans']:
            start, end = span['start'], span['end']
            indices = torch.tensor([rng.randrange(len(pool)) for _ in range(end-start)],
                                   dtype=torch.long, device=tokens.device)
            altered[index,start:end] = replacement[indices]
    return altered, changed


def role_contrastive_loss(features, labels, temperature=0.2):
    """Pull positive contexts of one role together; ignore NONE/quote outers.

    Uses the existing classifier hidden representation, with no new weights.
    A role with just one example in the batch contributes no anchor loss.
    """
    keep = labels > 0
    x, y = features[keep], labels[keep]
    if len(y) < 2:
        return features.sum() * 0
    x = nn.functional.normalize(x, dim=-1)
    similarity = x @ x.T / temperature
    diagonal = torch.eye(len(y), dtype=torch.bool, device=y.device)
    same = (y[:, None] == y[None, :]) & ~diagonal
    count = same.sum(-1)
    eligible = count > 0
    if not bool(eligible.any()):
        return features.sum() * 0
    denominator = similarity.masked_fill(diagonal, -torch.inf).logsumexp(-1)
    log_probability = similarity - denominator[:, None]
    return -(log_probability.masked_fill(~same, 0).sum(-1)[eligible] / count[eligible]).mean()


def span_set_loss(logits, candidates, targets, lengths, conditional_count=False):
    """Conditional likelihood of the complete nonoverlapping gold span set.

    All sampled typed intervals compete with a zero-score skip transition.
    This penalizes extra spans jointly, rather than just asking every gold
    span to beat one unrelated hard negative. No parameters or text rules.
    Parameter-free rows are excluded: the separate whole-sentence presence
    query handles their context aliases. Ranking-only candidates compete.
    """
    batch = len(lengths)
    positive = targets > 0
    if not bool(positive.any()):
        return logits.sum() * 0
    width = int(lengths.max())
    row, start, end = candidates.unbind(-1)
    # A type score relative to NONE, summed over the four entity types.
    evidence = logits[:, 1:].logsumexp(-1) - logits[:, 0]
    table = logits.new_full((batch, width + 1, width + 1), -torch.inf)
    table[row, start, end] = evidence
    if conditional_count:
        # Training supervision only; inference uses the learned count head.
        # Finite floors prevent undefined gradients at unreachable DP states.
        counts=torch.zeros(batch,dtype=torch.long,device=logits.device).scatter_add(
            0,row[positive],torch.ones_like(row[positive]))
        maximum=int(counts.max())
        table=table.clamp_min(-1e4)
        initial=logits.new_full((batch,maximum+1),-1e4);initial[:,0]=0.
        alpha=[initial]
        for stop in range(1,width+1):
            prefix=torch.stack(alpha,-1)[:,:-1,:]
            include=(prefix+table[:,:stop,stop][:,None,:]).logsumexp(-1)
            next_nonzero=torch.logaddexp(alpha[-1][:,1:],include)
            alpha.append(torch.cat((logits.new_zeros(batch,1),next_nonzero),-1))
        partition=torch.stack(alpha,-1)[torch.arange(batch,device=logits.device),counts,lengths]
        true_evidence=logits[positive].gather(1,targets[positive,None]).squeeze(-1)-logits[positive,0]
        gold_score=logits.new_zeros(batch).scatter_add(0,row[positive],true_evidence)
        keep=counts>0
        return ((partition[keep]-gold_score[keep])/counts[keep]).mean()
    alpha = [logits.new_zeros(batch)]
    for stop in range(1, width + 1):
        prefix = torch.stack(alpha, -1)
        values = prefix + table[:, :stop, stop]
        # Always include a finite skip, even for endpoints with no candidates.
        alpha.append(torch.cat((alpha[-1][:, None], values), -1).logsumexp(-1))
    partition = torch.stack(alpha, -1)[torch.arange(batch, device=lengths.device), lengths]
    true_evidence = logits[positive].gather(1, targets[positive, None]).squeeze(-1) - logits[positive, 0]
    gold_score = logits.new_zeros(batch).scatter_add(0, row[positive], true_evidence)
    count = logits.new_zeros(batch).scatter_add(0, row[positive], torch.ones_like(true_evidence))
    keep = count > 0
    return ((partition[keep] - gold_score[keep]) / count[keep]).mean()


def endpoint_competition_loss(model, banks, rows):
    """Gold endpoints compete with every endpoint sharing the other boundary.

    Scores still receive only the candidate's exterior. This objective trains
    boundary selection directly instead of diluting it among random negatives.
    Other annotated intervals are excluded from a group's competitors.
    """
    candidates, groups = [], []
    for index, row in enumerate(rows):
        gold = {(s['start'], s['end']) for s in row['spans']}
        n = len(row['text'])
        for span in row['spans']:
            s, e = span['start'], span['end']
            for alternatives in ([(i,e) for i in range(e)], [(s,i) for i in range(s+1,n+1)]):
                pairs = [(s,e)] + [p for p in alternatives if p not in gold]
                begin = len(candidates)
                candidates.extend((index,a,b) for a,b in pairs)
                groups.append((begin,len(candidates),LABEL_IDS[span['type']]))
    if not groups:
        return banks[0].sum()*0
    c = torch.tensor(candidates,device=banks[0].device)
    logits = model.score(banks,c)
    sizes=torch.tensor([end-begin for begin,end,_ in groups],device=c.device)
    ids=torch.repeat_interleave(torch.arange(len(groups),device=c.device),sizes)
    labels=torch.repeat_interleave(torch.tensor([label for _,_,label in groups],device=c.device),sizes)
    evidence=logits.gather(1,labels[:,None]).squeeze(1)-logits[:,0]
    # Vectorized group logsumexp avoids hundreds of tiny GPU kernels/batch.
    maxima=logits.new_full((len(groups),),-torch.inf).scatter_reduce(0,ids,evidence,reduce='amax',include_self=True).detach()
    normalizers=logits.new_zeros(len(groups)).scatter_add(0,ids,(evidence-maxima[ids]).exp())
    gold=torch.tensor([begin for begin,_,_ in groups],device=c.device)
    return (maxima+normalizers.log()-evidence[gold]).mean()


@torch.inference_mode()
def validation_metrics(model, rows, device, batch_size=32, learned_count=False, count_policy='cap'):
    model.eval()
    correct, raw_correct, tp, total_pred, total_gold = 0, 0, 0, 0, 0
    loss_sum, count = 0.0, 0
    raw_groups = {}
    raw_tp = raw_predictions = raw_gold = 0
    count_correct = 0
    for begin in range(0, len(rows), batch_size):
        batch = rows[begin:begin + batch_size]
        scored = model.all_scores([row["text"] for row in batch], device, with_counts=learned_count)
        for row, item in zip(batch, scored):
            pairs, logits = item[:2]
            count_logits = item[2] if learned_count else None
            if learned_count:
                limit = 1 if count_policy == 'zero_only' else model.config.count_classes-1
                count_correct += int(count_logits.argmax()) == min(len(row['spans']), limit)
            probabilities = logits.softmax(dim=-1).numpy()
            raw_pred, raw_true = span_key(select_model_spans(pairs, logits, count_logits, count_policy)), span_key(row['spans'])
            raw_match = raw_pred == raw_true
            raw_tp += len(raw_pred & raw_true)
            raw_predictions += len(raw_pred)
            raw_gold += len(raw_true)
            raw_correct += raw_match
            category=row.get('category','')
            group = 'fresh_natural' if category.startswith('semantic_fresh_natural') else 'context_product' if category.startswith('context_product_development') else 'combination' if category.startswith('semantic_combination') else 'oral' if category.startswith('semantic_oral') else 'semantic' if category.startswith('semantic') else 'clean' if category.startswith('basic') else 'noisy'
            g = raw_groups.setdefault(group, {'rows':0,'correct':0})
            g['rows'] += 1; g['correct'] += int(raw_match)
            probabilities = constrain_probabilities(row["text"], pairs, probabilities)
            prediction = select_spans(pairs, probabilities)
            gold, pred = span_key(row["spans"]), span_key(prediction)
            correct += pred == gold
            tp += len(gold & pred)
            total_gold += len(gold)
            total_pred += len(pred)
            targets = torch.zeros(len(pairs), dtype=torch.long)
            pair_indices = {pair: i for i, pair in enumerate(pairs)}
            for s in row["spans"]:
                targets[pair_indices[(s["start"], s["end"])]] = LABEL_IDS[s["type"]]
            loss_sum += float(nn.functional.cross_entropy(logits, targets, reduction="sum"))
            count += len(pairs)
    precision = tp / max(total_pred, 1)
    recall = tp / max(total_gold, 1)
    raw_precision, raw_recall = raw_tp / max(raw_predictions, 1), raw_tp / max(raw_gold, 1)
    return {"exact": correct / len(rows), "raw_exact": raw_correct / len(rows), "precision": precision, "recall": recall,
            'raw_precision':raw_precision, 'raw_recall':raw_recall,
            'raw_f1':2*raw_precision*raw_recall/max(raw_precision+raw_recall,1e-9),
            'count_accuracy': count_correct/len(rows) if learned_count else None,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "candidate_nll": loss_sum / max(count, 1),
            'raw_group_exact':{k:v['correct']/v['rows'] for k,v in raw_groups.items()}}


def save_checkpoint(model, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": "filter1.0", **model.metadata(),
               "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               **(extra or {})}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def retain_correct_parent_logits(student, parent, targets, temperature=2.):
    """Training-only retention on gold-correct parent predictions.

    Incorrect parent predictions and ignored/ambiguous labels contribute no
    loss. The deployed model contains no parent, lookup or extra parameters.
    """
    teacher = parent.detach()
    keep = (targets >= 0) & (teacher.argmax(-1) == targets)
    if not bool(keep.any()):
        return student.sum()*0
    return nn.functional.kl_div((student[keep]/temperature).log_softmax(-1),
                                (teacher[keep]/temperature).softmax(-1),
                                reduction='batchmean')*temperature**2


def train(data_dir="data", output="artifacts/filter1.0.pt", device="auto", epochs=30,
          batch_size=32, seed=42, threads=4, resume=None, run=1, learning_rate=0.001, posttrain=False,
          adaptive=False, ranking_weight=0.0, scheduler=False, quote_competitors=False, raw_negatives=False,
          balanced_validation=False,positive_loss_weight=0.0,contrastive_weight=0.0,count_weight=0.0,count_policy='cap',
          structured_weight=0.0,length_bucket_size=0,counterfactual_weight=0.0,counterfactual_probability=.5,
          exclude_context_aliases=False,online_hard_negatives=0,online_pool_size=512,epoch_samples=0,boundary_weight=0.,boundary_mix=0.,endpoint_weight=0.,local_window=0,local_mix=0.,skip_initial_mining=False,separate_aux_heads=False,exact_counts=False,conditional_span_set=False,literal_negative_width=0,parent_retention_weight=0.):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available; use --device cpu")
    model = ContextSpanModel().to(device)
    model.boundary_mix=boundary_mix
    if local_window<0 or not 0<=local_mix<=1:raise ValueError('Invalid local context settings')
    model.local_window=local_window;model.local_mix=local_mix
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if checkpoint.get('training',{}).get('learned_count') and not count_weight:
            raise ValueError('Resuming a count-query checkpoint requires --count-weight to preserve its learned decoding task')
        model=ContextSpanModel(ModelConfig(**checkpoint['config'])).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.boundary_mix=boundary_mix;model.local_window=local_window;model.local_mix=local_mix
    if separate_aux_heads:model.enable_auxiliary_heads()
    if exact_counts:model.enable_exact_counts()
    parent_model = None
    parent_has_count = False
    if parent_retention_weight < 0:
        raise ValueError('Parent retention weight cannot be negative')
    if parent_retention_weight:
        if not resume:
            raise ValueError('Parent retention requires an existing checkpoint')
        from .harness import FilterHarness
        parent_harness = FilterHarness(resume, device)
        parent_model = parent_harness.model
        parent_has_count = parent_harness.learned_count
        for parameter in parent_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    weights = torch.tensor([0.25, 1, 1, 1, 1], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=2, min_lr=0.000005) if scheduler else None
    train_rows = read_jsonl(Path(data_dir) / "train.jsonl")
    extra_path = Path(data_dir) / "train_extra.jsonl"
    if run > 1 and extra_path.exists():
        train_rows += read_jsonl(extra_path)
    validation = read_jsonl(Path(data_dir) / "validation.jsonl")
    rng = random.Random(seed)
    alias_info = prepare_context_aliases(train_rows) if exclude_context_aliases else None
    if alias_info:
        print(json.dumps({'event':'context_alias_preparation',**alias_info}),flush=True)
    if posttrain and not skip_initial_mining:
        if not resume:
            raise ValueError("Post-training requires an existing checkpoint")
        model.eval()
        mining_rows = [row for row in train_rows if row.get("source") in ('posttrain_structural_augmentation', 'adaptive_augmentation', 'noisy_augmentation', 'semantic_augmentation', 'joint_supervision')]
        if len(mining_rows)>24000 and all(row.get('source')=='joint_supervision' for row in mining_rows):
            mining_rows=rng.sample(mining_rows,24000)
        with torch.inference_mode():
            for begin in range(0, len(mining_rows), batch_size):
                if begin % (batch_size*16)==0:
                    print(json.dumps({'event':'mining_progress','seen':begin,'total':len(mining_rows)}),flush=True)
                batch = mining_rows[begin:begin + batch_size]
                scores = model.all_scores([row["text"] for row in batch], device)
                for row, (pairs, logits) in zip(batch, scores):
                    probabilities = logits.softmax(-1).numpy()
                    if not raw_negatives:
                        probabilities = constrain_probabilities(row["text"], pairs, probabilities)
                    positives = {(s["start"], s["end"]) for s in row["spans"]}
                    confidence = probabilities[:, 1:].max(axis=1)
                    eligible_indices = np.flatnonzero(confidence > .05)
                    # Mine the deployed decoder's extra spans first. Sorting
                    # softmax confidence alone saturates at 1.0 and can fill
                    # all eight slots with quote/boundary competitors, missing
                    # a separate hallucinated parameter elsewhere in the sentence.
                    evidence = (logits[:, 1:].amax(-1) - logits[:, 0]).numpy()
                    indices = eligible_indices[np.argsort(evidence[eligible_indices])[::-1]]
                    wrong_selected = [(s['start'], s['end']) for s in select_spans(pairs, probabilities)
                                      if (s['start'], s['end']) not in positives]
                    hard = list(dict.fromkeys(wrong_selected + [pairs[int(i)] for i in indices
                                             if pairs[int(i)] not in positives]))[:12]
                    row["hard_negatives"] = [list(p) for p in hard]
                if begin % (batch_size * 160) == 0:
                    print(json.dumps({'event':'mining_progress','rows':min(begin+batch_size,len(mining_rows)),
                                      'total':len(mining_rows)}),flush=True)
        print(json.dumps({"event": "hard_negative_mining", "training_rows_only": len(mining_rows),
                          "negatives": sum(len(r.get("hard_negatives", [])) for r in mining_rows)}), flush=True)
    best, stale, history = -1, 0, []
    started = time.time()
    print(json.dumps({"event": "start", "device": device, "parameters": model.metadata()["parameters"],
                      "train_rows": len(train_rows), "run": run}), flush=True)
    if resume:
        baseline=validation_metrics(model,validation,device,learned_count=bool(count_weight),count_policy=count_policy)
        tie=baseline['raw_f1' if raw_negatives else 'f1']
        best=(min(baseline['raw_group_exact'].values()) if balanced_validation else baseline['raw_exact' if adaptive else 'exact'])+.001*tie
        print(json.dumps({'event':'resume_baseline','validation':baseline}),flush=True)
        inherited=dict(checkpoint.get('training',{}))
        inherited.update(resume_baseline_on_current_validation=baseline,baseline_run=run,
                         training_updates_in_this_run=0,inherited_training_metadata_describes_parent=True,
                         boundary_mix=boundary_mix,local_window=local_window,local_mix=local_mix,
                         neural_only=raw_negatives,learned_count=bool(count_weight),count_policy=count_policy,
                         count_classes=model.config.count_classes)
        save_checkpoint(model,output,{'training':inherited})
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        rng.shuffle(train_rows)
        epoch_rows=train_rows[:epoch_samples] if epoch_samples else train_rows
        if length_bucket_size:
            # Sorting only within newly shuffled blocks avoids padding every
            # batch to the longest payload while retaining epoch randomness.
            for begin in range(0,len(epoch_rows),length_bucket_size):
                epoch_rows[begin:begin+length_bucket_size]=sorted(
                    epoch_rows[begin:begin+length_bucket_size],key=lambda r:len(r['text']))
        losses = []
        for begin in range(0, len(epoch_rows), batch_size):
            batch = epoch_rows[begin:begin + batch_size]
            tokens, lengths = encode_texts([row["text"] for row in batch], device)
            candidates, targets, groups = [], [], []
            for index, row in enumerate(batch):
                pairs, labels = sample_candidates(row, rng, quote_competitors=quote_competitors, raw_negatives=raw_negatives)
                if count_weight and not row['spans']:
                    # Interior-blind candidates can hide a negation or a
                    # whole concept statement, becoming indistinguishable
                    # from a genuine opaque value. Teach these no-argument
                    # sentences through the full-sentence count query instead.
                    labels = [-1] * len(labels)
                groups.append((len(targets), len(targets) + len(labels)))
                candidates.extend((index, s, e) for s, e in pairs)
                targets.extend(labels)
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=device)
            target_tensor = torch.tensor(targets, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            banks = model.contexts(tokens, lengths)
            if online_hard_negatives:
                existing=set(candidates)
                mined=[c for c in online_candidates(model,banks,batch,rng,online_hard_negatives,online_pool_size,literal_negative_width) if c not in existing]
                if mined:
                    candidate_tensor=torch.cat((candidate_tensor,torch.tensor(mined,dtype=torch.long,device=device)))
                    target_tensor=torch.cat((target_tensor,torch.zeros(len(mined),dtype=torch.long,device=device)))
            features = model.candidate_features(banks, candidate_tensor)
            logits = model.blend_boundaries(model.classifier(features),banks,candidate_tensor)
            loss = criterion(logits, target_tensor) if bool((target_tensor >= 0).any()) else logits.sum()*0
            if parent_model is not None:
                with torch.no_grad():
                    parent_banks = parent_model.contexts(tokens, lengths)
                    parent_logits = parent_model.score(parent_banks, candidate_tensor)
                loss += parent_retention_weight*retain_correct_parent_logits(logits, parent_logits, target_tensor)
            if endpoint_weight:
                loss += endpoint_weight*endpoint_competition_loss(model,banks,batch)
            if boundary_weight:
                start_logits,end_logits=model.boundary_logits(banks)
                start_targets=torch.zeros(start_logits.shape[:2],dtype=torch.long,device=device);end_targets=start_targets.clone()
                for index,row in enumerate(batch):
                    for span in row['spans']:
                        start_targets[index,span['start']]=1;end_targets[index,span['end']]=1
                valid=torch.arange(start_targets.shape[1],device=device)[None,:]<=lengths[:,None]
                weights=logits.new_tensor([.1,1.,.1,.1,.1])
                boundary_loss=nn.functional.cross_entropy(start_logits[valid],start_targets[valid],weight=weights)+nn.functional.cross_entropy(end_logits[valid],end_targets[valid],weight=weights)
                loss+=boundary_weight*boundary_loss
            if count_weight:
                limit = 1 if count_policy == 'zero_only' else model.config.count_classes-1
                counts = torch.tensor([min(len(row['spans']),limit) for row in batch],device=device)
                current_counts = model.count_logits(banks, lengths)
                loss = loss + count_weight * nn.functional.cross_entropy(current_counts, counts)
                if parent_model is not None and parent_has_count:
                    with torch.no_grad():
                        parent_counts = parent_model.count_logits(parent_banks, lengths)
                    if current_counts.shape == parent_counts.shape:
                        loss += parent_retention_weight*retain_correct_parent_logits(current_counts, parent_counts, counts)
            if contrastive_weight:
                hidden = model.classifier[1](model.classifier[0](features))
                loss = loss + contrastive_weight * role_contrastive_loss(hidden, target_tensor)
            if structured_weight:
                loss = loss + structured_weight * span_set_loss(logits, candidate_tensor, target_tensor, lengths,conditional_span_set)
            positive = target_tensor > 0
            if positive_loss_weight and bool(positive.any()):
                # Absolute positive/type supervision counters the pressure to
                # suppress unquoted literals sharing context with quote outers.
                loss = loss + positive_loss_weight * nn.functional.cross_entropy(logits[positive],target_tensor[positive])
            if ranking_weight:
                # Train correct spans to outrank the hardest competing
                # fragments. This changes the objective, not the architecture.
                entity_evidence = logits[:, 1:].amax(dim=-1) - logits[:, 0]
                positive = target_tensor > 0
                row_ids = candidate_tensor[:, 0]
                negative_evidence = entity_evidence.masked_fill(positive, -torch.inf)
                hardest = logits.new_full((len(batch),), -torch.inf).scatter_reduce(
                    0, row_ids, negative_evidence, reduce='amax', include_self=True)
                if bool(positive.any()):
                    true_evidence = logits[positive].gather(1, target_tensor[positive, None]).squeeze(1) - logits[positive, 0]
                    loss = loss + ranking_weight * nn.functional.softplus(hardest[row_ids[positive]] - true_evidence + .5).mean()
            if counterfactual_weight:
                altered, changed = counterfactual_tokens(tokens,batch,rng,counterfactual_probability)
                if bool(changed.any()):
                    keep = changed[candidate_tensor[:,0]]
                    remap = changed.long().cumsum(0)-1
                    alternate_candidates = candidate_tensor[keep].clone()
                    alternate_candidates[:,0] = remap[alternate_candidates[:,0]]
                    alternate_targets = target_tensor[keep]
                    alternate_lengths = lengths[changed]
                    alternate_banks = model.contexts(altered[changed],alternate_lengths)
                    alternate_logits = model.score(alternate_banks,alternate_candidates)
                    alternate_loss = criterion(alternate_logits,alternate_targets)
                    positives = alternate_targets > 0
                    if positive_loss_weight:
                        alternate_loss += positive_loss_weight * nn.functional.cross_entropy(
                            alternate_logits[positives],alternate_targets[positives])
                    if structured_weight:
                        alternate_loss += structured_weight * span_set_loss(
                            alternate_logits,alternate_candidates,alternate_targets,alternate_lengths,conditional_span_set)
                    if count_weight:
                        # Corruption never invents/removes an argument. The
                        # whole-sentence query must keep its original label.
                        limit = 1 if count_policy == 'zero_only' else model.config.count_classes-1
                        count_targets = torch.tensor([min(len(row['spans']),limit) for row,selected
                                                      in zip(batch,changed.tolist()) if selected],device=device)
                        alternate_loss += count_weight * nn.functional.cross_entropy(
                            model.count_logits(alternate_banks,alternate_lengths),count_targets)
                    loss += counterfactual_weight * alternate_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if begin%(batch_size*160)==0:
                print(json.dumps({'event':'batch','epoch':epoch,'seen':begin,'total':len(epoch_rows),'loss':round(losses[-1],5),'seconds':round(time.time()-epoch_start,1)}),flush=True)
        metrics = validation_metrics(model, validation, device, learned_count=bool(count_weight),count_policy=count_policy)
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": metrics,
                 "seconds": round(time.time() - epoch_start, 2)}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        # Exact sentence accuracy is the deployment objective; use F1 only to
        # break otherwise identical exact scores.
        tie_f1 = metrics['raw_f1' if raw_negatives else 'f1']
        selection_score = metrics['raw_exact' if adaptive else 'exact'] + 0.001 * tie_f1
        if balanced_validation:
            selection_score = min(metrics['raw_group_exact'].values()) + .001*tie_f1
        if lr_scheduler:
            lr_scheduler.step(selection_score)
        if selection_score > best + 1e-6:
            best, stale = selection_score, 0
            save_checkpoint(model, output, {"training": {"seed": seed, "epoch": epoch, "run": run,
                            "rows": len(train_rows), "device": device, "validation": metrics,
                            'data_dir':str(data_dir),
                            'data_sha256':{s:hashlib.sha256((Path(data_dir)/(s+'.jsonl')).read_bytes()).hexdigest()
                                           for s in ('train','validation','calibration')},
                              'train_extra_sha256':hashlib.sha256(extra_path.read_bytes()).hexdigest() if run>1 and extra_path.exists() else None,
                            "learning_rate": learning_rate, "batch_size": batch_size,
                            "posttrained": posttrain,
                            "adaptive": adaptive, "ranking_weight": ranking_weight,
                            'scheduler': scheduler, 'quote_competitors': quote_competitors,
                            'raw_negatives': raw_negatives, 'neural_only': raw_negatives,
                            'balanced_validation':balanced_validation,
                            'positive_loss_weight':positive_loss_weight,
                            'contrastive_weight':contrastive_weight,
                            'structured_weight':structured_weight,
                            'structured_objective':('sampled_gold_count_conditional_span_set_likelihood' if conditional_span_set else 'sampled_nonoverlapping_span_set_likelihood') if structured_weight else None,
                            'conditional_span_set':conditional_span_set,
                            'length_bucket_size':length_bucket_size,
                            'counterfactual_weight':counterfactual_weight,
                            'counterfactual_probability':counterfactual_probability,
                            'counterfactual_policy':'randomize_annotated_payloads_preserve_unicode_offsets_and_types' if counterfactual_weight else None,
                            'exclude_context_aliases':exclude_context_aliases,
                            'online_hard_negatives':online_hard_negatives,
                            'online_pool_size':online_pool_size,
                            'literal_negative_width':literal_negative_width,
                              'parent_retention_weight':parent_retention_weight,
                              'parent_retention_policy':'KL to frozen parent only on gold-correct training candidates/counts; no deployed teacher' if parent_retention_weight else None,
                            'epoch_samples':epoch_samples,'epoch_sample_policy':'reshuffle entire training population, then sample without replacement' if epoch_samples else 'all rows',
                            'boundary_weight':boundary_weight,'boundary_mix':boundary_mix,
                            'endpoint_weight':endpoint_weight,
                            'local_window':local_window,'local_mix':local_mix,
                            'separate_aux_heads':bool(model.config.auxiliary_heads),
                            'count_classes':model.config.count_classes,
                            'local_context_policy':'shared GRUs per bounded exterior; full banks for count' if local_window else None,
                            'skip_initial_mining':skip_initial_mining,
                            'boundary_query':('separate_readouts_prefix_shift4_suffix_shift6' if model.config.auxiliary_heads else 'shared_classifier_prefix_shift4_suffix_shift6') if boundary_weight else None,
                            'context_alias_preparation':alias_info,
                            'count_weight':count_weight, 'learned_count':bool(count_weight),
                            'count_policy':count_policy,
                            'sentence_query_task':'parameter_presence' if count_policy == 'zero_only' else 'parameter_count',
                            'count_query':'shared_classifier_full_context_shift_2' if count_weight else None,
                            'no_argument_candidate_ce':'ignored_context_aliases' if count_weight else 'NONE',
                            'mining_policy':'decoder_false_positives_then_logit_margin_top12',
                            'selection_tie_break':'raw_span_f1' if raw_negatives else 'quote_constrained_span_f1',
                            "parent_sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else "",
                            "validation_policy": "learned_count_exact_p0.7_otherwise_raw_threshold0.5" if count_policy=='exact' else "min_raw_group_exact_threshold_0.5" if balanced_validation else "raw_threshold_0.5" if adaptive else "quote_constraints_then_threshold_0.5"}})
        else:
            stale += 1
        # An explicit local stop marker lets an external evaluator stop after
        # the user's goal is met while preserving the best checkpoint/log.
        if stale >= 5 or Path(output).with_suffix('.stop').exists():
            break
    info = {"seed": seed, "run": run, "device": device, "elapsed_seconds": round(time.time() - started, 2),
            "history": history, "parameters": model.metadata()["parameters"]}
    log_path = Path(output).parent / f"training_run{run}.json"
    log_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info

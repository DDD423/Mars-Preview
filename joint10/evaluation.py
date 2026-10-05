"""Actual filter, constrained translator and workspace preview evaluation."""
import copy
import hashlib
import json
import tempfile
import time
import io
from pathlib import Path
import torch
from filter10 import FilterHarness
from filter10.harness import render_result
from davework.adapter import planner_input
from davework.runtime import Runtime
from davework.paths import Workspace
from davework.files import record_paths
from translator10.data import load_rows
from translator10.training import load_checkpoint,predict_rows,evaluate_predictions
from translator10.language import canonical,FILE_TOOLS
from translator10.inference import confidence,joint_features

def extract_rows(rows,checkpoint,device):
    f=FilterHarness(checkpoint,device);output=[];good=[]
    for begin in range(0,len(rows),32):
        batch=rows[begin:begin+32]
        scores=f.model.all_scores([r['raw'] for r in batch],device,with_counts=f.learned_count,with_evidence=f.neural_only)
        for r,item in zip(batch,scores):
            pairs,logits=item[:2];count=item[2] if f.learned_count else None
            evidence=item[3 if f.learned_count else 2] if f.neural_only else None
            status,spans,proposals=f.decode_scores(r['raw'],pairs,logits,count_logits=count,neural_evidence=evidence)
            out=render_result(r['raw'],status,spans,proposals,refine_paths=not f.neural_only).to_dict()
            out['checkpoint_sha256']=f.checkpoint_sha256
            assert out['original']==r['raw']
            output.append(out)
            good.append([(s['start'],s['end'],s['text']) for s in r['spans']]==[(s['start'],s['end'],r['raw'][s['start']:s['end']]) for s in out['spans']])
        if begin%320==0:print(json.dumps({'event':'joint_filter','seen':begin,'total':len(rows)}),flush=True)
    return output,good

def requests(rows,extractions):
    return [dict(r,request=planner_input({'id':'ctx_eval','filter':f,'settings':{'enabled_tools':r['request']['enabled_tools']}})) for r,f in zip(rows,extractions)]

def setup_fixture(row,root):
    """Seed only sources, existing directories and recovery records in a lab."""
    ws=Workspace(root,[]);body='原文 😀 {1}\n第二行'
    def path(v):
        if isinstance(v,str):return ws.path(v)
        if isinstance(v,dict) and 'slot' in v:return ws.path(row['bindings']['{'+str(v['slot'])+'}']['value'])
        return None
    def seed(v,directory=False):
        p=path(v)
        if p is None:return
        if isinstance(v,dict) and 'slot' in v:
            binding=row['bindings']['{'+str(v['slot'])+'}']
            if binding['type']=='NAME':
                # NAME sources live below a directory: predicting PATH must
                # not pass by coincidence just because the fixture is flat.
                p=ws.path(str(root/'已有资料'/binding['value']))
        p.parent.mkdir(parents=True,exist_ok=True)
        if directory:p.mkdir(exist_ok=True)
        elif not p.exists():p.write_bytes(body.encode('utf-8'))
    for s in row['target'].get('steps',[]):
        t=s['tool'];a=s['args']
        if t in ('fs.copy','fs.move','fs.rename'):seed(a['source'])
        elif t in ('fs.read_text','fs.append_text','fs.trash'):seed(a['path'])
        elif t=='fs.list':seed(a['path'],True)
        elif t=='fs.write_text' and a.get('overwrite'):seed(a['path'])
        elif t=='fs.restore' and 'slot' in a['record_id']:
            rid=row['bindings']['{'+str(a['record_id']['slot'])+'}']['value'];folder,meta,payload=record_paths(ws,rid)
            folder.mkdir(exist_ok=True);payload.write_bytes(body.encode('utf-8'));meta.write_text(json.dumps({'record_id':rid,'original':str(root/'restored.txt'),'kind':'trash'}),encoding='utf-8')
        for v in a.values():
            if isinstance(v,dict) and 'join' in v:seed(v['join']['directory'],True)
        for key in ('path','destination'):
            if key in a and t in ('fs.copy','fs.move','fs.write_text','fs.mkdir'):
                p=path(a[key])
                if p is not None:p.parent.mkdir(parents=True,exist_ok=True)
    return body

def preview_observation(row,extraction,prediction):
    """Observe real preview eligibility independently of gold correctness.

    Unbindable plans cannot be executed by Dave Work. They remain whole-request
    failures and are separately reported as rejected proposals, rather than
    counted as automatically accepted executable plans.
    """
    parent=Path('.davework-test-joint-preview').resolve();parent.mkdir(exist_ok=True)
    # TemporaryDirectory recursively cleans up only its verified child path.
    with tempfile.TemporaryDirectory(prefix='case-',dir=parent) as name:
        root=Path(name).resolve();assert root.parent==parent and root!=parent
        rt=None;fixture_ready=False;preview_valid=False
        try:
            setup_fixture(row,root);fixture_ready=True;rt=Runtime(root/'.state')
            rt.config.update(workspace=str(root),excludes=['.state'])
            gold=render_result(row['raw'],'ok',[dict(s,confidence=1.) for s in row['spans']],refine_paths=False).to_dict()
            ctx=rt.context('',extract=False);cid=ctx['context_id'];rt.contexts[cid]['filter']=extraction
            actual=copy.deepcopy(prediction);actual['context_id']=cid;actual=rt.preview(actual)
            preview_valid=actual['kind']=='plan'
            if row['target']['kind']!='plan':return False,preview_valid,'Execution proposed for a non-execution request'
            rt.contexts[cid]['filter']=gold
            target=copy.deepcopy(row['target']);target['context_id']=cid;expected=rt.preview(target)
            matched=expected['steps']==actual['steps']
            return matched,preview_valid,None if matched else 'Bound preview arguments differ from gold'
        except Exception as exc:return False,preview_valid,('Fixture preparation failed: ' if not fixture_ready else '')+str(exc)
        finally:
            if rt is not None:rt.close()


def preview_check(row,extraction,prediction):
    matched,_,error=preview_observation(row,extraction,prediction)
    return matched,error

def evaluate(root,filter_checkpoint,translator_checkpoint,device='cuda',split='validation',preview=True,final=False,oracle=True):
    torch.set_num_threads(4);root=Path(root);rows=load_rows(root/(split+'.jsonl'))
    # Snapshot bytes once: development evaluation may run on CPU while a GPU
    # trainer atomically replaces best.pt. Report the weights actually loaded.
    filter_bytes=Path(filter_checkpoint).read_bytes();translator_bytes=Path(translator_checkpoint).read_bytes()
    model,ck=load_checkpoint(io.BytesIO(translator_bytes),device)
    expected_filter=(ck.get('calibration') or {}).get('filter_sha256')
    if expected_filter and expected_filter != hashlib.sha256(filter_bytes).hexdigest():
        raise ValueError('Translator calibration belongs to another filter checkpoint')
    f,bindings_good=extract_rows(rows,io.BytesIO(filter_bytes),device);real=requests(rows,f)
    preds=predict_rows(model,real,device,progress=lambda n,total:print(json.dumps({'event':'joint_translator','seen':n,'total':total}),flush=True))
    for prediction,request in zip(preds,real):prediction['features']=joint_features(prediction,request['request'])
    puremetrics={'exact_accuracy':None}
    if oracle:
        pure=predict_rows(model,rows,device,progress=lambda n,total:print(json.dumps({'event':'oracle_translator','seen':n,'total':total}),flush=True))
        puremetrics,_=evaluate_predictions(rows,pure)
    _,plan_good=evaluate_predictions(real,preds)
    joint=[a and b for a,b in zip(bindings_good,plan_good)];preview_good=[];preview_errors=[];preview_valid=[]
    for i,(r,x,p,ok) in enumerate(zip(rows,f,preds,joint)):
        executable=p['plan'] is not None and p['plan']['kind']=='plan'
        matched,eligible,error=(preview_observation(r,x,p['plan']) if executable and preview else (ok,executable,None))
        preview_good.append(ok and matched);preview_errors.append(error);preview_valid.append(eligible)
        real[i]['request']['preview_valid']=eligible
        if i%320==0:print(json.dumps({'event':'joint_preview','seen':i,'total':len(rows)}),flush=True)
    cal=ck.get('calibration');threshold=cal.get('threshold',1.000001) if cal else 1.000001
    blocked=('error','unsupported','unsupported_length','uncertain','needs_context')
    usable=[r['request'].get('filter_status') not in blocked for r in real]
    proposed=[u and p['plan'] is not None and p['plan']['kind']=='plan' and confidence(p['features'],cal)>=threshold for u,p in zip(usable,preds)]
    accepted=[a and valid for a,valid in zip(proposed,preview_valid)]
    deployed=[u and g and (r['target']['kind']!='plan' or a) for u,r,g,a in zip(usable,rows,preview_good,accepted)]
    supported=sum(r['target']['kind']=='plan' for r in rows);count=sum(accepted);correct=sum(g and a for g,a in zip(preview_good,accepted))
    categories={}
    for r,g in zip(rows,deployed):
        c=categories.setdefault(r['category'],{'total':0,'correct':0});c['total']+=1;c['correct']+=int(g)
    metrics={'rows':len(rows),'filter_boundaries_exact':sum(bindings_good)/len(rows),'translator_oracle':puremetrics['exact_accuracy'],
             'joint_ungated_exact':sum(preview_good)/len(rows),'joint_gated_exact':sum(deployed)/len(rows),'preview_checked':preview,
             'fixture_preparation_failures':sum(bool(e) and e.startswith('Fixture preparation failed: ') for e in preview_errors),
             'accepted':count,'accepted_correct':correct,'accepted_precision':correct/count if count else None,
             'score_accepted_proposals':sum(proposed),
             'score_accepted_proposal_precision':sum(g and a for g,a in zip(preview_good,proposed))/sum(proposed) if any(proposed) else None,
             'preview_rejected_proposals':sum(a and not v for a,v in zip(proposed,preview_valid)),
             'acceptance_policy':'confidence threshold AND successful actual harness preview; all requests remain in accuracy denominator',
             'supported_coverage':sum(a and r['target']['kind']=='plan' for a,r in zip(accepted,rows))/supported,
             'wrong_accepted_modifying':sum(a and not g and any(s['tool'] not in ('fs.list','fs.read_text','fs.search') for s in p['plan']['steps']) for a,g,p in zip(accepted,preview_good,preds)),
             'categories':categories,'filter_sha256':hashlib.sha256(filter_bytes).hexdigest(),'translator_sha256':hashlib.sha256(translator_bytes).hexdigest(),
             'test_sha256':hashlib.sha256((root/(split+'.jsonl')).read_bytes()).hexdigest(),
             'negative_wrong_accepted_modifications':sum(a and r['target']['kind']!='plan' and any(s['tool'] not in ('fs.list','fs.read_text','fs.search') for s in p['plan']['steps']) for a,r,p in zip(accepted,rows,preds))}
    metrics['calibration_precision_target']=cal.get('precision_target',.98) if cal else None
    if cal and 'strict_98_threshold' in cal:
        strict=dict(cal,threshold=cal['strict_98_threshold'])
        strict_accepted=[u and v and p['plan'] is not None and p['plan']['kind']=='plan' and confidence(p['features'],strict)>=strict['threshold'] for u,v,p in zip(usable,preview_valid,preds)]
        strict_deployed=[u and g and (r['target']['kind']!='plan' or a) for u,r,g,a in zip(usable,rows,preview_good,strict_accepted)]
        strict_count=sum(strict_accepted);strict_correct=sum(a and g for a,g in zip(strict_accepted,preview_good))
        metrics['strict_98_comparison']={'joint_gated_exact':sum(strict_deployed)/len(rows),'accepted':strict_count,'accepted_correct':strict_correct,
            'accepted_precision':strict_correct/strict_count if strict_count else None,'threshold':strict['threshold']}
    errors=[{'raw':r['raw'],'expected':r['target'],'filter':x,'actual':p['plan'],'binding_good':b,'plan_good':g,'preview_error':pe,'accepted':a} for r,x,p,b,g,pe,a,d in zip(rows,f,preds,bindings_good,plan_good,preview_errors,accepted,deployed) if not d]
    return metrics,errors,(rows,real,preds,preview_good)

def real_training(root,filter_checkpoint,out,device='cuda',limit=24000):
    from translator10.language import masks
    from translator10.data import save_rows
    import random
    rows=load_rows(Path(root)/'train.jsonl');random.Random(10042033).shuffle(rows);rows=rows[:limit]
    f,good=extract_rows(rows,filter_checkpoint,device);real=requests(rows,f);output=[]
    for original,r,g in zip(rows,real,good):
        if not g:continue
        actual={s['slot']:s['type'] for s in r['request']['slots']}
        # Correct boundaries with conservative VALUE/text-role uncertainty can
        # teach the planner to rely on request semantics. Never train a plan
        # whose source would bind a nested name as a root-relative path, or
        # whose NAME lookup would interpret a real slash path as a basename.
        invalid=False
        for key,binding in original['bindings'].items():
            predicted=actual.get(int(key[1:-1]));kind=binding['type']
            if kind in ('NAME','PATH') and (predicted=='TEXT' or kind=='NAME' and predicted=='PATH' or kind=='PATH' and predicted=='NAME'):
                invalid=True;break
        if invalid:continue
        try:masks(r['request'],r['tokens'])
        except (ValueError,KeyError):continue
        output.append(r)
    info=save_rows(Path(out),output);info['source']='training only, exact raw boundaries, binding-safe NAME/PATH/VALUE or text-role uncertainty, grammar compatible';info['attempted']=len(rows);info['sample_seed']=10042033
    Path(out).with_suffix('.manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8');return info


def calibrate(root,filter_checkpoint,translator_checkpoint,device='cuda',feature_map='linear',precision_target=.98):
    """Fit only the real-flow calibration split, independently of final tests."""
    from translator10.evaluation import fit_calibration,acceptance,select_threshold
    if not 0 < precision_target <= 1:raise ValueError('Invalid precision target')
    metrics,_,(rows,real,preds,good)=evaluate(root,filter_checkpoint,translator_checkpoint,device,'calibration',preview=True,oracle=False)
    if hashlib.sha256(Path(translator_checkpoint).read_bytes()).hexdigest()!=metrics['translator_sha256'] or hashlib.sha256(Path(filter_checkpoint).read_bytes()).hexdigest()!=metrics['filter_sha256']:
        raise RuntimeError('Weights changed during calibration; stop training and calibrate immutable candidates')
    fit=list(range(0,len(rows),2));hold=list(range(1,len(rows),2))
    cal=fit_calibration([preds[i]['features'] for i in fit],[good[i] for i in fit],feature_map=feature_map,regularization=.01 if feature_map=='quadratic' else .002)
    cal.update(threshold=1.000001,method='real joint flow; confidence AND successful actual preview; even fit / odd threshold; empirical precision target',precision_target=precision_target,fit_rows=len(fit),threshold_rows=len(hold),acceptance_requires_actual_preview=True)
    cal['filter_sha256']=metrics['filter_sha256'];cal['data_sha256']=metrics['test_sha256']
    hr=[real[i] for i in hold];hp=[preds[i] for i in hold];hg=[good[i] for i in hold]
    strict,strict_metrics=select_threshold(hr,hp,hg,cal,.98)
    cal,m=select_threshold(hr,hp,hg,cal,precision_target);cal['threshold_metrics']=m
    cal['strict_98_threshold']=strict['threshold'];cal['strict_98_metrics']=strict_metrics
    cal['precision_coverage_curve']={str(p):select_threshold(hr,hp,hg,cal,p)[1] for p in (.90,.95,.97,.98,.99)}
    observations={'filter_sha256':metrics['filter_sha256'],'translator_source_sha256':metrics['translator_sha256'],
                  'data_sha256':metrics['test_sha256'],'split_policy':'even fit / odd threshold','rows':[]}
    for i,(r,p,ok) in enumerate(zip(real,preds,good)):
        plan=p['plan']
        observations['rows'].append({'index':i,'features':p['features'],'correct':bool(ok),
            'expected_kind':r['target']['kind'],'predicted_kind':plan['kind'] if plan else None,
            'modifying':bool(plan and plan['kind']=='plan' and any(s['tool'] not in ('fs.list','fs.read_text','fs.search') for s in plan['steps'])),
            'filter_status':r['request'].get('filter_status'),'preview_valid':r['request'].get('preview_valid',False)})
    Path(translator_checkpoint).with_suffix('.calibration-observations.json').write_text(json.dumps(observations,indent=2),encoding='utf-8')
    _,ck=load_checkpoint(translator_checkpoint,'cpu');ck['calibration']=cal
    destination=Path(translator_checkpoint);temporary=destination.with_suffix('.calibrating.pt')
    torch.save(ck,temporary);temporary.replace(destination)
    destination.with_suffix('.calibration.json').write_text(json.dumps(cal,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'calibration':cal,'ungated_calibration_metrics':metrics}

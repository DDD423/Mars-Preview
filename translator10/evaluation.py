import collections
import hashlib
import json
import time
import itertools
from pathlib import Path
import torch
from .data import load_rows
from .inference import confidence
from .language import canonical, nonexecution
from .training import load_checkpoint, predict_rows, evaluate_predictions

def fit_calibration(features,labels,feature_map='linear',regularization=.002):
    x=torch.tensor(features,dtype=torch.float64);y=torch.tensor(labels,dtype=torch.float64)
    mean=x.mean(0);scale=x.std(0).clamp_min(.01);x=(x-mean)/scale
    if feature_map=='quadratic':
        x=x.clamp(-4.,4.)
        x=torch.cat([x]+[(x[:,i]*x[:,j])[:,None] for i in range(x.shape[1]) for j in range(i,x.shape[1])],dim=1)
    elif feature_map!='linear':raise ValueError('Unknown calibration feature map')
    weights=torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True);bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([weights,bias],max_iter=120,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(x@weights+bias,y)+regularization*weights.square().sum();loss.backward();return loss
    opt.step(closure)
    return {"weights":weights.detach().tolist(),"bias":bias.item(),"mean":mean.tolist(),"scale":scale.tolist(),'feature_map':feature_map,'regularization':regularization}

def acceptance(rows,preds,good,cal):
    blocked=("error","unsupported","unsupported_length","uncertain","needs_context")
    accepted=[p["plan"] is not None and p["plan"]["kind"]=="plan" and confidence(p["features"],cal)>=cal["threshold"] and r["request"].get("filter_status") not in blocked and r['request'].get('preview_valid',True) for r,p in zip(rows,preds)]
    count=sum(accepted);correct=sum(a and ok for a,ok in zip(accepted,good));supported=sum(r["target"]["kind"]=="plan" for r in rows)
    return {"accepted":count,"accepted_correct":correct,"semantic_precision":correct/count if count else None,"supported":supported,"supported_coverage":sum(a and r["target"]["kind"]=="plan" for a,r in zip(accepted,rows))/supported if supported else None,"wrong_modifying_actions":sum(a and not ok and any(s["tool"] not in ("fs.list","fs.read_text","fs.search") for s in p["plan"]["steps"]) for a,ok,p in zip(accepted,good,preds))},accepted


def select_threshold(rows,preds,good,cal,precision=.98):
    """Exact threshold sweep, scoring each row once and keeping tied scores.

    Same acceptance criterion as the brute-force sweep; this avoids scoring
    1,000 calibration rows again for each of their 1,000 possible thresholds.
    """
    if not 0 < precision <= 1:raise ValueError('Invalid precision target')
    blocked=("error","unsupported","unsupported_length","uncertain","needs_context")
    records=[];supported=sum(r['target']['kind']=='plan' for r in rows)
    for r,p,ok in zip(rows,preds,good):
        if p['plan'] is None or p['plan']['kind']!='plan' or r['request'].get('filter_status') in blocked or not r['request'].get('preview_valid',True):continue
        records.append((confidence(p['features'],cal),bool(ok),r['target']['kind']=='plan'))
    records.sort(reverse=True);count=correct=covered=0;best=-1.;threshold=1.000001
    for score,group in itertools.groupby(records,key=lambda x:x[0]):
        for _,ok,plan in group:count+=1;correct+=ok;covered+=plan
        coverage=covered/supported if supported else 0.
        if count>=30 and correct/count>=precision and coverage>best:
            best=coverage;threshold=score
    result=dict(cal,threshold=threshold)
    metrics,_=acceptance(rows,preds,good,result)
    return result,metrics

def calibrate(data_dir,checkpoint,device):
    torch.set_num_threads(4);rows=load_rows(Path(data_dir)/"calibration.jsonl");model,ck=load_checkpoint(checkpoint,device)
    preds=predict_rows(model,rows,device,progress=lambda n,total:print(json.dumps({"event":"calibration","seen":n,"total":total}),flush=True))
    _,good=evaluate_predictions(rows,preds)
    # Fit/threshold separation within the calibration families, stratified by row.
    fit=list(range(0,len(rows),2));hold=list(range(1,len(rows),2))
    cal=fit_calibration([preds[i]["features"] for i in fit],[good[i] for i in fit])
    cal.update(threshold=1.000001,method="regularized logistic, even rows fit; odd rows threshold; >=98% empirical plan precision",fit_rows=len(fit),threshold_rows=len(hold))
    scores=[confidence(preds[i]["features"],cal) for i in hold]
    candidates=sorted(set(scores),reverse=True)
    best_coverage=-1
    hr=[rows[i] for i in hold];hp=[preds[i] for i in hold];hg=[good[i] for i in hold]
    for threshold in candidates:
        candidate=dict(cal,threshold=threshold);metrics,_=acceptance(hr,hp,hg,candidate)
        if metrics["accepted"]>=30 and metrics["semantic_precision"]>=.98 and metrics["supported_coverage"]>best_coverage:
            cal=candidate;best_coverage=metrics["supported_coverage"]
    metrics,_=acceptance(hr,hp,hg,cal);cal["threshold_metrics"]=metrics
    ck["calibration"]=cal;torch.save(ck,checkpoint)
    Path(checkpoint).with_suffix(".calibration.json").write_text(json.dumps(cal,ensure_ascii=False,indent=2),encoding="utf-8")
    return cal

def error_kind(expected,actual):
    if actual is None:return "decoding"
    if actual["kind"]!=expected["kind"]:return "refusal_or_action"
    if expected["kind"]!="plan":return "refusal_reason"
    es=expected["steps"];ps=actual["steps"]
    if len(es)!=len(ps):return "step_count_or_order"
    if [s["tool"] for s in es]!=[s["tool"] for s in ps]:return "tools_or_order"
    if [s.get("when") for s in es]!=[s.get("when") for s in ps]:return "condition"
    if any("result" in json.dumps(s["args"]) for s in es+ps):return "reference_or_argument"
    return "argument_role"

def error_dimensions(rows,preds):
    """Overlapping semantic dimensions, separate from the primary error bucket."""
    counts=collections.Counter({k:0 for k in ("tools","arguments","order","condition","reference","refusal_reason","kind","step_count","decoding")})
    def refs(value):
        if isinstance(value,dict):
            return ([value["result"]] if "result" in value else [])+sum((refs(v) for v in value.values()),[])
        if isinstance(value,list):return sum((refs(v) for v in value),[])
        return []
    for row,pred in zip(rows,preds):
        expected=row["target"];actual=pred["plan"]
        if actual is None:counts["decoding"]+=1;continue
        if expected["kind"]!=actual["kind"]:counts["kind"]+=1;continue
        if expected["kind"]!="plan":counts["refusal_reason"]+=int(expected["message"]!=actual["message"]);continue
        es=json.loads(canonical(expected))["steps"];ps=json.loads(canonical(actual))["steps"]
        et=[s["tool"] for s in es];pt=[s["tool"] for s in ps]
        counts["tools"]+=int(collections.Counter(et)!=collections.Counter(pt))
        counts["order"]+=int(et!=pt and collections.Counter(et)==collections.Counter(pt))
        counts["step_count"]+=int(len(es)!=len(ps))
        counts["arguments"]+=int([s["args"] for s in es]!=[s["args"] for s in ps])
        counts["condition"]+=int([s.get("when") for s in es]!=[s.get("when") for s in ps])
        counts["reference"]+=int(refs(es)!=refs(ps))
    return dict(counts)

def display_input(row):
    # Composition rows retain legacy normalized metadata from their first clause;
    # always render the complete structural input actually supplied to the model.
    segments=row.get("request",{}).get("segments")
    if segments is not None:return "".join(s["text"] if s["kind"]=="literal" else s["id"] for s in segments)
    return row.get("normalized",row.get("raw"))

def report(rows,preds,ck):
    metrics,good=evaluate_predictions(rows,preds);metrics["acceptance"],accepted=acceptance(rows,preds,good,ck["calibration"])
    deployed=[]
    for r,p,a,ok in zip(rows,preds,accepted,good):
        deployed.append(ok and (r["target"]["kind"]!="plan" or a))
    metrics["gated_whole_request_accuracy"]=sum(deployed)/len(rows)
    metrics["errors"]=dict(collections.Counter(error_kind(r["target"],p["plan"]) for r,p,ok in zip(rows,preds,good) if not ok))
    metrics["error_dimensions"]=error_dimensions(rows,preds)
    errors=[{"input":display_input(r),"expected":r["target"],"actual":p["plan"],"accepted":a,"category":r["category"]} for r,p,ok,a in zip(rows,preds,good,accepted) if not ok or (r["target"]["kind"]=="plan" and not a)]
    return metrics,errors

def evaluate(data_dir,checkpoint,device):
    root=Path(data_dir);out=Path(checkpoint).parent/"translator-evaluation"
    out.mkdir(exist_ok=True);final_marker=out/"FINAL_TEST_FROZEN.json"
    ckhash=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if final_marker.exists():
        prior=json.loads(final_marker.read_text())
        if prior["checkpoint_sha256"]!=ckhash:raise RuntimeError("Final tests were already run for another candidate; do not tune against them")
        if (out/"report.json").exists():return json.loads((out/"report.json").read_text(encoding="utf-8"))
    else:final_marker.write_text(json.dumps({"checkpoint_sha256":ckhash,"selected_before_test":True}),encoding="utf-8")
    torch.set_num_threads(4);model,ck=load_checkpoint(checkpoint,device);summary={"checkpoint_sha256":ckhash,"parameters":ck["parameters"],"selected_validation":ck["validation"],"calibration":ck["calibration"],"datasets":json.loads((root/"manifest.json").read_text(encoding="utf-8")),"suites":{}}
    training_files=[root/"train.jsonl"]+[Path("data")/name for name in ("translator1.0-round2.jsonl","translator1.0-round3.jsonl") if (Path("data")/name).exists()]
    seen_sequences=set()
    for path in training_files:
        for r in load_rows(path):
            if r["target"]["kind"]=="plan":seen_sequences.add(tuple(s["tool"] for s in r["target"]["steps"]))
    for split in ("test","combinations","independent"):
        rows=load_rows(root/(split+".jsonl"));preds=predict_rows(model,rows,device,progress=lambda n,total:print(json.dumps({"event":"test","suite":split,"seen":n,"total":total}),flush=True))
        metrics,errors=report(rows,preds,ck);summary["suites"][split+"_oracle"]=metrics
        if split=="combinations":
            unseen=[i for i,r in enumerate(rows) if tuple(s["tool"] for s in r["target"]["steps"]) not in seen_sequences]
            if unseen:
                summary["suites"]["strict_unseen_tool_sequences"],_=report([rows[i] for i in unseen],[preds[i] for i in unseen],ck)
            metrics["strict_unseen_sequence_rows"]=len(unseen)
            metrics["note"]="Intent-pair holdout is broader than strict tool-sequence holdout; both reported separately."
        (out/(split+"_errors.json")).write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding="utf-8")
    from filter10 import FilterHarness
    from davework.adapter import planner_input
    filter=FilterHarness(device=device);rows=load_rows(root/"independent.jsonl");real=[];binding_ok=[];extractions=[]
    for i,row in enumerate(rows):
        f=filter.extract(row["raw"]).to_dict();extractions.append(f)
        expected=[(s["start"],s["end"],s["text"]) for s in row["spans"]]
        actual=[(s["start"],s["end"],f["original"][s["start"]:s["end"]]) for s in f["spans"]]
        binding_ok.append(expected==actual)
        req=planner_input({"id":"ctx_eval","filter":f,"settings":{"enabled_tools":list(model_tools())}})
        real.append(dict(row,request=req))
        if i%40==0:print(json.dumps({"event":"real_filter","seen":i,"total":400}),flush=True)
    preds=predict_rows(model,real,device)
    metrics,plan_good=evaluate_predictions(real,preds);joint=[a and b for a,b in zip(plan_good,binding_ok)]
    metrics["exact_accuracy_with_bindings"]=sum(joint)/400;metrics["binding_exact_accuracy"]=sum(binding_ok)/400
    metrics["acceptance"],accepted=acceptance(real,preds,joint,ck["calibration"])
    final_good=[ok and (r["target"]["kind"]!="plan" or a) and r["request"].get("filter_status") not in ("error","unsupported","uncertain","needs_context") for r,ok,a in zip(real,joint,accepted)]
    metrics["gated_whole_request_accuracy"]=sum(final_good)/400
    metrics["filter_statuses"]=dict(collections.Counter(f["status"] for f in extractions))
    metrics["errors"]=dict(collections.Counter("filter_binding" if not b else error_kind(r["target"],p["plan"]) if not ok else "confidence_rejection" for r,p,b,ok,fg in zip(real,preds,binding_ok,plan_good,final_good) if not fg))
    metrics["error_dimensions"]=error_dimensions(real,preds)
    negatives=[i for i,r in enumerate(real) if r["target"]["kind"]!="plan" or (
        all(s["tool"] in ("fs.list","fs.read_text","fs.search") for s in r["target"]["steps"])
        and any(word in r["raw"] for word in ("不要","别","禁止","不允许","只读","不修改","不用","不删","勿")))]
    metrics["negative_safety"]={"cases":len(negatives),"accepted_wrong_modifications":sum(accepted[i] and any(s["tool"] not in ("fs.list","fs.read_text","fs.search") for s in preds[i]["plan"]["steps"]) for i in negatives)}
    summary["suites"]["independent_real_filter"]=metrics
    errors=[{"raw":r["raw"],"expected":r["target"],"extraction":f,"predicted":p["plan"],"bindings_correct":b,"accepted":a} for r,f,p,b,a,ok in zip(real,extractions,preds,binding_ok,accepted,final_good) if not ok]
    (out/"real_filter_errors.json").write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding="utf-8")
    safety_path=Path("data/translator1.0-safety.jsonl")
    if safety_path.exists():
        safety=load_rows(safety_path)
        for mode in ("oracle","real_filter"):
            cases=[]
            for row in safety:
                if mode=="oracle":cases.append(row);continue
                f=filter.extract(row["raw"]).to_dict()
                req=planner_input({"id":"ctx_eval","filter":f,"settings":{"enabled_tools":list(model_tools())}})
                cases.append(dict(row,request=req))
            pred=predict_rows(model,cases,device);sm,errs=report(cases,pred,ck)
            _,sg=evaluate_predictions(cases,pred);_,sa=acceptance(cases,pred,sg,ck["calibration"])
            sm["accepted_modifying_actions"]=sum(a and any(s["tool"] not in ("fs.list","fs.read_text","fs.search") for s in p["plan"]["steps"]) for a,p in zip(sa,pred))
            sm["all_modifying_predictions"]=sum(p["plan"] is not None and p["plan"]["kind"]=="plan" and any(s["tool"] not in ("fs.list","fs.read_text","fs.search") for s in p["plan"]["steps"]) for p in pred)
            sm["note"]="100 additional targeted checks authored during implementation; never used to tune; reported separately from pretraining-frozen 400."
            summary["suites"]["safety_"+mode]=sm
            (out/("safety_"+mode+"_errors.json")).write_text(json.dumps(errs,ensure_ascii=False,indent=2),encoding="utf-8")
        summary["safety_suite_sha256"]=hashlib.sha256(safety_path.read_bytes()).hexdigest()
    summary["targets"]={"translator_95":summary["suites"]["test_oracle"]["exact_accuracy"]>=.95,"joint_85":metrics["gated_whole_request_accuracy"]>=.85,"accepted_precision_98":(metrics["acceptance"]["semantic_precision"] or 0)>=.98,"coverage_85":(metrics["acceptance"]["supported_coverage"] or 0)>=.85}
    (out/"report.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    return summary

def model_tools():
    from .language import FILE_TOOLS
    return FILE_TOOLS

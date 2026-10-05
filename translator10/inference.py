"""Inference, independent confidence gating and the Dave Work planner interface."""
import math
from pathlib import Path
import torch
from .language import REASONS, nonexecution
from .training import load_checkpoint

def confidence(features,calibration):
    if not calibration:return 0.
    count=len(calibration['mean'])
    if len(features)<count:return 0.
    xs=[(x-m)/max(s,1e-6) for x,m,s in zip(features,calibration['mean'],calibration['scale'])]
    mapping=calibration.get('feature_map','linear')
    if mapping=='quadratic':
        xs=[max(-4.,min(4.,x)) for x in xs]
        xs=xs+[xs[i]*xs[j] for i in range(count) for j in range(i,count)]
    elif mapping!='linear':return 0.
    if len(xs)!=len(calibration['weights']):return 0.
    z=calibration["bias"]+sum(w*x for w,x in zip(calibration['weights'],xs))
    return 1/(1+math.exp(-max(-50.,min(50.,z))))

def joint_features(prediction,request):
    """Decoder evidence plus a numeric filter score; no binding text."""
    probability=max(1e-6,min(1.-1e-6,float(request.get('filter_confidence',1.))))
    extra=[]
    for key in ('filter_start_confidence','filter_end_confidence','filter_count_confidence'):
        p=max(1e-6,min(1.-1e-6,float(request.get(key,.5))))
        extra.append(math.log(p/(1.-p)))
    return list(prediction['features'][:3])+[math.log(probability/(1.-probability)),float(request.get('filter_boundary_margin',0.))]+extra+[float(request.get('filter_count_match',True))]

class TranslatorPlanner:
    def __init__(self,checkpoint=None,device="cpu"):
        torch.set_num_threads(4)
        path=checkpoint or Path(__file__).resolve().parent.parent/"artifacts"/"translator1.0.pt"
        self.model,self.checkpoint=load_checkpoint(path,device);self.device=device
    def plan(self,request):
        ctx=request.get("context_id","ctx_eval")
        if len(request["slots"])>8:
            return nonexecution("clarification","TOO_COMPLEX",ctx)
        if request.get("filter_status") in ("error","unsupported","unsupported_length","needs_context","uncertain"):
            return nonexecution("clarification","LOW_CONFIDENCE",ctx)
        expected = (self.checkpoint.get('calibration') or {}).get('filter_sha256')
        if expected and request.get('filter_sha256') != expected:
            return nonexecution('clarification', 'LOW_CONFIDENCE', ctx)
        try:pred=self.model.generate([request],self.device,beam=4)[0]
        except (ValueError,KeyError) as exc:
            return nonexecution("clarification","MISSING_PARAMETER",ctx)
        cal=self.checkpoint.get("calibration");score=confidence(joint_features(pred,request),cal)
        plan=pred["plan"]
        if plan is None or (plan["kind"]=="plan" and (not cal or score<cal["threshold"])):
            return nonexecution("clarification","LOW_CONFIDENCE",ctx)
        # Protocol plans intentionally have no confidence/metadata extra fields.
        return plan

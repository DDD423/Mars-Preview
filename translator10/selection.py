"""Select a single weight file using validation only (including weight means).

Parameter averaging is still one Transformer at inference, not an ensemble.
Final test suites are not opened here.
"""
import copy
import json
from pathlib import Path
import torch
from .data import load_rows
from .training import load_checkpoint,predict_rows,evaluate_predictions

def snapshot(out_dir):
    root=Path(out_dir);folder=root/"snapshots";folder.mkdir(exist_ok=True)
    made=[]
    for name in ("best.pt","last.pt"):
        try:
            ck=torch.load(root/name,map_location="cpu",weights_only=False)
            dest=folder/("epoch_%02d.pt"%ck["epoch"])
            if not dest.exists():torch.save(ck,dest);made.append(str(dest))
        except (OSError,RuntimeError,EOFError):pass # writer may be publishing last.pt
    return made

def select(data_dir,checkpoint="artifacts/translator1.0.pt",device="cuda"):
    torch.set_num_threads(4);valid=load_rows(Path(data_dir)/"validation.jsonl")
    records=[]
    for i in (1,2,3):
        path=Path("artifacts")/("translator-round"+str(i))/"best.pt"
        _,ck=load_checkpoint(path,"cpu");records.append((ck["validation"]["exact_accuracy"],path,ck))
    score,path,best=max(records,key=lambda x:x[0]);selection=[{"candidate":str(p),"validation_accuracy":s} for s,p,_ in records]
    folder=Path("artifacts/translator-round3");snapshot(folder)
    snaps=[]
    for p in (folder/"snapshots").glob("*.pt"):
        ck=torch.load(p,map_location="cpu",weights_only=False);snaps.append((ck["validation"]["exact_accuracy"],p,ck))
    snaps=sorted(snaps,key=lambda x:x[0],reverse=True)[:5]
    for n in (2,3,5):
        if len(snaps)<n:continue
        candidate=copy.deepcopy(snaps[0][2]);candidate["model"]={k:torch.stack([r[2]["model"][k] for r in snaps[:n]]).mean(0) for k in candidate["model"]}
        p=folder/("average_%d.pt"%n);torch.save(candidate,p)
        model,_=load_checkpoint(p,device);pred=predict_rows(model,valid,device)
        metrics,_=evaluate_predictions(valid,pred);s=metrics["exact_accuracy"]
        candidate["validation"]=metrics;candidate["selection"]={"method":"parameter_mean","source_epochs":[r[2]["epoch"] for r in snaps[:n]]};torch.save(candidate,p)
        selection.append({"candidate":str(p),"validation_accuracy":s});print(json.dumps(selection[-1]),flush=True)
        if s>score:score,path,best=s,p,candidate
        del model
        if device=="cuda":torch.cuda.empty_cache()
    best["selected_from"]=str(path);best["selection_candidates"]=selection;torch.save(best,checkpoint)
    report={"selected":str(path),"validation_accuracy":score,"candidates":selection}
    Path(checkpoint).with_suffix(".selection.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

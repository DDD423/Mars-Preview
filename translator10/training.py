"""Reproducible GPU training and validation; never reads frozen test files."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .data import load_rows
from .language import IDS, TOKENS, canonical, fingerprint, masks, FILE_TOOLS
from .model import Config, TranslatorModel, encode_request

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def prepare(rows):
    items=[]
    for r in rows:
        target=r["tokens"]; allowed,_=masks(r["request"],target)
        mask=torch.full((len(target)-1,len(TOKENS)),-10000.,dtype=torch.float32)
        for i, tokens in enumerate(allowed):mask[i,tokens]=0
        items.append((torch.tensor(encode_request(r["request"])),torch.tensor(target),mask))
    return items

def batch(items,device):
    x=pad_sequence([r[0] for r in items],batch_first=True)
    target=pad_sequence([r[1] for r in items],batch_first=True)
    width=target.shape[1]-1
    mask=torch.zeros(len(items),width,len(TOKENS))
    for i,r in enumerate(items):mask[i,:r[2].shape[0]]=r[2]
    return x.to(device),target.to(device),mask.to(device)

def predict_rows(model,rows,device,beam=4,batch_size=64,progress=None):
    output=[None]*len(rows)
    order=sorted(range(len(rows)),key=lambda i:(rows[i].get("category",""),len(encode_request(rows[i]["request"]))))
    for start in range(0,len(order),batch_size):
        ids=order[start:start+batch_size]
        preds=model.generate([rows[i]["request"] for i in ids],device=device,beam=beam)
        for i,p in zip(ids,preds):output[i]=p
        if progress and start%512==0:progress(start,len(order))
    return output

def evaluate_predictions(rows,preds):
    good=[p["plan"] is not None and canonical(p["plan"])==canonical(r["target"]) for r,p in zip(rows,preds)]
    counts={}
    for r,ok in zip(rows,good):
        c=counts.setdefault(r["category"],{"correct":0,"total":0});c["total"]+=1;c["correct"]+=int(ok)
    return {"correct":sum(good),"total":len(good),"exact_accuracy":sum(good)/len(good),"categories":counts},good

def load_checkpoint(path,device="cpu"):
    ck=torch.load(path,map_location="cpu",weights_only=False)
    if ck["schema_fingerprint"]!=fingerprint() or ck["tokens"]!=TOKENS:raise ValueError("Translator tool/vocabulary fingerprint mismatch")
    model=TranslatorModel(Config(**ck["config"]));model.load_state_dict(ck["model"]);model.to(device).eval()
    return model,ck


def training_rows(path):
    """Retain only model inputs/targets; raw bindings are not training inputs."""
    rows=[]
    with Path(path).open(encoding='utf-8') as file:
        for line in file:
            row=json.loads(line)
            rows.append({'request':row['request'],'tokens':row['tokens']})
    return rows

def train(data_dir,out_dir,device="cuda",epochs=50,batch_size=64,seed=42,resume=None,extra=None,learning_rate=.0003,epoch_samples=0,threads=4):
    seed_all(seed);torch.set_num_threads(threads)
    if device=="cuda":
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; pass --device cpu explicitly")
        torch.zeros(8,device="cuda").add_(1).cpu()
        torch.backends.cuda.matmul.allow_tf32=True
    root=Path(data_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=training_rows(root/"train.jsonl");valid=load_rows(root/"validation.jsonl")
    if extra:rows+=training_rows(extra)
    model=TranslatorModel().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    if resume:
        old,ck=load_checkpoint(resume,device);model=old
        optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-5)
    lengths=[len(FILE_TOOLS)+1+sum(len(p['text']) if p['kind']=='literal' else 1 for p in r['request']['segments']) for r in rows]
    scaler=torch.amp.GradScaler("cuda",enabled=device=="cuda")
    best=-1.;stale=0;history=[];started=time.time()
    print(json.dumps({"event":"start","parameters":model.metadata()["parameters"],"device":device,"gpu":torch.cuda.get_device_name() if device=="cuda" else None,"train":len(rows),"validation":len(valid),"batch":batch_size}),flush=True)
    if resume:
        baseline,_=evaluate_predictions(valid,predict_rows(model,valid,device))
        best=baseline['exact_accuracy']
        baseline_ck=copy.deepcopy(ck);baseline_ck['calibration']=None
        torch.save(baseline_ck,out/'best.pt')
        (out/'resume_baseline.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
        print(json.dumps({'event':'resume_baseline','validation':baseline}),flush=True)
    for epoch in range(1,epochs+1):
        model.train();order=list(range(len(rows)));random.shuffle(order)
        if epoch_samples:order=order[:epoch_samples]
        # Local length bucketing reduces padding, preserving randomized batches.
        for start in range(0,len(order),batch_size*20):order[start:start+batch_size*20]=sorted(order[start:start+batch_size*20],key=lambda i:lengths[i])
        losses=[];t=time.time()
        for start in range(0,len(order),batch_size):
            ids=order[start:start+batch_size]
            try:
                x,y,mask=batch(prepare([rows[i] for i in ids]),device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device=="cuda"):
                    logits=model(x,y[:,:-1]).float()+mask
                    loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),y[:,1:].flatten(),ignore_index=0)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();losses.append(loss.item())
            except torch.cuda.OutOfMemoryError:
                if batch_size<=16:raise
                batch_size//=2;torch.cuda.empty_cache()
                print(json.dumps({"event":"oom","new_batch":batch_size,"retry_epoch":epoch}),flush=True)
                return train(data_dir,out_dir,device,epochs,batch_size,seed,resume,extra,learning_rate,epoch_samples,threads)
            if start%(batch_size*200)==0:print(json.dumps({"event":"batch","epoch":epoch,"seen":start,"loss":round(loss.item(),5),"seconds":round(time.time()-t,1)}),flush=True)
        train_seconds=time.time()-t;t=time.time()
        preds=predict_rows(model,valid,device,beam=4,batch_size=64,progress=lambda done,total:print(json.dumps({"event":"validation","epoch":epoch,"seen":done,"total":total,"seconds":round(time.time()-t,1)}),flush=True))
        metrics,good=evaluate_predictions(valid,preds)
        row={"epoch":epoch,"loss":float(np.mean(losses)),"validation":metrics,"train_seconds":train_seconds,"validation_seconds":time.time()-t,"elapsed_seconds":time.time()-started}
        history.append(row);print(json.dumps({"event":"epoch",**row}),flush=True)
        ck={"version":"translator1.0","config":model.metadata()["config"],"parameters":model.metadata()["parameters"],"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"tokens":TOKENS,"schema_fingerprint":fingerprint(),"seed":seed,"epoch":epoch,"validation":metrics,"data_manifest":json.loads((root/"manifest.json").read_text(encoding="utf-8")),"calibration":None,
            "training_recipe":{"device":device,"threads":threads,"batch":batch_size,"batch_preparation":"on-demand, raw bindings discarded on load","learning_rate":learning_rate,"scheduler":"validation plateau x.5 patience2 minimum1e-5","epoch_samples":epoch_samples,"clip":1.,"max_epochs":epochs,"patience":7,"resume":str(resume) if resume else None,"resume_sha256":hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,"extra":str(extra) if extra else None,"extra_sha256":hashlib.sha256(Path(extra).read_bytes()).hexdigest() if extra else None}}
        temporary=out/'saving.pt';torch.save(ck,temporary);temporary.replace(out/'last.pt')
        scheduler.step(metrics['exact_accuracy'])
        if metrics["exact_accuracy"]>best:
            best=metrics["exact_accuracy"];stale=0;torch.save(ck,temporary);temporary.replace(out/'best.pt')
            (out/"validation_errors.json").write_text(json.dumps([{ "request":r["request"],"target":r["target"],"predicted":p["plan"],"category":r["category"]} for r,p,ok in zip(valid,preds,good) if not ok],ensure_ascii=False,indent=2),encoding="utf-8")
        else:stale+=1
        (out/"history.json").write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding="utf-8")
        if stale>=7 or (out/"STOP_AFTER_EPOCH").exists():break
    return {"best_validation":best,"epochs":len(history),"path":str(out/"best.pt")}

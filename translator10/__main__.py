import argparse
import json
import sys
from pathlib import Path

def main():
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser(prog="python -m translator10")
    sub=p.add_subparsers(dest="command",required=True)
    g=sub.add_parser("generate");g.add_argument("--data",default="data/translator1.0")
    t=sub.add_parser("train");t.add_argument("--data",default="data/translator1.0-frozen");t.add_argument("--out",default="artifacts/translator-round1");t.add_argument("--device",default="cuda");t.add_argument("--epochs",type=int,default=50);t.add_argument("--batch",type=int,default=64);t.add_argument("--resume");t.add_argument("--extra");t.add_argument('--learning-rate',type=float,default=.0003);t.add_argument('--epoch-samples',type=int,default=0)
    t.add_argument('--threads',type=int,default=4)
    for name in ("calibrate","evaluate"):
        q=sub.add_parser(name);q.add_argument("--data",default="data/translator1.0-frozen");q.add_argument("--checkpoint",default="artifacts/translator1.0.pt");q.add_argument("--device",default="cuda")
    for name in ("predict","chat"):
        q=sub.add_parser(name);q.add_argument("text",nargs="?");q.add_argument("--checkpoint",default="artifacts/translator1.0.pt");q.add_argument("--device",default="cpu")
        q.add_argument('--filter-checkpoint', help='Matching filter checkpoint for a named model pair')
    q=sub.add_parser("selftest");q.add_argument("--checkpoint",default="artifacts/translator1.0.pt");q.add_argument("--device",default="cpu")
    a=p.parse_args()
    if a.command=="generate":
        from .data import generate
        print(json.dumps(generate(a.data),ensure_ascii=False,indent=2))
    elif a.command=="train":
        from .training import train
        print(json.dumps(train(a.data,a.out,a.device,a.epochs,a.batch,resume=a.resume,extra=a.extra,learning_rate=a.learning_rate,epoch_samples=a.epoch_samples,threads=a.threads)))
    elif a.command in ("calibrate","evaluate"):
        from .evaluation import calibrate,evaluate
        print(json.dumps((calibrate if a.command=="calibrate" else evaluate)(a.data,a.checkpoint,a.device),ensure_ascii=False,indent=2))
    elif a.command=="selftest":
        from .selftest import selftest
        r=selftest(a.checkpoint,a.device);print(json.dumps({"successful":r["successful"],"total":r["total"]}))
    else:
        from .inference import TranslatorPlanner
        from filter10 import FilterHarness
        from davework.adapter import planner_input
        model=TranslatorPlanner(a.checkpoint,a.device)
        filter=FilterHarness(a.filter_checkpoint,a.device) if a.filter_checkpoint else FilterHarness(device=a.device)
        def predict(text):
            f=filter.extract(text).to_dict()
            r=planner_input({"id":"ctx_cli","filter":f,"settings":{"enabled_tools":None}})
            print(json.dumps({"filter":f,"translator":model.plan(r)},ensure_ascii=False,indent=2))
        if a.command=="predict":predict(a.text or input("中文请求 > "))
        else:
            while True:
                try:text=input("中文请求（exit 退出）> ")
                except EOFError:break
                if text.strip().lower() in ("exit","quit"):break
                predict(text)

if __name__=="__main__":main()

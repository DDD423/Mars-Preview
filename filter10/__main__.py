import argparse
import json
import sys


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(prog="python -m filter10", description="filter1.0 本地文字片段提取")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="生成按表达家族隔离的数据，冻结独立审计集")
    generate.add_argument("--data-dir", default="data")
    generate.add_argument("--seed", type=int, default=42)
    augment = sub.add_parser("augment", help="用训练家族的控制语句同义变体补充训练数据")
    augment.add_argument("--data-dir", default="data")
    augment.add_argument("--count", type=int, default=8000)
    augment.add_argument("--seed", type=int, default=91)
    post_generate = sub.add_parser("postgenerate", help="生成后训练数据，保持新的独立测试集冻结")
    post_generate.add_argument("--data-dir", default="data/posttrain")
    post_generate.add_argument("--seed", type=int, default=117)
    train = sub.add_parser("train", help="从零训练或继续训练")
    train.add_argument("--data-dir", default="data")
    train.add_argument("--output", default="artifacts/filter1.0.pt")
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--resume")
    train.add_argument("--run", type=int, default=1)
    train.add_argument("--posttrain", action="store_true")
    train.add_argument("--adaptive", action="store_true")
    train.add_argument("--ranking-weight", type=float, default=0.0)
    train.add_argument('--scheduler', action='store_true')
    train.add_argument('--quote-competitors', action='store_true')
    train.add_argument('--raw-negatives', action='store_true')
    train.add_argument('--balanced-validation', action='store_true')
    train.add_argument('--positive-loss-weight',type=float,default=0.0)
    train.add_argument('--contrastive-weight',type=float,default=0.0)
    train.add_argument('--structured-weight',type=float,default=0.0)
    train.add_argument('--length-bucket-size',type=int,default=0)
    train.add_argument('--conditional-span-set',action='store_true')
    train.add_argument('--counterfactual-weight',type=float,default=0.0)
    train.add_argument('--counterfactual-probability',type=float,default=0.5)
    train.add_argument('--exclude-context-aliases',action='store_true')
    train.add_argument('--online-hard-negatives',type=int,default=0)
    train.add_argument('--online-pool-size',type=int,default=512)
    train.add_argument('--literal-negative-width',type=int,default=0)
    train.add_argument('--parent-retention-weight',type=float,default=0.)
    train.add_argument('--epoch-samples',type=int,default=0)
    train.add_argument('--boundary-weight',type=float,default=0.)
    train.add_argument('--boundary-mix',type=float,default=0.)
    train.add_argument('--endpoint-weight',type=float,default=0.)
    train.add_argument('--local-window',type=int,default=0)
    train.add_argument('--local-mix',type=float,default=0.)
    train.add_argument('--skip-initial-mining',action='store_true')
    train.add_argument('--separate-aux-heads',action='store_true')
    train.add_argument('--exact-counts',action='store_true')
    train.add_argument('--count-weight',type=float,default=0.0)
    train.add_argument('--count-policy',choices=['cap','zero_only','off','exact','range'],default='cap')
    semantic = sub.add_parser('semanticgenerate',help='冻结按句式家族隔离的语义/语序挑战')
    semantic.add_argument('--directory',default='data/semantic')
    semantic.add_argument('--seed',type=int,default=631)
    role = sub.add_parser('roleaugment',help='生成只用于训练的角色与连接词组合，不改冻结测试')
    role.add_argument('--directory',default='data/neural_v2')
    role.add_argument('--count',type=int,default=42000)
    role.add_argument('--seed',type=int,default=839)
    role_composition = sub.add_parser('rolecompose',help='训练数据中的角色与操作组合，不改冻结测试')
    role_composition.add_argument('--directory',default='data/neural_v3')
    role_composition.add_argument('--count',type=int,default=42000)
    role_composition.add_argument('--seed',type=int,default=947)
    anchors = sub.add_parser('anchoraugment',help='短句训练回放，防止噪音测试掩盖基础提取退化')
    anchors.add_argument('--directory',default='data/neural_v4')
    anchors.add_argument('--count',type=int,default=14000)
    anchors.add_argument('--seed',type=int,default=1109)
    constituents = sub.add_parser('constituentaugment',help='组合参数短语和操作语句，只生成训练数据')
    constituents.add_argument('--directory',default='data/neural_v5')
    constituents.add_argument('--count',type=int,default=38000)
    constituents.add_argument('--seed',type=int,default=1223)
    paraphrases = sub.add_parser('paraphraseaugment',help='训练控制语言的等价表达，名称内容保持不变')
    paraphrases.add_argument('--directory',default='data/neural_v6')
    paraphrases.add_argument('--count',type=int,default=44000)
    paraphrases.add_argument('--seed',type=int,default=1361)
    relations = sub.add_parser('relationaugment',help='自然角色关系和多参数训练，冻结测试保持不变')
    relations.add_argument('--directory',default='data/neural_v7')
    relations.add_argument('--count',type=int,default=30000)
    relations.add_argument('--seed',type=int,default=1459)
    context = sub.add_parser('contextaugment',help='语法自然的角色短语与关系组合，只用于训练')
    context.add_argument('--directory',default='data/neural_v8')
    context.add_argument('--count',type=int,default=40000)
    context.add_argument('--seed',type=int,default=1543)
    context.add_argument('--natural',action='store_true',help='修正名称所属关系，写入新训练目录以保留历史数据')
    neutral = sub.add_parser('neutralaugment',help='参数角色与后续说明独立组合，只用于训练')
    neutral.add_argument('--directory',default='data/neural_v10')
    neutral.add_argument('--count',type=int,default=36000)
    neutral.add_argument('--seed',type=int,default=1733)
    incidental = sub.add_parser('incidentalaugment',help='插入说明、角色顺序、长参数和无参数概念训练，不改变测试')
    incidental.add_argument('--directory',default='data/neural_v11')
    incidental.add_argument('--count',type=int,default=32000)
    incidental.add_argument('--seed',type=int,default=1847)
    family_replay = sub.add_parser('familyreplay',help='保留原始语义训练家族，减轻多轮组合回放后的遗忘')
    family_replay.add_argument('--directory',default='data/neural_v12')
    family_replay.add_argument('--seed',type=int,default=1961)
    scopes = sub.add_parser('scopeaugment',help='已提供/缺失参数独立组合及延后角色说明，只用于训练')
    scopes.add_argument('--directory',default='data/neural_v13')
    scopes.add_argument('--count',type=int,default=30000)
    scopes.add_argument('--seed',type=int,default=2081)
    inventory = sub.add_parser('testinventory',help='审计冻结测试的多样性和训练重叠，不读取预测错误')
    inventory.add_argument('--data-dir',default='data/neural_v7')
    inventory.add_argument('--output',default='artifacts/benchmark_inventory.json')
    weight_average = sub.add_parser('weightaverage',help='同一后训练链上的参数平均，保持单模型和参数量')
    weight_average.add_argument('--checkpoints',nargs='+',required=True)
    weight_average.add_argument('--output',required=True)
    weight_average.add_argument('--coefficients',nargs='+',type=float)
    comprehensive = sub.add_parser('neurevaluate',help='纯模型在噪音、语义和固定规则失败子集上的综合评测')
    comprehensive.add_argument('--checkpoint',default='artifacts/noisy.pt')
    comprehensive.add_argument('--device',default='cpu')
    comprehensive.add_argument('--final',action='store_true')
    personal = sub.add_parser('selftest',help='用自己预先标注的JSONL测纯模型，不修改冻结基准')
    personal.add_argument('--input',required=True)
    personal.add_argument('--checkpoint',default='artifacts/filter1.0.pt')
    personal.add_argument('--device',default='cpu')
    personal.add_argument('--output',default='artifacts/personal_evaluation.json')
    noisy = sub.add_parser('noisegenerate', help='冻结噪音难度测试集并生成独立训练实例')
    noisy.add_argument('--directory', default='data/noisy')
    noisy.add_argument('--seed', type=int, default=521)
    noisy.add_argument('--train-count', type=int, default=70000)
    noisy_eval = sub.add_parser('noiseevaluate', help='固定0.5阈值、无语法修正的纯模型噪音评测')
    noisy_eval.add_argument('--checkpoint', default='artifacts/noisy.pt')
    noisy_eval.add_argument('--directory', default='data/noisy')
    noisy_eval.add_argument('--device', default='cpu')
    noisy_eval.add_argument('--split', choices=['challenge','validation','calibration'], default='challenge')
    noisy_eval.add_argument('--final', action='store_true', help='评估完成后保存错误，训练不使用它们')
    adapt_generate = sub.add_parser('adaptgenerate', help='生成口语、规则缺口、指代训练数据和冻结测试')
    adapt_generate.add_argument('--data-dir', default='data/adaptive')
    adapt_generate.add_argument('--seed', type=int, default=203)
    for name, directory, count, seed in [('adaptaugment', 'data/adaptive_v2', 26000, 317),
                                          ('adaptcompose', 'data/adaptive_v3', 42000, 419)]:
        p = sub.add_parser(name, help='生成仅用于训练的组合语句；不修改冻结测试')
        p.add_argument('--directory', default=directory)
        p.add_argument('--count', type=int, default=count)
        p.add_argument('--seed', type=int, default=seed)
    for name in ('adaptcalibrate', 'adaptevaluate'):
        p = sub.add_parser(name)
        p.add_argument('--checkpoint', default='artifacts/adaptive.pt')
        p.add_argument('--data-dir', default='data/adaptive')
        p.add_argument('--device', default='cpu')
        if name == 'adaptevaluate':
            p.add_argument('--splits', nargs='+')
    post_eval = sub.add_parser("postevaluate", help="评估新冻结测试集及模型/harness 消融对比")
    post_eval.add_argument("--checkpoint", default="artifacts/posttrained.pt")
    post_eval.add_argument("--device", default="cpu")
    diag = sub.add_parser("diagnostics", help="测量 CPU 推理耗时并保存命令演示")
    diag.add_argument("--checkpoint", default="artifacts/filter1.0.pt")
    diag.add_argument('--data-dir',help='纯模型实际使用的训练目录，用于交付校验记录')
    for name in ["calibrate", "evaluate", "predict", "interactive", 'compare']:
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", default="artifacts/filter1.0.pt")
        p.add_argument("--device", default="cpu")
        if name in ("calibrate", "evaluate"):
            p.add_argument("--data-dir", default="data")
        if name == "evaluate":
            p.add_argument("--splits", nargs="+", default=["test", "audit"])
        if name in ("predict", 'compare'):
            p.add_argument("text", nargs="?", help="省略时从stdin读取完整文本")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "generate":
        from .data import generate
        from .audit import generate_audit
        audit = generate_audit(args["data_dir"])
        print(json.dumps({"data": generate(args["data_dir"], args["seed"]),
                          "audit": audit}, ensure_ascii=False, indent=2))
    elif command == "augment":
        from .augmentation import augment
        print(json.dumps(augment(**args), ensure_ascii=False, indent=2))
    elif command == "postgenerate":
        from .posttrain_data import generate_posttrain
        print(json.dumps(generate_posttrain(**args), ensure_ascii=False, indent=2))
    elif command == 'adaptgenerate':
        from .adaptation_data import generate
        print(json.dumps(generate(**args), ensure_ascii=False, indent=2))
    elif command == 'noisegenerate':
        from .noisy_data import generate
        print(json.dumps(generate(**args), ensure_ascii=False, indent=2))
    elif command == 'noiseevaluate':
        from .noisy_eval import evaluate
        evaluate(**args)
    elif command == 'semanticgenerate':
        from .semantic_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'roleaugment':
        from .role_augmentation import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'rolecompose':
        from .role_composition import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'anchoraugment':
        from .anchor_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'constituentaugment':
        from .constituent_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'paraphraseaugment':
        from .paraphrase_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'relationaugment':
        from .relation_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'contextaugment':
        from .context_algebra import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'neutralaugment':
        from .neutral_tail_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'incidentalaugment':
        from .incidental_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'familyreplay':
        from .family_replay_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'scopeaugment':
        from .scope_data import generate
        print(json.dumps(generate(**args),ensure_ascii=False,indent=2))
    elif command == 'testinventory':
        from .benchmark_inventory import inventory
        inventory(**args)
    elif command == 'weightaverage':
        from .weight_average import average
        print(json.dumps(average(**args),ensure_ascii=False,indent=2))
    elif command == 'neurevaluate':
        from .noisy_eval import comprehensive
        comprehensive(**args)
    elif command == 'selftest':
        from .personal_eval import evaluate
        evaluate(**args)
    elif command == 'adaptcalibrate':
        from .adaptation_eval import calibrate
        calibrate(**args)
    elif command in ('adaptaugment', 'adaptcompose'):
        if command == 'adaptaugment':
            from .adaptation_augment import generate
        else:
            from .adaptation_composition import generate
        print(json.dumps(generate(**args), ensure_ascii=False, indent=2))
    elif command == 'adaptevaluate':
        from .adaptation_eval import evaluate
        evaluate(**args)
    elif command == "train":
        from .training import train
        train(**args)
    elif command == "calibrate":
        from .evaluation import calibrate
        print(json.dumps(calibrate(**args), ensure_ascii=False, indent=2))
    elif command == "evaluate":
        from .evaluation import evaluate
        evaluate(**args)
    elif command == "postevaluate":
        from .posttrain_evaluation import evaluate_posttrain
        evaluate_posttrain(**args)
    elif command == "diagnostics":
        from .diagnostics import diagnostics
        diagnostics(**args)
    else:
        from .harness import FilterHarness
        import torch
        torch.set_num_threads(4)
        text = args.pop("text", None)
        harness = FilterHarness(**args)
        if command in ("predict", 'compare'):
            text = sys.stdin.read() if text is None else text
            if command == 'compare':
                from .compare import compare
                result = compare(harness, text)
            else:
                result = harness.extract(text).to_dict()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("filter1.0：输入操作句，输入 :quit 退出。")
            while True:
                try:
                    text = input("> ")
                except (EOFError, KeyboardInterrupt):
                    break
                if text == ":quit":
                    break
                print(json.dumps(harness.extract(text).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

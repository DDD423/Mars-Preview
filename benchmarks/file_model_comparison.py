"""Raw Chinese -> plan -> real isolated Dave Work execution, paired benchmark.

No model sees gold labels, gold spans, gold workspace locations or answers.
Qwen extracts its own literal bindings; the custom pair uses its real filter.
Both get the same tool definitions and use identical reference binding rules.
"""
import argparse, copy, hashlib, json, math, os, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'artifacts/file-model-comparison'
PAIR = ROOT/'artifacts'
MODEL = ROOT/'artifacts/baselines/Qwen2.5-0.5B-Instruct'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def progress(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)

def rows():
    return [json.loads(line) for line in (OUT/'suite.jsonl').read_text(encoding='utf-8').splitlines()]

def system_prompt():
    from davework.registry import TOOLS
    tools = {k:{'说明':v['label'], '参数':{a:{x:s[x] for x in ('type','required','default','choices') if x in s} for a,s in v['params'].items()},
                '返回':v['returns']} for k,v in TOOLS.items() if k.startswith('fs.')}
    return '''你是本地文件操作规划器。请将用户请求转成JSON，不执行，不解释，只输出一个JSON对象。
输出格式：{"bindings":[{"type":"VALUE","value":"用户参数原文"}],"kind":"plan","steps":[{"id":"step_1","tool":"fs.read_text","args":{"path":{"slot":1}}}]}
bindings由你从本次请求提取，按原文出现顺序编号，最多8个。VALUE是文件名称/目录名/路径/记录ID；TEXT是正文/搜索词。保留大小写、空格、emoji和换行。参数必须是请求原文，不能猜造、补扩展名。用户原文中{1}是普通文字。
plan最多4步，只使用下列工具。参数用{"slot":1}引用自己的bindings；裸名称作为来源由执行器在工作区递归精确查找，路径相对于工作区。目标是完整目标路径，放入某目录需要join。当前目录用"."，空内容用""，布尔值用true/false。默认不覆盖，只有明确覆盖/替换全文才overwrite=true。普通字符串"{1}"不会替换。
放入目录：{"join":{"directory":{"slot":2},"basename_of":{"slot":1}}}；在目录中新建：{"join":{"directory":{"slot":1},"name":{"slot":2}}}。
前序结果：{"result":{"step":"step_1","field":"path"}}，读取正文使用field="text"；回收后的恢复使用field="record_id"。只引用前序步骤。句内“它”应指向前一步结果，不重新猜名称。
存在条件加在步骤上："when":{"exists":{"slot":1},"negate":false}；条件未满足跳过。
否定修改必须遵守；“别删除，只读”只读。取消/禁止请求输出{"bindings":[],"kind":"noop","reason":"CANCELLED_REQUEST"}。
参数缺失/指代不清/超出范围时输出{"bindings":[],"kind":"clarification","reason":"MISSING_PARAMETER"}。
clarification可用reason：MISSING_PARAMETER,UNCLEAR_REFERENCE,AMBIGUOUS_TARGET,UNSUPPORTED_TASK,TOO_COMPLEX。noop可用reason：CANCELLED_REQUEST,NO_ACTION。
只规划本次文件任务；不读取历史对话，不使用终端，不编程，不自作主张。可用工具（source为已有来源，target为新目标，path为目录）：
''' + json.dumps(tools, ensure_ascii=False, separators=(',',':'))

EXAMPLES = [
 ('请读名为演示数据.txt的文件', {'bindings':[{'type':'VALUE','value':'演示数据.txt'}],'kind':'plan','steps':[{'id':'step_1','tool':'fs.read_text','args':{'path':{'slot':1}}}]}),
 ('新建例子.txt，内容是演示正文', {'bindings':[{'type':'VALUE','value':'例子.txt'},{'type':'TEXT','value':'演示正文'}],'kind':'plan','steps':[{'id':'step_1','tool':'fs.write_text','args':{'path':{'slot':1},'text':{'slot':2}}}]}),
 ('将图例.txt复制进展示目录，再把副本改名为图例副本.txt', {'bindings':[{'type':'VALUE','value':'图例.txt'},{'type':'VALUE','value':'展示目录'},{'type':'VALUE','value':'图例副本.txt'}],'kind':'plan','steps':[{'id':'step_1','tool':'fs.copy','args':{'source':{'slot':1},'destination':{'join':{'directory':{'slot':2},'basename_of':{'slot':1}}}}},{'id':'step_2','tool':'fs.rename','args':{'source':{'result':{'step':'step_1','field':'path'}},'new_name':{'slot':3}}}]}),
 ('把之前说的那个文件删了', {'bindings':[],'kind':'clarification','reason':'UNCLEAR_REFERENCE'}),
 ('不要删除演示数据.txt', {'bindings':[],'kind':'noop','reason':'CANCELLED_REQUEST'}),
]

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    for filename in ('suite.jsonl','prompt.json'):
        source=ROOT/'benchmarks/data'/filename
        target=OUT/filename
        if target.exists():assert target.read_bytes()==source.read_bytes()
        else:target.write_bytes(source.read_bytes())
    info={'product':'Mars Preview','rows':620,'suite_sha256':sha(OUT/'suite.jsonl'),'prompt_sha256':sha(OUT/'prompt.json'),
          'filter_sha256':sha(PAIR/'filter1.0.pt'),'translator_sha256':sha(PAIR/'translator1.0.pt'),
          'protocol':'Frozen 620 requests; no tuning or retries; same original prompt; raw input -> own bindings -> common real file execution.'}
    write(OUT/'MANIFEST.json',info)
    progress(event='frozen',rows=620)

def predict_pair():
    import torch
    from joint10.evaluation import extract_rows, requests
    from translator10.training import load_checkpoint, predict_rows
    from translator10.inference import confidence, joint_features
    torch.set_num_threads(4);data=rows();start=time.perf_counter()
    extraction,good=extract_rows(data,PAIR/'filter1.0.pt','cuda')
    real=requests(data,extraction);model,ck=load_checkpoint(PAIR/'translator1.0.pt','cuda')
    assert ck['calibration']['filter_sha256']==sha(PAIR/'filter1.0.pt')
    preds=predict_rows(model,real,'cuda',progress=lambda n,t:progress(event='pair_translator',seen=n,total=t))
    results=[]
    for r,x,p,inp,boundary in zip(data,extraction,preds,real,good):
        score=confidence(joint_features(p,inp['request']),ck['calibration'])
        usable=x['status'] not in ('error','unsupported','unsupported_length','uncertain','needs_context')
        accept=usable and p['plan'] is not None and (p['plan']['kind']!='plan' or score>=ck['calibration']['threshold'])
        results.append({'id':r['comparison_id'],'extraction':x,'plan':p['plan'],'confidence':score,'accept_before_preview':accept,'filter_boundaries_exact':boundary})
    write(OUT/'pair_predictions.json',results)
    write(OUT/'pair_runtime.json',{'seconds':time.perf_counter()-start,'parameters':270488+sum(p.numel() for p in model.parameters()),'device':torch.cuda.get_device_name(0)})
    progress(event='pair_predictions_complete',rows=len(results))

def decode_qwen(raw, response):
    from translator10.language import nonexecution, REASONS
    # Markdown fence stripping is syntax-only. No guessed JSON corrections.
    text=response.strip()
    if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1]).strip()
    obj=json.loads(text)
    if not isinstance(obj,dict):raise ValueError('Output must be an object')
    kind=obj.get('kind');allowed={'bindings','kind','steps'} if kind=='plan' else {'bindings','kind','reason'}
    if set(obj)!=allowed:raise ValueError('Invalid output fields')
    if not isinstance(obj['bindings'],list) or len(obj['bindings'])>8:raise ValueError('Invalid bindings')
    bindings={}
    for i,b in enumerate(obj['bindings'],1):
        if not isinstance(b,dict) or set(b)!={'type','value'} or b['type'] not in ('VALUE','NAME','PATH','TEXT') or not isinstance(b['value'],str):raise ValueError('Invalid binding')
        value=b['value']
        if not value or value not in raw:raise ValueError('Binding is not literal request text')
        start=raw.find(value);bindings['{'+str(i)+'}']={'type':b['type'],'value':value,'start':start,'end':start+len(value)}
    extraction={'original':raw,'normalized':raw,'status':'ok','bindings':bindings,'spans':[],'segments':[{'kind':'literal','text':raw}]}
    if kind=='plan':
        if not isinstance(obj['steps'],list) or not 1<=len(obj['steps'])<=4:raise ValueError('Invalid steps')
        plan={'protocol':'davework/1','context_id':'ctx_eval','kind':'plan','steps':obj['steps']}
        if any(s.get('tool') not in __import__('translator10.language',fromlist=['FILE_TOOLS']).FILE_TOOLS for s in plan['steps']):raise ValueError('Tool not enabled')
    elif kind in ('clarification','noop'):
        reason=obj['reason']
        allowed_reasons={'MISSING_PARAMETER','UNCLEAR_REFERENCE','AMBIGUOUS_TARGET','UNSUPPORTED_TASK','TOO_COMPLEX'} if kind=='clarification' else {'CANCELLED_REQUEST','NO_ACTION'}
        if reason not in allowed_reasons:raise ValueError('Invalid reason')
        plan=nonexecution(kind,reason)
    else:raise ValueError('Invalid kind')
    return extraction,plan

def predict_qwen():
    sys.path.insert(0,str(ROOT/'artifacts/comparison-runtime'))
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(4);torch.manual_seed(100505)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False,attn_implementation='sdpa').to('cuda').eval()
    prompt=json.loads((OUT/'prompt.json').read_text(encoding='utf-8'));base=[{'role':'system','content':prompt['system']}]
    for question,answer in prompt['examples']:base += [{'role':'user','content':question},{'role':'assistant','content':json.dumps(answer,ensure_ascii=False,separators=(',',':'))}]
    data=rows();file=OUT/'qwen_predictions.jsonl';done=[]
    if file.exists():done=[json.loads(line) for line in file.read_text(encoding='utf-8').splitlines()]
    start=time.perf_counter();tokens=0
    with file.open('a',encoding='utf-8') as output:
        for begin in range(len(done),len(data),12):
            batch=data[begin:begin+12]
            texts=[tokenizer.apply_chat_template(base+[{'role':'user','content':r['raw']}],tokenize=False,add_generation_prompt=True) for r in batch]
            inputs=tokenizer(texts,return_tensors='pt',padding=True).to('cuda')
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=512,do_sample=False,num_beams=1,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
            for r,g in zip(batch,generated[:,inputs.input_ids.shape[1]:]):
                response=tokenizer.decode(g,skip_special_tokens=True)
                # Generation uses EOS for padding; tokenizer.pad_token_id differs.
                ends=(g==tokenizer.eos_token_id).nonzero()
                length=int(ends[0]) if len(ends) else len(g);tokens+=length
                record={'id':r['comparison_id'],'response':response,'generated_tokens':length,'batch_seconds':elapsed,'truncated':length>=512}
                try:
                    record['extraction'],record['plan']=decode_qwen(r['raw'],response);record['parse_error']=None
                except Exception as exc:record.update(extraction=None,plan=None,parse_error=str(exc))
                output.write(json.dumps(record,ensure_ascii=False)+'\n')
            output.flush();progress(event='qwen',seen=begin+len(batch),total=len(data),batch_seconds=round(elapsed,2))
    write(OUT/'qwen_runtime.json',{'seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'dtype':'bfloat16','generated_tokens_this_run':tokens,'device':torch.cuda.get_device_name(0),'download_manifest':json.loads((MODEL/'DOWNLOAD_MANIFEST.json').read_text())})

def timing_pair():
    """Rerun only the custom pair after Qwen finishes, with load excluded.

    Does not replace original predictions or select checkpoints. Asserts exact
    reproduction of the original plans and extraction boundaries.
    """
    import torch
    from filter10 import FilterHarness
    from filter10.harness import render_result
    from joint10.evaluation import requests
    from translator10.training import load_checkpoint,predict_rows
    torch.set_num_threads(4);data=rows();load_start=time.perf_counter()
    f=FilterHarness(PAIR/'filter1.0.pt','cuda');model,ck=load_checkpoint(PAIR/'translator1.0.pt','cuda')
    torch.cuda.synchronize();load_seconds=time.perf_counter()-load_start
    start=time.perf_counter();extractions=[]
    for begin in range(0,len(data),32):
        batch=data[begin:begin+32]
        scores=f.model.all_scores([r['raw'] for r in batch],'cuda',with_counts=f.learned_count,with_evidence=f.neural_only)
        for row,item in zip(batch,scores):
            pairs,logits=item[:2];count=item[2] if f.learned_count else None
            evidence=item[3 if f.learned_count else 2] if f.neural_only else None
            status,spans,proposals=f.decode_scores(row['raw'],pairs,logits,count_logits=count,neural_evidence=evidence)
            out=render_result(row['raw'],status,spans,proposals,refine_paths=not f.neural_only).to_dict();out['checkpoint_sha256']=f.checkpoint_sha256
            extractions.append(out)
    torch.cuda.synchronize();filter_seconds=time.perf_counter()-start
    real=requests(data,extractions);trans_start=time.perf_counter();preds=predict_rows(model,real,'cuda')
    torch.cuda.synchronize();translator_seconds=time.perf_counter()-trans_start;total=time.perf_counter()-start
    original=json.loads((OUT/'pair_predictions.json').read_text(encoding='utf-8'))
    assert [r['plan'] for r in original]==[p['plan'] for p in preds]
    assert [[(s['start'],s['end'],s['type']) for s in r['extraction']['spans']] for r in original]==[[(s['start'],s['end'],s['type']) for s in x['spans']] for x in extractions]
    assert [r['extraction']['bindings'] for r in original]==[x['bindings'] for x in extractions]
    info=json.loads((OUT/'pair_runtime.json').read_text(encoding='utf-8'))
    info.update(loaded_rerun_seconds=total,loaded_rerun_filter_seconds=filter_seconds,loaded_rerun_translator_seconds=translator_seconds,
        measured_loading_seconds=load_seconds,amortized_seconds_per_request=total/len(data),timing_repeat_identical_predictions=True)
    write(OUT/'pair_runtime.json',info)
    progress(event='pair_timing_complete',timing=info)

def fixture(row, root):
    from joint10.evaluation import setup_fixture
    setup_fixture(row,root)
    (root/'无关哨兵.txt').write_text('无关文件应保持不变',encoding='utf-8')
    for step in row['target'].get('steps',[]):
        if step['tool']=='fs.search':
            query=step['args']['query'];value=row['bindings']['{'+str(query['slot'])+'}']['value'] if isinstance(query,dict) and 'slot' in query else query
            folder=root/'检索资料';folder.mkdir(exist_ok=True)
            if step['args'].get('mode','name')=='name' and not any(c in value for c in '\\/:*?"<>|') and not any(ord(c)<32 for c in value):
                (folder/(value+'-匹配.txt')).write_text('检索样本',encoding='utf-8')
            else:(folder/'内容匹配.txt').write_text('前缀 '+value+' 后缀',encoding='utf-8')

def stable_plan(plan):
    p=copy.deepcopy(plan)
    if p['kind']!='plan':return p
    ids={s['id']:'step_'+str(i) for i,s in enumerate(p['steps'],1)}
    def walk(v):
        if isinstance(v,dict):
            if set(v)=={'result'}:v['result']['step']=ids[v['result']['step']]
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(p)
    for s in p['steps']:s['id']=ids[s['id']]
    return p

def run(row, extraction, plan, root):
    from davework.runtime import Runtime
    from translator10.language import FILE_TOOLS
    from joint10.execution import snapshot, normalize
    fixture(row,root);rt=Runtime(root/'.state')
    try:
        rt.config.update(workspace=str(root),excludes=['.state'],enabled_tools=list(FILE_TOOLS))
        ctx=rt.context('',extract=False);cid=ctx['context_id'];rt.contexts[cid]['filter']=extraction
        p=stable_plan(plan);p['context_id']=cid
        pre=rt.preview(p)
        if p['kind']!='plan':return {'status':p['kind'],'files':snapshot(root)},True,None
        # Remove volatile snapshot metadata and normalize absolute child roots.
        semantic=normalize([{k:s[k] for k in ('id','tool','args','when')} for s in pre['steps']],root)
        tid=rt.execute(pre['preview_id'],pre['digest'])['task_id'];deadline=time.monotonic()+15
        while rt.events(tid)['status']=='running':
            if time.monotonic()>deadline:rt.cancel(tid);raise TimeoutError('Fixture deadline exceeded')
            time.sleep(.005)
        task=rt.tasks[tid]
        observed={'status':task['status'],'files':snapshot(root),'results':normalize(task['results'],root),
                  'errors':[e['data'].get('code') for e in task['events'] if e['kind']=='error']}
        return observed,True,semantic
    finally:rt.close()

def audit_gold():
    """Check the frozen fixtures while the GPU baseline runs independently."""
    from filter10.harness import render_result
    parent=OUT/'gold-workspaces';parent.mkdir(exist_ok=True);observations=[]
    for row in rows():
        record={'id':row['comparison_id']}
        with tempfile.TemporaryDirectory(prefix='case-',dir=parent) as name:
            root=Path(name).resolve();assert root.parent==parent.resolve()
            gold=render_result(row['raw'],'ok',row['spans'],refine_paths=False).to_dict()
            try:
                expected,valid,semantic=run(row,gold,row['target'],root)
                record.update(gold_valid=expected['status'] in ('completed','noop','clarification'),gold_observation=expected,gold_semantic=semantic)
            except Exception as exc:record.update(gold_valid=False,gold_error=str(exc),gold_observation=None,gold_semantic=None)
        observations.append(record)
        if len(observations)%100==0:progress(event='gold_fixture_audit',seen=len(observations),failures=sum(not x['gold_valid'] for x in observations))
    write(OUT/'gold_observations.json',observations)
    progress(event='gold_fixture_audit_complete',rows=len(observations),failures=sum(not x['gold_valid'] for x in observations))

def score():
    from filter10.harness import render_result
    data=rows();pair=json.loads((OUT/'pair_predictions.json').read_text(encoding='utf-8'))
    qwen=[json.loads(line) for line in (OUT/'qwen_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(pair)==len(qwen)==len(data)
    cached_gold=json.loads((OUT/'gold_observations.json').read_text(encoding='utf-8')) if (OUT/'gold_observations.json').exists() else None
    parent=OUT/'workspaces';parent.mkdir(exist_ok=True);checks=[]
    for i,(r,a,b) in enumerate(zip(data,pair,qwen)):
        assert a['id']==b['id']==i
        item={'id':i,'raw':r['raw'],'group':r['comparison_group'],'category':r['category'],'expected':r['target'],'expected_kind':r['target']['kind'],'models':{}}
        with tempfile.TemporaryDirectory(prefix='case-',dir=parent) as name:
            lab=Path(name).resolve();assert lab.parent==parent.resolve()
            if cached_gold is not None:
                entry=cached_gold[i];assert entry['id']==i
                item.update({k:v for k,v in entry.items() if k.startswith('gold_') and k!='gold_semantic'})
                expected=entry['gold_observation'];semantic=entry['gold_semantic']
            else:
                gold_root=lab/'gold';gold_root.mkdir()
                gold=render_result(r['raw'],'ok',r['spans'],refine_paths=False).to_dict()
                try:
                    expected,valid,semantic=run(r,gold,r['target'],gold_root)
                    item['gold_valid']=expected['status'] in ('completed','noop','clarification');item['gold_observation']=expected
                except Exception as exc:item.update(gold_valid=False,gold_error=str(exc));expected=None;semantic=None
            for label,record in [('pair',a),('qwen',b)]:
                detail={'proposal':record['plan'],'parse_error':record.get('parse_error'),'preview_valid':False,'execution_correct':False,'bound_plan_correct':False,'accepted':False,'file_check_seconds':0.}
                if label=='qwen':
                    raw_json=record['response'].strip()
                    if raw_json.startswith('```') and raw_json.endswith('```'):raw_json='\n'.join(raw_json.splitlines()[1:-1]).strip()
                    try:json.loads(raw_json);detail['json_syntax_valid']=True
                    except json.JSONDecodeError:detail['json_syntax_valid']=False
                    error=record.get('parse_error') or ''
                    detail['output_failure_kind']=('json_syntax' if not detail['json_syntax_valid'] else 'nonliteral_binding' if 'literal request text' in error else 'output_contract_or_binding_type' if error else None)
                else:detail['json_syntax_valid']=record['plan'] is not None;detail['output_failure_kind']=None if record['plan'] is not None else 'generation'
                if record['plan'] is not None and record['extraction'] is not None:
                    root=lab/label;root.mkdir()
                    tick=time.perf_counter()
                    try:
                        observed,valid,actual_semantic=run(r,record['extraction'],record['plan'],root)
                        detail.update(observation=observed,preview_valid=valid)
                        # Query/mode are part of search semantics, even when an
                        # impossible filename query legitimately matches zero files.
                        expected_search=[s['args'] for s in (semantic or []) if s['tool']=='fs.search']
                        actual_search=[s['args'] for s in (actual_semantic or []) if s['tool']=='fs.search']
                        detail['execution_correct']=bool(item['gold_valid'] and observed==expected and actual_search==expected_search)
                        detail['bound_plan_correct']=bool(item['gold_valid'] and record['plan']['kind']==r['target']['kind'] and (actual_semantic==semantic if r['target']['kind']=='plan' else record['plan'].get('message')==r['target'].get('message')))
                        detail['accepted']=bool(valid and record['plan']['kind']=='plan' and (record['accept_before_preview'] if label=='pair' else True))
                    except Exception as exc:detail['execution_error']=str(exc)
                    finally:detail['file_check_seconds']=time.perf_counter()-tick
                detail['deployed_correct']=bool(detail['execution_correct'] and (r['target']['kind']!='plan' or detail['accepted']))
                item['models'][label]=detail
        checks.append(item)
        if (i+1)%40==0:
            write(OUT/'checks.partial.json',checks);progress(event='execution_scoring',seen=i+1,total=len(data))
    write(OUT/'checks.json',checks)
    metrics={}
    for group in ('all','fixed580','authored40'):
        chosen=[r for r in checks if group=='all' or r['group']==group];m={'rows':len(chosen),'gold_fixture_failures':sum(not r['gold_valid'] for r in chosen)}
        for label in ('pair','qwen'):
            supported=[r for r in chosen if r['expected_kind']=='plan'];accepted=[r for r in chosen if r['models'][label]['accepted']]
            details=[r['models'][label] for r in chosen]
            cm={'ungated_execution_correct':sum(x['execution_correct'] for x in details),'deployed_execution_correct':sum(x['deployed_correct'] for x in details),
                'strict_bound_plan_correct':sum(x['bound_plan_correct'] for x in details),'accepted_plans':len(accepted),
                'accepted_correct':sum(r['models'][label]['execution_correct'] for r in accepted),'supported_rows':len(supported),
                'supported_ungated_execution_correct':sum(r['models'][label]['execution_correct'] for r in supported),
                'supported_deployed_execution_correct':sum(r['models'][label]['deployed_correct'] for r in supported),
                'wrong_accepted_modifying':sum(not r['models'][label]['execution_correct'] and any(s['tool'] not in ('fs.list','fs.read_text','fs.search') for s in r['models'][label]['proposal']['steps']) for r in accepted),
                'nonexecution_wrong_accepted_modifying':sum(r['expected_kind']!='plan' and any(s['tool'] not in ('fs.list','fs.read_text','fs.search') for s in r['models'][label]['proposal']['steps']) for r in accepted),
                'parse_or_generation_failure':sum(x['proposal'] is None for x in details),'preview_or_execution_errors':sum(bool(x.get('execution_error')) for x in details)}
            syntactic_valid=sum(x['proposal'] is not None for x in details)
            cm['parsed_output_rows']=syntactic_valid
            cm['execution_accuracy_among_parsed_outputs']=cm['ungated_execution_correct']/syntactic_valid if syntactic_valid else None
            cm['json_syntax_errors']=sum(not x['json_syntax_valid'] for x in details)
            cm['nonliteral_binding_errors']=sum(x['output_failure_kind']=='nonliteral_binding' for x in details)
            cm['other_output_contract_errors']=sum(x['output_failure_kind']=='output_contract_or_binding_type' for x in details)
            cm['nonexecution_rows']=len(chosen)-len(supported)
            cm['nonexecution_correct']=sum(r['expected_kind']!='plan' and r['models'][label]['execution_correct'] for r in chosen)
            cm['file_check_seconds']=sum(x['file_check_seconds'] for x in details)
            cm.update(ungated_execution_accuracy=cm['ungated_execution_correct']/len(chosen),deployed_execution_accuracy=cm['deployed_execution_correct']/len(chosen),
                strict_bound_plan_accuracy=cm['strict_bound_plan_correct']/len(chosen),accepted_precision=cm['accepted_correct']/len(accepted) if accepted else None,
                supported_coverage=sum(r['models'][label]['accepted'] for r in supported)/len(supported))
            cm['categories']={c:{'total':sum(r['category']==c for r in chosen),'correct':sum(r['category']==c and r['models'][label]['execution_correct'] for r in chosen)} for c in sorted({r['category'] for r in chosen})}
            m[label]=cm
        m['paired_ungated']={'both_correct':sum(r['models']['pair']['execution_correct'] and r['models']['qwen']['execution_correct'] for r in chosen),
            'pair_only':sum(r['models']['pair']['execution_correct'] and not r['models']['qwen']['execution_correct'] for r in chosen),
            'qwen_only':sum(r['models']['qwen']['execution_correct'] and not r['models']['pair']['execution_correct'] for r in chosen),
            'both_wrong':sum(not r['models']['pair']['execution_correct'] and not r['models']['qwen']['execution_correct'] for r in chosen)}
        metrics[group]=m
    write(OUT/'metrics.json',metrics);progress(event='complete',metrics=metrics)

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('freeze','pair','qwen','score','timing-pair','gold'));a=p.parse_args()
    {'freeze':freeze,'pair':predict_pair,'qwen':predict_qwen,'score':score,'timing-pair':timing_pair,'gold':audit_gold}[a.phase]()

if __name__=='__main__':main()

"""Compare final file state and observations against gold in isolated labs."""
import copy,hashlib,json,tempfile,time
from pathlib import Path
from .evaluation import setup_fixture
from davework.runtime import Runtime
from filter10.harness import render_result

def snapshot(root):
    files={}
    for p in sorted(root.rglob('*')):
        relative=p.relative_to(root)
        if relative.parts[0] in ('.state','.davework-trash'):continue
        files[str(relative)]='directory' if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    return files

def normalize(value,root):
    if isinstance(value,str):return value.replace(str(root),'<WORKSPACE>')
    if isinstance(value,list):return [normalize(v,root) for v in value]
    if isinstance(value,dict):return {k:normalize(v,root) for k,v in value.items() if k not in ('record_id','backup_id')}
    return value

def execute_fixture(row,extraction,plan,root):
    setup_fixture(row,root);rt=Runtime(root/'.state')
    try:
        rt.config.update(workspace=str(root),excludes=['.state'])
        ctx=rt.context('',extract=False);cid=ctx['context_id'];rt.contexts[cid]['filter']=extraction
        request=copy.deepcopy(plan);request['context_id']=cid
        if request['kind']!='plan':return {'status':request['kind'],'files':snapshot(root)}
        pre=rt.preview(request);tid=rt.execute(pre['preview_id'],pre['digest'])['task_id'];deadline=time.monotonic()+15
        while rt.events(tid)['status']=='running':
            if time.monotonic()>deadline:rt.cancel(tid);raise TimeoutError('File fixture timed out')
            time.sleep(.01)
        task=rt.tasks[tid]
        return {'status':task['status'],'files':snapshot(root),'results':normalize(task['results'],root),'errors':[e['data'].get('code') for e in task['events'] if e['kind']=='error']}
    finally:rt.close()

def execution_check(row,extraction,plan):
    parent=Path('.davework-test-joint-execution').resolve();parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case-',dir=parent) as name:
        lab=Path(name).resolve();assert lab.parent==parent and lab!=parent
        expected_root=lab/'gold';actual_root=lab/'actual';expected_root.mkdir();actual_root.mkdir()
        gold=render_result(row['raw'],'ok',row['spans'],refine_paths=False).to_dict()
        expected=execute_fixture(row,gold,row['target'],expected_root)
        actual=execute_fixture(row,extraction,plan,actual_root)
        return expected==actual and expected['status'] in ('completed','noop','clarification'),{'expected':expected,'actual':actual}

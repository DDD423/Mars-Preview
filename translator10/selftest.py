"""Real filter + translator + preview + executor in disposable project fixtures."""
import json
import tempfile
import time
from pathlib import Path
from davework.runtime import Runtime

def selftest(checkpoint="artifacts/translator1.0.pt",device="cpu"):
    parent=Path(".davework-test-translator-selftest").resolve();parent.mkdir(exist_ok=True)
    records=[]
    cases=[
      ("rename","把叫做草稿的文件改名为最终版"),
      ("read","读取叫做草稿的文件"),
      ("empty","创建叫做新文件的空文件"),
      ("write","把文字新内容写入新文件新文件"),
      ("append","向文件草稿追加文字新内容"),
      ("copy","复制文件草稿到完整路径./副本"),
      ("move","移动文件草稿到完整路径./移后"),
      ("copy_into","复制文件草稿到目录目标目录里面"),
      ("trash","删除叫做草稿的文件"),
      ("restore","回收文件草稿，再根据删除记录恢复它"),
      ("reference","把文件草稿拷贝为./副本，然后把它改名叫最终版"),
      ("condition","如果路径./不存在存在，就读取文件./不存在"),
      ("negation","不要删除文件草稿，只读取文件草稿的内容"),
    ]
    for name,text in cases:
        # These fixtures remain in the project for review; no recursive deletion.
        root=Path(tempfile.mkdtemp(prefix=name+"-",dir=parent));(root/"目标目录").mkdir()
        # Explicit bytes avoid Windows write_text translating LF to CRLF.
        # The executor intentionally preserves the on-disk newline bytes.
        body="原文 😀 {1}\n第二行";(root/"草稿").write_bytes(body.encode("utf-8"))
        runtime=Runtime(root/".state");runtime.configure({"workspace":str(root),"translator_checkpoint":str(Path(checkpoint).resolve()),"translator_device":device,"device":device,"excludes":[".state"]})
        row={"case":name,"text":text,"workspace":str(root)}
        try:
            ctx=runtime.context(text);row["filter"]=ctx["filter"];row["plan"]=ctx["planner"]
            if ctx["planner"]["kind"]!="plan":row.update(success=False,reason="planner_refused")
            else:
                pre=runtime.preview(ctx["planner"]);row["preview"]=pre
                assert (root/"草稿").read_text(encoding="utf-8")==body,"Preview mutated a file"
                tid=runtime.execute(pre["preview_id"],pre["digest"])["task_id"];deadline=time.time()+10
                while runtime.events(tid)["status"]=="running":
                    if time.time()>deadline:raise RuntimeError("selftest timeout")
                    time.sleep(.02)
                task=runtime.tasks[tid];row["task_status"]=task["status"];row["events"]=task["events"]
                if task["status"]!="completed":row.update(success=False,reason="execution_failed")
                else:
                    checks={
                      "rename":lambda:(root/"最终版").read_text(encoding="utf-8")==body and not (root/"草稿").exists(),
                      "read":lambda:any(v.get("text")==body for v in task["results"].values()),
                      "empty":lambda:(root/"新文件").read_bytes()==b"",
                      "write":lambda:(root/"新文件").read_text(encoding="utf-8")=="新内容",
                      "append":lambda:(root/"草稿").read_text(encoding="utf-8")==body+"新内容",
                      "copy":lambda:(root/"副本").read_text(encoding="utf-8")==body and (root/"草稿").exists(),
                      "move":lambda:(root/"移后").read_text(encoding="utf-8")==body and not (root/"草稿").exists(),
                      "copy_into":lambda:(root/"目标目录"/"草稿").read_text(encoding="utf-8")==body,
                      "trash":lambda:not (root/"草稿").exists() and any("record_id" in v for v in task["results"].values()),
                      "restore":lambda:(root/"草稿").read_text(encoding="utf-8")==body and len(task["results"])==2,
                      "reference":lambda:(root/"最终版").read_text(encoding="utf-8")==body and (root/"草稿").exists() and not (root/"副本").exists(),
                      "condition":lambda:not task["results"] and any(e["kind"]=="step_skipped" for e in task["events"]),
                      "negation":lambda:(root/"草稿").read_text(encoding="utf-8")==body and any(v.get("text")==body for v in task["results"].values()),
                    }
                    row["success"]=bool(checks[name]());row["reason"]="checked_final_state"
        except Exception as exc:row.update(success=False,reason=str(exc))
        finally:runtime.close()
        records.append(row);print(json.dumps({"case":name,"success":row["success"],"reason":row["reason"]},ensure_ascii=False),flush=True)
    result={"total":len(records),"successful":sum(r["success"] for r in records),"cases":records,"note":"13 development smoke cases, not the frozen independent test and not a general accuracy estimate"}
    out=Path("artifacts/translator-evaluation");out.mkdir(exist_ok=True)
    (out/"execution-smoke.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

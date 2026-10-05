"""Finite typed plan language; no Chinese parsing or action inference here."""
import copy
import hashlib
import json
from davework.registry import TOOLS

FILE_TOOLS = tuple(sorted(k for k in TOOLS if k.startswith("fs.")))
REASONS = {
    "AMBIGUOUS_TARGET": "请说明目标是完整文件路径还是某个目录。",
    "MISSING_PARAMETER": "请补充明确的文件、路径或内容参数。",
    "UNCLEAR_REFERENCE": "请在本次请求中明确说明所指的文件，首版不读取之前的对话。",
    "UNSUPPORTED_TASK": "首版只规划文件操作，不处理终端、编程或其他电脑任务。",
    "TOO_COMPLEX": "首版最多支持 4 个步骤和 8 个参数，请拆分请求。",
    "LOW_CONFIDENCE": "模型对该请求没有足够把握，请换一种明确的说法。",
    "NO_ACTION": "没有需要执行的文件操作。",
    "CANCELLED_REQUEST": "已遵守取消或不执行要求，不进行操作。",
}
CONSTANTS = {"DOT": ".", "EMPTY": "", "UTF8": "utf-8", "GB": "gb18030", "NAME_MODE": "name", "TEXT_MODE": "text", "TRUE": True, "FALSE": False}
TOKENS = ["PAD", "BOS", "EOS", "PLAN", "CLARIFY", "NOOP", "NEXT", "END", "WHEN", "NO_WHEN", "JOIN_NAME", "JOIN_BASE"]
TOKENS += list(REASONS) + list(CONSTANTS) + ["S"+str(i) for i in range(1,9)]
TOKENS += list(FILE_TOOLS) + ["K:"+k for k in sorted({p for t in FILE_TOOLS for p in TOOLS[t]["params"]})]
TOKENS += ["R%d:%s" % (i, field) for i in range(1,5) for field in sorted({f for t in FILE_TOOLS for f in TOOLS[t]["returns"]})]
IDS = {t:i for i,t in enumerate(TOKENS)}


def fingerprint():
    return hashlib.sha256(json.dumps({"tools":[TOOLS[k] for k in FILE_TOOLS],"tokens":TOKENS},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def nonexecution(kind, reason, context_id="ctx_eval"):
    return {"protocol":"davework/1","kind":kind,"context_id":context_id,"message":REASONS[reason]}


def arg_tokens(value):
    if isinstance(value, dict):
        if "slot" in value:
            return ["S"+str(value["slot"])]
        if "result" in value:
            ref=value["result"]
            return ["R%s:%s" % (ref["step"].split("_")[-1],ref["field"])]
        join=value["join"]
        return ["JOIN_NAME" if "name" in join else "JOIN_BASE"] + arg_tokens(join["directory"]) + arg_tokens(join.get("name",join.get("basename_of")))
    for token, constant in CONSTANTS.items():
        if type(value) is type(constant) and value == constant:
            return [token]
    raise ValueError("Non-symbolic argument in target: " + repr(value))


def serialize(plan):
    if plan["kind"] != "plan":
        reason=next(k for k,v in REASONS.items() if v==plan["message"])
        return [IDS["BOS"],IDS["CLARIFY" if plan["kind"]=="clarification" else "NOOP"],IDS[reason],IDS["EOS"]]
    output=["BOS","PLAN"]
    for i, step in enumerate(plan["steps"]):
        if i: output.append("NEXT")
        tool=step["tool"]; output.append(tool)
        for key, spec in TOOLS[tool]["params"].items():
            output.append("K:"+key)
            output += arg_tokens(step["args"].get(key,copy.deepcopy(spec.get("default"))))
        if "when" in step:
            output.append("WHEN"); output+=arg_tokens(step["when"]["exists"])
            output+=arg_tokens(step["when"].get("negate",False))
        else: output.append("NO_WHEN")
        output.append("END")
    output.append("EOS")
    return [IDS[t] for t in output]


def canonical(plan):
    p=copy.deepcopy(plan); p.pop("context_id",None)
    if p["kind"]=="plan":
        ids={s["id"]:"step_"+str(i+1) for i,s in enumerate(p["steps"])}
        def walk(v):
            if isinstance(v,dict):
                if "result" in v: v["result"]["step"]=ids[v["result"]["step"]]
                for x in v.values(): walk(x)
            elif isinstance(v,list):
                for x in v: walk(x)
        for s in p["steps"]:
            s["id"]=ids[s["id"]]
            for k,spec in TOOLS[s["tool"]]["params"].items(): s["args"].setdefault(k,copy.deepcopy(spec.get("default")))
        walk(p)
    return json.dumps(p,sort_keys=True,ensure_ascii=False)


class Grammar:
    def __init__(self, request):
        self.slots={s["slot"]:s["type"] for s in request["slots"]}
        available=request.get("enabled_tools")
        if available is None: available=FILE_TOOLS
        self.available=[k for k in FILE_TOOLS if k in available]
        self.state="kind"; self.steps=[]; self.current=None; self.keys=[]; self.key_index=0
        self.stack=[]; self.pending=None; self.kind=None; self.reason=None

    def allowed(self):
        state=self.state
        if state=="kind": return [IDS[t] for t in ("PLAN","CLARIFY","NOOP")]
        if state=="reason": return [IDS[t] for t in (list(REASONS)[:6] if self.kind=="clarification" else ["NO_ACTION","CANCELLED_REQUEST"])]
        if state=="eos": return [IDS["EOS"]]
        if state=="done": return [IDS["PAD"]]
        if state=="tool": return [IDS[t] for t in self.available]
        if state=="key": return [IDS["K:"+self.keys[self.key_index]]]
        if state=="condition": return [IDS["WHEN"],IDS["NO_WHEN"]]
        if state=="end": return [IDS["END"]]
        if state=="next": return [IDS["EOS"]]+([IDS["NEXT"]] if len(self.steps)<4 else [])
        spec=self.pending["spec"]; kind=spec["type"]
        tokens=[]
        if kind=="boolean": return [IDS["TRUE"],IDS["FALSE"]]
        if spec.get("choices"):
            return [IDS[k] for k,v in CONSTANTS.items() if type(v)==str and v in spec["choices"]]
        for i,t in self.slots.items():
            if 1<=i<=8 and not (kind in ("source","target","path","directory","name") and t=="TEXT") and not (kind=="name" and t=="PATH"):
                tokens.append("S"+str(i))
        for i,previous in enumerate(self.steps,1):
            for field, typ in TOOLS[previous["tool"]]["returns"].items():
                if typ=="string" and kind!="name" and (kind not in ("source","target","path","directory") or field=="path"):
                    tokens.append("R%d:%s"%(i,field))
        if kind in ("path","directory") or self.pending.get("key")=="path" and self.current["tool"]=="fs.list": tokens.append("DOT")
        if kind=="string" and self.pending.get("key")=="text": tokens.append("EMPTY")
        if kind in ("target","path") and not self.stack: tokens += ["JOIN_NAME","JOIN_BASE"]
        return [IDS[t] for t in tokens]

    def start_value(self, spec, dest, key=None):
        self.pending={"spec":spec,"dest":dest,"key":key}; self.state="value"

    def complete_value(self, value):
        if self.stack:
            join=self.stack[-1]
            if "directory" not in join["value"]:
                join["value"]["directory"]=value
                self.pending={"spec":{"type":"name" if join["mode"]=="name" else "source"},"dest":None}
                return
            join["value"][join["mode"]]=value
            self.stack.pop(); value={"join":join["value"]}; self.pending=join["parent"]
        dest=self.pending["dest"]
        if dest=="arg":
            self.current["args"][self.pending["key"]]=value; self.key_index+=1
            self.state="key" if self.key_index<len(self.keys) else "condition"
        elif dest=="exists":
            self.current["when"]={"exists":value}
            self.start_value({"type":"boolean"},"negate")
        else:
            self.current["when"]["negate"]=value; self.state="end"

    def advance(self, token):
        if token not in self.allowed(): raise ValueError("Invalid plan-language transition: "+TOKENS[token])
        t=TOKENS[token]; state=self.state
        if state=="kind":
            self.kind={"PLAN":"plan","CLARIFY":"clarification","NOOP":"noop"}[t]
            self.state="tool" if t=="PLAN" else "reason"
        elif state=="reason": self.reason=t; self.state="eos"
        elif state in ("eos","next") and t=="EOS": self.state="done"
        elif state=="next": self.state="tool"
        elif state=="tool":
            self.current={"id":"step_"+str(len(self.steps)+1),"tool":t,"args":{}}
            self.keys=list(TOOLS[t]["params"]); self.key_index=0; self.state="key" if self.keys else "condition"
        elif state=="key":
            key=self.keys[self.key_index]; self.start_value(TOOLS[self.current["tool"]]["params"][key],"arg",key)
        elif state=="condition":
            if t=="WHEN": self.start_value({"type":"source"},"exists")
            else: self.state="end"
        elif state=="end": self.steps.append(self.current); self.state="next"
        elif state=="value":
            if t in ("JOIN_NAME","JOIN_BASE"):
                self.stack.append({"parent":self.pending,"mode":"name" if t=="JOIN_NAME" else "basename_of","value":{}})
                self.pending={"spec":{"type":"directory"},"dest":None}
            else:
                if t.startswith("S") and t[1:].isdigit(): value={"slot":int(t[1:])}
                elif t.startswith("R"):
                    index,field=t[1:].split(":"); value={"result":{"step":"step_"+index,"field":field}}
                else: value=CONSTANTS[t]
                self.complete_value(value)

    def plan(self, context_id):
        if self.state!="done": raise ValueError("Incomplete generated program")
        if self.kind!="plan": return nonexecution(self.kind,self.reason,context_id)
        return {"protocol":"davework/1","kind":"plan","context_id":context_id,"steps":copy.deepcopy(self.steps)}


def masks(request, target):
    g=Grammar(request); rows=[]; decisions=[]
    for token in target[1:]:
        allowed=g.allowed(); rows.append(allowed); decisions.append(len(allowed)>1); g.advance(token)
    return rows, decisions

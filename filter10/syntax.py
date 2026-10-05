"""Literal boundary evidence for a bounded Chinese workspace command grammar.

Literals are consumed as opaque intervals. We never run keyword recognition
inside an interval already recognized as a name, path or text. No name list,
filesystem lookup or execution is involved. Unsupported/ambiguous syntax is
reported rather than treated as a successful parameter-free command.
"""
import re
from dataclasses import dataclass, field
from .boundaries import quoted_interiors, QUOTE_PAIRS

NAME = r"(?:新(?:名字|名称)|文件(?:名字|名称|名)|文档名|目录名|名字|名称)(?:叫做|改为|改成|叫|是|为)|名为|名叫|叫做|重命名(?:为|成)|改名(?:为|成|叫)|命名为|(?:名字|名称)改(?:为|成)"
PATH = r"路径(?:为|是)?"
TEXT = r"(?<=[把将])(?:内容|文字)(?=[\s\S]+(?:写入|写到|追加到))|(?:搜索|查找)(?:正文|内容|文字内容)?(?:中|里)?(?:包含|含有|有)(?:文字|原文)?|(?:初始)?内容为|(?:写入|追加|添加)(?:原文|文字|内容)?|(?:正文|内容)(?:中|里)?(?:包含|含有|有)(?:文字|原文)?|(?:包含|含有)(?:文字|原文)?|(?:搜索|查找)(?:关键词|原文|文字|内容)|原文|(?:文字|内容)(?=[\s\S]+(?:写入|写到|追加到))"
ROOT_FILE = r"(?:复制到|移动到|复制至|移动至|移到|移入)(?:文件夹|目录)|(?:打开|查看|读取|删除|删掉|提取|给|向|复制|移动)(?:文件夹|文件|文档|目录)(?!名(?:为|是|叫|字|称))|(?:把|将)(?:文件夹|文件|文档|目录)(?!名(?:为|是|叫|字|称))"
ROOT_VALUE = r"(?:打开|查看|读取|删除|删掉|复制|移动|搜索|查找)|(?:把|将)(?=.+的(?:名字|名称)改)"
RAW_PATH = r"(?<![\w])(?:[A-Za-z]:[\\/]|\\\\[^\\/\r\n]+[\\/]|\.{1,2}[\\/]|/(?!/))"
ANCHORS = re.compile(f"(?P<path>{PATH})|(?P<name>{NAME})|(?P<text>{TEXT})|(?P<file>{ROOT_FILE})|(?P<value>{ROOT_VALUE})|(?P<raw>{RAW_PATH})")
OBJECT_END = re.compile(r"的(?:那份|这份)?(?:文件夹|文件|文档|目录|文本)(?=$|[，,。；;！？!?\s]|里|中|内|下|的(?:正文|内容)|内容|正文|文字|打开|读取|删除|复制|移动|改名|重命名|名称|名字|追加|添加|写入|移|备份|提取|到|至)")
CONTROL_END = re.compile(r"这个(?:文件夹|文件|文档|目录)|的(?:名字|名称)改(?:成|为)|重命名(?:为|成)|改名(?:为|成|叫)|中的(?:文字|文本)|里的(?:文字|文本)|的(?:正文|内容)")
PUNCT = re.compile(r"[，,。；;！？!?]")
NO_LITERALS = re.compile(r"^(?:(?:请|先|现在|暂时|帮我|麻烦|请先|请帮我|注意：|我希望|这次|如果方便，|请先不要)\s*)*(?:"
    r"(?:先)?(?:暂停|停止|取消|等待|保持|不用操作|等我|不需要|暂时不需要|不用|关闭|如果没有|如果文件不存在|检查|统计|显示|列出|展示|只处理|仅处理|保留所有|禁止删除|不要删除任何|不要修改|不要覆盖|不要移动这些|不要写入|先不要写入|先不要执行|只读取不要修改).+|"
    r"暂停|取消操作|停止操作|不用操作|保持当前状态|暂停所有任务|只读取不要修改|"
    r"(?:打开|查看|读取|删除|复制|移动|关闭|不要删除|只打开|只复制|只读取)(?:当前|选中|所有|全部|这些|空|任何).+|"
    r"(?:请)?不要删除当前选中的文件|把当前文件复制到选中的目录|保持文件名称不变)\s*$")


@dataclass
class SyntaxResult:
    spans: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    no_literals: bool = False
    ambiguous: bool = False


def _inside(position, spans):
    return any(s.get("marker_start", s["start"]) <= position < s["end"] for s in spans)


def _quoted(text, start, quotes):
    for s, e in quotes:
        if start == s - 1:
            return s, e
    return None


def _bounds(text, start, kind, mode, quotes):
    while start < len(text) and text[start].isspace():
        start += 1
    quoted = _quoted(text, start, quotes)
    if quoted:
        return quoted, False
    if start >= len(text) or text[start] in QUOTE_PAIRS:
        return None, True
    punctuation = PUNCT.search(text, start)
    stop = punctuation.start() if punctuation else len(text)
    object_matches = [m for m in OBJECT_END.finditer(text, start) if m.start() <= stop
                      and not any(s <= m.start() < e for s, e in quotes)]
    strong_objects = [m for m in object_matches if m.end() == len(text) or not text[m.end()].isspace()]
    object_ends = [m.start() for m in strong_objects]
    control_ends = [m.start() for m in CONTROL_END.finditer(text, start) if start < m.start() <= stop
                    and not any(s <= m.start() < e for s, e in quotes)]
    if kind in ('PATH', 'TEXT') or re.match(r"(?:[A-Za-z]:[\\/]|\\\\|/|\.{1,2}[\\/]|~[\\/])", text[start:]):
        control_ends = []
    if control_ends:
        object_ends = [end for end in object_ends if end <= min(control_ends)]
    if mode in ("file", "value", "terminal", "content"):
        object_ends = []
    if mode == "terminal":
        control_ends = []
    ends = ([min(object_ends)] if object_ends else []) + control_ends
    if mode in ("file", "value"):
        if mode == "file":
            ends += [m.start() for m in re.finditer(r"追加(?:原文|文字|内容)|添加(?:原文|文字|内容)|写入(?:原文|文字|内容)|改名(?:为|成|叫)", text)
                     if start < m.start() <= stop and not any(s <= m.start() < e for s, e in quotes)]
        # A transfer delimiter must separate two arguments; marked destination
        # names/paths are handled by their own anchor later.
        transfer = [m for m in re.finditer(r"(?:复制到|移动到|移到|移入|复制至|移动至|到|至)", text[start:stop])
                    if not any(s <= start + m.start() < e for s, e in quotes)]
        if transfer:
            strong = [m for m in transfer if len(m.group()) > 1]
            ends.append(start + (strong[-1] if strong else transfer[0]).start())
        declaration = re.match(r"(?:那个|这个|这份|那份|一个)?(?:" + NAME + r"|" + PATH + r")", text[start:])
        remainder = text[start+declaration.end():] if declaration else ''
        literal_marker = mode == 'file' and (not remainder or PUNCT.match(remainder)
                                              or remainder.startswith(('的名字', '的名称')))
        if mode == 'value' and declaration and not literal_marker:
            return None, remainder.startswith(('到', '至'))
        if mode == "value" and text[start:].startswith(("文件", "文档")):
            return None, False
    if kind == "TEXT":
        for m in re.finditer(r"(?:写入|写到|追加到)(?=名称|名字|名为|叫做|路径|[“「\"\'])", text[start:stop]):
            ends.append(start + m.start())
    end = min(ends) if ends else stop
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return None, True
    value = text[start:end]
    if mode in ("file", "value") and (re.match(r"^(?:当前|选中|所有|全部|任何|这些|空文件)", value) or
                                       re.fullmatch(r"(?:它|它们|这个|那个|该文件|这份文件)(?:的内容|的正文)?", value)):
        return None, False
    # An unquoted transfer operand containing several plausible '到/至'
    # boundaries is ambiguous. Asking for quotes preserves the opaque value.
    if mode == "value" and len(re.findall(r"到|至", text[start:stop])) > 1:
        return None, True
    return (start, end), False


def analyze_syntax(text):
    result = SyntaxResult()
    explicit = any(m.lastgroup in ("name", "path", "raw") for m in ANCHORS.finditer(text))
    text_parameter = any(m.lastgroup == 'text' and text[m.end():].strip() not in
                         ('', '内容', '文字', '原文', '任何内容', '任何文字', '任何原文')
                         for m in ANCHORS.finditer(text))
    if not text or (NO_LITERALS.fullmatch(text) and not explicit and not text_parameter):
        result.no_literals = True
        return result
    quotes = quoted_interiors(text)
    for match in ANCHORS.finditer(text):
        if _inside(match.start(), result.spans):
            continue
        # Keyword matches inside quoted literal content belong to that content.
        if any(s <= match.start() < e for s, e in quotes):
            continue
        mode = match.lastgroup
        if mode == 'name' and match.group().startswith('文件') and text[match.end():].startswith(('的名字', '的名称', '移动', '复制', '改名', '重命名')):
            # "文件名字是..." may be a declaration or a literal named
            # "名字是". With no value before the next operator, require quotes.
            result.ambiguous = True
            result.pending.append({'start': match.start(), 'end': match.end(), 'type': 'NAME'})
            continue
        clause = re.split(r"[，,。；;！？!?]", text[:match.start()])[-1] + text[match.start():]
        if result.spans and NO_LITERALS.fullmatch(clause):
            continue
        if mode == "value" and result.spans and (not text[match.end():].strip() or re.fullmatch(r"(?:一下|它|正文|内容|文字|文本)\s*[。！!]?", text[match.end():])):
            continue
        if mode == "text" and any(s["type"] == "TEXT" for s in result.spans) and re.match(r"(?:到)?(?:名称|名字|名为|名叫|叫做|路径)", text[match.end():]):
            continue
        if mode == "value" and result.spans and match.group() in ("读取", "查看", "提取") and re.fullmatch(r"(?:正文|内容|文字|文本)\s*[。！!]?", text[match.end():]):
            continue
        if mode == "value" and result.spans and text[match.end():].startswith(("到", "至")):
            continue
        if mode in ("file", "value") and re.search(r"怎么|如何|为什么|什么意思|是否应该", text[:match.start()]):
            continue
        kind = {"path": "PATH", "raw": "PATH", "name": "NAME", "file": "NAME", "text": "TEXT", "value": "VALUE"}[mode]
        if mode == "value" and match.group() in ("搜索", "查找"):
            kind = "TEXT"
            if text[match.end():].startswith(("正文", "内容", "包含", "原文", "文字", "关键词")):
                continue
        if mode == "text" and text[match.end():].startswith(("文字", "内容", "原文")):
            # Longer nested text markers are consumed by the regex itself;
            # reject malformed leftover marker words instead of naming them.
            pass
        start = match.start() if mode == "raw" else match.end()
        boundary_mode = mode
        if mode == "name" and re.search(r"重命名|改名|命名|新名字|新名称|名字改|名称改|文件名|文件名字|文件名称|文档名|目录名", match.group()):
            boundary_mode = "terminal"
        if mode == "text" and re.match(r"写入|追加|添加|(?:初始)?内容为", match.group()):
            boundary_mode = "content"
        bounds, ambiguous = _bounds(text, start, kind, boundary_mode, quotes)
        result.ambiguous |= ambiguous
        if not bounds:
            if ambiguous or mode in ("name", "path", "text"):
                result.pending.append({"start": match.start(), "end": match.end(), "type": kind})
            continue
        s, e = bounds
        if any(s < other["end"] and e > other["start"] for other in result.spans):
            continue
        result.spans.append({"start": s, "end": e, "type": kind,
                             "marker_start": match.start(), "marker_end": match.end(), "verified_by_syntax": True})
        # Bare transfers have a second operand without a separate action word.
        if mode in ("file", "value"):
            link_start = e + 1 if e < len(text) and text[e] in '”」\"\'' else e
            link = re.match(r"(?:复制到|移动到|复制至|移动至|移到|移入|到|至)", text[link_start:])
            if link:
                destination = link_start + link.end()
                rest = text[destination:]
                if not re.match(r"(?:" + NAME + r"|" + PATH + r"|当前|选中)", rest):
                    dest_kind = "VALUE"
                    selector = re.match(r"(?:文件夹|目录)", rest)
                    if selector:
                        destination += selector.end()
                        dest_kind = "NAME"
                    dest_bounds, ambiguous = _bounds(text, destination, dest_kind, "terminal", quotes)
                    result.ambiguous |= ambiguous
                    if dest_bounds:
                        result.spans.append({"start": dest_bounds[0], "end": dest_bounds[1], "type": dest_kind,
                            "marker_start": e, "marker_end": destination, "verified_by_syntax": True})
                    else:
                        result.pending.append({"start": e, "end": destination, "type": dest_kind})
    result.spans.sort(key=lambda s: s["start"])
    return result

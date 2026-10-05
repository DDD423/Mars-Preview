"""Second acceptance corpus, frozen before stage-two grammar repairs.

Stage one fresh.jsonl is development data once its errors have been analyzed.
This corpus stays separate, and its scores must never drive further repairs.
"""
import hashlib
import json
from pathlib import Path
from .data import SLOT_RE, QUOTES, render, write_jsonl

FRAMES = [
    '先打开那个名称为[[NAME:a]]的文档',
    '先不要删除叫做[[NAME:a]]的那份文件',
    '帮我创建文档，名称为[[NAME:a]]',
    '建立一个目录，名字是[[NAME:a]]',
    '读取名字叫[[NAME:a]]的文本',
    '名称是[[NAME:a]]的文件，请打开',
    '文档名是[[NAME:a]]，先复制它',
    '把名叫[[NAME:a]]的文档重命名为[[NAME:b]]',
    '请将文件[[NAME:a]]的名称改成[[NAME:b]]',
    '把路径为[[PATH:a]]的文档复制到路径[[PATH:b]]',
    '将名称是[[NAME:a]]的文件移入名为[[NAME:b]]的文件夹',
    '从路径为[[PATH:a]]的文本中提取文字',
    '不要修改，打开路径[[PATH:a]]的文件',
    '只读取路径[[PATH:a]]的文件内容',
    '若已经备份，删除名叫[[NAME:a]]的文档',
    '查找文字内容包含[[TEXT:a]]的文档',
    '找出内容里含有[[TEXT:a]]的文件',
    '在文件中查找原文[[TEXT:a]]',
    '搜索关键词[[TEXT:a]]',
    '给名叫[[NAME:a]]的文档添加原文[[TEXT:b]]',
    '向路径为[[PATH:a]]的文件写入文字[[TEXT:b]]',
    '把内容[[TEXT:a]]写入叫做[[NAME:b]]的文档',
    '将文字[[TEXT:a]]写到路径[[PATH:b]]的文本里',
    '创建名叫[[NAME:a]]的文档，初始内容为[[TEXT:b]]',
    '复制[[VALUE:a]]至[[VALUE:b]]',
    '移动[[VALUE:a]]到[[VALUE:b]]',
    '打开[[VALUE:a]]，不要修改',
    '读取文件[[NAME:a]]，如果不存在就停止',
    '将文件[[NAME:a]]复制到选中的目录',
    '名字为[[NAME:a]]的文件，先读取它',
    '在路径[[PATH:a]]的目录里搜索包含文字[[TEXT:b]]的文档',
    '请把原文[[TEXT:a]]追加到名叫[[NAME:b]]的文件中',
]
NAMES = ['改名成', '追加文字', '复制至', '路径是', '把文件', '的文件', '文本', '包含文字',
         'alpha β 2027.md', '星站🪐.csv', 'abc{2}XYZ', '从未使用的名称𠀀',
         '重命名为', '文件名字', '先不要修改', '未知名-Z17']
TEXTS = ['先不要移动', '添加原文', '改名成', '另一行\n再一行', '💡abc β', '名字叫',
         '没有目录', '{2} literal', '短文本', '空白 保留', 'content中包含', '新增ζ关键词',
         '重命名为', '中文english', '没有匹配', '保持文本原样']
PATHS = ['E:/评测二/重命名为.txt', '\\\\nas-two\\新共享\\复制至.md', './不同 空间/追加文字.log',
         '/opt/星站🪐/a β.txt', '../首次路径/的文件.csv', 'F:\\新名\\改名成.md',
         '~/second/包含文字.txt', 'E:/文本/{2}.md']
NONE = ['暂停操作', '等待下一步', '显示所有文件', '统计目录数量', '列出当前目录所有文件',
        '只读取不要修改', '停止全部操作', '关闭当前文档', '打开选中的目录', '不要写入任何文字',
        '不要删除这些文件', '复制所有文档到当前目录', '如果没有文件就停止',
        '保留所有文件', '检查选中的文件', '保持文件名称不变']


def generate_acceptance(data_dir='data/posttrain'):
    directory = Path(data_dir)
    path = directory / 'acceptance.jsonl'
    if path.exists():
        manifest = json.loads(path.with_name('acceptance_manifest.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
            raise ValueError('Frozen acceptance corpus changed')
        return manifest
    rows = []
    for family, frame in enumerate(FRAMES):
        for index in range(16):
            values, quotes = {}, {}
            for m in SLOT_RE.finditer(frame):
                kind, key = m.groups()
                k = (index + (5 if key == 'b' else 0)) % 16
                pool = PATHS if kind == 'PATH' else TEXTS if kind == 'TEXT' else NAMES
                values[key] = pool[k % len(pool)]
                if kind == 'VALUE' and index % 2:
                    values[key] = PATHS[k % len(PATHS)]
                quotes[key] = QUOTES[1 + index % 4] if index % 3 == 0 else ('', '')
            text, spans = render(frame, values, quotes)
            rows.append({'id': f'acceptance_{family:02}_{index:02}', 'family': f'acceptance_{family:02}',
                         'text': text, 'spans': spans, 'source': 'assistant_authored_acceptance'})
    for prefix in ('', '请', '麻烦', '现在', '帮我'):
        for index, text in enumerate(NONE):
            rows.append({'id': f'acceptance_none_{len(rows)}', 'family': 'acceptance_none',
                         'text': prefix + text, 'spans': [], 'source': 'assistant_authored_acceptance'})
    write_jsonl(path, rows)
    manifest = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'frozen_before_stage_two_repairs': True, 'used_for_tuning': False,
                'source': 'assistant-authored; 32 families, 16 values each, 80 no-literal examples'}
    path.with_name('acceptance_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

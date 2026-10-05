# Dave Work 指令协议 `davework/1`

工具参数和返回字段的唯一权威定义是 `registry.py` 中的 `TOOLS`；浏览器表单、后端校验和未来规划模型输入都使用此定义。运行时通过 `GET /api/tools` 获取完整目录。此文档说明通用语义，不复制另一套参数表。

## 请求上下文

`POST /api/context` 提交 `{"text":"中文请求","extract":true}`。返回独立 `context_id`、工作区快照、filter 的结构化提取结果和 translator1.0 的 plan / clarification / noop。页面仅对 plan 自动请求预览，不自动执行。`extract:false` 创建手动工具上下文，不加载模型。模型失败时手动工具可继续使用。

上下文由服务端保存，不接受客户端修改绑定。最多保留 100 个上下文，重启服务后需要重建。配置变化只影响后续上下文。所有偏移按 Unicode 码点计数，左闭右开；前端通过结构化 segments 渲染和还原，不按 JavaScript UTF-16 索引切割原文。

## 计划

```json
{
  "protocol": "davework/1",
  "kind": "plan",
  "context_id": "ctx_由提取接口返回",
  "steps": [
    {"id":"rename_1","tool":"fs.rename","args":{"source":{"slot":1},"new_name":{"slot":2}}}
  ]
}
```

1–64 个步骤顺序执行，id 为非空且唯一的字符串。工具的必需字段不可省略，可选字段使用注册表默认值。未知工具、多余字段、缺失字段、非法类型均报错。没有任意表达式求值、动态工具名或 shell 命令拼接。

- `{"slot":1}` 只引用当前 context 的 `{1}` 原文。不得附加 context_id。原始名称由 harness 保存；主模型适配输入只传 literal / slot 片段和 slot 类型，不传绑定值、原始输入或 proposals。
- 普通字符串 `"{1}"` 是字面文字。即使原文含 `{1}`，也不会误替换。
- 文件来源字段的 NAME 引用递归精确匹配完整 basename。Windows 使用系统大小写规则；不补扩展名、不模糊猜测。唯一匹配才能绑定。PATH/VALUE 和直接字符串按工作区内路径解析。
- 创建目标及 new_name 不查找已有名称。TEXT 不能绑定到文件路径字段；new_name 不允许目录分隔符、冒号和尾随点 / 空格。
- 显式覆盖写入的 NAME 来源查找已有文件；普通创建目标继续按新路径处理。
- `{"join":{"directory":{"slot":2},"basename_of":{"slot":1}}}` 将来源实际完整名称放入指定目录。`{"join":{"directory":{"slot":1},"name":{"slot":2}}}` 在目录下使用一个新名称。仅可用于 target / path 字段；不嵌套、不求值。directory 的 NAME 精确查找已有目录，name 不查找；directory / basename_of 可引用前序 path 结果。组合依赖同样固定、校验范围及检查预览变化。重名选择键如 `step_1.destination.join.directory`。
- `{"result":{"step":"copy_1","field":"path"}}` 引用前序步骤已声明的返回字段，类型必须兼容。可预测的路径在预览中展开；运行后才能得到的内容或删除记录 ID 在预览中保留结构化引用，执行到该步骤时绑定并显示真实参数。前序步骤跳过或失败时不得继续使用无效结果。程序参数 argv 中也可使用这些引用。
- `when:{"exists":"relative/path","negate":false}`：文件存在时运行；negate:true 表示不存在时运行。exists 也支持结构化引用。不满足条件则发出 step_skipped。不存在任何其他条件表达式。

非执行结果：

```json
{"protocol":"davework/1","kind":"clarification","context_id":"ctx_...","message":"哪一个同名文件？"}
```

`kind:"noop"` 使用相同字段。两者不会调用工具。

## 预览与执行

- `POST /api/preview`：`{"plan":计划,"selections":{}}`。重名时返回 AMBIGUOUS_NAME、selection_key 和真实路径 candidates。选择后重新提交，例如 `selections:{"rename_1.source":"工作区内真实路径"}`。预览显示固定工作区与绑定参数。
- 返回 preview_id 与计划 SHA-256 digest，有效期 600 秒。修改编辑器会清除页面预览。
- `POST /api/execute` 只接受 `{"preview_id":"pre_...","digest":"..."}`；不接受新计划。预览只能执行一次。服务端在执行前和每一步检查路径的身份、大小、修改时间及父目录，重新校验真实路径范围。目标发生变化需重新预览，绝不重新猜测名称。
- 运行观察引用如 stdout 不能在执行前得到实际内容，因此预览明确展示“待前序结果”的结构引用。若该观察被用作路径，执行时仍重新进行工作区限制检查。任务开始后的 step_started 事件显示最终真实参数。
- 文件失败、结果失效、权限不足、程序非零退出、超时都会停止后续步骤。

## 文件与终端行为

文件路径只能位于捕获的工作区真实路径内。`.davework` 和 `.davework-trash` 为保留目录。工作区根不能修改或删除。递归操作拒绝子树内链接 / junction；索引不跟随链接。检索排除目录可以配置，保留目录始终排除。索引排除不影响用户明确提供路径访问普通目录。

写入默认不覆盖；显式覆盖会备份旧内容，返回 backup_id。删除仅移入 `.davework-trash`，返回 record_id。`fs.restore` 可恢复删除或备份；目标已存在时失败，先把目标移走再恢复。复制、移动、重命名遇到冲突失败。写入先落临时文件，再原子发布；复制使用临时目标，取消时清理。

`fs.append_text(path,text,encoding)` 必须针对已有文本文件，缺失或编码不匹配即失败，不创建新文件。先验证明确的 UTF-8 / GB18030 编码，备份旧字节，保留原字节后追加编码后的内容，通过临时文件原子替换，返回 backup_id。不会自动猜测编码。

UTF-8 / GB18030 必须明确选择，解码失败不猜测编码。读取上限 1 MiB，搜索最多 1000 条结果，截断以 truncated 标记。大文件内容搜索仅检查读取范围并标记截断。二进制文件、PDF、DOCX、图片不提取文字。

被截断的读取结果不能作为 `result.text` 转交写入或追加，执行器返回 `TRUNCATED_REFERENCE`，避免把部分正文当成完整正文复制。结果依赖和目录组合的来源路径在预览时固定；复制后仍检查只读来源是否发生变化。

`terminal.exec` 用程序与 argv 数组调用非交互进程，不经过 shell。不直接执行 .bat/.cmd，避免 Windows 的隐式 shell；需要时明确使用 PowerShell 脚本工具。PowerShell 脚本保持原文，参数通过 `DAVE_SLOT_1` 等环境变量提供；脚本中的 `{1}` 不替换，ErrorActionPreference 设置为 Stop。Python 和 PowerShell 使用 UTF-8 输出，其他程序应自行选择 UTF-8；非 UTF-8 字节会显示替代字符。终端独立运行，默认工作目录为任务工作区，不继承上次 cd / 环境更改。Windows 在进程恢复执行前绑定 Job Object；取消、超时、正常结束和服务退出均结束剩余子进程。累计输出上限每次 4 MiB，超限后仍排空输出。

**终端拥有当前用户权限，工作区不是操作系统沙箱。** 文件路径检查不约束终端脚本可以访问的地方。

## API 与事件

所有 `/api/*` 请求使用 `Authorization: Bearer 启动令牌`。POST 必须使用 application/json 且 Origin 与本服务相同；请求体上限 2 MiB。服务仅监听 127.0.0.1，拒绝未知 Host、不提供跨来源许可。启动 URL 用 fragment 传令牌，页面将它存入 sessionStorage 后清除 fragment。

| API | 方法 | 用途 |
|---|---|---|
| `/api/config` | GET / POST | 读取状态 / 更新配置；CUDA 不可用拒绝更新 |
| `/api/context` | POST | filter 提取或空手动上下文 |
| `/api/tools` | GET | 完整工具目录、参数及返回结构 |
| `/api/preview` | POST | 校验、绑定和生成一次性预览 |
| `/api/execute` | POST | 执行已预览的计划 |
| `/api/cancel` | POST | `{"task_id":"task_..."}` 停止任务 |
| `/api/tasks` | GET | 最近任务 |
| `/api/events?task_id=...&after=序号` | GET | 补取序号之后的事件 |

事件序号从 1 连续增长，客户端只补取新事件。重连不重放执行请求。最近 20 个任务和配置保存在项目 `.davework`；一个任务事件最多 8 MiB，后续内容明确标记省略，状态继续保存，历史重复结果字段最多 8192 字符。记录包含原始参数和输出，仅保存在本机。首版支持一个活动任务，取消不回滚已经成功完成的步骤。

## 主模型适配

`adapter.planner_input(context, observations)` 提供结构片段、slot 类型、可用工具和显式观察。`translator10.inference.TranslatorPlanner.plan(request)` 实现此接口；不读绑定值、context_id 内容或观察正文，只编码 literal / slot / type 和工具开关。文件正文通过 result 引用转交执行器。主模型限制四步八参数和固定文件工具，手动 JSON 仍可提交协议支持的 64 步以及终端。低置信返回固定澄清；模型无法跳过预览。`NoModelPlanner` 只负责手动空上下文。

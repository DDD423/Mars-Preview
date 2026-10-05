# Mars Preview

**用两个小模型，把中文请求变成本地文件操作。**

Mars Preview 是一个本地运行的 **Filter–Translator** 模型组合与 Agent 框架。两个模型合计约 **360 万参数**，提供训练好的权重、Python 接口和浏览器工作台 Dave Work。

它研究一个简单的问题：如果只处理一类明确任务，小模型能否通过合理分工，完成原本需要更大模型的工作？目前它仍是实验预览版。

## 它如何工作

```text
中文请求 → Filter 提取原文参数 → Translator 生成计划 → Dave Work 绑定、预览 → 点击执行
```

- **filter1.0（约27万参数）**：找出文件名、路径、搜索词和正文，用 `{1}`、`{2}` 代替，并保存原文。它使用两组 GRU，根据候选前后的上下文判断；候选自身的文字不进入自己的分类评分，不需要名称词典。
- **translator1.0（约333万参数）**：一个小型 Transformer，把带占位符的请求转成工具计划。文件名、路径和正文原文不进入它的输入，它只规划动作、参数关系和顺序。
- **Dave Work**：普通程序负责精确名称查找、路径绑定、工具校验、预览和执行。它不根据中文关键词替模型选择动作。

例如：

```text
把叫做草稿.txt的文件改名为最终版.txt
                  ↓
把叫做{1}的文件改名为{2}
                  ↓
fs.rename(source=SLOT_1, new_name=SLOT_2)
```

参数原文由框架保管，模型计划经过预览才会执行。输入本身含有字面 `{1}` 时，也不会被全局字符串替换误改。

候选评分不读取候选原文，并不意味着整套提取对名称完全不敏感：整句数量判断和其他候选的上下文仍可能受到名称影响。

## 可以做什么

列目录、读取文本、搜索名称或正文、创建目录、创建/写入/追加文本、复制、移动、重命名、移入回收目录和恢复。最多规划4个步骤、8个参数，支持部分句内指代、否定和存在性条件。

首版主要处理常见中文文件请求，最长256个Unicode字符。不处理跨轮指代、任意长文、PDF/OCR或自由编程。终端保留为手动工具，规划模型不生成终端命令。

## 与 Qwen 0.5B 的同题对照

双方处理同样的 **620个原始中文请求**，并在独立临时工作区核对文件变化和工具输出；包含500个支持的操作任务和120个不执行请求。图中使用**不启用置信拒绝门槛**的结果，拒绝本应执行的任务仍算错。

![参数量与文件任务正确率](docs/assets/parameters-vs-accuracy.png)

![做题时间与文件任务正确率](docs/assets/time-vs-accuracy.png)

| 项目 | Mars Preview | Qwen2.5-0.5B-Instruct |
|---|---:|---:|
| 参数量 | 3,599,021 | 494,032,768 |
| 全部620题，实际结果正确 | **420/620，67.74%** | 32/620，5.16% |
| 其中500个支持的操作任务 | **313/500，62.60%** | 3/500，0.60% |
| 单独40个人工编写请求 | 32/40，80.00% | 6/40，15.00% |
| 完成620题的推理时间 | **13.18秒** | 902.53秒 |

这是一项**具体系统配置的对照**，不是通用语言能力排名。Mars Preview 经过文件任务专项训练并使用约束解码；Qwen 使用固定提示词、5个示例和自由JSON输出，没有专项训练或重试。Qwen出现228个JSON语法错误、76个非原文参数错误以及152个其他协议错误，不能把它的5.16%解释成通用能力只有5%。

时间在同一台RTX 5060 Laptop GPU上测得，排除下载与模型加载；这是批推理完成整套题的时间，不是单句交互延迟。580题此前重复验收过，另40题是此前未预测的人工编写请求；它们都是实验室测试，不代表真实用户总体成功率。

默认发布模型仍启用现有置信拒绝门槛，同一620题正确率为 **416/620，67.10%**；接受计划的实际结果正确率为 **309/370，83.51%**。结果尚未达到稳定实用的目标。详见[评测方法、原始数据与复现](docs/EVALUATION.md)。

## 快速开始

建议使用 Python 3.9–3.12。Windows 提供完整工作台与终端进程树管理；模型支持CPU或CUDA。仓库已经包含两个自研模型权重，Qwen权重不随仓库分发。

```powershell
git clone https://github.com/DDD423/Mars-Preview.git
cd Mars-Preview
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m davework
```

浏览器打开后，在左侧选择一个测试工作区，输入中文请求。页面显示参数绑定与模型计划，核对后点击执行。

也可以只生成计划，不执行文件操作：

```powershell
.venv\Scripts\python.exe -m translator10 predict "把叫做草稿.txt的文件改名为最终版.txt"
.venv\Scripts\python.exe -m translator10 chat
```

CPU版本安装见[PyTorch官方说明](https://pytorch.org/get-started/previous-versions/)。需要CUDA 12.8时可安装：

```powershell
.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## 开发与开源内容

包含自研训练权重、推理代码、训练API、数据生成器、文件工具、版本化计划协议、测试，以及此次对照的题集、提示词和原始输出。训练入门见[TRAINING.md](docs/TRAINING.md)，工具协议见[PROTOCOL.md](davework/PROTOCOL.md)，模型说明与哈希见[MODEL_CARD.md](docs/MODEL_CARD.md)。训练脚本可以复用，但短命令重新训练不保证复现当前成绩。

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
New-Item -ItemType Directory -Force .test-workspaces | Out-Null
.venv\Scripts\python.exe -m pytest -q --basetemp .test-workspaces/local
```

文件工具限制在所选工作区。终端使用当前用户权限，工作区不是操作系统沙箱。请先在测试目录中体验；模型计划仍可能有误。

## License

代码和自研权重使用 [MIT License](LICENSE)。Qwen是独立的第三方模型，遵循其[官方仓库许可](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。

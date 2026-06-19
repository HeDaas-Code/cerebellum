---
title: Cerebellum
version: 0.1.0
type: ai-agent-library
language: Python
license: MIT
status: active-development
last_updated: 2026-06-19
---

# Cerebellum 智能代理库

> 基于 DeepAgents 框架的 Python 库——提供阿里百炼 LLM + Daytona 沙盒 + 技能系统 + 联网搜索，让 AI 智能代理能处理真实任务（文档解析、代码生成、文件上传分析）。

## 这是什么

Cerebellum = "小脑"——负责把高层意图转化为具体动作的执行层。

**核心定位**：把 `DeepAgents` 框架工程化——加上：

- **多 LLM 后端**（阿里百炼 glm-5、SiliconFlow 兼容层）
- **Daytona 沙盒**（安全的远程代码执行环境）
- **技能系统**（加载自定义技能处理特定任务，类 Claude Skills 模式）
- **联网搜索**（Tavily 集成）
- **文件数据返回**（不落盘，直接 bytes/str 流回调用方）

**适用场景**：批量文档处理、自动化报告生成、需要执行代码的研究型任务。

## 仓库结构

```
cerebellum/
├── cbot.py                 # 使用示例入口（参考用）
├── cerebellum/             # 核心库（pip install -e . 后作为包导入）
│   ├── __init__.py         # Cerebellum/CerebellumConfig/run_task 公开 API
│   ├── config.py           # 配置类
│   ├── parser.py           # 文档解析（PDF/DOCX/YAML）
│   ├── security.py         # 输入过滤 + 沙盒权限
│   ├── upload.py           # 文件上传处理
│   ├── orchestrator/       # 多代理协调
│   ├── reflection/         # 自我反思与修正
│   ├── skills/             # 技能加载器（运行时注册）
│   ├── tools/              # 工具（沙盒调用、搜索）
│   ├── data/               # 任务编码、缓存、相似度
│   └── utils/              # 日志等
├── docs/
│   └── 参考.md             # 内部参考文档（设计记录 + 决策）
├── requirements.txt
└── README.md               # 你正在读
```

## 安装

```bash
pip install -r requirements.txt

# 可选依赖（用于文件解析）
pip install pdfplumber python-docx pyyaml
```

## 快速开始

最小可运行示例（见 `cbot.py`）：

```python
from cerebellum import Cerebellum, CerebellumConfig, SandboxConfig
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

config = CerebellumConfig(
    skills_dir=Path("./cerebellum/skills"),
    database_path=Path("./my_cache.db"),
    debug=True,
)

sandbox_config = SandboxConfig(
    timeout_seconds=300,
    max_retries=3,
    auto_cleanup=True,
    workdir="/home/daytona/workspace",
)

config.sandbox = sandbox_config

with Cerebellum(config=config) as agent:
    result = agent.run("请帮我生成一个斐波那契数列的图片")
    # result = {
    #     "success": True,
    #     "message": "处理结果...",
    #     "files": [FileData(name="fib.png", content=b"...", type="image/png")]
    # }
```

**更多用法**（带文件上传、沙盒预装包、错误重试）：见 [[cerebellum/README]]。

## 环境配置

**必填**（`.env`）：

| 变量 | 用途 |
|---|---|
| `DASHSCOPE_API_KEY` | 阿里百炼 LLM 密钥 |
| `TAVILY_API_KEY` | 联网搜索 |
| `DAYTONA_API_KEY` | 沙盒环境 |

**可选**：

| 变量 | 默认 | 用途 |
|---|---|---|
| `SILICONFLOW_API_KEY` | — | 切换到 SiliconFlow 后端 |
| `CEREBELLUM_DEBUG` | `false` | 调试日志开关 |

## 开发

```bash
# 运行测试
pytest

# 覆盖率
pytest --cov=cerebellum

# 本地编辑模式
pip install -e .
```

## 项目状态

- ✅ **LLM 后端抽象层** — commit `bf80c73`（2026-06-15）
- ✅ **沙盒环境配置** — 同上
- ✅ **技能系统** — 支持运行时注册
- ⏳ **文档待补** — API 参考、`docs/参考.md` 部分章节为占位
- ❌ **未发布 PyPI 包** — 当前仅源码使用

## 关联笔记

- [[cerebellum/README]] — 库 API 详细文档
- [[docs/参考]] — 设计决策与内部参考（中文）
- [[CONTEXT]] — 项目领域术语表（待补）

## 许可

MIT
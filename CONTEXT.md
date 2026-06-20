---
title: cerebellum 领域术语表
type: context
status: active
last_updated: 2026-06-19
---

# CONTEXT.md — cerebellum 领域术语表

## 项目核心概念

### Cerebellum（小脑）

基于 DeepAgents 框架的 Python 智能代理库。

**为什么叫"小脑"**：类比神经科学——大脑管思考，小脑管协调执行。Cerebellum = 把高层意图转化为具体动作的执行层。

### 核心组件

| 组件 | 模块 | 职责 |
|---|---|---|
| Cerebellum | `__init__.py` | 主入口类（context manager） |
| CerebellumConfig | `config.py` | 配置（skills_dir/database_path/debug） |
| SandboxConfig | `tools/sandbox.py` | Daytona 沙盒配置 |
| Parser | `parser.py` | 文档解析（PDF/DOCX/YAML） |
| Security | `security.py` | 输入过滤 + 沙盒权限 |
| Upload | `upload.py` | 文件上传处理 |
| Orchestrator | `orchestrator/` | 多代理协调 |
| Reflection | `reflection/` | 自我反思与修正 |
| Skills | `skills/` | 技能加载器（运行时注册） |
| Tools | `tools/` | 工具（沙盒调用、搜索） |

### 依赖矩阵

| 依赖 | 用途 |
|---|---|
| `deepagents>=0.1.0` | 核心框架 |
| `langchain>=0.3.0` | LLM 编排 |
| `langgraph>=0.2.0` | 状态图 |
| `langchain-openai` | OpenAI 兼容 |
| `langchain-daytona` | Daytona 沙盒 |
| `daytona-sdk` | 沙盒 SDK |
| `tavily` / `langchain-tavily` | 联网搜索 |
| `sentence-transformers` | 嵌入/相似度 |

### LLM 后端

- **主后端**：阿里百炼 `glm-5`（需 `DASHSCOPE_API_KEY`）
- **兼容后端**：SiliconFlow（OpenAI 兼容接口）

### Daytona 沙盒

**远程代码执行环境**—— agent 在沙盒里跑代码、生成图表、写文件。

配置示例（`cbot.py`）：

```python
SandboxConfig(
    timeout_seconds=300,
    max_retries=3,
    auto_cleanup=True,
    workdir="/home/daytona/workspace",
    pre_install=[
        {"name": "fonts-wqy-zenhei", "type": "apt"},
        {"name": "fonts-wqy-microhei", "type": "apt"},
        {"name": "fc-cache", "type": "apt"},
    ]
)
```

### 技能系统

- `skills/` 目录存放技能模块
- 运行时通过 `skills_dir` 加载
- 类似 Claude Skills——注册后可被 agent 调用

## 项目特定命名

| 术语 | 含义 |
|---|---|
| **CerebellumConfig** | 主配置类（skills_dir / database_path / sandbox） |
| **SandboxConfig** | 沙盒配置（独立于 CerebellumConfig） |
| **run_task** | 便捷函数（无 context manager） |
| **FileData** | 文件返回类型（name/content/type） |
| **DASHSCOPE** | 阿里百炼平台名 |
| **Daytona** | 远程开发环境平台 |

## 不混淆概念

- **cerebellum ≠ obsilo**——前者 Python 库，后者 TypeScript Obsidian 插件
- **cerebellum ≠ Neo_Agent**——前者基于 DeepAgents，后者基于 LangChain + LangGraph
- **SandboxConfig ≠ CerebellumConfig**——前者是后者的字段之一
- **Daytona ≠ Docker**——前者远程云沙盒，后者本地容器
- **DASHSCOPE ≠ OpenAI**——前者阿里百炼，后者 OpenAI
- **`__init__.py` ≠ 库入口脚本**——前者包初始化，后者导出公开 API

## 待补

- [ ] orchestrator 子模块清单
- [ ] reflection 子模块清单
- [ ] skills/ 已注册技能清单
- [ ] test/ 测试覆盖范围
- [ ] docs/参考.md 设计决策（中文）
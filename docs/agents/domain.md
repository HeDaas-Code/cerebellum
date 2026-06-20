---
title: cerebellum Domain Docs
type: agents-config
parent: AGENTS.md
---

# Domain Docs

**单上下文布局**——`CONTEXT.md`（根） + `docs/adr/`（入仓） + `工作Wiki/`（不入仓）。

## 何时开 ADR

- 新增 LLM 后端（provider）
- 改变 SandboxConfig 字段
- 重命名公开 API（Cerebellum / CerebellumConfig）
- 替换底层框架（DeepAgents → 别的）

## 何时不开

- bug fix
- 文档更新
- 技能注册
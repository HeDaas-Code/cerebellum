---
title: cerebellum Agent Skills Configuration
type: agents-config
status: active
last_updated: 2026-06-19
---

# AGENTS.md — cerebellum Agent Skills Configuration

> 本文件配置 cerebellum 项目使用 Hermes / Matt Pocock engineering skills 所需的项目级上下文。

## Agent skills

### Issue tracker

项目使用 GitHub Issues。详见 [[docs/agents/issue-tracker]]。

### Triage labels

Matt Pocock 默认 5 类状态机标签。详见 [[docs/agents/triage-labels]]。

### Domain docs

单上下文布局——根 `CONTEXT.md` + `docs/adr/`。详见 [[docs/agents/domain]]。

## 沟通约定

- **本项目代码、注释、issue/PR 标题使用中文**
- **变量/函数命名使用英文**（Python 业界惯例）
- **Commit message**：`feat: <中文简述>` / `fix: <中文简述>` / `chore: <中文简述>` / `docs: <中文简述>`
- **分支命名**：`feature/<功能>` / `fix/<问题>` / `chore/<任务>`

## 快速命令

```bash
# 安装依赖
pip install -r requirements.txt

# 可选依赖（用于文件解析）
pip install pdfplumber python-docx pyyaml

# 运行示例
python cbot.py

# 测试
pytest
```

## 关联笔记

- [[README]] — 仓库根 README（项目门面）
- [[cerebellum/README]] — 库 API 详细文档
- [[工作Wiki/README]] — 工作 Wiki 入口
- [[工作Wiki/00-index/README]] — 项目速览
- [[CONTEXT]] — 项目领域术语
# Deep Agents 配置项目

## 安装依赖
```bash
pip install -r requirements.txt
```

## 环境配置
复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```
# SiliconFlow API 配置
SILICONFLOW_API_KEY=your_api_key_here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=Qwen/Qwen2.5-7B-Instruct

# Daytona 沙盒配置
DAYTONA_API_KEY=your_daytona_api_key_here

# 可选：OpenAI 兼容模型配置
# 如果你想使用其他模型，可以在这里配置
```

## 使用方法
```bash
python agent.py
```

## 可用技能 (Skills)
- docx: Word 文档处理
- pdf: PDF 文档处理
- pptx: PowerPoint 演示文稿处理
- xlsx: Excel 表格处理
- skill-creator: 创建新技能
- theme-factory: 主题工厂
- slack-gif-creator: Slack GIF 创建器
- internal-comms: 内部通讯
- doc-coauthoring: 文档协作

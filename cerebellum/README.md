# Cerebellum 智能代理库

基于 DeepAgents 框架的强大 AI 智能代理库。

## 核心功能

- **阿里百炼 LLM** - 使用 glm-5 模型
- **Daytona 沙盒** - 安全的代码执行环境
- **技能系统** - 加载自定义技能处理特定任务
- **联网搜索** - 支持 Tavily 搜索
- **文件数据返回** - 直接返回文件内容（bytes/str），不保存到本地
- **文件上传处理** - 支持上传文件并让 AI 分析处理

## 安装

```bash
pip install -r requirements.txt

# 可选依赖（用于文件解析）
pip install pdfplumber python-docx pyyaml
```

## 快速开始

### 方式1: 使用上下文管理器

```python
from cerebellum import Cerebellum, CerebellumConfig
from pathlib import Path

config = CerebellumConfig(
    skills_dir=Path("./skills"),  # 可选：指定技能文件夹
    debug=True
)

with Cerebellum(config=config) as agent:
    result = agent.run("请帮我生成一个斐波那契数列的图片")
    
    # result 包含:
    # {
    #     "success": True,
    #     "message": "处理结果...",
    #     "files": [
    #         FileData(name="fibonacci.png", content=b"...", type="image/png")
    #     ]
    # }
```

### 方式2: 使用便捷函数

```python
from cerebellum import run_task

result = run_task("请帮我生成一个斐波那契数列的图片", debug=True)
```

### 方式3: 带文件上传的任务

```python
from cerebellum import Cerebellum, CerebellumConfig

config = CerebellumConfig(debug=True)

# 读取本地文件
with open("document.pdf", "rb") as f:
    file_data = f.read()

# 上传文件并处理（files 参数可选）
with Cerebellum(config=config) as agent:
    result = agent.run(
        task="请总结这份文档的主要内容",
        files=[(file_data, "document.pdf")]  # 可选
    )
    
    # result 包含:
    # {
    #     "success": True,
    #     "message": "处理结果...",
    #     "files": [...],        # 沙盒生成的文件
    #     "uploaded_files": [...]  # 上传的文件信息
    # }
```

### 方式4: 便捷函数 + 文件

```python
from cerebellum import run_task

# 读取文件
with open("report.pdf", "rb") as f:
    pdf_data = f.read()

# 带文件的任务
result = run_task(
    task="分析这份报告",
    files=[(pdf_data, "report.pdf")],
    debug=True
)
```

## API 参考

### CerebellumConfig

配置类，用于初始化 Cerebellum。

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dashscope_api_key` | str | "" | 阿里百炼 API Key |
| `dashscope_base_url` | str | "https://coding.dashscope.aliyuncs.com/v1" | API 地址 |
| `dashscope_model` | str | "glm-5" | 模型名称 |
| `daytona_api_key` | str | "" | Daytona API Key |
| `tavily_api_key` | str | "" | Tavily 搜索 API Key |
| `skills_dir` | Path | None | 技能文件夹路径（可选） |
| `debug` | bool | False | 调试模式 |

### Cerebellum

主类，提供智能代理功能。

#### 方法

##### `run(task: str, files: Optional[List[tuple]] = None) -> Dict[str, Any]`

执行任务。

**参数:**
- `task: str` - 任务描述
- `files: Optional[List[tuple]]` - 可选，上传的文件列表 [(file_data, filename), ...]

**返回:**
```python
{
    "success": bool,           # 是否成功
    "message": str,            # 处理结果文本
    "files": [                 # 沙盒生成的文件列表
        FileData(
            name="file.txt",   # 文件名
            content=b"...",    # 文件内容 (bytes 或 str)
            type="text/plain"  # 文件类型
        )
    ],
    "uploaded_files": [        # 上传的文件信息（如果有）
        {"filename": "doc.pdf", "size": 1024, "content_type": "application/pdf"}
    ]
}
```

### FileData

文件数据类（沙盒下载的文件）。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `name` | str | 文件名 |
| `content` | bytes \| str | 文件内容 |
| `type` | str | MIME 类型 |

#### 方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `get_bytes()` | bytes | 获取字节内容 |
| `get_text()` | str | 获取文本内容 |
| `get_base64()` | str | 获取 Base64 编码 |

### 文件上传和处理

#### FileUploader

文件上传处理器。

```python
from cerebellum import FileUploader, upload_file

uploader = FileUploader(
    max_file_size=10 * 1024 * 1024,  # 10MB
    parse=True,    # 自动解析
    check_security=True  # 安全检查
)

result = uploader.upload(file_data, filename)

if result.success:
    file = result.file
    print(f"文件名: {file.filename}")
    print(f"解析内容: {file.text}")
```

#### FileSecurityChecker

文件安全检查器。

```python
from cerebellum import FileSecurityChecker, check_file

# 检查文件
result = check_file(file_data, filename)

if result.is_safe:
    print("文件安全")
else:
    print(f"不安全: {result.error}")
```

#### FileParserManager

文件解析管理器，支持格式：
- 文本：txt, md, log, rst
- 表格：csv, tsv
- 文档：pdf, docx
- 数据：json, xml, yaml
- 图片：png, jpg, gif, bmp, webp, svg

```python
from cerebellum import parse_file

result = parse_file(file_data, "document.pdf")

if result.is_valid:
    print(result.text)  # 提取的文本
    print(result.metadata)  # 元数据
```

#### CerebellumPlus

支持文件上传的增强版代理。

```python
from cerebellum import CerebellumPlus

with CerebellumPlus(config=config) as agent:
    # 单文件处理
    result = agent.process_file(file_data, "file.pdf", "总结内容")
    
    # 批量处理
    result = agent.process_files(
        files=[
            (file_data1, "file1.pdf"),
            (file_data2, "file2.txt")
        ],
        task="比较这两个文件"
    )
```

## 文件筛选

自动筛选有效文件，排除：
- 隐藏文件（以 `.` 开头）
- 系统目录（`.cache`, `.git`, `node_modules` 等）

保留的文件类型：
- 图片：`png`, `jpg`, `jpeg`, `gif`, `bmp`, `webp`, `svg`, `ico`
- 文档：`pdf`, `doc`, `docx`, `txt`, `md`, `rtf`
- 表格：`csv`, `xlsx`, `xls`, `ods`
- 代码：`py`, `js`, `ts`, `java`, `c`, `cpp`, `go`, `rs`...
- 数据：`json`, `xml`, `yaml`, `toml`...
- 其他：`html`, `css`, `zip`, `tar`, `gz`...

## 命令行工具 (CBot)

```bash
# 执行任务
python cbot.py "帮我写一个Hello World程序"

# 调试模式
python cbot.py --debug "生成一个斐波那契数列图片"

# 指定技能目录
python cbot.py --skills ./custom_skills "搜索明日方舟终末地最新资讯"
```

## 环境变量

在 `.env` 文件中配置：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
DASHSCOPE_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
DASHSCOPE_MODEL=glm-5
DAYTONA_API_KEY=your_daytona_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 完整示例

### 示例1: 基本使用

```python
from cerebellum import Cerebellum, CerebellumConfig
from pathlib import Path

config = CerebellumConfig(
    skills_dir=Path("./skills"),
    debug=True
)

with Cerebellum(config=config) as agent:
    result = agent.run("生成一个斐波那契数列图片")
    
    if result["success"]:
        print(f"结果: {result['message']}")
        
        for file in result["files"]:
            with open(f"output/{file.name}", "wb") as f:
                f.write(file.get_bytes())
```

### 示例2: 文件上传处理

```python
from cerebellum import CerebellumPlus, CerebellumConfig

config = CerebellumConfig(debug=True)

with CerebellumPlus(config=config) as agent:
    # 读取 PDF 文件
    with open("report.pdf", "rb") as f:
        pdf_data = f.read()
    
    # 上传并分析
    result = agent.process_file(
        file_data=pdf_data,
        filename="report.pdf",
        task="请总结这份报告的主要内容，并提取关键数据"
    )
    
    print(f"结果: {result['message']}")
    print(f"生成文件: {result['files']}")
```

### 示例3: 批量文件处理

```python
from cerebellum import CerebellumPlus

files = [
    ("数据1.csv", open("data1.csv", "rb").read()),
    ("数据2.csv", open("data2.csv", "rb").read()),
]

with CerebellumPlus() as agent:
    result = agent.process_files(files, "分析这两个CSV文件的差异")
```

### 示例4: 手动文件解析

```python
from cerebellum import parse_file, check_file

# 安全检查
security = check_file(file_data, "document.pdf")
if not security.is_safe:
    print(f"不安全: {security.error}")
    exit()

# 解析内容
parsed = parse_file(file_data, "document.pdf")
if parsed.is_valid:
    print(f"提取的文本:\n{parsed.text}")
    print(f"元数据: {parsed.metadata}")
```

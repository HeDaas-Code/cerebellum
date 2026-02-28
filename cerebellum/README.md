# Cerebellum 智能代理库

基于 DeepAgents 框架的强大 AI 智能代理库。

## 核心功能

- **阿里百炼 LLM** - 使用 glm-5 模型
- **Daytona 沙盒** - 安全的代码执行环境
- **技能系统** - 加载自定义技能处理特定任务
- **联网搜索** - 支持 Tavily 搜索
- **文件数据返回** - 直接返回文件内容（bytes/str），不保存到本地

## 安装

```bash
pip install -r requirements.txt
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

### 方式3: 配置对象

```python
from cerebellum import Cerebellum, CerebellumConfig

config = CerebellumConfig(
    dashscope_api_key="your_api_key",
    dashscope_model="glm-5",
    daytona_api_key="your_daytona_key",
    tavily_api_key="your_tavily_key",
    skills_dir=Path("./skills"),
    debug=True
)

with Cerebellum(config=config) as agent:
    result = agent.run("帮我搜集关于明日方舟终末地的信息")
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

##### `__init__(config: Optional[CerebellumConfig] = None, debug: bool = False)`

初始化代理。

##### `initialize() -> Cerebellum`

手动初始化（创建 LLM、沙盒、Agent）。通常不需要手动调用，`run()` 会自动初始化。

##### `run(task: str) -> Dict[str, Any]`

执行任务。

**返回:**
```python
{
    "success": bool,           # 是否成功
    "message": str,            # 处理结果文本
    "files": [                 # 文件列表
        FileData(
            name="file.txt",   # 文件名
            content=b"...",    # 文件内容 (bytes 或 str)
            type="text/plain"  # 文件类型
        )
    ]
}
```

##### `__enter__() / __exit__()`

上下文管理器支持。

### FileData

文件数据类。

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

### run_task()

便捷函数，快速执行任务。

```python
def run_task(
    task: str,
    config: Optional[CerebellumConfig] = None,
    debug: bool = False
) -> Dict[str, Any]
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

项目提供了命令行工具 `cbot.py`：

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

```python
from cerebellum import Cerebellum, CerebellumConfig
from pathlib import Path

# 配置
config = CerebellumConfig(
    skills_dir=Path("./skills"),
    debug=True
)

# 执行任务
with Cerebellum(config=config) as agent:
    result = agent.run("生成一个斐波那契数列图片")
    
    if result["success"]:
        print(f"结果: {result['message']}")
        
        # 处理返回的文件
        for file in result["files"]:
            print(f"\n文件: {file.name}")
            print(f"类型: {file.type}")
            print(f"大小: {len(file.get_bytes())} bytes")
            
            # 保存到本地
            with open(f"output/{file.name}", "wb") as f:
                f.write(file.get_bytes())
```

## 注意事项

1. **文件不保存到本地** - `run()` 返回 `FileData` 对象，需要手动保存
2. **沙盒自动清理** - 任务完成后自动清理沙盒资源
3. **技能目录** - 可通过 `skills_dir` 指定自定义技能文件夹

"""
Deep Agents 智能代理主程序

该程序创建一个基于 deepagents 的智能代理，具备以下能力：
1. 使用阿里百炼 LLM 作为语言模型
2. 使用 Daytona 沙盒执行代码和命令
3. 加载预设的技能（Skills）来处理各种任务

用法:
    python agent.py                    # 交互模式
    python agent.py "任务描述"          # 执行单个任务
    python agent.py --debug            # 调试模式
    python agent.py --debug "任务描述"  # 调试模式执行任务
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

try:
    from daytona import Daytona
    from daytona_sdk.common.daytona import DaytonaConfig
    from langchain_daytona import DaytonaSandbox
except ImportError:
    print("警告: Daytona SDK 未安装，将使用本地后端")
    Daytona = None
    DaytonaConfig = None
    DaytonaSandbox = None

try:
    from tavily import TavilyClient
except ImportError:
    print("警告: Tavily SDK 未安装，将无法使用联网搜索")
    TavilyClient = None

try:
    from deepagents import create_deep_agent
    from deepagents.backends.utils import create_file_data
except ImportError:
    print("错误: deepagents 未安装，请运行: pip install deepagents")
    sys.exit(1)


# 技能目录路径
SKILLS_DIR = Path(__file__).parent / "skills"

# 全局变量存储加载的技能文件和后端
skills_files_cache = {}
backend_cache = None


def load_environment():
    """加载环境变量配置"""
    load_dotenv()
    
    # 检查必要的环境变量
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_key or dashscope_key == "your_dashscope_api_key_here":
        print("警告: DASHSCOPE_API_KEY 未设置，请在 .env 文件中配置")
    
    return {
        "dashscope_api_key": dashscope_key or os.getenv("DASHSCOPE_API_KEY", ""),
        "dashscope_base_url": os.getenv("DASHSCOPE_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1"),
        "dashscope_model": os.getenv("DASHSCOPE_MODEL", "glm-5"),
        "daytona_api_key": os.getenv("DAYTONA_API_KEY", ""),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
    }


def create_llm(config: Dict[str, str]) -> ChatOpenAI:
    """
    创建阿里百炼 LLM 客户端
    
    Args:
        config: 包含 API 配置的字典
    
    Returns:
        ChatOpenAI 实例
    """
    return ChatOpenAI(
        model=config["dashscope_model"],
        base_url=config["dashscope_base_url"],
        api_key=config["dashscope_api_key"],
        temperature=0.7,
        max_tokens=4096,
    )


def create_sandbox_backend(config: Dict[str, str]):
    """
    创建 Daytona 沙盒后端
    
    Args:
        config: 包含 Daytona API 密钥的字典
    
    Returns:
        tuple: (后端实例, 原始沙盒实例)，如果失败则返回 (None, None)
    """
    if Daytona is None or DaytonaSandbox is None or DaytonaConfig is None:
        print(" Daytona SDK 不可用，将使用默认后端")
        return None, None
    
    try:
        # 使用 DaytonaConfig 配置
        daytona_config = DaytonaConfig(
            api_key=config["daytona_api_key"] if config["daytona_api_key"] else None
        )
        daytona = Daytona(config=daytona_config)
        
        # 创建沙盒实例
        print("  正在创建沙盒...")
        sandbox = daytona.create()
        
        # 等待沙盒完全启动
        print("  等待沙盒启动...")
        sandbox.wait_for_sandbox_start(timeout=60)
        print(f"  沙盒状态: {sandbox.state}")
        
        # 获取沙盒工作目录
        work_dir = sandbox.get_work_dir()
        print(f"  沙盒工作目录: {work_dir}")
        
        # 初始化沙盒环境：创建 workspace 目录（相对于工作目录）
        print("  正在初始化沙盒环境...")
        # 使用相对路径 "workspace"（Daytona 会自动解析为 /home/[username]/workspace）
        init_result = sandbox._process.exec("mkdir -p workspace && chmod 755 workspace", timeout=30)
        
        # 验证 workspace 目录创建成功
        verify_result = sandbox._process.exec("ls -la workspace", timeout=30)
        if verify_result.exit_code == 0:
            print(f"  ✓ workspace 目录初始化完成")
            logging.info(f"沙盒 workspace 目录初始化成功: {verify_result.result.strip()}")
        else:
            logging.warning(f"沙盒 workspace 目录验证失败: {verify_result.result}")
        
        # 创建沙盒后端 (langchain 封装)
        backend = DaytonaSandbox(sandbox=sandbox)
        
        print(f"✓ Daytona 沙盒创建成功 (ID: {sandbox.id})")
        # 返回 backend 和原始 sandbox 对象
        return backend, sandbox
        
    except Exception as e:
        print(f"✗ 创建 Daytona 沙盒失败: {e}")
        print("  将使用默认后端")
        return None, None


def cleanup_sandbox(sandbox: Any, backend: Any = None, agent_response: str = ""):
    """
    清理沙盒资源，并在沙盒关闭前检查并下载输出文件
    
    Args:
        sandbox: Daytona 沙盒实例
        backend: DaytonaSandbox 后端实例（可选，用于下载文件）
        agent_response: 智能体的回复内容（用于提取文件路径）
    """
    if sandbox is not None:
        try:
            print("\n正在检查输出文件...")
            # 检查本地 output 目录
            local_output = Path("output")
            files_in_output = list(local_output.glob("*")) if local_output.exists() else []
            downloaded_files = []
            
            if not files_in_output:
                # 方法1: 尝试使用 Daytona SDK 的 fs.download_file 方法
                print("  方法1: 尝试使用 Daytona SDK 下载...")
                try:
                    # 列出沙盒中的文件
                    files = sandbox.fs.list_files("output")
                    if not files:
                        files = sandbox.fs.list_files("workspace")
                    
                    if files:
                        local_output.mkdir(exist_ok=True)
                        print(f"  发现 {len(files)} 个文件，使用 SDK 下载...")
                        
                        for f in files:
                            if not f.is_dir:
                                try:
                                    # 使用 SDK 下载文件
                                    content = sandbox.fs.download_file(f"output/{f.name}" if not f.name.startswith('/') else f.name)
                                    local_path = local_output / f.name
                                    with open(local_path, 'wb') as wf:
                                        wf.write(content)
                                    print(f"  ✓ SDK下载成功: {f.name}")
                                    downloaded_files.append(f.name)
                                except Exception as sdk_err:
                                    print(f"  SDK下载 {f.name} 失败: {sdk_err}, 尝试 exec 方法")
                                    # 降级到 exec 方法
                                    try:
                                        file_path = f"/home/daytona/output/{f.name}"
                                        content_result = sandbox._process.exec(f"cat '{file_path}'", timeout=30)
                                        cstdout = content_result.stdout if hasattr(content_result, 'stdout') else content_result.output if hasattr(content_result, 'output') else ""
                                        
                                        if f.name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                            b64_result = sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=30)
                                            bstdout = b64_result.stdout if hasattr(b64_result, 'stdout') else b64_result.output if hasattr(b64_result, 'output') else ""
                                            import base64
                                            content = base64.b64decode(bstdout.strip())
                                            with open(local_path, 'wb') as wf:
                                                wf.write(content)
                                        else:
                                            with open(local_path, 'w', encoding='utf-8') as wf:
                                                wf.write(cstdout)
                                        print(f"  ✓ Exec下载成功: {f.name}")
                                        downloaded_files.append(f.name)
                                    except Exception as exec_err:
                                        print(f"  ✗ 下载失败 {f.name}: {exec_err}")
                        
                        if downloaded_files:
                            print(f"\n✓ 文件已保存到 ./output/ 文件夹")
                            return
                except Exception as sdk_err:
                    print(f"  SDK 方法失败: {sdk_err}")
                
                # 方法2: 从智能体回复中提取文件路径
                if agent_response and not downloaded_files:
                    print("  方法2: 从智能体回复中提取文件路径...")
                    import re
                    # 匹配常见文件路径格式
                    patterns = [
                        r'/home/daytona/([^\s\)\]\.]+\.(?:png|jpg|jpeg|pdf|csv|xlsx|txt|py))',
                        r'output/([^\s\)\]\.]+\.(?:png|jpg|jpeg|pdf|csv|xlsx|txt|py))',
                        r'文件[路径名]*[：:]\s*(/[^\s]+)',
                    ]
                    
                    found_paths = []
                    for pattern in patterns:
                        matches = re.findall(pattern, agent_response, re.IGNORECASE)
                        found_paths.extend(matches)
                    
                    if found_paths:
                        local_output.mkdir(exist_ok=True)
                        print(f"  从回复中找到 {len(found_paths)} 个文件路径...")
                        
                        for file_path in found_paths:
                            try:
                                # 标准化路径
                                if not file_path.startswith('/'):
                                    file_path = f"/home/daytona/{file_path}"
                                if not file_path.startswith('/home/daytona/'):
                                    file_path = f"/home/daytona/{file_path}"
                                
                                file_name = Path(file_path).name
                                local_path = local_output / file_name
                                
                                # 下载文件
                                content_result = sandbox._process.exec(f"cat '{file_path}'", timeout=30)
                                cstdout = content_result.stdout if hasattr(content_result, 'stdout') else content_result.output if hasattr(content_result, 'output') else ""
                                
                                if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                    b64_result = sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=30)
                                    bstdout = b64_result.stdout if hasattr(b64_result, 'stdout') else b64_result.output if hasattr(b64_result, 'output') else ""
                                    import base64
                                    content = base64.b64decode(bstdout.strip())
                                    with open(local_path, 'wb') as wf:
                                        wf.write(content)
                                else:
                                    with open(local_path, 'w', encoding='utf-8') as wf:
                                        wf.write(cstdout)
                                
                                print(f"  ✓ 从回复下载成功: {file_name}")
                                downloaded_files.append(file_name)
                            except Exception as e:
                                print(f"  ✗ 下载失败 {file_path}: {e}")
                        
                        if downloaded_files:
                            print(f"\n✓ 文件已保存到 ./output/ 文件夹")
                            return
                
                # 方法3: 直接搜索沙盒目录
                if not downloaded_files:
                    print("  方法3: 搜索沙盒目录...")
                    try:
                        file_result = sandbox._process.exec("find /home/daytona -type f 2>/dev/null | head -20", timeout=30)
                        fstdout = file_result.stdout if hasattr(file_result, 'stdout') else file_result.output if hasattr(file_result, 'output') else ""
                        files = [f.strip() for f in fstdout.strip().split('\n') if f.strip()]
                        
                        if files:
                            local_output.mkdir(exist_ok=True)
                            print(f"  发现 {len(files)} 个文件...")
                            
                            for file_path in files:
                                try:
                                    file_name = Path(file_path).name
                                    local_path = local_output / file_name
                                    
                                    content_result = sandbox._process.exec(f"cat '{file_path}'", timeout=30)
                                    cstdout = content_result.stdout if hasattr(content_result, 'stdout') else content_result.output if hasattr(content_result, 'output') else ""
                                    
                                    if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                        b64_result = sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=30)
                                        bstdout = b64_result.stdout if hasattr(b64_result, 'stdout') else b64_result.output if hasattr(b64_result, 'output') else ""
                                        import base64
                                        content = base64.b64decode(bstdout.strip())
                                        with open(local_path, 'wb') as wf:
                                            wf.write(content)
                                    else:
                                        with open(local_path, 'w', encoding='utf-8') as wf:
                                            wf.write(cstdout)
                                    
                                    print(f"  ✓ 下载成功: {file_name}")
                                    downloaded_files.append(file_name)
                                except Exception as e:
                                    print(f"  ✗ 下载失败 {file_path}: {e}")
                            
                            if downloaded_files:
                                print(f"\n✓ 文件已保存到 ./output/ 文件夹")
                    except Exception as e:
                        print(f"  搜索失败: {e}")
                
                if not downloaded_files:
                    print("  未发现需要交付的文件")
            else:
                print(f"  本地已有 {len(files_in_output)} 个文件，无需下载")
            
            # 清理沙盒资源
            print("\n正在清理沙盒资源...")
            sandbox.stop()
            print("✓ 沙盒已停止")
        except Exception as e:
            print(f"  警告: 清理沙盒时出错: {e}")


def download_sandbox_files(sandbox: Any, backend: Any = None, output_dir: Path = None) -> List[Path]:
    """
    下载沙盒中的文件到本地
    
    Args:
        sandbox: Daytona 沙盒实例（原始对象）
        backend: DaytonaSandbox 后端实例（可选）
        output_dir: 输出目录（默认为 ./output）
    
    Returns:
        下载的文件路径列表
    """
    if sandbox is None and backend is None:
        return []
    
    if output_dir is None:
        output_dir = Path.cwd() / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    
    # 使用 backend 如果可用，否则使用 sandbox
    executor = backend if backend is not None else sandbox
    
    try:
        # 使用相对路径列出 workspace 目录中的文件（Daytona 会自动解析为工作目录下的 workspace）
        result = executor.execute("find workspace -type f 2>/dev/null")
        
        # 处理不同的返回格式
        stdout = result.stdout if hasattr(result, 'stdout') else result.output if hasattr(result, 'output') else ""
        
        if not stdout:
            logging.info("沙盒中没有找到文件")
            return []
        
        # 解析文件列表
        remote_files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]
        
        if not remote_files:
            logging.info("沙盒中没有找到文件")
            return []
        
        logging.info(f"开始下载 {len(remote_files)} 个文件...")
        
        # 确定使用哪个对象来下载文件
        fs_target = sandbox if sandbox is not None else getattr(backend, '_sandbox', None)
        
        if fs_target is None:
            logging.warning("无法获取沙盒文件系统对象")
            return []
        
        for remote_path in remote_files:
            try:
                # 下载文件
                content = fs_target.fs.download_file(remote_path)
                local_path = output_dir / Path(remote_path).name
                local_path.write_bytes(content)
                downloaded_files.append(local_path)
                logging.info(f"  已下载: {remote_path} -> {local_path}")
            except Exception as e:
                logging.warning(f"  下载失败 {remote_path}: {e}")
        
        if downloaded_files:
            print(f"\n✓ 已下载 {len(downloaded_files)} 个文件到: {output_dir}")
        else:
            print("\n没有文件需要下载")
        
        return downloaded_files
        
    except Exception as e:
        logging.warning(f"下载沙盒文件时出错: {e}")
        return []


def list_available_skills() -> List[str]:
    """
    列出可用的技能目录
    
    Returns:
        技能名称列表
    """
    if not SKILLS_DIR.exists():
        return []
    
    skills = []
    for item in SKILLS_DIR.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item.name)
    
    return skills


def load_skills_files(skills_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载 skills 目录中的文件，供 StateBackend 使用
    
    Args:
        skills_path: 技能目录路径
    
    Returns:
        格式化的 files 字典
    """
    target_path = skills_path if skills_path and skills_path.exists() else SKILLS_DIR
    
    if not target_path.exists():
        logging.warning(f"技能目录不存在: {target_path}")
        return {}
    
    skills_files = {}
    for skill_dir in target_path.iterdir():
        if not skill_dir.is_dir():
            continue
        
        skill_name = skill_dir.name
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(target_path)
                virtual_path = f"/skills/{skill_name}/{relative_path}".replace("\\", "/")
                
                try:
                    content = file_path.read_text(encoding="utf-8")
                    skills_files[virtual_path] = create_file_data(content)
                except Exception as e:
                    logging.warning(f"无法读取技能文件 {file_path}: {e}")
    
    logging.info(f"已加载 {len(skills_files)} 个技能文件")
    return skills_files


def create_agent(
    llm: ChatOpenAI,
    backend: Optional[Any] = None,
    skills_path: Optional[Path] = None,
    debug: bool = False,
) -> Any:
    """
    创建 Deep Agent 智能代理
    
    Args:
        llm: 语言模型实例
        backend: 沙盒后端（可选）
        skills_path: 技能目录路径（可选）
        debug: 是否启用调试模式
    
    Returns:
        配置好的 deep agent 实例
    """
    # 系统提示词
    system_prompt = """你是一个强大的 AI 智能代理，具有以下增强能力：

## 核心能力
1. **文件操作**: 读取、创建、编辑文件和目录
2. **代码执行**: 在沙盒环境中执行代码 - 这是你的核心能力！
3. **文档处理**: 使用专业技能处理 Word、PDF、PowerPoint、Excel 等文档
4. **问题解答**: 回答问题并提供解决方案
5. **Python脚本执行**: 可以创建、编写和执行 Python 脚本来辅助完成复杂计算、数据处理或自动化操作

## 意图-能力识别（重要！）
在开始执行任务前，你必须先分析用户的**最终输出需求**，然后选择最合适的交付方式：

### 输出类型识别与响应策略

| 用户需求类型 | 识别关键词 | 正确响应方式 |
|-------------|----------|-------------|
| **图片/图表** | 图片、图表、绘图、可视化、画图、生成图片 | 在沙盒中用 Python(matplotlib/pyplot) 生成图片，返回**图片文件路径** |
| **PDF文档** | PDF、报告、文档 | 在沙盒中生成 PDF 文件，返回**文件路径** |
| **代码文件** | 代码、脚本、程序 | 在沙盒中创建代码文件，返回**文件路径** |
| **数据/表格** | 数据、表格、CSV、Excel | 在沙盒中生成数据文件，返回**文件路径** |
| **文本内容** | 文本、文字、内容 | 如果简单直接返回；如果复杂生成文件返回 |
| **执行结果** | 运行、执行、计算结果 | 在沙盒中执行代码，返回**执行结果** |

### 关键原则
- **用户要什么就给什么**：如果用户说要图片，你就给图片文件路径，不是文字描述
- **不要过度解释**：用户不需要知道过程，只需要结果
- **文件优先**：如果任务涉及生成图片/文件，优先在沙盒中生成
- **必须交付给用户**：生成文件后，你必须**从沙盒下载文件到本地 ./output 文件夹**，然后告知用户文件已保存！这是你的核心职责！
- **可用工具**：execute(执行Python) + write_file(创建文件) + 读取文件内容交付给用户

## 增强功能

### 1. 任务分析与拆分
- 接收用户任务后，进行系统性分析
- 将复杂任务拆分为可执行的子任务
- 明确各子任务间的依赖关系和执行顺序

### 2. 工具与技能规划
- 针对拆分后的子任务，自动规划所需使用的工具和技能
- 形成详细的执行方案

### 3. TODO列表生成
- 为每个子任务创建清晰、具体的TODO项目
- 包含任务描述、预期结果

### 4. 分步执行机制
- 按照规划的步骤逐步执行任务
- 在每个步骤完成后进行结果验证
- 确保符合预期后再进入下一阶段

### 5. 技能扩展功能
- 当现有技能库(Skill)不足以完成当前任务时
- 自动调用 skill-creator 技能创建新的定制化技能
- 确保任务能够顺利完成

### 6. 文档处理专项要求
- 所有涉及文档处理的任务
- 必须在沙盒环境中创建并交付符合指定格式要求的文档
- 确保文档内容准确、结构完整

### 7. 代码质量标准
- 最终交付的新代码应达到高质量水平
- 包括：代码规范、模块化设计、性能优化、错误处理和注释完整性

## 重要提示
- 当用户要求创建文件、执行代码、或处理任何需要生成文件的任务时，你必须使用沙盒环境
- 使用 execute 工具在沙盒中执行命令
- 使用 write_file 工具在沙盒中创建文件
- **任务完成后，如果需要将文件交给用户，你必须从沙盒中取回文件！**
- 沙盒的工作目录是 /home/daytona，请在此目录下操作文件

## 可用工具
- execute: 执行 shell 命令（包括 Python 脚本）
- write_file: 在沙盒中创建文件
- read_file: 读取沙盒中的文件
- ls: 列出目录内容
- glob: 查找文件
- grep: 搜索文件内容
- write_todos: 管理 TODO 列表

## 文件下载说明
- 沙盒中的文件路径是 `/home/daytona/xxx`
- 下载文件到本地：`read_file` 读取文件后你可以将内容保存到本地
- 本地输出目录：`./output`（当前目录下的 output 文件夹）
- 例如：沙盒中 `/home/daytona/fibonacci.png` → 下载到 `./output/fibonacci.png`

## 技能使用
- 你可以通过加载技能（Skills）来处理特定领域的任务
- **重要**：文件交付技能(file-delivery)会自动加载，当你在沙盒中生成文件后，必须使用此技能将文件保存到本地 ./output 文件夹
- 当需要处理文档时，系统会自动选择合适的技能来帮助你
- 如果发现现有的技能无法满足用户需求，使用 skill-creator 技能创建新的自定义技能

## 联网搜索
- 当你需要获取最新信息、实时数据或不确定的知识时，可以使用 Tavily 搜索工具
- Tavily 可以搜索互联网获取准确的信息
- **重要**：对于需要准确信息的任务（如游戏攻略、数据查询等），必须先使用联网搜索获取准确信息，再进行后续处理

请始终以专业、友好的方式与用户交互。"""
    
    # 构建 agent 配置
    agent_kwargs = {
        "model": llm,
        "system_prompt": system_prompt,
        "debug": debug,
    }
    
    # 添加沙盒后端
    if backend is not None:
        agent_kwargs["backend"] = backend
    
    # 添加 Tavily 联网搜索工具
    try:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if tavily_api_key:
            tavily_tool = TavilySearchResults(api_key=tavily_api_key, max_results=5)
            agent_kwargs["tools"] = [tavily_tool]
            print("✓ 已加载 Tavily 联网搜索工具")
            logging.info("已加载 Tavily 联网搜索工具")
    except Exception as e:
        print(f"警告: 无法加载 Tavily 搜索工具: {e}")
    
    # 添加技能目录
    # 注意: deepagents 需要 POSIX 风格的路径 (使用正斜杠)
    skills_posix_path = str(skills_path).replace("\\", "/") if skills_path else None
    
    if skills_path is not None and skills_path.exists():
        agent_kwargs["skills"] = [skills_posix_path]
        print(f"✓ 已加载技能目录: {skills_path}")
        logging.info(f"加载技能目录: {skills_posix_path}")
    else:
        # 使用默认技能目录
        if SKILLS_DIR.exists():
            default_skills_path = str(SKILLS_DIR).replace("\\", "/")
            agent_kwargs["skills"] = [default_skills_path]
            print(f"✓ 已加载技能目录: {SKILLS_DIR}")
            logging.info(f"加载技能目录: {default_skills_path}")
    
    # 创建 agent
    agent = create_deep_agent(**agent_kwargs)
    
    return agent


def run_interactive_mode(agent: Any, sandbox: Any = None):
    """
    运行交互模式
    
    Args:
        agent: 已配置的 deep agent 实例
        sandbox: Daytona 沙盒实例（可选，用于清理）
    """
    print("\n" + "=" * 60)
    print("  Deep Agents 智能代理 - 交互模式")
    print("=" * 60)
    print("输入你的问题或任务，按 Enter 发送")
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    thread_id = "default"
    
    try:
        while True:
            try:
                user_input = input("\n用户 > ").strip()
                
                if user_input.lower() in ["quit", "exit", "退出"]:
                    print("再见!")
                    break
                
                if not user_input:
                    continue
                
                # 调用 agent 处理请求
                print("\n[正在处理...]")
                result = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_input
                            }
                        ],
                        "files": skills_files_cache
                    },
                    config={"configurable": {"thread_id": thread_id}}
                )
                
                # 输出响应
                if result and "messages" in result:
                    last_message = result["messages"][-1]
                    if hasattr(last_message, "content"):
                        print(f"\n助手 > {last_message.content}")
                    else:
                        print(f"\n助手 > {last_message}")
                        
            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                print(f"\n错误: {e}")
    finally:
        # 清理沙盒资源（交互模式下不传递回复内容）
        cleanup_sandbox(sandbox, agent_response="")


def run_single_task(agent: Any, task: str, sandbox: Any = None):
    """
    运行单个任务
    
    Args:
        agent: 已配置的 deep agent 实例
        task: 要执行的任务描述
        sandbox: Daytona 沙盒实例（可选，用于清理）
    """
    print(f"\n执行任务: {task}\n")
    agent_response = ""
    
    try:
        print("[正在处理...]")
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": task
                    }
                ],
                "files": skills_files_cache
            }
        )
        
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                agent_response = last_message.content
                print(f"\n结果:\n{last_message.content}")
            else:
                agent_response = str(last_message)
                print(f"\n结果:\n{last_message}")
                
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 清理沙盒资源
        cleanup_sandbox(sandbox, agent_response=agent_response)


def main():
    """主函数"""
    # 解析命令行参数
    debug_mode = "--debug" in sys.argv
    
    # 移除 --debug 参数
    sys.argv = [arg for arg in sys.argv if arg != "--debug"]
    
    # 配置日志
    if debug_mode:
        # 创建日志目录
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # 生成日志文件名（包含时间戳）
        from datetime import datetime
        log_file = log_dir / f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 配置日志：同时输出到控制台和文件
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        print(f"✓ 调试模式已启用")
        print(f"✓ 日志文件: {log_file}\n")
    
    print("正在初始化 Deep Agents 智能代理...\n")
    
    # 加载配置
    config = load_environment()
    
    # 列出可用技能
    available_skills = list_available_skills()
    if available_skills:
        print(f"可用技能: {', '.join(available_skills)}")
    
    # 创建 LLM
    print("\n正在连接阿里百炼 LLM...")
    llm = create_llm(config)
    print(f"  模型: {config['dashscope_model']}")
    print(f"  API: {config['dashscope_base_url']}")
    
    # 创建沙盒后端
    print("\n正在创建沙盒环境...")
    logging.info("开始创建沙盒环境...")
    backend, sandbox = create_sandbox_backend(config)
    global backend_cache
    backend_cache = backend
    
    if backend is None:
        logging.warning("沙盒创建失败，将使用默认后端")
    else:
        logging.info(f"沙盒创建成功，ID: {sandbox.id}")
    
    # 加载技能文件（供 StateBackend 使用）
    global skills_files_cache
    skills_files_cache = load_skills_files(SKILLS_DIR)
    
    # 创建 Agent (启用调试模式)
    print("\n正在初始化 Agent...")
    agent = create_agent(llm, backend, SKILLS_DIR, debug=debug_mode)
    print("✓ Agent 初始化完成\n")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 执行单个任务
        task = " ".join(sys.argv[1:])
        run_single_task(agent, task, sandbox)
    else:
        # 交互模式
        run_interactive_mode(agent, sandbox)


if __name__ == "__main__":
    main()

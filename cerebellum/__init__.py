"""
Cerebellum 智能代理库

该库提供一个强大的 AI 智能代理，具备以下能力：
1. 使用阿里百炼 LLM 作为语言模型
2. 使用 Daytona 沙盒执行代码和命令
3. 加载预设的技能（Skills）来处理各种任务
4. 支持 Tavily 联网搜索

用法:
    from cerebellum import Cerebellum, CerebellumConfig
    
    config = CerebellumConfig(
        skills_dir=Path("./skills"),  # 可选：指定技能文件夹
        debug=True
    )
    
    agent = Cerebellum(config=config)
    result = agent.run("你的任务描述")
    
    # result 包含:
    # {
    #     "success": True,
    #     "message": "处理结果...",
    #     "files": [
    #         {"name": "file.txt", "content": b"文件内容", "type": "text"}
    #     ]
    # }
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from base64 import b64encode, b64decode

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

try:
    from daytona import Daytona
    from daytona_sdk.common.daytona import DaytonaConfig
    from langchain_daytona import DaytonaSandbox
except ImportError:
    print("警告: Daytona SDK 未安装，将使用默认后端")
    Daytona = None
    DaytonaConfig = None
    DaytonaSandbox = None

try:
    from deepagents import create_deep_agent
    from deepagents.backends.utils import create_file_data
except ImportError:
    print("错误: deepagents 未安装，请运行: pip install deepagents")
    sys.exit(1)


# 默认技能目录路径
DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class FileData:
    """文件数据类"""
    name: str
    content: Union[bytes, str]
    type: str = "text"
    
    def __post_init__(self):
        if isinstance(self.content, bytes):
            self.type = "binary"
        else:
            self.type = "text"
    
    def get_bytes(self) -> bytes:
        """获取字节内容"""
        if isinstance(self.content, bytes):
            return self.content
        return self.content.encode('utf-8')
    
    def get_text(self) -> str:
        """获取文本内容"""
        if isinstance(self.content, str):
            return self.content
        return self.content.decode('utf-8')
    
    def get_base64(self) -> str:
        """获取 Base64 编码内容"""
        return b64encode(self.get_bytes()).decode('ascii')


@dataclass
class CerebellumConfig:
    """Cerebellum 配置类"""
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    dashscope_model: str = "glm-5"
    daytona_api_key: str = ""
    tavily_api_key: str = ""
    skills_dir: Optional[Path] = None
    debug: bool = False


class Cerebellum:
    """
    Cerebellum 智能代理类
    
    提供一个强大的 AI 智能代理，支持：
    - 阿里百炼 LLM
    - Daytona 沙盒
    - 技能系统
    - 联网搜索
    - 返回文件数据而非本地保存
    """
    
    def __init__(self, config: Optional[CerebellumConfig] = None, debug: bool = False):
        """
        初始化 Cerebellum 智能代理
        
        Args:
            config: CerebellumConfig 配置对象
            debug: 是否启用调试模式
        """
        self.debug = debug or (config.debug if config else False)
        self.config = config or CerebellumConfig()
        
        load_dotenv()
        
        if not self.config.dashscope_api_key:
            self.config.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.config.dashscope_base_url:
            self.config.dashscope_base_url = os.getenv("DASHSCOPE_BASE_URL", "https://coding.dashscope.aliyuncs.com/v1")
        if not self.config.dashscope_model:
            self.config.dashscope_model = os.getenv("DASHSCOPE_MODEL", "glm-5")
        if not self.config.daytona_api_key:
            self.config.daytona_api_key = os.getenv("DAYTONA_API_KEY", "")
        if not self.config.tavily_api_key:
            self.config.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        
        self.llm = None
        self.backend = None
        self.sandbox = None
        self.agent = None
        self.skills_files = {}
        
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志"""
        if self.debug:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            log_file = log_dir / f"cerebellum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            
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
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 客户端"""
        return ChatOpenAI(
            model=self.config.dashscope_model,
            base_url=self.config.dashscope_base_url,
            api_key=self.config.dashscope_api_key,
            temperature=0.7,
            max_tokens=4096,
        )
    
    def _create_sandbox(self):
        """创建沙盒"""
        if Daytona is None:
            print("警告: Daytona SDK 不可用")
            return None, None
        
        try:
            daytona_config = DaytonaConfig(
                api_key=self.config.daytona_api_key if self.config.daytona_api_key else None
            )
            daytona = Daytona(config=daytona_config)
            
            print("  正在创建沙盒...")
            sandbox = daytona.create()
            
            print("  等待沙盒启动...")
            sandbox.wait_for_sandbox_start(timeout=60)
            print(f"  沙盒状态: {sandbox.state}")
            
            work_dir = sandbox.get_work_dir()
            print(f"  沙盒工作目录: {work_dir}")
            
            print("  正在初始化沙盒环境...")
            sandbox._process.exec("mkdir -p workspace && chmod 755 workspace", timeout=30)
            
            backend = DaytonaSandbox(sandbox=sandbox)
            
            print(f"✓ Daytona 沙盒创建成功 (ID: {sandbox.id})")
            return backend, sandbox
            
        except Exception as e:
            print(f"✗ 创建沙盒失败: {e}")
            return None, None
    
    def _load_skills_files(self, skills_path: Path) -> Dict[str, Any]:
        """加载技能文件"""
        if not skills_path.exists():
            logging.warning(f"技能目录不存在: {skills_path}")
            return {}
        
        skills_files = {}
        for skill_dir in skills_path.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_name = skill_dir.name
            for file_path in skill_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(skills_path)
                    virtual_path = f"/skills/{skill_name}/{relative_path}".replace("\\", "/")
                    
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        skills_files[virtual_path] = create_file_data(content)
                    except Exception as e:
                        logging.warning(f"无法读取技能文件 {file_path}: {e}")
        
        logging.info(f"已加载 {len(skills_files)} 个技能文件")
        return skills_files
    
    def _get_file_type(self, filename: str) -> str:
        """根据文件名判断文件类型"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        image_exts = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico']
        if ext in image_exts:
            return f"image/{ext}"
        return "text/plain"
    
    def _is_valid_file(self, filename: str) -> bool:
        """检查是否为有效的用户文件（排除系统文件）"""
        # 排除以点开头的隐藏文件
        if filename.startswith('.'):
            return False
        
        # 排除常见系统目录
        exclude_dirs = ['.cache', '.local', '.npm', '.git', 'node_modules', '__pycache__', '.venv']
        for ex_dir in exclude_dirs:
            if ex_dir in filename:
                return False
        
        # 只保留有意义的文件后缀
        valid_extensions = [
            # 图片
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico',
            # 文档
            'pdf', 'doc', 'docx', 'txt', 'md', 'rtf', 'odt',
            # 表格
            'csv', 'xlsx', 'xls', 'ods'
        ]
        
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return ext in valid_extensions
    
    def _download_files_from_sandbox(self) -> List[FileData]:
        """从沙盒下载文件并返回 FileData 列表"""
        files_data = []
        
        if self.sandbox is None:
            return files_data
        
        try:
            print("\n正在获取沙盒文件列表...")
            downloaded_names = set()
            
            # 方法1: 使用 exec 命令搜索 /home/daytona 目录
            print("  使用 exec 方法搜索文件...")
            file_result = self.sandbox._process.exec("find /home/daytona -type f 2>/dev/null | head -30", timeout=30)
            stdout = file_result.stdout if hasattr(file_result, 'stdout') else file_result.output if hasattr(file_result, 'output') else ""
            files = [f.strip() for f in stdout.strip().split('\n') if f.strip()]
            
            if files:
                print(f"  发现 {len(files)} 个文件...")
                for file_path in files:
                    try:
                        file_name = Path(file_path).name
                        
                        # 筛选有效文件
                        if not self._is_valid_file(file_name):
                            continue
                        
                        if file_name in downloaded_names:
                            continue
                        
                        # 读取文件内容
                        content_result = self.sandbox._process.exec(f"cat '{file_path}'", timeout=30)
                        cstdout = content_result.stdout if hasattr(content_result, 'stdout') else content_result.output if hasattr(content_result, 'output') else ""
                        
                        # 根据文件类型处理
                        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            b64_result = self.sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=30)
                            bstdout = b64_result.stdout if hasattr(b64_result, 'stdout') else b64_result.output if hasattr(b64_result, 'output') else ""
                            content = b64decode(bstdout.strip())
                        else:
                            content = cstdout
                        
                        files_data.append(FileData(
                            name=file_name,
                            content=content,
                            type=self._get_file_type(file_name)
                        ))
                        downloaded_names.add(file_name)
                        print(f"  ✓ 已读取: {file_name}")
                        
                    except Exception as e:
                        print(f"  ✗ 读取失败 {file_path}: {e}")
            
            # 如果 exec 方法没有找到文件，尝试使用 SDK 方法
            if not files_data:
                print("  尝试使用 SDK 方法...")
                try:
                    # 尝试列出 output 和 workspace 目录
                    for dir_name in ["output", "workspace", ""]:
                        try:
                            files = self.sandbox.fs.list_files(dir_name) if dir_name else self.sandbox.fs.list_files(".")
                            for f in files:
                                # 筛选有效文件
                                if not self._is_valid_file(f.name):
                                    continue
                                if f.is_dir or f.name in downloaded_names:
                                    continue
                                try:
                                    # 构建完整路径
                                    file_path = f"/home/daytona/{dir_name}/{f.name}" if dir_name else f"/home/daytona/{f.name}"
                                    content = self.sandbox.fs.download_file(file_path)
                                    files_data.append(FileData(
                                        name=f.name,
                                        content=content,
                                        type=self._get_file_type(f.name)
                                    ))
                                    downloaded_names.add(f.name)
                                    print(f"  ✓ SDK下载成功: {f.name}")
                                except Exception as sdk_err:
                                    print(f"  SDK下载 {f.name} 失败: {sdk_err}")
                        except Exception:
                            continue
                except Exception as sdk_err:
                    print(f"  SDK 方法失败: {sdk_err}")
                        
        except Exception as e:
            print(f"  获取文件失败: {e}")
        
        return files_data
    
    def _cleanup_sandbox(self):
        """清理沙盒资源"""
        if self.sandbox is None:
            return
        
        try:
            print("\n正在清理沙盒资源...")
            self.sandbox.stop()
            self.sandbox.delete()
            print("✓ 沙盒已停止并删除")
            self.sandbox = None
        except Exception as e:
            print(f"  警告: 清理沙盒时出错: {e}")
    
    def _create_agent(self):
        """创建 Agent"""
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
| **图片/图表** | 图片、图表、绘图、可视化、画图、生成图片 | 在沙盒中用 Python(matplotlib/pyplot) 生成图片 |
| **PDF文档** | PDF、报告、文档 | 在沙盒中生成 PDF 文件 |
| **代码文件** | 代码、脚本、程序 | 在沙盒中创建代码文件 |
| **数据/表格** | 数据、表格、CSV、Excel | 在沙盒中生成数据文件 |
| **文本内容** | 文本、文字、内容 | 如果简单直接返回；如果复杂生成文件返回 |
| **执行结果** | 运行、执行、计算结果 | 在沙盒中执行代码 |

### 关键原则
- **用户要什么就给什么**：如果用户说要图片，你就给图片文件
- **不要过度解释**：用户不需要知道过程，只需要结果
- **必须返回文件数据**：任务完成后，将生成的文件数据返回给调用者
- **可用工具**：execute(执行Python) + write_file(创建文件)

## 增强功能
- 任务分析与拆分
- 工具与技能规划
- TODO列表生成
- 分步执行机制
- 技能扩展功能
- 文档处理专项要求
- 代码质量标准

## 重要提示
- 当用户要求创建文件、执行代码时，必须使用沙盒环境
- 沙盒的工作目录是 /home/daytona

## 联网搜索
- 当你需要获取最新信息时，可以使用 Tavily 搜索工具
- 对于需要准确信息的任务（如游戏攻略、数据查询等），必须先搜索再处理

## 技能使用
- 可以通过加载技能（Skills）来处理特定领域的任务

请始终以专业、友好的方式与用户交互。"""
        
        agent_kwargs = {
            "model": self.llm,
            "system_prompt": system_prompt,
            "debug": self.debug,
        }
        
        if self.backend is not None:
            agent_kwargs["backend"] = self.backend
        
        if self.config.tavily_api_key:
            try:
                tavily_tool = TavilySearchResults(api_key=self.config.tavily_api_key, max_results=5)
                agent_kwargs["tools"] = [tavily_tool]
                print("✓ 已加载 Tavily 联网搜索工具")
            except Exception as e:
                print(f"警告: 无法加载 Tavily 搜索工具: {e}")
        
        skills_path = self.config.skills_dir or DEFAULT_SKILLS_DIR
        if skills_path.exists():
            skills_posix_path = str(skills_path).replace("\\", "/")
            agent_kwargs["skills"] = [skills_posix_path]
            print(f"✓ 已加载技能目录: {skills_path}")
        
        self.agent = create_deep_agent(**agent_kwargs)
        print("✓ Agent 初始化完成\n")
    
    def initialize(self):
        """初始化 Agent（创建 LLM、沙盒、Agent）"""
        print("正在初始化 Cerebellum 智能代理...\n")
        
        print("正在连接阿里百炼 LLM...")
        self.llm = self._create_llm()
        print(f"  模型: {self.config.dashscope_model}")
        
        print("\n正在创建沙盒环境...")
        logging.info("开始创建沙盒环境...")
        self.backend, self.sandbox = self._create_sandbox()
        
        skills_path = self.config.skills_dir or DEFAULT_SKILLS_DIR
        self.skills_files = self._load_skills_files(skills_path)
        
        print("\n正在初始化 Agent...")
        self._create_agent()
        
        return self
    
    def run(self, task: str) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
        
        Returns:
            包含结果和文件信息的字典
            {
                "success": True,
                "message": "处理结果...",
                "files": [
                    {"name": "file.txt", "content": "文件内容", "type": "text"},
                    {"name": "image.png", "content": b"...", "type": "image/png"}
                ]
            }
        """
        if self.agent is None:
            self.initialize()
        
        print(f"执行任务: {task}\n")
        result = {
            "success": False,
            "message": "",
            "files": []
        }
        
        try:
            print("[正在处理...]")
            agent_result = self.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": task
                        }
                    ],
                    "files": self.skills_files
                }
            )
            
            if agent_result and "messages" in agent_result:
                last_message = agent_result["messages"][-1]
                result["message"] = last_message.content if hasattr(last_message, "content") else str(last_message)
                print(f"\n结果:\n{result['message']}")
                result["success"] = True
            
        except Exception as e:
            print(f"错误: {e}")
            result["message"] = str(e)
        
        finally:
            # 下载文件数据（不保存到本地）
            result["files"] = self._download_files_from_sandbox()
            
            # 清理沙盒
            self._cleanup_sandbox()
        
        return result
    
    def __enter__(self):
        """上下文管理器入口"""
        return self.initialize()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._cleanup_sandbox()
            



def run_task(task: str, config: Optional[CerebellumConfig] = None, debug: bool = False) -> Dict[str, Any]:
    """
    快速运行任务的便捷函数
    
    Args:
        task: 任务描述
        config: 配置对象（可选）
        debug: 是否启用调试模式
    
    Returns:
        包含结果和文件信息的字典
    """
    with Cerebellum(config=config, debug=debug) as agent:
        return agent.run(task)

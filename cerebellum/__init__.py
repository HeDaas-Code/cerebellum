"""
Cerebellum 智能代理库

该库提供一个强大的 AI 智能代理，具备以下能力：
1. 使用阿里百炼 LLM 作为语言模型
2. 使用 Daytona 沙盒执行代码和命令
3. 加载预设的技能（Skills）来处理各种任务
4. 支持 Tavily 联网搜索
5. 多智能体编排架构
6. 任务缓存与去重
7. 自主反思链错误处理

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
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults

from .utils import setup_logging, logger, log_function_call, log_execution_time

try:
    from daytona import Daytona
    from daytona_sdk.common.daytona import DaytonaConfig
    from langchain_daytona import DaytonaSandbox
except ImportError:
    logger.warning("Daytona SDK 未安装，将使用默认后端")
    Daytona = None
    DaytonaConfig = None
    DaytonaSandbox = None

try:
    from deepagents import create_deep_agent
    from deepagents.backends.utils import create_file_data
except ImportError:
    logger.error("deepagents 未安装，请运行: pip install deepagents")
    sys.exit(1)

from .types import (
    FileData, SubTask, SubTaskStatus, TaskStatus, TaskResult,
    ReflectionHistory, CacheResult, CacheAction
)
from .config import CerebellumConfig, CacheConfig, ReflectionConfig, SandboxConfig, LLMBackend
from .data import DatabaseManager
from .data.task_encoder import TaskEncoder
from .data.similarity import SimilarityChecker
from .data.cache_manager import CacheManager
from .tools import SandboxManager
from .reflection import ReflectionChainExecutor
from .orchestrator import TaskPlanner, TaskScheduler, TaskReporter


DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"

# 支持的文件扩展名（用于文件路径提取和下载）
VALID_FILE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg',
    '.pdf', '.xlsx', '.xls', '.csv', '.txt', '.md', '.json',
    '.docx', '.doc', '.pptx', '.ppt', '.html', '.xml',
    '.py', '.js', '.ts', '.sh', '.zip', '.tar', '.gz',
}

# 安全的沙盒路径前缀
SAFE_PATH_PREFIXES = ('/home/daytona/workspace/', '/tmp/')

# 致命连接错误类型名称 — 不可恢复，应立即终止而非重试
FATAL_ERROR_NAMES = frozenset({
    'RemoteDisconnected',
    'ConnectionResetError',
    'ConnectionRefusedError',
    'ConnectionAbortedError',
    'BrokenPipeError',
    'ProtocolError',
})

# agent.stream() 默认超时（秒）
AGENT_STREAM_TIMEOUT = 600

# execute_with_reflection 最大递归深度
MAX_REFLECTION_RECURSION_DEPTH = 3


def _is_fatal_connection_error(error: Exception) -> bool:
    """
    检测是否为致命的连接错误（不可恢复）
    
    这些错误表示底层网络连接已断开，
    重试只会无限等待，应立即终止执行。
    """
    error_type = type(error).__name__
    if error_type in FATAL_ERROR_NAMES:
        return True
    
    # 检查异常链中的嵌套错误
    cause = error.__cause__ or error.__context__
    while cause:
        if type(cause).__name__ in FATAL_ERROR_NAMES:
            return True
        cause = getattr(cause, '__cause__', None) or getattr(cause, '__context__', None)
    
    # 检查错误消息中的连接错误特征
    error_str = str(error)
    fatal_patterns = [
        'RemoteDisconnected',
        'Remote end closed connection',
        'Connection reset by peer',
        'Connection refused',
        'Broken pipe',
    ]
    return any(pattern in error_str for pattern in fatal_patterns)


def extract_llm_content(response) -> str:
    """
    统一提取 LLM 响应内容（兼容所有后端格式）
    
    处理字符串、列表格式（Anthropic 兼容）等多种内容格式
    
    Args:
        response: LLM 响应对象或内容
    
    Returns:
        提取的文本内容
    """
    content = response
    if hasattr(response, 'content'):
        content = response.content
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    text_parts.append(block.get('text', ''))
                elif 'text' in block:
                    text_parts.append(block['text'])
            elif isinstance(block, str):
                text_parts.append(block)
        return '\n'.join(text_parts)
    
    return str(content) if content else ""


class Cerebellum:
    """
    Cerebellum 智能代理类
    
    提供一个强大的 AI 智能代理，支持：
    - 阿里百炼 LLM
    - Daytona 沙盒
    - 技能系统
    - 联网搜索
    - 多智能体编排
    - 任务缓存与去重
    - 自主反思链
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
        
        # 从环境变量解析默认后端（仅当 config 未显式指定时）
        default_backend = os.getenv("DEFAULT_BACKEND", "").lower()
        if default_backend:
            backend_map = {
                "minimax": LLMBackend.MINIMAX,
                "openai": LLMBackend.OPENAI,
                "anthropic": LLMBackend.ANTHROPIC,
                "dashscope": LLMBackend.DASHSCOPE,
            }
            if default_backend in backend_map:
                self.config.llm_backend = backend_map[default_backend]
                # 后端切换后重新应用默认的 model 和 base_url
                self.config.apply_backend_defaults()
        
        # 统一解析 API 密钥、base_url、model（按后端类型从环境变量获取）
        env_key_map = {
            LLMBackend.DASHSCOPE: ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "DASHSCOPE_MODEL"),
            LLMBackend.MINIMAX: ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_MODEL"),
            LLMBackend.OPENAI: ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL"),
            LLMBackend.ANTHROPIC: ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL"),
        }
        env_keys = env_key_map.get(self.config.llm_backend, env_key_map[LLMBackend.DASHSCOPE])
        
        if not self.config.api_key:
            self.config.api_key = os.getenv(env_keys[0], "")
        if not self.config.daytona_api_key:
            self.config.daytona_api_key = os.getenv("DAYTONA_API_KEY", "")
        if not self.config.tavily_api_key:
            self.config.tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        
        # 允许环境变量覆盖 base_url 和 model
        env_base_url = os.getenv(env_keys[1], "")
        if env_base_url:
            self.config.base_url = env_base_url
        env_model = os.getenv(env_keys[2], "")
        if env_model:
            self.config.model = env_model
        
        self.llm = None
        self.backend = None
        self.sandbox = None
        self.sandbox_manager = None
        self.agent = None
        self.skills_files = {}
        
        self.db = None
        self.cache_manager = None
        self.similarity_checker = None
        self.reflection_executor = None
        self.planner = None
        self.scheduler = None
        self.reporter = None
        self.web_search = None
        self.summarizer = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        setup_logging(debug=self.debug)
        if self.debug:
            logger.debug("调试模式已启用")
    
    def _create_llm(self):
        """创建 LLM 客户端（支持 DashScope/GLM5、MiniMax、Anthropic、OpenAI）"""
        logger.debug(f"创建 LLM 客户端: backend={self.config.llm_backend.value}, model={self.config.model}")
        
        if self.config.llm_backend in (LLMBackend.MINIMAX, LLMBackend.ANTHROPIC):
            # MiniMax (Anthropic 兼容) 和 Anthropic 使用 ChatAnthropic
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError:
                logger.warning("langchain-anthropic 未安装，回退到 ChatOpenAI")
                return ChatOpenAI(
                    model=self.config.model,
                    base_url=self.config.base_url,
                    api_key=self.config.api_key,
                    temperature=0.7,
                    max_tokens=4096,
                )
            
            kwargs = {
                "model": self.config.model,
                "anthropic_api_key": self.config.api_key,
                "temperature": 0.7,
                "max_tokens": 4096,
            }
            
            # MiniMax 使用自定义 base_url
            if self.config.llm_backend == LLMBackend.MINIMAX:
                kwargs["anthropic_api_url"] = self.config.base_url
            elif self.config.base_url and "anthropic.com" not in self.config.base_url:
                kwargs["anthropic_api_url"] = self.config.base_url
            
            return ChatAnthropic(**kwargs)
        
        # DashScope (GLM5) 和 OpenAI 使用 ChatOpenAI
        return ChatOpenAI(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=0.7,
            max_tokens=4096,
        )
    
    def _create_sandbox(self):
        """创建沙盒"""
        if Daytona is None:
            logger.warning("Daytona SDK 不可用")
            return None, None
        
        try:
            logger.info("正在创建沙盒环境...")
            daytona_config = DaytonaConfig(
                api_key=self.config.daytona_api_key if self.config.daytona_api_key else None
            )
            daytona = Daytona(config=daytona_config)
            
            # 构建沙盒创建参数
            create_kwargs = {}
            
            # 自定义镜像
            if self.config.sandbox.image:
                logger.info(f"使用自定义镜像: {self.config.sandbox.image}")
                create_kwargs["image"] = self.config.sandbox.image
            
            logger.debug("调用 daytona.create()...")
            sandbox = daytona.create(**create_kwargs)
            
            logger.debug("等待沙盒启动...")
            sandbox.wait_for_sandbox_start(timeout=60)
            logger.info(f"沙盒已启动，状态: {sandbox.state}")
            
            work_dir = sandbox.get_work_dir()
            logger.debug(f"沙盒工作目录: {work_dir}")
            
            # 设置工作目录
            workdir = self.config.sandbox.workdir
            logger.debug("初始化沙盒环境...")
            sandbox._process.exec(f"mkdir -p {workdir} && chmod 755 {workdir}", timeout=30)
            
            # 设置环境变量
            if self.config.sandbox.env_vars:
                logger.info(f"设置环境变量: {list(self.config.sandbox.env_vars.keys())}")
                for key, value in self.config.sandbox.env_vars.items():
                    sandbox._process.exec(f'export {key}="{value}" && echo "export {key}=***" >> ~/.bashrc', timeout=10)
            
            # 预安装依赖
            if self.config.sandbox.pre_install:
                logger.info(f"预安装依赖: {self.config.sandbox.pre_install}")
                for package in self.config.sandbox.pre_install:
                    if isinstance(package, str):
                        sandbox._process.exec(f"pip install {package}", timeout=120)
                    elif isinstance(package, dict):
                        pkg_name = package.get("name", "")
                        pkg_type = package.get("type", "pip")
                        if pkg_type == "pip":
                            sandbox._process.exec(f"pip install {pkg_name}", timeout=120)
                        elif pkg_type == "apt":
                            sandbox._process.exec(f"apt-get update && apt-get install -y {pkg_name}", timeout=180)
            
            backend = DaytonaSandbox(sandbox=sandbox)
            
            logger.success(f"Daytona 沙盒创建成功 (ID: {sandbox.id})")
            return backend, sandbox
            
        except Exception as e:
            logger.error(f"创建沙盒失败: {e}")
            return None, None
    
    def _load_skills_files(self, skills_path: Path) -> Dict[str, Any]:
        """加载技能文件"""
        if not skills_path.exists():
            logger.warning(f"技能目录不存在: {skills_path}")
            return {}
        
        logger.debug(f"加载技能目录: {skills_path}")
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
                        logger.warning(f"无法读取技能文件 {file_path}: {e}")
        
        logger.info(f"已加载 {len(skills_files)} 个技能文件")
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
        if filename.startswith('.'):
            return False
        
        exclude_dirs = ['.cache', '.local', '.npm', '.git', 'node_modules', '__pycache__', '.venv']
        for ex_dir in exclude_dirs:
            if ex_dir in filename:
                return False
        
        valid_extensions = [
            'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico',
            'pdf', 'doc', 'docx', 'txt', 'md', 'rtf', 'odt',
            'csv', 'xlsx', 'xls', 'ods'
        ]
        
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        return ext in valid_extensions
    
    def _extract_file_paths(self, message: str) -> List[str]:
        """从智能体回复中提取文件路径"""
        import re
        import os.path
        file_paths = []
        
        # 匹配文件路径的正则表达式
        ext_pattern = '|'.join(ext.lstrip('.') for ext in VALID_FILE_EXTENSIONS)
        patterns = [
            rf'/home/daytona/workspace/[\w\-\./]+\.(?:{ext_pattern})',  # 完整路径（含子目录）
            rf'/tmp/[\w\-\./]+\.(?:{ext_pattern})',  # /tmp 目录
            rf'\b([\w\-]+)\.({ext_pattern})\b'  # 独立文件名
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, message)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        if len(match) >= 2 and match[0]:
                            filename = f"{match[0]}.{match[1]}"
                            file_paths.append(f"/home/daytona/workspace/{filename}")
                    else:
                        cleaned_path = re.sub(r'[`"\']', '', match)
                        if cleaned_path:
                            file_paths.append(cleaned_path)
        
        # 过滤无效路径并防止路径遍历攻击
        valid_paths = []
        for path in file_paths:
            if not any(path.lower().endswith(ext) for ext in VALID_FILE_EXTENSIONS):
                continue
            # 规范化路径并确保在安全目录内
            normalized = os.path.normpath(path)
            if any(normalized.startswith(prefix) for prefix in SAFE_PATH_PREFIXES):
                valid_paths.append(normalized)
        
        # 去重（保持顺序）
        seen = set()
        result = []
        for path in valid_paths:
            if path not in seen:
                seen.add(path)
                result.append(path)
        return result
    
    def _download_files_from_sandbox(self, specific_paths: List[str] = None) -> List[FileData]:
        """从沙盒下载文件并返回 FileData 列表"""
        files_data = []
        
        if self.sandbox is None:
            return files_data
        
        try:
            logger.info("正在获取沙盒文件列表...")
            downloaded_names = set()
            
            files = []
            
            # 优先使用指定的文件路径
            if specific_paths:
                files = specific_paths
                logger.info(f"使用指定的文件路径: {len(files)} 个")
            else:
                # 常规文件搜索
                all_files = []
                
                try:
                    result = self.sandbox._process.exec("ls -la /home/daytona/workspace/", timeout=10)
                    if hasattr(result, 'result') and result.result:
                        logger.debug(f"沙盒工作目录文件:\n{result.result[:500]}")
                        for line in result.result.strip().split('\n'):
                            if line.strip() and not line.startswith('total') and not line.startswith('drwxr'):
                                file_name = line.split()[-1]
                                all_files.append(f"/home/daytona/workspace/{file_name}")
                except Exception as e:
                    logger.debug(f"ls workspace 失败: {e}")
                
                try:
                    result = self.sandbox._process.exec(
                        "find /home/daytona/workspace -type f 2>/dev/null",
                        timeout=15
                    )
                    if hasattr(result, 'result') and result.result:
                        for line in result.result.strip().split('\n'):
                            if line.strip() and line.strip() not in all_files:
                                all_files.append(line.strip())
                except Exception as e:
                    logger.debug(f"find 失败: {e}")
                
                valid_ext_tuple = tuple(VALID_FILE_EXTENSIONS)
                files = [f for f in all_files if any(f.lower().endswith(ext) for ext in valid_ext_tuple)]
            
            if not files:
                logger.info("未发现有效文件")
                return files_data
            
            logger.info(f"发现 {len(files)} 个文件")
            valid_files = []
            
            # 先验证所有文件是否真实存在且有效
            for file_path in files:
                file_name = Path(file_path).name
                if file_name in downloaded_names:
                    continue
                
                # 获取文件大小
                file_size = self._get_file_size(file_path)
                if file_size <= 0:
                    logger.warning(f"文件不存在或为空: {file_path}")
                    continue
                
                valid_files.append((file_path, file_name, file_size))
            
            logger.info(f"有效文件: {len(valid_files)} 个")
            
            for file_path, file_name, file_size in valid_files:
                try:
                    logger.debug(f"下载文件: {file_name} (大小: {file_size} 字节)")
                    
                    # 根据文件类型选择下载方式
                    content = self._download_file_with_retry(file_path, file_name)
                    
                    if not content or len(content) < 10:
                        logger.warning(f"文件内容无效: {file_name} (下载大小: {len(content) if content else 0} 字节)")
                        continue
                    
                    # 验证文件内容
                    if not self._validate_file_content(file_name, content):
                        logger.warning(f"文件验证失败: {file_name}")
                        continue
                    
                    files_data.append(FileData(
                        name=file_name,
                        content=content,
                        type=self._get_file_type(file_name)
                    ))
                    downloaded_names.add(file_name)
                    logger.info(f"已下载文件: {file_name} ({len(content)} 字节)")
                    
                except Exception as e:
                    logger.warning(f"读取文件失败 {file_path}: {e}")
                    continue
                        
        except Exception as e:
            logger.error(f"获取文件失败: {e}")
        
        return files_data
    
    def _get_file_size(self, file_path: str) -> int:
        """获取沙盒中文件的大小"""
        try:
            result = self.sandbox._process.exec(f"stat -c%s '{file_path}' 2>/dev/null || wc -c < '{file_path}'", timeout=10)
            if hasattr(result, 'result') and result.result:
                return int(result.result.strip().split()[0])
        except Exception:
            pass
        return 0
    
    def _download_file_with_retry(self, file_path: str, file_name: str) -> bytes:
        """使用多种方法下载文件，带重试机制"""
        content = b""
        
        # 方法1: 使用 base64 编码（推荐用于二进制文件）
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.xlsx', '.zip')):
            content = self._download_via_base64(file_path)
            if content and len(content) > 100:
                logger.debug(f"base64 方法下载成功: {len(content)} 字节")
                return content
            logger.debug(f"base64 方法失败，尝试其他方法...")
        
        # 方法2: 使用 Python 读取并 base64 编码
        content = self._download_via_python(file_path)
        if content and len(content) > 100:
            logger.debug(f"Python 方法下载成功: {len(content)} 字节")
            return content
        
        # 方法3: 使用 cat 命令（仅适用于文本文件）
        if file_name.endswith(('.txt', '.md', '.csv', '.json')):
            content = self._download_via_cat(file_path)
            if content:
                logger.debug(f"cat 方法下载成功: {len(content)} 字节")
                return content
        
        return content
    
    def _download_via_base64(self, file_path: str) -> bytes:
        """通过 base64 命令下载文件"""
        try:
            from base64 import b64decode
            result = self.sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=60)
            if hasattr(result, 'result') and result.result:
                b64_data = result.result.strip()
                # 检查 base64 数据是否有效
                if len(b64_data) < 10:
                    return b""
                # 补齐 padding
                padding = 4 - len(b64_data) % 4
                if padding != 4:
                    b64_data += '=' * padding
                return b64decode(b64_data)
        except Exception as e:
            logger.debug(f"base64 下载失败: {e}")
        return b""
    
    def _download_via_python(self, file_path: str) -> bytes:
        """通过 Python 脚本下载文件"""
        try:
            import base64
            python_code = f'''
import base64
with open("{file_path}", "rb") as f:
    data = f.read()
print(base64.b64encode(data).decode('ascii'))
'''
            result = self.sandbox._process.exec(f"python3 -c '{python_code}'", timeout=60)
            if hasattr(result, 'result') and result.result:
                b64_data = result.result.strip()
                if len(b64_data) < 10:
                    return b""
                return base64.b64decode(b64_data)
        except Exception as e:
            logger.debug(f"Python 下载失败: {e}")
        return b""
    
    def _download_via_cat(self, file_path: str) -> bytes:
        """通过 cat 命令下载文本文件"""
        try:
            result = self.sandbox._process.exec(f"cat '{file_path}'", timeout=60)
            if hasattr(result, 'result') and result.result:
                return result.result.encode('utf-8')
        except Exception as e:
            logger.debug(f"cat 下载失败: {e}")
        return b""
    
    def _validate_file_content(self, file_name: str, content: bytes) -> bool:
        """验证文件内容是否有效"""
        if not content or len(content) < 10:
            return False
        
        # 检查文件魔数（签名）
        if file_name.endswith('.png'):
            return content[:8] == b'\x89PNG\r\n\x1a\n'
        elif file_name.endswith(('.jpg', '.jpeg')):
            return content[:2] == b'\xff\xd8'
        elif file_name.endswith('.pdf'):
            return content[:4] == b'%PDF'
        elif file_name.endswith('.zip'):
            return content[:4] == b'PK\x03\x04'
        elif file_name.endswith('.xlsx'):
            # xlsx 实际上是 zip 格式
            return content[:4] == b'PK\x03\x04'
        
        # 文本文件检查
        if file_name.endswith(('.txt', '.md', '.csv', '.json')):
            try:
                content.decode('utf-8')
                return True
            except UnicodeDecodeError:
                return False
        
        return True
    
    def _cleanup_sandbox(self):
        """清理沙盒资源"""
        if self.sandbox is None:
            return
        
        try:
            logger.info("正在清理沙盒资源...")
            self.sandbox.stop()
            self.sandbox.delete()
            logger.success("沙盒已停止并删除")
            self.sandbox = None
        except Exception as e:
            logger.warning(f"清理沙盒时出错: {e}")
    
    def _create_agent(self):
        """创建 Agent"""
        logger.debug("创建 Agent...")
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

### 图片生成最佳实践（重要！）
生成图片时必须遵循以下步骤：

1. **使用 matplotlib 生成图片**：
```python
import matplotlib.pyplot as plt
# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体避免乱码
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))
# ... 绑定数据 ...

# 保存图片（必须使用 savefig）
plt.savefig('/home/daytona/workspace/chart_name.png', dpi=150, bbox_inches='tight')
plt.close()  # 关闭图形释放内存
```

2. **验证图片已生成**：
```python
import os
file_path = '/home/daytona/workspace/chart_name.png'
if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
    print(f"图片已成功生成: {file_path}, 大小: {os.path.getsize(file_path)} 字节")
else:
    print("图片生成失败！")
```

3. **注意事项**：
- 必须使用 `plt.savefig()` 保存图片，不能只使用 `plt.show()`
- 图片路径必须是 `/home/daytona/workspace/` 目录下
- 生成后必须验证文件存在且大小大于 0

### 关键原则
- **用户要什么就给什么**：如果用户说要图片，你就给图片文件
- **不要过度解释**：用户不需要知道过程，只需要结果
- **必须返回文件数据**：任务完成后，将生成的文件数据返回给调用者
- **可用工具**：execute(执行Python) + write_file(创建文件)

## 联网搜索
- 当你需要获取最新信息时，可以使用 Tavily 搜索工具
- 对于需要准确信息的任务（如游戏攻略、数据查询等），必须先搜索再处理

## 技能使用
- 你可以使用 skills 工具来调用各种技能
- 可用技能包括:
  - **pdf**: 处理 PDF 文件（读取、创建、合并、拆分等）
  - **xlsx**: 处理 Excel 文件（数据分析、图表生成、格式设置）
  - **docx**: 处理 Word 文档
  - **pptx**: 创建 PowerPoint 演示文稿
  - **image-analyzer**: 分析图片内容
  - **file-delivery**: 文件传递和下载

请始终以专业、友好的方式与用户交互。"""
        
        agent_kwargs = {
            "model": self.llm,
            "system_prompt": system_prompt,
            "debug": False,
        }
        
        if self.backend is not None:
            agent_kwargs["backend"] = self.backend
        
        if self.config.tavily_api_key:
            try:
                try:
                    from langchain_tavily import TavilySearch
                    self.web_search = TavilySearch(api_key=self.config.tavily_api_key, max_results=5)
                except ImportError:
                    from langchain_community.tools import TavilySearchResults
                    self.web_search = TavilySearchResults(api_key=self.config.tavily_api_key, max_results=5)
                agent_kwargs["tools"] = [self.web_search]
                logger.info("已加载 Tavily 联网搜索工具")
            except Exception as e:
                logger.warning(f"无法加载 Tavily 搜索工具: {e}")
        
        skills_path = self.config.skills_dir or DEFAULT_SKILLS_DIR
        if skills_path.exists():
            skills_posix_path = str(skills_path).replace("\\", "/")
            agent_kwargs["skills"] = [skills_posix_path]
            logger.info(f"已加载技能目录: {skills_path}")
        
        self.agent = create_deep_agent(**agent_kwargs)
        logger.success("Agent 初始化完成")
    
    def _init_cache_system(self):
        """初始化缓存系统"""
        if self.config.database_path:
            logger.info(f"初始化缓存系统: {self.config.database_path}")
            self.db = DatabaseManager(self.config.database_path)
            
            # 相似度检查器始终启用（不依赖 tavily_api_key）
            self.similarity_checker = SimilarityChecker(
                llm=self.llm,
                db=self.db,
                threshold=self.config.cache.similarity_threshold
            )
            logger.info("相似度检查器已启用")
            
            self.cache_manager = CacheManager(
                db=self.db,
                similarity_checker=self.similarity_checker,
                config=self.config.cache
            )
            logger.success("缓存系统初始化完成")
    
    def _init_orchestrator(self):
        """初始化编排器"""
        logger.debug("初始化编排器...")
        self.planner = TaskPlanner(llm=self.llm)
        self.reporter = TaskReporter()
        
        if self.sandbox_manager and self.llm:
            self.reflection_executor = ReflectionChainExecutor(
                llm=self.llm,
                sandbox=self.sandbox_manager,
                web_search=self.web_search
            )
            
            self.scheduler = TaskScheduler(
                llm=self.llm,
                sandbox=self.sandbox_manager,
                reflection_executor=self.reflection_executor
            )
        
        # 初始化工作流总结器
        from .summary import WorkflowSummarizer
        self.summarizer = WorkflowSummarizer(llm=self.llm, db=self.db)
        
        logger.success("编排器初始化完成")
    
    def initialize(self):
        """初始化 Agent（创建 LLM、沙盒、Agent）"""
        logger.info("正在初始化 Cerebellum 智能代理...")
        
        backend_names = {
            LLMBackend.DASHSCOPE: "阿里百炼",
            LLMBackend.MINIMAX: "MiniMax",
            LLMBackend.OPENAI: "OpenAI",
            LLMBackend.ANTHROPIC: "Anthropic",
        }
        backend_name = backend_names.get(self.config.llm_backend, "未知")
        logger.info(f"正在连接 {backend_name} LLM...")
        self.llm = self._create_llm()
        logger.info(f"模型: {self.config.model}")
        
        logger.info("正在创建沙盒环境...")
        self.backend, self.sandbox = self._create_sandbox()
        
        if self.sandbox:
            self.sandbox_manager = SandboxManager(
                api_key=self.config.daytona_api_key,
                config=self.config.sandbox
            )
            self.sandbox_manager._sandbox = self.sandbox
            self.sandbox_manager._daytona = Daytona(config=DaytonaConfig(
                api_key=self.config.daytona_api_key if self.config.daytona_api_key else None
            )) if Daytona else None
        
        skills_path = self.config.skills_dir or DEFAULT_SKILLS_DIR
        self.skills_files = self._load_skills_files(skills_path)
        
        logger.info("正在初始化 Agent...")
        self._create_agent()
        
        logger.info("正在初始化缓存系统...")
        self._init_cache_system()
        
        logger.info("正在初始化编排器...")
        self._init_orchestrator()
        
        logger.success("Cerebellum 初始化完成")
        return self
    
    def _check_cache(self, task: str, files: List[tuple] = None) -> Optional[CacheResult]:
        """检查任务缓存"""
        if not self.cache_manager:
            logger.debug("缓存管理器未初始化，跳过缓存检查")
            return None
        
        logger.info("检查任务缓存...")
        task_hash, normalized, intent, file_hashes = TaskEncoder.encode(task, files)
        
        cache_result = self.cache_manager.check_cache(task_hash, normalized, intent)
        
        if cache_result.found:
            logger.info(f"缓存命中，相似度: {cache_result.similarity_score:.2f}")
            if cache_result.action == CacheAction.DIRECT_RETURN:
                logger.info("直接返回缓存结果")
            else:
                logger.info("继续处理任务（相似任务）")
        
        return cache_result
    
    def _store_cache(self, task: str, result_message: str, files: List[FileData]):
        """存储任务结果到缓存"""
        if not self.cache_manager:
            logger.debug("缓存管理器未初始化，跳过缓存存储")
            return
        
        logger.info("存储任务结果到缓存...")
        task_hash, _, _, _ = TaskEncoder.encode(task)
        self.cache_manager.store_cache(task_hash, result_message, files)
        logger.success("缓存存储完成")
    
    def run(
        self, 
        task: str, 
        files: Optional[List[tuple]] = None
    ) -> Dict[str, Any]:
        """
        执行任务
        
        Args:
            task: 任务描述
            files: 可选，要上传处理的文件列表 [(file_data, filename), ...]
        
        Returns:
            包含结果和文件信息的字典
            {
                "success": True,
                "message": "处理结果...",
                "files": [...],
                "uploaded_files": [...],
                "subtasks": [...],
                "skills_used": [...],
                "reflection_chains": [...]
            }
        """
        if self.agent is None:
            self.initialize()
        
        start_time = time.time()
        
        logger.info(f"执行任务: {task}")
        result = {
            "success": False,
            "message": "",
            "files": [],
            "uploaded_files": [],
            "subtasks": [],
            "skills_used": [],
            "reflection_chains": []
        }
        
        cache_result = self._check_cache(task, files)
        if cache_result and cache_result.action == CacheAction.DIRECT_RETURN:
            result["success"] = True
            result["message"] = cache_result.cached_result.get("message", "从缓存返回")
            result["files"] = [
                {"name": f.name, "content": f.content, "type": f.type}
                for f in cache_result.cached_files
            ]
            logger.success("从缓存返回结果")
            return result
        
        if files and self.sandbox_manager:
            logger.info("正在上传文件到沙盒...")
            for file_data, filename in files:
                if isinstance(file_data, bytes):
                    sandbox_path = f"/home/daytona/workspace/{filename}"
                    if self.sandbox_manager.upload(file_data, sandbox_path):
                        logger.debug(f"已上传: {filename} -> {sandbox_path}")
                    else:
                        logger.warning(f"上传失败: {filename}")
        
        try:
            logger.info("=" * 60)
            logger.info("【任务规划器】阶段 1: 分析与拆分")
            logger.info("=" * 60)
            
            intent, expected_outputs = self.planner.analyze_intent(task)
            logger.info(f"[任务规划器] 意图分析: {intent}")
            logger.info(f"[任务规划器] 预期输出类型: {expected_outputs}")
            
            files_info = [f[1] for f in files] if files else None
            subtasks = self.planner.split_task(task, files_info)
            
            if subtasks:
                logger.info(f"[任务规划器] 任务拆分为 {len(subtasks)} 个子任务:")
                for i, st in enumerate(subtasks, 1):
                    deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
                    logger.info(f"  {i}. [{st.priority}] {st.name}{deps}")
                result["subtasks"] = [
                    {"id": st.id, "name": st.name, "description": st.description, "priority": st.priority}
                    for st in subtasks
                ]
            
            # 提取技能名称（而不是文件路径）
            skill_names = set()
            for skill_path in self.skills_files.keys():
                # 从路径中提取技能名称: /skills/pdf/SKILL.md -> pdf
                parts = skill_path.split("/")
                if len(parts) >= 3 and parts[1] == "skills":
                    skill_names.add(parts[2])
            
            available_skills = list(skill_names)
            skill_mapping = {}
            if available_skills:
                logger.info(f"[任务规划器] 可用技能: {available_skills}")
                skill_mapping = self.planner.match_skills(subtasks, available_skills)
                if skill_mapping:
                    logger.info("[任务规划器] 技能匹配结果:")
                    for st_id, skill in skill_mapping.items():
                        st = next((s for s in subtasks if s.id == st_id), None)
                        if st:
                            logger.info(f"  - {st.name} -> {skill}")
                        result["skills_used"].append(skill)
            
            logger.info("=" * 60)
            logger.info("【智能体执行】阶段 2: 任务执行")
            logger.info("=" * 60)
            
            # 显示技能使用提示
            if result["skills_used"]:
                unique_skills = list(set(result["skills_used"]))
                logger.info(f"[智能体] 已加载技能: {unique_skills}")
                logger.info(f"[智能体] 技能文件已注入到智能体上下文中，智能体可以直接参考这些技能知识")
            
            task_prompt = task
            if files:
                task_prompt = self._build_file_prompt(task, files, result)
            
            # 构建技能提示
            skill_hints = ""
            if result["skills_used"]:
                unique_skills = list(set(result["skills_used"][:5]))
                logger.info(f"[任务规划器] 推荐技能: {unique_skills}")
                skill_hints = f"""

【可用技能】以下技能可以帮助你完成任务:
{chr(10).join([f'- {s}' for s in unique_skills])}

请使用 skills 工具调用这些技能。例如:
- 处理 PDF 文件: 使用 pdf 技能
- 处理 Excel 文件: 使用 xlsx 技能
- 创建 Word 文档: 使用 docx 技能
- 创建 PPT 演示: 使用 pptx 技能
"""
            
            enhanced_prompt = f"""{task_prompt}
{skill_hints}
请按照以下步骤执行:
1. 分析任务需求
2. 选择合适的工具和技能
3. 执行任务
4. 返回结果

如果需要生成文件，请保存到 /home/daytona/workspace/ 目录。"""
            
            agent_result = None
            agent_error = None
            step_count = 0
            reflection_count = 0
            max_reflections = 10
            recursion_depth = 0
            
            def execute_with_reflection(prompt_messages, current_step=0):
                """带反思链中断的执行函数（含致命错误检测和递归深度限制）"""
                nonlocal agent_result, agent_error, step_count, reflection_count, recursion_depth
                
                # 递归深度检查：防止无限递归导致程序挂起
                recursion_depth += 1
                if recursion_depth > MAX_REFLECTION_RECURSION_DEPTH:
                    logger.warning(f"[智能体] 达到最大递归深度 ({MAX_REFLECTION_RECURSION_DEPTH})，停止反思")
                    recursion_depth -= 1
                    return None
                
                try:
                    for event in self.agent.stream(
                        {"messages": prompt_messages, "files": self.skills_files},
                        stream_mode="values"
                    ):
                        if not event:
                            continue
                        
                        step_count += 1
                        if "messages" not in event:
                            continue
                            
                        msgs = event["messages"]
                        if not msgs:
                            continue
                        
                        last_msg = msgs[-1]
                        msg_type = type(last_msg).__name__
                        content = ""
                        if hasattr(last_msg, "content"):
                            raw_content = last_msg.content
                            if isinstance(raw_content, list):
                                text_parts = []
                                for block in raw_content:
                                    if isinstance(block, dict):
                                        if block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                        elif 'text' in block:
                                            text_parts.append(block['text'])
                                    elif isinstance(block, str):
                                        text_parts.append(block)
                                content = '\n'.join(text_parts)
                            else:
                                content = str(raw_content) if raw_content else ""
                        
                        if msg_type == "AIMessage" and content.strip():
                            if "tool_calls" in last_msg.additional_kwargs:
                                tool_calls = last_msg.additional_kwargs["tool_calls"]
                                if tool_calls:
                                    for tc in tool_calls:
                                        func_name = tc.get("function", {}).get("name", "unknown")
                                        func_args = tc.get("function", {}).get("arguments", "{}")
                                        
                                        # 检测技能调用
                                        if func_name == "skills" or "skill" in func_name.lower():
                                            logger.info(f"[智能体] 步骤 {step_count}: 【技能调用】{func_name}")
                                            try:
                                                import json
                                                args = json.loads(func_args) if isinstance(func_args, str) else func_args
                                                if args:
                                                    logger.info(f"[智能体] 技能参数: {args}")
                                            except Exception:
                                                pass
                                        else:
                                            logger.info(f"[智能体] 步骤 {step_count}: 调用工具 {func_name}")
                            elif content.strip():
                                logger.info(f"[智能体] 步骤 {step_count}: {content}")
                        
                        elif msg_type == "ToolMessage":
                            tool_name = getattr(last_msg, 'name', 'unknown')
                            raw_tool_content = last_msg.content if last_msg.content else ""
                            
                            if isinstance(raw_tool_content, list):
                                text_parts = []
                                for block in raw_tool_content:
                                    if isinstance(block, dict):
                                        if block.get('type') == 'text':
                                            text_parts.append(block.get('text', ''))
                                        elif 'text' in block:
                                            text_parts.append(block['text'])
                                    elif isinstance(block, str):
                                        text_parts.append(block)
                                tool_content = '\n'.join(text_parts)
                            else:
                                tool_content = str(raw_tool_content)
                            
                            if ("Error" in tool_content or "error" in tool_content or 
                                "失败" in tool_content or "Exception" in tool_content):
                                
                                logger.warning(f"[智能体] 步骤 {step_count}: 工具 {tool_name} 返回错误")
                                
                                if self.reflection_executor and reflection_count < max_reflections:
                                    reflection_count += 1
                                    
                                    logger.info("=" * 60)
                                    logger.info(f"【反思链】中断执行，开始错误分析与修复 (第 {reflection_count} 次)")
                                    logger.info("=" * 60)
                                    logger.info(f"[反思链] 错误来源: {tool_name}")
                                    logger.debug(f"[反思链] 错误详情: {tool_content}")
                                    
                                    chain = self.reflection_executor.execute_chain(
                                        error_message=tool_content,
                                        context={
                                            "tool": tool_name,
                                            "step": step_count,
                                            "task": task,
                                            "subtasks": [{"id": st.id, "name": st.name} for st in subtasks] if subtasks else []
                                        },
                                        chain_number=reflection_count,
                                        previous_chains=result["reflection_chains"]
                                    )
                                    
                                    result["reflection_chains"].append({
                                        "chain_id": chain.chain_id,
                                        "chain_number": chain.chain_number,
                                        "error": chain.error_message[:200],
                                        "problem": chain.problem_identified,
                                        "solution": chain.proposed_solution,
                                        "success": chain.is_success
                                    })
                                    
                                    if chain.is_success:
                                        logger.success(f"[反思链] 修复成功: {chain.result_message}")
                                        
                                        # 如果反思链生成了可执行代码，先执行它
                                        if chain.solution_code:
                                            logger.info(f"[反思链] 执行解决方案代码...")
                                            try:
                                                exec_result = self.sandbox._process.exec(chain.solution_code, timeout=60)
                                                if hasattr(exec_result, 'result'):
                                                    logger.debug(f"[反思链] 执行结果: {exec_result.result[:200] if exec_result.result else '无输出'}")
                                                logger.success(f"[反思链] 解决方案代码执行完成")
                                            except Exception as exec_err:
                                                logger.warning(f"[反思链] 解决方案代码执行失败: {exec_err}")
                                                if _is_fatal_connection_error(exec_err):
                                                    logger.error(f"[反思链] 沙盒连接已断开，终止执行")
                                                    recursion_depth -= 1
                                                    return None
                                        
                                        logger.info("[反思链] 恢复任务执行...")
                                        
                                        recovery_prompt = f"""【反思链修复完成】

问题: {chain.problem_identified}
根本原因: {chain.root_cause}
解决方案: {chain.proposed_solution}

请继续执行任务。如果之前有失败的步骤，请根据上述解决方案重新尝试。"""
                                        
                                        new_messages = prompt_messages + [
                                            {"role": "user", "content": recovery_prompt}
                                        ]
                                        return execute_with_reflection(new_messages, step_count)
                                    else:
                                        logger.warning(f"[反思链] 修复失败: {chain.result_message}")
                                else:
                                    logger.warning(f"[反思链] 已达到最大反思次数 ({max_reflections})，继续执行")
                            else:
                                logger.info(f"[智能体] 步骤 {step_count}: 工具 {tool_name} 返回成功")
                    
                    return event
                    
                except Exception as e:
                    agent_error = e
                    logger.error(f"[智能体] 执行错误: {type(e).__name__}: {e}")
                    
                    # 检测致命连接错误 — 沙盒连接已断开，不可恢复
                    if _is_fatal_connection_error(e):
                        logger.error(f"[智能体] 检测到致命连接错误 ({type(e).__name__})，终止执行")
                        recursion_depth -= 1
                        return None
                    
                    if self.reflection_executor and reflection_count < max_reflections:
                        reflection_count += 1
                        
                        logger.info("=" * 60)
                        logger.info(f"【反思链】中断执行，开始错误分析与修复 (第 {reflection_count} 次)")
                        logger.info("=" * 60)
                        logger.info(f"[反思链] 错误类型: {type(e).__name__}")
                        
                        chain = self.reflection_executor.execute_chain(
                            error_message=str(e),
                            context={
                                "tool": "agent",
                                "step": step_count,
                                "task": task,
                                "exception_type": type(e).__name__,
                                "subtasks": [{"id": st.id, "name": st.name} for st in subtasks] if subtasks else []
                            },
                            chain_number=reflection_count,
                            previous_chains=result["reflection_chains"]
                        )
                        
                        result["reflection_chains"].append({
                            "chain_id": chain.chain_id,
                            "chain_number": chain.chain_number,
                            "error": chain.error_message[:200],
                            "problem": chain.problem_identified,
                            "solution": chain.proposed_solution,
                            "success": chain.is_success
                        })
                        
                        if chain.is_success:
                            logger.success(f"[反思链] 修复成功: {chain.result_message}")
                            
                            # 如果反思链生成了可执行代码，先执行它
                            if chain.solution_code:
                                logger.info(f"[反思链] 执行解决方案代码...")
                                try:
                                    exec_result = self.sandbox._process.exec(chain.solution_code, timeout=60)
                                    if hasattr(exec_result, 'result'):
                                        logger.debug(f"[反思链] 执行结果: {exec_result.result[:200] if exec_result.result else '无输出'}")
                                    logger.success(f"[反思链] 解决方案代码执行完成")
                                except Exception as exec_err:
                                    logger.warning(f"[反思链] 解决方案代码执行失败: {exec_err}")
                                    # 解决方案代码执行中出现致命连接错误，终止
                                    if _is_fatal_connection_error(exec_err):
                                        logger.error(f"[反思链] 沙盒连接已断开，终止执行")
                                        recursion_depth -= 1
                                        return None
                            
                            logger.info("[反思链] 恢复任务执行...")
                            
                            recovery_prompt = f"""【反思链修复完成】

异常: {type(e).__name__}: {str(e)}
问题: {chain.problem_identified}
根本原因: {chain.root_cause}
解决方案: {chain.proposed_solution}

请继续执行任务。如果之前有失败的步骤，请根据上述解决方案重新尝试。"""
                            
                            new_messages = [
                                {"role": "user", "content": enhanced_prompt},
                                {"role": "user", "content": recovery_prompt}
                            ]
                            return execute_with_reflection(new_messages, step_count)
                    
                    recursion_depth -= 1
                    return None
            
            agent_result = execute_with_reflection([{"role": "user", "content": enhanced_prompt}])
            
            if agent_error and not result["reflection_chains"]:
                raise agent_error
            
            logger.info("=" * 60)
            logger.info("【结果收集】阶段 3: 文件下载与验证")
            logger.info("=" * 60)
            
            if agent_result and "messages" in agent_result:
                last_message = agent_result["messages"][-1]
                message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                if isinstance(message_content, list):
                    text_parts = []
                    for block in message_content:
                        if isinstance(block, dict):
                            if block.get('type') == 'text':
                                text_parts.append(block.get('text', ''))
                            elif 'text' in block:
                                text_parts.append(block['text'])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    result["message"] = '\n'.join(text_parts)
                else:
                    result["message"] = str(message_content)
                
                logger.info("[结果收集] 任务处理完成")
                result["success"] = True
            else:
                logger.warning(f"[结果收集] Agent 返回异常")
                result["message"] = "Agent 未返回有效结果"
            
            file_paths_from_message = self._extract_file_paths(result["message"])
            if file_paths_from_message:
                logger.info(f"[结果收集] 从智能体回复中提取到 {len(file_paths_from_message)} 个文件路径")
                for path in file_paths_from_message:
                    logger.debug(f"  - {path}")
            
            downloaded_files = self._download_files_from_sandbox(file_paths_from_message)
            if downloaded_files:
                result["files"] = [
                    {"name": f.name, "content": f.content, "type": f.type}
                    for f in downloaded_files
                ]
                logger.info(f"[结果收集] 已下载 {len(downloaded_files)} 个文件")
                
                for f in downloaded_files:
                    if self._validate_file_content(f.name, f.content):
                        logger.success(f"[结果收集] 文件验证通过: {f.name} ({len(f.content)} 字节)")
                    else:
                        logger.warning(f"[结果收集] 文件验证失败: {f.name}")
            
            if result["success"]:
                self._store_cache(task, result["message"], downloaded_files)
                
        except TimeoutError as e:
            result["message"] = f"执行超时: {str(e)}"
            logger.error(f"任务执行超时: {e}")
        except Exception as e:
            result["message"] = f"执行失败: {str(e)}"
            logger.error(f"任务执行失败: {e}")
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        result["execution_time_ms"] = execution_time_ms
        log_execution_time("任务", execution_time_ms)
        
        # 生成工作流总结
        if self.summarizer and result.get("reflection_chains"):
            try:
                summary = self.summarizer.summarize(
                    task=task,
                    result=result,
                    intent=result.get("intent", ""),
                    reflection_chains=result.get("reflection_chains", []),
                    skills_used=result.get("skills_used", []),
                    execution_time_ms=execution_time_ms,
                )
                result["summary"] = summary.to_dict()
                logger.info(f"[总结] 生成工作流总结: {len(summary.constraints)} 个约束")
            except Exception as e:
                logger.debug(f"生成工作流总结失败: {e}")
        
        return result
    
    def _build_file_prompt(self, task: str, files: List[tuple], result: Dict) -> str:
        """构建包含文件信息的提示"""
        file_descriptions = []
        uploaded_info = []
        
        for file_data, filename in files:
            if isinstance(file_data, bytes):
                try:
                    content = file_data.decode('utf-8')
                    file_descriptions.append(f"- {filename}:\n```\n{content}\n```")
                except UnicodeDecodeError:
                    file_descriptions.append(f"- {filename}: [二进制文件，大小: {len(file_data)} 字节]")
            else:
                file_descriptions.append(f"- {filename}:\n```\n{str(file_data)}\n```")
            
            uploaded_info.append({"name": filename, "size": len(file_data) if isinstance(file_data, bytes) else len(str(file_data))})
        
        result["uploaded_files"] = uploaded_info
        
        return f"""用户上传了以下文件:

{chr(10).join(file_descriptions)}

文件已上传到沙盒目录: /home/daytona/workspace/
可以使用 read_file 工具读取文件内容。

任务: {task}

请根据上传的文件内容完成用户的任务。"""
    
    def process_file(self, file_data: bytes, filename: str, task: str) -> Dict[str, Any]:
        """
        处理单个文件的便捷方法
        
        Args:
            file_data: 文件数据
            filename: 文件名
            task: 任务描述
        
        Returns:
            处理结果
        """
        return self.run(task, files=[(file_data, filename)])
    
    def process_files(self, files: List[tuple], task: str) -> Dict[str, Any]:
        """
        处理多个文件的便捷方法
        
        Args:
            files: 文件列表 [(file_data, filename), ...]
            task: 任务描述
        
        Returns:
            处理结果
        """
        return self.run(task, files=files)
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_sandbox()


__all__ = [
    "Cerebellum",
    "CerebellumConfig",
    "CacheConfig",
    "ReflectionConfig",
    "SandboxConfig",
    "LLMBackend",
    "FileData",
    "SubTask",
    "SubTaskStatus",
    "TaskStatus",
    "TaskResult",
    "ReflectionHistory",
    "CacheResult",
    "CacheAction",
    "DatabaseManager",
    "TaskEncoder",
    "SimilarityChecker",
    "CacheManager",
    "SandboxManager",
    "ReflectionChainExecutor",
    "TaskPlanner",
    "TaskScheduler",
    "TaskReporter",
    "VALID_FILE_EXTENSIONS",
    "extract_llm_content",
    "process_file",
    "process_files",
]

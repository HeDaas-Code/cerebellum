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
from .config import CerebellumConfig, CacheConfig, ReflectionConfig, SandboxConfig
from .data import DatabaseManager
from .data.task_encoder import TaskEncoder
from .data.similarity import SimilarityChecker
from .data.cache_manager import CacheManager
from .tools import SandboxManager
from .reflection import ReflectionChainExecutor
from .orchestrator import TaskPlanner, TaskScheduler, TaskReporter


DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


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
        
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        setup_logging(debug=self.debug)
        if self.debug:
            logger.debug("调试模式已启用")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 客户端"""
        logger.debug(f"创建 LLM 客户端: model={self.config.dashscope_model}")
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
            logger.warning("Daytona SDK 不可用")
            return None, None
        
        try:
            logger.info("正在创建沙盒环境...")
            daytona_config = DaytonaConfig(
                api_key=self.config.daytona_api_key if self.config.daytona_api_key else None
            )
            daytona = Daytona(config=daytona_config)
            
            logger.debug("调用 daytona.create()...")
            sandbox = daytona.create()
            
            logger.debug("等待沙盒启动...")
            sandbox.wait_for_sandbox_start(timeout=60)
            logger.info(f"沙盒已启动，状态: {sandbox.state}")
            
            work_dir = sandbox.get_work_dir()
            logger.debug(f"沙盒工作目录: {work_dir}")
            
            logger.debug("初始化沙盒环境...")
            sandbox._process.exec("mkdir -p workspace && chmod 755 workspace", timeout=30)
            
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
    
    def _download_files_from_sandbox(self) -> List[FileData]:
        """从沙盒下载文件并返回 FileData 列表"""
        files_data = []
        
        if self.sandbox is None:
            return files_data
        
        try:
            logger.info("正在获取沙盒文件列表...")
            downloaded_names = set()
            
            all_files = []
            
            try:
                result = self.sandbox._process.exec("ls /home/daytona/workspace/", timeout=10)
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            all_files.append(f"/home/daytona/workspace/{line.strip()}")
            except Exception as e:
                logger.debug(f"ls workspace 失败: {e}")
            
            try:
                result = self.sandbox._process.exec(
                    "find /home/daytona/workspace -type f -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.pdf' 2>/dev/null",
                    timeout=15
                )
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip() and line.strip() not in all_files:
                            all_files.append(line.strip())
            except Exception as e:
                logger.debug(f"find 失败: {e}")
            
            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.pdf', '.xlsx', '.csv', '.txt', '.md')
            files = [f for f in all_files if any(f.lower().endswith(ext) for ext in valid_extensions)]
            
            if not files:
                logger.info("未发现有效文件")
                return files_data
            
            logger.info(f"发现 {len(files)} 个文件")
            for file_path in files:
                try:
                    file_name = Path(file_path).name
                    
                    if file_name in downloaded_names:
                        continue
                    
                    if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        from base64 import b64decode
                        b64_result = self.sandbox._process.exec(f"base64 -w0 '{file_path}'", timeout=30)
                        bstdout = b64_result.stdout if hasattr(b64_result, 'stdout') else ""
                        content = b64decode(bstdout.strip()) if bstdout.strip() else b""
                    else:
                        content_result = self.sandbox._process.exec(f"cat '{file_path}'", timeout=30)
                        content = content_result.stdout if hasattr(content_result, 'stdout') else ""
                        if isinstance(content, str):
                            content = content.encode('utf-8')
                    
                    if not content:
                        logger.warning(f"文件内容为空: {file_name}")
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

### 关键原则
- **用户要什么就给什么**：如果用户说要图片，你就给图片文件
- **不要过度解释**：用户不需要知道过程，只需要结果
- **必须返回文件数据**：任务完成后，将生成的文件数据返回给调用者
- **可用工具**：execute(执行Python) + write_file(创建文件)

## 联网搜索
- 当你需要获取最新信息时，可以使用 Tavily 搜索工具
- 对于需要准确信息的任务（如游戏攻略、数据查询等），必须先搜索再处理

## 技能使用
- 可以通过加载技能（Skills）来处理特定领域的任务

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
            logger.debug(f"初始化缓存系统: {self.config.database_path}")
            self.db = DatabaseManager(self.config.database_path)
            
            if self.config.tavily_api_key:
                self.similarity_checker = SimilarityChecker(
                    llm=self.llm,
                    db=self.db,
                    threshold=self.config.cache.similarity_threshold
                )
            
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
        
        logger.success("编排器初始化完成")
    
    def initialize(self):
        """初始化 Agent（创建 LLM、沙盒、Agent）"""
        logger.info("正在初始化 Cerebellum 智能代理...")
        
        logger.info("正在连接阿里百炼 LLM...")
        self.llm = self._create_llm()
        logger.info(f"模型: {self.config.dashscope_model}")
        
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
            return None
        
        logger.debug("检查任务缓存...")
        task_hash, normalized, intent, file_hashes = TaskEncoder.encode(task, files)
        
        cache_result = self.cache_manager.check_cache(task_hash, normalized, intent)
        
        if cache_result.found:
            logger.info(f"缓存命中，相似度: {cache_result.similarity_score:.2f}")
            self.reporter.report_cache_hit(
                task_hash, 
                cache_result.similarity_score,
                "直接返回" if cache_result.action == CacheAction.DIRECT_RETURN else "继续处理"
            )
        
        return cache_result
    
    def _store_cache(self, task: str, result_message: str, files: List[FileData]):
        """存储任务结果到缓存"""
        if not self.cache_manager:
            return
        
        logger.debug("存储任务结果到缓存...")
        task_hash, _, _, _ = TaskEncoder.encode(task)
        self.cache_manager.store_cache(task_hash, result_message, files)
        logger.debug("缓存存储完成")
    
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
                "files": [
                    {"name": "file.txt", "content": "文件内容", "type": "text"},
                    {"name": "image.png", "content": b"...", "type": "image/png"}
                ],
                "uploaded_files": [...]  # 上传的文件信息（如果有）
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
            "skills_used": []
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
            logger.info("=" * 50)
            logger.info("阶段 1: 任务规划")
            logger.info("=" * 50)
            
            intent, expected_outputs = self.planner.analyze_intent(task)
            logger.info(f"任务意图: {intent}")
            logger.info(f"预期输出: {expected_outputs}")
            
            files_info = [f[1] for f in files] if files else None
            subtasks = self.planner.split_task(task, files_info)
            
            if subtasks:
                logger.info(f"任务拆分为 {len(subtasks)} 个子任务:")
                for i, st in enumerate(subtasks, 1):
                    deps = f" (依赖: {', '.join(st.dependencies)})" if st.dependencies else ""
                    logger.info(f"  {i}. [{st.priority}] {st.name}{deps}")
                result["subtasks"] = [
                    {"id": st.id, "name": st.name, "description": st.description, "priority": st.priority}
                    for st in subtasks
                ]
            
            available_skills = list(self.skills_files.keys()) if self.skills_files else []
            if available_skills:
                logger.info(f"可用技能: {len(available_skills)} 个")
                skill_mapping = self.planner.match_skills(subtasks, available_skills)
                if skill_mapping:
                    logger.info("技能匹配结果:")
                    for st_id, skill in skill_mapping.items():
                        st = next((s for s in subtasks if s.id == st_id), None)
                        if st:
                            logger.info(f"  - {st.name} -> {skill}")
                        result["skills_used"].append(skill)
            
            logger.info("=" * 50)
            logger.info("阶段 2: 任务执行")
            logger.info("=" * 50)
            
            task_prompt = task
            if files:
                task_prompt = self._build_file_prompt(task, files, result)
            
            skill_hints = ""
            if result["skills_used"]:
                skill_hints = f"\n\n可用技能: {', '.join(set(result['skills_used'][:5]))}\n可以通过 skills 工具调用这些技能来完成任务。"
            
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
            
            try:
                for event in self.agent.stream(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": enhanced_prompt
                            }
                        ],
                        "files": self.skills_files
                    },
                    stream_mode="values"
                ):
                    if event:
                        step_count += 1
                        if "messages" in event:
                            msgs = event["messages"]
                            if msgs:
                                last_msg = msgs[-1]
                                msg_type = type(last_msg).__name__
                                content = ""
                                if hasattr(last_msg, "content"):
                                    content = str(last_msg.content)[:150] if last_msg.content else ""
                                
                                if msg_type == "AIMessage" and content.strip():
                                    if "tool_calls" in last_msg.additional_kwargs:
                                        tool_calls = last_msg.additional_kwargs["tool_calls"]
                                        if tool_calls:
                                            for tc in tool_calls:
                                                func_name = tc.get("function", {}).get("name", "unknown")
                                                logger.info(f"[步骤 {step_count}] 调用工具: {func_name}")
                                    elif content.strip():
                                        logger.info(f"[步骤 {step_count}] AI: {content}...")
                                elif msg_type == "ToolMessage":
                                    logger.debug(f"[步骤 {step_count}] 工具返回")
                
                agent_result = event if event else None
                
            except Exception as e:
                agent_error = e
                logger.error(f"Stream 错误: {type(e).__name__}: {e}")
            
            if agent_error:
                raise agent_error
            
            logger.info("=" * 50)
            logger.info("阶段 3: 结果收集")
            logger.info("=" * 50)
            
            if agent_result and "messages" in agent_result:
                last_message = agent_result["messages"][-1]
                result["message"] = last_message.content if hasattr(last_message, 'content') else str(last_message)
                logger.info("任务处理完成")
                result["success"] = True
            else:
                logger.warning(f"Agent 返回异常: {agent_result}")
                result["message"] = "Agent 未返回有效结果"
            
            downloaded_files = self._download_files_from_sandbox()
            if downloaded_files:
                result["files"] = [
                    {"name": f.name, "content": f.content, "type": f.type}
                    for f in downloaded_files
                ]
                logger.info(f"已获取 {len(downloaded_files)} 个文件")
            
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
        
        return result
    
    def _build_file_prompt(self, task: str, files: List[tuple], result: Dict) -> str:
        """构建包含文件信息的提示"""
        file_descriptions = []
        uploaded_info = []
        
        for file_data, filename in files:
            if isinstance(file_data, bytes):
                try:
                    content = file_data.decode('utf-8')
                    preview = content[:500] + "..." if len(content) > 500 else content
                    file_descriptions.append(f"- {filename}:\n```\n{preview}\n```")
                except UnicodeDecodeError:
                    file_descriptions.append(f"- {filename}: [二进制文件，大小: {len(file_data)} 字节]")
            else:
                preview = str(file_data)[:500] + "..." if len(str(file_data)) > 500 else str(file_data)
                file_descriptions.append(f"- {filename}:\n```\n{preview}\n```")
            
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
    "process_file",
    "process_files",
]

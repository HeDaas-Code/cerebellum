"""
Cerebellum 类型定义模块

定义所有核心数据类型
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from base64 import b64encode, b64decode


class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class SubTaskStatus(str, Enum):
    """子任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReflectionStatus(str, Enum):
    """反思链状态"""
    SUCCESS = "success"
    FAILURE = "failure"


class CacheAction(str, Enum):
    """缓存操作类型"""
    DIRECT_RETURN = "direct_return"
    CONTINUE_PROCESSING = "continue_processing"
    NEW_TASK = "new_task"


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
class SubTask:
    """子任务定义"""
    id: str
    name: str
    description: str
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: SubTaskStatus = SubTaskStatus.PENDING
    assigned_skill: Optional[str] = None
    assigned_agent: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class TaskReport:
    """任务汇报"""
    timestamp: datetime
    phase: str
    subtask_id: Optional[str]
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ReflectionChain:
    """
    一次完整的反思链
    
    定义: 从错误发生到产生结果的完整过程
    """
    chain_id: str
    chain_number: int
    
    error_message: str
    error_context: Dict[str, Any] = field(default_factory=dict)
    
    problem_identified: str = ""
    problem_scope: str = ""
    
    search_queries: List[str] = field(default_factory=list)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    
    root_cause: str = ""
    proposed_solution: str = ""
    solution_code: Optional[str] = None
    
    execution_log: str = ""
    
    result_status: ReflectionStatus = ReflectionStatus.FAILURE
    result_message: str = ""
    
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    @property
    def is_success(self) -> bool:
        """反思链是否成功解决问题"""
        return self.result_status == ReflectionStatus.SUCCESS
    
    @property
    def duration_seconds(self) -> float:
        """反思链耗时"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0


@dataclass
class ReflectionHistory:
    """反思历史 - 记录所有反思链"""
    chains: List[ReflectionChain] = field(default_factory=list)
    max_chains: int = 10
    
    @property
    def current_chain_number(self) -> int:
        """当前是第几轮反思链"""
        return len(self.chains) + 1
    
    @property
    def can_continue(self) -> bool:
        """是否可以继续反思"""
        return len(self.chains) < self.max_chains
    
    @property
    def last_chain(self) -> Optional[ReflectionChain]:
        """最后一条反思链"""
        return self.chains[-1] if self.chains else None
    
    def add_chain(self, chain: ReflectionChain):
        """添加反思链"""
        self.chains.append(chain)


@dataclass
class TaskRecord:
    """任务记录（数据库存储）"""
    task_hash: str
    task_content: str
    normalized_content: str
    intent: str
    parameters: Dict[str, Any]
    file_hashes: List[str]
    status: TaskStatus
    created_at: datetime
    sandbox_id: Optional[str] = None
    result_path: Optional[str] = None


@dataclass
class CacheResult:
    """缓存结果"""
    found: bool
    task_hash: str
    similarity_score: float
    cached_files: List[FileData] = field(default_factory=list)
    cached_result: Optional[Dict[str, Any]] = None
    action: CacheAction = CacheAction.NEW_TASK


@dataclass
class SimilarityResult:
    """相似度比对结果"""
    is_similar: bool
    score: float
    matched_task: Optional[TaskRecord] = None
    reasoning: str = ""


@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    message: str
    files: List[FileData] = field(default_factory=list)
    uploaded_files: List[Dict[str, Any]] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    subtasks: List[SubTask] = field(default_factory=list)
    reflection_history: Optional[ReflectionHistory] = None
    execution_time_ms: int = 0


@dataclass
class ExecuteResult:
    """命令执行结果"""
    exit_code: int
    stdout: str
    stderr: str
    command: str
    timeout: bool = False


@dataclass
class ErrorDecision:
    """错误处理决策"""
    action: str
    delay: float = 0.0
    package: Optional[str] = None
    alternative: Optional[str] = None
    error: Optional[Exception] = None
    options: Optional[List[str]] = None
    message: str = ""
    solution: Optional[str] = None
    solution_code: Optional[str] = None
    reflection_count: int = 0

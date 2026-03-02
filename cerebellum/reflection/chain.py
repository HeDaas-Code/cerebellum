"""
Cerebellum 反思链定义

定义反思链数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ..types import ReflectionStatus


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

"""
Cerebellum 工作流总结器

总结反思链和整个工作流程，形成新的约束和路径
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..utils import logger


@dataclass
class PathConstraint:
    """路径约束"""
    constraint_type: str  # "positive" (好的路径) 或 "negative" (坏的约束)
    description: str
    source_task: str  # 来源任务描述
    confidence: float = 0.5  # 置信度 0-1
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "constraint_type": self.constraint_type,
            "description": self.description,
            "source_task": self.source_task,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PathConstraint":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()
        return cls(
            constraint_type=data.get("constraint_type", "positive"),
            description=data.get("description", ""),
            source_task=data.get("source_task", ""),
            confidence=data.get("confidence", 0.5),
            created_at=created_at,
            tags=data.get("tags", []),
        )


@dataclass
class WorkflowSummary:
    """工作流总结"""
    task: str
    intent: str
    success: bool
    total_steps: int
    reflection_count: int
    skills_used: List[str]
    constraints: List[PathConstraint]
    key_decisions: List[str]
    execution_time_ms: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "intent": self.intent,
            "success": self.success,
            "total_steps": self.total_steps,
            "reflection_count": self.reflection_count,
            "skills_used": self.skills_used,
            "constraints": [c.to_dict() for c in self.constraints],
            "key_decisions": self.key_decisions,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowSummary":
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now()
        return cls(
            task=data.get("task", ""),
            intent=data.get("intent", ""),
            success=data.get("success", False),
            total_steps=data.get("total_steps", 0),
            reflection_count=data.get("reflection_count", 0),
            skills_used=data.get("skills_used", []),
            constraints=[PathConstraint.from_dict(c) for c in data.get("constraints", [])],
            key_decisions=data.get("key_decisions", []),
            execution_time_ms=data.get("execution_time_ms", 0),
            created_at=created_at,
        )
    
    def get_positive_constraints(self) -> List[PathConstraint]:
        """获取正向约束（好的路径）"""
        return [c for c in self.constraints if c.constraint_type == "positive"]
    
    def get_negative_constraints(self) -> List[PathConstraint]:
        """获取负向约束（坏的约束）"""
        return [c for c in self.constraints if c.constraint_type == "negative"]


class WorkflowSummarizer:
    """
    工作流总结器
    
    总结反思链和整个工作流程，形成新的约束和路径参考
    """
    
    def __init__(self, llm=None, db=None):
        """
        初始化工作流总结器
        
        Args:
            llm: LLM 客户端（可选，用于 LLM 增强总结）
            db: 数据库管理器（可选，用于持久化存储）
        """
        self.llm = llm
        self.db = db
        self._summaries: List[WorkflowSummary] = []
        logger.debug("工作流总结器初始化完成")
    
    def _call_llm(self, prompt: str) -> str:
        """统一调用 LLM"""
        response = self.llm.invoke(prompt)
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
            return str(content)
        return str(response)
    
    def summarize(
        self,
        task: str,
        result: Dict[str, Any],
        intent: str = "",
        reflection_chains: List[Dict[str, Any]] = None,
        skills_used: List[str] = None,
        execution_time_ms: int = 0,
    ) -> WorkflowSummary:
        """
        总结工作流程
        
        Args:
            task: 原始任务
            result: 执行结果
            intent: 任务意图
            reflection_chains: 反思链记录
            skills_used: 使用的技能
            execution_time_ms: 执行时间(ms)
        
        Returns:
            工作流总结
        """
        logger.info("正在总结工作流程...")
        
        reflection_chains = reflection_chains or []
        skills_used = skills_used or []
        success = result.get("success", False)
        
        # 从反思链中提取约束和关键决策
        constraints = self._extract_constraints(task, reflection_chains, success)
        key_decisions = self._extract_key_decisions(reflection_chains)
        
        # 如果有 LLM，使用 LLM 增强总结
        if self.llm and reflection_chains:
            try:
                llm_constraints = self._llm_extract_constraints(task, reflection_chains, success)
                constraints.extend(llm_constraints)
            except Exception as e:
                logger.debug(f"LLM 增强总结失败: {e}")
        
        summary = WorkflowSummary(
            task=task,
            intent=intent,
            success=success,
            total_steps=len(result.get("subtasks", [])),
            reflection_count=len(reflection_chains),
            skills_used=skills_used,
            constraints=constraints,
            key_decisions=key_decisions,
            execution_time_ms=execution_time_ms,
        )
        
        self._summaries.append(summary)
        
        # 持久化存储
        if self.db:
            self._store_summary(summary)
        
        logger.success(f"工作流总结完成: {len(constraints)} 个约束, {len(key_decisions)} 个关键决策")
        return summary
    
    def _extract_constraints(
        self,
        task: str,
        reflection_chains: List[Dict[str, Any]],
        success: bool
    ) -> List[PathConstraint]:
        """从反思链中提取约束"""
        constraints = []
        
        for chain in reflection_chains:
            chain_success = chain.get("success", False)
            problem = chain.get("problem", "")
            solution = chain.get("solution", "")
            
            if chain_success and solution:
                # 成功的解决方案 -> 正向约束
                constraints.append(PathConstraint(
                    constraint_type="positive",
                    description=f"解决 '{problem}' 的有效方法: {solution}",
                    source_task=task,
                    confidence=0.8,
                    tags=["reflection", "solution"],
                ))
            elif not chain_success and solution:
                # 失败的尝试 -> 负向约束
                constraints.append(PathConstraint(
                    constraint_type="negative",
                    description=f"针对 '{problem}' 无效的方法: {solution}",
                    source_task=task,
                    confidence=0.7,
                    tags=["reflection", "failed_attempt"],
                ))
        
        return constraints
    
    def _extract_key_decisions(
        self,
        reflection_chains: List[Dict[str, Any]]
    ) -> List[str]:
        """提取关键决策"""
        decisions = []
        
        for chain in reflection_chains:
            if chain.get("success"):
                solution = chain.get("solution", "")
                if solution:
                    decisions.append(f"修复: {solution}")
        
        return decisions
    
    def _llm_extract_constraints(
        self,
        task: str,
        reflection_chains: List[Dict[str, Any]],
        success: bool
    ) -> List[PathConstraint]:
        """使用 LLM 提取更高级的约束"""
        chains_text = json.dumps(reflection_chains, ensure_ascii=False, indent=2)
        
        prompt = f"""分析以下任务执行记录，提取有价值的经验约束。

任务: {task}
执行结果: {"成功" if success else "失败"}

反思链记录:
{chains_text[:3000]}

请提取约束，返回 JSON 数组:
[
  {{"type": "positive", "description": "好的做法描述", "confidence": 0.8, "tags": ["标签"]}},
  {{"type": "negative", "description": "应避免的做法", "confidence": 0.7, "tags": ["标签"]}}
]

只返回 JSON，不要其他内容。"""
        
        try:
            content = self._call_llm(prompt).strip()
            content = content.replace("```json", "").replace("```", "").strip()
            raw_constraints = json.loads(content)
            
            constraints = []
            for rc in raw_constraints:
                constraints.append(PathConstraint(
                    constraint_type=rc.get("type", "positive"),
                    description=rc.get("description", ""),
                    source_task=task,
                    confidence=rc.get("confidence", 0.5),
                    tags=rc.get("tags", []),
                ))
            return constraints
        except Exception:
            return []
    
    def get_relevant_constraints(
        self,
        task: str,
        max_results: int = 10
    ) -> List[PathConstraint]:
        """
        获取与当前任务相关的约束参考
        
        Args:
            task: 当前任务描述
            max_results: 最大返回数
        
        Returns:
            相关约束列表
        """
        all_constraints = []
        for summary in self._summaries:
            all_constraints.extend(summary.constraints)
        
        # 从数据库加载
        if self.db:
            stored = self._load_constraints()
            all_constraints.extend(stored)
        
        # 简单相关性评分：基于关键词匹配
        task_lower = task.lower()
        scored = []
        for constraint in all_constraints:
            score = 0.0
            desc_lower = constraint.description.lower()
            source_lower = constraint.source_task.lower()
            
            # 关键词重叠
            task_words = set(task_lower.split())
            desc_words = set(desc_lower.split())
            source_words = set(source_lower.split())
            
            if task_words & desc_words:
                score += len(task_words & desc_words) / max(len(task_words), 1) * 0.5
            if task_words & source_words:
                score += len(task_words & source_words) / max(len(task_words), 1) * 0.5
            
            score *= constraint.confidence
            scored.append((score, constraint))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:max_results]]
    
    def _store_summary(self, summary: WorkflowSummary):
        """存储总结到数据库"""
        try:
            if self.db:
                self.db.store_summary(summary.to_dict())
        except Exception as e:
            logger.debug(f"存储总结失败: {e}")
    
    def _load_constraints(self) -> List[PathConstraint]:
        """从数据库加载约束"""
        try:
            if self.db:
                raw = self.db.load_constraints()
                if raw:
                    return [PathConstraint.from_dict(c) for c in raw]
        except Exception as e:
            logger.debug(f"加载约束失败: {e}")
        return []
    
    @property
    def summaries(self) -> List[WorkflowSummary]:
        """获取所有总结"""
        return list(self._summaries)

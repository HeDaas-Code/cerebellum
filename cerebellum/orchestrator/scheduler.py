"""
Cerebellum 任务调度模块

调度和执行子任务（支持并行执行）
"""

import concurrent.futures
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

from ..types import SubTask, SubTaskStatus, TaskResult, FileData
from ..tools.sandbox import SandboxManager
from ..reflection.executor import ReflectionChainExecutor
from ..reflection.chain import ReflectionHistory
from ..utils import logger


class TaskScheduler:
    """
    任务调度器
    
    调度子任务的执行顺序，支持独立子任务并行处理
    """
    
    def __init__(
        self, 
        llm=None, 
        sandbox: SandboxManager = None,
        reflection_executor: Optional[ReflectionChainExecutor] = None,
        max_parallel: int = 3
    ):
        """
        初始化任务调度器
        
        Args:
            llm: LLM 客户端
            sandbox: 沙盒管理器
            reflection_executor: 反思链执行器
            max_parallel: 最大并行子任务数
        """
        self.llm = llm
        self.sandbox = sandbox
        self.reflection_executor = reflection_executor
        self.max_parallel = max_parallel
    
    def get_execution_order(self, subtasks: List[SubTask]) -> List[SubTask]:
        """
        获取子任务执行顺序（拓扑排序）
        
        Args:
            subtasks: 子任务列表
        
        Returns:
            排序后的子任务列表
        """
        sorted_tasks = []
        remaining = list(subtasks)
        completed_ids = set()
        
        while remaining:
            for task in remaining[:]:
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    sorted_tasks.append(task)
                    completed_ids.add(task.id)
                    remaining.remove(task)
                    break
            else:
                if remaining:
                    sorted_tasks.append(remaining.pop(0))
        
        return sorted_tasks
    
    def get_parallel_groups(self, subtasks: List[SubTask]) -> List[List[SubTask]]:
        """
        将子任务分组为可并行执行的批次
        
        独立的（无依赖或依赖已完成的）子任务可以并行执行
        
        Args:
            subtasks: 子任务列表
        
        Returns:
            并行批次列表，每个批次中的子任务可以并行执行
        """
        groups = []
        remaining = list(subtasks)
        completed_ids = set()
        
        while remaining:
            # 找出所有依赖已满足的子任务（可并行执行）
            ready_tasks = []
            for task in remaining[:]:
                if all(dep_id in completed_ids for dep_id in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # 循环依赖处理：取第一个未完成的任务
                if remaining:
                    ready_tasks = [remaining[0]]
                    logger.warning(f"检测到循环依赖，强制执行: {remaining[0].name}")
                else:
                    break
            
            groups.append(ready_tasks)
            for task in ready_tasks:
                completed_ids.add(task.id)
                remaining.remove(task)
        
        return groups
    
    def execute_subtask(
        self, 
        subtask: SubTask, 
        context: Dict[str, Any],
        skill_content: str = None
    ) -> tuple:
        """
        执行单个子任务
        
        Args:
            subtask: 子任务
            context: 执行上下文
            skill_content: 技能内容
        
        Returns:
            (是否成功, 结果消息)
        """
        subtask.status = SubTaskStatus.RUNNING
        
        skill_context = ""
        if skill_content:
            skill_context = f"\n\n参考技能指导:\n{skill_content[:2000]}"
        
        prompt = f"""执行以下子任务:

子任务: {subtask.name}
描述: {subtask.description}

上下文: {context}
{skill_context}

请执行任务并返回结果。如果需要创建文件，请使用 write_file 工具。"""
        
        max_attempts = 3
        reflection_history = ReflectionHistory()
        
        for attempt in range(max_attempts):
            try:
                if self.reflection_executor and attempt > 0:
                    chain = self.reflection_executor.execute_chain(
                        subtask.error or "执行失败",
                        context,
                        reflection_history.current_chain_number,
                        [{"solution": c.proposed_solution, "result": c.result_status.value} 
                         for c in reflection_history.chains]
                    )
                    reflection_history.add_chain(chain)
                    
                    if chain.is_success:
                        subtask.status = SubTaskStatus.COMPLETED
                        subtask.result = chain.result_message
                        return True, chain.result_message
                    
                    if not reflection_history.can_continue:
                        break
                else:
                    response = self.llm.invoke(prompt)
                    content = response.content if hasattr(response, 'content') else str(response)
                    
                    # 处理列表格式的 content（Anthropic 兼容）
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
                        content = '\n'.join(text_parts)
                    
                    subtask.status = SubTaskStatus.COMPLETED
                    subtask.result = content
                    return True, content
                    
            except Exception as e:
                subtask.error = str(e)
        
        subtask.status = SubTaskStatus.FAILED
        return False, subtask.error or "执行失败"
    
    def execute_parallel_group(
        self,
        group: List[SubTask],
        context: Dict[str, Any],
        skill_mapping: Dict[str, str] = None,
        skills_content: Dict[str, str] = None
    ) -> List[tuple]:
        """
        并行执行一组子任务
        
        Args:
            group: 可并行执行的子任务列表
            context: 执行上下文
            skill_mapping: 子任务ID到技能名的映射
            skills_content: 技能名到内容的映射
        
        Returns:
            [(subtask, success, message), ...]
        """
        if len(group) == 1:
            # 单个任务直接执行，无需线程池
            subtask = group[0]
            skill_content = None
            if skill_mapping and skills_content:
                skill_name = skill_mapping.get(subtask.id)
                if skill_name:
                    skill_content = skills_content.get(skill_name)
            
            success, message = self.execute_subtask(subtask, context, skill_content)
            return [(subtask, success, message)]
        
        results = []
        max_workers = min(len(group), self.max_parallel)
        logger.info(f"并行执行 {len(group)} 个子任务 (workers={max_workers})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_subtask = {}
            
            for subtask in group:
                skill_content = None
                if skill_mapping and skills_content:
                    skill_name = skill_mapping.get(subtask.id)
                    if skill_name:
                        skill_content = skills_content.get(skill_name)
                
                future = executor.submit(
                    self.execute_subtask,
                    subtask,
                    context,
                    skill_content
                )
                future_to_subtask[future] = subtask
            
            for future in concurrent.futures.as_completed(future_to_subtask):
                subtask = future_to_subtask[future]
                try:
                    success, message = future.result()
                    results.append((subtask, success, message))
                except Exception as e:
                    subtask.status = SubTaskStatus.FAILED
                    subtask.error = str(e)
                    results.append((subtask, False, str(e)))
                    logger.warning(f"并行子任务 '{subtask.name}' 执行异常: {e}")
        
        return results

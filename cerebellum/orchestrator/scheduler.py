"""
Cerebellum 任务调度模块

调度和执行子任务
"""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

from ..types import SubTask, SubTaskStatus, TaskResult, FileData
from ..tools.sandbox import SandboxManager
from ..reflection.executor import ReflectionChainExecutor
from ..reflection.chain import ReflectionHistory


class TaskScheduler:
    """
    任务调度器
    
    调度子任务的执行顺序
    """
    
    def __init__(
        self, 
        llm: ChatOpenAI, 
        sandbox: SandboxManager,
        reflection_executor: Optional[ReflectionChainExecutor] = None
    ):
        """
        初始化任务调度器
        
        Args:
            llm: LLM 客户端
            sandbox: 沙盒管理器
            reflection_executor: 反思链执行器
        """
        self.llm = llm
        self.sandbox = sandbox
        self.reflection_executor = reflection_executor
    
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
                    subtask.status = SubTaskStatus.COMPLETED
                    subtask.result = response.content
                    return True, response.content
                    
            except Exception as e:
                subtask.error = str(e)
        
        subtask.status = SubTaskStatus.FAILED
        return False, subtask.error or "执行失败"

"""
scheduler.py 单元测试

测试任务调度器
"""

import pytest
from unittest.mock import MagicMock, patch

from cerebellum.orchestrator.scheduler import TaskScheduler
from cerebellum.types import SubTask, SubTaskStatus, TaskResult, FileData
from cerebellum.tools.sandbox import SandboxManager
from cerebellum.reflection.executor import ReflectionChainExecutor


class TestTaskScheduler:
    """TaskScheduler 测试"""
    
    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(content="执行成功"))
        return mock
    
    @pytest.fixture
    def mock_sandbox(self):
        """模拟沙盒"""
        mock = MagicMock(spec=SandboxManager)
        return mock
    
    @pytest.fixture
    def mock_reflection(self):
        """模拟反思执行器"""
        mock = MagicMock(spec=ReflectionChainExecutor)
        return mock
    
    @pytest.fixture
    def scheduler(self, mock_llm, mock_sandbox, mock_reflection):
        """任务调度器实例"""
        return TaskScheduler(
            llm=mock_llm,
            sandbox=mock_sandbox,
            reflection_executor=mock_reflection
        )
    
    def test_get_execution_order_no_dependencies(self, scheduler):
        """测试无依赖的任务排序"""
        subtasks = [
            SubTask(id="t1", name="任务1", description="描述1", dependencies=[]),
            SubTask(id="t2", name="任务2", description="描述2", dependencies=[]),
            SubTask(id="t3", name="任务3", description="描述3", dependencies=[])
        ]
        
        result = scheduler.get_execution_order(subtasks)
        
        assert len(result) == 3
        assert set(s.id for s in result) == {"t1", "t2", "t3"}
    
    def test_get_execution_order_with_dependencies(self, scheduler):
        """测试有依赖的任务排序"""
        subtasks = [
            SubTask(id="t1", name="任务1", description="描述1", dependencies=[]),
            SubTask(id="t2", name="任务2", description="描述2", dependencies=["t1"]),
            SubTask(id="t3", name="任务3", description="描述3", dependencies=["t2"])
        ]
        
        result = scheduler.get_execution_order(subtasks)
        
        assert result[0].id == "t1"
        assert result[1].id == "t2"
        assert result[2].id == "t3"
    
    def test_get_execution_order_complex_dependencies(self, scheduler):
        """测试复杂依赖关系"""
        subtasks = [
            SubTask(id="t1", name="任务1", description="描述1", dependencies=[]),
            SubTask(id="t2", name="任务2", description="描述2", dependencies=["t1"]),
            SubTask(id="t3", name="任务3", description="描述3", dependencies=["t1"]),
            SubTask(id="t4", name="任务4", description="描述4", dependencies=["t2", "t3"])
        ]
        
        result = scheduler.get_execution_order(subtasks)
        
        t1_idx = next(i for i, s in enumerate(result) if s.id == "t1")
        t2_idx = next(i for i, s in enumerate(result) if s.id == "t2")
        t3_idx = next(i for i, s in enumerate(result) if s.id == "t3")
        t4_idx = next(i for i, s in enumerate(result) if s.id == "t4")
        
        assert t1_idx < t2_idx
        assert t1_idx < t3_idx
        assert t2_idx < t4_idx
        assert t3_idx < t4_idx
    
    def test_get_execution_order_circular_dependency(self, scheduler):
        """测试循环依赖（应打破循环）"""
        subtasks = [
            SubTask(id="t1", name="任务1", description="描述1", dependencies=["t2"]),
            SubTask(id="t2", name="任务2", description="描述2", dependencies=["t1"])
        ]
        
        result = scheduler.get_execution_order(subtasks)
        
        assert len(result) == 2
    
    def test_execute_subtask_success(self, scheduler, mock_llm):
        """测试子任务执行成功"""
        subtask = SubTask(
            id="st1",
            name="测试任务",
            description="执行测试"
        )
        
        success, message = scheduler.execute_subtask(
            subtask,
            context={"files": ["test.txt"]}
        )
        
        assert success is True
        assert subtask.status == SubTaskStatus.COMPLETED
        assert subtask.result is not None
    
    def test_execute_subtask_with_skill(self, scheduler, mock_llm):
        """测试带技能指导的执行"""
        subtask = SubTask(
            id="st1",
            name="生成文档",
            description="创建 Word 文档"
        )
        
        success, message = scheduler.execute_subtask(
            subtask,
            context={},
            skill_content="使用 python-docx 库创建文档..."
        )
        
        assert success is True
        call_args = mock_llm.invoke.call_args[0][0]
        assert "python-docx" in call_args
    
    def test_execute_subtask_with_reflection(self, scheduler, mock_llm, mock_reflection):
        """测试带反思链的执行"""
        subtask = SubTask(
            id="st1",
            name="测试任务",
            description="执行测试"
        )
        subtask.error = "ModuleNotFoundError"
        
        from cerebellum.types import ReflectionStatus
        mock_chain = MagicMock()
        mock_chain.is_success = True
        mock_chain.result_message = "问题已解决"
        mock_reflection.execute_chain.return_value = mock_chain
        
        mock_llm.invoke.side_effect = [Exception("First fail"), MagicMock(content="Success")]
        
        success, message = scheduler.execute_subtask(subtask, context={})
        
        assert subtask.status == SubTaskStatus.COMPLETED
    
    def test_execute_subtask_max_attempts(self, scheduler, mock_llm, mock_reflection):
        """测试最大重试次数"""
        subtask = SubTask(
            id="st1",
            name="测试任务",
            description="执行测试"
        )
        
        mock_llm.invoke.side_effect = Exception("Always fail")
        
        from cerebellum.types import ReflectionStatus
        mock_chain = MagicMock()
        mock_chain.is_success = False
        mock_reflection.execute_chain.return_value = mock_chain
        
        success, message = scheduler.execute_subtask(subtask, context={})
        
        assert success is False
        assert subtask.status == SubTaskStatus.FAILED
    
    def test_execute_subtask_updates_status(self, scheduler, mock_llm):
        """测试状态更新"""
        subtask = SubTask(
            id="st1",
            name="测试任务",
            description="执行测试",
            status=SubTaskStatus.PENDING
        )
        
        scheduler.execute_subtask(subtask, context={})
        
        assert subtask.status == SubTaskStatus.COMPLETED
    
    def test_scheduler_without_reflection(self, mock_llm, mock_sandbox):
        """测试无反思执行器的调度器"""
        scheduler = TaskScheduler(
            llm=mock_llm,
            sandbox=mock_sandbox,
            reflection_executor=None
        )
        
        subtask = SubTask(id="st1", name="任务", description="描述")
        
        success, message = scheduler.execute_subtask(subtask, context={})
        
        assert success is True

"""
测试并行调度功能
"""

import pytest
from unittest.mock import MagicMock

from cerebellum.orchestrator.scheduler import TaskScheduler
from cerebellum.types import SubTask, SubTaskStatus


class TestParallelScheduler:
    """并行调度测试"""
    
    @pytest.fixture
    def scheduler(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="执行完成")
        return TaskScheduler(llm=mock_llm, sandbox=MagicMock())
    
    def test_get_parallel_groups_no_deps(self, scheduler):
        """测试无依赖时所有任务在同一组"""
        subtasks = [
            SubTask(id="1", name="任务A", description="描述A"),
            SubTask(id="2", name="任务B", description="描述B"),
            SubTask(id="3", name="任务C", description="描述C"),
        ]
        
        groups = scheduler.get_parallel_groups(subtasks)
        
        assert len(groups) == 1
        assert len(groups[0]) == 3
    
    def test_get_parallel_groups_with_deps(self, scheduler):
        """测试有依赖时分组"""
        subtasks = [
            SubTask(id="1", name="任务A", description="描述A"),
            SubTask(id="2", name="任务B", description="描述B", dependencies=["1"]),
            SubTask(id="3", name="任务C", description="描述C"),
        ]
        
        groups = scheduler.get_parallel_groups(subtasks)
        
        assert len(groups) == 2
        # 第一组包含 A 和 C（无依赖）
        first_ids = {t.id for t in groups[0]}
        assert "1" in first_ids
        assert "3" in first_ids
        # 第二组包含 B（依赖 A）
        second_ids = {t.id for t in groups[1]}
        assert "2" in second_ids
    
    def test_get_parallel_groups_chain(self, scheduler):
        """测试链式依赖"""
        subtasks = [
            SubTask(id="1", name="任务A", description="描述A"),
            SubTask(id="2", name="任务B", description="描述B", dependencies=["1"]),
            SubTask(id="3", name="任务C", description="描述C", dependencies=["2"]),
        ]
        
        groups = scheduler.get_parallel_groups(subtasks)
        
        assert len(groups) == 3
    
    def test_execute_parallel_group_single(self, scheduler):
        """测试单个任务组执行"""
        subtasks = [
            SubTask(id="1", name="任务A", description="描述A"),
        ]
        
        results = scheduler.execute_parallel_group(subtasks, {})
        
        assert len(results) == 1
        assert results[0][1] is True  # success
    
    def test_execute_parallel_group_multiple(self, scheduler):
        """测试多个任务并行执行"""
        subtasks = [
            SubTask(id="1", name="任务A", description="描述A"),
            SubTask(id="2", name="任务B", description="描述B"),
        ]
        
        results = scheduler.execute_parallel_group(subtasks, {})
        
        assert len(results) == 2
        assert all(r[1] is True for r in results)
    
    def test_max_parallel_limit(self):
        """测试最大并行数限制"""
        scheduler = TaskScheduler(
            llm=MagicMock(invoke=MagicMock(return_value=MagicMock(content="ok"))),
            sandbox=MagicMock(),
            max_parallel=2
        )
        assert scheduler.max_parallel == 2

"""
types.py 单元测试

测试所有核心数据类型
"""

import pytest
from datetime import datetime
from base64 import b64decode

from cerebellum.types import (
    FileData, SubTask, SubTaskStatus, TaskStatus, TaskRecord,
    ReflectionChain, ReflectionStatus, ReflectionHistory,
    CacheResult, CacheAction, SimilarityResult, TaskResult,
    ExecuteResult, ErrorDecision
)


class TestFileData:
    """FileData 测试"""
    
    def test_create_text_file(self):
        """测试创建文本文件"""
        file = FileData(name="test.txt", content="Hello World")
        
        assert file.name == "test.txt"
        assert file.content == "Hello World"
        assert file.type == "text"
    
    def test_create_binary_file(self):
        """测试创建二进制文件"""
        file = FileData(name="image.png", content=b"\x89PNG\r\n\x1a\n")
        
        assert file.name == "image.png"
        assert file.type == "binary"
    
    def test_get_bytes_from_text(self):
        """测试从文本获取字节"""
        file = FileData(name="test.txt", content="测试内容")
        
        result = file.get_bytes()
        
        assert isinstance(result, bytes)
        assert result == "测试内容".encode('utf-8')
    
    def test_get_bytes_from_binary(self):
        """测试从二进制获取字节"""
        content = b"\x00\x01\x02\x03"
        file = FileData(name="data.bin", content=content)
        
        result = file.get_bytes()
        
        assert result == content
    
    def test_get_text_from_string(self):
        """测试从字符串获取文本"""
        file = FileData(name="test.txt", content="文本内容")
        
        result = file.get_text()
        
        assert result == "文本内容"
    
    def test_get_text_from_bytes(self):
        """测试从字节获取文本"""
        file = FileData(name="test.txt", content="文本内容".encode('utf-8'))
        
        result = file.get_text()
        
        assert result == "文本内容"
    
    def test_get_base64(self):
        """测试 Base64 编码"""
        file = FileData(name="test.txt", content="Hello")
        
        result = file.get_base64()
        
        decoded = b64decode(result)
        assert decoded == b"Hello"


class TestSubTask:
    """SubTask 测试"""
    
    def test_create_subtask(self):
        """测试创建子任务"""
        subtask = SubTask(
            id="task-001",
            name="测试任务",
            description="这是一个测试任务"
        )
        
        assert subtask.id == "task-001"
        assert subtask.name == "测试任务"
        assert subtask.status == SubTaskStatus.PENDING
        assert subtask.priority == 0
        assert subtask.dependencies == []
    
    def test_subtask_with_dependencies(self):
        """测试带依赖的子任务"""
        subtask = SubTask(
            id="task-002",
            name="后续任务",
            description="依赖前一个任务",
            dependencies=["task-001"]
        )
        
        assert "task-001" in subtask.dependencies
    
    def test_subtask_default_values(self):
        """测试子任务默认值"""
        subtask = SubTask(id="t1", name="n", description="d")
        
        assert subtask.retry_count == 0
        assert subtask.assigned_skill is None
        assert subtask.assigned_agent is None
        assert subtask.result is None
        assert subtask.error is None
        assert subtask.completed_at is None
        assert isinstance(subtask.created_at, datetime)


class TestReflectionChain:
    """ReflectionChain 测试"""
    
    def test_create_reflection_chain(self):
        """测试创建反思链"""
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error"
        )
        
        assert chain.chain_id == "chain-001"
        assert chain.chain_number == 1
        assert chain.error_message == "Test error"
        assert chain.result_status == ReflectionStatus.FAILURE
    
    def test_is_success_property(self):
        """测试 is_success 属性"""
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error",
            result_status=ReflectionStatus.SUCCESS
        )
        
        assert chain.is_success is True
    
    def test_is_success_when_failure(self):
        """测试失败状态的 is_success"""
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error",
            result_status=ReflectionStatus.FAILURE
        )
        
        assert chain.is_success is False
    
    def test_duration_seconds(self):
        """测试耗时计算"""
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error",
            started_at=datetime(2024, 1, 1, 10, 0, 0),
            completed_at=datetime(2024, 1, 1, 10, 0, 30)
        )
        
        assert chain.duration_seconds == 30.0
    
    def test_duration_seconds_not_completed(self):
        """测试未完成时的耗时"""
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error"
        )
        
        assert chain.duration_seconds == 0.0


class TestReflectionHistory:
    """ReflectionHistory 测试"""
    
    def test_create_empty_history(self):
        """测试创建空历史"""
        history = ReflectionHistory()
        
        assert history.chains == []
        assert history.current_chain_number == 1
        assert history.can_continue is True
        assert history.last_chain is None
    
    def test_add_chain(self):
        """测试添加反思链"""
        history = ReflectionHistory()
        chain = ReflectionChain(
            chain_id="chain-001",
            chain_number=1,
            error_message="Test error"
        )
        
        history.add_chain(chain)
        
        assert len(history.chains) == 1
        assert history.current_chain_number == 2
        assert history.last_chain == chain
    
    def test_can_continue_limit(self):
        """测试反思次数限制"""
        history = ReflectionHistory(max_chains=3)
        
        for i in range(3):
            chain = ReflectionChain(
                chain_id=f"chain-{i}",
                chain_number=i + 1,
                error_message="Test error"
            )
            history.add_chain(chain)
        
        assert history.can_continue is False


class TestTaskRecord:
    """TaskRecord 测试"""
    
    def test_create_task_record(self):
        """测试创建任务记录"""
        record = TaskRecord(
            task_hash="abc123",
            task_content="测试任务",
            normalized_content="测试 任务",
            intent="test",
            parameters={},
            file_hashes=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        assert record.task_hash == "abc123"
        assert record.status == TaskStatus.PENDING
        assert record.sandbox_id is None
        assert record.result_path is None


class TestCacheResult:
    """CacheResult 测试"""
    
    def test_cache_hit(self):
        """测试缓存命中"""
        result = CacheResult(
            found=True,
            task_hash="abc123",
            similarity_score=0.95,
            action=CacheAction.DIRECT_RETURN
        )
        
        assert result.found is True
        assert result.action == CacheAction.DIRECT_RETURN
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        result = CacheResult(
            found=False,
            task_hash="new123",
            similarity_score=0.0,
            action=CacheAction.NEW_TASK
        )
        
        assert result.found is False
        assert result.action == CacheAction.NEW_TASK


class TestSimilarityResult:
    """SimilarityResult 测试"""
    
    def test_similar_result(self):
        """测试相似结果"""
        result = SimilarityResult(
            is_similar=True,
            score=0.92,
            reasoning="任务内容相似"
        )
        
        assert result.is_similar is True
        assert result.score == 0.92
    
    def test_not_similar_result(self):
        """测试不相似结果"""
        result = SimilarityResult(
            is_similar=False,
            score=0.3,
            reasoning="任务差异较大"
        )
        
        assert result.is_similar is False


class TestTaskResult:
    """TaskResult 测试"""
    
    def test_success_result(self):
        """测试成功结果"""
        result = TaskResult(
            success=True,
            message="执行成功",
            files=[FileData(name="output.txt", content="结果")]
        )
        
        assert result.success is True
        assert len(result.files) == 1
    
    def test_failure_result(self):
        """测试失败结果"""
        result = TaskResult(
            success=False,
            message="执行失败"
        )
        
        assert result.success is False
    
    def test_default_values(self):
        """测试默认值"""
        result = TaskResult(success=True, message="OK")
        
        assert result.files == []
        assert result.uploaded_files == []
        assert result.report == {}
        assert result.subtasks == []
        assert result.reflection_history is None
        assert result.execution_time_ms == 0


class TestExecuteResult:
    """ExecuteResult 测试"""
    
    def test_success_execution(self):
        """测试成功执行"""
        result = ExecuteResult(
            exit_code=0,
            stdout="输出内容",
            stderr="",
            command="echo test"
        )
        
        assert result.exit_code == 0
        assert result.timeout is False
    
    def test_failed_execution(self):
        """测试失败执行"""
        result = ExecuteResult(
            exit_code=1,
            stdout="",
            stderr="Error: something went wrong",
            command="false"
        )
        
        assert result.exit_code == 1
    
    def test_timeout_execution(self):
        """测试超时执行"""
        result = ExecuteResult(
            exit_code=-1,
            stdout="",
            stderr="",
            command="sleep 100",
            timeout=True
        )
        
        assert result.timeout is True


class TestErrorDecision:
    """ErrorDecision 测试"""
    
    def test_install_decision(self):
        """测试安装决策"""
        decision = ErrorDecision(
            action="install",
            package="requests",
            message="需要安装 requests"
        )
        
        assert decision.action == "install"
        assert decision.package == "requests"
    
    def test_retry_decision(self):
        """测试重试决策"""
        decision = ErrorDecision(
            action="retry",
            delay=2.0,
            message="等待后重试"
        )
        
        assert decision.action == "retry"
        assert decision.delay == 2.0
    
    def test_abort_decision(self):
        """测试中止决策"""
        decision = ErrorDecision(
            action="abort",
            message="无法恢复的错误"
        )
        
        assert decision.action == "abort"

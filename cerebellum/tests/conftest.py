"""
Cerebellum 测试配置

共享 fixtures 和测试工具
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from cerebellum.types import (
    FileData, SubTask, SubTaskStatus, TaskStatus, TaskRecord,
    ReflectionChain, ReflectionStatus, ReflectionHistory,
    CacheResult, CacheAction, SimilarityResult, TaskResult,
    ExecuteResult, ErrorDecision
)
from cerebellum.config import (
    CerebellumConfig, CacheConfig, ReflectionConfig, SandboxConfig
)


@pytest.fixture
def temp_dir():
    """临时目录 fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir):
    """临时数据库路径"""
    return temp_dir / "test_cache.db"


@pytest.fixture
def test_config(temp_db_path):
    """测试用配置"""
    return CerebellumConfig(
        dashscope_api_key="test-api-key",
        dashscope_model="test-model",
        daytona_api_key="test-daytona-key",
        tavily_api_key="test-tavily-key",
        debug=True,
        database_path=temp_db_path,
        cache=CacheConfig(
            similarity_threshold=0.85,
            max_cache_size=100,
            cache_ttl_days=7
        ),
        reflection=ReflectionConfig(
            max_reflections=3,
            enable_web_search=False
        ),
        sandbox=SandboxConfig(
            timeout_seconds=60,
            max_retries=2
        )
    )


@pytest.fixture
def sample_task_content():
    """标准测试任务内容"""
    return "分析这份文档并生成摘要报告"


@pytest.fixture
def sample_files():
    """测试用文件数据"""
    return [
        FileData(name="test.txt", content="这是测试文件内容"),
        FileData(name="data.bin", content=b"\x00\x01\x02\x03")
    ]


@pytest.fixture
def sample_file_tuples():
    """测试用文件元组 (bytes, name)"""
    return [
        ("测试文档内容".encode('utf-8'), "document.txt"),
        (b"\x89PNG\r\n\x1a\n", "image.png")
    ]


@pytest.fixture
def sample_subtask():
    """测试用子任务"""
    return SubTask(
        id="subtask-001",
        name="分析文档",
        description="读取并分析文档内容",
        priority=1,
        dependencies=[],
        status=SubTaskStatus.PENDING,
        assigned_skill="doc-analyzer"
    )


@pytest.fixture
def sample_subtasks(sample_subtask):
    """测试用子任务列表"""
    return [
        sample_subtask,
        SubTask(
            id="subtask-002",
            name="生成摘要",
            description="根据分析结果生成摘要",
            priority=2,
            dependencies=["subtask-001"],
            status=SubTaskStatus.PENDING,
            assigned_skill="text-generator"
        ),
        SubTask(
            id="subtask-003",
            name="格式化输出",
            description="将摘要格式化为报告",
            priority=3,
            dependencies=["subtask-002"],
            status=SubTaskStatus.PENDING,
            assigned_skill="formatter"
        )
    ]


@pytest.fixture
def sample_task_record():
    """测试用任务记录"""
    return TaskRecord(
        task_hash="abc123def456",
        task_content="分析文档并生成报告",
        normalized_content="分析文档 生成报告",
        intent="document_analysis",
        parameters={"output_format": "markdown"},
        file_hashes=["file_hash_1", "file_hash_2"],
        status=TaskStatus.COMPLETED,
        created_at=datetime.now()
    )


@pytest.fixture
def sample_reflection_chain():
    """测试用反思链"""
    return ReflectionChain(
        chain_id="chain-001",
        chain_number=1,
        error_message="ModuleNotFoundError: No module named 'requests'",
        error_context={"command": "python script.py"},
        problem_identified="缺少 requests 模块",
        problem_scope="依赖缺失",
        search_queries=["python install requests"],
        search_results=[{"title": "pip install requests", "url": "https://example.com"}],
        root_cause="虚拟环境中未安装 requests 库",
        proposed_solution="使用 pip 安装 requests",
        solution_code="pip install requests",
        execution_log="Successfully installed requests",
        result_status=ReflectionStatus.SUCCESS,
        result_message="问题已解决"
    )


@pytest.fixture
def sample_reflection_history(sample_reflection_chain):
    """测试用反思历史"""
    history = ReflectionHistory()
    history.add_chain(sample_reflection_chain)
    return history


@pytest.fixture
def sample_cache_result():
    """测试用缓存结果"""
    return CacheResult(
        found=True,
        task_hash="abc123",
        similarity_score=0.95,
        cached_files=[
            FileData(name="report.md", content="# 分析报告\n\n这是生成的报告内容")
        ],
        cached_result={"summary": "文档分析完成"},
        action=CacheAction.DIRECT_RETURN
    )


@pytest.fixture
def sample_similarity_result(sample_task_record):
    """测试用相似度结果"""
    return SimilarityResult(
        is_similar=True,
        score=0.92,
        matched_task=sample_task_record,
        reasoning="两个任务都是分析文档并生成报告"
    )


@pytest.fixture
def sample_task_result(sample_files, sample_subtasks):
    """测试用任务结果"""
    return TaskResult(
        success=True,
        message="任务执行成功",
        files=sample_files,
        subtasks=sample_subtasks,
        execution_time_ms=1500
    )


@pytest.fixture
def sample_execute_result():
    """测试用执行结果"""
    return ExecuteResult(
        exit_code=0,
        stdout="执行成功\n输出内容",
        stderr="",
        command="python script.py",
        timeout=False
    )


@pytest.fixture
def sample_error_decision():
    """测试用错误决策"""
    return ErrorDecision(
        action="install",
        delay=0.0,
        package="requests",
        message="需要安装 requests 模块",
        solution="pip install requests"
    )


@pytest.fixture
def mock_llm():
    """模拟 LLM 响应"""
    mock = MagicMock()
    mock.invoke = MagicMock(return_value="模拟 LLM 响应内容")
    mock.ainvoke = AsyncMock(return_value="模拟异步 LLM 响应")
    return mock


@pytest.fixture
def mock_sandbox():
    """模拟沙盒环境"""
    mock = MagicMock()
    mock.create_sandbox = MagicMock(return_value="sandbox-123")
    mock.execute = MagicMock(return_value=ExecuteResult(
        exit_code=0,
        stdout="执行成功",
        stderr="",
        command="test"
    ))
    mock.delete_sandbox = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_similarity_checker():
    """模拟相似度检查器"""
    mock = MagicMock()
    mock.check_similarity = MagicMock(return_value=SimilarityResult(
        is_similar=False,
        score=0.0,
        matched_task=None,
        reasoning="无相似任务"
    ))
    return mock


@pytest.fixture
def mock_cache_manager(sample_cache_result):
    """模拟缓存管理器"""
    mock = MagicMock()
    mock.check_cache = MagicMock(return_value=sample_cache_result)
    mock.save_to_cache = MagicMock(return_value=True)
    mock.clear_cache = MagicMock(return_value=True)
    return mock


class MockDatabase:
    """模拟数据库"""
    
    def __init__(self):
        self.records = {}
        self._id_counter = 0
    
    def save_task(self, task_record):
        self._id_counter += 1
        self.records[task_record.task_hash] = task_record
        return self._id_counter
    
    def get_task(self, task_hash):
        return self.records.get(task_hash)
    
    def find_similar_tasks(self, normalized_content, threshold=0.85):
        return []
    
    def delete_task(self, task_hash):
        if task_hash in self.records:
            del self.records[task_hash]
            return True
        return False
    
    def cleanup_expired(self, days=30):
        return 0


@pytest.fixture
def mock_database():
    """模拟数据库"""
    return MockDatabase()

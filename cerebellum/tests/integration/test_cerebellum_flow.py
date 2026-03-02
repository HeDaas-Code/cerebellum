"""
cerebellum 完整流程集成测试

测试 Cerebellum 类的端到端功能
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import os

from cerebellum import Cerebellum
from cerebellum.types import (
    SubTask, SubTaskStatus, TaskResult, TaskRecord, TaskStatus,
    FileData, CacheAction, ReflectionStatus
)
from cerebellum.config import CerebellumConfig, CacheConfig, ReflectionConfig


class TestCerebellumInitialization:
    """Cerebellum 初始化测试"""
    
    def test_init_with_config(self, test_config):
        """测试使用配置初始化"""
        agent = Cerebellum(config=test_config)
        
        assert agent.config == test_config
    
    def test_init_default_config(self):
        """测试默认配置初始化"""
        agent = Cerebellum()
        
        assert agent.config is not None
        assert agent.config.database_path is not None


class TestCerebellumRun:
    """Cerebellum 运行测试"""
    
    def test_run_with_cache_hit(self, test_config):
        """测试缓存命中直接返回"""
        agent = Cerebellum(config=test_config)
        
        cache_result = MagicMock()
        cache_result.found = True
        cache_result.action = CacheAction.DIRECT_RETURN
        cache_result.cached_files = [
            FileData(name="report.txt", content="缓存结果")
        ]
        cache_result.cached_result = {"message": "从缓存返回"}
        
        with patch.object(agent, '_check_cache', return_value=cache_result):
            result = agent.run("测试任务")
            
            assert result["success"] is True
            assert "缓存" in result["message"]
    
    def test_run_with_files(self, test_config):
        """测试带文件的任务执行"""
        agent = Cerebellum(config=test_config)
        
        files = [
            ("文档内容".encode('utf-8'), "document.txt"),
            (b"\x89PNG", "image.png")
        ]
        
        cache_result = MagicMock()
        cache_result.found = True
        cache_result.action = CacheAction.DIRECT_RETURN
        cache_result.cached_files = [
            FileData(name="output.txt", content="结果")
        ]
        cache_result.cached_result = {"message": "处理完成"}
        
        with patch.object(agent, '_check_cache', return_value=cache_result):
            result = agent.run("处理这些文件", files=files)
            
            assert result["success"] is True


class TestCerebellumContextManager:
    """Cerebellum 上下文管理器测试"""
    
    def test_context_manager_enter_exit(self, test_config):
        """测试上下文管理器"""
        with Cerebellum(config=test_config) as agent:
            assert agent is not None
            assert isinstance(agent, Cerebellum)


class TestFileHandling:
    """文件处理测试"""
    
    @pytest.fixture
    def agent(self, test_config):
        """创建 Cerebellum 实例"""
        return Cerebellum(config=test_config)
    
    def test_build_file_prompt_with_files(self, agent):
        """测试构建带文件的提示"""
        files = [
            ("文档内容".encode('utf-8'), "report.txt"),
            (b"\x89PNG", "image.png")
        ]
        
        result = {
            "uploaded_files": []
        }
        
        prompt = agent._build_file_prompt("分析这些文件", files, result)
        
        assert "report.txt" in prompt
        assert len(result["uploaded_files"]) == 2
    
    def test_build_file_prompt_with_empty_files(self, agent):
        """测试构建空文件列表的提示"""
        result = {
            "uploaded_files": []
        }
        
        prompt = agent._build_file_prompt("分析任务", [], result)
        
        assert "分析任务" in prompt
    
    def test_get_file_type_returns_mime_type(self, agent):
        """测试获取文件类型返回 MIME 类型"""
        file_type = agent._get_file_type("document.txt")
        
        assert "text" in file_type
    
    def test_get_file_type_image(self, agent):
        """测试获取图片文件类型"""
        file_type = agent._get_file_type("image.png")
        
        assert "image" in file_type
    
    def test_is_valid_file_txt(self, agent):
        """测试验证 txt 文件"""
        assert agent._is_valid_file("document.txt") is True


class TestReporterIntegration:
    """Reporter 集成测试"""
    
    def test_reporter_initialization(self):
        """测试 Reporter 初始化"""
        from cerebellum.orchestrator.reporter import TaskReporter
        
        reporter = TaskReporter()
        
        assert reporter is not None
    
    def test_report_task_start(self):
        """测试报告任务开始"""
        from cerebellum.orchestrator.reporter import TaskReporter
        
        reporter = TaskReporter()
        subtasks = [
            SubTask(
                id="st1",
                name="子任务1",
                description="描述",
                priority=1,
                dependencies=[],
                status=SubTaskStatus.PENDING,
                assigned_skill="test"
            )
        ]
        
        reporter.report_task_start("测试任务", subtasks)
    
    def test_report_subtask_complete(self):
        """测试报告子任务完成"""
        from cerebellum.orchestrator.reporter import TaskReporter
        
        reporter = TaskReporter()
        subtask = SubTask(
            id="st1",
            name="子任务1",
            description="描述",
            priority=1,
            dependencies=[],
            status=SubTaskStatus.COMPLETED,
            assigned_skill="test"
        )
        
        reporter.report_subtask_complete(subtask, "执行结果")
    
    def test_report_cache_hit(self):
        """测试报告缓存命中"""
        from cerebellum.orchestrator.reporter import TaskReporter
        
        reporter = TaskReporter()
        
        reporter.report_cache_hit("abc123", 0.95, "direct_return")


class TestCacheSystem:
    """缓存系统测试"""
    
    def test_check_cache_returns_none_when_no_cache_manager(self, test_config):
        """测试无缓存管理器时返回 None"""
        agent = Cerebellum(config=test_config)
        agent.cache_manager = None
        
        result = agent._check_cache("测试任务", None)
        
        assert result is None


class TestConfigLoading:
    """配置加载测试"""
    
    def test_config_defaults(self):
        """测试配置默认值"""
        config = CerebellumConfig()
        
        assert config.cache.similarity_threshold == 0.85
        assert config.cache.cache_ttl_days == 30
        assert config.reflection.max_reflections == 10
    
    def test_config_custom_values(self):
        """测试自定义配置值"""
        config = CerebellumConfig(
            dashscope_api_key="test_key",
            cache=CacheConfig(similarity_threshold=0.9),
            reflection=ReflectionConfig(max_reflections=5)
        )
        
        assert config.dashscope_api_key == "test_key"
        assert config.cache.similarity_threshold == 0.9
        assert config.reflection.max_reflections == 5

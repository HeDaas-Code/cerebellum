"""
similarity.py 单元测试

测试相似度检查器
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
import json

from cerebellum.data.similarity import SimilarityChecker
from cerebellum.types import TaskRecord, TaskStatus, SimilarityResult
from cerebellum.data.database import DatabaseManager


class TestSimilarityChecker:
    """SimilarityChecker 测试"""
    
    @pytest.fixture
    def mock_db(self):
        """模拟数据库"""
        mock = MagicMock(spec=DatabaseManager)
        mock.get_all_tasks = MagicMock(return_value=[])
        mock.save_similarity = MagicMock(return_value=True)
        return mock
    
    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value=MagicMock(
            content='{"score": 0.85, "reasoning": "任务相似"}'
        ))
        return mock
    
    @pytest.fixture
    def similarity_checker(self, mock_llm, mock_db):
        """相似度检查器实例"""
        return SimilarityChecker(
            llm=mock_llm,
            db=mock_db,
            threshold=0.85
        )
    
    def _create_task_record(self, task_hash, task_content, normalized_content, intent, status=TaskStatus.COMPLETED):
        """辅助方法：创建 TaskRecord"""
        return TaskRecord(
            task_hash=task_hash,
            task_content=task_content,
            normalized_content=normalized_content,
            intent=intent,
            parameters={},
            file_hashes=[],
            status=status,
            created_at=datetime.now()
        )
    
    def test_check_no_historical_tasks(self, similarity_checker, mock_db):
        """测试无历史任务"""
        mock_db.get_all_tasks.return_value = []
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.is_similar is False
        assert result.score == 0.0
        assert result.matched_task is None
        assert "没有历史任务记录" in result.reasoning
    
    def test_check_with_similar_task(self, similarity_checker, mock_db, mock_llm):
        """测试找到相似任务"""
        historical_task = self._create_task_record(
            "def456", "分析报告", "分析 报告", "analyze"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.92, "reasoning": "都是分析任务"}'
        )
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.is_similar is True
        assert result.score == 0.92
        assert result.matched_task == historical_task
    
    def test_check_below_threshold(self, similarity_checker, mock_db, mock_llm):
        """测试相似度低于阈值"""
        historical_task = self._create_task_record(
            "def456", "生成图片", "生成 图片", "generate"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.5, "reasoning": "任务不同"}'
        )
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.is_similar is False
        assert result.score == 0.5
    
    def test_check_skip_same_task(self, similarity_checker, mock_db):
        """测试跳过相同任务"""
        historical_task = self._create_task_record(
            "abc123", "分析文档", "分析 文档", "analyze"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.is_similar is False
        assert result.score == 0.0
    
    def test_check_multiple_tasks_best_match(self, similarity_checker, mock_db, mock_llm):
        """测试多个任务选择最佳匹配"""
        tasks = [
            self._create_task_record("task1", "生成报告", "生成 报告", "generate"),
            self._create_task_record("task2", "分析文档", "分析 文档", "analyze")
        ]
        mock_db.get_all_tasks.return_value = tasks
        
        call_count = [0]
        def mock_invoke(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return MagicMock(content='{"score": 0.6, "reasoning": "部分相似"}')
            else:
                return MagicMock(content='{"score": 0.95, "reasoning": "非常相似"}')
        
        mock_llm.invoke.side_effect = mock_invoke
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.score == 0.95
        assert result.matched_task.task_hash == "task2"
    
    def test_llm_parse_error(self, similarity_checker, mock_db, mock_llm):
        """测试 LLM 响应解析错误"""
        historical_task = self._create_task_record(
            "def456", "分析报告", "分析 报告", "analyze"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        mock_llm.invoke.return_value = MagicMock(content="invalid json response")
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.score == 0.0
    
    def test_llm_exception(self, similarity_checker, mock_db, mock_llm):
        """测试 LLM 调用异常"""
        historical_task = self._create_task_record(
            "def456", "分析报告", "分析 报告", "analyze"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        result = similarity_checker.check("abc123", "分析文档", "analyze")
        
        assert result.score == 0.0
    
    def test_save_similarity_called(self, similarity_checker, mock_db, mock_llm):
        """测试保存相似度记录"""
        historical_task = self._create_task_record(
            "def456", "分析报告", "分析 报告", "analyze"
        )
        mock_db.get_all_tasks.return_value = [historical_task]
        
        mock_llm.invoke.return_value = MagicMock(
            content='{"score": 0.9, "reasoning": "相似"}'
        )
        
        similarity_checker.check("abc123", "分析文档", "analyze")
        
        mock_db.save_similarity.assert_called_once_with("abc123", "def456", 0.9)
    
    def test_custom_threshold(self, mock_llm, mock_db):
        """测试自定义阈值"""
        checker = SimilarityChecker(
            llm=mock_llm,
            db=mock_db,
            threshold=0.95
        )
        
        assert checker.threshold == 0.95

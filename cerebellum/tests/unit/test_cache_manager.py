"""
cache_manager.py 单元测试

测试缓存管理器
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from cerebellum.data.cache_manager import CacheManager
from cerebellum.types import CacheAction, FileData, TaskRecord, TaskStatus
from cerebellum.config import CacheConfig
from cerebellum.data.similarity import SimilarityChecker
from cerebellum.data.database import DatabaseManager


class TestCacheManager:
    """CacheManager 测试"""
    
    @pytest.fixture
    def mock_db(self):
        """模拟数据库"""
        mock = MagicMock(spec=DatabaseManager)
        mock.get_result_cache = MagicMock(return_value=None)
        mock.save_result_cache = MagicMock(return_value=True)
        mock.cleanup_old_cache = MagicMock(return_value=0)
        return mock
    
    @pytest.fixture
    def mock_similarity(self):
        """模拟相似度检查器"""
        mock = MagicMock(spec=SimilarityChecker)
        mock.check = MagicMock(return_value=None)
        return mock
    
    @pytest.fixture
    def cache_manager(self, mock_db, mock_similarity):
        """缓存管理器实例"""
        return CacheManager(
            db=mock_db,
            similarity_checker=mock_similarity,
            config=CacheConfig(similarity_threshold=0.85)
        )
    
    def _create_task_record(self, task_hash):
        """辅助方法：创建 TaskRecord"""
        return TaskRecord(
            task_hash=task_hash,
            task_content="测试任务",
            normalized_content="测试 任务",
            intent="test",
            parameters={},
            file_hashes=[],
            status=TaskStatus.COMPLETED,
            created_at=datetime.now()
        )
    
    def test_check_cache_exact_match(self, cache_manager, mock_db):
        """测试精确匹配缓存"""
        mock_db.get_result_cache.return_value = {
            "result_message": "执行成功",
            "files_data": [{"name": "output.txt", "content": "结果内容".encode('utf-8')}],
            "files_metadata": [{"name": "output.txt", "type": "text"}]
        }
        
        result = cache_manager.check_cache("abc123", "分析文档", "analyze")
        
        assert result.found is True
        assert result.similarity_score == 1.0
        assert result.action == CacheAction.DIRECT_RETURN
        mock_db.get_result_cache.assert_called_once_with("abc123")
    
    def test_check_cache_no_match(self, cache_manager, mock_db, mock_similarity):
        """测试缓存未命中"""
        mock_db.get_result_cache.return_value = None
        mock_similarity.check.return_value = type('obj', (object,), {
            'is_similar': False,
            'score': 0.0,
            'matched_task': None
        })()
        
        result = cache_manager.check_cache("abc123", "分析文档", "analyze")
        
        assert result.found is False
        assert result.action == CacheAction.NEW_TASK
    
    def test_check_cache_similar_match(self, cache_manager, mock_db, mock_similarity):
        """测试相似匹配缓存"""
        mock_db.get_result_cache.return_value = None
        
        matched_task = self._create_task_record("def456")
        
        mock_similarity.check.return_value = type('obj', (object,), {
            'is_similar': True,
            'score': 0.92,
            'matched_task': matched_task
        })()
        
        mock_db.get_result_cache.side_effect = lambda h: {
            "result_message": "执行成功",
            "files_data": [],
            "files_metadata": []
        } if h == "def456" else None
        
        result = cache_manager.check_cache("abc123", "分析文档", "analyze")
        
        assert result.found is True
        assert result.similarity_score == 0.92
        assert result.action == CacheAction.CONTINUE_PROCESSING
    
    def test_store_cache(self, cache_manager, mock_db):
        """测试存储缓存"""
        files = [
            FileData(name="output.txt", content="结果内容"),
            FileData(name="data.bin", content=b"\x00\x01")
        ]
        
        cache_manager.store_cache("abc123", "执行成功", files)
        
        mock_db.save_result_cache.assert_called_once()
        call_args = mock_db.save_result_cache.call_args
        assert call_args[0][0] == "abc123"
        assert call_args[0][1] == "执行成功"
    
    def test_serialize_files(self, cache_manager):
        """测试文件序列化"""
        files = [
            FileData(name="test.txt", content="文本内容"),
            FileData(name="binary.bin", content=b"\x00\x01\x02")
        ]
        
        data, metadata = cache_manager._serialize_files(files)
        
        assert len(data) == 2
        assert len(metadata) == 2
        assert data[0]["name"] == "test.txt"
        assert metadata[0]["type"] == "text"
        assert metadata[1]["type"] == "binary"
    
    def test_deserialize_files(self, cache_manager):
        """测试文件反序列化"""
        files_data = [
            {"name": "test.txt", "content": "内容".encode('utf-8')},
            {"name": "data.bin", "content": b"\x00\x01"}
        ]
        files_metadata = [
            {"name": "test.txt", "type": "text"},
            {"name": "data.bin", "type": "binary"}
        ]
        
        result = cache_manager._deserialize_files(files_data, files_metadata)
        
        assert len(result) == 2
        assert result[0].name == "test.txt"
        assert result[1].type == "binary"
    
    def test_cleanup(self, cache_manager, mock_db):
        """测试清理过期缓存"""
        mock_db.cleanup_old_cache.return_value = 5
        
        result = cache_manager.cleanup()
        
        assert result == 5
        mock_db.cleanup_old_cache.assert_called_once_with(
            cache_manager.config.cache_ttl_days
        )
    
    def test_check_cache_without_similarity_checker(self, mock_db):
        """测试无相似度检查器时的缓存检查"""
        manager = CacheManager(db=mock_db, similarity_checker=None)
        mock_db.get_result_cache.return_value = None
        
        result = manager.check_cache("abc123", "分析文档", "analyze")
        
        assert result.found is False
        assert result.action == CacheAction.NEW_TASK
    
    def test_config_default_values(self, cache_manager):
        """测试配置默认值"""
        assert cache_manager.config.similarity_threshold == 0.85

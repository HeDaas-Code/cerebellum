"""
database.py 集成测试

测试数据库管理器（使用真实 SQLite）
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from cerebellum.data.database import DatabaseManager
from cerebellum.types import TaskRecord, TaskStatus


class TestDatabaseManager:
    """DatabaseManager 集成测试"""
    
    @pytest.fixture
    def db(self, temp_db_path):
        """数据库管理器实例"""
        return DatabaseManager(temp_db_path)
    
    def test_init_creates_database_file(self, temp_db_path):
        """测试初始化创建数据库文件"""
        db = DatabaseManager(temp_db_path)
        
        assert temp_db_path.exists()
    
    def test_init_creates_tables(self, db, temp_db_path):
        """测试初始化创建数据表"""
        import sqlite3
        
        conn = sqlite3.connect(str(temp_db_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        assert "tasks" in tables
        assert "result_cache" in tables
        assert "similarity_cache" in tables
    
    def test_save_and_get_task(self, db):
        """测试保存和获取任务"""
        task = TaskRecord(
            task_hash="test_hash_001",
            task_content="分析文档",
            normalized_content="分析 文档",
            intent="analyze",
            parameters={"output": "markdown"},
            file_hashes=["file1", "file2"],
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        task_id = db.save_task(task)
        
        assert task_id > 0
        
        retrieved = db.get_task_by_hash("test_hash_001")
        
        assert retrieved is not None
        assert retrieved.task_hash == "test_hash_001"
        assert retrieved.task_content == "分析文档"
        assert retrieved.intent == "analyze"
        assert retrieved.parameters == {"output": "markdown"}
        assert retrieved.file_hashes == ["file1", "file2"]
    
    def test_get_nonexistent_task(self, db):
        """测试获取不存在的任务"""
        result = db.get_task_by_hash("nonexistent_hash")
        
        assert result is None
    
    def test_save_task_update(self, db):
        """测试更新任务"""
        task = TaskRecord(
            task_hash="test_hash_002",
            task_content="原始任务",
            normalized_content="原始 任务",
            intent="unknown",
            parameters={},
            file_hashes=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        db.save_task(task)
        
        updated_task = TaskRecord(
            task_hash="test_hash_002",
            task_content="更新后的任务",
            normalized_content="更新 后 任务",
            intent="analyze",
            parameters={"new": "param"},
            file_hashes=["new_file"],
            status=TaskStatus.COMPLETED,
            created_at=datetime.now()
        )
        
        db.save_task(updated_task)
        
        retrieved = db.get_task_by_hash("test_hash_002")
        
        assert retrieved.task_content == "更新后的任务"
        assert retrieved.intent == "analyze"
    
    def test_get_all_tasks(self, db):
        """测试获取所有任务"""
        for i in range(5):
            task = TaskRecord(
                task_hash=f"hash_{i}",
                task_content=f"任务{i}",
                normalized_content=f"任务 {i}",
                intent="test",
                parameters={},
                file_hashes=[],
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
            db.save_task(task)
        
        tasks = db.get_all_tasks(limit=10)
        
        assert len(tasks) == 5
    
    def test_get_all_tasks_with_limit(self, db):
        """测试获取任务限制数量"""
        for i in range(10):
            task = TaskRecord(
                task_hash=f"limit_hash_{i}",
                task_content=f"任务{i}",
                normalized_content=f"任务 {i}",
                intent="test",
                parameters={},
                file_hashes=[],
                status=TaskStatus.PENDING,
                created_at=datetime.now()
            )
            db.save_task(task)
        
        tasks = db.get_all_tasks(limit=3)
        
        assert len(tasks) == 3
    
    def test_save_and_get_result_cache(self, db):
        """测试保存和获取结果缓存"""
        files_data = [
            {"name": "output.txt", "content": "结果内容".encode('utf-8')}
        ]
        files_metadata = [
            {"name": "output.txt", "type": "text"}
        ]
        
        db.save_result_cache(
            task_hash="cache_test_001",
            result_message="执行成功",
            files_data=files_data,
            files_metadata=files_metadata
        )
        
        result = db.get_result_cache("cache_test_001")
        
        assert result is not None
        assert result["result_message"] == "执行成功"
        assert len(result["files_data"]) == 1
        assert result["files_data"][0]["name"] == "output.txt"
    
    def test_get_result_cache_updates_access_count(self, db):
        """测试获取缓存更新访问计数"""
        db.save_result_cache(
            task_hash="access_test",
            result_message="测试",
            files_data=[],
            files_metadata=[]
        )
        
        db.get_result_cache("access_test")
        result = db.get_result_cache("access_test")
        
        assert result["access_count"] == 2
    
    def test_get_nonexistent_cache(self, db):
        """测试获取不存在的缓存"""
        result = db.get_result_cache("nonexistent_cache")
        
        assert result is None
    
    def test_save_similarity(self, db):
        """测试保存相似度记录"""
        db.save_similarity(
            new_task_hash="new_task",
            matched_task_hash="matched_task",
            score=0.92
        )
        
        import sqlite3
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM similarity_cache WHERE new_task_hash = ?", ("new_task",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
    
    def test_cleanup_old_cache(self, db):
        """测试清理过期缓存"""
        import sqlite3
        
        old_date = (datetime.now() - timedelta(days=40)).isoformat()
        
        conn = sqlite3.connect(str(db.db_path))
        cursor = conn.cursor()
        
        files_data = pickle.dumps([{"name": "old.txt", "content": b"old"}])
        cursor.execute("""
            INSERT INTO result_cache (task_hash, result_message, files_data, files_metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("old_cache", "旧缓存", files_data, "[]", old_date))
        
        new_files_data = pickle.dumps([{"name": "new.txt", "content": b"new"}])
        cursor.execute("""
            INSERT INTO result_cache (task_hash, result_message, files_data, files_metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("new_cache", "新缓存", new_files_data, "[]", datetime.now().isoformat()))
        conn.commit()
        conn.close()
        
        deleted = db.cleanup_old_cache(days=30)
        
        assert deleted >= 1
        
        result = db.get_result_cache("new_cache")
        assert result is not None
    
    def test_cleanup_keeps_recent_cache(self, db):
        """测试清理保留近期缓存"""
        db.save_result_cache(
            task_hash="recent_cache",
            result_message="近期缓存",
            files_data=[],
            files_metadata=[]
        )
        
        db.cleanup_old_cache(days=30)
        
        result = db.get_result_cache("recent_cache")
        
        assert result is not None
    
    def test_database_path_creation(self, temp_dir):
        """测试数据库路径自动创建"""
        nested_path = temp_dir / "nested" / "dir" / "test.db"
        
        db = DatabaseManager(nested_path)
        
        assert nested_path.parent.exists()
        assert nested_path.exists()
    
    def test_concurrent_access(self, db):
        """测试并发访问安全性"""
        import threading
        
        errors = []
        
        def save_task(i):
            try:
                task = TaskRecord(
                    task_hash=f"concurrent_{i}",
                    task_content=f"并发任务{i}",
                    normalized_content=f"并发 任务 {i}",
                    intent="test",
                    parameters={},
                    file_hashes=[],
                    status=TaskStatus.PENDING,
                    created_at=datetime.now()
                )
                db.save_task(task)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=save_task, args=(i,)) for i in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        
        tasks = db.get_all_tasks(limit=20)
        concurrent_tasks = [t for t in tasks if t.task_hash.startswith("concurrent_")]
        assert len(concurrent_tasks) == 10
    
    def test_unicode_content(self, db):
        """测试 Unicode 内容存储"""
        task = TaskRecord(
            task_hash="unicode_test",
            task_content="分析中文文档，生成报告📊",
            normalized_content="分析 中文 文档 生成 报告",
            intent="analyze",
            parameters={"emoji": "🎉", "chinese": "测试"},
            file_hashes=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        db.save_task(task)
        
        retrieved = db.get_task_by_hash("unicode_test")
        
        assert "📊" in retrieved.task_content
        assert retrieved.parameters["emoji"] == "🎉"

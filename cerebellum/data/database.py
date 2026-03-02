"""
Cerebellum 数据库管理模块

使用 SQLite 存储任务记录和缓存结果
"""

import sqlite3
import json
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from ..types import TaskRecord, TaskStatus
from ..utils import logger


class DatabaseManager:
    """
    SQLite 数据库管理器
    
    管理任务记录和缓存结果
    """
    
    def __init__(self, db_path: Path):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"初始化数据库: {self.db_path}")
        self._init_tables()
    
    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_tables(self):
        """初始化数据表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_hash TEXT UNIQUE NOT NULL,
                    task_content TEXT NOT NULL,
                    normalized_content TEXT,
                    intent TEXT,
                    parameters TEXT,
                    file_hashes TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    execution_time_ms INTEGER
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS result_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_hash TEXT NOT NULL,
                    result_message TEXT,
                    files_data BLOB,
                    files_metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    FOREIGN KEY (task_hash) REFERENCES tasks(task_hash)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS similarity_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    new_task_hash TEXT NOT NULL,
                    matched_task_hash TEXT NOT NULL,
                    similarity_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_hash ON tasks(task_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_result_task_hash ON result_cache(task_hash)
            """)
    
    def save_task(self, task: TaskRecord) -> int:
        """
        保存任务记录
        
        Args:
            task: 任务记录
        
        Returns:
            任务 ID
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tasks 
                (task_hash, task_content, normalized_content, intent, parameters, file_hashes, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_hash,
                task.task_content,
                task.normalized_content,
                task.intent,
                json.dumps(task.parameters, ensure_ascii=False),
                json.dumps(task.file_hashes),
                task.status.value,
                task.created_at.isoformat()
            ))
            return cursor.lastrowid
    
    def get_task_by_hash(self, task_hash: str) -> Optional[TaskRecord]:
        """
        根据哈希获取任务
        
        Args:
            task_hash: 任务哈希
        
        Returns:
            任务记录或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks WHERE task_hash = ?", (task_hash,))
            row = cursor.fetchone()
            
            if row:
                return TaskRecord(
                    task_hash=row["task_hash"],
                    task_content=row["task_content"],
                    normalized_content=row["normalized_content"],
                    intent=row["intent"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    file_hashes=json.loads(row["file_hashes"]) if row["file_hashes"] else [],
                    status=TaskStatus(row["status"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            return None
    
    def get_all_tasks(self, limit: int = 100) -> List[TaskRecord]:
        """
        获取所有任务
        
        Args:
            limit: 最大数量
        
        Returns:
            任务列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            
            tasks = []
            for row in rows:
                tasks.append(TaskRecord(
                    task_hash=row["task_hash"],
                    task_content=row["task_content"],
                    normalized_content=row["normalized_content"],
                    intent=row["intent"],
                    parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                    file_hashes=json.loads(row["file_hashes"]) if row["file_hashes"] else [],
                    status=TaskStatus(row["status"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                ))
            return tasks
    
    def save_result_cache(self, task_hash: str, result_message: str, files_data: List, files_metadata: List[Dict]):
        """
        保存结果缓存
        
        Args:
            task_hash: 任务哈希
            result_message: 结果消息
            files_data: 文件数据列表
            files_metadata: 文件元数据列表
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO result_cache 
                (task_hash, result_message, files_data, files_metadata, created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                task_hash,
                result_message,
                pickle.dumps(files_data),
                json.dumps(files_metadata, ensure_ascii=False),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
    
    def get_result_cache(self, task_hash: str) -> Optional[Dict[str, Any]]:
        """
        获取结果缓存
        
        Args:
            task_hash: 任务哈希
        
        Returns:
            缓存结果或 None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM result_cache WHERE task_hash = ?", (task_hash,))
            row = cursor.fetchone()
            
            if row:
                cursor.execute(
                    "UPDATE result_cache SET last_accessed = ?, access_count = access_count + 1 WHERE task_hash = ?",
                    (datetime.now().isoformat(), task_hash)
                )
                
                return {
                    "result_message": row["result_message"],
                    "files_data": pickle.loads(row["files_data"]) if row["files_data"] else [],
                    "files_metadata": json.loads(row["files_metadata"]) if row["files_metadata"] else [],
                    "created_at": row["created_at"],
                    "access_count": row["access_count"] + 1
                }
            return None
    
    def save_similarity(self, new_task_hash: str, matched_task_hash: str, score: float):
        """
        保存相似度记录
        
        Args:
            new_task_hash: 新任务哈希
            matched_task_hash: 匹配的任务哈希
            score: 相似度分数
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO similarity_cache (new_task_hash, matched_task_hash, similarity_score)
                VALUES (?, ?, ?)
            """, (new_task_hash, matched_task_hash, score))
    
    def cleanup_old_cache(self, days: int = 30) -> int:
        """
        清理过期缓存
        
        Args:
            days: 过期天数
        
        Returns:
            删除的记录数
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM result_cache 
                WHERE datetime(created_at) < datetime('now', ?)
            """, (f"-{days} days",))
            return cursor.rowcount

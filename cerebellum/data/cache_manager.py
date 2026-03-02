"""
Cerebellum 缓存管理模块

管理任务结果缓存
"""

from typing import Optional, Dict, Any, List
from pathlib import Path

from ..types import CacheResult, CacheAction, FileData
from ..config import CacheConfig
from ..utils import logger
from .database import DatabaseManager
from .similarity import SimilarityChecker


class CacheManager:
    """
    缓存管理器
    
    管理任务结果缓存，支持相似度匹配
    """
    
    def __init__(
        self, 
        db: DatabaseManager, 
        similarity_checker: Optional[SimilarityChecker] = None,
        config: Optional[CacheConfig] = None
    ):
        """
        初始化缓存管理器
        
        Args:
            db: 数据库管理器
            similarity_checker: 相似度检查器
            config: 缓存配置
        """
        self.db = db
        self.similarity_checker = similarity_checker
        self.config = config or CacheConfig()
        logger.debug("缓存管理器初始化完成")
    
    def check_cache(
        self, 
        task_hash: str, 
        normalized_content: str, 
        intent: str
    ) -> CacheResult:
        """
        检查缓存
        
        Args:
            task_hash: 任务哈希
            normalized_content: 标准化内容
            intent: 意图
        
        Returns:
            缓存结果
        """
        logger.debug(f"检查缓存: task_hash={task_hash[:8]}...")
        exact_cache = self.db.get_result_cache(task_hash)
        if exact_cache:
            logger.debug("缓存命中（精确匹配）")
            return CacheResult(
                found=True,
                task_hash=task_hash,
                similarity_score=1.0,
                cached_files=self._deserialize_files(exact_cache["files_data"], exact_cache["files_metadata"]),
                cached_result={"message": exact_cache["result_message"]},
                action=CacheAction.DIRECT_RETURN
            )
        
        if self.similarity_checker:
            logger.debug("检查相似度匹配...")
            similarity_result = self.similarity_checker.check(task_hash, normalized_content, intent)
            
            if similarity_result.is_similar and similarity_result.matched_task:
                logger.debug(f"相似度匹配成功: score={similarity_result.score:.2f}")
                cached = self.db.get_result_cache(similarity_result.matched_task.task_hash)
                if cached:
                    return CacheResult(
                        found=True,
                        task_hash=similarity_result.matched_task.task_hash,
                        similarity_score=similarity_result.score,
                        cached_files=self._deserialize_files(cached["files_data"], cached["files_metadata"]),
                        cached_result={"message": cached["result_message"]},
                        action=CacheAction.CONTINUE_PROCESSING
                    )
        
        logger.debug("缓存未命中")
        return CacheResult(
            found=False,
            task_hash=task_hash,
            similarity_score=0.0,
            action=CacheAction.NEW_TASK
        )
    
    def store_cache(
        self, 
        task_hash: str, 
        result_message: str, 
        files: List[FileData]
    ):
        """
        存储缓存
        
        Args:
            task_hash: 任务哈希
            result_message: 结果消息
            files: 文件列表
        """
        logger.debug(f"存储缓存: task_hash={task_hash[:8]}...")
        files_data, files_metadata = self._serialize_files(files)
        self.db.save_result_cache(task_hash, result_message, files_data, files_metadata)
        logger.debug("缓存存储完成")
    
    def _serialize_files(self, files: List[FileData]) -> tuple:
        """序列化文件列表"""
        files_data = []
        files_metadata = []
        
        for f in files:
            files_data.append({
                "name": f.name,
                "content": f.get_bytes()
            })
            files_metadata.append({
                "name": f.name,
                "type": f.type
            })
        
        return files_data, files_metadata
    
    def _deserialize_files(self, files_data: List, files_metadata: List) -> List[FileData]:
        """反序列化文件列表"""
        files = []
        
        for data, meta in zip(files_data, files_metadata):
            files.append(FileData(
                name=data.get("name", meta.get("name", "unknown")),
                content=data.get("content", b""),
                type=meta.get("type", "binary")
            ))
        
        return files
    
    def cleanup(self) -> int:
        """
        清理过期缓存
        
        Returns:
            删除的记录数
        """
        return self.db.cleanup_old_cache(self.config.cache_ttl_days)

"""
Cerebellum 任务编码模块

对任务进行标准化编码，生成唯一哈希
"""

import hashlib
import json
from typing import List, Tuple, Any


class TaskEncoder:
    """
    任务编码器
    
    对任务内容进行标准化处理并生成唯一哈希
    """
    
    @staticmethod
    def normalize_task(task: str) -> str:
        """
        标准化任务描述
        
        Args:
            task: 原始任务描述
        
        Returns:
            标准化后的任务描述
        """
        normalized = task.strip().lower()
        normalized = ' '.join(normalized.split())
        return normalized
    
    @staticmethod
    def extract_intent(task: str) -> str:
        """
        提取任务意图
        
        Args:
            task: 任务描述
        
        Returns:
            意图关键词
        """
        intent_keywords = {
            '总结': 'summarize',
            '分析': 'analyze',
            '生成': 'generate',
            '创建': 'create',
            '转换': 'convert',
            '提取': 'extract',
            '比较': 'compare',
            '计算': 'calculate',
            '图表': 'chart',
            '图片': 'image',
            '文档': 'document',
            '报告': 'report',
            '表格': 'table',
        }
        
        task_lower = task.lower()
        intents = []
        
        for keyword, intent in intent_keywords.items():
            if keyword in task_lower:
                intents.append(intent)
        
        return ','.join(intents) if intents else 'unknown'
    
    @staticmethod
    def compute_file_hash(file_data: bytes) -> str:
        """
        计算文件哈希
        
        Args:
            file_data: 文件数据
        
        Returns:
            文件哈希值
        """
        return hashlib.sha256(file_data).hexdigest()[:16]
    
    @classmethod
    def encode(
        cls, 
        task: str, 
        files: List[Tuple[bytes, str]] = None
    ) -> Tuple[str, str, str, List[str]]:
        """
        编码任务
        
        Args:
            task: 任务描述
            files: 文件列表 [(data, filename), ...]
        
        Returns:
            (task_hash, normalized_content, intent, file_hashes)
        """
        normalized = cls.normalize_task(task)
        intent = cls.extract_intent(task)
        
        file_hashes = []
        if files:
            for file_data, _ in files:
                if isinstance(file_data, bytes):
                    file_hashes.append(cls.compute_file_hash(file_data))
        
        hash_content = {
            "normalized": normalized,
            "intent": intent,
            "file_hashes": sorted(file_hashes)
        }
        
        hash_str = json.dumps(hash_content, sort_keys=True)
        task_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:32]
        
        return task_hash, normalized, intent, file_hashes

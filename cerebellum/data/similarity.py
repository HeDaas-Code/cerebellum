"""
Cerebellum 相似度比对模块

使用 LLM 比对任务相似度
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI

from ..types import TaskRecord, SimilarityResult
from .database import DatabaseManager


class SimilarityChecker:
    """
    任务相似度检查器
    
    使用 LLM 比对新任务与历史任务的相似度
    """
    
    def __init__(self, llm: ChatOpenAI, db: DatabaseManager, threshold: float = 0.85):
        """
        初始化相似度检查器
        
        Args:
            llm: LLM 客户端
            db: 数据库管理器
            threshold: 相似度阈值
        """
        self.llm = llm
        self.db = db
        self.threshold = threshold
    
    def check(
        self, 
        task_hash: str, 
        normalized_content: str, 
        intent: str
    ) -> SimilarityResult:
        """
        检查任务相似度
        
        Args:
            task_hash: 新任务哈希
            normalized_content: 标准化内容
            intent: 意图
        
        Returns:
            相似度结果
        """
        historical_tasks = self.db.get_all_tasks(limit=50)
        
        if not historical_tasks:
            return SimilarityResult(
                is_similar=False,
                score=0.0,
                matched_task=None,
                reasoning="没有历史任务记录"
            )
        
        best_match = None
        best_score = 0.0
        best_reasoning = ""
        
        for hist_task in historical_tasks:
            if hist_task.task_hash == task_hash:
                continue
            
            score, reasoning = self._compare_with_llm(
                normalized_content, 
                intent,
                hist_task.normalized_content,
                hist_task.intent
            )
            
            if score > best_score:
                best_score = score
                best_match = hist_task
                best_reasoning = reasoning
        
        is_similar = best_score >= self.threshold
        
        if best_match:
            self.db.save_similarity(task_hash, best_match.task_hash, best_score)
        
        return SimilarityResult(
            is_similar=is_similar,
            score=best_score,
            matched_task=best_match,
            reasoning=best_reasoning
        )
    
    def _compare_with_llm(
        self, 
        new_content: str, 
        new_intent: str,
        hist_content: str, 
        hist_intent: str
    ) -> tuple:
        """
        使用 LLM 比对两个任务
        
        Args:
            new_content: 新任务内容
            new_intent: 新任务意图
            hist_content: 历史任务内容
            hist_intent: 历史任务意图
        
        Returns:
            (相似度分数, 理由)
        """
        prompt = f"""比较以下两个任务的相似度，返回 0-1 之间的分数。

新任务:
- 内容: {new_content}
- 意图: {new_intent}

历史任务:
- 内容: {hist_content}
- 意图: {hist_intent}

请分析:
1. 任务目标是否相同？
2. 处理方式是否相似？
3. 预期输出是否相似？

返回 JSON 格式:
{{"score": 0.85, "reasoning": "理由说明"}}
"""
        
        try:
            response = self.llm.invoke(prompt)
            import json
            result = json.loads(response.content.strip().replace("```json", "").replace("```", ""))
            return result.get("score", 0.0), result.get("reasoning", "")
        except Exception:
            return 0.0, "比对失败"

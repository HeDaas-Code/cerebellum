"""
Cerebellum 相似度比对模块

使用 LLM 比对任务相似度
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI

from ..types import TaskRecord, SimilarityResult
from .database import DatabaseManager
from ..utils import logger


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
        logger.debug(f"[相似度检查] 开始检查: {normalized_content[:50]}...")
        historical_tasks = self.db.get_all_tasks(limit=50)
        
        if not historical_tasks:
            logger.debug("[相似度检查] 没有历史任务记录")
            return SimilarityResult(
                is_similar=False,
                score=0.0,
                matched_task=None,
                reasoning="没有历史任务记录"
            )
        
        logger.debug(f"[相似度检查] 找到 {len(historical_tasks)} 个历史任务")
        
        best_match = None
        best_score = 0.0
        best_reasoning = ""
        
        for i, hist_task in enumerate(historical_tasks):
            if hist_task.task_hash == task_hash:
                continue
            
            score, reasoning = self._compare_with_llm(
                normalized_content, 
                intent,
                hist_task.normalized_content,
                hist_task.intent
            )
            
            logger.debug(f"[相似度检查] 对比任务 {i+1}/{len(historical_tasks)}: 分数={score:.2f}")
            
            if score > best_score:
                best_score = score
                best_match = hist_task
                best_reasoning = reasoning
        
        is_similar = best_score >= self.threshold
        
        if best_match:
            self.db.save_similarity(task_hash, best_match.task_hash, best_score)
            logger.debug(f"[相似度检查] 最佳匹配: {best_match.normalized_content[:30]}..., 分数: {best_score:.2f}")
        
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
            
            content = response.content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                        elif 'text' in block:
                            text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                content = '\n'.join(text_parts)
            else:
                content = str(content)
            
            result = json.loads(content.strip().replace("```json", "").replace("```", ""))
            score_val = result.get("score", 0.0)
            reasoning_val = result.get("reasoning", "")
            logger.debug(f"[LLM相似度比对] 分数: {score_val}, 理由: {reasoning_val}")
            return score_val, reasoning_val
        except Exception as e:
            logger.debug(f"[LLM相似度比对] 比对失败: {e}")
            return 0.0, "比对失败"

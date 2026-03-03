"""
Cerebellum 任务规划模块

分析和拆分任务
"""

import json
import uuid
import re
from typing import List, Tuple, Dict
from langchain_openai import ChatOpenAI

from ..types import SubTask, SubTaskStatus
from ..utils import logger


class TaskPlanner:
    """
    任务规划器
    
    分析任务意图并拆分为子任务
    """
    
    INTENT_KEYWORDS: Dict[str, List[str]] = {
        "图表生成": ["图表", "可视化", "绘制", "plot", "chart", "graph"],
        "文档总结": ["总结", "摘要", "概括", "归纳", "summarize"],
        "数据分析": ["分析", "统计", "计算", "数据", "analysis"],
        "代码生成": ["代码", "编写", "实现", "开发", "code"],
        "文件处理": ["文件", "读取", "转换", "处理", "file"],
        "搜索查询": ["搜索", "查询", "查找", "search", "find"],
    }
    
    OUTPUT_TYPE_KEYWORDS: Dict[str, List[str]] = {
        "图表": ["图表", "可视化", "绘制", "plot", "chart", "graph", "图片"],
        "文本": ["总结", "摘要", "报告", "文档", "文本"],
        "代码": ["代码", "脚本", "程序"],
        "数据": ["数据", "表格", "csv", "excel"],
    }
    
    SKILL_KEYWORDS: Dict[str, List[str]] = {
        "pdf": ["pdf", "文档", "报告", "report"],
        "xlsx": ["excel", "表格", "数据", "xlsx", "csv", "spreadsheet"],
        "docx": ["word", "文档", "docx", "document"],
        "pptx": ["ppt", "演示", "幻灯片", "pptx", "presentation"],
        "chart": ["图表", "可视化", "绘制", "plot", "chart", "graph", "图片"],
        "image": ["图片", "图像", "图片处理", "image"],
        "web": ["网页", "网站", "web", "html"],
        "api": ["api", "接口", "请求"],
    }
    
    def __init__(self, llm: ChatOpenAI):
        """
        初始化任务规划器
        
        Args:
            llm: LLM 客户端
        """
        self.llm = llm
        logger.debug("任务规划器初始化完成")
    
    def analyze_intent(self, task: str) -> Tuple[str, List[str]]:
        """
        分析任务意图（基于关键词匹配）
        
        Args:
            task: 任务描述
        
        Returns:
            (意图描述, 预期输出类型列表)
        """
        logger.debug(f"分析任务意图: {task[:50]}...")
        
        detected_intents = []
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in task.lower():
                    detected_intents.append(intent)
                    break
        
        if not detected_intents:
            detected_intents = ["通用任务"]
        
        outputs = []
        for output_type, keywords in self.OUTPUT_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in task.lower():
                    outputs.append(output_type)
                    break
        
        if not outputs:
            outputs = ["文本"]
        
        intent_str = "、".join(detected_intents)
        logger.debug(f"意图分析完成: intent={intent_str}, outputs={outputs}")
        return intent_str, outputs
    
    def split_task(self, task: str, files_info: List[str] = None) -> List[SubTask]:
        """
        拆分任务为子任务（基于规则）
        
        Args:
            task: 任务描述
            files_info: 上传文件信息
        
        Returns:
            子任务列表
        """
        logger.debug(f"拆分任务: {task[:50]}...")
        
        subtasks = []
        task_lower = task.lower()
        
        if files_info:
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                name="读取文件",
                description=f"读取并分析上传的文件: {', '.join(files_info)}",
                priority=5,
                status=SubTaskStatus.PENDING
            ))
        
        if any(kw in task_lower for kw in ["图表", "可视化", "绘制", "chart", "plot"]):
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                name="生成图表",
                description="根据数据生成可视化图表",
                priority=4,
                status=SubTaskStatus.PENDING
            ))
        
        if any(kw in task_lower for kw in ["总结", "摘要", "概括", "summarize"]):
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                name="总结内容",
                description="总结文档或数据的核心内容",
                priority=4,
                status=SubTaskStatus.PENDING
            ))
        
        if any(kw in task_lower for kw in ["分析", "统计", "analysis"]):
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                name="数据分析",
                description="分析数据并提取关键信息",
                priority=4,
                status=SubTaskStatus.PENDING
            ))
        
        if not subtasks:
            subtasks.append(SubTask(
                id=str(uuid.uuid4())[:8],
                name="执行任务",
                description=task,
                priority=5,
                status=SubTaskStatus.PENDING
            ))
        
        if len(subtasks) > 1:
            for i in range(1, len(subtasks)):
                subtasks[i].dependencies = [subtasks[i-1].id]
        
        logger.debug(f"任务拆分完成: {len(subtasks)} 个子任务")
        return subtasks
    
    def match_skills(self, subtasks: List[SubTask], available_skills: List[str]) -> dict:
        """
        匹配子任务与技能（基于关键词匹配）
        
        Args:
            subtasks: 子任务列表
            available_skills: 可用技能列表
        
        Returns:
            子任务ID到技能的映射
        """
        logger.debug(f"匹配技能: {len(subtasks)} 个子任务, 可用技能: {available_skills}")
        skill_mapping = {}
        
        available_lower = [s.lower() for s in available_skills]
        
        for subtask in subtasks:
            subtask_text = f"{subtask.name} {subtask.description}".lower()
            
            for skill_pattern, keywords in self.SKILL_KEYWORDS.items():
                for kw in keywords:
                    if kw.lower() in subtask_text:
                        # 直接匹配技能名称
                        for i, skill in enumerate(available_lower):
                            if skill_pattern == skill:
                                skill_mapping[subtask.id] = available_skills[i]
                                logger.debug(f"子任务 '{subtask.name}' 匹配技能: {available_skills[i]}")
                                break
                        break
                if subtask.id in skill_mapping:
                    break
        
        logger.debug(f"技能匹配完成: {len(skill_mapping)} 个映射")
        return skill_mapping

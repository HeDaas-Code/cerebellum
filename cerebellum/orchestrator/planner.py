"""
Cerebellum 任务规划模块

分析和拆分任务（支持 LLM 增强和关键词回退）
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
    
    分析任务意图并拆分为子任务，优先使用 LLM，关键词匹配作为回退
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
    
    def __init__(self, llm=None):
        """
        初始化任务规划器
        
        Args:
            llm: LLM 客户端（可选，为 None 时使用纯关键词匹配）
        """
        self.llm = llm
        logger.debug("任务规划器初始化完成")
    
    def _call_llm(self, prompt: str) -> str:
        """
        统一调用 LLM，兼容不同后端的响应格式
        
        Args:
            prompt: 提示词
        
        Returns:
            LLM 响应文本内容
        """
        response = self.llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            content = response.content
            
            if isinstance(content, str):
                return content
            
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
                return '\n'.join(text_parts)
            
            return str(content)
        
        return str(response)
    
    def analyze_intent(self, task: str) -> Tuple[str, List[str]]:
        """
        分析任务意图（优先 LLM，回退关键词匹配）
        
        Args:
            task: 任务描述
        
        Returns:
            (意图描述, 预期输出类型列表)
        """
        logger.debug(f"分析任务意图: {task[:50]}...")
        
        # 优先使用 LLM 分析
        if self.llm:
            try:
                prompt = f"""分析以下任务的意图和预期输出类型。

任务: {task}

返回 JSON 格式:
{{"intent": "任务意图描述", "expected_outputs": ["输出类型1", "输出类型2"]}}

只返回 JSON，不要其他内容。"""
                
                content = self._call_llm(prompt).strip()
                content = content.replace("```json", "").replace("```", "").strip()
                result = json.loads(content)
                
                intent = result.get("intent", "未知意图")
                outputs = result.get("expected_outputs", [])
                
                logger.debug(f"LLM 意图分析完成: intent={intent}, outputs={outputs}")
                return intent, outputs
            except Exception as e:
                logger.debug(f"LLM 意图分析失败: {e}，回退到关键词匹配")
        
        # 回退：关键词匹配
        return self._analyze_intent_by_keywords(task)
    
    def _analyze_intent_by_keywords(self, task: str) -> Tuple[str, List[str]]:
        """基于关键词匹配分析意图（回退方案）"""
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
        logger.debug(f"关键词意图分析完成: intent={intent_str}, outputs={outputs}")
        return intent_str, outputs
    
    def split_task(self, task: str, files_info: List[str] = None) -> List[SubTask]:
        """
        拆分任务为子任务（优先 LLM，回退规则匹配）
        
        Args:
            task: 任务描述
            files_info: 上传文件信息
        
        Returns:
            子任务列表
        """
        logger.debug(f"拆分任务: {task[:50]}...")
        
        # 优先使用 LLM 拆分
        if self.llm:
            try:
                files_context = ""
                if files_info:
                    files_context = f"\n上传的文件: {', '.join(files_info)}"
                
                prompt = f"""将以下任务拆分为子任务。{files_context}

任务: {task}

返回 JSON 数组格式:
[
  {{"name": "子任务名称", "description": "子任务描述", "priority": 5, "dependencies": []}}
]

注意:
- 每个子任务的 dependencies 是它依赖的其他子任务的 name 列表
- priority 范围 1-5，5 为最高优先级
- 只返回 JSON 数组，不要其他内容"""
                
                content = self._call_llm(prompt).strip()
                content = content.replace("```json", "").replace("```", "").strip()
                subtask_defs = json.loads(content)
                
                if isinstance(subtask_defs, list) and len(subtask_defs) > 0:
                    subtasks = []
                    name_to_id = {}
                    
                    for st_def in subtask_defs:
                        st_id = str(uuid.uuid4())[:8]
                        name = st_def.get("name", "未知任务")
                        name_to_id[name] = st_id
                        
                        subtasks.append(SubTask(
                            id=st_id,
                            name=name,
                            description=st_def.get("description", ""),
                            priority=st_def.get("priority", 5),
                            dependencies=[],
                            status=SubTaskStatus.PENDING
                        ))
                    
                    # 解析依赖关系（name -> id）
                    for i, st_def in enumerate(subtask_defs):
                        deps = st_def.get("dependencies", [])
                        if deps:
                            resolved_deps = []
                            for dep_name in deps:
                                dep_id = name_to_id.get(dep_name)
                                if dep_id:
                                    resolved_deps.append(dep_id)
                            subtasks[i].dependencies = resolved_deps
                    
                    logger.debug(f"LLM 任务拆分完成: {len(subtasks)} 个子任务")
                    return subtasks
            except Exception as e:
                logger.debug(f"LLM 任务拆分失败: {e}，回退到规则匹配")
        
        # 回退：规则匹配
        return self._split_task_by_rules(task, files_info)
    
    def _split_task_by_rules(self, task: str, files_info: List[str] = None) -> List[SubTask]:
        """基于规则拆分任务（回退方案）"""
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
        
        logger.debug(f"规则任务拆分完成: {len(subtasks)} 个子任务")
        return subtasks
    
    def match_skills(self, subtasks: List[SubTask], available_skills: List[str]) -> dict:
        """
        匹配子任务与技能（优先 LLM，回退关键词，无匹配时使用 skill-creator）
        
        Args:
            subtasks: 子任务列表
            available_skills: 可用技能列表
        
        Returns:
            子任务ID到技能的映射
        """
        logger.debug(f"匹配技能: {len(subtasks)} 个子任务, 可用技能: {available_skills}")
        skill_mapping = {}
        
        for subtask in subtasks:
            matched_skill = None
            
            # 优先使用 LLM 匹配
            if self.llm:
                try:
                    prompt = f"""从以下可用技能中选择最适合执行此子任务的技能。

子任务: {subtask.name} - {subtask.description}

可用技能: {', '.join(available_skills)}

如果没有合适的技能，返回 "skill-creator" 来自主创建新技能。

只返回技能名称，不要其他内容。"""
                    
                    content = self._call_llm(prompt).strip()
                    
                    # 验证返回的技能名称
                    if content in available_skills or content == "skill-creator":
                        matched_skill = content
                        logger.debug(f"LLM 匹配子任务 '{subtask.name}' -> 技能: {content}")
                except Exception as e:
                    logger.debug(f"LLM 技能匹配失败: {e}，回退到关键词匹配")
            
            # 回退：关键词匹配
            if matched_skill is None:
                matched_skill = self._match_skill_by_keywords(subtask, available_skills)
            
            if matched_skill:
                skill_mapping[subtask.id] = matched_skill
        
        logger.debug(f"技能匹配完成: {len(skill_mapping)} 个映射")
        return skill_mapping
    
    def _match_skill_by_keywords(self, subtask: SubTask, available_skills: List[str]) -> str:
        """基于关键词匹配技能（回退方案，无匹配时返回 skill-creator）"""
        available_lower = [s.lower() for s in available_skills]
        subtask_text = f"{subtask.name} {subtask.description}".lower()
        
        for skill_pattern, keywords in self.SKILL_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in subtask_text:
                    for i, skill in enumerate(available_lower):
                        if skill_pattern == skill:
                            return available_skills[i]
                    break
        
        # 无匹配时回退到 skill-creator（自主创建新技能）
        logger.debug(f"关键词匹配未找到技能: '{subtask.name}'，回退到 skill-creator")
        return "skill-creator"

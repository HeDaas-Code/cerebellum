"""
planner.py 单元测试

测试任务规划器
"""

import pytest
from unittest.mock import MagicMock, patch
import json

from cerebellum.orchestrator.planner import TaskPlanner
from cerebellum.types import SubTask, SubTaskStatus


class TestTaskPlanner:
    """TaskPlanner 测试"""
    
    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def planner(self, mock_llm):
        """任务规划器实例"""
        return TaskPlanner(llm=mock_llm)
    
    def test_analyze_intent_success(self, planner, mock_llm):
        """测试意图分析成功"""
        mock_llm.invoke.return_value = MagicMock(
            content='{"intent": "文档分析", "expected_outputs": ["报告", "图表"]}'
        )
        
        intent, outputs = planner.analyze_intent("分析这份文档")
        
        assert intent == "文档分析"
        assert "报告" in outputs
        assert "图表" in outputs
    
    def test_analyze_intent_parse_error(self, planner, mock_llm):
        """测试意图分析解析错误"""
        mock_llm.invoke.return_value = MagicMock(content="invalid json")
        
        intent, outputs = planner.analyze_intent("做一些事情")
        
        assert intent == "通用任务"
        assert outputs == ["文本"]
    
    def test_analyze_intent_with_markdown(self, planner, mock_llm):
        """测试带 markdown 的响应"""
        mock_llm.invoke.return_value = MagicMock(
            content='```json\n{"intent": "数据分析", "expected_outputs": ["图表"]}\n```'
        )
        
        intent, outputs = planner.analyze_intent("分析数据")
        
        assert intent == "数据分析"
        assert "图表" in outputs
    
    def test_analyze_intent_exception(self, planner, mock_llm):
        """测试 LLM 异常"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        intent, outputs = planner.analyze_intent("做一些事情")
        
        assert intent == "通用任务"
        assert outputs == ["文本"]
    
    def test_split_task_success(self, planner, mock_llm):
        """测试任务拆分成功"""
        mock_llm.invoke.return_value = MagicMock(
            content='''[
                {"name": "读取文档", "description": "读取并解析文档内容", "priority": 5, "dependencies": []},
                {"name": "生成摘要", "description": "生成文档摘要", "priority": 4, "dependencies": ["读取文档"]}
            ]'''
        )
        
        subtasks = planner.split_task("分析文档并生成摘要")
        
        assert len(subtasks) == 2
        assert subtasks[0].name == "读取文档"
        assert subtasks[1].name == "生成摘要"
        assert subtasks[0].id in subtasks[1].dependencies
    
    def test_split_task_with_files(self, planner, mock_llm):
        """测试带文件的任务拆分"""
        mock_llm.invoke.return_value = MagicMock(
            content='[{"name": "处理文件", "description": "处理上传的文件", "priority": 5, "dependencies": []}]'
        )
        
        subtasks = planner.split_task("处理文件", files_info=["report.pdf", "data.xlsx"])
        
        assert len(subtasks) == 1
        call_args = mock_llm.invoke.call_args[0][0]
        assert "report.pdf" in call_args
        assert "data.xlsx" in call_args
    
    def test_split_task_parse_error(self, planner, mock_llm):
        """测试任务拆分解析错误时创建默认任务"""
        mock_llm.invoke.return_value = MagicMock(content="invalid json")
        
        subtasks = planner.split_task("做一些事情")
        
        assert len(subtasks) == 1
        assert subtasks[0].name == "执行任务"
        assert subtasks[0].description == "做一些事情"
    
    def test_split_task_exception(self, planner, mock_llm):
        """测试 LLM 异常时创建默认任务"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        subtasks = planner.split_task("分析文档")
        
        assert len(subtasks) == 1
        assert subtasks[0].status == SubTaskStatus.PENDING
    
    def test_split_task_generates_unique_ids(self, planner, mock_llm):
        """测试生成唯一 ID"""
        mock_llm.invoke.return_value = MagicMock(
            content='''[
                {"name": "任务1", "description": "描述1", "priority": 5, "dependencies": []},
                {"name": "任务2", "description": "描述2", "priority": 4, "dependencies": []}
            ]'''
        )
        
        subtasks = planner.split_task("测试任务")
        
        ids = [s.id for s in subtasks]
        assert len(ids) == len(set(ids))
    
    def test_match_skills_success(self, planner, mock_llm):
        """测试技能匹配成功"""
        subtasks = [
            SubTask(id="st1", name="生成图表", description="创建数据图表"),
            SubTask(id="st2", name="创建文档", description="生成 Word 文档")
        ]
        
        mock_llm.invoke.side_effect = [
            MagicMock(content="chart-creator"),
            MagicMock(content="docx")
        ]
        
        mapping = planner.match_skills(subtasks, ["chart-creator", "docx", "pdf"])
        
        assert mapping["st1"] == "chart-creator"
        assert mapping["st2"] == "docx"
    
    def test_match_skills_fallback_to_creator(self, planner, mock_llm):
        """测试无匹配技能时回退到 skill-creator"""
        subtasks = [
            SubTask(id="st1", name="特殊任务", description="执行特殊操作")
        ]
        
        mock_llm.invoke.return_value = MagicMock(content="skill-creator")
        
        mapping = planner.match_skills(subtasks, ["docx", "pdf"])
        
        assert mapping["st1"] == "skill-creator"
    
    def test_match_skills_exception(self, planner, mock_llm):
        """测试技能匹配异常时回退到 skill-creator"""
        subtasks = [
            SubTask(id="st1", name="任务", description="描述")
        ]
        
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        mapping = planner.match_skills(subtasks, ["skill1"])
        
        # 关键词匹配无法找到时，返回 skill-creator
        assert mapping.get("st1") == "skill-creator"
    
    def test_match_skills_invalid_skill(self, planner, mock_llm):
        """测试返回无效技能名称时回退到 skill-creator"""
        subtasks = [
            SubTask(id="st1", name="任务", description="描述")
        ]
        
        mock_llm.invoke.return_value = MagicMock(content="invalid-skill")
        
        mapping = planner.match_skills(subtasks, ["skill1", "skill2"])
        
        # LLM 返回无效技能名 -> 关键词匹配 -> 无匹配 -> skill-creator
        assert mapping.get("st1") == "skill-creator"
    
    def test_keyword_fallback_returns_skill_creator(self):
        """测试关键词匹配无结果时返回 skill-creator"""
        planner = TaskPlanner(llm=None)
        
        subtask = SubTask(id="st1", name="特殊操作", description="执行一些不常见的操作")
        result = planner._match_skill_by_keywords(subtask, ["pdf", "xlsx"])
        
        assert result == "skill-creator"
    
    def test_keyword_fallback_returns_matched_skill(self):
        """测试关键词匹配找到技能时返回正确技能"""
        planner = TaskPlanner(llm=None)
        
        subtask = SubTask(id="st1", name="创建 PDF 报告", description="生成 PDF 格式的报告")
        result = planner._match_skill_by_keywords(subtask, ["pdf", "xlsx", "chart"])
        
        # "报告" 匹配 pdf 技能关键词
        assert result == "pdf"
    
    def test_match_skills_all_skill_creator_no_llm(self):
        """测试无 LLM 且无关键词匹配时，所有子任务都返回 skill-creator"""
        planner = TaskPlanner(llm=None)
        
        subtasks = [
            SubTask(id="st1", name="自定义任务1", description="执行自定义操作"),
            SubTask(id="st2", name="自定义任务2", description="另一个操作"),
        ]
        
        mapping = planner.match_skills(subtasks, ["pdf", "docx"])
        
        assert mapping.get("st1") == "skill-creator"
        assert mapping.get("st2") == "skill-creator"

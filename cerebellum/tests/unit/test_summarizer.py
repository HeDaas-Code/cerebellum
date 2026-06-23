"""
测试工作流总结器
"""

import pytest
from datetime import datetime

from cerebellum.summary.summarizer import (
    WorkflowSummarizer, WorkflowSummary, PathConstraint
)


class TestPathConstraint:
    """PathConstraint 测试"""
    
    def test_create_positive_constraint(self):
        """测试创建正向约束"""
        c = PathConstraint(
            constraint_type="positive",
            description="使用 pip install 安装依赖",
            source_task="安装Python包",
            confidence=0.8,
            tags=["python", "install"],
        )
        assert c.constraint_type == "positive"
        assert c.confidence == 0.8
        assert "pip" in c.description
    
    def test_create_negative_constraint(self):
        """测试创建负向约束"""
        c = PathConstraint(
            constraint_type="negative",
            description="不要使用 sudo pip install",
            source_task="安装Python包",
            confidence=0.7,
        )
        assert c.constraint_type == "negative"
    
    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        c = PathConstraint(
            constraint_type="positive",
            description="测试约束",
            source_task="测试任务",
            confidence=0.9,
            tags=["test"],
        )
        d = c.to_dict()
        assert d["constraint_type"] == "positive"
        assert d["confidence"] == 0.9
        
        c2 = PathConstraint.from_dict(d)
        assert c2.constraint_type == c.constraint_type
        assert c2.description == c.description
        assert c2.confidence == c.confidence


class TestWorkflowSummary:
    """WorkflowSummary 测试"""
    
    def test_create_summary(self):
        """测试创建总结"""
        s = WorkflowSummary(
            task="生成PDF报告",
            intent="文档生成",
            success=True,
            total_steps=3,
            reflection_count=1,
            skills_used=["pdf"],
            constraints=[],
            key_decisions=["使用reportlab"],
        )
        assert s.success is True
        assert s.total_steps == 3
    
    def test_get_positive_constraints(self):
        """测试获取正向约束"""
        s = WorkflowSummary(
            task="test",
            intent="test",
            success=True,
            total_steps=1,
            reflection_count=0,
            skills_used=[],
            constraints=[
                PathConstraint(constraint_type="positive", description="好的", source_task="t"),
                PathConstraint(constraint_type="negative", description="坏的", source_task="t"),
                PathConstraint(constraint_type="positive", description="也好", source_task="t"),
            ],
            key_decisions=[],
        )
        pos = s.get_positive_constraints()
        neg = s.get_negative_constraints()
        assert len(pos) == 2
        assert len(neg) == 1
    
    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        s = WorkflowSummary(
            task="test",
            intent="test",
            success=True,
            total_steps=1,
            reflection_count=0,
            skills_used=["pdf"],
            constraints=[
                PathConstraint(constraint_type="positive", description="好的", source_task="t"),
            ],
            key_decisions=["决策1"],
            execution_time_ms=1234,
        )
        d = s.to_dict()
        assert d["success"] is True
        assert d["execution_time_ms"] == 1234
        
        s2 = WorkflowSummary.from_dict(d)
        assert s2.task == "test"
        assert len(s2.constraints) == 1


class TestWorkflowSummarizer:
    """WorkflowSummarizer 测试"""
    
    def test_init_without_llm(self):
        """测试无 LLM 初始化"""
        s = WorkflowSummarizer()
        assert s.llm is None
        assert s.summaries == []
    
    def test_summarize_basic(self):
        """测试基础总结"""
        s = WorkflowSummarizer()
        
        result = s.summarize(
            task="生成报告",
            result={"success": True, "subtasks": [{"id": "1"}]},
            intent="文档生成",
            reflection_chains=[
                {
                    "success": True,
                    "problem": "缺少依赖",
                    "solution": "安装 reportlab",
                }
            ],
            skills_used=["pdf"],
            execution_time_ms=5000,
        )
        
        assert result.success is True
        assert result.task == "生成报告"
        assert len(result.constraints) >= 1
        assert len(result.key_decisions) >= 1
        assert result.execution_time_ms == 5000
    
    def test_summarize_with_failed_chains(self):
        """测试包含失败反思链的总结"""
        s = WorkflowSummarizer()
        
        result = s.summarize(
            task="数据分析",
            result={"success": False},
            reflection_chains=[
                {
                    "success": False,
                    "problem": "文件格式错误",
                    "solution": "转换文件格式",
                },
                {
                    "success": True,
                    "problem": "编码问题",
                    "solution": "使用UTF-8编码",
                },
            ],
        )
        
        positive = result.get_positive_constraints()
        negative = result.get_negative_constraints()
        assert len(positive) >= 1
        assert len(negative) >= 1
    
    def test_get_relevant_constraints(self):
        """测试获取相关约束"""
        s = WorkflowSummarizer()
        
        s.summarize(
            task="生成 PDF 报告",
            result={"success": True},
            reflection_chains=[
                {"success": True, "problem": "缺少PDF库", "solution": "安装 reportlab"},
            ],
        )
        
        constraints = s.get_relevant_constraints("创建 PDF 文件")
        assert len(constraints) >= 1
    
    def test_summaries_property(self):
        """测试总结列表属性"""
        s = WorkflowSummarizer()
        
        s.summarize(task="任务1", result={"success": True}, reflection_chains=[])
        s.summarize(task="任务2", result={"success": False}, reflection_chains=[])
        
        assert len(s.summaries) == 2

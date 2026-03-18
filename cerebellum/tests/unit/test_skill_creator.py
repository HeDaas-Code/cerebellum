"""
测试自主技能创建器
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from cerebellum.orchestrator.skill_creator import SkillCreator


class TestSkillCreator:
    """SkillCreator 测试"""
    
    def test_init_without_llm(self):
        """测试无 LLM 初始化"""
        creator = SkillCreator()
        assert creator.llm is None
        assert creator.skills_dir is None
    
    def test_create_basic_skill(self):
        """测试基础技能创建（无 LLM）"""
        creator = SkillCreator()
        
        result = creator.create_skill(
            name="test-skill",
            task_description="处理测试文件",
        )
        
        assert result is not None
        assert result["name"] == "test-skill"
        assert "处理测试文件" in result["content"]
    
    def test_create_skill_with_llm(self):
        """测试使用 LLM 创建技能"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"name": "data-cleaner", "description": "Clean data files", '
                    '"title": "Data Cleaner", "content": "Clean and transform data", '
                    '"dependencies": ["pandas"], "usage": "Use for data cleaning"}'
        )
        
        creator = SkillCreator(llm=mock_llm)
        
        result = creator.create_skill(
            name="data-cleaner",
            task_description="清洗数据文件",
        )
        
        assert result is not None
        assert result["name"] == "data-cleaner"
        assert "pandas" in result.get("dependencies", [])
    
    def test_create_skill_with_llm_fallback(self):
        """测试 LLM 失败回退到基础创建"""
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        creator = SkillCreator(llm=mock_llm)
        
        result = creator.create_skill(
            name="fallback-skill",
            task_description="回退测试",
        )
        
        assert result is not None
        assert result["name"] == "fallback-skill"
    
    def test_save_skill_to_dir(self, tmp_path):
        """测试保存技能到目录"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        
        creator = SkillCreator(skills_dir=skills_dir)
        
        result = creator.create_skill(
            name="saved-skill",
            task_description="测试保存",
        )
        
        assert result is not None
        assert result["path"] is not None
        
        skill_path = skills_dir / "saved-skill" / "SKILL.md"
        assert skill_path.exists()
        content = skill_path.read_text()
        assert "saved-skill" in content
    
    def test_create_skill_with_subtask(self):
        """测试带子任务描述的技能创建"""
        creator = SkillCreator()
        
        result = creator.create_skill(
            name="data-viz",
            task_description="分析数据并生成图表",
            subtask_description="使用 matplotlib 生成柱状图",
        )
        
        assert result is not None
        assert result["name"] == "data-viz"
        assert "matplotlib" in result["content"] or "分析数据" in result["content"]
    
    def test_create_skill_overwrites_existing(self, tmp_path):
        """测试覆盖已存在的技能"""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        
        creator = SkillCreator(skills_dir=skills_dir)
        
        # 第一次创建
        result1 = creator.create_skill(name="test-skill", task_description="任务v1")
        assert result1 is not None
        
        # 第二次创建（覆盖）
        result2 = creator.create_skill(name="test-skill", task_description="任务v2")
        assert result2 is not None
        
        content = (skills_dir / "test-skill" / "SKILL.md").read_text()
        assert "任务v2" in content
    
    def test_create_skill_learned_from_reflection(self):
        """测试从反思链学习创建技能"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"name": "learned-fix", "description": "Fix encoding issues", '
                    '"title": "Encoding Fix", "content": "Handle UTF-8 encoding properly", '
                    '"dependencies": [], "usage": "Use when encoding errors occur"}'
        )
        
        creator = SkillCreator(llm=mock_llm)
        
        result = creator.create_skill(
            name="learned-fix",
            task_description="处理文件编码",
            subtask_description="从反思链学习的解决方案:\n问题: UnicodeDecodeError\n解决: 使用 UTF-8 编码",
        )
        
        assert result is not None
        assert result["name"] == "learned-fix"

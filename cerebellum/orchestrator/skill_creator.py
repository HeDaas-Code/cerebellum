"""
Cerebellum 自主技能创建模块

在运行时自主创建新的 Skills 并保存到技能目录
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils import logger


class SkillCreator:
    """
    自主技能创建器
    
    当现有技能无法满足任务需求时，自主创建新技能
    """
    
    SKILL_TEMPLATE = """---
name: {name}
description: {description}
---

# {title}

{content}

## 依赖

{dependencies}

## 使用方法

{usage}
"""
    
    def __init__(self, llm=None, skills_dir: Path = None):
        """
        初始化技能创建器
        
        Args:
            llm: LLM 客户端（可选，用于 LLM 增强创建）
            skills_dir: 技能目录路径
        """
        self.llm = llm
        self.skills_dir = skills_dir
        logger.debug("技能创建器初始化完成")
    
    def _call_llm(self, prompt: str) -> str:
        """统一调用 LLM"""
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
    
    def create_skill(
        self,
        name: str,
        task_description: str,
        subtask_description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        自主创建新技能
        
        Args:
            name: 技能名称
            task_description: 任务描述
            subtask_description: 子任务描述
        
        Returns:
            技能定义字典 {"name": ..., "content": ..., "path": ...} 或 None
        """
        logger.info(f"正在自主创建技能: {name}")
        
        if self.llm:
            return self._create_skill_with_llm(name, task_description, subtask_description)
        else:
            return self._create_skill_basic(name, task_description, subtask_description)
    
    def _create_skill_with_llm(
        self,
        name: str,
        task_description: str,
        subtask_description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """使用 LLM 创建技能"""
        prompt = f"""为以下任务创建一个新的技能定义。

技能名称: {name}
任务描述: {task_description}
子任务: {subtask_description}

请生成技能定义，返回 JSON 格式:
{{
    "name": "{name}",
    "description": "技能描述（英文，用于触发匹配）",
    "title": "技能标题",
    "content": "详细的技能使用指导（包括步骤、注意事项等）",
    "dependencies": ["依赖包1", "依赖包2"],
    "usage": "使用方法说明"
}}

只返回 JSON，不要其他内容。"""
        
        try:
            content = self._call_llm(prompt).strip()
            content = content.replace("```json", "").replace("```", "").strip()
            skill_def = json.loads(content)
            
            # 生成 SKILL.md 内容
            skill_md = self.SKILL_TEMPLATE.format(
                name=skill_def.get("name", name),
                description=skill_def.get("description", task_description),
                title=skill_def.get("title", name),
                content=skill_def.get("content", ""),
                dependencies="\n".join(
                    f"- {dep}" for dep in skill_def.get("dependencies", [])
                ) or "无",
                usage=skill_def.get("usage", ""),
            )
            
            # 保存到技能目录
            saved_path = self._save_skill(name, skill_md)
            
            logger.success(f"技能创建成功: {name}")
            return {
                "name": name,
                "content": skill_md,
                "path": str(saved_path) if saved_path else None,
                "dependencies": skill_def.get("dependencies", []),
            }
            
        except Exception as e:
            logger.warning(f"LLM 技能创建失败: {e}")
            return self._create_skill_basic(name, task_description, subtask_description)
    
    def _create_skill_basic(
        self,
        name: str,
        task_description: str,
        subtask_description: str = "",
    ) -> Optional[Dict[str, Any]]:
        """基础技能创建（无 LLM）"""
        skill_md = self.SKILL_TEMPLATE.format(
            name=name,
            description=task_description[:200],
            title=name.replace("-", " ").title(),
            content=f"执行以下任务:\n\n{task_description}\n\n{subtask_description}",
            dependencies="无",
            usage=f"当需要 {task_description[:50]}... 时使用此技能",
        )
        
        saved_path = self._save_skill(name, skill_md)
        
        logger.info(f"基础技能创建完成: {name}")
        return {
            "name": name,
            "content": skill_md,
            "path": str(saved_path) if saved_path else None,
            "dependencies": [],
        }
    
    def _save_skill(self, name: str, content: str) -> Optional[Path]:
        """保存技能到目录"""
        if not self.skills_dir:
            logger.debug("未指定技能目录，跳过保存")
            return None
        
        try:
            skill_dir = self.skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            
            skill_path = skill_dir / "SKILL.md"
            skill_path.write_text(content, encoding="utf-8")
            
            logger.debug(f"技能已保存: {skill_path}")
            return skill_path
        except Exception as e:
            logger.warning(f"保存技能失败: {e}")
            return None

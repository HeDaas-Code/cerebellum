"""
Cerebellum 任务汇报模块

输出任务执行进度
"""

from datetime import datetime
from typing import List, Optional

from ..types import SubTask, SubTaskStatus, TaskResult


class TaskReporter:
    """
    任务汇报器
    
    输出任务执行进度信息
    """
    
    @staticmethod
    def _print_box(title: str, content: str = ""):
        """打印带边框的信息框"""
        lines = content.split('\n') if content else []
        max_width = max(len(title), max((len(line) for line in lines), default=0))
        width = min(max_width, 60)
        
        print("═" * (width + 4))
        print(f"  {title}")
        if content:
            print("─" * (width + 4))
            for line in lines:
                print(f"  {line}")
        print("═" * (width + 4))
        print()
    
    def report_task_start(self, task: str, subtasks: List[SubTask]):
        """汇报任务开始"""
        subtask_list = "\n".join([
            f"{i+1}. {st.name} [优先级: {st.priority}]"
            for i, st in enumerate(subtasks)
        ])
        
        self._print_box(
            "📋 任务规划完成",
            f"任务: {task}\n子任务数量: {len(subtasks)}\n\n┌─ 子任务列表 ─────────┐\n{subtask_list}\n└──────────────────────┘"
        )
    
    def report_skill_usage(self, skill_name: str, action: str):
        """汇报技能使用"""
        self._print_box(
            f"🔧 使用技能: {skill_name}",
            f"操作: {action}"
        )
    
    def report_subtask_complete(self, subtask: SubTask, result: str = ""):
        """汇报子任务完成"""
        status_icon = "✅" if subtask.status == SubTaskStatus.COMPLETED else "❌"
        
        self._print_box(
            f"{status_icon} 子任务{'完成' if subtask.status == SubTaskStatus.COMPLETED else '失败'}: {subtask.name}",
            f"结果: {result[:200]}..." if len(result) > 200 else f"结果: {result}"
        )
    
    def report_task_complete(self, result: TaskResult):
        """汇报任务完成"""
        files_info = "\n".join([
            f"  - {f.name}"
            for f in result.files
        ]) if result.files else "  无文件"
        
        subtask_summary = "\n".join([
            f"{'✓' if st.status == SubTaskStatus.COMPLETED else '✗'} {st.name}"
            for st in result.subtasks
        ])
        
        self._print_box(
            "🎉 任务完成" if result.success else "❌ 任务失败",
            f"总耗时: {result.execution_time_ms / 1000:.2f} 秒\n"
            f"生成文件: {len(result.files)} 个\n{files_info}\n\n"
            f"┌─ 执行摘要 ─────────┐\n{subtask_summary}\n└────────────────────┘"
        )
    
    def report_cache_hit(self, task_hash: str, similarity_score: float, action: str):
        """汇报缓存命中"""
        self._print_box(
            "💾 缓存命中",
            f"任务哈希: {task_hash[:16]}...\n相似度: {similarity_score:.2%}\n操作: {action}"
        )

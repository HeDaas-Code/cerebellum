"""
Cerebellum 反思链执行器

执行完整的反思链
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_openai import ChatOpenAI

from .chain import ReflectionChain, ReflectionHistory
from ..types import ReflectionStatus
from ..tools.sandbox import SandboxManager
from ..utils import logger


class ReflectionChainExecutor:
    """
    反思链执行器
    
    执行完整的反思链：错误 → 联网搜索 → 分析 → 方案 → 任务编排 → 执行 → 结果
    """
    
    def __init__(
        self, 
        llm: ChatOpenAI, 
        sandbox: SandboxManager,
        web_search=None
    ):
        """
        初始化反思链执行器
        
        Args:
            llm: LLM 客户端
            sandbox: 沙盒管理器
            web_search: 联网搜索工具
        """
        self.llm = llm
        self.sandbox = sandbox
        self.web_search = web_search
        logger.debug("反思链执行器初始化完成")
    
    def execute_chain(
        self, 
        error_message: str, 
        context: Dict[str, Any],
        chain_number: int,
        previous_chains: list = None
    ) -> ReflectionChain:
        """
        执行一次完整的反思链
        
        流程:
        1. 识别问题
        2. 联网搜索解决方案
        3. 分析根本原因
        4. 构建解决方案
        5. 任务编排
        6. 执行方案
        7. 评估结果
        
        Args:
            error_message: 错误信息
            context: 错误上下文
            chain_number: 第几轮反思链
            previous_chains: 之前的反思链记录
        
        Returns:
            完整的反思链记录
        """
        chain = ReflectionChain(
            chain_id=str(uuid.uuid4()),
            chain_number=chain_number,
            error_message=error_message,
            error_context=context,
            started_at=datetime.now()
        )
        
        logger.info(f"[反思链 #{chain_number}] 开始执行")
        logger.debug(f"[反思链 #{chain_number}] 错误信息: {error_message[:100]}...")
        
        # 步骤1: 识别问题
        chain.problem_identified = self._identify_problem(error_message, context, previous_chains)
        logger.debug(f"[反思链 #{chain_number}] 问题识别: {chain.problem_identified}")
        
        # 步骤2: 联网搜索解决方案（优先执行）
        chain.search_queries = self._generate_search_queries(chain.problem_identified)
        chain.search_results = []
        
        if self.web_search:
            logger.info(f"[反思链 #{chain_number}] 正在联网搜索解决方案...")
            for query in chain.search_queries:
                try:
                    results = self.web_search.invoke(query)
                    chain.search_results.append({"query": query, "results": results})
                    logger.debug(f"[反思链 #{chain_number}] 搜索 '{query}' 完成")
                except Exception as e:
                    logger.warning(f"[反思链 #{chain_number}] 搜索失败: {e}")
        else:
            logger.warning(f"[反思链 #{chain_number}] 联网搜索工具不可用")
        
        logger.info(f"[反思链 #{chain_number}] 搜索完成，找到 {len(chain.search_results)} 个结果")
        
        # 步骤3: 分析根本原因并构建解决方案
        chain.root_cause, chain.proposed_solution, chain.solution_code = \
            self._build_solution_with_search(chain, context, previous_chains)
        logger.info(f"[反思链 #{chain_number}] 根本原因: {chain.root_cause}")
        logger.info(f"[反思链 #{chain_number}] 解决方案: {chain.proposed_solution}")
        
        # 步骤4: 任务编排（将解决方案拆分为可执行步骤）
        execution_steps = self._orchestrate_solution(chain.proposed_solution, chain.solution_code)
        if execution_steps:
            logger.info(f"[反思链 #{chain_number}] 任务编排: {len(execution_steps)} 个步骤")
        
        # 步骤5: 执行方案
        if execution_steps:
            chain.execution_log = self._execute_steps(execution_steps)
        elif chain.solution_code:
            chain.execution_log = self._execute_solution(chain.solution_code)
        else:
            chain.execution_log = "无需执行代码"
        
        # 步骤6: 评估结果
        chain.result_status, chain.result_message = self._evaluate_result(
            chain.execution_log, error_message, context
        )
        chain.completed_at = datetime.now()
        
        if chain.is_success:
            logger.success(f"[反思链 #{chain_number}] 成功: {chain.result_message}")
        else:
            logger.warning(f"[反思链 #{chain_number}] 失败: {chain.result_message}")
        
        return chain
    
    def _identify_problem(
        self, 
        error_message: str, 
        context: Dict[str, Any],
        previous_chains: list = None
    ) -> str:
        """识别问题"""
        prev_info = ""
        if previous_chains:
            prev_info = f"\n\n之前尝试过的方法:\n"
            for pc in previous_chains:
                prev_info += f"- {pc.get('solution', '未知')} (结果: {pc.get('result', '未知')})\n"
        
        prompt = f"""分析以下错误，识别具体问题:

错误信息: {error_message}
上下文: {json.dumps(context, ensure_ascii=False, indent=2)}
{prev_info}

请用一句话描述具体问题是什么。只返回问题描述，不要其他内容。"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception:
            return "无法识别问题"
    
    def _generate_search_queries(self, problem: str) -> list:
        """生成搜索关键词"""
        prompt = f"""针对以下问题，生成3个搜索关键词，用于查找解决方案:

问题: {problem}

返回 JSON 数组格式: ["关键词1", "关键词2", "关键词3"]"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception:
            return [problem]
    
    def _build_solution_with_search(
        self, 
        chain: ReflectionChain, 
        context: Dict[str, Any],
        previous_chains: list = None
    ) -> tuple:
        """结合搜索结果构建解决方案"""
        prev_info = ""
        if previous_chains:
            prev_info = f"\n\n之前尝试过的方法（都已失败）:\n"
            for pc in previous_chains:
                prev_info += f"- {pc.get('solution', '未知')}\n"
            prev_info += "\n请尝试不同的方法。\n"
        
        search_info = ""
        if chain.search_results:
            search_info = "\n\n联网搜索结果:\n"
            for sr in chain.search_results[:3]:
                if isinstance(sr.get('results'), str):
                    search_info += f"- {sr.get('results', '无结果')[:500]}\n"
                elif isinstance(sr.get('results'), list):
                    for r in sr.get('results', [])[:2]:
                        if isinstance(r, dict):
                            search_info += f"- {r.get('content', r.get('snippet', '无内容'))[:300]}\n"
        
        prompt = f"""基于以下信息，构建解决方案:

问题: {chain.problem_identified}
{search_info}
{prev_info}

请分析:
1. 根本原因是什么?
2. 解决方案是什么?
3. 如果需要执行命令或代码，请提供具体代码。

返回 JSON 格式:
{{"root_cause": "根本原因", "solution": "解决方案描述", "code": "具体代码（如果需要，不需要则为空字符串）"}}"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "")
            result = json.loads(content)
            return (
                result.get("root_cause", ""),
                result.get("solution", ""),
                result.get("code") or None
            )
        except Exception as e:
            return "未知原因", "无法生成解决方案", None
    
    def _orchestrate_solution(self, solution: str, code: str = None) -> List[Dict[str, Any]]:
        """将解决方案编排为可执行步骤"""
        if code:
            return [{"type": "execute_code", "code": code, "description": solution}]
        
        prompt = f"""将以下解决方案拆分为具体的执行步骤:

解决方案: {solution}

返回 JSON 数组格式:
[
  {{"step": 1, "action": "执行动作描述", "command": "具体命令（如果有）"}},
  ...
]"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "")
            steps = json.loads(content)
            return steps
        except Exception:
            return []
    
    def _execute_steps(self, steps: List[Dict[str, Any]]) -> str:
        """执行编排好的步骤"""
        logs = []
        for i, step in enumerate(steps, 1):
            logger.info(f"[反思链] 执行步骤 {i}: {step.get('action', step.get('description', '未知'))}")
            
            if step.get("type") == "execute_code" or step.get("command"):
                code = step.get("code") or step.get("command", "")
                result = self._execute_solution(code)
                logs.append(f"步骤 {i}: {result}")
            else:
                logs.append(f"步骤 {i}: 跳过（无执行内容）")
        
        return "\n".join(logs)
    
    def _execute_solution(self, solution_code: Optional[str]) -> str:
        """执行解决方案代码"""
        if not solution_code:
            return "无需执行代码"
        
        try:
            result = self.sandbox.execute(solution_code, timeout=60)
            return f"Exit Code: {result.exit_code}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        except Exception as e:
            return f"执行失败: {str(e)}"
    
    def _evaluate_result(
        self, 
        execution_log: str, 
        original_error: str,
        context: Dict[str, Any]
    ) -> tuple:
        """评估执行结果"""
        if "Exit Code: 0" in execution_log and "Error" not in execution_log:
            return ReflectionStatus.SUCCESS, "解决方案执行成功"
        elif "执行失败" in execution_log:
            return ReflectionStatus.FAILURE, f"解决方案执行失败: {execution_log}"
        else:
            return ReflectionStatus.FAILURE, f"解决方案可能未完全解决问题"

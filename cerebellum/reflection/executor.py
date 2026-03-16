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
        llm, 
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
    
    def _call_llm(self, prompt: str) -> str:
        """
        统一调用 LLM，兼容不同后端
        
        Args:
            prompt: 提示词
        
        Returns:
            LLM 响应内容
        """
        try:
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
        except Exception as e:
            logger.error(f"[反思链] LLM 调用失败: {e}")
            raise
    
    def execute_chain(
        self, 
        error_message: str, 
        context: Dict[str, Any],
        chain_number: int,
        previous_chains: list = None,
        parent_chain_id: str = None
    ) -> ReflectionChain:
        """
        执行一次完整的反思链（支持递归修复）
        
        流程:
        1. 识别问题
        2. 联网搜索解决方案
        3. 分析根本原因
        4. 构建解决方案
        5. 任务编排
        6. 执行方案（如果执行失败，递归调用反思链修复）
        7. 评估结果
        
        Args:
            error_message: 错误信息
            context: 错误上下文
            chain_number: 第几轮反思链
            previous_chains: 之前的反思链记录
            parent_chain_id: 父反思链ID（用于递归追踪）
        
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
        
        if parent_chain_id:
            logger.info(f"[反思链 #{chain_number}] 【递归修复】开始执行，父反思链: {parent_chain_id}")
        else:
            logger.info(f"[反思链 #{chain_number}] 开始执行")
        logger.debug(f"[反思链 #{chain_number}] 错误信息: {error_message[:200]}...")
        
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
        
        # 步骤5: 执行方案（支持递归修复和两阶段切换）
        if execution_steps:
            chain.execution_log = self._execute_steps_with_recursion(
                execution_steps, chain_number, parent_chain_id, context, phase=1
            )
        elif chain.solution_code:
            chain.execution_log = self._execute_solution_with_recursion(
                chain.solution_code, chain_number, parent_chain_id, context, phase=1
            )
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
    
    def _execute_steps_with_recursion(
        self, 
        steps: List[Dict[str, Any]], 
        parent_chain_number: int,
        parent_chain_id: str = None,
        original_context: Dict[str, Any] = None,
        phase: int = 1
    ) -> str:
        """执行步骤，支持递归修复和两阶段切换"""
        logs = []
        max_recursion = 5  # 最大递归修复次数
        phase_switch_threshold = 3  # 第三层递归后切换到第二阶段
        
        context = original_context or {}
        
        for i, step in enumerate(steps, 1):
            step_desc = step.get('action', step.get('description', '未知'))
            logger.info(f"[反思链] 执行步骤 {i}: {step_desc}")
            
            if step.get("type") == "execute_code" or step.get("command"):
                code = step.get("code") or step.get("command", "")
                
                # 执行代码
                result = self._execute_solution(code)
                logs.append(f"步骤 {i}: {result}")
                
                # 检查是否需要切换到第二阶段
                if parent_chain_number >= phase_switch_threshold and phase == 1:
                    logger.warning(f"[反思链] 递归 {parent_chain_number} 次仍失败，切换到第二阶段...")
                    
                    # 执行第二阶段：放弃当前方法，尝试替代方案
                    phase2_result = self._execute_phase2(
                        error_message=result,
                        context={**context, "step": i, "description": step_desc, "failed_code": code},
                        chain_number=parent_chain_number + 1,
                        parent_chain_id=parent_chain_id
                    )
                    
                    logs.append(f"[第二阶段] {phase2_result}")
                    
                    if "成功" in phase2_result or "success" in phase2_result.lower():
                        return "\n".join(logs)
                    
                    continue
                
                # 检查是否需要递归修复
                if self._needs_recursion(result) and parent_chain_number < max_recursion:
                    logger.warning(f"[反思链] 步骤 {i} 执行失败，启动递归修复...")
                    
                    # 创建新的反思链来修复错误
                    new_chain_number = parent_chain_number + 1
                    fix_chain = self.execute_chain(
                        error_message=result,
                        context={**context, "step": i, "description": step_desc, "failed_code": code},
                        chain_number=new_chain_number,
                        previous_chains=None,
                        parent_chain_id=parent_chain_id or str(uuid.uuid4())
                    )
                    
                    logs.append(f"[递归修复 #{new_chain_number}] {fix_chain.result_message}")
                    
                    # 如果修复成功，尝试用修复后的方法重新执行
                    if fix_chain.is_success and fix_chain.solution_code:
                        logger.info(f"[反思链] 使用修复后的方案重新执行步骤 {i}...")
                        result = self._execute_solution(fix_chain.solution_code)
                        logs.append(f"步骤 {i} (修复后): {result}")
            else:
                logs.append(f"步骤 {i}: 跳过（无执行内容）")
        
        return "\n".join(logs)
    
    def _execute_solution_with_recursion(
        self, 
        solution_code: str, 
        parent_chain_number: int,
        parent_chain_id: str = None,
        original_context: Dict[str, Any] = None,
        phase: int = 1
    ) -> str:
        """执行解决方案代码，支持递归修复和两阶段切换"""
        if not solution_code:
            return "无需执行代码"
        
        max_recursion = 5
        phase_switch_threshold = 3  # 第三层递归后切换到第二阶段
        current_code = solution_code
        current_chain_number = parent_chain_number
        context = original_context or {}
        
        for attempt in range(max_recursion):
            try:
                result = self._execute_solution(current_code)
                logs = f"尝试 {attempt + 1}: Exit Code: {result.exit_code}\n"
                
                if result.exit_code == 0 and "Error" not in result.stderr:
                    return f"{logs}执行成功"
                
                logs += f"Stderr: {result.stderr}"
                
                # 检查是否需要切换到第二阶段
                if current_chain_number >= phase_switch_threshold and phase == 1:
                    logger.warning(f"[反思链] 递归 {current_chain_number} 次仍失败，切换到第二阶段...")
                    
                    # 执行第二阶段：放弃当前方法，尝试替代方案
                    phase2_result = self._execute_phase2(
                        error_message=result.stderr or str(result),
                        context={**context, "attempt": attempt + 1, "failed_code": current_code},
                        chain_number=current_chain_number + 1,
                        parent_chain_id=parent_chain_id
                    )
                    
                    logs += f"\n[第二阶段] {phase2_result}"
                    
                    if "成功" in phase2_result or "success" in phase2_result.lower():
                        return logs
                    
                    continue
                
                # 检查是否需要继续递归修复
                if attempt < max_recursion - 1:
                    logger.warning(f"[反思链] 执行失败 (尝试 {attempt + 1})，启动递归修复...")
                    
                    fix_chain = self.execute_chain(
                        error_message=result.stderr or str(result),
                        context={**context, "attempt": attempt + 1, "failed_code": current_code, "exit_code": result.exit_code},
                        chain_number=current_chain_number + 1,
                        previous_chains=None,
                        parent_chain_id=parent_chain_id or str(uuid.uuid4())
                    )
                    
                    logs += f"\n[递归修复 #{current_chain_number + 1}] {fix_chain.result_message}"
                    
                    if fix_chain.is_success and fix_chain.solution_code:
                        current_code = fix_chain.solution_code
                        current_chain_number += 1
                        continue
                
                return logs
                
            except Exception as e:
                # 检查是否需要切换到第二阶段
                if current_chain_number >= phase_switch_threshold and phase == 1:
                    logger.warning(f"[反思链] 异常 {e}，切换到第二阶段...")
                    
                    phase2_result = self._execute_phase2(
                        error_message=str(e),
                        context={**context, "attempt": attempt + 1, "failed_code": current_code},
                        chain_number=current_chain_number + 1,
                        parent_chain_id=parent_chain_id
                    )
                    
                    return f"执行异常: {str(e)}\n[第二阶段] {phase2_result}"
                
                if attempt < max_recursion - 1:
                    logger.warning(f"[反思链] 执行异常: {e}，尝试修复...")
                    fix_chain = self.execute_chain(
                        error_message=str(e),
                        context={**context, "attempt": attempt + 1, "failed_code": current_code},
                        chain_number=current_chain_number + 1,
                        previous_chains=None,
                        parent_chain_id=parent_chain_id or str(uuid.uuid4())
                    )
                    
                    if fix_chain.is_success and fix_chain.solution_code:
                        current_code = fix_chain.solution_code
                        current_chain_number += 1
                        continue
                
                return f"执行异常: {str(e)}"
        
        return f"达到最大递归次数 ({max_recursion})，执行终止"
    
    def _needs_recursion(self, execution_result: str) -> bool:
        """判断是否需要递归修复"""
        error_indicators = [
            "Exit Code: -1",
            "Error",
            "Exception",
            "Traceback",
            "失败",
            "无法",
            "not found",
            "No such file",
            "Permission denied"
        ]
        result_lower = execution_result.lower()
        return any(indicator.lower() in result_lower for indicator in error_indicators)
    
    def _execute_phase2(
        self,
        error_message: str,
        context: Dict[str, Any],
        chain_number: int,
        parent_chain_id: str = None
    ) -> str:
        """
        第二阶段：放弃当前方法，尝试替代方案
        
        当第一阶段递归修复失败3次后，触发第二阶段
        第二阶段会：
        1. 分析当前方法失败的根本原因
        2. 识别可能的替代技术或策略
        3. 生成完全不同的新解决方案路径
        """
        logger.info("=" * 60)
        logger.info(f"[第二阶段 #{chain_number}] 放弃当前方案，尝试替代方法")
        logger.info("=" * 60)
        
        task = context.get("task", "未知任务")
        failed_code = context.get("failed_code", "")
        step_desc = context.get("description", "")
        
        # 构建第二阶段分析提示
        prompt = f"""你是一个高级问题解决专家。现在面临一个困难问题，需要你分析失败原因并找到全新的解决方案。

## 当前任务
{task}

## 失败详情
- 失败步骤: {step_desc}
- 失败的代码/命令: {failed_code}
- 错误信息: {error_message}

## 之前尝试过的方法（都失败了）
请分析为什么这些方法都失败了，找出根本问题。

## 第二阶段要求
请按照以下格式提供替代方案：

1. **失败原因分析**: 为什么之前的方法都失败了？
2. **替代方案**: 完全不同的新方法（必须与之前的方法有本质区别）
3. **新代码**: 新的解决方案代码

返回 JSON 格式:
{{
    "root_cause": "失败的根本原因",
    "alternative_approach": "完全不同的替代方法描述",
    "new_code": "新的解决方案代码（Python）",
    "explanation": "为什么这个新方法可以绕过之前的问题"
}}"""
        
        try:
            content = self._call_llm(prompt).strip()
            content = content.replace("```json", "").replace("```", "")
            result = json.loads(content)
            
            root_cause = result.get("root_cause", "未知")
            alternative = result.get("alternative_approach", "无")
            new_code = result.get("new_code", "")
            explanation = result.get("explanation", "")
            
            logger.info(f"[第二阶段 #{chain_number}] 失败原因: {root_cause}")
            logger.info(f"[第二阶段 #{chain_number}] 替代方案: {alternative}")
            logger.info(f"[第二阶段 #{chain_number}] 新代码: {new_code[:200] if new_code else '无'}...")
            
            # 执行新代码
            if new_code:
                logger.info(f"[第二阶段 #{chain_number}] 执行新方案...")
                exec_result = self._execute_solution(new_code)
                
                if exec_result.exit_code == 0:
                    logger.success(f"[第二阶段 #{chain_number}] 新方案执行成功")
                    return f"成功 - 新方案执行成功\n原因: {root_cause}\n替代方案: {alternative}"
                else:
                    logger.warning(f"[第二阶段 #{chain_number}] 新方案执行失败: {exec_result.stderr}")
                    return f"失败 - 新方案执行失败\n原因: {root_cause}\n错误: {exec_result.stderr}"
            else:
                return f"未生成新代码 - 原因: {root_cause}, 替代方案: {alternative}"
                
        except Exception as e:
            logger.error(f"[第二阶段 #{chain_number}] 分析失败: {e}")
            return f"第二阶段执行异常: {str(e)}"
    
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
            return self._call_llm(prompt).strip()
        except Exception:
            return "无法识别问题"
    
    def _generate_search_queries(self, problem: str) -> list:
        """生成搜索关键词"""
        prompt = f"""针对以下问题，生成3个搜索关键词，用于查找解决方案:

问题: {problem}

返回 JSON 数组格式: ["关键词1", "关键词2", "关键词3"]"""
        
        try:
            content = self._call_llm(prompt).strip()
            content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception:
            return [problem]
    
    def _build_solution(
        self,
        chain: ReflectionChain,
        context: Dict[str, Any],
        previous_chains: list = None
    ) -> tuple:
        """
        构建解决方案（_build_solution_with_search 的别名）
        
        Args:
            chain: 反思链
            context: 上下文
            previous_chains: 之前的反思链记录
        
        Returns:
            (root_cause, solution, code)
        """
        return self._build_solution_with_search(chain, context, previous_chains)
    
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
            content = self._call_llm(prompt).strip()
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
            content = self._call_llm(prompt).strip()
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

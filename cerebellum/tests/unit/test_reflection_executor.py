"""
reflection/executor.py 单元测试

测试反思链执行器
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from concurrent.futures import TimeoutError as FuturesTimeoutError

from cerebellum.reflection.executor import ReflectionChainExecutor, LLM_CALL_TIMEOUT
from cerebellum.types import ReflectionStatus, ExecuteResult
from cerebellum.tools.sandbox import SandboxManager


class TestReflectionChainExecutor:
    """ReflectionChainExecutor 测试"""
    
    @pytest.fixture
    def mock_llm(self):
        """模拟 LLM"""
        mock = MagicMock()
        return mock
    
    @pytest.fixture
    def mock_sandbox(self):
        """模拟沙盒"""
        mock = MagicMock(spec=SandboxManager)
        mock.execute = MagicMock(return_value=ExecuteResult(
            exit_code=0,
            stdout="Success",
            stderr="",
            command="test"
        ))
        return mock
    
    @pytest.fixture
    def mock_web_search(self):
        """模拟联网搜索"""
        mock = MagicMock()
        mock.invoke = MagicMock(return_value="搜索结果内容")
        return mock
    
    @pytest.fixture
    def executor(self, mock_llm, mock_sandbox, mock_web_search):
        """反思链执行器实例"""
        return ReflectionChainExecutor(
            llm=mock_llm,
            sandbox=mock_sandbox,
            web_search=mock_web_search
        )
    
    def test_execute_chain_success(self, executor, mock_llm, mock_sandbox):
        """测试反思链执行成功"""
        mock_llm.invoke.side_effect = [
            MagicMock(content="缺少依赖模块"),
            MagicMock(content='["pip install requests"]'),
            MagicMock(content='{"root_cause": "未安装依赖", "solution": "安装 requests", "code": "pip install requests"}')
        ]
        
        chain = executor.execute_chain(
            error_message="ModuleNotFoundError: No module named 'requests'",
            context={"command": "python script.py"},
            chain_number=1
        )
        
        assert chain.chain_number == 1
        assert chain.problem_identified == "缺少依赖模块"
        assert chain.root_cause == "未安装依赖"
        assert chain.proposed_solution == "安装 requests"
    
    def test_execute_chain_with_web_search(self, executor, mock_llm, mock_web_search):
        """测试带联网搜索的反思链"""
        mock_llm.invoke.side_effect = [
            MagicMock(content="配置错误"),
            MagicMock(content='["python config fix"]'),
            MagicMock(content='{"root_cause": "配置问题", "solution": "修复配置", "code": ""}')
        ]
        
        chain = executor.execute_chain(
            error_message="ConfigurationError",
            context={},
            chain_number=1
        )
        
        assert len(chain.search_queries) > 0
        mock_web_search.invoke.assert_called()
    
    def test_execute_chain_with_previous_chains(self, executor, mock_llm):
        """测试带历史反思链的执行"""
        previous_chains = [
            {"solution": "方法1", "result": "failure"},
            {"solution": "方法2", "result": "failure"}
        ]
        
        mock_llm.invoke.side_effect = [
            MagicMock(content="新问题分析"),
            MagicMock(content='["新搜索词"]'),
            MagicMock(content='{"root_cause": "新原因", "solution": "新方法", "code": ""}')
        ]
        
        chain = executor.execute_chain(
            error_message="Persistent error",
            context={},
            chain_number=3,
            previous_chains=previous_chains
        )
        
        assert chain.chain_number == 3
    
    def test_identify_problem_success(self, executor, mock_llm):
        """测试问题识别成功"""
        mock_llm.invoke.return_value = MagicMock(content="缺少依赖包")
        
        result = executor._identify_problem("ModuleNotFoundError", {})
        
        assert result == "缺少依赖包"
    
    def test_identify_problem_exception(self, executor, mock_llm):
        """测试问题识别异常"""
        mock_llm.invoke.side_effect = Exception("LLM error")
        
        result = executor._identify_problem("Error", {})
        
        assert result == "无法识别问题"
    
    def test_generate_search_queries_success(self, executor, mock_llm):
        """测试生成搜索关键词成功"""
        mock_llm.invoke.return_value = MagicMock(
            content='["python install", "pip install", "module not found"]'
        )
        
        result = executor._generate_search_queries("缺少模块")
        
        assert len(result) == 3
        assert "python install" in result
    
    def test_generate_search_queries_parse_error(self, executor, mock_llm):
        """测试搜索关键词解析错误"""
        mock_llm.invoke.return_value = MagicMock(content="invalid json")
        
        result = executor._generate_search_queries("问题")
        
        assert result == ["问题"]
    
    def test_build_solution_success(self, executor, mock_llm):
        """测试构建解决方案成功"""
        from cerebellum.types import ReflectionChain
        
        chain = ReflectionChain(
            chain_id="test",
            chain_number=1,
            error_message="Error",
            problem_identified="缺少模块"
        )
        
        mock_llm.invoke.return_value = MagicMock(
            content='{"root_cause": "依赖未安装", "solution": "运行 pip install", "code": "pip install requests"}'
        )
        
        root_cause, solution, code = executor._build_solution(chain, {})
        
        assert root_cause == "依赖未安装"
        assert solution == "运行 pip install"
        assert code == "pip install requests"
    
    def test_build_solution_no_code(self, executor, mock_llm):
        """测试构建无代码的解决方案"""
        from cerebellum.types import ReflectionChain
        
        chain = ReflectionChain(
            chain_id="test",
            chain_number=1,
            error_message="Error",
            problem_identified="配置问题"
        )
        
        mock_llm.invoke.return_value = MagicMock(
            content='{"root_cause": "配置错误", "solution": "修改配置文件", "code": ""}'
        )
        
        root_cause, solution, code = executor._build_solution(chain, {})
        
        assert code is None
    
    def test_execute_solution_success(self, executor, mock_sandbox):
        """测试执行解决方案成功"""
        result = executor._execute_solution("pip install requests")
        
        mock_sandbox.execute.assert_called_once()
        assert "Exit Code: 0" in result
    
    def test_execute_solution_no_code(self, executor):
        """测试无代码执行"""
        result = executor._execute_solution(None)
        
        assert result == "无需执行代码"
    
    def test_execute_solution_exception(self, executor, mock_sandbox):
        """测试执行解决方案异常"""
        mock_sandbox.execute.side_effect = Exception("Sandbox error")
        
        result = executor._execute_solution("some code")
        
        assert "执行失败" in result
    
    def test_evaluate_result_success(self, executor):
        """测试评估成功结果"""
        log = "Exit Code: 0\nStdout: Success\nStderr: "
        
        status, message = executor._evaluate_result(log, "Error", {})
        
        assert status == ReflectionStatus.SUCCESS
    
    def test_evaluate_result_failure(self, executor):
        """测试评估失败结果"""
        log = "Exit Code: 1\nStdout: \nStderr: Error occurred"
        
        status, message = executor._evaluate_result(log, "Error", {})
        
        assert status == ReflectionStatus.FAILURE
    
    def test_evaluate_result_execution_failed(self, executor):
        """测试执行失败结果"""
        log = "执行失败: Sandbox error"
        
        status, message = executor._evaluate_result(log, "Error", {})
        
        assert status == ReflectionStatus.FAILURE
        assert "执行失败" in message
    
    def test_chain_timing(self, executor, mock_llm):
        """测试反思链时间记录"""
        mock_llm.invoke.side_effect = [
            MagicMock(content="问题"),
            MagicMock(content='["搜索"]'),
            MagicMock(content='{"root_cause": "原因", "solution": "方案", "code": ""}')
        ]
        
        chain = executor.execute_chain("Error", {}, 1)
        
        assert chain.started_at is not None
        assert chain.completed_at is not None
        assert chain.duration_seconds >= 0
    
    def test_executor_without_web_search(self, mock_llm, mock_sandbox):
        """测试无联网搜索的执行器"""
        executor = ReflectionChainExecutor(
            llm=mock_llm,
            sandbox=mock_sandbox,
            web_search=None
        )
        
        mock_llm.invoke.side_effect = [
            MagicMock(content="问题"),
            MagicMock(content='["搜索"]'),
            MagicMock(content='{"root_cause": "原因", "solution": "方案", "code": ""}')
        ]
        
        chain = executor.execute_chain("Error", {}, 1)
        
        assert chain.search_results == []

    def test_call_llm_timeout(self, executor, mock_llm):
        """测试 LLM 调用超时保护"""
        import time
        def slow_invoke(prompt):
            time.sleep(10)
            return MagicMock(content="response")
        mock_llm.invoke.side_effect = slow_invoke
        
        with pytest.raises(TimeoutError, match="LLM 调用超时"):
            executor._call_llm("test prompt", timeout=1)

    def test_call_llm_respects_timeout_param(self, executor, mock_llm):
        """测试 LLM 调用超时参数传递"""
        mock_llm.invoke.return_value = MagicMock(content="快速响应")
        
        result = executor._call_llm("test prompt", timeout=5)
        assert result == "快速响应"

    def test_call_llm_default_timeout(self, executor, mock_llm):
        """测试 LLM 调用默认超时值存在"""
        assert LLM_CALL_TIMEOUT == 60


class TestFatalConnectionErrorDetection:
    """致命连接错误检测测试"""
    
    def test_remote_disconnected_is_fatal(self):
        """测试 RemoteDisconnected 被识别为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        # 模拟 RemoteDisconnected 错误
        class RemoteDisconnected(Exception):
            pass
        
        error = RemoteDisconnected("Remote end closed connection without response")
        assert _is_fatal_connection_error(error) is True
    
    def test_connection_reset_is_fatal(self):
        """测试 ConnectionResetError 被识别为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        error = ConnectionResetError("Connection reset by peer")
        assert _is_fatal_connection_error(error) is True
    
    def test_connection_refused_is_fatal(self):
        """测试 ConnectionRefusedError 被识别为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        error = ConnectionRefusedError("Connection refused")
        assert _is_fatal_connection_error(error) is True
    
    def test_broken_pipe_is_fatal(self):
        """测试 BrokenPipeError 被识别为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        error = BrokenPipeError("Broken pipe")
        assert _is_fatal_connection_error(error) is True
    
    def test_normal_error_is_not_fatal(self):
        """测试普通错误不被误判为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        error = ValueError("some value error")
        assert _is_fatal_connection_error(error) is False
    
    def test_not_found_error_is_not_fatal(self):
        """测试 NotFoundError(404) 不被误判为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        error = Exception("Error code: 404")
        assert _is_fatal_connection_error(error) is False
    
    def test_nested_remote_disconnected_is_fatal(self):
        """测试嵌套的 RemoteDisconnected 也被识别为致命错误"""
        from cerebellum import _is_fatal_connection_error
        
        class RemoteDisconnected(Exception):
            pass
        
        inner = RemoteDisconnected("Remote end closed connection")
        outer = Exception("Retrying after connection broken")
        outer.__cause__ = inner
        
        assert _is_fatal_connection_error(outer) is True
    
    def test_error_message_with_remote_disconnected_is_fatal(self):
        """测试错误消息包含 RemoteDisconnected 的情况"""
        from cerebellum import _is_fatal_connection_error
        
        error = Exception("RemoteDisconnected('Remote end closed connection without response')")
        assert _is_fatal_connection_error(error) is True

    def test_max_reflection_recursion_depth_constant(self):
        """测试最大递归深度常量存在"""
        from cerebellum import MAX_REFLECTION_RECURSION_DEPTH
        assert MAX_REFLECTION_RECURSION_DEPTH >= 1
        assert MAX_REFLECTION_RECURSION_DEPTH <= 10

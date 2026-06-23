import sys
import time
import types

import pytest


STUB_OPTIONAL_MODULES = [
    "daytona",
    "daytona_sdk",
    "daytona_sdk.common",
    "daytona_sdk.common.daytona",
    "langchain_daytona",
    "sentence_transformers",
]


def _stub_deepagents_and_llm(agent_instance):
    if "deepagents" not in sys.modules:
        da = types.ModuleType("deepagents")
        da.create_deep_agent = lambda *args, **kwargs: agent_instance
        sys.modules["deepagents"] = da
    else:
        sys.modules["deepagents"].create_deep_agent = lambda *args, **kwargs: agent_instance
    if "deepagents.backends" not in sys.modules:
        sys.modules["deepagents.backends"] = types.ModuleType("deepagents.backends")
    if "deepagents.backends.utils" not in sys.modules:
        utils = types.ModuleType("deepagents.backends.utils")
        utils.create_file_data = lambda *args, **kwargs: None
        sys.modules["deepagents.backends.utils"] = utils
    if "langchain_openai" not in sys.modules:
        lc = types.ModuleType("langchain_openai")

        class _DummyChatOpenAI:
            def __init__(self, *args, **kwargs):
                ...

        lc.ChatOpenAI = _DummyChatOpenAI
        sys.modules["langchain_openai"] = lc
    if "langchain_community" not in sys.modules:
        comm = types.ModuleType("langchain_community")
        comm.tools = types.ModuleType("langchain_community.tools")

        class _DummyTavily:
            def __init__(self, *args, **kwargs):
                ...

        comm.tools.TavilySearchResults = _DummyTavily
        sys.modules["langchain_community"] = comm
        sys.modules["langchain_community.tools"] = comm.tools
    for name in STUB_OPTIONAL_MODULES:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


class _HangingAgent:
    def stream(self, *args, **kwargs):
        # 模拟工具调用长时间无响应
        while True:
            time.sleep(5)


def test_agent_stream_timeout_triggers():
    hanging_agent = _HangingAgent()
    _stub_deepagents_and_llm(hanging_agent)

    from cerebellum import Cerebellum
    from cerebellum.config import CerebellumConfig, SandboxConfig

    config = CerebellumConfig(sandbox=SandboxConfig(timeout_seconds=1))
    cb = Cerebellum(config=config)
    cb.agent = hanging_agent

    with pytest.raises(TimeoutError, match="智能体执行超时"):
        for _ in cb._stream_agent_events([{"role": "user", "content": "ping"}], timeout=1):
            pass

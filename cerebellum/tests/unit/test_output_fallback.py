import sys
import types

import pytest


def _stub_deepagents_and_llm():
    if "deepagents" not in sys.modules:
        da = types.ModuleType("deepagents")
        da.create_deep_agent = lambda *args, **kwargs: None
        sys.modules["deepagents"] = da
    if "deepagents.backends" not in sys.modules:
        sys.modules["deepagents.backends"] = types.ModuleType("deepagents.backends")
    if "deepagents.backends.utils" not in sys.modules:
        utils = types.ModuleType("deepagents.backends.utils")
        utils.create_file_data = lambda *args, **kwargs: None
        sys.modules["deepagents.backends.utils"] = utils
    if "langchain_openai" not in sys.modules:
        lc = types.ModuleType("langchain_openai")
        class _DummyChatOpenAI:
            def __init__(self, *args, **kwargs): ...
        lc.ChatOpenAI = _DummyChatOpenAI
        sys.modules["langchain_openai"] = lc
    if "langchain_community" not in sys.modules:
        comm = types.ModuleType("langchain_community")
        comm.tools = types.ModuleType("langchain_community.tools")
        class _DummyTavily:
            def __init__(self, *args, **kwargs): ...
        comm.tools.TavilySearchResults = _DummyTavily
        sys.modules["langchain_community"] = comm
        sys.modules["langchain_community.tools"] = comm.tools
    for name in ["daytona", "daytona_sdk", "daytona_sdk.common", "daytona_sdk.common.daytona", "langchain_daytona", "sentence_transformers"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def test_apply_output_fallback_creates_file_when_missing():
    _stub_deepagents_and_llm()
    from cerebellum import Cerebellum
    from cerebellum.config import CerebellumConfig

    cb = Cerebellum(CerebellumConfig())
    result = {"success": True, "message": "final summary"}

    files = cb._apply_output_fallback(result, [])

    assert len(files) == 1
    assert files[0].name == "output.md"
    assert "final summary" in files[0].content
    assert files[0].type == "text"


def test_apply_output_fallback_noop_when_files_exist():
    _stub_deepagents_and_llm()
    from cerebellum import Cerebellum, FileData
    from cerebellum.config import CerebellumConfig

    existing = [FileData(name="report.md", content=b"ok", type="text")]
    cb = Cerebellum(CerebellumConfig())
    result = {"success": True, "message": "ignored"}

    files = cb._apply_output_fallback(result, existing)

    assert files is existing
    assert len(files) == 1
    assert files[0].name == "report.md"

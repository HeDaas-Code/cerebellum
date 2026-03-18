from unittest.mock import MagicMock

from cerebellum import Cerebellum
from cerebellum.config import CerebellumConfig
from cerebellum.utils import logger


SANDBOX_LOG_MSG = "正在创建沙盒环境..."


def test_initialize_logs_sandbox_creation_once(monkeypatch, tmp_path):
    """Ensure sandbox creation log is emitted only once during initialization."""
    config = CerebellumConfig(
        api_key="test-key",
        base_url="https://example.com",
        model="test-model",
        database_path=tmp_path / "cache.db",
        debug=True,
    )
    cb = Cerebellum(config=config)

    # Avoid actual external calls
    monkeypatch.setattr(cb, "_create_llm", lambda: MagicMock(name="llm"))

    def _mock_create_sandbox():
        logger.info(SANDBOX_LOG_MSG)
        return MagicMock(name="backend"), MagicMock(name="sandbox")

    monkeypatch.setattr(cb, "_create_sandbox", _mock_create_sandbox)
    monkeypatch.setattr(cb, "_create_agent", lambda: None)
    monkeypatch.setattr(cb, "_init_cache_system", lambda: None)
    monkeypatch.setattr(cb, "_init_orchestrator", lambda: None)
    monkeypatch.setattr(cb, "_load_skills_files", lambda *args, **kwargs: {})

    captured = []
    sink_id = logger.add(lambda message: captured.append(str(message).strip()))
    try:
        cb.initialize()
    finally:
        logger.remove(sink_id)

    sandbox_logs = [msg for msg in captured if SANDBOX_LOG_MSG in msg]
    assert len(sandbox_logs) == 1

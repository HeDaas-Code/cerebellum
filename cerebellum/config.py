"""
Cerebellum 配置模块

定义所有配置类
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from enum import Enum


class LLMBackend(Enum):
    """LLM 后端类型"""
    DASHSCOPE = "dashscope"
    MINIMAX = "minimax"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class CacheConfig:
    """缓存配置"""
    similarity_threshold: float = 0.85
    max_cache_size: int = 1000
    cache_ttl_days: int = 30
    enable_auto_cleanup: bool = True
    cleanup_interval_hours: int = 24


@dataclass
class ReflectionConfig:
    """反思配置"""
    max_reflections: int = 10
    enable_web_search: bool = True
    search_max_results: int = 5


@dataclass
class SandboxConfig:
    """沙盒配置"""
    timeout_seconds: int = 300
    max_retries: int = 3
    auto_cleanup: bool = True
    image: Optional[str] = None
    resources: dict = field(default_factory=lambda: {
        "cpu": 2,
        "memory": 4
    })
    env_vars: dict = field(default_factory=dict)
    workdir: str = "/home/daytona/workspace"
    pre_install: list = field(default_factory=list)


@dataclass
class CerebellumConfig:
    """Cerebellum 主配置类"""
    llm_backend: LLMBackend = LLMBackend.DASHSCOPE
    api_key: str = ""
    base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    model: str = "glm-5"
    daytona_api_key: str = ""
    tavily_api_key: str = ""
    skills_dir: Optional[Path] = None
    debug: bool = False
    
    cache: CacheConfig = field(default_factory=CacheConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    
    database_path: Optional[Path] = None
    
    # 各后端的默认配置 (base_url, model)
    BACKEND_DEFAULTS = {
        LLMBackend.DASHSCOPE: ("https://coding.dashscope.aliyuncs.com/v1", "glm-5"),
        LLMBackend.MINIMAX: ("https://api.minimaxi.com/anthropic", "MiniMax-M2.5"),
        LLMBackend.ANTHROPIC: ("https://api.anthropic.com", "claude-sonnet-4-20250514"),
        LLMBackend.OPENAI: ("https://api.openai.com/v1", "gpt-4o"),
    }
    
    def __post_init__(self):
        if self.database_path is None:
            self.database_path = Path.home() / ".cerebellum" / "cache.db"
        
        self.apply_backend_defaults()
    
    def apply_backend_defaults(self):
        """根据当前 llm_backend 应用默认的 base_url 和 model
        
        仅当 base_url/model 仍为其他后端的默认值时才覆盖，
        避免覆盖用户显式设置的自定义值。
        """
        # 收集所有后端的默认 base_url 和 model
        all_default_urls = {v[0] for v in self.BACKEND_DEFAULTS.values()}
        all_default_models = {v[1] for v in self.BACKEND_DEFAULTS.values()}
        
        defaults = self.BACKEND_DEFAULTS.get(self.llm_backend)
        if defaults:
            # 仅当 base_url 还是某个后端的默认值（或空）时才更新
            if not self.base_url or self.base_url in all_default_urls:
                self.base_url = defaults[0]
            # 仅当 model 还是某个后端的默认值（或空）时才更新
            if not self.model or self.model in all_default_models:
                self.model = defaults[1]

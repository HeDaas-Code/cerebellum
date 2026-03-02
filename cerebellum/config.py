"""
Cerebellum 配置模块

定义所有配置类
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


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


@dataclass
class CerebellumConfig:
    """Cerebellum 主配置类"""
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://coding.dashscope.aliyuncs.com/v1"
    dashscope_model: str = "glm-5"
    daytona_api_key: str = ""
    tavily_api_key: str = ""
    skills_dir: Optional[Path] = None
    debug: bool = False
    
    cache: CacheConfig = field(default_factory=CacheConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    
    database_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.database_path is None:
            self.database_path = Path.home() / ".cerebellum" / "cache.db"

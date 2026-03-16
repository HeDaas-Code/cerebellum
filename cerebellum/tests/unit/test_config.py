"""
config.py 单元测试

测试配置类
"""

import pytest
from pathlib import Path

from cerebellum.config import (
    CerebellumConfig, CacheConfig, ReflectionConfig, SandboxConfig, LLMBackend
)


class TestCacheConfig:
    """CacheConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = CacheConfig()
        
        assert config.similarity_threshold == 0.85
        assert config.max_cache_size == 1000
        assert config.cache_ttl_days == 30
        assert config.enable_auto_cleanup is True
        assert config.cleanup_interval_hours == 24
    
    def test_custom_values(self):
        """测试自定义值"""
        config = CacheConfig(
            similarity_threshold=0.9,
            max_cache_size=500,
            cache_ttl_days=7
        )
        
        assert config.similarity_threshold == 0.9
        assert config.max_cache_size == 500
        assert config.cache_ttl_days == 7


class TestReflectionConfig:
    """ReflectionConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = ReflectionConfig()
        
        assert config.max_reflections == 10
        assert config.enable_web_search is True
        assert config.search_max_results == 5
    
    def test_disabled_web_search(self):
        """测试禁用网络搜索"""
        config = ReflectionConfig(enable_web_search=False)
        
        assert config.enable_web_search is False


class TestSandboxConfig:
    """SandboxConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = SandboxConfig()
        
        assert config.timeout_seconds == 300
        assert config.max_retries == 3
        assert config.auto_cleanup is True
    
    def test_custom_timeout(self):
        """测试自定义超时"""
        config = SandboxConfig(timeout_seconds=600)
        
        assert config.timeout_seconds == 600


class TestCerebellumConfig:
    """CerebellumConfig 测试"""
    
    def test_default_values(self):
        """测试默认值"""
        config = CerebellumConfig()
        
        assert config.api_key == ""
        assert config.model == "glm-5"
        assert config.debug is False
        assert config.skills_dir is None
    
    def test_custom_values(self):
        """测试自定义值"""
        config = CerebellumConfig(
            api_key="test-key",
            model="custom-model",
            debug=True
        )
        
        assert config.api_key == "test-key"
        assert config.model == "custom-model"
        assert config.debug is True
    
    def test_default_database_path(self):
        """测试默认数据库路径"""
        config = CerebellumConfig()
        
        assert config.database_path is not None
        assert ".cerebellum" in str(config.database_path)
        assert config.database_path.name == "cache.db"
    
    def test_custom_database_path(self, temp_db_path):
        """测试自定义数据库路径"""
        config = CerebellumConfig(database_path=temp_db_path)
        
        assert config.database_path == temp_db_path
    
    def test_nested_config_defaults(self):
        """测试嵌套配置默认值"""
        config = CerebellumConfig()
        
        assert isinstance(config.cache, CacheConfig)
        assert isinstance(config.reflection, ReflectionConfig)
        assert isinstance(config.sandbox, SandboxConfig)
    
    def test_nested_config_custom(self):
        """测试嵌套配置自定义"""
        config = CerebellumConfig(
            cache=CacheConfig(similarity_threshold=0.95),
            reflection=ReflectionConfig(max_reflections=5)
        )
        
        assert config.cache.similarity_threshold == 0.95
        assert config.reflection.max_reflections == 5
    
    def test_skills_dir_path(self, temp_dir):
        """测试技能目录路径"""
        skills_path = temp_dir / "skills"
        config = CerebellumConfig(skills_dir=skills_path)
        
        assert config.skills_dir == skills_path
    
    def test_api_keys(self):
        """测试 API 密钥配置"""
        config = CerebellumConfig(
            api_key="dash-key",
            daytona_api_key="daytona-key",
            tavily_api_key="tavily-key"
        )
        
        assert config.api_key == "dash-key"
        assert config.daytona_api_key == "daytona-key"
        assert config.tavily_api_key == "tavily-key"
    
    def test_base_url_default(self):
        """测试默认 Base URL"""
        config = CerebellumConfig()
        
        assert "dashscope" in config.base_url
    
    def test_minimax_backend(self):
        """测试 MiniMax 后端配置"""
        config = CerebellumConfig(llm_backend=LLMBackend.MINIMAX)
        
        assert "minimaxi" in config.base_url
        assert config.model == "MiniMax-M2.5"
    
    def test_anthropic_backend(self):
        """测试 Anthropic 后端配置"""
        config = CerebellumConfig(llm_backend=LLMBackend.ANTHROPIC)
        
        assert "anthropic" in config.base_url
    
    def test_apply_backend_defaults_switches_model(self):
        """测试 apply_backend_defaults 在后端切换后更新 model 和 base_url"""
        config = CerebellumConfig()  # 默认 DASHSCOPE: model=glm-5
        assert config.model == "glm-5"
        
        # 模拟 env var 切换后端
        config.llm_backend = LLMBackend.MINIMAX
        config.apply_backend_defaults()
        
        assert config.model == "MiniMax-M2.5"
        assert "minimaxi" in config.base_url
    
    def test_apply_backend_defaults_preserves_custom_model(self):
        """测试 apply_backend_defaults 不覆盖用户自定义 model"""
        config = CerebellumConfig(
            llm_backend=LLMBackend.MINIMAX,
            model="MiniMax-M2.1",
        )
        # model="MiniMax-M2.1" 不在默认值列表中，不应被覆盖
        assert config.model == "MiniMax-M2.1"
    
    def test_apply_backend_defaults_preserves_custom_base_url(self):
        """测试 apply_backend_defaults 不覆盖用户自定义 base_url"""
        config = CerebellumConfig(
            llm_backend=LLMBackend.MINIMAX,
            base_url="https://custom.api.example.com/v1",
        )
        assert config.base_url == "https://custom.api.example.com/v1"
    
    def test_backend_defaults_class_variable(self):
        """测试 BACKEND_DEFAULTS 类变量包含所有后端"""
        defaults = CerebellumConfig.BACKEND_DEFAULTS
        assert LLMBackend.DASHSCOPE in defaults
        assert LLMBackend.MINIMAX in defaults
        assert LLMBackend.ANTHROPIC in defaults
        assert LLMBackend.OPENAI in defaults

"""
Cerebellum 沙盒管理模块

Daytona 沙盒抽象层
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..types import ExecuteResult
from ..config import SandboxConfig

try:
    from daytona import Daytona
    from daytona_sdk.common.daytona import DaytonaConfig
except ImportError:
    Daytona = None
    DaytonaConfig = None


@dataclass
class FileInfo:
    """文件信息"""
    name: str
    path: str
    is_dir: bool
    size: int = 0


class SandboxManager:
    """
    沙盒管理器
    
    统一的沙盒操作接口，确保：
    1. 沙盒的一次性使用
    2. 资源的正确清理
    3. 操作的可追踪性
    4. 支持自定义沙盒环境配置
    """
    
    def __init__(self, api_key: str, config: Optional[SandboxConfig] = None):
        """
        初始化沙盒管理器
        
        Args:
            api_key: Daytona API Key
            config: 沙盒配置
        """
        self.api_key = api_key
        self.config = config or SandboxConfig()
        self._sandbox = None
        self._daytona = None
        self._logger = logging.getLogger(__name__)
    
    def create(self) -> str:
        """
        创建新沙盒
        
        Returns:
            沙盒 ID
        """
        if Daytona is None:
            raise RuntimeError("Daytona SDK 未安装")
        
        try:
            daytona_config = DaytonaConfig(
                api_key=self.api_key if self.api_key else None
            )
            self._daytona = Daytona(config=daytona_config)
            
            self._logger.info("正在创建沙盒...")
            
            # 构建沙盒创建参数
            create_kwargs = {}
            
            # 自定义镜像
            if self.config.image:
                self._logger.info(f"使用自定义镜像: {self.config.image}")
                create_kwargs["image"] = self.config.image
            
            # 资源配置
            if self.config.resources:
                self._logger.debug(f"资源配置: {self.config.resources}")
                # Daytona SDK 可能支持资源参数，具体取决于 SDK 版本
            
            self._sandbox = self._daytona.create(**create_kwargs)
            
            self._logger.info("等待沙盒启动...")
            self._sandbox.wait_for_sandbox_start(timeout=60)
            
            self._logger.info(f"沙盒创建成功: {self._sandbox.id}")
            
            # 设置工作目录
            workdir = self.config.workdir
            self.execute(f"mkdir -p {workdir} && chmod 755 {workdir}")
            
            # 设置环境变量
            if self.config.env_vars:
                self._logger.info(f"设置环境变量: {list(self.config.env_vars.keys())}")
                for key, value in self.config.env_vars.items():
                    self.execute(f'export {key}="{value}" && echo "export {key}=***" >> ~/.bashrc')
            
            # 预安装依赖
            if self.config.pre_install:
                self._logger.info(f"预安装依赖: {self.config.pre_install}")
                for package in self.config.pre_install:
                    if isinstance(package, str):
                        self.install([package])
                    elif isinstance(package, dict):
                        # 支持更复杂的安装配置
                        pkg_name = package.get("name", "")
                        pkg_type = package.get("type", "pip")
                        if pkg_type == "pip":
                            self.install([pkg_name])
                        elif pkg_type == "apt":
                            self.execute(f"apt-get update && apt-get install -y {pkg_name}", timeout=120)
            
            return self._sandbox.id
            
        except Exception as e:
            self._logger.error(f"创建沙盒失败: {e}")
            raise
    
    def execute(self, command: str, timeout: int = 30) -> ExecuteResult:
        """
        执行命令
        
        Args:
            command: Shell 命令
            timeout: 超时时间
        
        Returns:
            执行结果
        """
        if self._sandbox is None:
            raise RuntimeError("沙盒未创建")
        
        try:
            result = self._sandbox._process.exec(command, timeout=timeout)
            
            return ExecuteResult(
                exit_code=result.exit_code if hasattr(result, 'exit_code') else 0,
                stdout=result.stdout if hasattr(result, 'stdout') else str(result),
                stderr=result.stderr if hasattr(result, 'stderr') else "",
                command=command,
                timeout=False
            )
        except Exception as e:
            return ExecuteResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                command=command,
                timeout=True
            )
    
    def execute_python(self, code: str, timeout: int = 60) -> ExecuteResult:
        """
        执行 Python 代码
        
        Args:
            code: Python 代码
            timeout: 超时时间
        
        Returns:
            执行结果
        """
        escaped_code = code.replace("'", "'\"'\"'")
        command = f"python3 -c '{escaped_code}'"
        return self.execute(command, timeout)
    
    def install(self, packages: List[str]) -> ExecuteResult:
        """
        安装依赖包
        
        Args:
            packages: 包名列表
        
        Returns:
            安装结果
        """
        packages_str = " ".join(packages)
        command = f"pip install {packages_str}"
        return self.execute(command, timeout=120)
    
    def upload(self, file_data: bytes, path: str) -> bool:
        """
        上传文件到沙盒
        
        Args:
            file_data: 文件数据
            path: 目标路径
        
        Returns:
            是否成功
        """
        if self._sandbox is None:
            raise RuntimeError("沙盒未创建")
        
        try:
            import base64
            b64_data = base64.b64encode(file_data).decode('ascii')
            
            dir_path = "/".join(path.split("/")[:-1])
            if dir_path:
                self.execute(f"mkdir -p {dir_path}")
            
            self.execute(f"echo '{b64_data}' | base64 -d > {path}")
            return True
        except Exception as e:
            self._logger.error(f"上传文件失败: {e}")
            return False
    
    def download(self, path: str) -> Optional[bytes]:
        """
        从沙盒下载文件
        
        Args:
            path: 文件路径
        
        Returns:
            文件数据或 None
        """
        if self._sandbox is None:
            raise RuntimeError("沙盒未创建")
        
        try:
            import base64
            
            result = self.execute(f"base64 -w0 {path}")
            if result.exit_code == 0:
                return base64.b64decode(result.stdout.strip())
            return None
        except Exception as e:
            self._logger.error(f"下载文件失败: {e}")
            return None
    
    def list_files(self, directory: str = "/home/daytona") -> List[FileInfo]:
        """
        列出文件
        
        Args:
            directory: 目录路径
        
        Returns:
            文件信息列表
        """
        result = self.execute(f"find {directory} -type f 2>/dev/null | head -100")
        
        files = []
        if result.exit_code == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    path = line.strip()
                    name = path.split('/')[-1]
                    files.append(FileInfo(
                        name=name,
                        path=path,
                        is_dir=False
                    ))
        
        return files
    
    def destroy(self):
        """
        销毁沙盒（任务完成后必须调用）
        """
        if self._sandbox is None:
            return
        
        try:
            self._logger.info("正在销毁沙盒...")
            self._sandbox.stop()
            self._sandbox.delete()
            self._logger.info("沙盒已销毁")
            self._sandbox = None
            self._daytona = None
        except Exception as e:
            self._logger.warning(f"销毁沙盒时出错: {e}")
    
    def __enter__(self):
        self.create()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

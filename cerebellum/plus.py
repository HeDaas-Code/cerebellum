"""
Cerebellum 扩展模块 - 支持文件上传和处理

提供增强的 Cerebellum 类：
- 文件上传
- 文件解析
- AI 处理上传的文件
"""

from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from dataclasses import dataclass

from .upload import FileUploader, UploadResult, UploadedFile
from .parser import ParsedContent
from .security import SecurityCheckResult


class CerebellumPlus:
    """
    支持文件上传和处理的 Cerebellum 代理
    
    提供完整的文件处理流程：
    1. 上传文件 → 安全检查 → 解析内容
    2. 将解析内容整合到任务提示
    3. 让 AI 处理
    4. 返回处理结果和生成的文件
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        debug: bool = False,
        max_file_size: int = 10 * 1024 * 1024,
        check_security: bool = True,
        auto_parse: bool = True
    ):
        """
        初始化 Cerebellum Plus
        
        Args:
            config: CerebellumConfig 配置对象
            debug: 调试模式
            max_file_size: 最大文件大小（字节）
            check_security: 是否进行安全检查
            auto_parse: 是否自动解析文件
        """
        self.config = config
        self.debug = debug
        self.max_file_size = max_file_size
        self.check_security = check_security
        self.auto_parse = auto_parse
        
        # 初始化文件上传器
        self.uploader = FileUploader(
            max_file_size=max_file_size,
            parse=auto_parse,
            check_security=check_security
        )
        
        # 代理实例（延迟加载）
        self._agent = None
    
    def _get_agent(self):
        """获取或创建 Cerebellum 代理"""
        if self._agent is None:
            from . import Cerebellum
            self._agent = Cerebellum(config=self.config, debug=self.debug)
            self._agent.initialize()
        return self._agent
    
    def upload_file(
        self,
        file_data: Union[bytes, str],
        filename: str
    ) -> UploadResult:
        """
        上传文件
        
        Args:
            file_data: 文件内容
            filename: 文件名
        
        Returns:
            UploadResult: 上传结果
        """
        return self.uploader.upload(file_data, filename)
    
    def upload_files(
        self,
        files: List[tuple]
    ) -> List[UploadResult]:
        """
        批量上传文件
        
        Args:
            files: 文件列表 [(file_data, filename), ...]
        
        Returns:
            List[UploadResult]: 上传结果列表
        """
        return self.uploader.upload_multiple(files)
    
    def process_file(
        self,
        file_data: Union[bytes, str],
        filename: str,
        task: str
    ) -> Dict[str, Any]:
        """
        上传文件并让 AI 处理
        
        Args:
            file_data: 文件内容
            filename: 文件名
            task: 处理任务描述
        
        Returns:
            处理结果
        """
        # 1. 上传文件
        upload_result = self.upload_file(file_data, filename)
        if not upload_result.success:
            return {
                "success": False,
                "error": upload_result.error,
                "warnings": upload_result.warnings
            }
        
        file = upload_result.file
        
        # 2. 构建任务提示
        task_prompt = self._build_task_prompt(file, task)
        
        # 3. 执行任务
        agent = self._get_agent()
        result = agent.run(task_prompt)
        
        # 4. 添加上传文件信息
        result["uploaded_file"] = {
            "filename": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "parsed_metadata": file.parsed.metadata if file.parsed else None
        }
        
        return result
    
    def process_files(
        self,
        files: List[tuple],
        task: str
    ) -> Dict[str, Any]:
        """
        上传多个文件并让 AI 处理
        
        Args:
            files: 文件列表 [(file_data, filename), ...]
            task: 处理任务描述
        
        Returns:
            处理结果
        """
        # 1. 上传文件
        upload_results = self.upload_files(files)
        
        # 检查是否有失败
        failed = [r for r in upload_results if not r.success]
        if failed:
            return {
                "success": False,
                "error": "部分文件上传失败",
                "failed_files": [{"error": r.error} for r in failed]
            }
        
        # 2. 构建任务提示
        task_prompt = task + "\n\n"
        task_prompt += "## 上传的文件\n\n"
        
        uploaded_files = []
        for i, result in enumerate(upload_results):
            file = result.file
            task_prompt += f"### 文件 {i+1}: {file.filename}\n"
            task_prompt += f"- 大小: {file.size} bytes\n"
            task_prompt += f"- 类型: {file.content_type}\n\n"
            
            if file.parsed and file.parsed.is_valid:
                task_prompt += f"内容:\n{file.parsed.text}\n\n"
            else:
                task_prompt += "[文件内容无法自动解析]\n\n"
            
            uploaded_files.append({
                "filename": file.filename,
                "size": file.size,
                "content_type": file.content_type
            })
        
        # 3. 执行任务
        agent = self._get_agent()
        result = agent.run(task_prompt)
        
        # 4. 添加上传文件信息
        result["uploaded_files"] = uploaded_files
        
        return result
    
    def _build_task_prompt(self, file: UploadedFile, task: str) -> str:
        """构建任务提示"""
        prompt = f"{task}\n\n"
        prompt += f"## 上传的文件\n\n"
        prompt += f"文件名: {file.filename}\n"
        prompt += f"文件大小: {file.size} bytes\n"
        prompt += f"文件类型: {file.content_type}\n\n"
        
        if file.parsed and file.parsed.is_valid:
            prompt += f"文件内容:\n{file.parsed.text}"
        else:
            prompt += "[文件内容无法自动解析，请根据文件类型自行处理]"
        
        return prompt
    
    def close(self):
        """关闭代理和清理资源"""
        if self._agent is not None:
            self._agent._cleanup_sandbox()
            self._agent = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def process_file(
    file_data: Union[bytes, str],
    filename: str,
    task: str,
    config: Optional[Any] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    便捷函数：上传文件并让 AI 处理
    
    Args:
        file_data: 文件内容
        filename: 文件名
        task: 处理任务
        config: 配置对象
        debug: 调试模式
    
    Returns:
        处理结果
    """
    with CerebellumPlus(config=config, debug=debug) as agent:
        return agent.process_file(file_data, filename, task)


def process_files(
    files: List[tuple],
    task: str,
    config: Optional[Any] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    便捷函数：上传多个文件并让 AI 处理
    
    Args:
        files: 文件列表 [(file_data, filename), ...]
        task: 处理任务
        config: 配置对象
        debug: 调试模式
    
    Returns:
        处理结果
    """
    with CerebellumPlus(config=config, debug=debug) as agent:
        return agent.process_files(files, task)

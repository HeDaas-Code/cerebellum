"""
Cerebellum 文件上传模块

提供文件上传和处理功能：
- 文件上传接口
- 文件安全检查
- 文件解析
"""

import io
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .security import FileSecurityChecker, SecurityCheckResult
from .parser import FileParserManager, ParsedContent


@dataclass
class UploadedFile:
    """上传的文件"""
    filename: str
    content: bytes
    size: int
    content_type: str
    parsed: Optional[ParsedContent] = None
    security_result: Optional[SecurityCheckResult] = None
    
    @property
    def is_safe(self) -> bool:
        return self.security_result is None or self.security_result.is_safe
    
    @property
    def text(self) -> str:
        """获取解析后的文本内容"""
        if self.parsed:
            return self.parsed.text
        if isinstance(self.content, bytes):
            try:
                return self.content.decode('utf-8')
            except:
                return ""
        return str(self.content)


@dataclass
class UploadResult:
    """上传结果"""
    success: bool
    file: Optional[UploadedFile] = None
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class FileUploader:
    """文件上传处理器"""
    
    def __init__(
        self,
        max_file_size: int = 10 * 1024 * 1024,
        parse: bool = True,
        check_security: bool = True
    ):
        """
        初始化上传处理器
        
        Args:
            max_file_size: 最大文件大小（字节）
            parse: 是否自动解析文件
            check_security: 是否进行安全检查
        """
        self.max_file_size = max_file_size
        self.parse = parse
        self.check_security = check_security
        self.security_checker = FileSecurityChecker(max_file_size=max_file_size)
        self.parser = FileParserManager()
    
    def upload(
        self,
        file_data: Union[bytes, str, io.BytesIO],
        filename: str
    ) -> UploadResult:
        """
        上传文件
        
        Args:
            file_data: 文件内容（bytes、str 或 BytesIO）
            filename: 文件名
        
        Returns:
            UploadResult: 上传结果
        """
        # 1. 转换文件数据为 bytes
        if isinstance(file_data, str):
            content = file_data.encode('utf-8')
        elif isinstance(file_data, io.BytesIO):
            content = file_data.getvalue()
        elif isinstance(file_data, bytes):
            content = file_data
        else:
            return UploadResult(
                success=False,
                error=f"不支持的文件数据类型: {type(file_data)}"
            )
        
        # 2. 安全检查
        if self.check_security:
            security_result = self.security_checker.check(content, filename)
            if not security_result.is_safe:
                return UploadResult(
                    success=False,
                    error=security_result.error,
                    warnings=security_result.warnings
                )
        else:
            security_result = None
        
        # 3. 解析文件
        parsed = None
        if self.parse:
            try:
                parsed = self.parser.parse(content, filename)
            except Exception as e:
                return UploadResult(
                    success=False,
                    error=f"文件解析失败: {str(e)}"
                )
        
        # 4. 创建上传文件对象
        uploaded_file = UploadedFile(
            filename=filename,
            content=content,
            size=len(content),
            content_type=self._get_content_type(filename),
            parsed=parsed,
            security_result=security_result
        )
        
        return UploadResult(
            success=True,
            file=uploaded_file,
            warnings=security_result.warnings if security_result else []
        )
    
    def upload_multiple(
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
        results = []
        for file_data, filename in files:
            result = self.upload(file_data, filename)
            results.append(result)
        return results
    
    def _get_content_type(self, filename: str) -> str:
        """获取文件 MIME 类型"""
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"


def upload_file(file_data: Union[bytes, str], filename: str) -> UploadResult:
    """
    便捷函数：上传并解析文件
    
    Args:
        file_data: 文件内容
        filename: 文件名
    
    Returns:
        UploadResult: 上传结果
    """
    uploader = FileUploader()
    return uploader.upload(file_data, filename)

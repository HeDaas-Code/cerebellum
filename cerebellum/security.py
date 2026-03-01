"""
Cerebellum 安全检查模块

提供文件上传安全检查功能：
- 文件大小限制
- 格式验证
- 恶意文件检测
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass


# 允许的文件扩展名
ALLOWED_EXTENSIONS = {
    # 文档
    'txt', 'md', 'rtf', 'pdf', 'doc', 'docx', 'odt',
    # 表格
    'csv', 'xlsx', 'xls', 'ods',
    # 图片（仅用于分析）
    'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg',
    # 代码
    'py', 'js', 'ts', 'java', 'c', 'cpp', 'h', 'go', 'rs', 'rb', 'php', 'swift', 'kt', 'sql',
    # 数据格式
    'json', 'xml', 'yaml', 'yml', 'toml', 'ini', 'cfg', 'env',
    # 其他
    'html', 'css', 'scss', 'sql', 'sh', 'bat', 'ps1', 'log'
}

# 危险文件扩展名（禁止上传）
DANGEROUS_EXTENSIONS = {
    'exe', 'dll', 'so', 'dylib', 'bin',
    'sh', 'bash', 'zsh', 'bat', 'cmd', 'ps1', 'vbs', 'js', 'vba', 'macro',
    'jar', 'class', 'pyc', 'pyo', 'pyd',
    'zip', 'rar', '7z', 'tar', 'gz', 'bz2',
    'htm', 'html', 'shtml', 'xhtml',
    'asp', 'aspx', 'jsp', 'jspx',
    'php', 'phtml', 'php3', 'php4', 'php5',
    'cgi', 'pl', 'perl',
    'lua', 'tcl', 'rb',
    'msi', 'msp', 'cab',
    'scr', 'pif', 'com',
    'url', 'lnk', 'inf',
    'reg', 'ini', 'cfg', 'conf'
}

# 允许的 MIME 类型
ALLOWED_MIME_TYPES = {
    'text/plain',
    'text/markdown',
    'text/rtf',
    'text/csv',
    'text/html',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/json',
    'application/xml',
    'application/x-yaml',
    'image/png',
    'image/jpeg',
    'image/gif',
    'image/bmp',
    'image/webp',
    'image/svg+xml'
}

# 文件大小限制（10MB）
MAX_FILE_SIZE = 10 * 1024 * 1024


@dataclass
class SecurityCheckResult:
    """安全检查结果"""
    is_safe: bool
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class FileSecurityChecker:
    """文件安全检查器"""
    
    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        allowed_extensions: Optional[set] = None,
        allowed_mime_types: Optional[set] = None
    ):
        """
        初始化安全检查器
        
        Args:
            max_file_size: 最大文件大小（字节）
            allowed_extensions: 允许的文件扩展名集合
            allowed_mime_types: 允许的 MIME 类型集合
        """
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or ALLOWED_EXTENSIONS
        self.allowed_mime_types = allowed_mime_types or ALLOWED_MIME_TYPES
    
    def check(self, file_data: bytes, filename: str) -> SecurityCheckResult:
        """
        检查文件安全性
        
        Args:
            file_data: 文件内容（字节）
            filename: 文件名
        
        Returns:
            SecurityCheckResult: 检查结果
        """
        warnings = []
        
        # 1. 检查文件大小
        if len(file_data) > self.max_file_size:
            return SecurityCheckResult(
                is_safe=False,
                error=f"文件大小超过限制 ({self.max_file_size // (1024*1024)}MB)"
            )
        
        # 2. 检查文件扩展名
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext in DANGEROUS_EXTENSIONS:
            return SecurityCheckResult(
                is_safe=False,
                error=f"不允许上传危险文件类型: {ext}"
            )
        
        if ext not in self.allowed_extensions:
            return SecurityCheckResult(
                is_safe=False,
                error=f"不支持的文件类型: {ext}"
            )
        
        # 3. 检查 MIME 类型
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type and mime_type not in self.allowed_mime_types:
            warnings.append(f"非常规 MIME 类型: {mime_type}")
        
        # 4. 检查文件内容（基本检查）
        content_check = self._check_content(file_data, ext)
        if not content_check[0]:
            return SecurityCheckResult(is_safe=False, error=content_check[1])
        
        if content_check[1]:
            warnings.extend(content_check[1])
        
        # 5. 检查文件名
        filename_check = self._check_filename(filename)
        if not filename_check[0]:
            return SecurityCheckResult(is_safe=False, error=filename_check[1])
        
        if filename_check[1]:
            warnings.extend(filename_check[1])
        
        return SecurityCheckResult(is_safe=True, warnings=warnings)
    
    def _check_content(self, file_data: bytes, ext: str) -> Tuple[bool, any]:
        """检查文件内容"""
        warnings = []
        
        # 检查是否为空文件
        if len(file_data) == 0:
            return False, "文件为空"
        
        # 检查是否是有效的 UTF-8（文本文件）
        if ext in {'txt', 'md', 'csv', 'json', 'yaml', 'yml', 'xml', 'py', 'js', 'ts', 'html', 'css'}:
            try:
                file_data.decode('utf-8')
            except UnicodeDecodeError:
                return False, f"{ext} 文件编码无效"
        
        # 检查文件头部魔数
        if ext == 'pdf':
            if not file_data.startswith(b'%PDF'):
                return False, "无效的 PDF 文件"
        
        if ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
            # 简单的图片文件头检查
            valid_starts = {
                'png': b'\x89PNG',
                'jpg': b'\xff\xd8\xff',
                'jpeg': b'\xff\xd8\xff',
                'gif': b'GIF8',
                'bmp': b'BM',
                'webp': b'RIFF'
            }
            if ext == 'jpg':
                ext = 'jpeg'
            if ext in valid_starts:
                if not file_data.startswith(valid_starts[ext]):
                    warnings.append("图片文件头可能不完整")
        
        return True, warnings if warnings else None
    
    def _check_filename(self, filename: str) -> Tuple[bool, any]:
        """检查文件名"""
        warnings = []
        
        # 检查文件名长度
        if len(filename) > 255:
            return False, "文件名过长"
        
        # 检查是否有危险字符
        dangerous_chars = ['..', '~', '$', '|', '&', ';', '`', '<', '>']
        for char in dangerous_chars:
            if char in filename:
                return False, f"文件名包含危险字符: {char}"
        
        # 检查是否为隐藏文件
        if filename.startswith('.'):
            warnings.append("隐藏文件")
        
        return True, warnings if warnings else None


def check_file(file_data: bytes, filename: str) -> SecurityCheckResult:
    """
    便捷函数：检查文件安全性
    
    Args:
        file_data: 文件内容
        filename: 文件名
    
    Returns:
        SecurityCheckResult: 检查结果
    """
    checker = FileSecurityChecker()
    return checker.check(file_data, filename)

"""
Cerebellum 文件解析模块

提供文件解析功能：
- 支持多种文件格式
- 提取文本内容
- 返回结构化数据
"""

import io
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


@dataclass
class ParsedContent:
    """解析后的内容"""
    text: str
    metadata: Dict[str, Any]
    error: Optional[str] = None
    
    @property
    def is_valid(self) -> bool:
        """是否解析成功"""
        return self.error is None


class FileParser(ABC):
    """文件解析器基类"""
    
    @abstractmethod
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        """解析文件"""
        pass
    
    @abstractmethod
    def supports(self, ext: str) -> bool:
        """是否支持该扩展名"""
        pass


class TextParser(FileParser):
    """纯文本解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'txt', 'text', 'log', 'md', 'markdown', 'rst'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        try:
            # 尝试 UTF-8 解码
            text = file_data.decode('utf-8')
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ['gbk', 'gb2312', 'big5', 'latin-1']:
                try:
                    text = file_data.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return ParsedContent(
                    text="",
                    metadata={},
                    error="无法解码文件，请检查编码"
                )
        
        return ParsedContent(
            text=text,
            metadata={
                "encoding": "utf-8",
                "lines": len(text.splitlines()),
                "chars": len(text)
            }
        )


class CSVParser(FileParser):
    """CSV 解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'csv', 'tsv', 'dat'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        try:
            text = file_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = file_data.decode('gbk')
            except UnicodeDecodeError:
                return ParsedContent(
                    text="",
                    metadata={},
                    error="CSV 文件编码错误"
                )
        
        try:
            lines = text.splitlines()
            reader = csv.reader(io.StringIO('\n'.join(lines)))
            rows = list(reader)
            
            # 转换为 Markdown 表格
            if rows:
                md = []
                headers = rows[0]
                md.append("| " + " | ".join(headers) + " |")
                md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows[1:]:
                    md.append("| " + " | ".join(row) + " |")
                
                table_text = "\n".join(md)
                text = f"# CSV 数据 ({len(rows)} 行, {len(headers)} 列)\n\n{table_text}"
            
            return ParsedContent(
                text=text,
                metadata={
                    "rows": len(rows),
                    "columns": len(rows[0]) if rows else 0
                }
            )
        except Exception as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"CSV 解析错误: {str(e)}"
            )


class JSONParser(FileParser):
    """JSON 解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext == 'json'
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        try:
            text = file_data.decode('utf-8')
            data = json.loads(text)
            
            # 格式化 JSON
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            
            return ParsedContent(
                text=formatted,
                metadata={
                    "type": type(data).__name__,
                    "keys": list(data.keys()) if isinstance(data, dict) else None,
                    "length": len(data) if isinstance(data, (list, dict)) else 0
                }
            )
        except json.JSONDecodeError as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"JSON 解析错误: {str(e)}"
            )


class XMLParser(FileParser):
    """XML 解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'xml', 'svg'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        try:
            text = file_data.decode('utf-8')
            root = ET.fromstring(text)
            
            # 转换为文本表示
            lines = []
            lines.append(f"# 根元素: {root.tag}")
            
            def parse_element(elem, indent=0):
                prefix = "  " * indent
                attrs = " ".join([f'{k}="{v}"' for k, v in elem.attrib.items()])
                attr_str = f" ({attrs})" if attrs else ""
                lines.append(f"{prefix}{elem.tag}{attr_str}")
                for child in elem:
                    parse_element(child, indent + 1)
                if elem.text and elem.text.strip():
                    lines.append(f"{prefix}  = {elem.text.strip()}")
            
            parse_element(root)
            
            return ParsedContent(
                text="\n".join(lines),
                metadata={
                    "root": root.tag,
                    "elements": len(list(root.iter()))
                }
            )
        except ET.ParseError as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"XML 解析错误: {str(e)}"
            )


class PDFParser(FileParser):
    """PDF 解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext == 'pdf'
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        if not HAS_PDFPLUMBER:
            return ParsedContent(
                text="",
                metadata={},
                error="PDF 解析需要安装 pdfplumber: pip install pdfplumber"
            )
        
        try:
            with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages.append(f"--- 第 {i+1} 页 ---\n{text}")
                
                full_text = "\n\n".join(pages)
                
                return ParsedContent(
                    text=full_text,
                    metadata={
                        "pages": len(pdf.pages),
                        "extracted_pages": len(pages)
                    }
                )
        except Exception as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"PDF 解析错误: {str(e)}"
            )


class DocxParser(FileParser):
    """Word 文档解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'docx', 'docm'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        if not HAS_DOCX:
            return ParsedContent(
                text="",
                metadata={},
                error="Word 文档解析需要安装 python-docx: pip install python-docx"
            )
        
        try:
            with io.BytesIO(file_data) as f:
                doc = Document(f)
                
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text)
                
                tables_text = []
                for table in doc.tables:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data.append(row_data)
                    
                    if table_data:
                        # 转换为 Markdown 表格
                        md = []
                        headers = table_data[0]
                        md.append("| " + " | ".join(headers) + " |")
                        md.append("| " + " | ".join(["---"] * len(headers)) + " |")
                        for row in table_data[1:]:
                            md.append("| " + " | ".join(row) + " |")
                        tables_text.append("\n".join(md))
                
                text = "\n\n".join(paragraphs)
                if tables_text:
                    text += "\n\n## 表格\n\n" + "\n\n".join(tables_text)
                
                return ParsedContent(
                    text=text,
                    metadata={
                        "paragraphs": len(paragraphs),
                        "tables": len(doc.tables)
                    }
                )
        except Exception as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"Word 文档解析错误: {str(e)}"
            )


class YAMLParser(FileParser):
    """YAML 解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'yaml', 'yml'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        try:
            import yaml
            
            text = file_data.decode('utf-8')
            data = yaml.safe_load(text)
            
            # 转换为 JSON 格式
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            
            return ParsedContent(
                text=formatted,
                metadata={
                    "type": type(data).__name__,
                    "keys": list(data.keys()) if isinstance(data, dict) else None
                }
            )
        except ImportError:
            return ParsedContent(
                text=file_data.decode('utf-8', errors='ignore'),
                metadata={"warning": "YAML 解析需要安装 pyyaml"}
            )
        except Exception as e:
            return ParsedContent(
                text="",
                metadata={},
                error=f"YAML 解析错误: {str(e)}"
            )


class ImageParser(FileParser):
    """图片解析器"""
    
    def supports(self, ext: str) -> bool:
        return ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'svg', 'ico'}
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        # 返回 Base64 编码的图片
        import base64
        b64 = base64.b64encode(file_data).decode('ascii')
        
        ext = filename.lower().split('.')[-1]
        mime_types = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp',
            'svg': 'image/svg+xml',
            'ico': 'image/x-icon'
        }
        mime = mime_types.get(ext, 'image/png')
        
        return ParsedContent(
            text=f"![{filename}](data:{mime};base64,{b64})",
            metadata={
                "type": "image",
                "mime": mime,
                "size": len(file_data),
                "base64_length": len(b64)
            }
        )


class FileParserManager:
    """文件解析管理器"""
    
    def __init__(self):
        self.parsers: List[FileParser] = [
            TextParser(),
            CSVParser(),
            JSONParser(),
            XMLParser(),
            PDFParser(),
            DocxParser(),
            YAMLParser(),
            ImageParser(),
        ]
    
    def parse(self, file_data: bytes, filename: str) -> ParsedContent:
        """
        解析文件
        
        Args:
            file_data: 文件内容
            filename: 文件名
        
        Returns:
            ParsedContent: 解析结果
        """
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # 查找支持的解析器
        for parser in self.parsers:
            if parser.supports(ext):
                return parser.parse(file_data, filename)
        
        # 默认当作纯文本处理
        return TextParser().parse(file_data, filename)


def parse_file(file_data: bytes, filename: str) -> ParsedContent:
    """
    便捷函数：解析文件
    
    Args:
        file_data: 文件内容
        filename: 文件名
    
    Returns:
        ParsedContent: 解析结果
    """
    manager = FileParserManager()
    return manager.parse(file_data, filename)

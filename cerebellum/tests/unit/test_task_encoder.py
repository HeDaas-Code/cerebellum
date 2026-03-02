"""
task_encoder.py 单元测试

测试任务编码器
"""

import pytest

from cerebellum.data.task_encoder import TaskEncoder


class TestTaskEncoder:
    """TaskEncoder 测试"""
    
    def test_normalize_task_basic(self):
        """测试基本任务标准化"""
        result = TaskEncoder.normalize_task("分析这份文档")
        
        assert result == "分析这份文档"
    
    def test_normalize_task_whitespace(self):
        """测试空白字符处理"""
        result = TaskEncoder.normalize_task("  分析   文档  ")
        
        assert result == "分析 文档"
    
    def test_normalize_task_case(self):
        """测试大小写转换"""
        result = TaskEncoder.normalize_task("Analyze This Document")
        
        assert result == "analyze this document"
    
    def test_normalize_task_multiple_spaces(self):
        """测试多空格压缩"""
        result = TaskEncoder.normalize_task("分析    这份    文档")
        
        assert "    " not in result
        assert result == "分析 这份 文档"
    
    def test_extract_intent_summarize(self):
        """测试总结意图提取"""
        result = TaskEncoder.extract_intent("总结这份报告")
        
        assert "summarize" in result
    
    def test_extract_intent_analyze(self):
        """测试分析意图提取"""
        result = TaskEncoder.extract_intent("分析数据趋势")
        
        assert "analyze" in result
    
    def test_extract_intent_multiple(self):
        """测试多意图提取"""
        result = TaskEncoder.extract_intent("分析文档并生成报告")
        
        assert "analyze" in result
        assert "generate" in result
    
    def test_extract_intent_unknown(self):
        """测试未知意图"""
        result = TaskEncoder.extract_intent("做一些事情")
        
        assert result == "unknown"
    
    def test_extract_intent_chart(self):
        """测试图表意图"""
        result = TaskEncoder.extract_intent("创建一个图表")
        
        assert "chart" in result
        assert "create" in result
    
    def test_compute_file_hash_consistency(self):
        """测试文件哈希一致性"""
        data = b"test file content"
        
        hash1 = TaskEncoder.compute_file_hash(data)
        hash2 = TaskEncoder.compute_file_hash(data)
        
        assert hash1 == hash2
    
    def test_compute_file_hash_different(self):
        """测试不同文件哈希不同"""
        hash1 = TaskEncoder.compute_file_hash(b"content 1")
        hash2 = TaskEncoder.compute_file_hash(b"content 2")
        
        assert hash1 != hash2
    
    def test_compute_file_hash_length(self):
        """测试哈希长度"""
        result = TaskEncoder.compute_file_hash(b"test")
        
        assert len(result) == 16
    
    def test_encode_basic(self):
        """测试基本编码"""
        task_hash, normalized, intent, file_hashes = TaskEncoder.encode(
            "分析文档"
        )
        
        assert len(task_hash) == 32
        assert normalized == "分析文档"
        assert "analyze" in intent
        assert file_hashes == []
    
    def test_encode_with_files(self):
        """测试带文件的编码"""
        files = [
            (b"file content 1", "doc1.txt"),
            (b"file content 2", "doc2.txt")
        ]
        
        task_hash, normalized, intent, file_hashes = TaskEncoder.encode(
            "分析文档",
            files=files
        )
        
        assert len(task_hash) == 32
        assert len(file_hashes) == 2
        assert file_hashes[0] != file_hashes[1]
    
    def test_encode_same_task_same_hash(self):
        """测试相同任务相同哈希"""
        result1 = TaskEncoder.encode("分析文档")
        result2 = TaskEncoder.encode("分析文档")
        
        assert result1[0] == result2[0]
    
    def test_encode_different_task_different_hash(self):
        """测试不同任务不同哈希"""
        result1 = TaskEncoder.encode("分析文档")
        result2 = TaskEncoder.encode("总结报告")
        
        assert result1[0] != result2[0]
    
    def test_encode_files_affect_hash(self):
        """测试文件影响哈希"""
        task = "分析文档"
        
        result1 = TaskEncoder.encode(task)
        result2 = TaskEncoder.encode(task, files=[(b"content", "file.txt")])
        
        assert result1[0] != result2[0]
    
    def test_encode_file_order_independence(self):
        """测试文件顺序不影响哈希"""
        files1 = [
            (b"content a", "a.txt"),
            (b"content b", "b.txt")
        ]
        files2 = [
            (b"content b", "b.txt"),
            (b"content a", "a.txt")
        ]
        
        result1 = TaskEncoder.encode("分析", files=files1)
        result2 = TaskEncoder.encode("分析", files=files2)
        
        assert result1[0] == result2[0]
    
    def test_encode_normalized_content(self):
        """测试标准化内容"""
        result1 = TaskEncoder.encode("  分析  文档  ")
        result2 = TaskEncoder.encode("分析 文档")
        
        assert result1[0] == result2[0]
        assert result1[1] == result2[1]

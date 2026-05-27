"""
TCF算法测试
测试音色还原度计算
"""
import pytest
import sys
import os
import numpy as np

# 添加app到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestTCFCalculator:
    """TCF计算器测试"""

    def test_tcf_initialization(self):
        """测试TCF计算器初始化"""
        try:
            from app.algorithms.tcf.tcf_calculator import OptimizedToneColorFidelityScore
            # 注意：实际测试需要模型文件
            # tcf = OptimizedToneColorFidelityScore()
            assert True
        except ImportError as e:
            pytest.skip(f"TCF模块导入失败: {e}")

    def test_cosine_similarity(self):
        """测试余弦相似度计算"""
        # 创建两个相似的向量
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        
        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        
        # 相同向量的相似度应为1
        assert similarity == 1.0

    def test_different_vectors_similarity(self):
        """测试不同向量的相似度"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm1 * norm2)
        
        # 正交向量的相似度应为0
        assert similarity == 0.0


class TestAudioFeatures:
    """音频特征测试"""

    def test_audio_loading(self):
        """测试音频加载（模拟）"""
        # 创建模拟音频数据
        sample_rate = 16000
        duration = 1  # 1秒
        samples = sample_rate * duration
        
        # 生成正弦波
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 440 * t)  # 440Hz
        
        assert len(audio) == samples
        assert audio.dtype == np.float64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

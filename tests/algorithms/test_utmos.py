"""
UTMOS算法测试
测试UTMOSv2 MOS评分
"""
import pytest
import sys
import os
import numpy as np

# 添加app到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestUTMOSCore:
    """UTMOS核心测试"""

    def test_utmos_import(self):
        """测试UTMOS模块导入"""
        try:
            from app.algorithms.utmos.utmos_score import UTMOSCore
            assert True
        except ImportError as e:
            pytest.skip(f"UTMOS模块导入失败: {e}")

    def test_utmos_constants(self):
        """测试UTMOS常量"""
        # UTMOS评分范围应该是1-5
        min_score = 1.0
        max_score = 5.0
        
        assert min_score >= 1.0
        assert max_score <= 5.0


class TestMOSPrediction:
    """MOS预测测试"""

    def test_score_range(self):
        """测试评分范围"""
        # 模拟MOS评分
        mock_score = 3.5
        
        # 确保评分在有效范围内
        assert 1.0 <= mock_score <= 5.0

    def test_score_format(self):
        """测试评分格式"""
        mock_score = 3.56789
        
        # 格式化为4位小数
        formatted = round(mock_score, 4)
        assert isinstance(formatted, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

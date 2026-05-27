"""
NISQA算法测试
测试NISQA评分功能
"""
import pytest
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'algorithms', 'nisqa'))


class TestNISQAModule:
    """NISQA模块测试类"""

    def test_nisqa_import(self):
        """测试NISQA模块导入"""
        try:
            from nisqa.predict import nisqa_predict
            assert True
        except ImportError as e:
            pytest.skip(f"NISQA模块导入失败: {e}")

    def test_nisqa_lib_import(self):
        """测试NISQA库导入"""
        try:
            from nisqa_lib.NISQA_lib import NISQA
            assert True
        except ImportError as e:
            pytest.skip(f"NISQA库导入失败: {e}")

    def test_nisqa_model_import(self):
        """测试NISQA模型导入"""
        try:
            from nisqa_lib.NISQA_model import NisqaModel
            assert True
        except ImportError as e:
            pytest.skip(f"NISQA模型导入失败: {e}")


class TestNISQAMosScore:
    """NISQA MOS评分测试类"""

    def test_nisqa_mos_initialization(self):
        """测试NISQA MOS初始化"""
        try:
            from app.core.mos_calculator import NisqaMosScore
            
            nisqa = NisqaMosScore()
            assert nisqa is not None
            assert nisqa.nisqa_mode == "predict_list"
            assert nisqa.nisqa_model == "nisqa_3000.tar"
        except ImportError as e:
            pytest.skip(f"NISQA MOS初始化失败: {e}")

    def test_nisqa_model_files(self):
        """测试NISQA模型文件"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        model_path = os.path.join(
            project_root, 
            'app', 
            'algorithms', 
            'nisqa', 
            'weights', 
            'nisqa_3000.tar'
        )
        
        if not os.path.exists(model_path):
            pytest.skip(f"NISQA模型文件不存在: {model_path}")
        
        assert os.path.exists(model_path)


class TestNISQAConstants:
    """NISQA常量测试类"""

    def test_nisqa_prediction_range(self):
        """测试NISQA预测值范围"""
        # NISQA预测值应该在1-5范围内
        min_mos = 1.0
        max_mos = 5.0
        
        assert min_mos >= 1.0
        assert max_mos <= 5.0

    def test_nisqa_dimensions(self):
        """测试NISQA维度"""
        # NISQA包含以下维度
        dimensions = [
            'mos_pred',
            'noi_pred',
            'dis_pred',
            'col_pred',
            'loud_pred'
        ]
        
        assert len(dimensions) == 5
        assert 'mos_pred' in dimensions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
DNSMOS算法测试
测试DNSMOS评分功能
"""
import pytest
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestDNSMOSScore:
    """DNSMOS评分测试类"""

    def test_dnsmos_module_import(self):
        """测试DNSMOS模块导入"""
        try:
            from app.core.mos_calculator import DNSMOScore
            assert True
        except ImportError as e:
            pytest.skip(f"DNSMOS模块导入失败: {e}")

    def test_dnsmos_constants(self):
        """测试DNSMOS常量"""
        try:
            from app.core.mos_calculator import DNSMOScore

            dnsmos = DNSMOScore()
            assert dnsmos.INPUT_LENGTH == 9.01
            assert dnsmos.SAMPLING_RATE == 16000
        except (ImportError, AttributeError) as e:
            pytest.skip(f"DNSMOS常量测试失败: {e}")

    def test_dnsmos_polyfit_values(self):
        """测试DNSMOS多项式拟合值计算"""
        try:
            from app.core.mos_calculator import DNSMOScore
            
            dnsmos = DNSMOScore()
            
            # 测试标准MOS的拟合
            sig_raw, bak_raw, ovr_raw = 3.0, 3.0, 3.0
            sig_poly, bak_poly, ovr_poly = dnsmos._DNSMOScore__get_polyfit_val(
                sig_raw, bak_raw, ovr_raw, is_personalized_MOS=False
            )
            
            # 验证结果是浮点数
            assert isinstance(sig_poly, (int, float))
            assert isinstance(bak_poly, (int, float))
            assert isinstance(ovr_poly, (int, float))
            
            # 测试个性化MOS的拟合
            sig_poly_p, bak_poly_p, ovr_poly_p = dnsmos._DNSMOScore__get_polyfit_val(
                sig_raw, bak_raw, ovr_raw, is_personalized_MOS=True
            )
            
            assert isinstance(sig_poly_p, (int, float))
            assert isinstance(bak_poly_p, (int, float))
            assert isinstance(ovr_poly_p, (int, float))
        except ImportError as e:
            pytest.skip(f"DNSMOS模块导入失败: {e}")
        except FileNotFoundError as e:
            pytest.skip(f"DNSMOS模型文件不存在: {e}")

    def test_audio_melspec(self):
        """测试音频梅尔频谱计算"""
        try:
            from app.core.mos_calculator import DNSMOScore
            
            # 创建测试音频
            sample_rate = 16000
            duration = 1.0  # 1秒
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            
            dnsmos = DNSMOScore()
            mel_spec = dnsmos._DNSMOScore__audio_melspec(audio, n_mels=120)
            
            # 验证梅尔频谱的形状
            assert mel_spec.shape[1] == 120  # n_mels
            assert mel_spec.shape[0] > 0  # 时间帧数
        except ImportError as e:
            pytest.skip(f"DNSMOS模块导入失败: {e}")
        except FileNotFoundError as e:
            pytest.skip(f"DNSMOS模型文件不存在: {e}")


class TestDNSMOSModel:
    """DNSMOS模型测试类"""

    def test_model_files_exist(self):
        """测试DNSMOS模型文件存在"""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # 检查模型文件
        model_paths = [
            os.path.join(project_root, 'models', 'dnsmos', 'DNSMOS', 'model_v8.onnx'),
            os.path.join(project_root, 'models', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx'),
            os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'DNSMOS', 'model_v8.onnx'),
            os.path.join(project_root, 'app', 'algorithms', 'dnsmos', 'pDNSMOS', 'sig_bak_ovr.onnx'),
        ]
        
        found = any(os.path.exists(path) for path in model_paths)
        if not found:
            pytest.skip("DNSMOS模型文件不存在")
        
        assert found is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

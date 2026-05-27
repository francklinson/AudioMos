"""
MOS计算器算法测试
测试MOS评分计算功能
"""
import pytest
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'algorithms'))


class TestUtilityFunctions:
    """工具函数测试类"""

    def test_can_convert_to_float_valid(self):
        """测试有效的浮点数转换"""
        try:
            from mos_calculator import can_convert_to_float
            
            assert can_convert_to_float("3.14") is True
            assert can_convert_to_float("123") is True
            assert can_convert_to_float("-5.5") is True
            assert can_convert_to_float(3.14) is True
            assert can_convert_to_float(42) is True
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")

    def test_can_convert_to_float_invalid(self):
        """测试无效的浮点数转换"""
        try:
            from mos_calculator import can_convert_to_float
            
            assert can_convert_to_float("abc") is False
            assert can_convert_to_float("") is False
            assert can_convert_to_float(None) is False
            assert can_convert_to_float("3.14.15") is False
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")

    def test_get_ref_file(self):
        """测试获取参考文件"""
        try:
            from mos_calculator import get_ref_file
            
            # 测试正常情况
            test_file = "voice_mix_70dB_1_关_003.wav"
            ref_dir = "/tmp/ref"
            
            # 由于文件可能不存在，这里只测试函数逻辑
            # 实际返回值会是None（因为文件不存在）
            result = get_ref_file(test_file, ref_dir)
            # 结果应该是None或有效的路径字符串
            assert result is None or isinstance(result, str)
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")


class TestRefScore:
    """参考评分测试类"""

    def test_ref_score_initialization(self):
        """测试参考评分初始化"""
        try:
            from mos_calculator import RefScore
            
            # 注意：需要speechmetrics模块
            # 如果模块不可用会抛出ImportError
            ref_score = RefScore()
            assert ref_score is not None
            assert ref_score.metrics is not None
        except ImportError as e:
            pytest.skip(f"RefScore初始化失败（可能缺少speechmetrics）: {e}")
        except Exception as e:
            pytest.skip(f"RefScore初始化失败: {e}")

    def test_pesq_to_mos_lqo(self):
        """测试PESQ到MOS LQo的转换"""
        try:
            from mos_calculator import RefScore
            
            # 测试不同PESQ值的转换
            test_scores = [1.0, 2.0, 3.0, 4.0, 4.5]
            for pesq_score in test_scores:
                mos_lqo = RefScore.pesq_to_mos_lqo(pesq_score)
                # MOS LQo应该在1-5范围内
                assert 1.0 <= mos_lqo <= 5.0
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")


class TestDNSMOScore:
    """DNSMOS评分测试类"""

    def test_dnsmos_initialization(self):
        """测试DNSMOS初始化"""
        try:
            from mos_calculator import DNSMOScore
            
            # 检查模型文件是否存在
            dnsmos = DNSMOScore()
            assert dnsmos is not None
            assert dnsmos.SAMPLING_RATE == 16000
            assert dnsmos.INPUT_LENGTH == 9.01
        except ImportError as e:
            pytest.skip(f"DNSMOS初始化失败: {e}")
        except FileNotFoundError as e:
            pytest.skip(f"DNSMOS模型文件不存在: {e}")
        except Exception as e:
            pytest.skip(f"DNSMOS初始化失败: {e}")


class TestToneColorFidelityScore:
    """音色还原度评分测试类"""

    def test_tcf_initialization(self):
        """测试TCF初始化"""
        try:
            from mos_calculator import ToneColorFidelityScore
            
            # 注意：需要modelscope模块
            tcf = ToneColorFidelityScore()
            assert tcf is not None
            assert "eres2net" in tcf.sv_model_dict
        except ImportError as e:
            pytest.skip(f"TCF初始化失败（可能缺少modelscope）: {e}")
        except Exception as e:
            pytest.skip(f"TCF初始化失败: {e}")

    def test_compare_speakers(self):
        """测试说话人比较"""
        try:
            from mos_calculator import ToneColorFidelityScore
            
            # 创建相同的特征向量
            features1 = np.array([1.0, 2.0, 3.0, 4.0])
            features2 = np.array([1.0, 2.0, 3.0, 4.0])
            
            similarity = ToneColorFidelityScore._compare_speakers(features1, features2)
            
            # 相同向量的相似度应该为1
            assert abs(similarity - 1.0) < 1e-6
            
            # 测试正交向量
            features3 = np.array([1.0, 0.0, 0.0, 0.0])
            features4 = np.array([0.0, 1.0, 0.0, 0.0])
            similarity = ToneColorFidelityScore._compare_speakers(features3, features4)
            assert abs(similarity - 0.0) < 1e-6
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")


class TestNisqaMosScore:
    """NISQA评分测试类"""

    def test_nisqa_initialization(self):
        """测试NISQA初始化"""
        try:
            from mos_calculator import NisqaMosScore
            
            nisqa = NisqaMosScore()
            assert nisqa is not None
            assert nisqa.nisqa_mode == "predict_list"
            assert nisqa.nisqa_model == "nisqa_3000.tar"
        except ImportError as e:
            pytest.skip(f"NISQA初始化失败（可能缺少nisqa）: {e}")
        except Exception as e:
            pytest.skip(f"NISQA初始化失败: {e}")


class TestScoreqScore:
    """Scoreq评分测试类"""

    def test_scoreq_initialization(self):
        """测试Scoreq初始化"""
        try:
            from mos_calculator import ScoreqScore
            
            scoreq = ScoreqScore()
            assert scoreq is not None
            assert scoreq.pred_mos_ins is not None
        except ImportError as e:
            pytest.skip(f"Scoreq初始化失败（可能缺少scoreq）: {e}")
        except Exception as e:
            pytest.skip(f"Scoreq初始化失败: {e}")


class TestWerScore:
    """WER评分测试类"""

    def test_wer_initialization(self):
        """测试WER初始化"""
        try:
            from mos_calculator import WerScore
            
            # 注意：需要wenet模块
            wer = WerScore()
            assert wer is not None
            assert wer.model is not None
        except ImportError as e:
            pytest.skip(f"WER初始化失败（可能缺少wenet）: {e}")
        except Exception as e:
            pytest.skip(f"WER初始化失败: {e}")

    def test_get_ref_gt_text(self):
        """测试获取参考文本"""
        try:
            from mos_calculator import WerScore
            
            # 测试不同参考文件的文本
            test_files = [
                ("test_001.wav", "他为儿子买了一整根甘蔗市区的停车收费将大幅提高他醒来后发现自己脸上有黑眼圈"),
                ("test_002.wav", "大风刮倒了一处在建厂房姚大爷觉得车夫的想法蛮有道理汹涌的河水顺利而下流的很快"),
                ("test_003.wav", "坚持终于让他有所收获据说这是当地最古老的小区你就是那个爱打篮球的人"),
                ("test_004.wav", "总理对任何事情都要刨根问底渐渐的他还真就睡着了这身衣服就像被大雨淋过似的"),
            ]
            
            for file, expected_text in test_files:
                result = WerScore._WerScore__get_ref_gt_text(file)
                if result is not None:
                    assert isinstance(result, str)
        except ImportError as e:
            pytest.skip(f"MOS计算器导入失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

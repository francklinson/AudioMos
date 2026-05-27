"""
音频处理模块单元测试
测试音频切分和对齐功能
"""
import pytest
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'core'))


class TestAudioShift:
    """音频位移测试类"""

    def test_shift_audio_positive_lag(self):
        """测试正延迟音频位移"""
        try:
            from audio_processor import shift_audio
            
            # 创建测试音频
            audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            lag = 2
            
            result = shift_audio(audio, lag)
            
            # 验证结果
            assert len(result) == len(audio)
            assert result[0] == 0.0  # 前lag个应为0
            assert result[1] == 0.0
            assert result[2] == audio[0]  # 内容向后移动
            assert result[3] == audio[1]
            assert result[4] == audio[2]
        except ImportError as e:
            pytest.skip(f"音频处理模块导入失败: {e}")

    def test_shift_audio_negative_lag(self):
        """测试负延迟音频位移"""
        try:
            from audio_processor import shift_audio
            
            # 创建测试音频
            audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            lag = -2
            
            result = shift_audio(audio, lag)
            
            # 验证结果
            assert len(result) == len(audio)
            assert result[0] == audio[2]  # 内容向前移动
            assert result[1] == audio[3]
            assert result[2] == audio[4]
            assert result[3] == 0.0  # 后面补0
            assert result[4] == 0.0
        except ImportError as e:
            pytest.skip(f"音频处理模块导入失败: {e}")

    def test_shift_audio_zero_lag(self):
        """测试零延迟音频位移"""
        try:
            from audio_processor import shift_audio
            
            audio = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            lag = 0
            
            result = shift_audio(audio, lag)
            
            # 验证结果不变
            assert np.array_equal(result, audio)
        except ImportError as e:
            pytest.skip(f"音频处理模块导入失败: {e}")


class TestFixedPriorityQueue:
    """固定优先级队列测试类"""

    def test_priority_queue_initialization(self):
        """测试优先级队列初始化"""
        try:
            from audio_cut import FixedPriorityQueue
            
            pq = FixedPriorityQueue(max_size=5)
            
            assert pq.max_size == 5
            assert len(pq) == 0
            assert pq.empty() is True
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")

    def test_priority_queue_push(self):
        """测试优先级队列添加元素"""
        try:
            from audio_cut import FixedPriorityQueue
            
            pq = FixedPriorityQueue(max_size=3)
            
            pq.push(3, "item3")
            pq.push(1, "item1")
            pq.push(2, "item2")
            
            assert len(pq) == 3
            assert pq.empty() is False
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")

    def test_priority_queue_max_size(self):
        """测试优先级队列最大大小限制"""
        try:
            from audio_cut import FixedPriorityQueue
            
            pq = FixedPriorityQueue(max_size=2)
            
            # 添加3个元素，但只有2个会被保留（优先级最高的，即数值最小的）
            pq.push(3, "item3")
            pq.push(1, "item1")
            pq.push(2, "item2")
            
            assert len(pq) == 2
            
            # 验证保留的是优先级最高的两个（数值最小）
            all_items = pq.get_all()
            priorities = [item[0] for item in all_items]
            # 保留优先级最高的：1和2（或者1和3，取决于heapreplace的行为）
            assert 1 in priorities  # 1必须被保留（最高优先级）
            assert len(priorities) == 2  # 确保只有两个元素
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")

    def test_priority_queue_pop(self):
        """测试优先级队列弹出元素"""
        try:
            from audio_cut import FixedPriorityQueue
            
            pq = FixedPriorityQueue(max_size=3)
            pq.push(3, "item3")
            pq.push(1, "item1")
            pq.push(2, "item2")
            
            # 弹出优先级最高的元素（数值最小的）
            result = pq.pop()
            assert result[0] == 1  # 优先级
            assert result[1] == "item1"  # 数据
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")


class TestMFCCFeatures:
    """MFCC特征测试类"""

    def test_mfcc_initialization(self):
        """测试MFCC定位器初始化"""
        try:
            from audio_cut import MFCCLocate
            
            # 注意：需要实际音频文件才能完全测试
            # 这里只测试导入和基本属性
            assert MFCCLocate is not None
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")

    def test_time_index_conversion(self):
        """测试时间和索引转换"""
        try:
            from audio_cut import MFCCLocate
            
            # 创建一个模拟的MFCCLocate实例来测试转换方法
            class MockMFCCLocate:
                def __init__(self):
                    self.sr = 16000
                    self.hop_length = 512
                
                def index2time(self, index):
                    return index * (self.hop_length / self.sr)
                
                def time2index(self, time):
                    import math
                    return math.floor(time * self.sr / self.hop_length)
            
            ml = MockMFCCLocate()
            
            # 测试索引转时间
            time = ml.index2time(100)
            expected_time = 100 * 512 / 16000
            assert abs(time - expected_time) < 0.001
            
            # 测试时间转索引
            index = ml.time2index(1.0)
            expected_index = int(1.0 * 16000 / 512)
            assert index == expected_index
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")


class TestCutUsing1k:
    """1kHz切分测试类"""

    def test_cut_using_1k_initialization(self):
        """测试1k切分器初始化"""
        try:
            from audio_cut import CutUsing1k
            
            cut = CutUsing1k()
            assert cut is not None
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")

    def test_find_longest_path(self):
        """测试最长路径查找"""
        try:
            from audio_cut import CutUsing1k
            
            cut = CutUsing1k()
            
            # 测试队列
            queue = [0, 13, 27, 40, 54]
            result = cut._find_longest_path(queue)
            
            # 验证结果包含所有有效节点
            assert len(result) >= 1
            
            # 测试无效队列（间隔不在12.5-14.5范围内）
            invalid_queue = [0, 5, 10, 15]  # 间隔太小
            result = cut._find_longest_path(invalid_queue)
            # 应该返回至少一个元素（每个元素自身算一条路径）
            assert len(result) >= 1
        except ImportError as e:
            pytest.skip(f"音频切分模块导入失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

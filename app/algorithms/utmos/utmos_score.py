"""
UTMOS评分模块
基于UTMOSv2的MOS预测系统
"""
import os
from typing import List, Dict
import warnings

warnings.filterwarnings("ignore")

# 设置HuggingFace离线模式，使用本地缓存
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 尝试导入UTMOS
try:
    import utmosv2
    UTMOS_AVAILABLE = True
except ImportError:
    UTMOS_AVAILABLE = False
    print("警告: utmosv2未安装，UTMOS评分将不可用")

# 导入torch并启用cuDNN - 已安装cuDNN 9.8.0，与PyTorch 2.8.0+cu128兼容
import torch
torch.backends.cudnn.enabled = True


class UTMOSCore:
    """UTMOS评分核心类"""
    
    def __init__(self):
        if not UTMOS_AVAILABLE:
            raise ImportError("utmosv2未安装")
        
        print("正在初始化UTMOS模型...")
        # 使用CUDA
        import torch
        if torch.cuda.is_available():
            self.model = utmosv2.create_model(pretrained=True, device='cuda')
            print("✓ UTMOS模型初始化完成 (CUDA模式)")
        else:
            self.model = utmosv2.create_model(pretrained=True, device='cpu')
            print("✓ UTMOS模型初始化完成 (CPU模式)")
        self.model.eval()
    
    def predict_file(self, file_path: str) -> float:
        """
        预测单个文件的MOS分数
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            MOS分数 (1-5)
        """
        try:
            mos = self.model.predict(input_path=file_path)
            return float(mos)
        except Exception as e:
            print(f"UTMOS预测失败 {file_path}: {e}")
            return 0.0
    
    def predict_files(self, file_list: List[str]) -> Dict[str, List[float]]:
        """
        预测多个文件的MOS分数
        
        Args:
            file_list: 音频文件路径列表
            
        Returns:
            包含UTMOS分数的字典
        """
        utmos_scores = []
        
        for file_path in file_list:
            score = self.predict_file(file_path)
            utmos_scores.append(score)
        
        return {"utmos": utmos_scores}


# 全局UTMOS实例
_utmos_instance = None


def get_utmos_instance():
    """获取UTMOS实例(单例模式)"""
    global _utmos_instance
    if _utmos_instance is None:
        _utmos_instance = UTMOSCore()
    return _utmos_instance


def predict_utmos(file_list: List[str]) -> Dict[str, List[float]]:
    """
    预测UTMOS分数的便捷函数
    
    Args:
        file_list: 音频文件路径列表
        
    Returns:
        包含UTMOS分数的字典
    """
    if not UTMOS_AVAILABLE:
        return {"utmos": [0.0] * len(file_list)}
    
    try:
        utmos = get_utmos_instance()
        return utmos.predict_files(file_list)
    except Exception as e:
        print(f"UTMOS预测失败: {e}")
        return {"utmos": [0.0] * len(file_list)}


if __name__ == "__main__":
    # 测试
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        if os.path.exists(test_file):
            print(f"测试文件: {test_file}")
            result = predict_utmos([test_file])
            print(f"UTMOS分数: {result['utmos'][0]:.4f}")
        else:
            print(f"文件不存在: {test_file}")
    else:
        print("用法: python utmos_score.py <wav_file>")

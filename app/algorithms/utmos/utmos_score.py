"""
UTMOS评分模块
基于UTMOSv2的MOS预测系统
"""
import os
import time
from typing import List, Dict
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# 设置HuggingFace离线模式，使用本地缓存
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 尝试导入UTMOS
try:
    import utmosv2
    UTMOS_AVAILABLE = True
    print("[UTMOS] 成功导入utmosv2模块")
except ImportError as e:
    UTMOS_AVAILABLE = False
    print(f"[UTMOS] 警告: utmosv2未安装，UTMOS评分将不可用 - {e}")

# 导入torch并启用cuDNN - 已安装cuDNN 9.8.0，与PyTorch 2.8.0+cu128兼容
import torch
torch.backends.cudnn.enabled = True


class UTMOSCore:
    """UTMOS评分核心类"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("=" * 60)
        self._log("[UTMOS] 开始初始化UTMOS模型")
        self._log("=" * 60)

        if not UTMOS_AVAILABLE:
            raise ImportError("utmosv2未安装")

        # 记录开始时间
        init_start = time.time()

        # 检查CUDA可用性
        self._log("\n[UTMOS] 步骤1/3: 检查运行环境")
        if torch.cuda.is_available():
            self.device = 'cuda'
            device_name = torch.cuda.get_device_name(0)
            self._log(f"  ✓ CUDA可用: {device_name}")
            self._log(f"  ✓ 使用设备: {self.device}")
        else:
            self.device = 'cpu'
            self._log(f"  ⚠ CUDA不可用，使用CPU模式")
            self._log(f"  ✓ 使用设备: {self.device}")

        # 检查模型路径
        self._log("\n[UTMOS] 步骤2/3: 检查模型路径")
        from utmosv2.utils._constants import _UTMOSV2_CHACHE
        model_dir = _UTMOSV2_CHACHE / "models" / "fusion_stage3"
        self._log(f"  模型缓存路径: {_UTMOSV2_CHACHE}")
        self._log(f"  模型目录: {model_dir}")

        if model_dir.exists():
            model_files = list(model_dir.glob("*.pth"))
            self._log(f"  ✓ 找到 {len(model_files)} 个模型文件:")
            for f in sorted(model_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                self._log(f"    - {f.name} ({size_mb:.1f} MB)")
        else:
            self._log(f"  ⚠ 模型目录不存在，将尝试下载")

        # 创建模型
        self._log("\n[UTMOS] 步骤3/3: 创建UTMOSv2模型")
        self._log(f"  配置: fusion_stage3")
        self._log(f"  使用fold: 0")

        try:
            self.model = utmosv2.create_model(pretrained=True, device=self.device)
            init_time = time.time() - init_start
            self._log(f"\n✓ UTMOS模型初始化完成 (耗时: {init_time:.2f}s)")
            self._log(f"  设备: {self.device}")
            self._log(f"  模型类型: SSLMultiSpecExtModelV2")
            self.model.eval()
        except Exception as e:
            self._log(f"\n✗ UTMOS模型初始化失败: {e}")
            raise
        
        self._log("=" * 60)
    
    def _log(self, msg: str, end: str = "\n"):
        """打印日志"""
        if self.verbose:
            print(msg, end=end)
    
    def predict_file(self, file_path: str) -> float:
        """
        预测单个文件的MOS分数
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            MOS分数 (1-5)
        """
        start_time = time.time()
        self._log(f"\n[UTMOS] 开始预测: {Path(file_path).name}")
        
        try:
            # 检查文件
            if not os.path.exists(file_path):
                self._log(f"  ✗ 文件不存在: {file_path}")
                return 0.0
            
            file_size = os.path.getsize(file_path) / 1024  # KB
            self._log(f"  文件大小: {file_size:.1f} KB")
            
            # 预测 - 使用初始化时的设备
            predict_start = time.time()
            mos = self.model.predict(input_path=file_path, device=self.device, verbose=False)
            predict_time = time.time() - predict_start
            
            self._log(f"  ✓ 预测完成: MOS={mos:.4f} (耗时: {predict_time:.3f}s)")
            return float(mos)
            
        except Exception as e:
            self._log(f"  ✗ UTMOS预测失败: {e}")
            import traceback
            self._log(f"  错误详情: {traceback.format_exc()}")
            return 0.0
    
    def predict_files(self, file_list: List[str]) -> Dict[str, List[float]]:
        """
        预测多个文件的MOS分数
        
        Args:
            file_list: 音频文件路径列表
            
        Returns:
            包含UTMOS分数的字典
        """
        total_start = time.time()
        self._log(f"\n{'=' * 60}")
        self._log(f"[UTMOS] 批量预测开始")
        self._log(f"  文件数量: {len(file_list)}")
        self._log(f"{'=' * 60}")
        
        utmos_scores = []
        success_count = 0
        
        for i, file_path in enumerate(file_list, 1):
            self._log(f"\n[{i}/{len(file_list)}] ", end="")
            score = self.predict_file(file_path)
            utmos_scores.append(score)
            if score > 0:
                success_count += 1
        
        total_time = time.time() - total_start
        avg_time = total_time / len(file_list) if file_list else 0
        
        self._log(f"\n{'=' * 60}")
        self._log(f"[UTMOS] 批量预测完成")
        self._log(f"  成功: {success_count}/{len(file_list)}")
        self._log(f"  总耗时: {total_time:.2f}s")
        self._log(f"  平均耗时: {avg_time:.3f}s/文件")
        self._log(f"{'=' * 60}")
        
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

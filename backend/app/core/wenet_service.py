"""
WeNet 语音识别服务
用于计算 WER (Word Error Rate) 词错误率
"""
import os
import logging
from typing import Optional, Dict, Any
import torch
import numpy as np

logger = logging.getLogger(__name__)


class WeNetService:
    """WeNet 语音识别服务"""

    def __init__(self):
        self.model = None
        self.language = "chinese"
        self._initialized = False
        self.model_dir = os.path.expanduser("~/.wenet/chinese")

    def initialize(self) -> bool:
        """初始化 WeNet 模型"""
        try:
            import wenet

            logger.info("正在初始化 WeNet 语音识别服务...")

            # 检查 CUDA 是否可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")

            # 检查模型是否已下载
            if not self._check_model_exists():
                logger.info("模型未找到，正在下载 WeNet 中文模型...")
                self._download_model()

            # 初始化 WeNet 模型
            # 优先级: 项目模型目录 > 本地缓存 > 自动下载
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            project_model = os.path.join(project_root, "models", "wenet")
            if os.path.exists(project_model) and os.path.exists(os.path.join(project_model, "final.pt")):
                self.model = wenet.load_model(project_model)
            elif os.path.exists(self.model_dir):
                self.model = wenet.load_model(self.model_dir)
            else:
                self.model = wenet.load_model(self.language)

            self._initialized = True
            logger.info("WeNet 语音识别服务初始化成功")
            return True

        except Exception as e:
            logger.error(f"WeNet 初始化失败: {e}")
            self._initialized = False
            return False

    def _check_model_exists(self) -> bool:
        """检查模型文件是否存在"""
        train_yaml = os.path.join(self.model_dir, "train.yaml")
        return os.path.exists(train_yaml)

    def _download_model(self) -> bool:
        """下载 WeNet 中文模型"""
        try:
            import urllib.request
            import tarfile
            import tempfile
            import shutil

            # WeNet 中文模型下载地址
            model_url = "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell2/20210618_unified_conformer_libtorch.tar.gz"

            logger.info(f"正在从 {model_url} 下载模型...")

            # 创建模型目录
            os.makedirs(self.model_dir, exist_ok=True)

            # 下载模型
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tar.gz") as tmp_file:
                urllib.request.urlretrieve(model_url, tmp_file.name)
                tmp_path = tmp_file.name

            logger.info("模型下载完成，正在解压...")

            # 解压模型
            with tarfile.open(tmp_path, "r:gz") as tar:
                # 解压到临时目录
                temp_extract_dir = tempfile.mkdtemp()
                tar.extractall(temp_extract_dir)

                # 查找解压后的目录
                extracted_dirs = [d for d in os.listdir(temp_extract_dir) if os.path.isdir(os.path.join(temp_extract_dir, d))]

                if extracted_dirs:
                    source_dir = os.path.join(temp_extract_dir, extracted_dirs[0])
                    # 移动文件到模型目录
                    for item in os.listdir(source_dir):
                        source = os.path.join(source_dir, item)
                        dest = os.path.join(self.model_dir, item)
                        if os.path.exists(dest):
                            shutil.rmtree(dest) if os.path.isdir(dest) else os.remove(dest)
                        shutil.move(source, dest)

                # 清理临时文件
                shutil.rmtree(temp_extract_dir)
                os.remove(tmp_path)

            logger.info(f"模型已安装到: {self.model_dir}")
            return True

        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            return False

    def recognize(self, audio_file: str) -> Optional[str]:
        """
        识别音频文件，返回识别文本

        Args:
            audio_file: 音频文件路径

        Returns:
            识别文本，失败返回 None
        """
        if not self._initialized:
            if not self.initialize():
                return None

        try:
            if not os.path.exists(audio_file):
                logger.error(f"音频文件不存在: {audio_file}")
                return None

            logger.info(f"正在识别音频: {audio_file}")

            # 使用 WeNet 进行识别
            result = self.model.transcribe(audio_file)

            # 提取文本（兼容多种返回类型）
            if hasattr(result, 'text'):
                text = result.text
            elif isinstance(result, dict):
                text = result.get("text", "")
            else:
                text = str(result)

            logger.info(f"识别结果: {text[:80]}...")
            return text

        except Exception as e:
            logger.error(f"音频识别失败: {e}")
            return None

    def calculate_wer(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        计算词错误率 (WER)

        Args:
            reference: 参考文本（标准答案）
            hypothesis: 识别文本（ASR结果）

        Returns:
            包含 WER 和 WCorr 的字典
        """
        try:
            # 导入项目中的 WER 计算模块
            import sys
            sys.path.insert(0, '/home/zhouchenghao/PycharmProjects/AudioMos')
            from wenet.wer import wer

            # 分词
            import jieba
            ref_words = list(jieba.cut(reference))
            hyp_words = list(jieba.cut(hypothesis))

            # 计算 WER
            wer_value, wcorr = wer(ref_words, hyp_words)

            return {
                "wer": float(wer_value),
                "wcorr": float(wcorr),
                "reference": reference,
                "hypothesis": hypothesis
            }

        except Exception as e:
            logger.error(f"WER 计算失败: {e}")
            return {
                "wer": 1.0,  # 完全错误
                "wcorr": 0.0,
                "reference": reference,
                "hypothesis": hypothesis,
                "error": str(e)
            }

    def process_audio_pair(self,
                          clean_audio: str,
                          enhanced_audio: str,
                          reference_text: Optional[str] = None) -> Dict[str, Any]:
        """
        处理音频对，计算增强后的 WER 改进

        Args:
            clean_audio: 干净音频路径（作为参考）
            enhanced_audio: 增强音频路径
            reference_text: 参考文本（可选，如果提供则直接使用）

        Returns:
            包含 WER 对比结果的字典
        """
        try:
            # 获取参考文本
            if reference_text:
                ref_text = reference_text
            else:
                # 使用干净音频作为参考
                ref_text = self.recognize(clean_audio)
                if not ref_text:
                    return {"error": "无法识别参考音频"}

            # 识别增强音频
            hyp_text = self.recognize(enhanced_audio)
            if not hyp_text:
                return {"error": "无法识别增强音频"}

            # 计算 WER
            result = self.calculate_wer(ref_text, hyp_text)

            return {
                "reference_text": ref_text,
                "hypothesis_text": hyp_text,
                **result
            }

        except Exception as e:
            logger.error(f"处理音频对失败: {e}")
            return {"error": str(e)}


# 全局服务实例
wenet_service = WeNetService()

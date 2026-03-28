"""
音频指纹识别系统
该模块用于实现音频指纹的生成和索引构建，可用于音频识别和匹配任务
"""
import csv
import hashlib
import os
import pickle
from typing import Dict, List, Tuple

import librosa
import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm


class AudioFingerprinter:
    def __init__(self, config: dict = None):
        """
        初始化音频指纹识别器

        Args:
            config: 配置字典，包含所有参数
        """
        # 默认配置
        self.config = {
            'audio_dir': "../DatasetCreation/audio_1000",  # 音频文件目录
            'metadata_csv': os.path.join("../DatasetCreation/audio_1000", "metadata.csv"),  # 元数据CSV文件路径
            'output_index': "fingerprints.pkl",  # 输出索引文件名
            'peak_neighborhood_size': 20,  # 峰值点邻域大小
            'fan_value': 50,  # 扇形值，用于控制匹配的频率范围
            'window_size': 4096,  # FFT窗口大小
            'overlap_ratio': 0.5,  # 重叠比率
            'min_hash_time_delta': 0,  # 最小哈希时间差
            'max_hash_time_delta': 200,  # 最大哈希时间差
            'energy_threshold_ratio': 0.3,  # 能量阈值比率
            'sample_rate': 22050  # 采样率
        }

        if config:
            self.config.update(config)  # 更新配置

        self.index: Dict[int, List[Tuple[int, int]]] = {}  # 初始化索引字典

    @staticmethod
    def _stable_hash(f1: int, f2: int, dt: int) -> int:
        """生成稳定的哈希值"""
        key = f"{f1}|{f2}|{dt}".encode("utf-8")  # 创建键字符串
        digest = hashlib.sha1(key).digest()  # 使用SHA1哈希算法
        return int.from_bytes(digest[:8], byteorder="big", signed=False)  # 转换为整数

    def _get_anchors(self, S: np.ndarray) -> List[Tuple[int, int]]:
        """获取频谱图中的峰值点"""
        footprint = np.ones((self.config['peak_neighborhood_size'],
                             self.config['peak_neighborhood_size']))  # 创建邻域模板
        local_max = maximum_filter(S, footprint=footprint) == S  # 应用最大值滤波器
        background = (S == 0)  # 获取背景
        eroded_bg = maximum_filter(background, footprint=footprint)
        peaks = local_max & ~eroded_bg
        return np.argwhere(peaks).tolist()

    def _fingerprint(self, y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """生成单个音频文件的指纹"""
        hop_length = int(self.config['window_size'] * self.config['overlap_ratio'])
        S = np.abs(librosa.stft(y, n_fft=self.config['window_size'],
                                hop_length=hop_length))

        energy = np.sum(S ** 2, axis=0)
        thresh = np.median(energy) * self.config['energy_threshold_ratio']

        anchors = self._get_anchors(S)
        anchors = [(f, t) for f, t in anchors if energy[t] >= thresh]

        hashes = []
        for i, (f1, t1) in enumerate(anchors):
            for j in range(1, self.config['fan_value']):
                if i + j < len(anchors):
                    f2, t2 = anchors[i + j]
                    dt = t2 - t1
                    if self.config['min_hash_time_delta'] <= dt <= self.config['max_hash_time_delta']:
                        h = self._stable_hash(f1, f2, dt)
                        hashes.append((h, t1))
        return hashes

    def fingerprint_file(self, path: str) -> List[Tuple[int, int]]:
        """处理单个音频文件"""
        y, sr = librosa.load(path, sr=self.config['sample_rate'], mono=True)
        return self._fingerprint(y, sr)

    def build_index(self) -> None:
        """构建指纹索引"""
        with open(self.config['metadata_csv'], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Fingerprinting"):
                tid = int(row["track_id"])
                mp3_path = os.path.join(self.config['audio_dir'], f"{tid}.mp3")
                if not os.path.isfile(mp3_path):
                    tqdm.write(f"Missing file: {mp3_path}")
                    continue
                try:
                    for h, offs in self.fingerprint_file(mp3_path):
                        self.index.setdefault(h, []).append((tid, offs))
                except Exception as e:
                    tqdm.write(f"Error on {tid}.mp3: {e}")

        with open(self.config['output_index'], "wb") as out:
            pickle.dump(self.index, out)
        print(f"Built index with {len(self.index)} unique hashes -> {self.config['output_index']}")

    @classmethod
    def from_config_file(cls, config_path: str):
        """从配置文件创建实例"""
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)


if __name__ == "__main__":
    fingerprinter = AudioFingerprinter()
    fingerprinter.build_index()

import os
import pickle
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from MusicFingerprintsComparison.DeepLearning.utils.BaseEvaluator import BaseEvaluator, BaseConfig

warnings.filterwarnings("ignore")


class CNNTunedConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_path = 'contrastive_model_tuned_best.pt'
        self.labelmap_path = 'label_map_tuned.pkl'
        self.test_data_dir = '../../DatasetCreation/TEST_METADATA/downloaded_tracks'
        self.test_metadata_csv = '../../DatasetCreation/TEST_METADATA/metadata_test_20.csv'
        self.output_dir = 'FINAL_TUNED_CNN_ANALYSIS_RESULTS'
        self.sr = 22050
        self.clip_duration = 5.0
        self.emb_dim = 128
        self.n_mels = 64


class CNNTunedEvaluator(BaseEvaluator):
    def __init__(self, config: CNNTunedConfig):
        super().__init__(config)

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

    def setup_model(self):
        """加载预训练模型"""
        self._load_labelmap()
        try:
            self.model = Encoder(emb_dim=256, n_classes=len(self.id2idx))
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {self.config.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        else:
            return True

    def _load_labelmap(self):
        """加载标签映射"""
        try:
            with open(self.config.labelmap_path, 'rb') as f:
                self.id2idx = pickle.load(f)
        except FileNotFoundError:
            print(f"Label map file not found at {self.config.labelmap_path}")
            raise
        else:
            return True

    def get_embedding(self, audio_clip):
        """获取音频嵌入"""
        mel_spec = self.compute_log_mel(audio_clip).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(mel_spec)
        return embedding.cpu()


class Encoder(nn.Module):
    def __init__(self, emb_dim=256, n_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes > 0 else None

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        z_unnorm = self.proj(h)
        return F.normalize(z_unnorm, dim=1)


def main():
    config = CNNTunedConfig()
    analyzer = CNNTunedEvaluator(config)
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()

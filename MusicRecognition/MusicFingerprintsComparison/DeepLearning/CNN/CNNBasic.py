import csv
import os
import random
from typing import Dict, Optional
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Config:
    """配置类，包含所有模型和训练相关的超参数"""

    def __init__(self):
        # 数据相关
        self.data_dir = '../../DatasetCreation/audio_1000'
        self.metadata_csv = os.path.join(self.data_dir, 'metadata_100.csv')
        self.model_path = 'contrastive_model_basic.pt'
        self.labelmap_path = 'label_map.pkl'
        self.checkpoint_dir = 'checkpoints'  # 检查点保存目录

        # 音频处理参数
        self.sr = 22050
        self.clip_duration = 5.0

        # 训练参数
        self.batch_size = 64
        self.epochs = 5
        self.lr = 1e-3
        self.temp = 0.1
        self.alpha = 1.0
        self.emb_dim = 128
        self.save_every = 5  # 每隔多少个epoch保存一次检查点


class AudioProcessor:
    """音频处理工具类"""

    @staticmethod
    def compute_log_mel(y: np.ndarray, sr: int) -> torch.Tensor:
        """计算对数梅尔频谱图

        Args:
            y: 音频信号
            sr: 采样率

        Returns:
            标准化后的对数梅尔频谱图
        """
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64
        )
        logm = librosa.power_to_db(melspec)
        logm = (logm - logm.mean()) / (logm.std() + 1e-6)
        return torch.from_numpy(logm).unsqueeze(0)


class ContrastiveAudioDataset(Dataset):
    """
    对比学习音频数据集

    用于音频对比学习任务：
        同一个音频的不同片段作为正样本对
        不同音频的片段作为负样本对
        常用于自监督学习或无监督学习场景
    """

    def __init__(self, csv_file: str, data_dir: str, clip_duration: float, sr: int):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            track_ids = [row['track_id'] for row in reader]

        self.ids = sorted(set(track_ids))
        self.id2idx = {tid: i for i, tid in enumerate(self.ids)}
        self.samples = track_ids

    def __len__(self) -> int:
        return len(self.samples)

    def _load_random_clip(self, track_id: str) -> np.ndarray:
        """
        从指定音轨ID加载一个随机音频片段
        参数:
            track_id (str): 音轨的唯一标识符
        返回:
            np.ndarray: 加载并处理后的音频数据数组
        """
        # 构建音频文件的完整路径
        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        # 使用librosa加载音频文件，采样率设为self.sr，转换为单声道
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        # 计算音频总时长（秒）
        total = len(y) / self.sr

        # 如果音频总长度大于目标片段长度
        if total > self.clip_dur:
            # 在有效范围内随机选择一个起始时间点
            start = random.uniform(0, total - self.clip_dur)
            # 将时间点转换为采样索引
            i0 = int(start * self.sr)
            i1 = int((start + self.clip_dur) * self.sr)
            # 截取指定长度的音频片段
            y = y[i0:i1]
        else:
            # 计算目标片段需要的采样点数
            target = int(self.clip_dur * self.sr)
            # 如果音频长度不足，使用wrap模式填充
            y = np.pad(y, (0, max(0, target - len(y))), mode='wrap')

        return y

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        根据给定的索引获取数据样本的方法
        参数:
            idx (int): 数据样本的索引值
        返回:
            Tuple[torch.Tensor, torch.Tensor, int]: 包含两个音频张量和一个标签的元组
                - x1: 第一个音频的梅尔频谱图
                - x2: 第二个音频的梅尔频谱图
                - label: 对应的标签索引
        """
        # 根据索引获取样本ID
        tid = self.samples[idx]
        # 获取样本对应的标签索引
        label = self.id2idx[tid]
        # 加载同一ID的两个随机音频剪辑
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)
        # 将音频转换为梅尔频谱图
        x1 = AudioProcessor.compute_log_mel(y1, self.sr)
        x2 = AudioProcessor.compute_log_mel(y2, self.sr)
        return x1, x2, label


class ContrastiveAudioDatasetCache(Dataset):
    """
    对比学习音频数据集

    用于音频对比学习任务：
        同一个音频的不同片段作为正样本对
        不同音频的片段作为负样本对
        常用于自监督学习或无监督学习场景
    """

    def __init__(self, csv_file: str, data_dir: str, clip_duration: float, sr: int):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.target_length = int(clip_duration * sr)

        # 添加音频缓存
        self._audio_cache = {}
        self._max_cache_size = 500

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            track_ids = [row['track_id'] for row in reader]

        self.ids = sorted(set(track_ids))
        self.id2idx = {tid: i for i, tid in enumerate(self.ids)}
        self.samples = track_ids
        self.cache_hit_times = 0

    def __len__(self) -> int:
        return len(self.samples)

    def _update_cache(self, track_id: str, audio: np.ndarray) -> None:
        """更新音频缓存"""
        if len(self._audio_cache) >= self._max_cache_size:
            # 移除最旧的缓存项
            oldest_key = next(iter(self._audio_cache))
            del self._audio_cache[oldest_key]
        self._audio_cache[track_id] = audio

    def _load_audio(self, track_id: str) -> np.ndarray:
        """加载音频，使用缓存机制"""
        if track_id in self._audio_cache:
            # 命中缓存
            self.cache_hit_times += 1
            if self.cache_hit_times % 500 == 0:
                print("Cache hit 500 times!")
            return self._audio_cache[track_id]

        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        # 使用更高效的加载参数
        y = librosa.load(
            path,
            sr=self.sr,
            mono=True,
            dtype=np.float32,  # 使用更节省内存的数据类型
            res_type='kaiser_fast'  # 使用更快的重采样方法
        )[0]

        self._update_cache(track_id, y)
        return y

    def _sample_random_segment(self, y: np.ndarray) -> np.ndarray:
        """更高效的随机采样方法"""
        if len(y) <= self.target_length:
            # 使用更高效的填充方法
            n_repeats = (self.target_length + len(y) - 1) // len(y)
            return np.tile(y, n_repeats)[:self.target_length]

        # 使用更高效的随机采样
        max_start = len(y) - self.target_length
        start_idx = np.random.randint(0, max_start + 1)
        return y[start_idx:start_idx + self.target_length]

    def _load_random_clip(self, track_id: str) -> np.ndarray:
        """
        从指定音轨ID加载一个随机音频片段
        参数:
            track_id (str): 音轨的唯一标识符
        返回:
            np.ndarray: 加载并处理后的音频数据数组
        """
        y = self._load_audio(track_id)
        return self._sample_random_segment(y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        根据给定的索引获取数据样本的方法
        参数:
            idx (int): 数据样本的索引值
        返回:
            Tuple[torch.Tensor, torch.Tensor, int]: 包含两个音频张量和一个标签的元组
                - x1: 第一个音频的梅尔频谱图
                - x2: 第二个音频的梅尔频谱图
                - label: 对应的标签索引
        """
        # 获取样本ID和标签
        tid = self.samples[idx]
        label = self.id2idx[tid]

        # 加载两个随机片段
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)

        # 转换为梅尔频谱图
        x1 = AudioProcessor.compute_log_mel(y1, self.sr)
        x2 = AudioProcessor.compute_log_mel(y2, self.sr)

        return x1, x2, label

    def clear_cache(self) -> None:
        """清理音频缓存"""
        self._audio_cache.clear()


class Encoder(nn.Module):
    """音频编码器模型"""

    def __init__(self, emb_dim: int = 128, n_classes: int = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes > 0 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播函数
        参数:
            x (torch.Tensor): 输入的张量数据
        返回:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - 第一个元素是经过标准化后的特征向量z
                - 第二个元素是分类器的输出logits（如果存在classifier），否则为None
        """
        # 通过卷积层处理输入，并展平特征图
        h = self.conv(x).view(x.size(0), -1)
        # 对特征向量进行标准化处理
        z = F.normalize(self.proj(h), dim=1)
        # 如果存在分类器，则计算logits；否则返回None
        logits = self.classifier(z) if self.classifier else None
        # 返回标准化后的特征向量和分类结果（如果有）
        return z, logits


class ContrastiveLoss:
    """对比学习损失函数"""

    @staticmethod
    def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
        """
        计算InfoNCE损失函数值
        InfoNCE (Noise Contrastive Estimation) 是一种用于对比学习的损失函数。
        它通过最大化正样本对之间的相似度，同时最小化负样本对之间的相似度来学习表征。

        Args:
            z1 (torch.Tensor): 第一个批次样本的特征向量，形状为 (batch_size, feature_dim)
            z2 (torch.Tensor): 第二个批次样本的特征向量，形状为 (batch_size, feature_dim)
            temp (float, optional): 温度参数，用于控制相似度分布的平滑程度。默认值为 0.1

        Returns:
            torch.Tensor: InfoNCE损失值，一个标量张量

        Note:
            - 该函数假设输入的两个批次中的样本是两两对应的正样本对
            - 使用余弦相似度计算样本之间的相似度
            - 温度参数 temp 越小，分布越尖锐；越大，分布越平滑
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temp
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(sim.device)
        return F.cross_entropy(sim, labels)


class AudioTrainer:
    """音频模型训练器"""

    def __init__(self, config: Config):
        self.config = config
        # 使用的设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Using device:", self.device)
        # 创建检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model: Encoder, optimizer: torch.optim.Optimizer, epoch: int, loss: float) -> None:
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, model: Encoder, optimizer: torch.optim.Optimizer) -> Tuple[int, float]:
        """加载最新的检查点"""
        # 如果已经存在最终的模型文件，
        checkpoints = [f for f in os.listdir(self.config.checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            return 0, 0.0
        else:
            print("Load checkpoint from previous training!")
        # 获取最新的检查点
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(self.config.checkpoint_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print(f"Resuming training from epoch {epoch}")
        return epoch, loss

    def train(self) -> Tuple[Encoder, Dict]:
        """训练模型"""
        # 准备数据集
        dataset = ContrastiveAudioDatasetCache(
            self.config.metadata_csv,
            self.config.data_dir,
            self.config.clip_duration,
            self.config.sr
        )
        torch.save(dataset.id2idx, self.config.labelmap_path)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,  # 多进程加载
            pin_memory=True,  # 加速GPU传输
            persistent_workers=True  # 保持worker进程
        )

        # 初始化模型
        model = Encoder(
            emb_dim=self.config.emb_dim,
            n_classes=len(dataset.ids)
        ).to(self.device)

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)

        # 尝试加载检查点
        start_epoch, _ = self.load_checkpoint(model, optimizer)
        print(f"Start training from epoch{start_epoch}!")
        # 训练循环
        for epoch in range(start_epoch + 1, self.config.epochs + 1):
            model.train()
            total_loss = 0.0

            for x1, x2, labels in tqdm(loader, desc=f"Epoch {epoch}"):
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                print("Data loaded to device!")
                z1, log1 = model(x1)
                z2, log2 = model(x2)

                loss_c = ContrastiveLoss.info_nce_loss(z1, z2, self.config.temp)
                loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
                loss = loss_c + self.config.alpha * loss_ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

            # 定期保存检查点
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(model, optimizer, epoch, avg_loss)

        # 保存最终模型
        torch.save(model.state_dict(), self.config.model_path)
        return model, dataset.id2idx


class AudioEvaluator:
    """音频模型评估器"""

    def __init__(self, config: Config):
        self.config = config

    def build_embeddings(self, model: Encoder, id2idx: Dict) -> Dict:
        """构建音轨嵌入"""
        device = next(model.parameters()).device
        inv_map = {v: k for k, v in id2idx.items()}
        embeddings = {}
        model.eval()

        with torch.no_grad():
            for idx in range(len(inv_map)):
                tid = inv_map[idx]
                y, _ = librosa.load(
                    os.path.join(self.config.data_dir, f"{tid}.mp3"),
                    sr=self.config.sr
                )
                clip = y[:int(self.config.clip_duration * self.config.sr)]
                x = AudioProcessor.compute_log_mel(clip, sr=self.config.sr).unsqueeze(0).to(device)
                z, _ = model(x)
                embeddings[tid] = z.cpu()

        return embeddings

    def evaluate(self, model: Encoder, embeddings: Dict) -> None:
        """评估模型性能"""
        device = next(model.parameters()).device
        correct = 0
        N = len(embeddings)
        model.eval()

        with torch.no_grad():
            for tid, z_ref in embeddings.items():
                y, _ = librosa.load(
                    os.path.join(self.config.data_dir, f"{tid}.mp3"),
                    sr=self.config.sr
                )
                start = random.uniform(0, max(0, len(y) / self.config.sr - self.config.clip_duration))
                i0, i1 = int(start * self.config.sr), int((start + self.config.clip_duration) * self.config.sr)
                clip = y[i0:i1]
                x = AudioProcessor.compute_log_mel(clip, sr=self.config.sr).unsqueeze(0).to(device)
                zq, _ = model(x)
                sims = {k: torch.cosine_similarity(zq.cpu(), v).item() for k, v in embeddings.items()}
                pred = max(sims, key=sims.get)
                if pred == tid:
                    correct += 1

        print(f"Top-1 accuracy (clean, 5s): {correct / N:.2%}")


if __name__ == '__main__':
    """主函数"""
    config = Config()
    trainer = AudioTrainer(config)
    evaluator = AudioEvaluator(config)

    # 训练模型
    model, id2idx = trainer.train()

    # 评估模型
    embeddings = evaluator.build_embeddings(model, id2idx)
    evaluator.evaluate(model, embeddings)

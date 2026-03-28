import csv
import os
import pickle
import random
import warnings
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise, Gain
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

warnings.filterwarnings("ignore")


@dataclass
class Config:
    # 数据配置
    data_dir: str = '../../DatasetCreation/audio_1000'
    metadata_csv: str = os.path.join(data_dir, 'metadata_100.csv')
    output_dir: str = 'AST_Results'
    model_path: str = os.path.join(output_dir, 'finetuned_ast_sota_model.pt')
    labelmap_path: str = os.path.join(output_dir, 'finetuned_ast_sota_label_map.pkl')

    # 模型配置
    model_name: str = "/home/zhouchenghao/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/f826b80d28226b62986cc218e5cec390b1096902"
    sr: int = 16000
    clip_duration: float = 10.0
    emb_dim: int = 256
    num_dropout_samples: int = 4

    # 训练配置
    batch_size: int = 12
    epochs: int = 5
    lr_head: float = 1e-4
    lr_backbone: float = 5e-5
    alpha: float = 0.5
    label_smoothing: float = 0.1
    validation_split: float = 0.2
    gradient_clip_val: float = 1.0
    warmup_ratio: float = 0.1
    accumulation_steps: int = 4


class WaveformDataset(Dataset):
    def __init__(self, csv_file, data_dir, track_ids, id2idx, feature_extractor, config, augment=False):
        self.data_dir = data_dir
        self.id2idx = id2idx
        self.feature_extractor = feature_extractor
        self.config = config
        self.augment = augment
        with open(csv_file, newline='', encoding='utf-8') as f:
            all_tracks = [row['track_id'] for row in csv.DictReader(f)]
        self.samples = [tid for tid in all_tracks if tid in track_ids]
        if self.augment:
            self.augmentation_pipeline = Compose([
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3)
            ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_random_clip(path: str, duration: float, sr: int) -> np.ndarray:
        """Load a random clip from an audio file"""
        target_len = int(duration * sr)
        try:
            y, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(target_len, dtype=np.float32)

        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        return y

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        path = os.path.join(self.data_dir, f"{tid}.mp3")
        y1 = self.load_random_clip(path, self.config.clip_duration, self.config.sr)
        y2 = self.load_random_clip(path, self.config.clip_duration, self.config.sr)
        if self.augment:
            y1 = self.augmentation_pipeline(samples=y1, sample_rate=self.config.sr)
            y2 = self.augmentation_pipeline(samples=y2, sample_rate=self.config.sr)
        inputs1 = self.feature_extractor(y1, sampling_rate=self.config.sr, return_tensors="pt")
        inputs2 = self.feature_extractor(y2, sampling_rate=self.config.sr, return_tensors="pt")
        return inputs1['input_values'].squeeze(0), inputs2['input_values'].squeeze(0), label


class AttentionPooling(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        attention_scores = self.attention(x)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_average = torch.sum(x * attention_weights, dim=1)
        return weighted_average


class ASTForSongID(nn.Module):
    def __init__(self, config):
        super().__init__()
        print(f"Loading pretrained model: {config.model_name}")
        self.base_model = AutoModel.from_pretrained(config.model_name)
        ast_embedding_size = self.base_model.config.hidden_size
        self.config = config

        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.pooling = AttentionPooling(in_features=ast_embedding_size)

        self.proj_head = nn.Sequential(
            nn.Linear(ast_embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, config.emb_dim)
        )
        self.classifier = nn.Linear(config.emb_dim, len(config.id2idx))

    def forward(self, x):
        ast_embedding = self.pooling(self.base_model(x).last_hidden_state)

        if self.training and self.config.num_dropout_samples > 1:
            z_unnorm_samples = torch.stack(
                [self.proj_head(ast_embedding) for _ in range(self.config.num_dropout_samples)],
                dim=0
            )
            z_unnorm = torch.mean(z_unnorm_samples, dim=0)
        else:
            z_unnorm = self.proj_head(ast_embedding)

        z_norm = F.normalize(z_unnorm, p=2, dim=1)
        logits = self.classifier(z_unnorm)
        return z_norm, logits


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)

        # 初始化特征提取器
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)

        # 加载数据
        self._prepare_data()
        # 初始化模型
        self.model = ASTForSongID(self.config).to(self.device)

        # 初始化优化器和学习率调度器
        self._init_optimizer()

        # 初始化混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        self.best_val_loss = float('inf')

    def _prepare_data(self):
        # 加载所有track IDs
        with open(self.config.metadata_csv, newline='', encoding='utf-8') as f:
            all_unique_track_ids = sorted(list(set(row['track_id'] for row in csv.DictReader(f))))

        # 创建标签映射
        self.config.id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}
        with open(self.config.labelmap_path, 'wb') as f:
            pickle.dump(self.config.id2idx, f)
        print(f"Saved label map with {len(self.config.id2idx)} tracks to {self.config.labelmap_path}")

        # 划分训练集和验证集
        train_ids, val_ids = train_test_split(
            all_unique_track_ids,
            test_size=self.config.validation_split,
            random_state=42
        )

        # 创建数据集
        train_ds = WaveformDataset(
            self.config.metadata_csv,
            self.config.data_dir,
            train_ids,
            self.config.id2idx,
            self.feature_extractor,
            self.config,
            augment=True
        )
        val_ds = WaveformDataset(
            self.config.metadata_csv,
            self.config.data_dir,
            val_ids,
            self.config.id2idx,
            self.feature_extractor,
            self.config,
            augment=False
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        self.val_ids = val_ids

    def _init_optimizer(self):
        param_groups = [
            {'params': self.model.base_model.parameters(), 'lr': self.config.lr_backbone},
            {'params': self.model.pooling.parameters(), 'lr': self.config.lr_head},
            {'params': self.model.proj_head.parameters(), 'lr': self.config.lr_head},
            {'params': self.model.classifier.parameters(), 'lr': self.config.lr_head},
            {'params': [self.model.temperature], 'lr': 1e-3}
        ]
        self.optimizer = torch.optim.AdamW(param_groups)

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = self._get_cosine_schedule_with_warmup(
            self.optimizer,
            warmup_steps,
            total_steps
        )

    def _get_cosine_schedule_with_warmup(self, optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + np.cos(
                np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)
            ))

        return LambdaLR(optimizer, lr_lambda)

    def _info_nce_loss(self, z1, z2, temp):
        temp = torch.clamp(temp, min=0.01)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temp
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).long().to(sim.device)
        return F.cross_entropy(sim, labels)

    def train_epoch(self):
        self.model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(self.train_loader, desc=f"Training", ncols=100)
        self.optimizer.zero_grad()

        for i, (y1, y2, labels) in enumerate(train_pbar):
            y1, y2, labels = y1.to(self.device), y2.to(self.device), labels.to(self.device)

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                z1, log1 = self.model(y1)
                z2, log2 = self.model(y2)
                loss_c = self._info_nce_loss(z1, z2, self.model.temperature)

                loss_ce = F.cross_entropy(log1, labels, label_smoothing=self.config.label_smoothing) + \
                          F.cross_entropy(log2, labels, label_smoothing=self.config.label_smoothing)
                loss = loss_c + self.config.alpha * loss_ce
                loss = loss / self.config.accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.config.accumulation_steps == 0 or (i + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            total_train_loss += loss.item() * self.config.accumulation_steps
            train_pbar.set_postfix(
                loss=f"{loss.item() * self.config.accumulation_steps:.4f}",
                temp=f"{self.model.temperature.item():.3f}"
            )

        return total_train_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for y1_v, y2_v, labels_v in self.val_loader:
                y1_v, y2_v, labels_v = y1_v.to(self.device), y2_v.to(self.device), labels_v.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    z1_v, log1_v = self.model(y1_v)
                    z2_v, log2_v = self.model(y2_v)
                    loss_c_val = self._info_nce_loss(z1_v, z2_v, self.model.temperature)
                    loss_ce_val = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                    val_loss = loss_c_val + self.config.alpha * loss_ce_val
                total_val_loss += val_loss.item()
        return total_val_loss / len(self.val_loader)

    def train(self):
        print(f"Starting training for {self.config.epochs} epochs...")
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            print(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Temp: {self.model.temperature.item():.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.model_path)
                print(f"New best model saved to {self.config.model_path}")

    def evaluate_final_model(self, n_clips_ref=5):
        if os.path.exists(self.config.model_path):
            print(f"Loading best model from {self.config.model_path}")
            self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        else:
            print("Warning: No saved model found. Evaluating with last epoch model.")

        self.model.eval()
        embeddings_db = {}
        print("\nBuilding reference embeddings for the validation set.")
        with torch.no_grad():
            for tid in tqdm(self.val_ids, desc="Building DB"):
                path = os.path.join(self.config.data_dir, f"{tid}.mp3")
                clip_embs = []
                for _ in range(n_clips_ref):
                    y_clip = WaveformDataset.load_random_clip(path, self.config.clip_duration,
                                                              self.config.sr)
                    inputs = self.feature_extractor(y_clip, sampling_rate=self.config.sr, return_tensors="pt").to(
                        self.device)
                    z, _ = self.model(inputs['input_values'])
                    clip_embs.append(z)
                if clip_embs:
                    embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

        correct_predictions = 0
        print("\nEvaluating Top-1 retrieval accuracy.")
        with torch.no_grad():
            for query_tid in tqdm(self.val_ids, desc="Querying"):
                y_query = WaveformDataset.load_random_clip(os.path.join(self.config.data_dir, f"{query_tid}.mp3"),
                                                           self.config.clip_duration, self.config.sr)
                inputs_query = self.feature_extractor(y_query, sampling_rate=self.config.sr, return_tensors="pt").to(
                    self.device)
                zq, _ = self.model(inputs_query['input_values'])
                zq_cpu = zq.cpu()
                sims = {tid: F.cosine_similarity(zq_cpu, emb).item() for tid, emb in embeddings_db.items()}
                predicted_tid = max(sims, key=sims.get)
                if predicted_tid == query_tid:
                    correct_predictions += 1

        accuracy = (correct_predictions / len(self.val_ids)) if self.val_ids else 0.0
        print("\n" + "=" * 50)
        print(f"Final Top-1 Accuracy: {accuracy:.2%}")
        print("=" * 50)


if __name__ == '__main__':
    config = Config()
    trainer = Trainer(config)
    trainer.train()
    trainer.evaluate_final_model()

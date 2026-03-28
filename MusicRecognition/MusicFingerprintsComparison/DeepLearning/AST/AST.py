import csv
import os
import pickle
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
class TrainConfig:
    """Configuration class for audio processing and training"""
    # Data paths
    data_dir: str = '../../DatasetCreation/audio_1000'
    metadata_csv: str = '../../DatasetCreation/audio_1000/metadata_100.csv'
    output_dir: str = 'AST_Results'

    # Model parameters
    # 在线下载模型
    # model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    # 本地文件
    model_name = "/home/zhouchenghao/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/f826b80d28226b62986cc218e5cec390b1096902"
    sr: int = 16000
    clip_duration: float = 10.0
    emb_dim: int = 256

    # Training parameters
    batch_size: int = 12
    epochs: int = 15
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    temp: float = 0.07
    alpha: float = 0.5
    validation_split: float = 0.2
    gradient_clip_val: float = 1.0
    warmup_ratio: float = 0.1


class AudioProcessor:
    """Handles audio processing and augmentation"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)
        self._setup_augmentation()

    def _setup_augmentation(self):
        """Setup audio augmentation pipeline"""
        self.augmentation_pipeline = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            Gain(min_gain_db=-6, max_gain_db=6, p=0.3)
        ])

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

    def process_audio(self, y: np.ndarray, augment: bool = False) -> torch.Tensor:
        """Process audio clip with optional augmentation"""
        if augment:
            y = self.augmentation_pipeline(samples=y, sample_rate=self.config.sr)

        inputs = self.feature_extractor(y, sampling_rate=self.config.sr, return_tensors="pt")
        return inputs['input_values'].squeeze(0)


class WaveformDataset(Dataset):
    """Dataset class for audio waveforms"""

    def __init__(self, csv_file: str, data_dir: str, track_ids: List[str],
                 id2idx: Dict[str, int], processor: AudioProcessor, augment: bool = False):
        self.data_dir = data_dir
        self.id2idx = id2idx
        self.processor = processor
        self.augment = augment

        with open(csv_file, newline='', encoding='utf-8') as f:
            all_tracks = [row['track_id'] for row in csv.DictReader(f)]
        self.samples = [tid for tid in all_tracks if tid in track_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tid = self.samples[idx]
        label = self.id2idx[tid]
        path = os.path.join(self.data_dir, f"{tid}.mp3")

        y1 = AudioProcessor.load_random_clip(path, self.processor.config.clip_duration,
                                             self.processor.config.sr)
        y2 = AudioProcessor.load_random_clip(path, self.processor.config.clip_duration,
                                             self.processor.config.sr)

        x1 = self.processor.process_audio(y1, self.augment)
        x2 = self.processor.process_audio(y2, self.augment)

        return x1, x2, label


class ASTModel(nn.Module):
    """Audio Spectrogram Transformer model for song identification"""

    def __init__(self, config: TrainConfig, n_classes: int):
        super().__init__()
        print(f"Loading pretrained model: {config.model_name}")
        self.base_model = AutoModel.from_pretrained(config.model_name)

        embedding_size = self.base_model.config.hidden_size

        self.proj_head = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, config.emb_dim)
        )
        self.classifier = nn.Linear(config.emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.base_model(x)
        embedding = torch.mean(outputs.last_hidden_state, dim=1)

        z_unnorm = self.proj_head(embedding)
        z_norm = F.normalize(z_unnorm, p=2, dim=1)
        logits = self.classifier(z_unnorm)

        return z_norm, logits


class AudioTrainer:
    """Handles training and evaluation of the audio model"""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AudioProcessor(config)

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Setup paths
        self.model_path = os.path.join(config.output_dir, 'finetuned_ast_model.pt')
        self.labelmap_path = os.path.join(config.output_dir, 'finetuned_ast_label_map.pkl')

        # Initialize model
        self.model = None
        self.id2idx = None

    def setup_data(self) -> Tuple[DataLoader, DataLoader]:
        """Setup training and validation data loaders"""
        # Load track IDs
        with open(self.config.metadata_csv, newline='', encoding='utf-8') as f:
            all_unique_track_ids = sorted(list(set(row['track_id'] for row in csv.DictReader(f))))

        # Create label mapping
        self.id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}
        with open(self.labelmap_path, 'wb') as f:
            pickle.dump(self.id2idx, f)

        # Split data
        train_ids, val_ids = train_test_split(all_unique_track_ids,
                                              test_size=self.config.validation_split,
                                              random_state=42)

        # Create datasets
        train_ds = WaveformDataset(self.config.metadata_csv, self.config.data_dir,
                                   train_ids, self.id2idx, self.processor, augment=True)
        val_ds = WaveformDataset(self.config.metadata_csv, self.config.data_dir,
                                 val_ids, self.id2idx, self.processor, augment=False)

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size,
                                  shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.batch_size,
                                shuffle=False, num_workers=2, pin_memory=True)

        return train_loader, val_loader

    def setup_model_and_optimizer(self) -> Tuple[torch.optim.Optimizer, LambdaLR]:
        """Setup model and optimizer"""
        self.model = ASTModel(self.config, len(self.id2idx)).to(self.device)

        param_groups = [
            {'params': self.model.base_model.parameters(), 'lr': self.config.lr_backbone},
            {'params': self.model.proj_head.parameters(), 'lr': self.config.lr_head},
            {'params': self.model.classifier.parameters(), 'lr': self.config.lr_head}
        ]
        optimizer = torch.optim.AdamW(param_groups)

        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = self._get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return optimizer, scheduler

    @staticmethod
    def _get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                         warmup_steps: int, total_steps: int) -> LambdaLR:
        """Create cosine learning rate schedule with warmup"""

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) /
                                       (total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float) -> torch.Tensor:
        """Calculate InfoNCE loss"""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temp
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)]).long().to(sim.device)
        return F.cross_entropy(sim, labels)

    def train_epoch(self, optimizer: torch.optim.Optimizer, scheduler: LambdaLR,
                    scaler: torch.cuda.amp.GradScaler) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Training", ncols=100)

        for y1, y2, labels in pbar:
            y1, y2, labels = y1.to(self.device), y2.to(self.device), labels.to(self.device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                z1, log1 = self.model(y1)
                z2, log2 = self.model(y2)
                loss_c = self.info_nce_loss(z1, z2, self.config.temp)
                loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
                loss = loss_c + self.config.alpha * loss_ce

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for y1, y2, labels in tqdm(self.val_loader, desc="Validation"):
                y1, y2, labels = y1.to(self.device), y2.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                    z1, log1 = self.model(y1)
                    z2, log2 = self.model(y2)
                    loss_c = self.info_nce_loss(z1, z2, self.config.temp)
                    loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
                    loss = loss_c + self.config.alpha * loss_ce

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def evaluate_model(self, n_clips_ref: int = 5) -> float:
        """Evaluate model using top-1 accuracy"""
        self.model.eval()

        # Build reference embeddings
        embeddings_db = {}
        print("\nBuilding reference embeddings...")
        with torch.no_grad():
            for tid in tqdm(self.val_ids, desc="Building DB"):
                path = os.path.join(self.config.data_dir, f"{tid}.mp3")
                clip_embs = []
                for _ in range(n_clips_ref):
                    y_clip = AudioProcessor.load_random_clip(path, self.config.clip_duration,
                                                             self.config.sr)
                    inputs = self.processor.process_audio(y_clip)
                    z, _ = self.model(inputs.unsqueeze(0).to(self.device))
                    clip_embs.append(z)
                if clip_embs:
                    embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

        # Evaluate accuracy
        correct = 0
        print("\nEvaluating...")
        with torch.no_grad():
            for query_tid in tqdm(self.val_ids, desc="Querying"):
                y_query = AudioProcessor.load_random_clip(
                    os.path.join(self.config.data_dir, f"{query_tid}.mp3"),
                    self.config.clip_duration, self.config.sr
                )
                inputs = self.processor.process_audio(y_query)
                zq, _ = self.model(inputs.unsqueeze(0).to(self.device))
                zq_cpu = zq.cpu()

                sims = {tid: F.cosine_similarity(zq_cpu, emb).item()
                        for tid, emb in embeddings_db.items()}
                predicted_tid = max(sims, key=sims.get)

                if predicted_tid == query_tid:
                    correct += 1

        return correct / len(self.val_ids)

    def train(self):
        """Main training loop"""
        print(f"Using device: {self.device}")

        # Setup data
        self.train_loader, self.val_loader = self.setup_data()
        self.val_ids = [tid for tid in self.id2idx.keys()
                        if tid not in self.train_loader.dataset.samples]

        # Setup model and optimizer
        optimizer, scheduler = self.setup_model_and_optimizer()
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self.train_epoch(optimizer, scheduler, scaler)
            val_loss = self.validate()

            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"New best model saved to {self.model_path}")

        # Final evaluation
        print("\nStarting final evaluation...")
        if os.path.exists(self.model_path):
            print(f"Loading best model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        accuracy = self.evaluate_model()
        print(f"\nFinal Top-1 Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    config = TrainConfig()
    trainer = AudioTrainer(config)
    trainer.train()

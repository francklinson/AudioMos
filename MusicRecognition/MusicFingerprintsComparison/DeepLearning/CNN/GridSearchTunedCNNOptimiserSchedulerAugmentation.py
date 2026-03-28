import os
import csv
import pickle
import random
import time
import itertools
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torchaudio.transforms as T
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise, Gain, Mp3Compression

@dataclass
class Config:
    # 数据配置
    data_dir: str = '../../DatasetCreation/audio_1000'
    metadata_csv: str = os.path.join(data_dir, 'metadata_100.csv')
    sr: int = 22050
    clip_duration: float = 5.0
    n_reference_clips: int = 5

    # 训练配置
    epochs: int = 15
    validation_split: float = 0.2
    patience_lr: int = 3
    patience_early_stop: int = 5

    # 模型配置
    emb_dim: int = 256
    n_classes: int = 1000

    # 损失函数参数
    temp: float = 0.07
    alpha: float = 0.25

class AudioProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.spec_augment_transform = nn.Sequential(
            T.FrequencyMasking(freq_mask_param=20),
            T.TimeMasking(time_mask_param=40)
        )
        self.audioment_pipelines = {
            'audioment_light': Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
            ]),
            'audioment_heavy': Compose([
                TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
                Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.2, backend='pydub'),
            ])
        }

    @staticmethod
    def compute_log_mel(y: np.ndarray, sr: int) -> torch.Tensor:
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        logm = librosa.power_to_db(melspec)
        logm = (logm - logm.mean()) / (logm.std() + 1e-6)
        return torch.from_numpy(logm).unsqueeze(0)

    def load_random_clip(self, path: str) -> np.ndarray:
        target_len = int(self.config.clip_duration * self.config.sr)
        try:
            y, _ = librosa.load(path, sr=self.config.sr, mono=True)
        except Exception:
            return np.zeros(target_len, dtype=np.float32)

        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        return y

    def apply_augmentation(self, y: np.ndarray, augment_config: str) -> np.ndarray:
        if augment_config in self.audioment_pipelines:
            pipeline = self.audioment_pipelines[augment_config]
            return pipeline(samples=y, sample_rate=self.config.sr)
        return y

class AudioDataset(Dataset):
    def __init__(self, config: Config, processor: AudioProcessor, track_ids: List[str],
                 id2idx: Dict[str, int], augment_config: str = 'none'):
        self.config = config
        self.processor = processor
        self.id2idx = id2idx
        self.augment_config = augment_config

        with open(config.metadata_csv, newline='', encoding='utf-8') as f:
            all_tracks = [row['track_id'] for row in csv.DictReader(f)]
        self.samples = [tid for tid in all_tracks if tid in track_ids]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        tid = self.samples[idx]
        label = self.id2idx[tid]
        path = os.path.join(self.config.data_dir, f"{tid}.mp3")

        y1 = self.processor.load_random_clip(path)
        y2 = self.processor.load_random_clip(path)

        y1 = self.processor.apply_augmentation(y1, self.augment_config)
        y2 = self.processor.apply_augmentation(y2, self.augment_config)

        x1 = self.processor.compute_log_mel(y1, self.config.sr)
        x2 = self.processor.compute_log_mel(y2, self.config.sr)

        if self.augment_config == 'specaugment':
            x1 = self.processor.spec_augment_transform(x1)
            x2 = self.processor.spec_augment_transform(x2)

        return x1, x2, label

class AudioEncoder(nn.Module):
    def __init__(self, config: Config, architecture: str = 'Encoder3Layer'):
        super().__init__()
        self.config = config
        self.architecture = architecture

        if architecture == 'Encoder3Layer':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1,1)),
            )
            self.proj = nn.Sequential(
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, config.emb_dim)
            )
        else:  # Encoder4Layer
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            )
            self.proj = nn.Sequential(
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, config.emb_dim)
            )

        self.classifier = nn.Linear(config.emb_dim, config.n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).view(x.size(0), -1)
        z_unnorm = self.proj(h)
        z = F.normalize(z_unnorm, dim=1)
        logits = self.classifier(z)
        return z, logits

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_model(self, model_arch: str) -> AudioEncoder:
        model = AudioEncoder(self.config, model_arch).to(self.device)
        return model

    def setup_optimizer(self, model: AudioEncoder, optimizer_name: str, lr: float):
        return getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)

    def setup_scheduler(self, optimizer, scheduler_name: str, train_loader_len: int):
        if scheduler_name == 'ReduceLROnPlateau':
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.patience_lr)
        else:
            return CosineAnnealingLR(optimizer, T_max=train_loader_len * self.config.epochs)

    def compute_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                    logits1: torch.Tensor, logits2: torch.Tensor,
                    labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Contrastive loss
        B = z1.size(0)
        if B == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.config.temp
        mask = torch.eye(2*B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -1e9)
        labels_contrastive = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).long().to(sim.device)
        loss_contrastive = F.cross_entropy(sim, labels_contrastive)

        # Classification loss
        loss_classification = F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels)

        # Combined loss
        total_loss = loss_contrastive + self.config.alpha * loss_classification

        return total_loss, loss_contrastive, loss_classification

    def evaluate(self, model: AudioEncoder, val_dataset: AudioDataset) -> float:
        model.eval()
        embeddings_db = {}

        # Build reference database
        for tid in tqdm(val_dataset.samples, desc="[Eval] Building DB", leave=False):
            path = os.path.join(val_dataset.config.data_dir, f"{tid}.mp3")
            clip_embs = []
            for _ in range(self.config.n_reference_clips):
                y_clip = val_dataset.processor.load_random_clip(path)
                x = val_dataset.processor.compute_log_mel(y_clip, self.config.sr).unsqueeze(0).to(self.device)
                z, _ = model(x)
                clip_embs.append(z.squeeze(0))
            if clip_embs:
                embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

        # Evaluate queries
        correct, total = 0, 0
        if not embeddings_db:
            return 0.0

        for query_tid in tqdm(val_dataset.samples, desc="[Eval] Querying", leave=False):
            path = os.path.join(val_dataset.config.data_dir, f"{query_tid}.mp3")
            y_query = val_dataset.processor.load_random_clip(path)
            x_query = val_dataset.processor.compute_log_mel(y_query, self.config.sr).unsqueeze(0).to(self.device)
            zq, _ = model(x_query)
            zq_cpu = zq.squeeze(0).cpu()

            sims = {tid: F.cosine_similarity(zq_cpu.unsqueeze(0), emb.unsqueeze(0)).item()
                   for tid, emb in embeddings_db.items()}
            if not sims:
                continue
            pred_tid = max(sims, key=sims.get)
            if pred_tid == query_tid:
                correct += 1
            total += 1

        return (correct / total * 100) if total > 0 else 0.0

    def train_epoch(self, model: AudioEncoder, train_loader: DataLoader,
                   optimizer, scheduler) -> float:
        model.train()
        total_loss = 0
        for x1, x2, labels in train_loader:
            x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            z1, logits1 = model(x1)
            z2, logits2 = model(x2)

            loss, _, _ = self.compute_loss(z1, z2, logits1, logits2, labels)
            loss.backward()
            optimizer.step()

            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate_epoch(self, model: AudioEncoder, val_loader: DataLoader) -> float:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x1, x2, labels in val_loader:
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                z1, logits1 = model(x1)
                z2, logits2 = model(x2)

                loss, _, _ = self.compute_loss(z1, z2, logits1, logits2, labels)
                total_loss += loss.item()

        return total_loss / len(val_loader)

class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.trainer = Trainer(config)
        self.processor = AudioProcessor(config)
        self.results_csv = 'hyperparameter_search_results.csv'

        # Load track IDs
        with open(config.metadata_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.all_track_ids = sorted(list(set(row['track_id'] for row in reader)))
        self.id2idx = {tid: i for i, tid in enumerate(self.all_track_ids)}

    def setup_datasets(self, train_ids: List[str], val_ids: List[str],
                      augment_config: str) -> Tuple[AudioDataset, AudioDataset]:
        train_ds = AudioDataset(self.config, self.processor, train_ids,
                              self.id2idx, augment_config)
        val_ds = AudioDataset(self.config, self.processor, val_ids,
                            self.id2idx, 'none')
        return train_ds, val_ds

    def run_experiment(self, exp_config: Dict) -> float:
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # Split data
        train_ids, val_ids = train_test_split(self.all_track_ids,
                                            test_size=self.config.validation_split,
                                            random_state=42)

        # Setup datasets
        train_ds, val_ds = self.setup_datasets(train_ids, val_ids,
                                              exp_config['augmentation'])

        # Setup data loaders
        train_loader = DataLoader(train_ds, batch_size=exp_config['batch_size'],
                                shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=exp_config['batch_size'],
                              shuffle=False)

        # Setup model
        model = self.trainer.setup_model(exp_config['model_arch'])

        # Setup optimizer and scheduler
        optimizer = self.trainer.setup_optimizer(model, exp_config['optimizer'],
                                                exp_config['learning_rate'])
        scheduler = self.trainer.setup_scheduler(optimizer, exp_config['scheduler'],
                                               len(train_loader))

        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0
        print("Starting training...")
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.trainer.train_epoch(model, train_loader, optimizer, scheduler)

            # Validate
            val_loss = self.trainer.validate_epoch(model, val_loader)

            # Update scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.config.patience_early_stop:
                break
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        print("Training complete.")
        # Final evaluation
        print("Start evaluation...")
        final_accuracy = self.trainer.evaluate(model, val_ds)
        return final_accuracy

    def run_all_experiments(self, search_space: Dict):
        # Generate all experiment configurations
        keys, values = zip(*search_space.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Check for existing results
        completed_configs = set()
        header = list(search_space.keys()) + ['accuracy', 'duration_minutes', 'timestamp']

        if os.path.exists(self.results_csv):
            print(f"Found existing results file at '{self.results_csv}'. Reading experiments...")
            try:
                with open(self.results_csv, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            config_from_row = {}
                            for key in search_space.keys():
                                original_value_type = type(search_space[key][0])
                                config_from_row[key] = original_value_type(row[key])
                            completed_configs.add(tuple(sorted(config_from_row.items())))
                        except (KeyError, ValueError) as e:
                            print(f"Warning: Could not parse a row in the CSV: {row}. {e}")
            except Exception as e:
                print(f"Error reading or parsing CSV file: {e}")
        else:
            with open(self.results_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Run experiments
        total_experiments = len(experiments)
        current_experiment_num = 0

        for config in experiments:
            config_tuple = tuple(sorted(config.items()))
            if config_tuple in completed_configs:
                continue

            current_experiment_num += 1
            start_time = time.time()

            print("\n" + "="*80)
            print(f"Running Experiment {current_experiment_num}/{total_experiments}")
            print(f"Config: {config}")
            print("="*80)

            try:
                accuracy = self.run_experiment(config)
            except Exception as e:
                print(f"EXPERIMENT FAILED: {e}")
                traceback.print_exc()
                accuracy = 'FAIL'

            duration_minutes = (time.time() - start_time) / 60
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            # Log results
            log_row = list(config.values()) + [
                f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy,
                f"{duration_minutes:.2f}",
                timestamp
            ]

            with open(self.results_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

            completed_configs.add(config_tuple)
            print(f"Experiment finished. Accuracy: {accuracy}")

        print("\n\nAll experiments complete")
        print(f"Results saved to: '{self.results_csv}'")

if __name__ == '__main__':
    # Configuration
    config = Config()

    # Search space
    SEARCH_SPACE = {
        'batch_size': [32],
        'learning_rate': [1e-4],
        'emb_dim': [256],
        'temp': [0.07],
        'alpha': [0.25],
        'optimizer': ['Adam', 'AdamW'],
        'scheduler': ['ReduceLROnPlateau', 'CosineAnnealingLR'],
        'model_arch': ['Encoder3Layer', 'Encoder4Layer'],
        'augmentation': [
            'none',
            'specaugment',
            'audioment_light',
            'audioment_heavy',
        ]
    }

    # Run experiments
    manager = ExperimentManager(config)
    manager.run_all_experiments(SEARCH_SPACE)


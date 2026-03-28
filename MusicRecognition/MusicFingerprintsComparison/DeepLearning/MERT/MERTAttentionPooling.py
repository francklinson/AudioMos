import os
import csv
import pickle
import random
import warnings

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


class MERTAttentionConfig:
    def __init__(self):
        self.DATA_DIR = '../../DatasetCreation/audio_1000'
        self.METADATA_CSV = os.path.join(self.DATA_DIR, 'metadata_100.csv')
        self.DRIVE_OUTPUT_DIR = './MERT_Results'
        os.makedirs(self.DRIVE_OUTPUT_DIR, exist_ok=True)
        self.MODEL_PATH = os.path.join(self.DRIVE_OUTPUT_DIR, 'finetuned_mert_attention_model.pt')
        self.LABELMAP_PATH = os.path.join(self.DRIVE_OUTPUT_DIR, 'finetuned_mert_attention_label_map.pkl')

        # self.MODEL_NAME = "m-a-p/MERT-v1-95M"
        self.MODEL_NAME = "/home/zhouchenghao/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
        self.SR = 24000
        self.CLIP_DURATION = 5.0
        self.BATCH_SIZE = 16
        self.EPOCHS = 5
        self.LR_HEAD = 1e-4
        self.LR_BACKBONE = 5e-5
        self.ALPHA = 0.5
        self.LABEL_SMOOTHING = 0.1
        self.EMB_DIM = 256
        self.VALIDATION_SPLIT = 0.2
        self.GRADIENT_CLIP_VAL = 1.0
        self.WARMUP_RATIO = 0.1


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


class MERTForSongIDWithAttention(nn.Module):
    def __init__(self, model_name, emb_dim, n_classes):
        super().__init__()
        print(f"Loading pretrained model: {model_name}")
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        mert_embedding_size = self.base_model.config.hidden_size

        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.pooling = AttentionPooling(in_features=mert_embedding_size)

        self.proj_head = nn.Sequential(
            nn.Linear(mert_embedding_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        mert_out = self.base_model(x).last_hidden_state
        mert_embedding = self.pooling(mert_out)
        z_unnorm = self.proj_head(mert_embedding)
        z_norm = F.normalize(z_unnorm, p=2, dim=1)
        logits = self.classifier(z_unnorm)
        return z_norm, logits


class MERTAttentionTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

    def setup_model(self, n_classes):
        self.model = MERTForSongIDWithAttention(
            model_name=self.config.MODEL_NAME,
            emb_dim=self.config.EMB_DIM,
            n_classes=n_classes
        ).to(self.device)

        param_groups = [
            {'params': self.model.base_model.parameters(), 'lr': self.config.LR_BACKBONE},
            {'params': self.model.pooling.parameters(), 'lr': self.config.LR_HEAD},
            {'params': self.model.proj_head.parameters(), 'lr': self.config.LR_HEAD},
            {'params': self.model.classifier.parameters(), 'lr': self.config.LR_HEAD},
            {'params': [self.model.temperature], 'lr': 1e-3}
        ]
        self.optimizer = torch.optim.AdamW(param_groups)

        total_steps = len(self.train_loader) * self.config.EPOCHS
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        self.scheduler = self.get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == 'cuda'))

    def prepare_data(self):
        with open(self.config.METADATA_CSV, newline='', encoding='utf-8') as f:
            all_unique_track_ids = sorted(list(set(row['track_id'] for row in csv.DictReader(f))))
        id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}

        with open(self.config.LABELMAP_PATH, 'wb') as f:
            pickle.dump(id2idx, f)

        train_ids, val_ids = train_test_split(
            all_unique_track_ids,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=42
        )

        train_ds = WaveformDataset(
            self.config.METADATA_CSV,
            self.config.DATA_DIR,
            train_ids,
            id2idx,
            self.feature_extractor,
            augment=True
        )
        val_ds = WaveformDataset(
            self.config.METADATA_CSV,
            self.config.DATA_DIR,
            val_ids,
            id2idx,
            self.feature_extractor,
            augment=False
        )

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        return val_ids, id2idx

    @staticmethod
    def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def info_nce_loss(z1, z2, temp):
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

        for y1, y2, labels in train_pbar:
            y1, y2, labels = y1.to(self.device), y2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                z1, log1 = self.model(y1)
                z2, log2 = self.model(y2)

                loss_c = self.info_nce_loss(z1, z2, self.model.temperature)
                loss_ce = F.cross_entropy(log1, labels, label_smoothing=self.config.LABEL_SMOOTHING) + \
                          F.cross_entropy(log2, labels, label_smoothing=self.config.LABEL_SMOOTHING)
                loss = loss_c + self.config.ALPHA * loss_ce

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_VAL)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_train_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", temp=f"{self.model.temperature.item():.3f}")

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
                    loss_c_val = self.info_nce_loss(z1_v, z2_v, self.model.temperature)
                    loss_ce_val = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                    val_loss = loss_c_val + self.config.ALPHA * loss_ce_val
                total_val_loss += val_loss.item()

        return total_val_loss / len(self.val_loader)

    def train(self):
        val_ids, id2idx = self.prepare_data()
        self.setup_model(len(id2idx))

        best_val_loss = float('inf')
        print(f"Starting MERT fine-tuning with attention pooling for {self.config.EPOCHS} epochs...")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()

            print(
                f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Temp: {self.model.temperature.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                print(f"New best model saved to {self.config.MODEL_PATH}")

        return val_ids, id2idx


class MERTAttentionEvaluator:
    def __init__(self, config, model, feature_extractor):
        self.config = config
        self.model = model
        self.feature_extractor = feature_extractor
        self.device = next(model.parameters()).device

    def evaluate_final_model(self, query_track_ids, id2idx, n_clips_ref=5):
        self.model.eval()
        embeddings_db = {}

        print("\nBuilding reference embeddings for the validation set.")
        with torch.no_grad():
            for tid in tqdm(query_track_ids, desc="Building DB"):
                path = os.path.join(self.config.DATA_DIR, f"{tid}.mp3")
                clip_embs = []
                for _ in range(n_clips_ref):
                    y_clip = WaveformDataset._load_random_clip(path)
                    inputs = self.feature_extractor(y_clip, sampling_rate=self.config.SR, return_tensors="pt").to(
                        self.device)
                    z, _ = self.model(inputs['input_values'])
                    clip_embs.append(z)
                if clip_embs:
                    embeddings_db[tid] = torch.stack(clip_embs).mean(dim=0).cpu()

        correct_predictions = 0
        print("\nEvaluating Top-1 retrieval accuracy.")
        with torch.no_grad():
            for query_tid in tqdm(query_track_ids, desc="Querying"):
                y_query = WaveformDataset._load_random_clip(os.path.join(self.config.DATA_DIR, f"{query_tid}.mp3"))
                inputs_query = self.feature_extractor(y_query, sampling_rate=self.config.SR, return_tensors="pt").to(
                    self.device)
                zq, _ = self.model(inputs_query['input_values'])
                zq_cpu = zq.cpu()

                sims = {tid: F.cosine_similarity(zq_cpu, emb).item() for tid, emb in embeddings_db.items()}
                predicted_tid = max(sims, key=sims.get)

                if predicted_tid == query_tid:
                    correct_predictions += 1

        accuracy = (correct_predictions / len(query_track_ids)) if query_track_ids else 0.0
        print("\n" + "=" * 50)
        print(f"Final Top-1 Accuracy (MERT + Attention Pooling): {accuracy:.2%}")
        print("=" * 50)


class WaveformDataset(Dataset):
    def __init__(self, csv_file, data_dir, track_ids, id2idx, feature_extractor, augment=False):
        self.data_dir = data_dir
        self.id2idx = id2idx
        self.feature_extractor = feature_extractor
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
    def _load_random_clip(path, duration=5.0, sr=24000):
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

        y1 = self._load_random_clip(path)
        y2 = self._load_random_clip(path)

        if self.augment:
            y1 = self.augmentation_pipeline(samples=y1, sample_rate=24000)
            y2 = self.augmentation_pipeline(samples=y2, sample_rate=24000)

        inputs1 = self.feature_extractor(y1, sampling_rate=24000, return_tensors="pt", padding=True)
        inputs2 = self.feature_extractor(y2, sampling_rate=24000, return_tensors="pt", padding=True)

        return inputs1['input_values'].squeeze(0), inputs2['input_values'].squeeze(0), label


if __name__ == '__main__':
    config = MERTAttentionConfig()
    trainer = MERTAttentionTrainer(config)

    val_ids, id2idx = trainer.train()

    print("\nFine-tuning Finished. Starting Final Evaluation.")
    if os.path.exists(config.MODEL_PATH):
        print(f"Loading best model from {config.MODEL_PATH}")
        trainer.model.load_state_dict(torch.load(config.MODEL_PATH, map_location=trainer.device))
    else:
        print("Warning: No saved model found. Evaluating with last epoch model.")

    evaluator = MERTAttentionEvaluator(config, trainer.model, trainer.feature_extractor)
    evaluator.evaluate_final_model(val_ids, id2idx)

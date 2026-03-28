import csv
import itertools
import os
import pickle
import random
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class AudioConfig:
    def __init__(self):
        self.DATA_DIR = '../../DatasetCreation/audio_1000'
        self.METADATA_CSV = os.path.join(self.DATA_DIR, 'metadata_100.csv')
        self.LABELMAP_PATH = 'label_map_gpu_grid_search.pkl'
        self.TEMP_MODEL_PATH = 'temp_best_model_for_trial.pt'

        self.SR = 22050
        self.CLIP_DURATION = 5.0
        self.VALIDATION_SPLIT = 0.15
        self.N_REFERENCE_CLIPS = 5
        self.PATIENCE_LR = 5
        self.PATIENCE_EARLY_STOP = 10

class AudioProcessor:
    @staticmethod
    def compute_log_mel(y, sr):
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
        logm = librosa.power_to_db(melspec)
        logm = (logm - logm.mean()) / (logm.std() + 1e-6)
        return torch.from_numpy(logm).unsqueeze(0)

    @staticmethod
    def info_nce_loss(z1, z2, temp):
        B = z1.size(0)
        if B == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / temp
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)
        labels_np = np.concatenate([np.arange(B, 2 * B), np.arange(0, B)])
        labels = torch.from_numpy(labels_np).long().to(sim.device)
        return F.cross_entropy(sim, labels)

class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr, track_ids_for_this_set, id2idx, mode='train'):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.id2idx = id2idx
        self.mode = mode
        self.samples = track_ids_for_this_set
        if not self.samples:
            print(f"Warning: No samples found for mode {self.mode}...")

    def __len__(self):
        return len(self.samples)

    def _load_random_clip(self, track_id):
        path = os.path.join(self.data_dir, f"{track_id}.mp3")
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}. Returning silent clip.")
            return np.zeros(int(self.clip_dur * self.sr), dtype=np.float32)

        total_duration_sec = len(y) / self.sr
        target_len_samples = int(self.clip_dur * self.sr)
        if total_duration_sec > self.clip_dur:
            start_time_sec = random.uniform(0, total_duration_sec - self.clip_dur)
            i0 = int(start_time_sec * self.sr)
            y_clip = y[i0:i0 + target_len_samples]
        else:
            y_clip = np.pad(y, (0, max(0, target_len_samples - len(y))), mode='constant')

        if len(y_clip) != target_len_samples:
            y_clip = np.pad(y_clip, (0, target_len_samples - len(y_clip)), mode='constant')[:target_len_samples]
        return y_clip

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]
        y1 = self._load_random_clip(tid)
        y2 = self._load_random_clip(tid)
        x1 = AudioProcessor.compute_log_mel(y1, self.sr)
        x2 = AudioProcessor.compute_log_mel(y2, self.sr)
        return x1, x2, label

class Encoder(nn.Module):
    def __init__(self, emb_dim, n_classes=0):
        super().__init__()
        # 4层CNN,和basic里不一样哦
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, n_classes) if n_classes > 0 else None

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        logits = self.classifier(z) if self.classifier else None
        return z, logits

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def setup_datasets(self, all_track_ids, id2idx):
        train_ids, val_ids = train_test_split(all_track_ids, test_size=self.config.VALIDATION_SPLIT, random_state=42)
        train_ds = ContrastiveAudioDataset(
            self.config.METADATA_CSV, self.config.DATA_DIR,
            self.config.CLIP_DURATION, self.config.SR,
            train_ids, id2idx, mode='train'
        )
        val_ds = ContrastiveAudioDataset(
            self.config.METADATA_CSV, self.config.DATA_DIR,
            self.config.CLIP_DURATION, self.config.SR,
            val_ids, id2idx, mode='val'
        )
        return train_ds, val_ds

    def setup_model(self, emb_dim, n_classes):
        model = Encoder(emb_dim=emb_dim, n_classes=n_classes).to(self.device)
        return model

    def setup_optimizer(self, model, lr):
        return torch.optim.Adam(model.parameters(), lr=lr)

    def setup_scheduler(self, optimizer):
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=self.config.PATIENCE_LR)

    def evaluate_model(self, model, val_ds):
        if not val_ds or len(val_ds.samples) == 0:
            return 0.0

        model.eval()
        embeddings_db = {}

        # Build reference database
        for track_id in tqdm(val_ds.samples, desc="[Eval] Building DB"):
            clip_embs = []
            for _ in range(self.config.N_REFERENCE_CLIPS):
                y_clip = val_ds._load_random_clip(track_id)
                x_clip = AudioProcessor.compute_log_mel(y_clip, sr=val_ds.sr)
                x_clip = x_clip.unsqueeze(0).to(self.device)
                z, _ = model(x_clip)
                clip_embs.append(z)

            if clip_embs:
                embeddings_db[track_id] = torch.mean(torch.cat(clip_embs, dim=0), dim=0).cpu()

        if not embeddings_db:
            return 0.0

        # Query database
        correct_predictions = 0
        for query_tid in tqdm(val_ds.samples, desc="[Eval] Querying"):
            y_query = val_ds._load_random_clip(query_tid)
            x_query = AudioProcessor.compute_log_mel(y_query, sr=val_ds.sr)
            x_query = x_query.unsqueeze(0).to(self.device)
            zq, _ = model(x_query)
            zq_cpu = zq.cpu()

            sims = {tid: F.cosine_similarity(zq_cpu, emb.unsqueeze(0)).item()
                   for tid, emb in embeddings_db.items()}
            predicted_tid = max(sims, key=sims.get)

            if predicted_tid == query_tid:
                correct_predictions += 1

        return (correct_predictions / len(val_ds.samples)) * 100 if len(val_ds.samples) > 0 else 0.0

    def train_model(self, model, train_loader, val_loader, optimizer, scheduler, config):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        model_saved = False

        for epoch in range(1, config['epochs'] + 1):
            # Training phase
            model.train()
            total_train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Ep {epoch} [Train]", leave=False, ncols=100)

            for x1, x2, labels in train_pbar:
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                z1, log1 = model(x1)
                z2, log2 = model(x2)

                loss_c = AudioProcessor.info_nce_loss(z1, z2, temp=config['temp'])
                loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
                loss = loss_c + config['alpha'] * loss_ce

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

            # Validation phase
            if val_loader and len(val_loader) > 0:
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for x1_val, x2_val, labels_val in val_loader:
                        x1_val, x2_val, labels_val = x1_val.to(self.device), x2_val.to(self.device), labels_val.to(self.device)
                        z1_val, log1_val = model(x1_val)
                        z2_val, log2_val = model(x2_val)
                        loss_c_val = AudioProcessor.info_nce_loss(z1_val, z2_val, temp=config['temp'])
                        loss_ce_val = F.cross_entropy(log1_val, labels_val) + F.cross_entropy(log2_val, labels_val)
                        loss_val = loss_c_val + config['alpha'] * loss_ce_val
                        total_val_loss += loss_val.item()

                current_val_loss = total_val_loss / len(val_loader)
                scheduler.step(current_val_loss)

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), self.config.TEMP_MODEL_PATH)
                    model_saved = True
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.config.PATIENCE_EARLY_STOP:
                        print(f"Early stopping at epoch {epoch}.")
                        break

            val_loss_display = f"{current_val_loss:.4f}" if not np.isinf(current_val_loss) else "N/A"
            print(f"E{epoch}: TrainLoss={avg_train_loss:.4f}, ValLoss={val_loss_display}")

        return best_val_loss, model_saved

class GridSearch:
    def __init__(self, config):
        self.config = config
        self.trainer = ModelTrainer(config)

    def setup_param_grid(self):
        return {
            'lr': [1e-4, 5e-4],
            'temp': [0.07, 0.1],
            'alpha': [0.25, 0.5, 0.75],
            'emb_dim': [128, 256],
            # 'epochs': [5],
            'epochs': [15],
            'batch_size': [32, 64]
        }

    def run_trial(self, config, trial_id, all_track_ids, id2idx):
        print(f"\n--- Starting Trial {trial_id + 1}: {config} ---")

        # Setup datasets
        train_ds, val_ds = self.trainer.setup_datasets(all_track_ids, id2idx)
        if len(train_ds) == 0:
            return float('inf'), 0.0

        # Setup data loaders
        num_workers = min(os.cpu_count() if os.cpu_count() is not None else 1, 4) if torch.cuda.is_available() else 0
        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            drop_last=True, num_workers=num_workers,
            pin_memory=True if self.trainer.device == 'cuda' else False
        )
        val_loader = DataLoader(
            val_ds, batch_size=config['batch_size'], shuffle=False,
            drop_last=False, num_workers=num_workers,
            pin_memory=True if self.trainer.device == 'cuda' else False
        ) if len(val_ds) > 0 else None

        # Setup model and training components
        model = self.trainer.setup_model(config['emb_dim'], len(id2idx))
        optimizer = self.trainer.setup_optimizer(model, config['lr'])
        scheduler = self.trainer.setup_scheduler(optimizer)

        # Train model
        best_val_loss, model_saved = self.trainer.train_model(model, train_loader, val_loader, optimizer, scheduler, config)

        # Evaluate model
        best_val_accuracy = 0.0
        if model_saved:
            print(f"--- Evaluating accuracy for Trial {trial_id+1}'s best model ---")
            best_model = self.trainer.setup_model(config['emb_dim'], len(id2idx))
            best_model.load_state_dict(torch.load(self.config.TEMP_MODEL_PATH))
            best_val_accuracy = self.trainer.evaluate_model(best_model, val_ds)
            print(f"--- Accuracy for Trial {trial_id+1}: {best_val_accuracy:.2f}% ---")

        return best_val_loss, best_val_accuracy

    def run_search(self, all_track_ids, id2idx):
        param_grid = self.setup_param_grid()
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        results = []
        for i, config in enumerate(combinations):
            val_loss, val_accuracy = self.run_trial(config, i, all_track_ids, id2idx)

            result = config.copy()
            result['best_val_loss'] = val_loss
            result['best_val_accuracy'] = val_accuracy
            results.append(result)

            # Save intermediate results
            df = pd.DataFrame(results)
            df.to_csv('gpu_grid_search_results.csv', index=False)

            valid_losses = df['best_val_loss'].replace([np.inf, -np.inf], np.nan).dropna()
            if not valid_losses.empty:
                print(f"Results for trial {i+1}/{len(combinations)} saved. Best val_loss so far: {valid_losses.min():.4f}")

        return results

def main():
    start_time = time.time()
    config = AudioConfig()

    # Validate paths
    if not os.path.isdir(config.DATA_DIR):
        print(f"Error: Data directory '{config.DATA_DIR}' not found.")
        exit()
    if not os.path.isfile(config.METADATA_CSV):
        print(f"Error: Metadata CSV '{config.METADATA_CSV}' not found.")
        exit()

    # Load data
    with open(config.METADATA_CSV, newline='') as f:
        reader = csv.DictReader(f)
        all_track_ids = sorted(list(set(row['track_id'] for row in reader)))

    id2idx = {tid: i for i, tid in enumerate(all_track_ids)}
    if not id2idx:
        raise ValueError("No track IDs found in metadata.csv.")

    with open(config.LABELMAP_PATH, 'wb') as f:
        pickle.dump(id2idx, f)
    print(f"Label map saved. Total unique tracks: {len(all_track_ids)}")

    # Run grid search
    grid_search = GridSearch(config)
    results = grid_search.run_search(all_track_ids, id2idx)

    # Process and save results
    df = pd.DataFrame(results)
    df['best_val_loss'] = df['best_val_loss'].replace([np.inf, -np.inf], np.nan)

    print("\nTop 5 Results (by Validation Loss):")
    print(df.sort_values(by='best_val_loss', ascending=True).head())

    print("\nTop 5 Results (by Validation Accuracy):")
    print(df.sort_values(by='best_val_accuracy', ascending=False).head())

    df.to_csv('gpu_grid_search_results_final.csv', index=False)

    if not df.empty and not df['best_val_loss'].isnull().all():
        best_config = df.sort_values(by='best_val_loss').iloc[0].to_dict()
        print("\nBest Configuration Found (by loss):")
        for key, value in best_config.items():
            print(f"  {key}: {value}")

    total_time = (time.time() - start_time) / 60
    print(f"\nTotal Duration: {total_time:.2f} minutes")

    if os.path.exists(config.TEMP_MODEL_PATH):
        os.remove(config.TEMP_MODEL_PATH)

if __name__ == '__main__':
    main()

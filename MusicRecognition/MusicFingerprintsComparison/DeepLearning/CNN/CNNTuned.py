import csv
import os
import pickle
import random
import time
from dataclasses import dataclass

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class Config:
    # 数据路径
    DATA_DIR = '../../DatasetCreation/audio_1000'
    METADATA_CSV = os.path.join(DATA_DIR, 'metadata_100.csv')
    MODEL_PATH = 'contrastive_model_tuned_best.pt'
    LABELMAP_PATH = 'label_map_tuned.pkl'

    # 音频处理参数
    SR = 22050
    CLIP_DURATION = 5.0
    N_REFERENCE_CLIPS = 5

    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 5
    LR = 1e-4
    TEMP = 0.07
    ALPHA = 0.25
    EMB_DIM = 256

    # 验证和早停参数
    VALIDATION_SPLIT = 0.2
    PATIENCE_LR = 5
    PATIENCE_EARLY_STOP = 10

    # 数据加载参数
    NUM_WORKERS = 4

    # 数据增强参数
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 40


class AudioProcessor:
    @staticmethod
    def compute_log_mel(y, sr=Config.SR):
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64
        )
        logm = librosa.power_to_db(melspec)
        logm = (logm - logm.mean()) / (logm.std() + 1e-6)
        return torch.from_numpy(logm).unsqueeze(0)

    @staticmethod
    def load_random_clip(track_id, data_dir, clip_duration, sr):
        path = os.path.join(data_dir, f"{track_id}.mp3")
        target_len_samples = int(clip_duration * sr)
        try:
            y, loaded_sr = librosa.load(path, sr=sr, mono=True)
            if loaded_sr != sr:
                y = librosa.resample(y, orig_sr=loaded_sr, target_sr=sr)
        except Exception as e:
            print(f"Error loading audio file {path}: {e}. Returning silent clip.")
            return np.zeros(target_len_samples, dtype=np.float32)

        total_duration_sec = len(y) / sr
        if total_duration_sec > clip_duration:
            start_time_sec = random.uniform(0, total_duration_sec - clip_duration)
            i0 = int(start_time_sec * sr)
            i1 = i0 + target_len_samples
            y_clip = y[i0:i1]
        else:
            y_clip = np.pad(y, (0, max(0, target_len_samples - len(y))), mode='constant')

        if len(y_clip) < target_len_samples:
            y_clip = np.pad(y_clip, (0, target_len_samples - len(y_clip)), mode='constant')
        elif len(y_clip) > target_len_samples:
            y_clip = y_clip[:target_len_samples]
        return y_clip


class ContrastiveAudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, clip_duration, sr, track_ids_for_this_set, id2idx, augment=False):
        self.data_dir = data_dir
        self.clip_dur = clip_duration
        self.sr = sr
        self.id2idx = id2idx
        self.augment = augment
        self.processor = AudioProcessor()

        all_samples_temp = []
        try:
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_samples_temp.append(row['track_id'])
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV not found at {csv_file}")

        self.samples = [tid for tid in all_samples_temp if tid in track_ids_for_this_set]

        if not self.samples:
            print(f"Warning: No samples found for the provided track IDs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tid = self.samples[idx]
        label = self.id2idx[tid]

        y1 = self.processor.load_random_clip(tid, self.data_dir, self.clip_dur, self.sr)
        y2 = self.processor.load_random_clip(tid, self.data_dir, self.clip_dur, self.sr)

        x1 = self.processor.compute_log_mel(y1, self.sr)
        x2 = self.processor.compute_log_mel(y2, self.sr)

        if self.augment:
            spec_augment_transforms = nn.Sequential(
                T.FrequencyMasking(freq_mask_param=Config.FREQ_MASK_PARAM),
                T.TimeMasking(time_mask_param=Config.TIME_MASK_PARAM)
            )
            x1 = spec_augment_transforms(x1)
            x2 = spec_augment_transforms(x2)

        return x1, x2, label


class Encoder(nn.Module):
    def __init__(self, emb_dim=Config.EMB_DIM, n_classes=0):
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
        z = F.normalize(self.proj(h), dim=1)
        logits = self.classifier(z) if self.classifier else None
        return z, logits


class ContrastiveTrainer:
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AudioProcessor()
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"PyTorch CUDA version: {torch.version.cuda}")

    def prepare_data(self):
        # 读取所有唯一的track_id
        with open(self.config.METADATA_CSV, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_unique_track_ids = sorted(list(set(row['track_id'] for row in reader)))

        if not all_unique_track_ids:
            raise ValueError("No track IDs found in metadata.csv. Cannot train.")

        # 创建标签映射
        self.id2idx = {tid: i for i, tid in enumerate(all_unique_track_ids)}
        with open(self.config.LABELMAP_PATH, 'wb') as f:
            pickle.dump(self.id2idx, f)
        print(f"Label map saved to {self.config.LABELMAP_PATH} with {len(self.id2idx)} unique tracks.")

        # 划分训练集和验证集
        train_ids, val_ids = train_test_split(
            all_unique_track_ids,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=42,
            stratify=None
        )
        print(f"Total unique tracks: {len(all_unique_track_ids)}")
        print(f"Training tracks (IDs): {len(train_ids)}, Validation tracks (IDs): {len(val_ids)}")

        # 创建数据集
        train_ds = ContrastiveAudioDataset(
            self.config.METADATA_CSV,
            self.config.DATA_DIR,
            self.config.CLIP_DURATION,
            self.config.SR,
            train_ids,
            self.id2idx,
            augment=True
        )
        val_ds = ContrastiveAudioDataset(
            self.config.METADATA_CSV,
            self.config.DATA_DIR,
            self.config.CLIP_DURATION,
            self.config.SR,
            val_ids,
            self.id2idx,
            augment=False
        )

        if len(train_ds) == 0:
            raise ValueError("Training dataset is empty. Check metadata, data directory, and splits.")

        # 创建数据加载器
        pin_memory_flag = True if self.device == 'cuda' else False
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=pin_memory_flag
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=pin_memory_flag
        ) if len(val_ds) > 0 else None

        return train_ids, val_ids

    def train_model(self):
        # 准备数据
        train_ids, val_ids = self.prepare_data()

        # 初始化模型和优化器
        model = Encoder(emb_dim=self.config.EMB_DIM, n_classes=len(self.id2idx)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.LR)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=self.config.PATIENCE_LR
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0

        print(f"Starting training for {self.config.EPOCHS} epochs...")
        for epoch in range(1, self.config.EPOCHS + 1):
            epoch_start_time = time.time()

            # 训练阶段
            model.train()
            total_train_loss = 0.0
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.EPOCHS} [Train]", ncols=100)
            for x1, x2, labels in train_pbar:
                x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                z1, log1 = model(x1)
                z2, log2 = model(x2)

                loss_c = self.info_nce_loss(z1, z2)
                loss_ce = F.cross_entropy(log1, labels) + F.cross_entropy(log2, labels)
                loss = loss_c + self.config.ALPHA * loss_ce

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            avg_train_loss = total_train_loss / len(self.train_loader)

            # 验证阶段
            current_val_loss = float('inf')
            if self.val_loader and len(self.val_loader) > 0:
                model.eval()
                total_val_loss = 0.0
                val_pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config.EPOCHS} [Val]", ncols=100)
                with torch.no_grad():
                    for x1_v, x2_v, labels_v in val_pbar:
                        x1_v, x2_v, labels_v = x1_v.to(self.device), x2_v.to(self.device), labels_v.to(self.device)
                        z1_v, log1_v = model(x1_v)
                        z2_v, log2_v = model(x2_v)

                        loss_c_val = self.info_nce_loss(z1_v, z2_v)
                        loss_ce_val = F.cross_entropy(log1_v, labels_v) + F.cross_entropy(log2_v, labels_v)
                        loss_val = loss_c_val + self.config.ALPHA * loss_ce_val
                        total_val_loss += loss_val.item()
                        val_pbar.set_postfix(loss=f"{loss_val.item():.4f}")
                current_val_loss = total_val_loss / len(self.val_loader)
                scheduler.step(current_val_loss)

                # 保存最佳模型
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    torch.save(model.state_dict(), self.config.MODEL_PATH)
                    print(
                        f"Epoch {epoch}: New best model saved to {self.config.MODEL_PATH} (Val Loss: {best_val_loss:.4f})")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

            # 打印epoch总结
            epoch_duration = time.time() - epoch_start_time
            train_loss_disp = f"{avg_train_loss:.4f}" if not np.isinf(avg_train_loss) else "N/A"
            val_loss_disp = f"{current_val_loss:.4f}" if not np.isinf(current_val_loss) else "N/A"
            print(
                f"Epoch {epoch}/{self.config.EPOCHS} Summary: Train Loss: {train_loss_disp}, Val Loss: {val_loss_disp}, Duration: {epoch_duration:.2f}s")

            # 早停检查
            if epochs_no_improve >= self.config.PATIENCE_EARLY_STOP:
                print(
                    f"Early stopping triggered after {self.config.PATIENCE_EARLY_STOP} epochs with no improvement on validation loss.")
                break

        print("Training finished.")

        # 加载最佳模型
        if os.path.exists(self.config.MODEL_PATH):
            print(f"Loading best model from {self.config.MODEL_PATH} for evaluation.")
            model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.device))
        else:
            print("Warning: Best model path not found. Evaluating with the last model state.")

        return model, val_ids

    def info_nce_loss(self, z1, z2):
        B = z1.size(0)
        if B == 0:
            return torch.tensor(0.0, device=z1.device, requires_grad=True)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.config.TEMP
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -torch.finfo(sim.dtype).max)

        labels_np = np.concatenate([np.arange(B, 2 * B), np.arange(0, B)])
        labels = torch.from_numpy(labels_np).long().to(sim.device)

        return F.cross_entropy(sim, labels)

    def build_track_embeddings(self, model, track_ids_to_embed):
        embeddings_db = {}
        model.eval()

        with torch.no_grad():
            for tid in tqdm(track_ids_to_embed, desc="Building Reference Embeddings"):
                # path = os.path.join(self.config.DATA_DIR, f"{tid}.mp3")
                # target_len_samples = int(self.config.CLIP_DURATION * self.config.SR)
                # try:
                #     y_full, loaded_sr = librosa.load(path, sr=self.config.SR, mono=True)
                #     if loaded_sr != self.config.SR:
                #         y_full = librosa.resample(y_full, orig_sr=loaded_sr, target_sr=self.config.SR)
                # except Exception as e:
                #     print(f"Error loading {path} for embedding: {e}. Skipping.")
                #     continue
                #
                # if len(y_full) == 0:
                #     print(f"Warning: {path} is empty. Skipping.")
                #     continue

                clip_embeddings_list = []
                for _ in range(self.config.N_REFERENCE_CLIPS):
                    y_clip = self.processor.load_random_clip(
                        tid,
                        self.config.DATA_DIR,
                        self.config.CLIP_DURATION,
                        self.config.SR
                    )
                    x = self.processor.compute_log_mel(y_clip, self.config.SR).unsqueeze(0).to(self.device)
                    z, _ = model(x)
                    clip_embeddings_list.append(z)

                if clip_embeddings_list:
                    avg_z = torch.stack(clip_embeddings_list).mean(dim=0)
                    embeddings_db[tid] = avg_z.cpu()
                else:
                    print(f"Could not generate clips for {tid}. Skipping.")
        return embeddings_db

    def evaluate_model(self, model, embeddings_to_search, query_track_ids):
        if not embeddings_to_search:
            print("Embeddings database is empty. Cannot evaluate.")
            return 0.0

        correct = 0
        total_evaluated = 0

        model.eval()
        with torch.no_grad():
            for query_tid in tqdm(query_track_ids, desc="Evaluating Model"):
                y_query_clip = self.processor.load_random_clip(
                    query_tid,
                    self.config.DATA_DIR,
                    self.config.CLIP_DURATION,
                    self.config.SR
                )

                x_query = self.processor.compute_log_mel(y_query_clip, self.config.SR).unsqueeze(0).to(self.device)
                zq_query, _ = model(x_query)
                zq_query_cpu = zq_query.cpu()

                sims = {}
                for db_tid, z_ref_db_cpu in embeddings_to_search.items():
                    sims[db_tid] = torch.cosine_similarity(zq_query_cpu, z_ref_db_cpu).item()

                if not sims: continue

                pred_tid = max(sims, key=sims.get)

                if pred_tid == query_tid:
                    correct += 1
                total_evaluated += 1

        accuracy = 0.0
        if total_evaluated > 0:
            accuracy = correct / total_evaluated
            print(
                f"Top-1 Accuracy on {len(query_track_ids)} query tracks (vs. {len(embeddings_to_search)} in DB): {accuracy:.2%}")
        else:
            print("No tracks were evaluated.")
        return accuracy


def main():
    overall_start_time = time.time()
    config = Config()

    # 检查必要的目录和文件
    if not os.path.isdir(config.DATA_DIR):
        print(f"Error: Data directory '{config.DATA_DIR}' not found. Please check the path.")
        exit()
    if not os.path.isfile(config.METADATA_CSV):
        print(f"Error: Metadata CSV '{config.METADATA_CSV}' not found in '{config.DATA_DIR}'.")
        exit()

    # 设置NUM_WORKERS
    if os.name != 'posix':
        config.NUM_WORKERS = 0
        print(f"Running on non-POSIX OS, setting NUM_WORKERS to 0 for DataLoader.")
    elif torch.cuda.is_available():
        config.NUM_WORKERS = min(os.cpu_count() if os.cpu_count() is not None else 1, 4)
        print(f"CUDA available, setting NUM_WORKERS to {config.NUM_WORKERS}.")
    else:
        config.NUM_WORKERS = min(os.cpu_count() if os.cpu_count() is not None else 1, 4)
        print(f"CUDA not available, setting NUM_WORKERS to {config.NUM_WORKERS}.")

    # 创建训练器实例
    trainer = ContrastiveTrainer(config)

    # 训练模型
    trained_model, validation_ids = trainer.train_model()

    # 构建验证集的嵌入向量
    print("\nBuilding track embeddings for the validation set...")
    validation_embeddings = trainer.build_track_embeddings(trained_model, validation_ids)

    # 评估模型
    if validation_embeddings:
        print(f"\nEvaluating model on the {len(validation_ids)} validation tracks...")
        trainer.evaluate_model(trained_model, validation_embeddings, validation_ids)
    else:
        print("Could not build reference embeddings for the validation set. Skipping evaluation.")

    overall_duration = time.time() - overall_start_time
    print(f"\nScript finished in {overall_duration / 60:.2f} minutes.")


if __name__ == '__main__':
    main()

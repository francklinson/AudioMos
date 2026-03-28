# BaseEvaluator.py
import os
import random
import warnings
from abc import ABC, abstractmethod

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from audiomentations import Compose, AddGaussianNoise
from sklearn.manifold import TSNE
from tqdm import tqdm


class BaseConfig:
    """Base configuration class"""

    def __init__(self):
        self.test_data_dir = None
        self.test_metadata_csv = None
        self.output_dir = None

        # Analysis parameters
        self.snr_levels = [None, 20, 10, 0]
        self.duration_levels = [2, 5, 10]
        self.db_clip_duration = 5.0
        self.t_sne_songs_to_plot = 30
        self.t_sne_clips_per_song = 5
        self.heatmap_songs_to_plot = 4
        self.heatmap_clips_per_song = 3
        self.num_hard_cases_to_show = 10

        self.sr = None
        self.n_mels = None
        self.model_path = None

        # Initialize SNR to amplitude mapping
        self.snr_to_amplitude_map = {
            20: 0.005,
            10: 0.015,
            0: 0.05,
        }


class BaseEvaluator(ABC):
    """Base class for model evaluation"""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.test_metadata = None
        self.test_track_ids = None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    @abstractmethod
    def setup_model(self):
        """Initialize and load the model"""
        pass

    @abstractmethod
    def get_embedding(self, audio_clip: np.ndarray) -> torch.Tensor:
        """Get embedding for an audio clip"""
        pass

    def setup_data(self):
        """Load test metadata and track IDs"""
        try:
            self.test_metadata = pd.read_csv(self.config.test_metadata_csv)
            self.test_track_ids = sorted(self.test_metadata['track_id'].astype(str).unique().tolist())
            print(f"Successfully read test metadata. Found {len(self.test_track_ids)} unique tracks.")
            return True
        except FileNotFoundError:
            print(
                f"ERROR: Test metadata file not found at {self.config.test_metadata_csv}. Please provide a valid test set.")
            return False
        except Exception as e:
            print(f"ERROR: Failed to process test metadata file. Details: {e}")
            return False

    def compute_log_mel(self, y: np.ndarray) -> torch.Tensor:
        """Compute log mel spectrogram."""
        melspec = librosa.feature.melspectrogram(y=y, sr=self.config.sr, n_fft=2048, hop_length=512,
                                                 n_mels=self.config.n_mels)
        logm = librosa.power_to_db(melspec)
        logm = (logm - logm.mean()) / (logm.std() + 1e-6)
        return torch.from_numpy(logm).unsqueeze(0)

    def load_random_clip(self, path: str, duration: float, ) -> np.ndarray:
        """Load a random clip from an audio file"""
        target_len = int(duration * self.config.sr)
        try:
            y, _ = librosa.load(path, sr=self.config.sr, mono=True)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return np.zeros(target_len, dtype=np.float32)

        if len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)), 'constant')
        return y

    def run_robustness_analysis(self):
        """Analyze model robustness to duration and noise"""
        print("\n" + "=" * 80)
        print("ANALYSIS 1: ROBUSTNESS TO DURATION AND NOISE")
        print("=" * 80)

        results = pd.DataFrame(
            index=[f"{snr if snr is not None else 'Clean'} dB" for snr in self.config.snr_levels],
            columns=[f"{dur}s" for dur in self.config.duration_levels]
        )

        print("Building clean reference database...")
        embeddings_db = {}
        for tid in tqdm(self.test_track_ids, desc="Building DB"):
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            clip_embs = [
                self.get_embedding(
                    self.load_random_clip(path, self.config.db_clip_duration, )
                )
                for _ in range(5)
            ]
            embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

        for snr in self.config.snr_levels:
            for duration in self.config.duration_levels:
                print(f"\n-- Evaluating: Duration={duration}s, SNR={snr if snr is not None else 'Clean'} dB --")
                correct_predictions = 0
                for query_tid in tqdm(self.test_track_ids, desc=f"Querying ({duration}s, {snr}dB)"):
                    path = os.path.join(self.config.test_data_dir, f"{query_tid}.mp3")
                    y_query_clean = self.load_random_clip(path, duration, )

                    if snr is not None:
                        amplitude = self.config.snr_to_amplitude_map.get(snr, 0)
                        noise_adder = Compose([
                            AddGaussianNoise(min_amplitude=amplitude, max_amplitude=amplitude, p=1.0)
                        ])
                        y_query_noisy = noise_adder(samples=y_query_clean, sample_rate=self.config.sr)
                    else:
                        y_query_noisy = y_query_clean

                    zq = self.get_embedding(y_query_noisy)
                    sims = {
                        tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item()
                        for tid, emb in embeddings_db.items()
                    }
                    predicted_tid = max(sims, key=sims.get)

                    if predicted_tid == query_tid:
                        correct_predictions += 1

                accuracy = (correct_predictions / len(self.test_track_ids)) * 100 if self.test_track_ids else 0.0
                results.loc[f"{snr if snr is not None else 'Clean'} dB", f"{duration}s"] = f"{accuracy:.2f}%"

        print("\n--- Robustness Analysis Results ---")
        print(results)
        results.to_csv(os.path.join(self.config.output_dir, 'robustness_analysis.csv'))
        print(f"Results table saved to {os.path.join(self.config.output_dir, 'robustness_analysis.csv')}")

    def run_tsne_analysis(self):
        """Generate t-SNE visualization of embedding space"""
        print("\n" + "=" * 80)
        print("ANALYSIS 2: t-SNE VISUALIZATION OF EMBEDDING SPACE")
        print("=" * 80)

        if 'genre' not in self.test_metadata.columns:
            print("WARNING: 'genre' column not found in metadata. Skipping t-SNE plot.")
            return

        print(f"Selecting {self.config.t_sne_songs_to_plot} random songs for visualization...")
        subset_df = self.test_metadata.sample(
            n=min(self.config.t_sne_songs_to_plot, len(self.test_metadata)),
            random_state=42
        )

        embeddings = []
        labels = []
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Generating t-SNE embeddings"):
            tid = row['track_id']
            genre = row['genre']
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            for _ in range(self.config.t_sne_clips_per_song):
                y_clip = self.load_random_clip(path, self.config.db_clip_duration, )
                embeddings.append(self.get_embedding(y_clip))
                labels.append(genre)

        embeddings_cat = torch.cat(embeddings, dim=0).numpy()

        print("Running t-SNE...")
        tsne = TSNE(
            n_components=2,
            verbose=1,
            perplexity=min(30, len(embeddings_cat) - 1),
            random_state=42
        )
        tsne_results = tsne.fit_transform(embeddings_cat)

        plot_df = pd.DataFrame({
            'x': tsne_results[:, 0],
            'y': tsne_results[:, 1],
            'genre': labels
        })

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x="x", y="y",
            hue="genre",
            palette=sns.color_palette("hsv", len(plot_df['genre'].unique())),
            data=plot_df,
            legend="full",
            alpha=0.8
        )
        plt.title('t-SNE Projection of Song Embeddings, Colored by Genre')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

        save_path = os.path.join(self.config.output_dir, 'tsne_genre_visualization.png')
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
        plt.close()

    def run_distance_distribution_analysis(self):
        """Analyze intra-class vs. inter-class similarity"""
        print("\n" + "=" * 80)
        print("ANALYSIS 3: INTRA-CLASS VS. INTER-CLASS SIMILARITY")
        print("=" * 80)
        self.model.eval()
        intra_class_sims = []
        inter_class_sims = []

        print("Generating embeddings for distance analysis...")
        song_embeddings = {}
        for tid in tqdm(self.test_track_ids, desc="Generating embeddings"):
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            clip1 = self.get_embedding(
                self.load_random_clip(path, self.config.db_clip_duration, )
            )
            clip2 = self.get_embedding(
                self.load_random_clip(path, self.config.db_clip_duration, )
            )
            song_embeddings[tid] = (clip1, clip2)

        print("Calculating similarities...")
        track_list = list(song_embeddings.keys())
        for i in range(len(track_list)):
            tid1 = track_list[i]
            z1_1, z1_2 = song_embeddings[tid1]

            intra_class_sims.append(F.cosine_similarity(z1_1, z1_2).item())

            j = i
            while j == i:
                j = random.randint(0, len(track_list) - 1)
            tid2 = track_list[j]
            z2_1, _ = song_embeddings[tid2]
            inter_class_sims.append(F.cosine_similarity(z1_1, z2_1).item())

        plt.figure(figsize=(10, 6))
        sns.histplot(
            intra_class_sims,
            color="blue",
            label='Intra-Class (Same Song)',
            kde=True,
            stat="density",
            element="step"
        )
        sns.histplot(
            inter_class_sims,
            color="red",
            label='Inter-Class (Different Songs)',
            kde=True,
            stat="density",
            element="step"
        )
        plt.title('Distribution of Embedding Similarities')
        plt.xlabel('Cosine Similarity')
        plt.legend()

        save_path = os.path.join(self.config.output_dir, 'similarity_distributions.png')
        plt.savefig(save_path)
        print(f"Distance distribution plot saved to {save_path}")
        plt.close()

    def run_similarity_matrix_analysis(self):
        """Generate similarity matrix heatmap"""
        print("\n" + "=" * 80)
        print("ANALYSIS 4: SIMILARITY MATRIX HEATMAP")
        print("=" * 80)

        if len(self.test_track_ids) < self.config.heatmap_songs_to_plot:
            print("Not enough unique songs in test set to generate heatmap. Skipping.")
            return

        selected_tids = random.sample(self.test_track_ids, self.config.heatmap_songs_to_plot)

        embeddings = []
        labels = []
        for tid in selected_tids:
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            for i in range(self.config.heatmap_clips_per_song):
                y_clip = self.load_random_clip(path, self.config.db_clip_duration, )
                embeddings.append(self.get_embedding(y_clip))
                labels.append(f"{str(tid)[:5]}-{i + 1}")

        embeddings_cat = torch.cat(embeddings, dim=0)
        sim_matrix = F.cosine_similarity(
            embeddings_cat.unsqueeze(1),
            embeddings_cat.unsqueeze(0),
            dim=-1
        )

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            sim_matrix.numpy(),
            xticklabels=labels,
            yticklabels=labels,
            cmap='viridis',
            annot=False
        )
        plt.title('Cosine Similarity Matrix of Sample Clips')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(self.config.output_dir, 'similarity_heatmap.png')
        plt.savefig(save_path)
        print(f"Similarity heatmap saved to {save_path}")
        plt.close()

    def run_hard_mining_analysis(self):
        """Analyze hard cases"""
        print("\n" + "=" * 80)
        print("ANALYSIS 5: HARD-CASE MINING")
        print("=" * 80)

        print("Building reference database for hard-case mining...")
        embeddings_db = {}
        for tid in tqdm(self.test_track_ids, desc="Building DB"):
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            clip_embs = [
                self.get_embedding(
                    self.load_random_clip(path, self.config.db_clip_duration, )
                )
                for _ in range(5)
            ]
            embeddings_db[tid] = torch.mean(torch.cat(clip_embs, dim=0), dim=0)

        print("Finding hardest negative examples...")
        hard_negatives = []
        for query_tid in tqdm(self.test_track_ids, desc="Mining Hard Negatives"):
            y_query = self.load_random_clip(
                os.path.join(self.config.test_data_dir, f"{query_tid}.mp3"),
                self.config.db_clip_duration,
            )
            zq = self.get_embedding(y_query)

            sims = {
                tid: F.cosine_similarity(zq, emb.unsqueeze(0)).item()
                for tid, emb in embeddings_db.items()
            }
            sims.pop(query_tid, None)

            if sims:
                hardest_negative_tid = max(sims, key=sims.get)
                score = sims[hardest_negative_tid]
                hard_negatives.append((query_tid, hardest_negative_tid, score))

        hard_negatives_df = pd.DataFrame(
            hard_negatives,
            columns=['Query Song ID', 'Most Confused With (Impostor)', 'Similarity Score']
        )
        hard_negatives_df = hard_negatives_df.sort_values(
            by='Similarity Score',
            ascending=False
        ).head(self.config.num_hard_cases_to_show)

        print("Finding hardest positive examples...")
        hard_positives = []
        for tid in tqdm(self.test_track_ids, desc="Mining Hard Positives"):
            path = os.path.join(self.config.test_data_dir, f"{tid}.mp3")
            clips = [
                self.load_random_clip(path, self.config.db_clip_duration)
                for _ in range(10)
            ]
            embeddings = [self.get_embedding(c) for c in clips]
            embeddings_cat = torch.cat(embeddings, dim=0)

            sim_matrix = F.cosine_similarity(
                embeddings_cat.unsqueeze(1),
                embeddings_cat.unsqueeze(0),
                dim=-1
            )
            sim_matrix.fill_diagonal_(1.0)

            min_sim_val, _ = torch.min(sim_matrix.view(-1), 0)
            hard_positives.append((tid, min_sim_val.item()))

        hard_positives_df = pd.DataFrame(
            hard_positives,
            columns=['Song ID', 'Lowest Intra-Song Similarity']
        )
        hard_positives_df = hard_positives_df.sort_values(
            by='Lowest Intra-Song Similarity',
            ascending=True
        ).head(self.config.num_hard_cases_to_show)

        print("\nHard Negative Analysis Results (Top Confusions)")
        print(hard_negatives_df)
        hard_negatives_df.to_csv(
            os.path.join(self.config.output_dir, 'hard_negatives_analysis.csv'),
            index=False
        )

        print("\nHard Positive Analysis Results (Most Dissimilar Clips from Same Song)")
        print(hard_positives_df)
        hard_positives_df.to_csv(
            os.path.join(self.config.output_dir, 'hard_positives_analysis.csv'),
            index=False
        )

    def run_all_analyses(self):
        """Run all evaluation analyses"""
        if not self.setup_data():
            print("Data setup failed!")
            return
        self.setup_model()

        self.run_robustness_analysis()
        self.run_tsne_analysis()
        self.run_distance_distribution_analysis()
        self.run_similarity_matrix_analysis()
        self.run_hard_mining_analysis()

        print("\n\nComprehensive analysis finished!")

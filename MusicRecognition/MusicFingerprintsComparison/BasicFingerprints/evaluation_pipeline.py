"""评估算法性能"""
import csv
import hashlib
import os
import pickle

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm


class AudioFingerprinter:
    def __init__(self):
        self.peak_neighborhood_size = 20
        self.fan_value = 50
        self.window_size = 4096
        self.overlap_ratio = 0.5
        self.min_hash_time_delta = 0
        self.max_hash_time_delta = 200
        self.energy_threshold_ratio = 0.3
        self.default_fs = 22050

    def stable_hash(self, f1, f2, dt):
        key = f"{f1}|{f2}|{dt}".encode("utf-8")
        digest = hashlib.sha1(key).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def get_anchors(self, S):
        fsz = self.peak_neighborhood_size
        footprint = np.ones((fsz, fsz))
        local_max = maximum_filter(S, footprint=footprint) == S
        background = (S == 0)
        eroded = maximum_filter(background, footprint=footprint)
        return np.argwhere(local_max & ~eroded)

    def fingerprint(self, y, sr=None):
        if sr is None:
            sr = self.default_fs
        hop = int(self.window_size * self.overlap_ratio)
        S = np.abs(librosa.stft(y, n_fft=self.window_size, hop_length=hop))
        energy = np.sum(S ** 2, axis=0)
        thresh = np.median(energy) * self.energy_threshold_ratio

        anchors = [(f, t) for f, t in self.get_anchors(S) if energy[t] >= thresh]

        hashes = []
        for i, (f1, t1) in enumerate(anchors):
            for j in range(1, self.fan_value):
                if i + j < len(anchors):
                    f2, t2 = anchors[i + j]
                    dt = t2 - t1
                    if self.min_hash_time_delta <= dt <= self.max_hash_time_delta:
                        h = self.stable_hash(f1, f2, dt)
                        hashes.append((h, t1))
        return hashes


class AudioIndex:
    def __init__(self, path=None):
        self.index = {}
        if path:
            self.load(path)

    def load(self, path):
        with open(path, "rb") as f:
            self.index = pickle.load(f)

    def get(self, key, default=None):
        return self.index.get(key, default)


class AudioEvaluator:
    def __init__(self, audio_dir, metadata_path):
        self.audio_dir = audio_dir
        self.metadata_path = metadata_path
        self.fingerprinter = AudioFingerprinter()
        self.rows = list(csv.DictReader(open(metadata_path, newline="", encoding="utf-8")))

    def recognize_scores(self, clip, index):
        votes = {}
        for h, offs in self.fingerprinter.fingerprint(clip):
            for tid, db_offs in index.get(h, []):
                dt = db_offs - offs
                votes[(tid, dt)] = votes.get((tid, dt), 0) + 1
        track_scores = {}
        for (tid, _), v in votes.items():
            track_scores[tid] = max(track_scores.get(tid, 0), v)
        return track_scores

    def process_audio_clip(self, tid, clip_dur, snr_db):
        path = os.path.join(self.audio_dir, f"{tid}.mp3")
        try:
            y, sr = librosa.load(path, sr=self.fingerprinter.default_fs, mono=True)
        except FileNotFoundError:
            print(f"File {path} not found, skipping")
            return None

        dur = len(y) / sr
        if dur > clip_dur:
            start = np.random.uniform(0, dur - clip_dur)
            i0, i1 = int(start * sr), int((start + clip_dur) * sr)
            clip = y[i0:i1]
        else:
            clip = y

        if snr_db >= 0:
            sig_pow = np.mean(clip ** 2)
            noise_pow = sig_pow / (10 ** (snr_db / 10))
            noise = np.random.randn(len(clip))
            noise = noise * np.sqrt(noise_pow) / np.std(noise)
            clip = clip + noise

        return clip

    def evaluate_top_n(self, index, clip_dur, snr_db, top_n):
        hits = 0
        total = 0
        for r in tqdm(self.rows, desc=f"{clip_dur}s @ {snr_db}dB (Top-{top_n})"):
            tid = int(r["track_id"])
            clip = self.process_audio_clip(tid, clip_dur, snr_db)
            if clip is None:
                continue

            scores = self.recognize_scores(clip, index)
            ranked = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
            if tid in ranked[:top_n]:
                hits += 1
            total += 1

        return hits / total if total > 0 else 0

    def compute_cmc(self, index, clip_dur, snr_db, max_rank=10):
        cmc_counts = np.zeros(max_rank, dtype=int)
        total = 0

        for r in tqdm(self.rows, desc=f"CMC {clip_dur}s @ {snr_db}dB"):
            tid = int(r["track_id"])
            clip = self.process_audio_clip(tid, clip_dur, snr_db)
            if clip is None:
                continue

            scores = self.recognize_scores(clip, index)
            ranked = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

            for k in range(1, max_rank + 1):
                if tid in ranked[:k]:
                    cmc_counts[k - 1] += 1
            total += 1

        return cmc_counts / total if total > 0 else np.zeros(max_rank)


class AudioVisualizer:
    @staticmethod
    def plot_accuracy_vs_duration(results, top_n, durations, snr_dbs):
        plt.figure()
        for snr in snr_dbs:
            ys = [results[(top_n, d, snr)] for d in durations]
            plt.plot(durations, ys, marker='o', label=f"{snr} dB")
        plt.xlabel("Clip duration (s)")
        plt.ylabel(f"Top-{top_n} accuracy")
        plt.title(f"Accuracy vs. Clip Length (Top-{top_n})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_vs_clip_top{top_n}.png", dpi=200)

    @staticmethod
    def plot_accuracy_vs_noise(results, top_n, durations, snr_dbs):
        plt.figure()
        for dur in durations:
            ys = [results[(top_n, dur, s)] for s in snr_dbs]
            plt.plot(snr_dbs, ys, marker='o', label=f"{dur}s")
        plt.xlabel("Noise level (SNR in dB)")
        plt.ylabel(f"Top-{top_n} accuracy")
        plt.title(f"Accuracy vs. Noise Level (Top-{top_n})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"accuracy_vs_noise_top{top_n}.png", dpi=200)

    @staticmethod
    def plot_accuracy_heatmap(results, top_n, durations, snr_dbs):
        data = np.array([[results[(top_n, d, s)] for s in snr_dbs] for d in durations])
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(len(snr_dbs)))
        ax.set_yticks(np.arange(len(durations)))
        ax.set_xticklabels([f"{s}dB" for s in snr_dbs])
        ax.set_yticklabels([f"{d}s" for d in durations])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j] * 100:4.1f}%", ha="center", va="center", color="w")
        ax.set_xlabel("Noise (dB SNR)")
        ax.set_ylabel("Clip duration (s)")
        ax.set_title(f"Top-{top_n} Accuracy Heatmap")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"accuracy_heatmap_top{top_n}.png", dpi=200)

    @staticmethod
    def plot_cmc_curve(cmc, max_rank):
        plt.figure()
        ranks = np.arange(1, max_rank + 1)
        plt.plot(ranks, cmc, marker='o')
        plt.xticks(ranks)
        plt.xlabel("Rank n")
        plt.ylabel("Recognition rate")
        plt.title("CMC Curve (5 s, 0 dB)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cmc_curve.png", dpi=200)


def print_results_table(results, top_n, durations, snr_dbs):
    print(f"\n**Top-{top_n} Accuracy Table**")
    header = ["SNR \\ Dur"] + [f"{d}s" for d in durations]
    print("| " + " | ".join(header) + " |")
    print("|" + "------|" * len(header))
    for snr in snr_dbs:
        row = [f"{snr}dB"] + [f"{results[(top_n, d, snr)] * 100:5.2f}%" for d in durations]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    # Constants
    AUDIO_DIR = "../DatasetCreation/audio_1000"
    METADATA_CSV = os.path.join(AUDIO_DIR, "metadata.csv")
    INDEX_PATH = "fingerprints.pkl"
    DURATIONS = [2.0, 5.0, 10.0]
    SNR_DBS = [0, 10, 20]
    TOP_NS = [1, 5]
    CMC_MAX_RANK = 10

    # Initialize components
    index = AudioIndex(INDEX_PATH)
    evaluator = AudioEvaluator(AUDIO_DIR, METADATA_CSV)
    visualizer = AudioVisualizer()

    # Run evaluation
    results = {}
    for top_n in TOP_NS:
        for dur in DURATIONS:
            for snr in SNR_DBS:
                acc = evaluator.evaluate_top_n(index, dur, snr, top_n)
                results[(top_n, dur, snr)] = acc
                print(f"→ Top-{top_n}: {dur}s @ {snr}dB → {acc:.2%}")

    # Print results tables
    for top_n in TOP_NS:
        print_results_table(results, top_n, DURATIONS, SNR_DBS)

    # Generate plots
    for top_n in TOP_NS:
        visualizer.plot_accuracy_vs_duration(results, top_n, DURATIONS, SNR_DBS)
        visualizer.plot_accuracy_vs_noise(results, top_n, DURATIONS, SNR_DBS)
        visualizer.plot_accuracy_heatmap(results, top_n, DURATIONS, SNR_DBS)

    # Generate CMC curve
    cmc = evaluator.compute_cmc(index, clip_dur=5.0, snr_db=0, max_rank=CMC_MAX_RANK)
    visualizer.plot_cmc_curve(cmc, CMC_MAX_RANK)
    print("\nPlots saved")

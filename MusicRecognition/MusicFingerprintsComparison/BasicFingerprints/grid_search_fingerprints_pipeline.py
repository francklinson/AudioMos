"""评估不同配置参数下的算法性能"""
import os
import csv
import time
import pickle
import hashlib
from itertools import product

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class AudioFingerprinter:
    @staticmethod
    def stable_hash(f1, f2, dt):
        key = f"{f1}|{f2}|{dt}".encode("utf-8")
        digest = hashlib.sha1(key).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    @staticmethod
    def get_anchors(S, peak_size):
        footprint = np.ones((peak_size, peak_size))
        local_max = maximum_filter(S, footprint=footprint) == S
        background = (S == 0)
        eroded_bg = maximum_filter(background, footprint=footprint)
        return np.argwhere(local_max & ~eroded_bg)

    def __init__(self, params):
        self.params = params

    def process(self, y):
        hop_length = int(self.params['WINDOW_SIZE'] * self.params['OVERLAP_RATIO'])
        S = np.abs(librosa.stft(y,
                             n_fft=self.params['WINDOW_SIZE'],
                             hop_length=hop_length))

        energy = np.sum(S**2, axis=0)
        threshold = np.median(energy) * self.params['ENERGY_THRESHOLD_RATIO']

        anchors = self.get_anchors(S, self.params['PEAK_NEIGHBORHOOD_SIZE'])
        anchors = [(f, t) for f, t in anchors if energy[t] >= threshold]

        hashes = []
        for i, (f1, t1) in enumerate(anchors):
            for j in range(1, self.params['FAN_VALUE']):
                if i + j < len(anchors):
                    f2, t2 = anchors[i + j]
                    dt = t2 - t1
                    if self.params['MIN_HASH_TIME_DELTA'] <= dt <= self.params['MAX_HASH_TIME_DELTA']:
                        h = self.stable_hash(f1, f2, dt)
                        hashes.append((h, t1))
        return hashes


class IndexBuilder:
    def __init__(self, audio_dir, metadata_csv, params):
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.fingerprinter = AudioFingerprinter(params)
        self.index = {}

    def build(self):
        with open(self.metadata_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Indexing"):
                track_id = int(row['track_id'])
                path = os.path.join(self.audio_dir, f"{track_id}.mp3")
                if not os.path.isfile(path):
                    continue
                y, sr = librosa.load(path, sr=self.fingerprinter.params.get('DEFAULT_FS', 22050), mono=True)
                for h, offs in self.fingerprinter.process(y):
                    self.index.setdefault(h, []).append((track_id, offs))
        return self.index


class Evaluator:
    def __init__(self, audio_dir, metadata_csv, params):
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.fingerprinter = AudioFingerprinter(params)

    def evaluate(self, index, clip_duration=5.0, noise_level=0.0, top_n=1):
        with open(self.metadata_csv, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        hits = 0
        total = len(rows)
        t_start = time.time()

        for row in tqdm(rows, desc="Evaluating"):
            tid = int(row['track_id'])
            path = os.path.join(self.audio_dir, f"{tid}.mp3")
            try:
                y, sr = librosa.load(path, sr=self.fingerprinter.params.get('DEFAULT_FS', 22050), mono=True)
            except FileNotFoundError:
                continue
            dur = len(y) / sr

            if dur > clip_duration:
                start = np.random.uniform(0, dur - clip_duration)
                clip = y[int(start * sr):int((start + clip_duration) * sr)]
            else:
                clip = y

            if noise_level > 0:
                noise = np.random.randn(len(clip))
                clip += noise_level * noise / np.std(noise)

            votes = {}
            for h, offs in self.fingerprinter.process(clip):
                for track_id, db_offs in index.get(h, []):
                    dt = db_offs - offs
                    votes[(track_id, dt)] = votes.get((track_id, dt), 0) + 1

            track_scores = {}
            for (track_id, _), v in votes.items():
                track_scores[track_id] = max(track_scores.get(track_id, 0), v)

            top = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)
            preds = [tid_score for tid_score, _ in top[:top_n]]

            if preds and tid in preds:
                hits += 1

        latency = (time.time() - t_start) / total
        accuracy = hits / total
        return accuracy, latency


class GridSearch:
    def __init__(self, audio_dir, metadata_csv, base_params, param_grid):
        self.audio_dir = audio_dir
        self.metadata_csv = metadata_csv
        self.base_params = base_params
        self.param_grid = param_grid
        self.results = []

    def run(self, top_n=1):
        print("Starting grid search over {} configurations...".format(
            np.prod([len(v) for v in self.param_grid.values()])))

        for pns, fan, eth in product(
                self.param_grid['PEAK_NEIGHBORHOOD_SIZE'],
                self.param_grid['FAN_VALUE'],
                self.param_grid['ENERGY_THRESHOLD_RATIO']):
            params = dict(self.base_params)
            params.update({
                'PEAK_NEIGHBORHOOD_SIZE': pns,
                'FAN_VALUE': fan,
                'ENERGY_THRESHOLD_RATIO': eth,
            })
            label = f"PNS={pns}_FAN={fan}_ETH={eth}"
            print(f"\n--- Testing {label} ---")

            t0 = time.time()
            builder = IndexBuilder(self.audio_dir, self.metadata_csv, params)
            index = builder.build()
            print(f"Index built ({len(index)} hashes) in {time.time() - t0:.1f}s")

            evaluator = Evaluator(self.audio_dir, self.metadata_csv, params)
            acc, lat = evaluator.evaluate(index, top_n=top_n)
            print(f"-> Top-1 Accuracy: {acc:.2%}, Latency: {lat:.3f}s/clip")
            self.results.append((label, pns, fan, eth, acc, lat))

        best = max(self.results, key=lambda x: x[4])
        print(f"\nBest config: {best[0]} with Top-1 Accuracy {best[4]:.2%}")

        out_csv = 'grid_search_results.csv'
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['label', 'PNS', 'FAN', 'ETH', 'accuracy', 'latency'])
            writer.writerows(self.results)
        print(f"Results saved to {out_csv}")


if __name__ == '__main__':
    AUDIO_DIR = "../DatasetCreation/audio_1000"
    METADATA_CSV = os.path.join(AUDIO_DIR, 'metadata.csv')

    base_params = {
        'WINDOW_SIZE': 4096,
        'OVERLAP_RATIO': 0.5,
        'MIN_HASH_TIME_DELTA': 0,
        'MAX_HASH_TIME_DELTA': 200,
        'DEFAULT_FS': 22050,
    }

    grid = {
        'PEAK_NEIGHBORHOOD_SIZE': [20, 30, 40],
        'FAN_VALUE': [15, 30, 50],
        'ENERGY_THRESHOLD_RATIO': [0.3, 0.5, 0.7],
    }

    grid_search = GridSearch(AUDIO_DIR, METADATA_CSV, base_params, grid)
    grid_search.run(top_n=1)

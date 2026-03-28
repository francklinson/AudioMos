import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

from MERTAttentionPooling import AttentionPooling
from MusicFingerprintsComparison.DeepLearning.utils.BaseEvaluator import BaseEvaluator, BaseConfig

warnings.filterwarnings("ignore")


class MERTAttentionPoolingConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.model_path = 'MERT_Results/finetuned_mert_attention_model.pt'
        self.labelmap_path = 'MERT_Results/finetuned_mert_attention_label_map.pkl'
        self.test_data_dir = '../../DatasetCreation/TEST_METADATA/downloaded_tracks'
        self.test_metadata_csv = '../../DatasetCreation/TEST_METADATA/metadata_test_20.csv'
        self.output_dir = 'FINAL_ANALYSIS_RESULTS_MERT_WithAttention'
        self.sr = 24000
        self.clip_duration = 5.0
        self.emb_dim = 256
        self.model_type = "MERT"
        self.model_hf_path = "/home/zhouchenghao/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/snapshots/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"

        os.makedirs(self.output_dir, exist_ok=True)


class MERTForSongID(nn.Module):
    def __init__(self, model_name="m-a-p/MERT-v1-95M", emb_dim=256, **kwargs):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        mert_embedding_size = self.base_model.config.hidden_size

        self.pooling = AttentionPooling(in_features=mert_embedding_size)
        self.proj_head = nn.Sequential(
            nn.Linear(mert_embedding_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state
        mert_embedding = self.pooling(hidden_states)

        z_unnorm = self.proj_head(mert_embedding)
        return F.normalize(z_unnorm, p=2, dim=1)


class MERTEvaluator(BaseEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.feature_extractor = None
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

    def setup_model(self):
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_hf_path)
            self.model = MERTForSongID(
                model_name=self.config.model_hf_path,
                emb_dim=256, )
            full_state_dict = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(full_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"ERROR: Failed to load model from {self.config.model_path}. {e}")
            return False
        return True

    @torch.no_grad()
    def get_embedding(self, audio_clip):
        inputs = self.feature_extractor(audio_clip, sampling_rate=self.config.sr, return_tensors="pt", padding=True).to(
            self.device)
        embedding = self.model(inputs['input_values'])
        return embedding.cpu()


def main():
    config = MERTAttentionPoolingConfig()
    evaluator = MERTEvaluator(config)
    evaluator.run_all_analyses()


if __name__ == "__main__":
    main()

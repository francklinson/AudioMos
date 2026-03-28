import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoFeatureExtractor

from MusicFingerprintsComparison.DeepLearning.utils.BaseEvaluator import BaseEvaluator, BaseConfig

warnings.filterwarnings("ignore")


@dataclass
class ASTConfig(BaseConfig):
    """Configuration class for model evaluation"""

    def __init__(self):
        super().__init__()
        self.model_path = 'AST_Results/finetuned_ast_model.pt'
        self.labelmap_path = 'AST_Results/finetuned_ast_label_map.pkl'
        self.test_data_dir = '../../DatasetCreation/TEST_METADATA/downloaded_tracks'
        self.test_metadata_csv = '../../DatasetCreation/TEST_METADATA/metadata_test_20.csv'
        self.output_dir = 'FINAL_ANALYSIS_RESULTS_AST_NoAttention'
        self.sr = 22050
        self.clip_duration = 5.0
        self.emb_dim = 128
        self.n_mels = 64
        self.model_type: str = "AST"
        # model_hf_path:str = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.model_hf_path: str = "/home/zhouchenghao/.cache/huggingface/hub/models--MIT--ast-finetuned-audioset-10-10-0.4593/snapshots/f826b80d28226b62986cc218e5cec390b1096902"

        os.makedirs(self.output_dir, exist_ok=True)


class ASTForSongID(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", emb_dim=256, **kwargs):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name)
        ast_embedding_size = self.base_model.config.hidden_size

        self.proj_head = nn.Sequential(
            nn.Linear(ast_embedding_size, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        hidden_states = self.base_model(x).last_hidden_state

        ast_embedding = torch.mean(hidden_states, dim=1)

        z_unnorm = self.proj_head(ast_embedding)
        return F.normalize(z_unnorm, p=2, dim=1)


class ASTEvaluator(BaseEvaluator):
    """Handles model evaluation and analysis"""

    def __init__(self, config: ASTConfig):
        super().__init__(config)
        self.feature_extractor = None

    def setup_model(self):
        """Initialize and load the model"""

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_hf_path)

        self.model = ASTForSongID(
            model_name=self.config.model_hf_path,
            emb_dim=256, )

        full_state_dict = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(full_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, audio_clip: np.ndarray) -> torch.Tensor:
        """Get embedding for an audio clip"""
        inputs = self.feature_extractor(
            audio_clip,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        embedding = self.model(inputs['input_values'])
        return embedding.cpu()


def main():
    config = ASTConfig()
    evaluator = ASTEvaluator(config)
    evaluator.run_all_analyses()


if __name__ == "__main__":
    main()

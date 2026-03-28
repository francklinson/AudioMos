import os
import warnings

import numpy as np
import torch

from CNNBasic import Encoder
from MusicFingerprintsComparison.DeepLearning.utils.BaseEvaluator import BaseEvaluator, BaseConfig

warnings.filterwarnings("ignore")


class CNNBasicConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.model_path = 'contrastive_model_basic.pt'
        self.labelmap_path = 'label_map.pkl'
        self.test_data_dir = '../../DatasetCreation/TEST_METADATA/downloaded_tracks'
        self.test_metadata_csv = '../../DatasetCreation/TEST_METADATA/metadata_test_20.csv'
        self.output_dir = 'FINAL_BASIC_CNN_ANALYSIS_RESULTS'
        self.sr = 22050
        self.clip_duration = 5.0
        self.n_mels = 64


class CNNBasicEvaluator(BaseEvaluator):
    def __init__(self, config: CNNBasicConfig):
        super().__init__(config)

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def setup_model(self) -> bool:
        """Load the trained model."""
        try:
            print(f"Loading trained CNN model...")
            self.model = Encoder(emb_dim=128, n_classes=len(self.test_track_ids))

            full_state_dict = torch.load(self.config.model_path, map_location=self.device)
            filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('classifier')}

            self.model.load_state_dict(filtered_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model from {self.config.model_path}. Details: {e}")
            return False

    @torch.no_grad()
    def get_embedding(self, audio_clip: np.ndarray) -> torch.Tensor:
        """Get embedding for an audio clip."""
        mel_spec = self.compute_log_mel(audio_clip).unsqueeze(0).to(self.device)
        embedding, _ = self.model(mel_spec)
        return embedding.cpu()


def main():
    # Configuration
    config = CNNBasicConfig()
    analyzer = CNNBasicEvaluator(config)
    analyzer.run_all_analyses()


if __name__ == "__main__":
    main()

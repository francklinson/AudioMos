"""
使用方式：
source  /mnt/test/scripts/label-studio-zch/mos-env/bin/active
cd /mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples
label-studio-ml start mos

"""
import os
import sys
from tqdm import tqdm
import soundfile as sf
import librosa
import numpy as np
import onnxruntime as ort
import pandas as pd
import warnings
import logging
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from typing import List, Dict, Optional
import wenet
from wenet.wer import wer
from nisqa.predict import nisqa_predict


class DNSMOScore:
    def __init__(self) -> None:
        p808_model_path = "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/DNSMOS/DNSMOS/model_v8.onnx"
        primary_model_path = "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/voice_mos/DNSMOS/pDNSMOS/sig_bak_ovr.onnx"
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        self.INPUT_LENGTH = 9.01
        self.SAMPLING_RATE = 16000
        self.p808_model_path = os.path.join('DNSMOS', 'DNSMOS', 'model_v8.onnx')

    @staticmethod
    def __audio_melspec(audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length,
                                                  n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    @staticmethod
    def __get_polyfit_val(sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
            p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
            p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])

        # print(sig)
        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __get_score(self, fpath, is_personalized_MOS=None):
        aud, input_fs = sf.read(fpath)
        # 双声道只取第一个声道
        if len(aud.shape) > 1:
            # 方法1：平均所有声道
            # mono_data = np.mean(aud, axis=1)
            # 方法2：只取第一个声道
            aud = aud[:, 0]
        fs = self.SAMPLING_RATE
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + self.INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            p808_input_features = np.array(self.__audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis,
            :,
            :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw,
                                                               is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)
        clip_dict = {'OVRL': np.mean(predicted_mos_ovr_seg), 'SIG': np.mean(predicted_mos_sig_seg),
                     'BAK': np.mean(predicted_mos_bak_seg), 'P808_MOS': np.mean(predicted_p808_mos)}
        return clip_dict

    def get_mos(self, audio_file):
        if not audio_file.endswith(".wav"):
            return {}
        data = self.__get_score(audio_file)
        res = {'OVRL': data['OVRL'], 'SIG': data['SIG'], 'BAK': data['BAK'], 'P808_MOS': data['P808_MOS']}
        return res


class NisqaMos:
    def __init__(self):
        self.nisqa_mode = "predict_file"
        self.nisqa_model = 'nisqa_3000.tar'

    def get_mos(self, audio_file) -> dict:
        nisqa_prediction: pd.DataFrame = nisqa_predict(mode=self.nisqa_mode, deg=audio_file,
                                                       model=self.nisqa_model)
        ret = nisqa_prediction.to_dict()
        ret.pop("deg")
        for k, v in ret.items():
            ret[k] = v[0]
        return ret


dnsmos_calc_ins = DNSMOScore()
nisqa_calc_ins = NisqaMos()


class MosModel(LabelStudioMLBase):
    """Custom ML Backend model
    """

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """

        print(f"Run prediction on {tasks}")
        print(f"Received context: {context}")
        print(f"Project ID: {self.project_id}")
        print(f"Label config: {self.label_config}")
        print(f"Parsed JSON Label config: {self.parsed_label_config}")
        print(f"Extra params: {self.extra_params}")

        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])
        try:
            path = self.get_local_path(tasks[0]['data']['audio'], task_id=tasks[0]['id'])
            print("解析地址成功! ", path)
        except Exception as e:
            print('获取音频地址失败: %s', e)
            return ModelResponse(predictions=[])
        # 初始化一个空字典用于存储所有评分结果
        data = {}

        # 计算并添加DNSMOS评分
        dnsmos_scores = dnsmos_calc_ins.get_mos(path)
        data.update(dnsmos_scores)

        # 计算并添加NISQA维度评分
        nisqa_dim_scores = nisqa_calc_ins.get_mos(path)
        data.update(nisqa_dim_scores)

        avg_mos = 0
        for v in data.values():
            avg_mos += v

        pred = 0 if not data else 2 * avg_mos / len(data)
        predictions = [{
            "model_version": "0.0.1",
            "result": [{'value': {'rating': pred},
                        'from_name': 'rating',
                        'to_name': 'audio',
                        'type': 'rating', }]
        }]
        print("*" * 10, "Predictions", "*" * 10)
        return ModelResponse(predictions=[predictions])

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')
        print('fit() completed successfully.')

import os
from errno import ELOOP
from typing import List, Dict, Optional

import librosa
import torchaudio
from audiobox_aesthetics.infer import initialize_predictor
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from music_recognition import predict_music
from utils.PEAQ_utils import PEAQ
from utils.hparam import hp

# TODO 批处理
aesmos_predictor = initialize_predictor(
    ckpt="/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/music_mos/pretrained_models/checkpoint.pt")
peaq = PEAQ()


class MusicMosModel(LabelStudioMLBase):
    """
    Custom ML Backend model
    """

    def setup(self):
        """
        Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict_peaq(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        # print(f'''\
        # Run prediction on {tasks}
        # Received context: {context}
        # Project ID: {self.project_id}
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}
        # Extra params: {self.extra_params}''')
        # 拿到音频路径
        test_music_path = self.get_local_path(tasks[0]['data']['audio'], task_id=tasks[0]['id'])

        # 先判断是哪首歌，获得歌曲名称和偏移量
        pre_ret_map = predict_music(test_music_path)
        music_name = pre_ret_map[test_music_path]["music_name"]
        if music_name is None:
            print("No proper source music found!!!!!")
            return
        # 找到对应的参考音频源文件路径
        ref_music_path = os.path.join(hp.fingerprint.path.music_path, music_name)

        music_offset = pre_ret_map[test_music_path]["music_offset"]
        # 偏移量 * 64ms ，即为和参考音频之间相差的时间
        # 偏移量为正，说明测试音频在参考音频之前；偏移量为负，说明测试音频在参考音频之后。
        t_delta = music_offset * 0.064  # 单位是s，帧间隔64ms

        # 从测试音频和参考音频中都切出10s进行peaq计算
        test_slice_start_time, ref_slice_start_time = 0.0, 0.0
        if music_offset > 0:
            print(f"test faster than ref {t_delta}s")
            test_slice_start_time = 0.0
            ref_slice_start_time = t_delta
        elif music_offset < 0:
            print(f"test slower than ref {-t_delta}s")
            test_slice_start_time = -t_delta
            ref_slice_start_time = 0.0
        elif music_offset == 0:
            print(f"test && ref align!")
            test_slice_start_time, ref_slice_start_time = 0.0, 0.0

        # 切片
        print("ref music src path:", ref_music_path)
        ref_music_slice, _ = librosa.load(str(ref_music_path), sr=48000, offset=ref_slice_start_time, duration=10.0)
        test_music_slice, _ = librosa.load(test_music_path, sr=48000, offset=test_slice_start_time, duration=10.0)
        # print(len(ref_music_slice))
        # print(len(test_music_slice))

        ref_music_slice = ref_music_slice.squeeze()
        test_music_slice = test_music_slice.squeeze()
        # 幅值归一化
        ref_max_val = max(abs(ref_music_slice))
        test_max_val = max(abs(test_music_slice))
        max_val = max(ref_max_val, test_max_val)

        ref_music_slice = ref_music_slice * 32768.0 * max_val / ref_max_val
        test_music_slice = test_music_slice * 32768.0 * max_val / test_max_val
        peaq.process(ref_music_slice, test_music_slice)
        odg = peaq.avg_get()["ODG"]
        # print("odg", odg)
        score = min(10 + 2 * odg, 10)
        # print("score", score)
        predictions = [{
            "model_version": "0.0.1",
            "result": [{'value': {'rating': score},
                        'from_name': 'rating',
                        'to_name': 'audio',
                        'type': 'rating', }]
        }]
        return ModelResponse(predictions=[predictions])

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')
        # 拿到音频路径
        test_music_path = self.get_local_path(tasks[0]['data']['audio'], task_id=tasks[0]['id'])

        # 先判断是哪首歌，获得歌曲名称和偏移量
        try:
            pre_ret_map = predict_music(test_music_path)
            music_name = pre_ret_map[test_music_path]["music_name"]
            max_hash_count = pre_ret_map[test_music_path]["max_hash_count"]
        except Exception as e:
            print(f'No proper source music found!!!!!  {str(e)}')
            return ModelResponse(predictions=[])

        # 匹配到的特征太少了
        if max_hash_count < 10:
            print("Not match music found in database! Audio slice start time set to default 0s!")
            music_offset = 0
        else:
            music_offset = pre_ret_map[test_music_path]["music_offset"]
        # 偏移量 * 64ms ，即为和参考音频之间相差的时间
        # 偏移量为正，说明测试音频在参考音频之前；偏移量为负，说明测试音频在参考音频之后。
        t_delta = abs(music_offset) * 0.064  # 单位是s，帧间隔64ms
        if music_offset > 0:
            print(f"test faster than ref {t_delta}s")
            t_delta = 0
        elif music_offset < 0:
            print(f"test slower than ref {t_delta}s")

        # 从测试音频中切出15s进行aesmos计算
        metadata = torchaudio.info(test_music_path)
        sample_rate = metadata.sample_rate
        test_music_slice, _ = torchaudio.load(uri=test_music_path,
                                              frame_offset=int(sample_rate * t_delta),
                                              num_frames=int(sample_rate * 15))
        # print(len(test_music_slice[0])) # 确保时间长度
        ret = aesmos_predictor.forward([{"path": test_music_slice, "sample_rate": sample_rate}, ])
        CE, CU, PC, PQ = ret[0]["CE"], ret[0]["CU"], ret[0]["PC"], ret[0]["PQ"]
        print('*' * 20)
        print(f"CE: {CE}\n")
        print(f"CU: {CU}\n")
        print(f"PC: {PC}\n")
        print(f"PQ: {PQ}\n")
        print('*' * 20)
        score = (CE + CU + PC + PQ) / 4
        # 3~8的分布，扩展到0~10
        score = 2 * score - 6
        score = min(score, 10)
        score = max(0, score)
        score = int(score + 0.5)  # 四舍五入
        # 四舍五入
        predictions = [{
            "model_version": "0.0.1",
            "result": [{'value': {'rating': score},
                        'from_name': 'rating',
                        'to_name': 'audio',
                        'type': 'rating', }]
        }]
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

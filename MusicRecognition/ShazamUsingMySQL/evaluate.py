# 从音频中随机截取片段（1s、5s、10s），加上噪声（0dB、10dB、20dB）并保存。


import os
import random
import re
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise

from core.STFTMusicProcessor import STFTMusicProcessorPredict
from database.MySQLConnector import MySQLConnector
from utils.hparam import hp
from utils.print_utils import print_message, print_error
import tqdm


def random_slice_audio(audio, slice_duration):
    """
    从音频中随机截取片段
    :param audio: 音频对象
    :param slice_duration: 可选的截取时长列表（单位：毫秒）
    :return: 截取后的音频片段
    """
    duration = len(audio)
    if slice_duration > duration:
        print("音频时长不足，无法截取指定长度的片段。")
        return audio
    start_time = random.randint(0, duration - slice_duration)
    end_time = start_time + slice_duration
    return audio[start_time:end_time]


def add_noise(audio, snr_db):
    """
    给音频添加随机噪声
    :param audio: 音频对象
    :param snr_db: 可选的信噪比列表（单位：dB）
    :return: 添加噪声后的音频
    """
    audio_rms = audio.rms
    noise_rms = audio_rms / (10 ** (snr_db / 20))

    noise = WhiteNoise().to_audio_segment(duration=len(audio), volume=0)
    noise_rms_current = noise.rms
    noise = noise - (20 * np.log10(noise_rms_current / noise_rms))

    return audio.overlay(noise)


def generate_clip_length_snr(input_file, output_folder):
    """
    处理音频文件，包括随机截取和添加噪声，并保存处理后的音频
    :param input_file: 输入音频文件路径
    :param output_folder: 输出文件夹路径
    """
    # 读取音频文件
    audio = AudioSegment.from_file(input_file)
    # 三个循环
    # 每个音频取5段
    for i in range(5):
        # 时长 3s 5s 10s
        for clip_length in [3000, 5000, 10000]:
            # 随机截取音频片段
            sliced_audio = random_slice_audio(audio, clip_length)
            # 信噪比 0 10 20dB
            for snr in [0, 10, 20]:
                # 添加噪声
                try:
                    noisy_audio = add_noise(sliced_audio, snr)
                    # 保存处理后的音频
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    base_name = os.path.basename(input_file).split('.')[0]
                    output_file = os.path.join(output_folder,
                                               f"{base_name}_clip_{i + 1}_{int(clip_length / 1000)}s_{snr}dB.wav")
                    noisy_audio.export(output_file, format="wav")
                    print(f"处理后的音频已保存到: {output_file}")
                except Exception as e:
                    print(e)


class EvaluationCalculation:
    def __init__(self):
        self.calc_dict = dict()
        for duration in [3, 5, 10]:
            for snr in [0, 10, 20]:
                self.calc_dict[f"sample_{duration}s_{snr}dB"] = 0
                self.calc_dict[f"correct_{duration}s_{snr}dB"] = 0

    def _find_clip_length_snr(self, text):

        # 匹配时长数值
        duration_pattern = re.compile(r'_(\d+)s_')
        duration_match = duration_pattern.search(text)
        duration = duration_match.group(1) if duration_match else None

        # 匹配信噪比数值
        snr_pattern = re.compile(r'_(\d+)dB\.')
        snr_match = snr_pattern.search(text)
        snr = snr_match.group(1) if snr_match else None

        if duration is None:
            raise RuntimeError("No duration time matched!!!")
        if snr is None:
            raise RuntimeError("No snr value matched!!!")
        return duration, snr

    def calculate(self, src: str, pred: str):
        """
        :param src:
        :param pred:
        :return:
        """
        duration, snr = self._find_clip_length_snr(src)
        self.calc_dict[f"sample_{duration}s_{snr}dB"] += 1
        if src[:5] == pred[:5]:
            self.calc_dict[f"correct_{duration}s_{snr}dB"] += 1

    def analyze(self):
        # 按照时长统计
        return self.calc_dict


def evaluate_predict_performance(query_path):
    """
    :param query_path:
    :return:
    """
    # 获取数据库的连接
    connector = MySQLConnector()
    # 获取核心的预测处理器
    music_processor = STFTMusicProcessorPredict()
    ec = EvaluationCalculation()
    # 逐个预测数据
    all_file_list = os.listdir(query_path)
    for _file in tqdm.trange(len(all_file_list)):
        path = all_file_list[_file]
        if not (path.endswith("wav") or path.endswith("mp3")):
            continue
        # 获取音乐的相对路径
        music_path = os.path.join(query_path, path)
        print(f"Processing: {path}")
        # 预测歌曲
        try:
            music_info = music_processor.predict_music(music_path=music_path, connector=connector)
        except Exception as e:
            print_error(e)
            continue
        # 根据music_info输出
        music_id = music_info['music_id']
        music_name = connector.find_music_name_by_music_id(music_id)
        print_message("预测歌曲：" + str(music_name) +
                      ", --- 线性匹配的Hash个数为：" + str(music_info['max_hash_count']) +
                      ", --- 歌曲偏移：" + str(music_info['music_offset']))
        print("\n")
        ec.calculate(path, music_name)
    print(ec.analyze())
    return


if __name__ == "__main__":
    # 生成数据
    # audio_dir = "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/mos/Music_MOS/dataset/music_key/"
    # for audio_file in os.listdir(audio_dir):
    #     input_file = os.path.join(audio_dir, audio_file)
    #     output_folder = "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/mos/Music_MOS/dataset/music_query/"  # 替换为你的输出文件夹路径
    #     generate_clip_length_snr(input_file, output_folder)

    evaluate_predict_performance(
        "/mnt/test/scripts/label-studio-zch/label-studio-ml-backend/label_studio_ml/examples/mos/Music_MOS/dataset/music_query/")

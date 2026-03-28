"""
计算两个音频文件的互相关来找到它们之间的最佳对齐位置，然后将测试音频文件移动到这个位置。
"""
import os
import librosa
import numpy as np
from scipy import signal
import soundfile as sf


def shift_audio(test_audio, lag):
    """
    用于将音频数据向前或向后移动指定的延迟（lag）个位置。
    如果lag为正，音频数据向前移动；
    如果lag为负，音频数据向后移动；
    如果lag为零，音频数据保持不变。
    """
    # 获取音频数据的长度
    audio_length = len(test_audio)
    # 创建一个和原音频长度相同的零数组，用于存储移动后的音频数据
    shifted_audio = np.zeros_like(test_audio)

    if lag > 0:
        # 如果 lag 大于零，将 test_audio 往前移动 lag 个位置
        shifted_audio[:audio_length - lag] = test_audio[lag:]
    elif lag < 0:
        # 如果 lag 小于零，将 test_audio 往后移动 abs(lag) 个位置
        shifted_audio[-lag:] = test_audio[:audio_length + lag]
    else:
        # 如果 lag 等于零，不进行移动
        shifted_audio = test_audio
    return shifted_audio


def align_audio(test_audio_path, reference_audio_path, output_file_path):
    """
    对齐两个音频文件
    1. 加载测试音频和参考音频，并确保它们的采样率相同。
    2. 计算测试音频和参考音频的互相关，找到最大互相关值的索引，这个索引表示了两个音频之间的延迟。
    3. 使用shift_audio函数将测试音频移动到这个延迟位置，并将对齐后的音频保存到指定的输出文件路径。
    """
    # 加载测试音频和参考音频
    test_audio, test_sr = librosa.load(test_audio_path, sr=None)
    reference_audio, reference_sr = librosa.load(reference_audio_path, sr=None)
    print(test_audio_path,test_sr)
    print(reference_audio_path,reference_sr)
    # 确保采样率一致
    if test_sr != reference_sr:
        raise ValueError("采样率不一致，请确保测试音频和参考音频的采样率相同。")

    # 计算互相关
    correlation = signal.correlate(test_audio, reference_audio, mode='full')
    # 找到互相关结果中的最大值的索引
    lag = np.argmax(correlation) - (len(test_audio) - 1)
    print("最大互相关值的延迟为：", lag / test_sr, "s")
    # 根据对齐结果调整测试音频
    aligned_test_audio = shift_audio(test_audio, lag)
    sf.write(output_file_path, aligned_test_audio, test_sr)
    return aligned_test_audio, test_sr, output_file_path


def align_splited_wav_in_dir(input_dir, ref_dir, output_dir, ):
    """
    对齐一个文件夹中的所有音频文件
    1. 遍历输入文件夹中的所有.wav文件，对于每个文件，它找到对应的参考音频文件（文件名以ref_开头，并且包含相同的文件名部分），
    2. 调用align_audio函数对齐这两个音频文件，并将对齐后的音频保存到输出文件夹。
    """
    # 遍历input_dir
    test_audio_file_list = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            # 获取音频文件的路径
            test_audio_path = os.path.join(input_dir, filename)
            test_audio_file_list.append(test_audio_path)
            # print("正在对齐音频文件：" + filename)
            # # 获取参考音频文件的路径
            # reference_audio_path = os.path.join(ref_dir, "ref_" + filename.removesuffix(".wav").split("_")[-1] + ".wav")
            # # 获取输出文件的路径
            # output_file_path = os.path.join(output_dir, filename)
            # # 对齐音频文件
            # align_audio(test_audio_path, reference_audio_path, output_file_path)
            # print(f"音频对齐完成，输出到 {output_file_path}")
    align_splited_wav_from_list(test_audio_file_list, ref_dir, output_dir)


def align_splited_wav_from_list(input_file_list, ref_dir, output_dir):
    """
    对输入的文件列表中的文件执行对齐操作
    """
    os.makedirs(output_dir, exist_ok=True)
    print(input_file_list)
    output_filepath_list = []
    for test_audio_path in input_file_list:
        reference_audio_path = os.path.join(ref_dir,
                                            "ref_" + test_audio_path.removesuffix(".wav").split("_")[-1] + ".wav")
        _, _, output_file_path = align_audio(test_audio_path=test_audio_path,
                                             reference_audio_path=reference_audio_path,
                                             output_file_path=os.path.join(output_dir, os.path.basename(test_audio_path)))
        output_filepath_list.append(output_file_path)
    return output_filepath_list


if __name__ == '__main__':
    align_splited_wav_in_dir(input_dir="split_out", ref_dir="ref_dir", output_dir="est_dir")

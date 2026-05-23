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
    
    注意：这里的lag是基于互相关计算的结果：
    - lag > 0: 测试音频内容比参考音频晚，需要向后移动（填充前面的空白）
    - lag < 0: 测试音频内容比参考音频早，需要向前移动（删除前面的内容）
    """
    # 获取音频数据的长度
    audio_length = len(test_audio)
    # 创建一个和原音频长度相同的零数组，用于存储移动后的音频数据
    shifted_audio = np.zeros_like(test_audio)

    if lag > 0:
        # 如果 lag 大于零，测试音频内容比参考音频晚
        # 需要向后移动：将内容向后推，前面补零
        shifted_audio[lag:] = test_audio[:-lag]
    elif lag < 0:
        # 如果 lag 小于零，测试音频内容比参考音频早
        # 需要向前移动：删除前面的内容
        shifted_audio[:audio_length + lag] = test_audio[-lag:]
    else:
        # 如果 lag 等于零，不进行移动
        shifted_audio = test_audio
    return shifted_audio


def align_audio(test_audio_path, reference_audio_path, output_file_path, split_redundancy=0.5):
    """
    对齐两个音频文件
    1. 加载测试音频和参考音频，并确保它们的采样率相同。
    2. 计算测试音频和参考音频的互相关，找到最大互相关值的索引，这个索引表示了两个音频之间的延迟。
    3. 使用shift_audio函数将测试音频移动到这个延迟位置，并将对齐后的音频保存到指定的输出文件路径。

    参数:
        split_redundancy: 切分时保留的前置冗余时间（秒），默认0.5s
                         切分后的音频中，目标内容从split_redundancy秒开始
    """
    # 加载测试音频和参考音频
    test_audio, test_sr = librosa.load(test_audio_path, sr=None)
    reference_audio, reference_sr = librosa.load(reference_audio_path, sr=None)
    print(f"【对齐算法】测试音频: {test_audio_path}, 采样率: {test_sr}, 长度: {len(test_audio)/test_sr:.3f}s")
    print(f"【对齐算法】参考音频: {reference_audio_path}, 采样率: {reference_sr}, 长度: {len(reference_audio)/reference_sr:.3f}s")

    # 确保采样率一致
    if test_sr != reference_sr:
        raise ValueError("采样率不一致，请确保测试音频和参考音频的采样率相同。")

    # 由于切分后的音频包含前置冗余（split_redundancy秒），
    # 真正的目标内容从 split_redundancy 秒开始
    # 因此需要去掉前置冗余，并裁剪到与参考音频相同长度后再进行对齐计算
    redundancy_samples = int(split_redundancy * test_sr)
    ref_length_samples = len(reference_audio)

    if len(test_audio) >= redundancy_samples + ref_length_samples:
        # 去掉前置冗余，并裁剪到与参考音频相同长度
        test_audio_for_align = test_audio[redundancy_samples:redundancy_samples + ref_length_samples]
        print(f"【对齐算法】去除前置冗余 {split_redundancy}s ({redundancy_samples}采样点)")
        print(f"【对齐算法】裁剪到与参考音频相同长度: {ref_length_samples}采样点 ({ref_length_samples/test_sr:.3f}s)")
    elif len(test_audio) >= redundancy_samples + 1000:
        # 如果长度不够，只去掉冗余
        test_audio_for_align = test_audio[redundancy_samples:]
        print(f"【对齐算法】去除前置冗余 {split_redundancy}s ({redundancy_samples}采样点)")
        print(f"【对齐算法】警告: 音频长度不足，无法裁剪到参考音频长度")
    else:
        test_audio_for_align = test_audio
        print(f"【对齐算法】音频长度不足，保留前置冗余进行对齐")

    # 计算互相关
    # 使用 signal.correlate(reference_audio, test_audio_for_align) 计算参考音频与测试音频的互相关
    # 这样计算出的 lag 表示：测试音频需要移动多少才能与参考音频对齐
    # lag > 0: 测试音频需要向前移动（内容提前）
    # lag < 0: 测试音频需要向后移动（内容延后）
    correlation = signal.correlate(reference_audio, test_audio_for_align, mode='full')
    # 找到互相关结果中的最大值的索引
    lag = np.argmax(correlation) - (len(reference_audio) - 1)
    lag_seconds = lag / test_sr

    print(f"【对齐算法】互相关计算: 延迟 = {lag} 采样点 = {lag_seconds:.3f}s")

    # 根据对齐结果调整原始测试音频
    # 注意：lag 是基于去掉冗余后的音频计算的
    # 如果 lag > 0，表示测试音频内容比参考音频晚，需要向后移动（补零）
    # 如果 lag < 0，表示测试音频内容比参考音频早，需要向前移动（删除）
    if lag > 0:
        print(f"【对齐算法】测试音频内容比参考音频晚 {lag_seconds:.3f}s，向后移动")
    elif lag < 0:
        print(f"【对齐算法】测试音频内容比参考音频早 {-lag_seconds:.3f}s，向前移动")
    else:
        print(f"【对齐算法】测试音频与参考音频已对齐，无需移动")

    aligned_test_audio = shift_audio(test_audio, lag)

    # 对齐后去掉前置冗余，使输出音频与参考音频长度和结构一致
    # 参考音频本身可能有前置静音，对齐后的音频应该与之匹配
    if len(aligned_test_audio) > redundancy_samples:
        aligned_test_audio = aligned_test_audio[redundancy_samples:]
        print(f"【对齐算法】去掉前置冗余 {split_redundancy}s，输出长度: {len(aligned_test_audio)/test_sr:.3f}s")

    # 确保输出音频与参考音频长度相同
    if len(aligned_test_audio) > ref_length_samples:
        aligned_test_audio = aligned_test_audio[:ref_length_samples]
        print(f"【对齐算法】裁剪到与参考音频相同长度: {ref_length_samples/test_sr:.3f}s")
    elif len(aligned_test_audio) < ref_length_samples:
        # 如果长度不足，后面补零
        aligned_test_audio = np.pad(aligned_test_audio, (0, ref_length_samples - len(aligned_test_audio)), mode='constant')
        print(f"【对齐算法】长度不足，补零至: {ref_length_samples/test_sr:.3f}s")

    sf.write(output_file_path, aligned_test_audio, test_sr)
    print(f"【对齐算法】对齐完成，已保存到: {output_file_path}")
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
    align_splited_wav_from_list(test_audio_file_list, ref_dir, output_dir)


def align_splited_wav_from_list(input_file_list, ref_dir, output_dir):
    """
    对输入的文件列表中的文件执行对齐操作
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"【对齐流程】输入文件列表: {[os.path.basename(f) for f in input_file_list]}")
    output_filepath_list = []

    for test_audio_path in input_file_list:
        # 提取文件名中的序号（如 xxx_001.wav 或 xxx_001_001.wav 中的 001）
        basename = os.path.basename(test_audio_path)
        name_without_ext = basename.removesuffix(".wav")
        parts = name_without_ext.split("_")

        # 尝试多种方式匹配参考文件
        # 方式1: 最后一部分是序号（如 xxx_001.wav -> ref_001.wav）
        ref_file_1 = f"ref_{parts[-1]}.wav"
        # 方式2: 倒数第二部分是序号（如 xxx_001_001.wav -> ref_001.wav）
        ref_file_2 = f"ref_{parts[-2]}.wav" if len(parts) >= 2 else ""

        reference_audio_path = None
        if os.path.exists(os.path.join(ref_dir, ref_file_1)):
            reference_audio_path = os.path.join(ref_dir, ref_file_1)
            print(f"【对齐流程】文件 {basename} 匹配参考文件: {ref_file_1} (方式1)")
        elif ref_file_2 and os.path.exists(os.path.join(ref_dir, ref_file_2)):
            reference_audio_path = os.path.join(ref_dir, ref_file_2)
            print(f"【对齐流程】文件 {basename} 匹配参考文件: {ref_file_2} (方式2)")
        else:
            # 尝试列出参考目录中的所有文件
            ref_files = [f for f in os.listdir(ref_dir) if f.endswith('.wav')]
            print(f"【对齐流程】警告: 无法匹配参考文件 for {basename}")
            print(f"【对齐流程】尝试匹配: {ref_file_1}, {ref_file_2}")
            print(f"【对齐流程】参考目录中的文件: {ref_files}")
            continue

        output_file_path = os.path.join(output_dir, basename)
        _, _, output_path = align_audio(
            test_audio_path=test_audio_path,
            reference_audio_path=reference_audio_path,
            output_file_path=output_file_path
        )
        output_filepath_list.append(output_path)

    return output_filepath_list


if __name__ == '__main__':
    align_splited_wav_in_dir(input_dir="split_out", ref_dir="ref_dir", output_dir="est_dir")

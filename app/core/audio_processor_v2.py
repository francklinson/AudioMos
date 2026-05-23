"""
改进版音频对齐算法
基于频谱特征的对齐，对噪声和长度变化更鲁棒
"""
import os
import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import soundfile as sf


def shift_audio(test_audio, lag):
    """
    用于将音频数据向前或向后移动指定的延迟（lag）个位置。
    如果lag为正，音频数据向前移动（内容提前）;
    如果lag为负，音频数据向后移动（内容延后）;
    如果lag为零，音频数据保持不变。
    """
    audio_length = len(test_audio)
    shifted_audio = np.zeros_like(test_audio)

    if lag > 0:
        # 如果 lag 大于零，将 test_audio 往前移动 lag 个位置
        shifted_audio[:audio_length - lag] = test_audio[lag:]
    elif lag < 0:
        # 如果 lag 小于零，将 test_audio 往后移动 abs(lag) 个位置
        shifted_audio[-lag:] = test_audio[:audio_length + lag]
    else:
        shifted_audio = test_audio
    return shifted_audio


def align_audio_spectral(test_audio_path, reference_audio_path, output_file_path, 
                         split_redundancy=0.5, n_fft=2048, hop_length=256):
    """
    基于频谱特征的音频对齐算法
    
    改进点：
    1. 使用STFT频谱而不是原始波形，对噪声更鲁棒
    2. 对频谱的每一帧计算互相关，然后取平均
    3. 考虑切分冗余，只对齐内容部分
    
    参数:
        test_audio_path: 测试音频路径
        reference_audio_path: 参考音频路径
        output_file_path: 输出文件路径
        split_redundancy: 切分时保留的前置冗余时间（秒），默认0.5s
        n_fft: FFT窗口大小
        hop_length: 帧移
    """
    # 加载音频
    test_audio, test_sr = librosa.load(test_audio_path, sr=None)
    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=None)
    
    print(f"【对齐算法】测试音频: {test_audio_path}, 采样率: {test_sr}, 长度: {len(test_audio)/test_sr:.3f}s")
    print(f"【对齐算法】参考音频: {reference_audio_path}, 采样率: {ref_sr}, 长度: {len(ref_audio)/ref_sr:.3f}s")

    if test_sr != ref_sr:
        raise ValueError("采样率不一致")

    # 计算STFT
    D_test = np.abs(librosa.stft(test_audio, n_fft=n_fft, hop_length=hop_length))
    D_ref = np.abs(librosa.stft(ref_audio, n_fft=n_fft, hop_length=hop_length))
    
    print(f"【对齐算法】测试音频STFT形状: {D_test.shape}, 参考音频STFT形状: {D_ref.shape}")

    # 去掉前置冗余（转换为帧数）
    redundancy_frames = int(split_redundancy * test_sr / hop_length)
    
    if D_test.shape[1] > redundancy_frames + D_ref.shape[1]:
        # 去掉冗余，并裁剪到与参考音频相同长度
        D_test_content = D_test[:, redundancy_frames:redundancy_frames + D_ref.shape[1]]
        print(f"【对齐算法】去除前置冗余 {split_redundancy}s ({redundancy_frames}帧)")
        print(f"【对齐算法】裁剪到与参考音频相同长度: {D_ref.shape[1]}帧")
    elif D_test.shape[1] > redundancy_frames:
        D_test_content = D_test[:, redundancy_frames:]
        print(f"【对齐算法】去除前置冗余 {split_redundancy}s ({redundancy_frames}帧)")
        print(f"【对齐算法】警告: 音频长度不足，无法裁剪到参考音频长度")
    else:
        D_test_content = D_test
        print(f"【对齐算法】音频长度不足，保留前置冗余进行对齐")

    # 对每一频率bin计算互相关
    correlations = []
    min_frames = min(D_test_content.shape[1], D_ref.shape[1])
    
    for freq_bin in range(D_test_content.shape[0]):
        # 计算该频率bin的互相关
        corr = signal.correlate(
            D_ref[freq_bin, :min_frames], 
            D_test_content[freq_bin, :min_frames], 
            mode='full'
        )
        correlations.append(corr)
    
    # 平均所有频率bin的互相关
    avg_correlation = np.mean(correlations, axis=0)
    
    # 找到峰值
    peak_idx = np.argmax(avg_correlation)
    lag_frames = peak_idx - (min_frames - 1)
    
    # 转换为采样点
    lag_samples = lag_frames * hop_length
    lag_seconds = lag_samples / test_sr
    
    print(f"【对齐算法】频谱互相关计算: 延迟 = {lag_frames}帧 = {lag_samples}采样点 = {lag_seconds:.3f}s")
    
    # 根据对齐结果调整原始测试音频
    if lag_samples > 0:
        print(f"【对齐算法】测试音频内容比参考音频晚 {lag_seconds:.3f}s，向前移动")
    elif lag_samples < 0:
        print(f"【对齐算法】测试音频内容比参考音频早 {-lag_seconds:.3f}s，向后移动")
    else:
        print(f"【对齐算法】测试音频与参考音频已对齐，无需移动")
    
    # 应用移动
    aligned_audio = shift_audio(test_audio, lag_samples)
    sf.write(output_file_path, aligned_audio, test_sr)
    print(f"【对齐算法】对齐完成，已保存到: {output_file_path}")
    
    return aligned_audio, test_sr, output_file_path


def align_splited_wav_from_list(input_file_list, ref_dir, output_dir, 
                                 split_redundancy=0.5, n_fft=2048, hop_length=512):
    """
    对输入的文件列表中的文件执行对齐操作（使用改进的频谱对齐算法）
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"【对齐流程】输入文件列表: {[os.path.basename(f) for f in input_file_list]}")
    output_filepath_list = []

    for test_audio_path in input_file_list:
        # 提取文件名中的序号
        basename = os.path.basename(test_audio_path)
        name_without_ext = basename.removesuffix(".wav")
        parts = name_without_ext.split("_")

        # 尝试多种方式匹配参考文件
        ref_file_1 = f"ref_{parts[-1]}.wav"
        ref_file_2 = f"ref_{parts[-2]}.wav" if len(parts) >= 2 else ""

        reference_audio_path = None
        if os.path.exists(os.path.join(ref_dir, ref_file_1)):
            reference_audio_path = os.path.join(ref_dir, ref_file_1)
            print(f"【对齐流程】文件 {basename} 匹配参考文件: {ref_file_1} (方式1)")
        elif ref_file_2 and os.path.exists(os.path.join(ref_dir, ref_file_2)):
            reference_audio_path = os.path.join(ref_dir, ref_file_2)
            print(f"【对齐流程】文件 {basename} 匹配参考文件: {ref_file_2} (方式2)")
        else:
            ref_files = [f for f in os.listdir(ref_dir) if f.endswith('.wav')]
            print(f"【对齐流程】警告: 无法匹配参考文件 for {basename}")
            print(f"【对齐流程】尝试匹配: {ref_file_1}, {ref_file_2}")
            print(f"【对齐流程】参考目录中的文件: {ref_files}")
            continue

        output_file_path = os.path.join(output_dir, basename)
        _, _, output_path = align_audio_spectral(
            test_audio_path=test_audio_path,
            reference_audio_path=reference_audio_path,
            output_file_path=output_file_path,
            split_redundancy=split_redundancy,
            n_fft=n_fft,
            hop_length=256  # 使用更小的hop_length提高精度
        )
        output_filepath_list.append(output_path)

    return output_filepath_list


if __name__ == '__main__':
    # 测试
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    task_id = "65427752-dd3e-4b13-b2ad-aa127d1c3525"
    split_dir = f"data/temp/{task_id}_split"
    ref_dir = "data/ref"
    output_dir = f"data/temp/{task_id}_aligned_v2"
    
    os.makedirs(output_dir, exist_ok=True)
    
    test_files = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.wav')]
    test_files.sort()
    
    align_splited_wav_from_list(test_files, ref_dir, output_dir)

"""
【废弃】此文件未被任何代码引用，仅供参考

改进版音频对齐算法（GCC-PHAT方案，未集成到主流程）
当前管道使用 matching_optimizer 的 HPSS谐波互相关精对齐。

保留此文件作为 GCC-PHAT 对齐的参考实现，不参与实际管道。
"""
import os
import librosa
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import soundfile as sf


def align_audio_gcc_phat(test_audio_path, reference_audio_path, output_file_path):
    """
    GCC-PHAT广义互相关对齐（参考实现，未集成）
    注：在低SNR下GCC-PHAT表现不如HPSS谐波互相关，详见 matching_optimizer._hpss_fine_align
    """
    test_audio, test_sr = librosa.load(test_audio_path, sr=None)
    reference_audio, reference_sr = librosa.load(reference_audio_path, sr=None)

    if test_sr != reference_sr:
        raise ValueError("采样率不一致")

    n = min(len(test_audio), len(reference_audio))
    test = test_audio[:n]
    ref = reference_audio[:n]

    n_fft = 2 ** (int(np.log2(2 * n - 1)) + 1)
    TEST_F = fft(test, n=n_fft)
    REF_F = fft(ref, n=n_fft)

    cross = REF_F * np.conj(TEST_F)
    cross /= np.abs(cross) + 1e-10
    gcc = np.real(ifft(cross))

    mid = n_fft // 2
    max_lag = int(2.0 * test_sr)
    s = max(0, mid - max_lag)
    e = min(len(gcc), mid + max_lag)
    pk = np.argmax(np.abs(gcc[s:e]))
    lag = (s + pk) - mid

    aligned = np.zeros_like(test_audio)
    if lag > 0:
        aligned[lag:] = test_audio[:-lag]
    elif lag < 0:
        aligned[:len(test_audio) + lag] = test_audio[-lag:]
    else:
        aligned = test_audio

    sf.write(output_file_path, aligned, test_sr)
    return aligned, test_sr, output_file_path

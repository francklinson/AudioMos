#!/usr/bin/env python3
"""
生成测试音频集

包含多种噪声类型、不同SNR级别的测试音频，用于验证降噪/修复算法效果。

输出目录: data/test_audio/
"""

import os
import numpy as np
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "test_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 音频参数 ──────────────────────────────────────────────
SR_16K = 16000
SR_48K = 48000


def make_speech(t, f0=200):
    """模拟语音: 基频 + 谐波 + 颤音"""
    vibrato = 1 + 0.005 * np.sin(2 * np.pi * 5 * t)
    s = 0.6 * np.sin(2 * np.pi * f0 * t * vibrato)
    s += 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
    s += 0.15 * np.sin(2 * np.pi * f0 * 3 * t)
    s += 0.07 * np.sin(2 * np.pi * f0 * 4 * t)
    # 幅度包络
    env = np.exp(-2 * np.abs(t - t.mean()) / (t.max() - t.min()))
    return s * env


def make_chirp(t):
    """线性调频信号（模拟复杂语音）"""
    f0, f1 = 300, 3000
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * t.max()))
    env = 0.5 * (1 - np.cos(2 * np.pi * t / t.max()))
    return np.sin(phase) * env * 0.5


def pink_noise(n):
    """粉红噪声 (1/f 频谱)"""
    white = np.random.randn(n)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # 避免除零
    fft = fft / np.sqrt(freqs)
    result = np.fft.irfft(fft, n)
    return result / np.max(np.abs(result)) * 0.3


def babble_noise(n, n_speakers=6):
    """模拟多人交谈噪声（多正弦混合）"""
    t = np.arange(n) / SR_16K
    noise = np.zeros(n)
    for i in range(n_speakers):
        f0 = np.random.uniform(100, 300)
        noise += 0.15 * np.sin(2 * np.pi * f0 * t + np.random.rand() * 2 * np.pi)
        noise += 0.08 * np.sin(2 * np.pi * f0 * 2 * t + np.random.rand() * 2 * np.pi)
    return noise / np.max(np.abs(noise)) * 0.3


def traffic_noise(n):
    """模拟交通噪声: 低频轰鸣 + 随机冲击"""
    t = np.arange(n) / SR_16K
    noise = 0.15 * np.sin(2 * np.pi * 60 * t)  # 低频引擎声
    noise += 0.1 * np.sin(2 * np.pi * 120 * t)
    # 随机鸣笛/冲击
    for _ in range(3):
        pos = np.random.randint(0, n - SR_16K // 2)
        dur = SR_16K // 4
        env = np.hanning(dur)
        noise[pos:pos + dur] += 0.4 * env * np.sin(2 * np.pi * 800 * t[:dur])
    return noise / np.max(np.abs(noise)) * 0.3


def station_noise(n):
    """模拟稳态噪声: 多频带恒定噪声"""
    t = np.arange(n) / SR_16K
    noise = np.zeros(n)
    for freq in [50, 100, 200, 400, 800]:
        noise += 0.06 * np.sin(2 * np.pi * freq * t + np.random.rand())
    noise += 0.08 * np.random.randn(n)
    return noise / np.max(np.abs(noise)) * 0.25


def mix_with_snr(signal, noise, snr_db):
    """按指定 SNR 混合信号和噪声"""
    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return signal
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return signal + noise * scale


def save(path, audio, sr):
    sf.write(path, audio.astype(np.float32), sr)
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  {os.path.basename(path):40s} {sr/1000:.0f}kHz {duration:.1f}s RMS={rms:.4f}")


# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("  生成测试音频集")
print("=" * 60)
print()

# ── 测试组 1: 不同噪声类型 (16kHz) ────────────────
print("【组1】不同噪声类型 @ 5dB SNR (16kHz)")

duration = 4.0
n = int(SR_16K * duration)
t = np.arange(n) / SR_16K
speech = make_speech(t, f0=220)

noise_types = {
    "white": lambda n: np.random.randn(n) * 0.3,
    "pink": lambda n: pink_noise(n),
    "babble": lambda n: babble_noise(n),
    "traffic": lambda n: traffic_noise(n),
    "station": lambda n: station_noise(n),
}

for name, noise_fn in noise_types.items():
    noise = noise_fn(n)
    # 保存纯净语音
    if name == "white":
        save(os.path.join(OUTPUT_DIR, "clean_speech_16k.wav"), speech, SR_16K)
    # 保存噪声
    save(os.path.join(OUTPUT_DIR, f"noise_{name}_16k.wav"), noise, SR_16K)
    # 混合
    noisy = mix_with_snr(speech, noise, snr_db=5)
    save(os.path.join(OUTPUT_DIR, f"noisy_{name}_5db_16k.wav"), noisy, SR_16K)

print()

# ── 测试组 2: 不同 SNR 级别 ────────────────────────
print("【组2】不同SNR级别 — 白噪声 (16kHz)")

for snr in [-5, 0, 5, 10, 20]:
    noise = np.random.randn(n) * 0.3
    noisy = mix_with_snr(speech, noise, snr_db=snr)
    label = f"{snr:+d}" if snr >= 0 else f"{snr}"
    save(os.path.join(OUTPUT_DIR, f"noisy_white_{label}db_16k.wav"), noisy, SR_16K)

print()

# ── 测试组 3: 48kHz 测试音频 ────────────────────────
print("【组3】48kHz 高采样率测试")

n_48k = int(SR_48K * 4.0)
t_48k = np.arange(n_48k) / SR_48K
speech_48k = make_speech(t_48k, f0=220)

save(os.path.join(OUTPUT_DIR, "clean_speech_48k.wav"), speech_48k, SR_48K)

noise_48k = np.random.randn(n_48k) * 0.3
noisy_48k = mix_with_snr(speech_48k, noise_48k, snr_db=5)
save(os.path.join(OUTPUT_DIR, "noisy_white_5db_48k.wav"), noisy_48k, SR_48K)

print()

# ── 测试组 4: 语音分离测试 ──────────────────────────
print("【组4】语音分离 — 双人混合 (16kHz)")

n_sep = int(SR_16K * 5.0)
t_sep = np.arange(n_sep) / SR_16K

# 说话人A (男声)
spk_a = make_speech(t_sep, f0=150)
# 说话人B (女声)
spk_b = make_speech(t_sep, f0=280)

save(os.path.join(OUTPUT_DIR, "speaker_a_16k.wav"), spk_a, SR_16K)
save(os.path.join(OUTPUT_DIR, "speaker_b_16k.wav"), spk_b, SR_16K)
mix_ab = (spk_a * 0.6 + spk_b * 0.5).astype(np.float32)
mix_ab = mix_ab / np.max(np.abs(mix_ab)) * 0.8
save(os.path.join(OUTPUT_DIR, "mixed_2speakers_16k.wav"), mix_ab, SR_16K)

print()

# ── 测试组 5: 极端场景 ─────────────────────────────
print("【组5】极端场景测试 (16kHz)")

# 超低SNR
noise = np.random.randn(n) * 0.5
noisy_vlow = mix_with_snr(speech, noise, snr_db=-10)
save(os.path.join(OUTPUT_DIR, "noisy_white_-10db_extreme_16k.wav"), noisy_vlow, SR_16K)

# 长音频 (10秒)
n_long = int(SR_16K * 10)
t_long = np.arange(n_long) / SR_16K
speech_long = np.tile(make_speech(t[:SR_16K * 2], f0=200), 5)
noise_long = pink_noise(n_long)
noisy_long = mix_with_snr(speech_long, noise_long, snr_db=5)
save(os.path.join(OUTPUT_DIR, "noisy_pink_5db_10s_16k.wav"), noisy_long, SR_16K)

# chirp信号（测试频率响应）
chirp = make_chirp(t)
save(os.path.join(OUTPUT_DIR, "chirp_clean_16k.wav"), chirp, SR_16K)
noisy_chirp = mix_with_snr(chirp, np.random.randn(n) * 0.3, snr_db=5)
save(os.path.join(OUTPUT_DIR, "noisy_chirp_5db_16k.wav"), noisy_chirp, SR_16K)

print()

# ── 汇总 ────────────────────────────────────────────
print("=" * 60)
total_size = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(path)
    total_size += size
    print(f"  {f:45s} {size/1024:7.1f} KB")

print(f"\n  共 {len(os.listdir(OUTPUT_DIR))} 个文件, {total_size/1024/1024:.1f} MB")
print(f"  输出目录: {OUTPUT_DIR}")
print("=" * 60)

# HPSS谐波互相关精对齐优化

**日期:** 2026-06-28
**作者:** zhouchenghao
**状态:** 已完成

## 解决的问题

DTW（MFCC帧级）定位在低SNR（8~10dB）场景下存在系统性偏移（最大-0.37s），导致切分后的音频与参考音频在样本级不对齐。原始波形互相关在噪声下质量极差（0.055），无法用于精对齐修正。

## 方案

在DTW定位切分后，增加**HPSS谐波互相关精对齐**步骤：

1. HPSS（Harmonic-Percussive Source Separation）从噪声中提取语音谐波分量
2. 在谐波域做互相关检测残留时间偏移（lag）
3. 修正offset = DTW offset - lag
4. 用修正后的offset重新切分

## 效果

| 指标 | DTW-only | DTW+HPSS |
|------|----------|----------|
| 降噪关平均偏移 | 0.52s | **0.22s** |
| 降噪关残留lag<10ms | — | **100%** |
| 降噪关残留lag>30ms | ~38% | **0%** |

全部80段（20文件×4参考段）端到端测试通过。

## 安全机制

- HPSS互相关质量 < 0.02 时跳过修正（保护降噪损伤场景）
- 检测lag > 2.0s 时跳过修正（防止DTW完全失效时误修正）

## 改动文件

- `app/core/matching_optimizer.py`:
  - `extract_harmonic_component()` — 增加 `kernel_size` 参数（默认51）
  - 新增 `OptimizedMatcher._hpss_fine_align()` — HPSS精对齐核心方法
  - `cut_all_audio_files_with_optimized_matcher()` — 集成HPSS精对齐

## 关键参数

- HPSS kernel_size = 51（谐波核大小）
- max_correction_s = 2.0（最大修正量±2s）
- min_quality = 0.02（最低互相关质量阈值）

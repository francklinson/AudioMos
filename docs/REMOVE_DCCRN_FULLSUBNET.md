# DCCRN 和 FullSubNet 移除说明

## 移除原因

由于 DCCRN 和 FullSubNet 的官方模型源不可用（HuggingFace 和 ModelScope 上的仓库都不存在或需要认证），无法自动下载这两个模型的预训练权重。为避免用户混淆和错误提示，决定从项目中移除这两种降噪方法。

## 已删除的文件

### 算法实现文件
- `app/algorithms/denoise/dccrn_denoiser.py` - DCCRN 降噪算法实现
- `app/algorithms/denoise/fullsubnet_denoiser.py` - FullSubNet 降噪算法实现

### 下载脚本和文档
- `scripts/download_dccrn_models.py` - DCCRN 模型下载脚本
- `scripts/download_fullsubnet_models.py` - FullSubNet 模型下载脚本
- `scripts/download_denoise_models.py` - 批量下载脚本
- `scripts/download_denoise_models_guide.py` - 下载指南
- `docs/DOWNLOAD_MODELS.md` - 模型下载文档

## 已修改的文件

### 1. `app/algorithms/denoise/__init__.py`
- 删除了 DCCRN 和 FullSubNet 的导入语句
- 删除了 `DCCRN_AVAILABLE` 和 `FULLSUBNET_AVAILABLE` 标志
- 从 `__all__` 中移除了相关导出

### 2. `app/algorithms/denoise/registry.py`
- 从 `DENOISER_DESCRIPTIONS` 中删除了 DCCRN 和 FullSubNet 的描述信息

### 3. `app/algorithms/restoration/__init__.py`
- 从 `RESTORER_PRESETS` 中删除了 DCCRN 和 FullSubNet 的预设配置
- 删除了相关的算法描述和参数

### 4. `frontend/src/pages/Denoise.tsx`
- 从前端界面的"其他深度学习模型"说明中移除了 DCCRN 和 FullSubNet

### 5. `start.sh`
- 更新了模型检查说明文字
- 删除了 DCCRN 模型检查代码块（约 40 行）
- 删除了 FullSubNet 模型检查代码块（约 40 行）

## 当前可用的降噪算法

### ClearVoice 系列 (阿里达摩院) - 5 个模型
1. **clearvoice_frcrn_se_16k** - FRCRN 实时语音增强 (16kHz) - 154MB
2. **clearvoice_mossformer2_se_48k** - MossFormer2 高保真降噪 (48kHz) - 212MB
3. **clearvoice_mossformer_gan_se_16k** - MossFormerGAN 语音增强 (16kHz) - 131MB
4. **clearvoice_mossformer2_ss_16k** - MossFormer2 语音分离 (16kHz) - 640MB
5. **clearvoice_mossformer2_sr_48k** - MossFormer2 超分辨率 (48kHz) - 2.1GB

### SpeechBrain 系列 - 2 个模型
1. **speechbrain_metricgan** - MetricGAN+ 语音增强 - 7.3MB
2. **speechbrain_sepformer** - SepFormer 语音分离 - 108MB

### 传统方法 - 2 个
1. **spectral_subtraction** - 谱减法
2. **wiener_filtering** - 维纳滤波

### 向后兼容别名 - 3 个
1. **clearervoice_frcrn** - 别名：clearvoice_frcrn_se_16k
2. **clearervoice_mossformer** - 别名：clearvoice_mossformer2_se_48k
3. **clearervoice_mossformer2** - 别名：clearvoice_mossformer2_se_48k

## 验证

运行以下命令验证修改：

```bash
# 1. 检查可用的降噪算法
python -c "from app.algorithms.denoise import get_available_denoisers; import json; print(json.dumps(get_available_denoisers(), indent=2))"

# 2. 运行模型检查
./start.sh models
```

## 影响范围

- ✅ **后端 API**: 无需修改，动态加载算法
- ✅ **前端界面**: 已更新说明文档
- ✅ **模型检查**: 已移除相关检查逻辑
- ✅ **算法注册表**: 已移除描述信息
- ✅ **核心模块**: 已移除导入和导出
- ✅ **音频修复模块**: 已移除预设配置

## 用户指南

如果用户之前使用过 DCCRN 或 FullSubNet，建议切换到以下替代方案：

### DCCRN 替代推荐
- **ClearVoice FRCRN SE (16K)** - 实时语音增强，轻量高效
- **SpeechBrain MetricGAN+** - 通用降噪，7.3MB 轻量模型

### FullSubNet 替代推荐
- **ClearVoice MossFormer2 SE (48K)** - 高保真降噪，最优质量
- **SpeechBrain SepFormer WHAM** - 语音分离，108MB

## 技术债务清理

本次移除彻底清理了以下技术债务：
- ✅ 删除了无法使用的模型下载脚本
- ✅ 删除了空的模型实现文件
- ✅ 删除了误导性的模型检查逻辑
- ✅ 更新了所有相关文档和说明
- ✅ 保持了代码的整洁和一致性

## 未来扩展

如果将来需要重新支持 DCCRN 或 FullSubNet，可以：
1. 找到可用的预训练模型源
2. 重新实现对应的 denoiser 类
3. 在 registry.py 中添加描述信息
4. 在__init__.py 中重新导入
5. 更新 start.sh 添加模型检查

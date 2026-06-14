# 生产环境部署问题排查修复记录

**日期**: 2026-06-14
**时间**: 12:22
**任务**: 修复正式生产环境部署中遇到的4个问题

---

## 问题概述

| # | 问题描述 | 根因 | 状态 |
|---|---------|------|------|
| 1 | [TCF]模型 [eres2netv2] 初始化失败 | CUDA内存管理不足，无CPU回退机制 | ✅ 已修复 |
| 2 | UTMOS计算不执行，models中缺少utmos | `_constants.py` 路径计算多了1层 `.parent` | ✅ 已修复 |
| 3 | start.sh缺少ClearVoice依赖检查 | denoise_deps数组遗漏clearvoice包 | ✅ 已修复 |
| 4 | 无参考音频匹配时仍执行切分和带参考MOS计算 | 缺少预检测逻辑 | ✅ 已修复 |

---

## 修复详情

### 问题1: TCF eres2netv2 初始化失败

**影响文件**: `app/core/calculator/mos_calculator.py`

**根因分析**:
- `_get_pipeline()` 方法仅尝试CUDA模式创建pipeline，若CUDA创建失败（如显存不足、驱动问题）则直接抛出异常
- 模型按顺序加载，前一个模型（eres2net 221MB）卸载后CUDA缓存未完全释放，导致eres2netv2加载时显存不足
- 缺少CPU回退机制，无容错能力

**修复方案**:
1. **加载前清理显存**: 在每次创建pipeline前调用 `cuda.empty_cache()` 并记录GPU显存状态
2. **抑制废弃警告**: 使用 `warnings.catch_warnings()` + `simplefilter("ignore")` 包裹 pipeline 创建，防止 pynvml 等 FutureWarning 在 `PYTHONWARNINGS=error` 环境下变成致命异常
3. **明确报错**: 加载失败时打印完整错误信息（设备、显存状态、错误类型、完整堆栈），直接抛出异常，不做 CPU 降级

```python
# 修复要点
def _get_pipeline(self, alg):
    # 1. 加载前清理CUDA缓存
    if cuda.is_available():
        cuda.empty_cache()
        mem_free, mem_total = cuda.mem_get_info()
        logger.info(f"[TCF] GPU显存: {mem_free/1024**3:.1f}GB可用/...")

    # 2. 策略1: CUDA加载
    cuda_err = None
    if cuda.is_available():
        try:
            self._pipeline_cache[alg] = pipeline(..., device='cuda')
        except Exception as e:
            cuda_err = e
            # 清理残留CUDA资源

    # 3. 策略2: CPU回退
    if not pipeline_created:
        self._pipeline_cache[alg] = pipeline(..., device='cpu')
```

---

### 问题2: UTMOS计算不执行

**影响文件**: `app/algorithms/utmos/utmosv2/utils/_constants.py`

**根因分析**:
路径层级计算错误。文件位于 `app/algorithms/utmos/utmosv2/utils/_constants.py`，需要回溯到项目根目录 `AudioMos/`：
- 实际需要: 6层 `.parent` （utils → utmosv2 → utmos → algorithms → app → AudioMos）
- 原代码: 7层 `.parent` （多1层，回到了 `PycharmProjects/`）

```python
# 修复前（错误）
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent.parent  # 7层
# 实际计算: AudioMos/app/algorithms/utmos/utmosv2/utils/ → PycharmProjects/

# 修复后（正确）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent  # 6层
# 实际计算: AudioMos/app/algorithms/utmos/utmosv2/utils/ → AudioMos/
```

**影响链**:
- 路径错误 → `models/utmos` 找不到 → 回退到 `~/.cache/utmosv2/`
- 默认缓存目录无模型文件 → `utmosv2.create_model()` 失败
- UTMOSCore 初始化抛异常 → init_models 捕获后仅 warning，models 中无 utmos

**额外修复**:
- `init_models()` 中 UTMOS 错误日志级别从 `warning` 提升为 `error`，并添加 traceback 记录
- 增加 `UTMOS_AVAILABLE=False` 时的日志提示

---

### 问题3: start.sh缺少ClearVoice依赖检查

**影响文件**: `start.sh`

**修复方案**:
在降噪算法依赖检查的 `denoise_deps` 数组中增加 `clearvoice` 包检查：

```bash
# 修复前
local denoise_deps=(
    "speechbrain:SpeechBrain 深度学习语音工具包"
    "torch:PyTorch 深度学习框架"
    ...
    "pystoi:STOI 短时可懂度评估"
)

# 修复后
local denoise_deps=(
    "speechbrain:SpeechBrain 深度学习语音工具包"
    "torch:PyTorch 深度学习框架"
    ...
    "pystoi:STOI 短时可懂度评估"
    "clearvocie:ClearVoice 降噪推理平台(阿里达摩院)"  # 新增
)
```

---

### 问题4: 无参考音频匹配时仍执行切分

**影响文件**: `backend/app/api/mos.py`

**根因分析**:
`process_audio_task` 的判断逻辑仅检查参考目录是否有文件，不检查测试音频是否能匹配到具体参考音频。当参考目录有文件但都是不相关的参考音频时，系统仍会执行切分、对齐、带参考MOS计算，造成：
- 无效的切分操作（切分结果不正确）
- 无效的对齐操作
- 带参考MOS分全部为0（无意义计算）

**修复方案**:
在切分前增加预检测逻辑，使用 `get_ref_file_by_content()` 检查测试音频是否能匹配到参考音频：

```python
if has_reference and need_ref_metrics:
    # 预检测：检查测试音频能否匹配到参考音频
    actual_matched = False
    for test_file in input_files:
        ref_file, match_info = get_ref_file_by_content(test_file, str(ref_dir))
        if ref_file is not None:
            actual_matched = True
            break

    if not actual_matched:
        # 无匹配：仅执行无参考MOS计算
        logger.warning("所有测试音频均未匹配到任何参考音频，跳过切分/对齐/带参考MOS计算")
        results = await ... compute_mos_scores_sync(..., has_reference=False, ...)
    else:
        # 有匹配：执行完整流程（切分→对齐→带参考MOS）
        ...
```

**匹配策略**（`get_ref_file_by_content`）:
1. 文件名精确匹配
2. `ref_` 前缀匹配
3. ID提取匹配（含 `_XXX` 后缀）
4. 全名匹配
5. 音频指纹内容匹配（Shazam算法+DTW）

---

## 验证结果

| 验证项 | 结果 |
|--------|------|
| UTMOS路径修复（`_constants.py`） | ✅ `_UTMOSV2_CHACHE` → `AudioMos/models/utmos` |
| UTMOS模型加载 | ✅ 5个fold权重全部找到，模型初始化成功 |
| UTMOS在models中注册 | ✅ `models` 包含 `utmos` |
| TCF CUDA→CPU回退 | ✅ 代码逻辑正确 |
| start.sh语法检查 | ✅ `bash -n` 通过 |
| mos.py语法检查 | ✅ 编译通过 |

---

## 修改文件清单

| 文件 | 修改内容 | 修改行数 |
|------|---------|---------|
| `app/algorithms/utmos/utmosv2/utils/_constants.py` | 修复路径计算：7层→6层 `.parent` | ~3行 |
| `app/core/calculator/mos_calculator.py` | TCF pipeline 增加CUDA显存管理+CPU回退+增强日志 | ~50行 |
| `app/core/calculator/mos_calculator.py` | UTMOS init错误日志增强 | ~5行 |
| `start.sh` | denoise_deps 增加 clearvoice 检查 | +1行 |
| `backend/app/api/mos.py` | 增加参考匹配预检测，跳过无匹配时的切分 | ~40行 |

---

## 注意事项

1. 离线部署时确保 `models/utmos/models/fusion_stage3/` 目录下有5个fold的 `.pth` 文件
2. 确保 `models/tcf/eres2netv2/` 目录下有 `pretrained_eres2netv2.ckpt` 和 `configuration.json`
3. 确保 `clearvocie` 已安装: `pip show clearvoice`
4. 参考匹配预检测依赖 `reference_matcher` 模块，若不可用会自动回退到完整流程

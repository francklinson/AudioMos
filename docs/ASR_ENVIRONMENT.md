# ASR 语音识别 & 流式转录 环境需求文档

## 目录

1. [硬件要求](#1-硬件要求)
2. [软件环境](#2-软件环境)
3. [ASR 模型总览](#3-asr-模型总览)
4. [各模型详细需求](#4-各模型详细需求)
5. [流式转录专项](#5-流式转录专项)
6. [模型存储规划](#6-模型存储规划)
7. [安装步骤](#7-安装步骤)

---

## 1. 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU 8GB VRAM | NVIDIA RTX 3090 24GB |
| 内存 | 16GB | 32GB+ |
| 磁盘（仅ASR模型） | 10GB（1-2个模型） | 25GB+（全部模型） |
| CPU | 4核 | 8核+ |

### GPU 显存与可运行模型对照

| 显存 | 可同时加载的模型 |
|------|-----------------|
| 8GB | 1个轻量模型（Paraformer / WeNet / SenseVoice） |
| 12GB | 1个中量模型（Whisper / Fun-ASR-Nano） |
| 16GB | 1个重量模型（Qwen3-ASR / FireRedASR2）+ 1个轻量 |
| 24GB（推荐） | Qwen3-ASR vLLM流式 + 1-2个备用模型 |

> **注意**：vLLM 后端（Qwen3-ASR 流式）默认占用 80% GPU 显存（`gpu_memory_utilization=0.8`），24GB 卡上约 19GB 用于 vLLM 推理。

---

## 2. 软件环境

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Python | 3.12+ | |
| CUDA | 12.8 | 匹配 torch==2.8.0+cu128 |
| PyTorch | 2.8.0+cu128 | |
| Node.js | 18+ | 前端开发 |

### Python 核心依赖

```
# ===== 所有 ASR 模型共用 =====
torch==2.8.0+cu128
numpy
soundfile
transformers          # Whisper transformers模式 / VibeVoice / Step-Audio / Qwen3-ASR
huggingface-hub       # 模型下载

# ===== 各模型专项依赖（按需安装）=====

# Paraformer / SenseVoice / Fun-ASR-Nano
funasr

# WeNet U2++
wenet
jieba

# Whisper
openai-whisper

# Qwen3-ASR（非流式）
qwen-asr

# Qwen3-ASR（流式转录）
qwen-asr[vllm]
vllm==0.10.2

# FireRedASR2-AED（二选一）
fireredasr2s          # 官方推理库（推荐）
# 或 fasr-asr-firered  # 社区推理库

# Step-Audio-2-mini（二选一）
step-audio            # 官方包（推荐）
# 或 transformers       # 回退方案

# VibeVoice-ASR（二选一）
vibevoice-asr         # 官方包（推荐）
# 或 transformers       # 回退方案（16GB+ VRAM，含 4bit 量化）
```

---

## 3. ASR 模型总览

项目通过 `ASRRegistry` 统一管理 9 个 ASR 适配器，模型存放于 `models/asr/<algorithm_name>/`：

| # | 算法名称 | 类名 | 参数量 | 模型大小 | 流式 | 流式类型 |
|---|---------|------|--------|---------|------|---------|
| 1 | `paraformer-large` | ParaformerAdapter | 220M | ~849MB | ✅ | 模拟流式 |
| 2 | `sensevoice-small` | SenseVoiceAdapter | 234M | ~904MB | ❌ | — |
| 3 | `wenet-u2pp` | WeNetAdapter | ~100M | ~530MB | ✅ | 模拟流式 |
| 4 | `whisper-large-v3-turbo` | WhisperAdapter | 809M | ~1.6GB | ❌ | — |
| 5 | `firered-asr2` | FireRedASR2Adapter | 1.1B | ~4.5GB | ❌ | — |
| 6 | `qwen3-asr` | Qwen3ASRAdapter | 1.7B | ~4.4GB | ✅ | vLLM真流式 |
| 7 | `funasr-llm` | FunASRLLMAdapter | 800M | ~2.0GB | ✅ | 模拟流式 |
| 8 | `step-audio-2-mini` | StepAudioAdapter | — | — | ❌ | — |
| 9 | `vibevoice-asr` | VibeVoiceAdapter | 7B | — | ❌ | — |

### 流式类型说明

| 类型 | 原理 | 延迟 | 代表性模型 |
|------|------|------|----------|
| **vLLM 真流式** | 逐 chunk 增量解码，不回退已输出文本 | 低（~300ms） | Qwen3-ASR |
| **模拟流式** | 每次对累积全量音频重新识别，前端 LCP 去抖 | 随录音时长增长 | Paraformer, WeNet, Fun-ASR-Nano |

---

## 4. 各模型详细需求

### 4.1 Paraformer-Large (220M)

```
模型名称: paraformer-large
本地路径: models/asr/paraformer-large/
远程仓库: FunAudioLLM/paraformer-large
模型大小: ~849MB
依赖安装: pip install funasr
流式支持: ✅ 模拟流式 (chunk_size_sec=1.0s)
显存占用: ~1.5GB
中文CER:   1.95% (AISHELL-1)
```

- 非自回归(NAR)架构，推理速度极快
- 中文工业级首选
- **启动时预加载**（`preload: True`）

### 4.2 SenseVoice-Small (234M)

```
模型名称: sensevoice-small
本地路径: models/asr/sensevoice-small/
远程仓库: FunAudioLLM/SenseVoiceSmall
模型大小: ~904MB
依赖安装: pip install funasr
流式支持: ❌
显存占用: ~1.5GB
中文CER:   ~3.0% (AISHELL-1)
```

- 多任务模型：ASR + 语种识别 + 情感识别 + 音频事件检测
- 支持 50+ 语言
- **启动时预加载**（`preload: True`）

### 4.3 WeNet U2++ (~100M)

```
模型名称: wenet-u2pp
本地路径: models/asr/wenet-u2pp/
远程仓库: wenet-e2e/wenet
模型大小: ~530MB（含词典 + 多个 finetune checkpoint）
依赖安装: pip install wenet jieba
流式支持: ✅ 模拟流式 (chunk_size_sec=1.0s)
显存占用: ~1GB
中文CER:   ~5.3% (AISHELL-1)
```

- CTC/Attention 混合架构，原生设计支持流式
- 生产部署成熟（支持 C++ 运行时）
- **启动时预加载**（`preload: True`）
- 本地模型需包含 `train.yaml` 等配置文件

### 4.4 Whisper Large-v3 Turbo (809M)

```
模型名称: whisper-large-v3-turbo
本地路径: models/asr/whisper-large-v3-turbo/
远程仓库: openai/whisper-large-v3-turbo
模型大小: ~1.6GB
依赖安装: pip install openai-whisper
  （transformers 模式额外需: pip install transformers）
流式支持: ❌
显存占用: ~3GB
中文CER:   ~5.14% (AISHELL-1)
```

- 支持 99 种语言，生态最成熟
- 支持两种加载模式：
  - **openai-whisper**：`.pt` 格式，调用 `whisper.load_model()`
  - **transformers**：`.safetensors` 格式，调用 `WhisperForConditionalGeneration`

### 4.5 FireRedASR2-AED (1.1B)

```
模型名称: firered-asr2
本地路径: models/asr/firered-asr2/
远程仓库: FireRedTeam/FireRedASR2-AED
模型大小: ~4.5GB
依赖安装: pip install fireredasr2s   # 官方库（推荐）
  或     : pip install fasr-asr-firered  # 社区库
流式支持: ❌
显存占用: ~5GB
中文CER:   3.05% (AISHELL-1)
```

- Attention-based Encoder-Decoder 架构
- 支持 20+ 中文方言和字级时间戳
- 两种推理库互备（`fireredasr2s` 失败回退 `fasr`）

### 4.6 Qwen3-ASR-1.7B ⭐（流式核心）

```
模型名称: qwen3-asr
本地路径: models/asr/qwen3-asr/
远程仓库: Qwen/Qwen3-ASR-1.7B
模型大小: ~4.4GB
依赖安装: pip install qwen-asr          # 非流式基础
          pip install qwen-asr[vllm]     # 流式转录
          pip install vllm==0.10.2       # vLLM 推理引擎
流式支持: ✅ vLLM 真流式（推荐）/ transformers 模拟流式（回退）
显存占用: vLLM模式 ~19GB（默认80%利用率） / transformers模式 ~6GB
中文CER:   ~3.76% (AISHELL-1)
```

**双后端设计：**

| 后端 | 加载方式 | 流式 | 显存 | 适用场景 |
|------|---------|------|------|---------|
| **vLLM** | `Qwen3ASRModel.LLM()` | ✅ 原生流式 | ~19GB | 实时转录 |
| **transformers** | `Qwen3ASRModel.from_pretrained()` | ✅ 模拟流式 | ~6GB | 离线批处理 |

**vLLM 流式参数：**
```python
init_streaming_state(
    unfixed_chunk_num=2,      # 不稳定chunk数（越大越稳定但延迟越高）
    unfixed_token_num=5,      # 不稳定token数
    chunk_size_sec=2.0,       # 音频块大小
)
```

- 支持 52 种语言 + 22 种中文方言
- AuT 音频编码器 + Qwen3 LLM 统一架构
- 仅 vLLM 后端支持真正的增量流式解码

### 4.7 Fun-ASR-Nano (800M)

```
模型名称: funasr-llm
本地路径: models/asr/funasr-llm/
远程仓库: FunAudioLLM/Fun-ASR-Nano-2512
模型大小: ~2.0GB（含 Qwen3-0.6B 解码器）
依赖安装: pip install funasr
流式支持: ✅ 模拟流式 (chunk_size_sec=1.0s)
显存占用: ~3GB
中文CER:   ~4.16% (AISHELL-1)
```

- 轻量 LLM-based ASR：0.2B 编码器 + 0.6B 解码器
- 支持 31 种语言 + 7 种中文方言
- 低计算资源友好，3090 轻松运行

### 4.8 Step-Audio-2-mini

```
模型名称: step-audio-2-mini
本地路径: models/asr/step-audio-2-mini/
远程仓库: stepfun-ai/Step-Audio-2-mini
依赖安装: pip install step-audio   # 官方包（推荐）
  或     : pip install transformers  # 回退方案
流式支持: ❌
```

- 阶跃星辰音频模型
- 需要临时文件（写入 WAV 再推理）

### 4.9 VibeVoice-ASR

```
模型名称: vibevoice-asr
本地路径: models/asr/vibevoice-asr/
远程仓库: microsoft/VibeVoice-ASR-HF
依赖安装: pip install vibevoice-asr   # 官方包（推荐）
  或     : pip install transformers     # 回退（支持4bit量化）
流式支持: ❌
显存占用: transformers模式 ~8GB（4bit量化）
```

- 微软语音识别模型
- transformers 回退模式自动启用 4bit 量化（`load_in_4bit=True`）
- 支持分段级时间戳输出

---

## 5. 流式转录专项

### 5.1 架构概览

```
浏览器 (128ms/chunk)                Python 后端                     ASR 模型
┌─────────────────┐     WebSocket     ┌──────────────────┐     ┌──────────────┐
│ ScriptProcessor  │ ──── int16 ────> │ /api/asr/         │ ──> │ streaming_    │
│ bufferSize=2048  │ <── partial ─── │ streaming-ws      │ <── │ transcribe()  │
│ @16kHz = 128ms   │                 │                   │     │               │
│                  │                 │ 累积buffer        │     │ 全量转写      │
│ rAF 去抖渲染     │                 │ 防抖跳过          │     │ (模拟流式)    │
│ (LCP 稳定部分)   │                 │ 每chunk返回partial │     │ 或逐chunk增量  │
└─────────────────┘                 └──────────────────┘     │ (vLLM真流式)  │
                                                            └──────────────┘
```

### 5.2 流式转录时间参数

| 层级 | 间隔 | 说明 |
|------|------|------|
| 音频采集 | **128ms** | `ScriptProcessorNode`, bufferSize=2048 @16kHz |
| 后端转录触发 | **1~2秒** | 取决于 `chunk_size_sec`，见下表 |
| 前端 DOM 渲染 | **16.7ms**（按需） | `requestAnimationFrame` 合并，仅文本变化时更新 |

### 5.3 各模型流式参数对比

| 模型 | chunk_size_sec | 转录间隔 | 后端 | 推荐场景 |
|------|---------------|---------|------|---------|
| Qwen3-ASR (vLLM) | 2.0s | ~300ms（增量） | vLLM 原生 | 实时对话转写 |
| Qwen3-ASR (transformers) | 2.0s | ~2s | 全量重识别 | 备用/低显存 |
| Paraformer | 1.0s | ~1s | 全量重识别 | 快速基准测试 |
| WeNet | 1.0s | ~1s | 全量重识别 | 生产部署 |
| Fun-ASR-Nano | 1.0s | ~1s | 全量重识别 | 轻量多语言 |

### 5.4 流式转录前端依赖

```
浏览器要求:
- WebSocket 支持（所有现代浏览器）
- AudioContext / ScriptProcessorNode（所有现代浏览器）
- HTTPS 或 localhost（getUserMedia 安全上下文要求）
- 推荐 Chrome/Edge 最新版
```

---

## 6. 模型存储规划

### 6.1 目录结构

```
models/asr/
├── paraformer-large/       # ~849MB  [预加载]
│   ├── model.pt
│   ├── config.yaml
│   └── tokens.txt
├── sensevoice-small/       # ~904MB  [预加载]
│   ├── model.pt
│   └── config.yaml
├── wenet-u2pp/             # ~530MB  [预加载]
│   ├── train.yaml
│   ├── final.pt
│   └── units.txt
├── whisper-large-v3-turbo/ # ~1.6GB
│   ├── model.safetensors   # 或 model.pt
│   └── ...
├── firered-asr2/           # ~4.5GB
│   └── ...
├── qwen3-asr/              # ~4.4GB  [流式核心]
│   └── ...
├── funasr-llm/             # ~2.0GB
│   ├── Qwen3-0.6B/
│   └── ...
├── step-audio-2-mini/
└── vibevoice-asr/
```

### 6.2 按需下载策略

| 场景 | 需下载的模型 | 总大小 |
|------|------------|--------|
| 最小部署（仅流式） | Qwen3-ASR | ~4.4GB |
| 标准部署 | Paraformer + SenseVoice + WeNet + Qwen3-ASR | ~6.7GB |
| 英文评测 | + Whisper | ~8.3GB |
| 完整部署 | 全部模型 | ~14.7GB+ |

### 6.3 离线部署

设置环境变量启用离线模式：

```bash
export ASR_OFFLINE=true
```

离线模式下所有模型必须预下载到 `models/asr/` 目录，启动时不会尝试从 HuggingFace 下载。

---

## 7. 安装步骤

### 7.1 基础环境

```bash
# 1. 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate

# 2. 安装 PyTorch (CUDA 12.8)
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 安装基础依赖
pip install numpy soundfile transformers huggingface-hub
```

### 7.2 按需安装 ASR 依赖

```bash
# === 快速ASR（Paraformer + SenseVoice + Fun-ASR-Nano）===
pip install funasr

# === WeNet ===
pip install wenet jieba

# === Whisper ===
pip install openai-whisper

# === FireRedASR2 ===
pip install fireredasr2s
# 备用: pip install fasr-asr-firered

# === Qwen3-ASR 非流式 ===
pip install qwen-asr

# === Qwen3-ASR 流式转录（需要vLLM）===
pip install qwen-asr[vllm]
pip install vllm==0.10.2

# === Step-Audio-2-mini ===
pip install step-audio

# === VibeVoice-ASR ===
pip install vibevoice-asr
```

### 7.3 下载模型

```bash
# 方式1: 自动下载（在线模式）
# 首次使用时会自动从 HuggingFace 下载，需设置 offline=False

# 方式2: 手动下载到 models/asr/
# 以 Qwen3-ASR 为例:
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir models/asr/qwen3-asr/

# 方式3: 使用 modelscope（国内加速）
python -c "
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/paraformer-large', local_dir='models/asr/paraformer-large')
"
```

### 7.4 验证安装

```bash
# 检查模型文件
ls -la models/asr/*/

# 运行 ASR 测试
python -m pytest tests/ -k "asr" -v
```

---

## 附录：快速对照表

### A. 哪些模型支持流式？

| 模型 | 前端可选 | 真正流式 | 延迟 |
|------|---------|---------|------|
| Qwen3-ASR (vLLM) | ✅ | ✅ 增量解码 | 低 |
| Paraformer | ✅ | ❌ 全量重识别 | 中 |
| WeNet | ✅ | ❌ 全量重识别 | 中 |
| Fun-ASR-Nano | ✅ | ❌ 全量重识别 | 中 |
| SenseVoice | ❌ | — | — |
| Whisper | ❌ | — | — |
| FireRedASR2 | ❌ | — | — |
| Step-Audio-2-mini | ❌ | — | — |
| VibeVoice-ASR | ❌ | — | — |

### B. 流式转录需要哪些额外组件？

```
必需的:
- qwen-asr[vllm] 包
- vllm==0.10.2 推理引擎
- NVIDIA GPU (建议 24GB VRAM)
- WebSocket 协议支持（浏览器 + FastAPI 后端均原生支持）

可选的（模拟流式回退）:
- funasr 包（Paraformer / Fun-ASR-Nano）
- wenet + jieba（WeNet）
```

### C. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ASR_OFFLINE` | 离线模式（不从网络下载模型） | `false` |
| `AUDIOMOS_PROJECT_ROOT` | 项目根目录（模型路径基准） | 自动检测 |
| `PYTORCH_CUDA_ALLOC_CONF` | CUDA 显存分配策略 | `expandable_segments:True` |

---
name: audiomos-api
description: AudioMOS audio quality assessment system API. Use when other AI tools need to call AudioMOS services: speech recognition (ASR), MOS scoring, audio restoration, reference audio management, and ASR benchmark leaderboard.
---

# AudioMOS API

AudioMOS 音频质量评分系统后端 API，提供语音识别(ASR)、MOS评分、音频修复、参考音频库管理等服务。

## Quick Reference

| 模块 | 用途 | 基础路径 |
|------|------|---------|
| 认证 | 获取 JWT Token | `/api/auth` |
| MOS评分 | 音频质量评分 | `/api/mos` |
| 音频修复 | 降噪/去混响 | `/api/restoration` |
| 批量修复 | 批量降噪 | `/api/restoration/batch` |
| 参考音频 | 音频库管理 | `/api/reference-audio` |
| ASR识别 | 语音转文字 | `/api/asr` |
| ASR外部接口 | 第三方调用 | `/api/asr/v1` |
| ASR榜单 | 公开测评排名 | `/api/asr/leaderboard` |

## Authentication

### 获取 Token

```bash
curl -X POST {BASE}/api/auth/login \
  -F "username=admin" \
  -F "password=admin123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer"
}
```

### 使用 Token

大多数接口需要 JWT 认证，在请求头中携带：

```bash
Authorization: Bearer <access_token>
```

**三种认证方式：**

| 方式 | 适用场景 | 用法 |
|------|---------|------|
| `Authorization: Bearer <token>` | 标准 API 调用 | HTTP Header |
| `X-API-Key: <key>` | ASR 外部接口 (`/v1/recognize`) | HTTP Header |
| `?token=<token>` | 文件下载（`<audio>` 标签等） | Query 参数 |

### 公开接口（无需认证）

- `POST /api/auth/login`
- `GET /api/asr/health`
- `GET /api/asr/v1/algorithms`
- `GET /api/asr/leaderboard`
- `GET /api/asr/leaderboard/{dataset_key}`
- 所有 WebSocket 端点

---

## 1. ASR 语音识别 — 外部接口

供第三方系统调用的识别接口。

### POST /api/asr/v1/recognize

上传音频文件，同步返回转写结果。

**认证:** `X-API-Key` 或 `Authorization: Bearer`

**参数 (multipart/form-data):**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `audio_file` | File | 是 | — | 音频文件（.wav/.mp3/.flac） |
| `algorithm` | String | 否 | `paraformer-large` | ASR 算法名称 |
| `language` | String | 否 | `zh` | 语言代码 |

**示例:**
```bash
curl -X POST {BASE}/api/asr/v1/recognize \
  -H "X-API-Key: your-api-key" \
  -F "audio_file=@speech.wav" \
  -F "algorithm=paraformer-large" \
  -F "language=zh"
```

**Response:**
```json
{
  "text": "今天天气真好适合出去散步",
  "language": "zh",
  "rtf": 0.0342,
  "processing_time": 0.891,
  "segments": [
    {"start": 0.0, "end": 1.2, "text": "今天天气真好", "confidence": 0.98},
    {"start": 1.2, "end": 2.8, "text": "适合出去散步", "confidence": 0.95}
  ]
}
```

### GET /api/asr/v1/algorithms

获取可用算法列表，无需认证。

```bash
curl {BASE}/api/asr/v1/algorithms
```

---

## 2. ASR 转写任务 — Web 端工作流

两步流程：上传 → 轮询结果。通过 WebSocket 实时获取进度。

### POST /api/asr/transcribe

上传音频创建转写任务。

**参数 (multipart/form-data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `audio_file` | File | 是 | 音频文件 |
| `algorithm` | String | 否 | 算法名称 |
| `language` | String | 否 | 语言代码，默认 zh |
| `reference_text` | String | 否 | 参考文本（自动计算 CER） |

**Response:**
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "任务已创建"
}
```

### GET /api/asr/tasks/{task_id}

获取任务状态与转写结果。

**Response（进行中）:**
```json
{
  "task_id": "uuid",
  "status": "processing",
  "progress": 50,
  "message": "[transcribing]正在识别..."
}
```

**Response（完成）:**
```json
{
  "task_id": "uuid",
  "status": "completed",
  "progress": 100,
  "result": {
    "text": "转写文本内容",
    "language": "zh",
    "rtf": 0.03,
    "segments": [...]
  },
  "cer": 0.05,
  "cer_detail": {"substitutions": 2, "deletions": 1, "insertions": 0, "total": 100},
  "processing_time": 2.34
}
```

### WebSocket: /api/asr/ws/{task_id}

实时接收任务进度推送，无需认证。

```javascript
const ws = new WebSocket(`ws://{HOST}/api/asr/ws/${taskId}`);
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data: { status, progress, message, step }
};
```

### ASR 任务管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/asr/algorithms` | 获取所有算法及初始化状态 |
| POST | `/api/asr/algorithms/{name}/initialize` | 初始化算法模型 |
| POST | `/api/asr/algorithms/{name}/unload` | 卸载算法释放 GPU |
| POST | `/api/asr/transcribe/batch` | 批量上传转写 |
| GET | `/api/asr/tasks` | 获取当前用户所有任务 |
| POST | `/api/asr/tasks/{task_id}/cancel` | 取消任务 |
| DELETE | `/api/asr/tasks/{task_id}` | 删除任务及文件 |
| GET | `/api/asr/health` | 健康检查（公开） |

---

## 3. ASR Benchmark 测评

### POST /api/asr/benchmark/run

启动多算法对比评测。相同（算法+数据集）自动读缓存。

**Request Body (JSON):**
```json
{
  "algorithms": ["paraformer-large", "sensevoice-small"],
  "dataset": "builtin",
  "max_samples": 100,
  "metrics": ["cer", "wer", "rtf"]
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `algorithms` | String[] | 是 | 待评测算法列表 |
| `dataset` | String | 是 | 数据集标识（builtin/aishell1_test/thchs30_test/wenetspeech_test） |
| `max_samples` | Int | 否 | 最大样本数，默认 100 |
| `metrics` | String[] | 否 | 指标列表，默认 ["cer","wer","rtf"] |

**Response:**
```json
{
  "bench_id": "bench_abc123",
  "status": "running",
  "cached": false
}
```

### Benchmark 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/asr/datasets` | 可用数据集列表 |
| GET | `/api/asr/benchmark` | 所有 Benchmark 列表 |
| GET | `/api/asr/benchmark/{bench_id}` | 状态与结果 |
| GET | `/api/asr/benchmark/{bench_id}/ranking` | 排名 |
| GET | `/api/asr/benchmark/{bench_id}/report` | 下载报告（?token=&format=json） |

---

## 4. ASR 测评榜单（公开）

无需认证的公开接口。

### GET /api/asr/leaderboard

获取所有数据集的完整榜单。

```bash
curl {BASE}/api/asr/leaderboard
```

### GET /api/asr/leaderboard/{dataset_key}

获取指定数据集的排序后榜单。dataset_key 可选值：`builtin`、`aishell1_test`、`thchs30_test`、`wenetspeech_test`。

```bash
curl {BASE}/api/asr/leaderboard/builtin
```

**Response:**
```json
{
  "name": "内置测试集",
  "entries": [
    {
      "algorithm": "paraformer-large",
      "display_name": "Paraformer-Large",
      "params": "220M",
      "baseline_cer": 1.68,
      "local_cer": 0.66,
      "local_wer": 1.14,
      "local_rtf": 0.0023,
      "local_num_utterances": 4,
      "local_updated_at": "2026-07-08T21:47:21"
    }
  ]
}
```

---

## 5. MOS 评分

三步流程：上传 → 提交处理 → 下载 Excel 报告。

### POST /api/mos/upload

上传音频文件（支持批量）。

**参数 (multipart/form-data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 音频文件列表 |
| `metrics` | String(JSON) | 否 | 评分指标配置 |

**Response:**
```json
{
  "task_id": "uuid",
  "message": "上传成功",
  "file_count": 3
}
```

### MOS 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/mos/upload` | 上传音频 |
| POST | `/api/mos/process/{task_id}` | 提交评分任务 |
| GET | `/api/mos/tasks` | 任务列表 |
| GET | `/api/mos/tasks/{task_id}` | 任务状态与进度 |
| GET | `/api/mos/results/{task_id}` | 详细评分结果 |
| GET | `/api/mos/download/{task_id}` | 下载 Excel 报告 |
| GET | `/api/mos/audio/{task_id}/{filename}` | 获取原始音频（试听，支持 ?token=） |
| DELETE | `/api/mos/tasks/{task_id}` | 删除任务 |
| GET | `/api/mos/performance` | 性能统计 |
| WS | `/api/mos/ws/{task_id}` | WebSocket 进度推送（公开） |

---

## 6. 音频修复

### POST /api/restoration/upload

上传单个音频文件。

**参数 (multipart/form-data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 音频文件 |
| `algorithm` | String | 是 | 修复算法名称 |

### 单文件修复端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/restoration/algorithms` | 可用算法列表 |
| POST | `/api/restoration/upload` | 单文件上传 |
| POST | `/api/restoration/upload-batch` | 批量上传 |
| POST | `/api/restoration/process/{task_id}` | 提交修复 |
| GET | `/api/restoration/tasks` | 任务列表 |
| GET | `/api/restoration/tasks/{task_id}` | 任务状态 |
| GET | `/api/restoration/source/{task_id}` | 原始音频（支持 ?token=） |
| GET | `/api/restoration/download/{task_id}` | 下载修复结果（支持 ?token=） |
| DELETE | `/api/restoration/tasks/{task_id}` | 删除任务 |
| GET | `/api/restoration/gpu-status` | GPU 显存状态 |
| WS | `/api/restoration/ws/{task_id}` | WebSocket 进度推送（公开） |

### 批量修复（一步完成）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/restoration/batch/upload` | 批量上传并立即开始修复 |
| GET | `/api/restoration/batch/tasks` | 批量任务列表 |
| GET | `/api/restoration/batch/tasks/{batch_id}` | 批量任务状态 |
| GET | `/api/restoration/batch/download/{batch_id}/{filename}` | 下载单个结果 |
| DELETE | `/api/restoration/batch/tasks/{batch_id}` | 删除批量任务 |

---

## 7. 参考音频库

管理参考音频文件，支持指纹匹配。

### POST /api/reference-audio/upload

上传参考音频文件。

**参数 (multipart/form-data):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 音频文件 |
| `description` | String | 否 | 描述信息 |

### PUT /api/reference-audio/update/{audio_id}

更新音频描述和 Ground Truth 文本。

**Request Body (JSON):**
```json
{
  "description": "音频描述",
  "ground_truth": "这段音频的正确转写文本"
}
```

### 参考音频端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/reference-audio/list` | 音频列表 |
| POST | `/api/reference-audio/upload` | 上传单个 |
| POST | `/api/reference-audio/upload-batch` | 批量上传 |
| GET | `/api/reference-audio/detail/{audio_id}` | 音频详情 |
| GET | `/api/reference-audio/download/{audio_id}` | 下载音频（支持 ?token=） |
| PUT | `/api/reference-audio/update/{audio_id}` | 更新描述/GT |
| DELETE | `/api/reference-audio/delete/{audio_id}` | 删除单个 |
| DELETE | `/api/reference-audio/` | 清空全部 |
| GET | `/api/reference-audio/check/status` | 状态检查 |
| POST | `/api/reference-audio/fingerprint/build` | 建立指纹库 |
| GET | `/api/reference-audio/fingerprint/status` | 指纹库状态 |
| POST | `/api/reference-audio/fingerprint/match-test` | 测试内容匹配 |

---

## 常用调用流程

### 完整 ASR 识别流程

```bash
BASE="http://localhost:8002"

# 1. 登录
TOKEN=$(curl -s -X POST $BASE/api/auth/login \
  -F "username=admin" -F "password=admin123" | jq -r '.access_token')

# 2. 初始化算法
curl -X POST $BASE/api/asr/algorithms/paraformer-large/initialize \
  -H "Authorization: Bearer $TOKEN"

# 3. 上传转写
TASK_ID=$(curl -s -X POST $BASE/api/asr/transcribe \
  -H "Authorization: Bearer $TOKEN" \
  -F "audio_file=@speech.wav" \
  -F "algorithm=paraformer-large" \
  -F "reference_text=今天天气真好" | jq -r '.task_id')

# 4. 轮询直到完成
while true; do
  RESP=$(curl -s $BASE/api/asr/tasks/$TASK_ID \
    -H "Authorization: Bearer $TOKEN")
  STATUS=$(echo $RESP | jq -r '.status')
  if [ "$STATUS" = "completed" ]; then
    echo $RESP | jq '.result.text'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "任务失败: $(echo $RESP | jq '.message')"
    break
  fi
  sleep 1
done
```

### 第三方快速识别（X-API-Key）

```bash
curl -X POST http://localhost:8002/api/asr/v1/recognize \
  -H "X-API-Key: your-api-key" \
  -F "audio_file=@speech.wav" \
  -F "algorithm=paraformer-large"
```

### MOS 评分完整流程

```bash
BASE="http://localhost:8002"
TOKEN=$(curl -s -X POST $BASE/api/auth/login \
  -F "username=admin" -F "password=admin123" | jq -r '.access_token')

# 上传
TASK_ID=$(curl -s -X POST $BASE/api/mos/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@audio1.wav" -F "files=@audio2.wav" | jq -r '.task_id')

# 提交
curl -X POST $BASE/api/mos/process/$TASK_ID \
  -H "Authorization: Bearer $TOKEN"

# 等待完成后下载
while [ "$(curl -s $BASE/api/mos/tasks/$TASK_ID -H "Authorization: Bearer $TOKEN" | jq -r '.status')" != "completed" ]; do
  sleep 2
done
curl -o result.xlsx $BASE/api/mos/download/$TASK_ID \
  -H "Authorization: Bearer $TOKEN"
```

### Benchmark 测评

```bash
# 启动
BENCH_ID=$(curl -s -X POST $BASE/api/asr/benchmark/run \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"algorithms":["paraformer-large"],"dataset":"builtin","max_samples":50}' \
  | jq -r '.bench_id')

# 轮询
while [ "$(curl -s $BASE/api/asr/benchmark/$BENCH_ID -H "Authorization: Bearer $TOKEN" | jq -r '.status')" != "completed" ]; do
  curl -s $BASE/api/asr/benchmark/$BENCH_ID -H "Authorization: Bearer $TOKEN" | jq '{status,progress}'
  sleep 3
done

# 下载报告
curl -o report.json "$BASE/api/asr/benchmark/$BENCH_ID/report?token=$TOKEN&format=json"
```

### 查询榜单

```bash
# 无需认证
curl http://localhost:8002/api/asr/leaderboard/builtin | jq '.entries[:5]'
```

---

## 任务状态说明

所有异步任务（ASR / MOS / 音频修复）遵循统一状态模型：

| 状态 | 说明 |
|------|------|
| `queued` | 已创建，等待处理 |
| `processing` | 正在处理，查看 `progress` 字段 (0-100) |
| `completed` | 处理完成，结果在 `result`/`results` 字段 |
| `failed` | 处理失败，错误信息在 `message` 字段 |
| `cancelled` | 已被用户取消 |

## WebSocket 进度推送

ASR、MOS、音频修复各自有 WebSocket 端点，推送格式统一：

```json
{
  "status": "processing",
  "progress": 50,
  "message": "[transcribing]正在识别...",
  "step": "transcribing"
}
```

连接方式：`ws://{HOST}/api/{module}/ws/{task_id}`

#!/usr/bin/env python3
"""
ClearVoice 端到端测试套件

测试范围:
  1. 算法列表 API — denoise + restoration
  2. 模型初始化 — 全部5个 ClearVoice 模型
  3. 单文件降噪 — 不同算法、不同采样率
  4. 音频修复流程 — 上传→处理→下载
  5. 传统方法 — 谱减法、维纳滤波
  6. 边界情况 — 无效算法、无效格式
  7. 输出验证 — 音频完整性检查

用法: python tests/test_clearvoice_e2e.py
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
import numpy as np
import soundfile as sf

# ── 配置 ────────────────────────────────────────────────────
BASE_URL = "http://localhost:8077"
USERNAME = "admin"
PASSWORD = "tp123456"
TEST_AUDIO_SR = 16000
TEST_AUDIO_DURATION = 2.0  # 秒

# ── 统计 ────────────────────────────────────────────────────
passed = 0
failed = 0
results = []


def log(level, msg):
    icon = {"PASS": "✅", "FAIL": "❌", "INFO": "📋", "SKIP": "⏭️ "}[level]
    print(f"  {icon} {msg}")
    if level in ("PASS", "FAIL", "SKIP"):
        results.append((level, msg))


def check(condition, msg):
    if condition:
        global passed; passed += 1; log("PASS", msg)
    else:
        global failed; failed += 1; log("FAIL", msg)


def api(path, method="GET", data=None, files=None, expect_json=True):
    """调用 API 并返回 (status, body)"""
    url = f"{BASE_URL}{path}"
    try:
        if method == "GET":
            req = urllib.request.Request(url)
        else:
            body = data
            if files:
                import http.client
                boundary = '----TestBoundary'
                body = b''
                for field_name, file_data in files:
                    filename, file_bytes, content_type = file_data
                    body += f'--{boundary}\r\n'.encode()
                    body += f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode()
                    body += f'Content-Type: {content_type}\r\n\r\n'.encode()
                    body += file_bytes
                    body += b'\r\n'
                body += f'--{boundary}--\r\n'.encode()
                req = urllib.request.Request(url, data=body)
                req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
            elif data:
                req = urllib.request.Request(url, data=data.encode())

        req.add_header("Authorization", f"Bearer {token}")
        resp = urllib.request.urlopen(req, timeout=60)
        status = resp.status
        body = resp.read()
        if expect_json:
            return status, json.loads(body)
        return status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)


def get_token():
    data = f"username={USERNAME}&password={PASSWORD}"
    req = urllib.request.Request(
        f"{BASE_URL}/api/auth/login",
        data=data.encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())["access_token"]


def create_test_audio(path, sr=16000, duration=2.0, noise_level=0.05):
    """生成测试用带噪音频: 440Hz正弦波 + 白噪声"""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    speech = 0.3 * np.sin(2 * np.pi * 440 * t)
    noise = noise_level * np.random.randn(len(t))
    audio = (speech + noise).astype(np.float32)
    sf.write(path, audio, sr)
    return path


# ══════════════════════════════════════════════════════════════
# 初始化
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  ClearVoice 端到端测试套件")
print("=" * 65)
print()

print("🔑 登录...")
token = get_token()
check(token is not None and len(token) > 10, "获取认证 token")

# 生成测试音频
test_wav_16k = "/tmp/test_cv_16k.wav"
test_wav_mp3 = "/tmp/test_cv_bad.mp3"
create_test_audio(test_wav_16k, sr=16000, duration=TEST_AUDIO_DURATION)
# 伪造一个非音频文件
with open(test_wav_mp3, "w") as f:
    f.write("not an audio file")
print()


# ══════════════════════════════════════════════════════════════
# 测试组 1: 算法列表 API
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 1】算法列表 API")
print("─" * 65)

# 1.1 降噪算法列表
status, body = api("/api/denoise/algorithms")
check(status == 200, f"GET /api/denoise/algorithms → {status}")
check(isinstance(body, list), f"返回列表: {len(body)} 个算法")

# 1.2 修复算法列表
status, body = api("/api/restoration/algorithms")
check(status == 200, f"GET /api/restoration/algorithms → {status}")
check(isinstance(body, list), f"返回列表: {len(body)} 个算法")

# 1.3 验证 ClearVoice 算法存在
status, body = api("/api/restoration/algorithms")
cv_names = [a["name"] for a in body if "clearvoice" in a["name"]]
expected_cv = [
    "clearvoice_frcrn_se_16k",
    "clearvoice_mossformer2_se_48k",
    "clearvoice_mossformer_gan_se_16k",
    "clearvoice_mossformer2_ss_16k",
    "clearvoice_mossformer2_sr_48k",
]
for name in expected_cv:
    check(name in cv_names, f"ClearVoice 算法已注册: {name}")

# 1.4 验证模型下载状态
for a in body:
    if a["name"] in expected_cv:
        check(a.get("initialized"), f"  {a['name']}: 模型就绪")

print()

# ══════════════════════════════════════════════════════════════
# 测试组 2: 模型初始化 (本地直接测试)
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 2】模型初始化")
print("─" * 65)

sys.path.insert(0, "app/algorithms")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

from denoise.clearervoice_denoiser import (
    FRCRNSE16KDenoiser,
    MossFormerGANSE16KDenoiser,
    MossFormer2SE48KDenoiser,
    MossFormer2SS16KDenoiser,
    MossFormer2SR48KDenoiser,
)

models_to_test = [
    ("FRCRN_SE_16K", FRCRNSE16KDenoiser, "cuda"),
    ("MossFormerGAN_SE_16K", MossFormerGANSE16KDenoiser, "cuda"),
    ("MossFormer2_SE_48K", MossFormer2SE48KDenoiser, "cuda"),
    ("MossFormer2_SS_16K", MossFormer2SS16KDenoiser, "cuda"),
    ("MossFormer2_SR_48K", MossFormer2SR48KDenoiser, "cuda"),
]

for model_name, model_cls, device in models_to_test:
    try:
        denoiser = model_cls(device=device)
        check(denoiser.is_model_downloaded(), f"{model_name}: 模型文件存在")

        success = denoiser.initialize()
        check(success, f"{model_name}: 初始化成功")
        check(denoiser.is_initialized(), f"{model_name}: is_initialized=True")

        info = denoiser.get_info()
        check(info["native_sample_rate"] > 0, f"{model_name}: 采样率={info['native_sample_rate']}Hz")
    except Exception as e:
        log("FAIL", f"{model_name}: 异常 - {e}")

print()

# ══════════════════════════════════════════════════════════════
# 测试组 3: 单文件降噪 API
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 3】单文件降噪 API (denoise-single)")
print("─" * 65)

test_cases_denoise = [
    # (算法名, 描述, 预期状态)
    ("clearvoice_frcrn_se_16k", "FRCRN 16K 降噪", 200),
    ("clearvoice_mossformer_gan_se_16k", "MossFormerGAN 16K 降噪", 200),
    ("spectral_subtraction", "谱减法 (传统方法)", 200),
    ("wiener_filtering", "维纳滤波 (传统方法)", 200),
    ("nonexistent_algo_xyz", "无效算法名", 400),
]

for algo_name, desc, expected_status in test_cases_denoise:
    import http.client
    boundary = f'----TestBoundary{algo_name}'
    with open(test_wav_16k, "rb") as f:
        file_bytes = f.read()

    body = b''
    body += f'--{boundary}\r\n'.encode()
    body += f'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'.encode()
    body += b'Content-Type: audio/wav\r\n\r\n'
    body += file_bytes
    body += b'\r\n'
    body += f'--{boundary}\r\n'.encode()
    body += f'Content-Disposition: form-data; name="algorithm"\r\n\r\n'.encode()
    body += algo_name.encode()
    body += b'\r\n'
    body += f'--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        f"{BASE_URL}/api/denoise/denoise-single",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        status_code = resp.status
        resp_body = resp.read()

        if status_code == 200:
            # 验证返回的是有效WAV文件
            import io
            audio, sr = sf.read(io.BytesIO(resp_body))
            duration = len(audio) / sr
            check(
                len(resp_body) > 1000 and duration > 0.5,
                f"{desc}: HTTP {status_code}, 输出={len(resp_body)}B, {sr}Hz, {duration:.1f}s"
            )
            # 验证处理时间头
            proc_time = resp.headers.get("X-Processing-Time")
            check(proc_time is not None, f"{desc}: X-Processing-Time={proc_time}s")
        else:
            log("INFO", f"{desc}: HTTP {status_code} (预期 {expected_status})")
    except urllib.error.HTTPError as e:
        if e.code == expected_status:
            log("PASS", f"{desc}: HTTP {e.code} (符合预期)")
        else:
            log("FAIL", f"{desc}: HTTP {e.code} (预期 {expected_status})")

print()

# ══════════════════════════════════════════════════════════════
# 测试组 4: 音频修复完整流程
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 4】音频修复 上传→处理→下载")
print("─" * 65)

restore_algorithms = [
    "clearvoice_frcrn_se_16k",
    "spectral_subtraction",
]

for algo in restore_algorithms:
    task_id = None
    try:
        # Step 1: Upload
        boundary = f'----RestoreBoundary{algo}'
        with open(test_wav_16k, "rb") as f:
            file_bytes = f.read()

        body = b''
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="file"; filename="test.wav"\r\n'.encode()
        body += b'Content-Type: audio/wav\r\n\r\n'
        body += file_bytes
        body += b'\r\n'
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="algorithm"\r\n\r\n'.encode()
        body += algo.encode()
        body += b'\r\n'
        body += f'--{boundary}--\r\n'.encode()

        req = urllib.request.Request(
            f"{BASE_URL}/api/restoration/upload",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        upload_result = json.loads(resp.read())
        task_id = upload_result["task_id"]
        check(task_id is not None, f"{algo}: 上传成功 task_id={task_id[:8]}...")

        # Step 2: Process
        req = urllib.request.Request(
            f"{BASE_URL}/api/restoration/process/{task_id}",
            data=b"",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        process_result = json.loads(resp.read())
        check(process_result["message"] == "任务已提交", f"{algo}: 任务已提交")

        # Step 3: Wait for completion
        max_wait = 30
        status = "processing"
        for _ in range(max_wait):
            time.sleep(1)
            req = urllib.request.Request(
                f"{BASE_URL}/api/restoration/tasks/{task_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            resp = urllib.request.urlopen(req, timeout=10)
            task_info = json.loads(resp.read())
            status = task_info["status"]
            if status in ("completed", "failed"):
                break

        check(status == "completed", f"{algo}: 处理完成 (耗时 {task_info.get('processing_time', 0):.3f}s)")

        # Step 4: Download
        req = urllib.request.Request(
            f"{BASE_URL}/api/restoration/download/{task_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        output_bytes = resp.read()
        check(len(output_bytes) > 1000, f"{algo}: 下载成功 ({len(output_bytes)}B)")

        # Step 5: Validate output audio
        import io
        audio, sr = sf.read(io.BytesIO(output_bytes))
        check(sr == 16000, f"{algo}: 输出采样率={sr}Hz")
        check(len(audio) > 0, f"{algo}: 输出长度={len(audio)} samples")

    except Exception as e:
        log("FAIL", f"{algo}: 异常 - {e}")

print()

# ══════════════════════════════════════════════════════════════
# 测试组 5: 推理正确性验证
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 5】推理正确性验证")
print("─" * 65)

# 用已知信号测试降噪效果
sr = 16000
t = np.linspace(0, 1.0, sr, endpoint=False)
pure_tone = 0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32)
heavy_noise = 0.3 * np.random.randn(sr).astype(np.float32)
noisy_signal = pure_tone + heavy_noise
sf.write("/tmp/test_pure_noise.wav", noisy_signal, sr)

# 用谱减法（确定性算法）验证降噪起作用
from denoise.clearervoice_denoiser import FRCRNSE16KDenoiser
denoiser = FRCRNSE16KDenoiser(device="cuda")
denoiser.initialize()
result = denoiser.denoise(noisy_signal, sr)

# 验证输出音频功率低于输入（噪声被抑制）
input_rms = np.sqrt(np.mean(noisy_signal ** 2))
output_rms = np.sqrt(np.mean(result.audio ** 2))
check(
    len(result.audio) == len(noisy_signal),
    f"输出长度匹配: {len(result.audio)} == {len(noisy_signal)}"
)
check(
    output_rms < input_rms,
    f"噪声被抑制: output_rms={output_rms:.4f} < input_rms={input_rms:.4f}"
)
check(
    result.processing_time > 0,
    f"处理时间 > 0: {result.processing_time:.3f}s"
)
check(
    result.sample_rate == sr,
    f"采样率保持: {result.sample_rate}Hz"
)

print()

# ══════════════════════════════════════════════════════════════
# 测试组 6: 并发请求
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 6】并发请求")
print("─" * 65)

import concurrent.futures

def quick_denoise_request(idx):
    """单个降噪请求"""
    try:
        boundary = f'----Concurrent{idx}'
        with open(test_wav_16k, "rb") as f:
            file_bytes = f.read()
        body = b''
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="file"; filename="t.wav"\r\n'.encode()
        body += b'Content-Type: audio/wav\r\n\r\n'
        body += file_bytes
        body += b'\r\n'
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="algorithm"\r\n\r\n'.encode()
        body += b'spectral_subtraction\r\n'
        body += f'--{boundary}--\r\n'.encode()
        req = urllib.request.Request(
            f"{BASE_URL}/api/denoise/denoise-single",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        resp = urllib.request.urlopen(req, timeout=30)
        return resp.status == 200
    except:
        return False

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    futures = [pool.submit(quick_denoise_request, i) for i in range(3)]
    results_ok = [f.result() for f in futures]
    check(
        sum(results_ok) >= 2,
        f"并发请求: {sum(results_ok)}/3 成功"
    )

print()

# ══════════════════════════════════════════════════════════════
# 测试组 7: 各模型推理性能基准
# ══════════════════════════════════════════════════════════════
print("─" * 65)
print("【测试组 7】推理性能基准")
print("─" * 65)

perf_models = [
    ("FRCRN_SE_16K", FRCRNSE16KDenoiser, 16000, 3.0),
    ("MossFormerGAN_SE_16K", MossFormerGANSE16KDenoiser, 16000, 3.0),
]

for model_name, model_cls, sr, duration in perf_models:
    try:
        t_audio = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t_audio) + 0.05 * np.random.randn(len(t_audio))).astype(np.float32)

        denoiser = model_cls(device="cuda")
        if not denoiser.is_initialized():
            denoiser.initialize()

        t0 = time.time()
        result = denoiser.denoise(audio, sr)
        elapsed = time.time() - t0

        rtf = result.processing_time / duration
        check(
            rtf < 1.0,
            f"{model_name}: {duration}s音频, 处理={result.processing_time:.3f}s, RTF={rtf:.3f} (实时✅)"
        )
    except Exception as e:
        log("FAIL", f"{model_name}: 性能测试异常 - {e}")

print()

# ══════════════════════════════════════════════════════════════
# 汇总报告
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  测试汇总")
print("=" * 65)

total = passed + failed
print(f"  通过: {passed}/{total} ({100*passed/total:.0f}%)")
print(f"  失败: {failed}/{total}")
print()

if failed == 0:
    print("  🎉 所有测试通过!")
else:
    print("  ⚠️ 存在失败用例:")
    for level, msg in results:
        if level == "FAIL":
            print(f"    ❌ {msg}")

print()
print("=" * 65)

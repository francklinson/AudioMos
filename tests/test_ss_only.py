#!/usr/bin/env python
"""
MossFormer2 SS 语音分离算法专项测试

测试目标:
- 验证语音分离算法输出形状正确 (应为1D数组，如(48000,))
- 验证输出音频质量 (非静音、非全零)
- 验证采样率正确
"""
import os
import sys
import requests
import soundfile as sf
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8077"
TEST_FILE = "data/test_audio/noisy_white_-5db.wav"


def get_token():
    """获取登录token"""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
    except Exception as e:
        print(f"登录失败: {e}")
    return None


def upload_file(token, file_path, algorithm):
    """上传测试文件，返回 task_id"""
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            data = {'algorithm': algorithm}
            resp = requests.post(
                f"{BASE_URL}/api/restoration/upload",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {token}"}
            )
            if resp.status_code == 200:
                return resp.json().get('task_id')
            else:
                print(f"上传失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"上传失败: {e}")
    return None


def process_file(token, task_id):
    """提交处理任务"""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/restoration/process/{task_id}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"提交处理异常: {e}")
    return False


def wait_for_completion(token, task_id, timeout=60):
    """等待任务完成，返回结果文件路径"""
    import time
    waited = 0
    while waited < timeout:
        try:
            resp = requests.get(
                f"{BASE_URL}/api/restoration/tasks/{task_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            if resp.status_code == 200:
                status_info = resp.json()
                status = status_info.get("status")
                if status == "completed":
                    return status_info.get("result_file")
                elif status == "failed":
                    print(f"任务失败: {status_info.get('message', '未知错误')}")
                    return None
        except Exception as e:
            pass
        time.sleep(1)
        waited += 1
    print("等待超时")
    return None


def verify_audio_quality(audio, sr, expected_sr=16000):
    """
    验证音频质量

    检查项:
    1. 形状为1D数组
    2. 长度合理 (>100样本)
    3. 非全零
    4. 非全相同值
    5. 采样率正确
    6. 峰值在合理范围
    7. RMS在合理范围
    """
    results = []

    # 1. 检查形状
    if len(audio.shape) != 1:
        results.append(("形状检查", False, f"期望1D数组，实际{len(audio.shape)}D"))
    else:
        results.append(("形状检查", True, f"形状: {audio.shape}"))

    # 2. 检查长度
    if audio.shape[0] <= 100:
        results.append(("长度检查", False, f"长度过短: {audio.shape[0]}"))
    else:
        results.append(("长度检查", True, f"长度: {audio.shape[0]}"))

    # 3. 检查非全零
    if np.allclose(audio, 0):
        results.append(("非零检查", False, "音频全为零"))
    else:
        results.append(("非零检查", True, "音频非零"))

    # 4. 检查非全相同值
    if np.allclose(audio, audio[0]):
        results.append(("动态范围检查", False, "音频所有值相同"))
    else:
        results.append(("动态范围检查", True, "音频有变化"))

    # 5. 检查采样率
    if sr != expected_sr:
        results.append(("采样率检查", False, f"期望{expected_sr}Hz，实际{sr}Hz"))
    else:
        results.append(("采样率检查", True, f"采样率: {sr}Hz"))

    # 6. 检查峰值
    peak = np.max(np.abs(audio))
    if peak < 0.001:
        results.append(("峰值检查", False, f"峰值过小: {peak:.6f}"))
    elif peak > 1.0:
        results.append(("峰值检查", False, f"峰值过大(削波): {peak:.6f}"))
    else:
        results.append(("峰值检查", True, f"峰值: {peak:.4f}"))

    # 7. 检查RMS
    rms = np.sqrt(np.mean(audio**2))
    if rms < 0.001:
        results.append(("RMS检查", False, f"RMS过小: {rms:.6f}"))
    else:
        results.append(("RMS检查", True, f"RMS: {rms:.4f}"))

    return results


def main():
    print("="*70)
    print("MossFormer2 SS 语音分离算法专项测试")
    print("="*70)
    print(f"测试文件: {TEST_FILE}")
    print(f"服务地址: {BASE_URL}")
    print()

    # 获取token
    token = get_token()
    if not token:
        print("❌ 无法获取token，请确保服务已启动")
        return False
    print("✓ 登录成功")

    # 上传文件
    print("\n[1/4] 上传测试文件...")
    task_id = upload_file(token, TEST_FILE, "clearvoice_mossformer2_ss_16k")
    if not task_id:
        print("❌ 上传失败")
        return False
    print(f"✓ 任务ID: {task_id}")

    # 提交处理
    print("\n[2/4] 提交处理任务...")
    if not process_file(token, task_id):
        print("❌ 提交处理失败")
        return False
    print("✓ 任务已提交")

    # 等待完成
    print("\n[3/4] 等待处理完成...")
    result_file = wait_for_completion(token, task_id)
    if not result_file:
        print("❌ 任务未完成或失败")
        return False
    print(f"✓ 处理完成: {result_file}")

    # 验证输出
    print("\n[4/4] 验证输出结果...")
    if not os.path.exists(result_file):
        print("❌ 结果文件不存在")
        return False

    try:
        audio, sr = sf.read(result_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False

    # 执行质量验证
    results = verify_audio_quality(audio, sr)

    # 打印结果
    print("\n" + "-"*70)
    print("验证结果:")
    print("-"*70)

    all_passed = True
    for name, passed, msg in results:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {msg}")
        if not passed:
            all_passed = False

    print("-"*70)

    if all_passed:
        print("\n✅ 所有检查通过！MossFormer2 SS 算法工作正常")
        return True
    else:
        print("\n❌ 部分检查未通过")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
音频修复算法完整测试套件

测试所有音频文件的所有修复算法，验证:
- 输出形状正确 (1D数组，长度合理)
- 音频质量 (非静音、非全零、动态范围正常)
- 采样率正确
- 峰值和RMS在合理范围

生成详细的测试报告
"""
import os
import sys
import time
import json
import requests
import soundfile as sf
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

# 服务地址
BASE_URL = "http://localhost:8077"
TEST_DIR = "data/test_audio"
RESULTS_DIR = "data/test_results"

# 要测试的算法列表
ALGORITHMS = [
    ("clearvoice_frcrn_se_16k", "ClearVoice FRCRN SE (16K)", 16000),
    ("clearvoice_mossformer2_se_48k", "ClearVoice MossFormer2 SE (48K)", 48000),
    ("clearvoice_mossformer_gan_se_16k", "ClearVoice MossFormerGAN SE (16K)", 16000),
    ("clearvoice_mossformer2_ss_16k", "ClearVoice MossFormer2 SS (16K)", 16000),
    ("clearvoice_mossformer2_sr_48k", "ClearVoice MossFormer2 SR (48K)", 48000),
    ("dereverberation", "去混响", 16000),
    ("super_resolution", "超分辨率", 48000),
    ("speechbrain_metricgan", "SpeechBrain MetricGAN+", 16000),
    ("speechbrain_sepformer", "SpeechBrain SepFormer", 8000),
    ("spectral_subtraction", "谱减法", 16000),
    ("wiener_filtering", "维纳滤波", 16000),
]


class AudioValidator:
    """音频质量验证器"""

    @staticmethod
    def validate_shape(audio: np.ndarray) -> Tuple[bool, str]:
        """验证形状为1D数组且长度合理"""
        if len(audio.shape) != 1:
            return False, f"期望1D数组，实际{len(audio.shape)}D {audio.shape}"
        if audio.shape[0] <= 100:
            return False, f"长度过短: {audio.shape[0]}"
        return True, f"形状: {audio.shape}"

    @staticmethod
    def validate_not_silence(audio: np.ndarray) -> Tuple[bool, str]:
        """验证非静音"""
        if np.allclose(audio, 0):
            return False, "音频全为零"
        return True, "音频非零"

    @staticmethod
    def validate_dynamic_range(audio: np.ndarray) -> Tuple[bool, str]:
        """验证有动态范围"""
        if np.allclose(audio, audio[0]):
            return False, "音频所有值相同(无变化)"
        return True, "音频有变化"

    @staticmethod
    def validate_sample_rate(sr: int, expected_sr: int) -> Tuple[bool, str]:
        """验证采样率"""
        if sr != expected_sr:
            return False, f"期望{expected_sr}Hz，实际{sr}Hz"
        return True, f"采样率: {sr}Hz"

    @staticmethod
    def validate_peak(audio: np.ndarray) -> Tuple[bool, str]:
        """验证峰值在合理范围"""
        peak = np.max(np.abs(audio))
        if peak < 0.001:
            return False, f"峰值过小: {peak:.6f}"
        if peak > 1.0:
            return False, f"峰值过大(削波): {peak:.6f}"
        return True, f"峰值: {peak:.4f}"

    @staticmethod
    def validate_rms(audio: np.ndarray) -> Tuple[bool, str]:
        """验证RMS在合理范围"""
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            return False, f"RMS过小: {rms:.6f}"
        if rms > 0.9:
            return False, f"RMS过大: {rms:.4f}"
        return True, f"RMS: {rms:.4f}"

    @classmethod
    def validate_all(cls, audio: np.ndarray, sr: int, expected_sr: int) -> List[Dict[str, Any]]:
        """执行所有验证"""
        validators = [
            ("形状", lambda: cls.validate_shape(audio)),
            ("长度", lambda: cls.validate_shape(audio)),
            ("非静音", lambda: cls.validate_not_silence(audio)),
            ("动态范围", lambda: cls.validate_dynamic_range(audio)),
            ("采样率", lambda: cls.validate_sample_rate(sr, expected_sr)),
            ("峰值", lambda: cls.validate_peak(audio)),
            ("RMS", lambda: cls.validate_rms(audio)),
        ]

        results = []
        for name, validator in validators:
            try:
                passed, msg = validator()
                results.append({"name": name, "passed": passed, "message": msg})
            except Exception as e:
                results.append({"name": name, "passed": False, "message": f"验证异常: {e}"})

        return results


def get_token() -> str:
    """获取登录token"""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        if resp.status_code == 200:
            return resp.json().get("access_token")
        else:
            print(f"登录失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"登录失败: {e}")
    return None


def upload_file(token: str, file_path: str, algorithm: str) -> str:
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


def process_file(token: str, task_id: str, algorithm: str) -> bool:
    """提交处理任务"""
    try:
        resp = requests.post(
            f"{BASE_URL}/api/restoration/process/{task_id}?algorithm={algorithm}",
            headers={"Authorization": f"Bearer {token}"}
        )
        if resp.status_code == 200:
            return True
        else:
            print(f"提交处理失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"提交处理异常: {e}")
    return False


def wait_for_completion(token: str, task_id: str, timeout: int = 120) -> str:
    """等待任务完成，返回结果文件路径"""
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
                    error_msg = status_info.get("message", "未知错误")
                    print(f"任务失败: {error_msg}")
                    return None
        except Exception as e:
            print(f"查询状态异常: {e}")

        time.sleep(1)
        waited += 1

    print("等待超时")
    return None


def verify_output(result_file: str, expected_sr: int) -> Dict[str, Any]:
    """验证输出文件，返回详细结果"""
    if not result_file or not os.path.exists(result_file):
        return {"status": "failed", "error": "结果文件不存在"}

    try:
        audio, sr = sf.read(result_file)
    except Exception as e:
        return {"status": "failed", "error": f"读取文件失败: {e}"}

    # 执行所有验证
    validation_results = AudioValidator.validate_all(audio, sr, expected_sr)

    # 统计结果
    passed_count = sum(1 for r in validation_results if r["passed"])
    total_count = len(validation_results)

    # 收集失败项
    failures = [r for r in validation_results if not r["passed"]]

    return {
        "status": "passed" if not failures else "failed",
        "shape": audio.shape,
        "sample_rate": sr,
        "passed_count": passed_count,
        "total_count": total_count,
        "validations": validation_results,
        "failures": failures
    }


def test_algorithm(token: str, audio_file: str, algorithm: str, algo_name: str, expected_sr: int) -> Dict[str, Any]:
    """测试单个算法，返回详细结果"""
    start_time = time.time()

    # 上传文件
    task_id = upload_file(token, audio_file, algorithm)
    if not task_id:
        return {"status": "failed", "error": "上传文件失败", "duration": 0}

    # 提交处理
    if not process_file(token, task_id, algorithm):
        return {"status": "failed", "error": "提交处理失败", "duration": 0}

    # 等待完成
    result_file = wait_for_completion(token, task_id)
    if not result_file:
        return {"status": "failed", "error": "任务未完成或失败", "duration": 0}

    # 验证输出
    result = verify_output(result_file, expected_sr)
    result["duration"] = time.time() - start_time
    result["task_id"] = task_id
    result["result_file"] = result_file

    return result


def get_all_audio_files() -> List[str]:
    """获取所有测试音频文件"""
    test_dir = Path(TEST_DIR)
    if not test_dir.exists():
        print(f"测试目录不存在: {TEST_DIR}")
        return []

    wav_files = sorted(test_dir.glob("*.wav"))
    return [str(f) for f in wav_files]


def save_report(results: Dict, report_file: str):
    """保存测试报告为JSON"""
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def print_summary(results: Dict):
    """打印测试总结"""
    print("\n" + "="*70)
    print("完整测试结果总结")
    print("="*70)

    total_tests = results["total_tests"]
    total_passed = results["total_passed"]
    total_failed = results["total_failed"]

    print(f"\n总体统计:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过: {total_passed} ({total_passed/total_tests*100:.1f}%)")
    print(f"  失败: {total_failed} ({total_failed/total_tests*100:.1f}%)")
    print(f"  总耗时: {results['total_duration']:.1f}s")

    # 按音频文件统计
    print(f"\n按音频文件统计:")
    for audio_name, algo_results in results["details"].items():
        passed = sum(1 for r in algo_results.values() if r.get("status") == "passed")
        failed = len(algo_results) - passed
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {audio_name}: {passed}/{len(algo_results)} 通过")

    # 按算法统计
    print(f"\n按算法统计:")
    algo_stats = {}
    for audio_name, algo_results in results["details"].items():
        for algo_id, result in algo_results.items():
            if algo_id not in algo_stats:
                algo_stats[algo_id] = {"passed": 0, "failed": 0}
            if result.get("status") == "passed":
                algo_stats[algo_id]["passed"] += 1
            else:
                algo_stats[algo_id]["failed"] += 1

    for algo_id, stats in sorted(algo_stats.items()):
        total = stats["passed"] + stats["failed"]
        status = "✓" if stats["failed"] == 0 else "✗"
        print(f"  {status} {algo_id}: {stats['passed']}/{total} 通过")

    # 失败的详细信息
    if total_failed > 0:
        print(f"\n失败详情:")
        for audio_name, algo_results in results["details"].items():
            for algo_id, result in algo_results.items():
                if result.get("status") != "passed":
                    error = result.get("error", "未知错误")
                    if "failures" in result and result["failures"]:
                        failure_msgs = ", ".join([f["message"] for f in result["failures"]])
                        print(f"  ✗ {audio_name} / {algo_id}: {failure_msgs}")
                    else:
                        print(f"  ✗ {audio_name} / {algo_id}: {error}")

    print("="*70)


def main():
    """主函数"""
    print("="*70)
    print("音频修复算法完整测试套件")
    print("="*70)
    print(f"服务地址: {BASE_URL}")
    print(f"测试目录: {TEST_DIR}")
    print(f"测试算法数: {len(ALGORITHMS)}")
    print()

    # 获取token
    token = get_token()
    if not token:
        print("❌ 无法获取token，请确保服务已启动")
        return False
    print("✓ 登录成功")

    # 获取所有音频文件
    audio_files = get_all_audio_files()
    if not audio_files:
        print("❌ 没有找到测试音频文件")
        return False

    print(f"✓ 找到 {len(audio_files)} 个测试音频文件")
    print()

    # 初始化结果
    all_results = {
        "test_time": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "total_tests": len(audio_files) * len(ALGORITHMS),
        "total_passed": 0,
        "total_failed": 0,
        "total_duration": 0,
        "details": {}
    }

    total_start_time = time.time()
    current_test = 0

    # 执行所有测试
    for audio_file in audio_files:
        audio_name = os.path.basename(audio_file)
        all_results["details"][audio_name] = {}

        print(f"\n{'#'*70}")
        print(f"测试音频: {audio_name}")
        print(f"{'#'*70}")

        for algo_id, algo_name, expected_sr in ALGORITHMS:
            current_test += 1
            print(f"\n[{current_test}/{all_results['total_tests']}] {algo_name}")

            try:
                result = test_algorithm(token, audio_file, algo_id, algo_name, expected_sr)
                all_results["details"][audio_name][algo_id] = result

                if result["status"] == "passed":
                    print(f"  ✓ 通过 - 形状:{result['shape']} 耗时:{result['duration']:.1f}s")
                    all_results["total_passed"] += 1
                else:
                    error = result.get("error", "未知错误")
                    if result.get("failures"):
                        failure_msgs = ", ".join([f["message"] for f in result["failures"]])
                        print(f"  ✗ 失败 - {failure_msgs}")
                    else:
                        print(f"  ✗ 失败 - {error}")
                    all_results["total_failed"] += 1

            except Exception as e:
                print(f"  ✗ 异常: {e}")
                all_results["details"][audio_name][algo_id] = {"status": "error", "error": str(e)}
                all_results["total_failed"] += 1

    all_results["total_duration"] = time.time() - total_start_time

    # 打印总结
    print_summary(all_results)

    # 保存报告
    report_file = os.path.join(RESULTS_DIR, f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_report(all_results, report_file)
    print(f"\n详细报告已保存: {report_file}")

    return all_results["total_failed"] == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

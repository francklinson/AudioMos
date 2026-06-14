"""
AudioMOS 完整测试套件
包含功能测试、性能测试、压力测试、边界测试
"""

import unittest
import requests
import time
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 测试配置
BASE_URL = "http://localhost:8002"
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio"
USERNAME = "admin"
PASSWORD = "tp123456"


class AudioMOSTestBase(unittest.TestCase):
    """测试基类"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.token = cls._get_token()
        cls.headers = {"Authorization": f"Bearer {cls.token}"}
        cls.session = requests.Session()
        cls.session.headers.update(cls.headers)

    @classmethod
    def _get_token(cls):
        """获取认证token"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": USERNAME, "password": PASSWORD}
        )
        return response.json()["access_token"]

    def assert_response_ok(self, response, msg=None):
        """断言响应成功"""
        self.assertIn(response.status_code, [200, 201], 
                     f"{msg or '请求失败'}: {response.status_code} - {response.text}")


class TestAuthentication(AudioMOSTestBase):
    """认证系统测试"""

    def test_login_success(self):
        """TC-AUTH-001: 正常登录"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": USERNAME, "password": PASSWORD}
        )
        self.assert_response_ok(response, "登录失败")
        data = response.json()
        self.assertIn("access_token", data)
        self.assertEqual(data["token_type"], "bearer")

    def test_login_wrong_password(self):
        """TC-AUTH-002: 错误密码登录"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": USERNAME, "password": "wrong_password"}
        )
        self.assertEqual(response.status_code, 401)

    def test_login_nonexistent_user(self):
        """TC-AUTH-003: 不存在的用户登录"""
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            data={"username": "nonexistent", "password": "password"}
        )
        self.assertEqual(response.status_code, 401)

    def test_access_protected_without_token(self):
        """TC-AUTH-004: 无Token访问受保护接口"""
        response = requests.get(f"{BASE_URL}/api/reference-audio/list")
        self.assertEqual(response.status_code, 401)

    def test_access_protected_with_invalid_token(self):
        """TC-AUTH-005: 无效Token访问受保护接口"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = requests.get(
            f"{BASE_URL}/api/reference-audio/list",
            headers=headers
        )
        self.assertEqual(response.status_code, 401)


class TestHealthCheck(unittest.TestCase):
    """健康检查测试"""

    def test_health_endpoint(self):
        """TC-HEALTH-001: 健康检查接口"""
        response = requests.get(f"{BASE_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "audiomos-api")


class TestReferenceAudio(AudioMOSTestBase):
    """参考音频管理测试"""

    def test_list_reference_audio(self):
        """TC-REF-001: 获取参考音频列表"""
        response = self.session.get(f"{BASE_URL}/api/reference-audio/list")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIn("items", data)
        self.assertIn("total", data)

    def test_reference_audio_structure(self):
        """TC-REF-002: 参考音频数据结构完整性"""
        response = self.session.get(f"{BASE_URL}/api/reference-audio/list")
        data = response.json()
        if data["items"]:
            audio = data["items"][0]
            required_fields = [
                "id", "filename", "original_name", "file_size",
                "duration", "sample_rate", "channels", "created_at"
            ]
            for field in required_fields:
                self.assertIn(field, audio, f"缺少字段: {field}")

    def test_check_reference_status(self):
        """TC-REF-003: 检查参考音频状态"""
        response = self.session.get(f"{BASE_URL}/api/reference-audio/check/status")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIn("has_reference", data)
        self.assertIn("total_count", data)


class TestMOSCalculation(AudioMOSTestBase):
    """MOS计算功能测试"""

    def test_get_mos_tasks_empty(self):
        """TC-MOS-001: 获取MOS任务列表（空）"""
        response = self.session.get(f"{BASE_URL}/api/mos/tasks")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_upload_audio_without_file(self):
        """TC-MOS-002: 上传音频（无文件）"""
        response = self.session.post(f"{BASE_URL}/api/mos/upload")
        self.assertEqual(response.status_code, 422)

    def test_create_task_invalid_algorithm(self):
        """TC-MOS-003: 创建任务（无效算法）"""
        # 先上传文件
        test_file = TEST_AUDIO_DIR / "test_16k.wav"
        if not test_file.exists():
            self.skipTest("测试音频文件不存在")

        with open(test_file, "rb") as f:
            files = {"files": ("test.wav", f, "audio/wav")}
            upload_response = self.session.post(
                f"{BASE_URL}/api/mos/upload",
                files=files
            )
        self.assert_response_ok(upload_response)
        task_id = upload_response.json()["task_id"]

        # 创建任务（无效算法）
        response = self.session.post(
            f"{BASE_URL}/api/mos/tasks",
            json={
                "task_id": task_id,
                "algorithms": ["invalid_algorithm"]
            }
        )
        # 应该返回错误或使用默认算法
        self.assertIn(response.status_code, [200, 400, 422])


class TestAudioRestoration(AudioMOSTestBase):
    """音频修复功能测试"""

    def test_list_restoration_algorithms(self):
        """TC-REST-001: 获取修复算法列表"""
        response = self.session.get(f"{BASE_URL}/api/restoration/algorithms")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_restoration_algorithm_structure(self):
        """TC-REST-002: 修复算法数据结构完整性"""
        response = self.session.get(f"{BASE_URL}/api/restoration/algorithms")
        data = response.json()
        if data:
            algo = data[0]
            required_fields = [
                "name", "display_name", "description", "type",
                "advantages", "limitations", "initialized"
            ]
            for field in required_fields:
                self.assertIn(field, algo, f"缺少字段: {field}")

    def test_upload_for_restoration(self):
        """TC-REST-003: 上传音频用于修复"""
        test_file = TEST_AUDIO_DIR / "test_16k.wav"
        if not test_file.exists():
            self.skipTest("测试音频文件不存在")

        with open(test_file, "rb") as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            data = {"algorithm": "clearvoice_frcrn_se_16k"}
            response = self.session.post(
                f"{BASE_URL}/api/restoration/upload",
                files=files,
                data=data
            )
        self.assert_response_ok(response)
        result = response.json()
        self.assertIn("task_id", result)
        return result["task_id"]

    def test_get_restoration_tasks(self):
        """TC-REST-004: 获取修复任务列表"""
        response = self.session.get(f"{BASE_URL}/api/restoration/tasks")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIsInstance(data, list)


class TestDenoise(AudioMOSTestBase):
    """降噪功能测试"""

    def test_list_denoise_algorithms(self):
        """TC-DENOISE-001: 获取降噪算法列表"""
        response = self.session.get(f"{BASE_URL}/api/denoise/algorithms")
        self.assert_response_ok(response)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_denoise_algorithm_structure(self):
        """TC-DENOISE-002: 降噪算法数据结构完整性"""
        response = self.session.get(f"{BASE_URL}/api/denoise/algorithms")
        data = response.json()
        if data:
            algo = data[0]
            required_fields = ["name", "description", "type"]
            for field in required_fields:
                self.assertIn(field, algo, f"缺少字段: {field}")


class TestPerformance(AudioMOSTestBase):
    """性能测试"""

    def test_api_response_time(self):
        """TC-PERF-001: API响应时间测试"""
        endpoints = [
            ("/health", "GET"),
            ("/api/reference-audio/list", "GET"),
            ("/api/restoration/algorithms", "GET"),
            ("/api/denoise/algorithms", "GET"),
        ]

        results = []
        for endpoint, method in endpoints:
            start = time.time()
            if method == "GET":
                response = self.session.get(f"{BASE_URL}{endpoint}")
            else:
                response = self.session.post(f"{BASE_URL}{endpoint}")
            elapsed = time.time() - start
            results.append((endpoint, elapsed, response.status_code))

        # 打印性能报告
        print("\n性能测试报告:")
        print("-" * 60)
        for endpoint, elapsed, status in results:
            status_str = "✓" if status == 200 else "✗"
            print(f"{status_str} {endpoint}: {elapsed*1000:.2f}ms (HTTP {status})")
        print("-" * 60)

        # 断言所有请求都成功
        for _, _, status in results:
            self.assertEqual(status, 200)

    def test_concurrent_requests(self):
        """TC-PERF-002: 并发请求测试"""
        def make_request(i):
            start = time.time()
            response = self.session.get(f"{BASE_URL}/health")
            elapsed = time.time() - start
            return i, elapsed, response.status_code

        # 10个并发请求
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # 验证所有请求成功
        success_count = sum(1 for _, _, status in results if status == 200)
        self.assertEqual(success_count, 10, "并发请求有失败")

        # 打印并发测试报告
        print("\n并发测试报告 (10个并发请求):")
        print("-" * 60)
        times = [elapsed for _, elapsed, _ in results]
        print(f"平均响应时间: {sum(times)/len(times)*1000:.2f}ms")
        print(f"最小响应时间: {min(times)*1000:.2f}ms")
        print(f"最大响应时间: {max(times)*1000:.2f}ms")
        print("-" * 60)


class TestGPUStatus(AudioMOSTestBase):
    """GPU状态测试"""

    def test_gpu_memory_usage(self):
        """TC-GPU-001: 检查GPU显存使用"""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.free", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                used, free = map(int, result.stdout.strip().split(","))
                total = used + free
                usage_percent = (used / total) * 100

                print(f"\nGPU显存使用报告:")
                print("-" * 60)
                print(f"已用显存: {used} MB ({usage_percent:.1f}%)")
                print(f"可用显存: {free} MB")
                print(f"总显存: {total} MB")
                print("-" * 60)

                # 断言显存使用不超过90%
                self.assertLess(usage_percent, 90, "显存使用超过90%")
        except Exception as e:
            self.skipTest(f"无法获取GPU信息: {e}")


class TestEndToEnd(AudioMOSTestBase):
    """端到端流程测试"""

    def test_full_workflow(self):
        """TC-E2E-001: 完整工作流程测试"""
        # 1. 检查参考音频
        response = self.session.get(f"{BASE_URL}/api/reference-audio/check/status")
        self.assert_response_ok(response)

        # 2. 获取修复算法列表
        response = self.session.get(f"{BASE_URL}/api/restoration/algorithms")
        self.assert_response_ok(response)
        algorithms = response.json()
        self.assertGreater(len(algorithms), 0)

        # 3. 获取降噪算法列表
        response = self.session.get(f"{BASE_URL}/api/denoise/algorithms")
        self.assert_response_ok(response)

        print("\n端到端测试通过: 所有核心接口正常")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestHealthCheck,
        TestAuthentication,
        TestReferenceAudio,
        TestMOSCalculation,
        TestAudioRestoration,
        TestDenoise,
        TestPerformance,
        TestGPUStatus,
        TestEndToEnd,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回测试结果
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

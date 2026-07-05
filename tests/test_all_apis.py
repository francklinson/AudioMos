"""
完整API测试套件
测试所有后端接口的可用性和完整性
"""
import pytest
import os
import sys
import io
import wave
import struct

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))


@pytest.fixture(scope="module")
def client():
    """创建测试客户端"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture(scope="module")
def auth_token(client):
    """获取认证token"""
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "tp123456"}
    )
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def sample_audio_bytes():
    """生成测试音频数据（1秒静音，16kHz，单声道）"""
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    samples = [0] * num_samples
    
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *samples))
    
    buffer.seek(0)
    return buffer.read()


class TestAllAPIs:
    """所有API接口综合测试"""

    # ========== 认证API测试 ==========
    def test_auth_login_success(self, client):
        """测试登录成功"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_auth_login_failure(self, client):
        """测试登录失败"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "wrong_password"}
        )
        assert response.status_code == 401

    def test_auth_get_current_user(self, client, auth_token):
        """测试获取当前用户"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        assert response.json()["username"] == "admin"

    def test_auth_logout(self, client, auth_token):
        """测试登出"""
        response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    # ========== MOS评分API测试 ==========
    def test_mos_upload_files(self, client, auth_token, sample_audio_bytes):
        """测试MOS上传文件"""
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test.wav", sample_audio_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "files" in data

    def test_mos_get_tasks(self, client, auth_token):
        """测试获取MOS任务列表"""
        response = client.get(
            "/api/mos/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_mos_get_performance(self, client, auth_token):
        """测试获取MOS性能统计"""
        response = client.get(
            "/api/mos/performance",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_mos_reset_performance(self, client, auth_token):
        """测试重置MOS性能统计"""
        response = client.post(
            "/api/mos/performance/reset",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    # ========== 音频修复API测试 ==========
    def test_restoration_get_algorithms(self, client, auth_token):
        """测试获取修复算法列表"""
        response = client.get(
            "/api/restoration/algorithms",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        algorithms = response.json()
        assert isinstance(algorithms, list)
        assert len(algorithms) > 0

    def test_restoration_upload_audio(self, client, auth_token, sample_audio_bytes):
        """测试修复上传音频"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={"algorithm": "spectral_subtraction"}
        )
        assert response.status_code == 200
        assert "task_id" in response.json()

    def test_restoration_upload_batch(self, client, auth_token, sample_audio_bytes):
        """测试修复批量上传"""
        response = client.post(
            "/api/restoration/upload-batch",
            headers={"Authorization": f"Bearer {auth_token}"},
            files=[
                ("files", ("audio1.wav", sample_audio_bytes, "audio/wav")),
                ("files", ("audio2.wav", sample_audio_bytes, "audio/wav"))
            ],
            data={"algorithm": "spectral_subtraction"}
        )
        assert response.status_code == 200

    def test_restoration_get_tasks(self, client, auth_token):
        """测试获取修复任务列表"""
        response = client.get(
            "/api/restoration/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    # ========== 参考音频API测试 ==========
    def test_reference_audio_list(self, client, auth_token):
        """测试获取参考音频列表"""
        response = client.get(
            "/api/reference-audio/list",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data

    def test_reference_audio_upload(self, client, auth_token, sample_audio_bytes):
        """测试上传参考音频"""
        response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("ref.wav", sample_audio_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        assert "id" in response.json()

    def test_reference_audio_upload_batch(self, client, auth_token, sample_audio_bytes):
        """测试批量上传参考音频"""
        response = client.post(
            "/api/reference-audio/upload-batch",
            headers={"Authorization": f"Bearer {auth_token}"},
            files=[
                ("files", ("ref1.wav", sample_audio_bytes, "audio/wav")),
                ("files", ("ref2.wav", sample_audio_bytes, "audio/wav"))
            ]
        )
        assert response.status_code == 200
        data = response.json()
        # 实际返回结构：{"message": "...", "results": {"success": [...], "failed": [...], "total": 2}}
        assert "results" in data or "total" in data
        if "results" in data:
            assert data["results"]["total"] == 2

    def test_reference_audio_check_status(self, client, auth_token):
        """测试检查参考音频状态"""
        response = client.get(
            "/api/reference-audio/check/status",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    def test_reference_audio_fingerprint_status(self, client, auth_token):
        """测试查询指纹数据库状态"""
        response = client.get(
            "/api/reference-audio/fingerprint/status",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200

    # ========== 健康检查API测试 ==========
    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200


class TestAuthenticationRequired:
    """认证必需接口测试"""

    def test_mos_upload_requires_auth(self, client):
        """测试MOS上传需要认证"""
        response = client.post("/api/mos/upload")
        assert response.status_code == 401

    def test_restoration_upload_requires_auth(self, client):
        """测试修复上传需要认证"""
        response = client.post("/api/restoration/upload")
        assert response.status_code == 401

    def test_reference_audio_list_requires_auth(self, client):
        """测试参考音频列表需要认证"""
        response = client.get("/api/reference-audio/list")
        assert response.status_code == 401

    def test_restoration_algorithms_requires_auth(self, client):
        """测试修复算法列表需要认证"""
        response = client.get("/api/restoration/algorithms")
        assert response.status_code == 401


class TestErrorHandling:
    """错误处理测试"""

    def test_mos_task_not_found(self, client, auth_token):
        """测试MOS任务不存在"""
        response = client.get(
            "/api/mos/tasks/nonexistent_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_restoration_task_not_found(self, client, auth_token):
        """测试修复任务不存在"""
        response = client.get(
            "/api/restoration/tasks/nonexistent_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_reference_audio_not_found(self, client, auth_token):
        """测试参考音频不存在"""
        response = client.get(
            "/api/reference-audio/detail/nonexistent_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_invalid_file_format(self, client, auth_token):
        """测试无效文件格式"""
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test.txt", b"invalid", "text/plain")}
        )
        # 应被过滤或返回错误
        assert response.status_code in [400, 200]  # 可能过滤掉无效文件返回空列表

    def test_invalid_restoration_algorithm(self, client, auth_token, sample_audio_bytes):
        """测试无效修复算法"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={"algorithm": "invalid_algorithm"}
        )
        assert response.status_code == 400


class TestAPIPerformance:
    """API性能测试"""

    def test_health_check_fast(self, client):
        """测试健康检查响应速度"""
        import time
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # 应在1秒内响应

    def test_login_fast(self, client):
        """测试登录响应速度"""
        import time
        start = time.time()
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0  # 应在2秒内响应


def run_all_tests():
    """运行所有测试"""
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode


if __name__ == "__main__":
    # 直接运行pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
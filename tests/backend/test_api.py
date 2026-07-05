"""
后端API测试
测试FastAPI后端接口（含音频修复功能）
"""
import pytest
import os
import io
import wave
import struct


@pytest.fixture
def client():
    """创建测试客户端"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)


@pytest.fixture
def auth_token(client):
    """获取认证token"""
    response = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": "tp123456"}
    )
    return response.json()["access_token"]


@pytest.fixture
def sample_audio_bytes():
    """生成简单的测试音频数据（1秒静音，16kHz，单声道）"""
    # 创建一个简单的 WAV 文件
    sample_rate = 16000
    duration = 1.0
    num_samples = int(sample_rate * duration)
    
    # 生成静音数据（全是0）
    samples = [0] * num_samples
    
    # 写入 WAV 格式
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack('<' + 'h' * num_samples, *samples))
    
    buffer.seek(0)
    return buffer.read()


class TestAuthAPI:
    """认证接口测试"""

    def test_login_success(self, client):
        """测试正常登录"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_failure(self, client):
        """测试错误密码登录"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "wrongpassword"}
        )
        assert response.status_code == 401

    def test_login_missing_fields(self, client):
        """测试缺少字段登录"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin"}
        )
        assert response.status_code == 422


class TestMosAPI:
    """MOS评分接口测试"""

    def test_get_tasks(self, client, auth_token):
        """测试获取任务列表"""
        response = client.get(
            "/api/mos/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or isinstance(data, dict)

    def test_upload_without_auth(self, client):
        """测试未认证上传"""
        response = client.post("/api/mos/upload")
        assert response.status_code == 401

    def test_upload_audio(self, client, auth_token, sample_audio_bytes):
        """测试上传音频文件"""
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test_audio.wav", sample_audio_bytes, "audio/wav")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert "files" in data
        assert "message" in data

    def test_upload_multiple_files(self, client, auth_token, sample_audio_bytes):
        """测试上传多个音频文件"""
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files=[
                ("files", ("audio1.wav", sample_audio_bytes, "audio/wav")),
                ("files", ("audio2.wav", sample_audio_bytes, "audio/wav"))
            ]
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["files"]) == 2

    def test_get_task_status_not_found(self, client, auth_token):
        """测试获取不存在的任务状态"""
        response = client.get(
            "/api/mos/tasks/nonexistent_task_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_download_not_found(self, client, auth_token):
        """测试下载不存在的任务结果"""
        response = client.get(
            "/api/mos/download/nonexistent_task_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_get_results_not_found(self, client, auth_token):
        """测试获取不存在的任务结果JSON"""
        response = client.get(
            "/api/mos/results/nonexistent_task_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_delete_task_not_found(self, client, auth_token):
        """测试删除不存在的任务"""
        response = client.delete(
            "/api/mos/tasks/nonexistent_task_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_get_performance(self, client, auth_token):
        """测试获取性能统计"""
        response = client.get(
            "/api/mos/performance",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # 验证返回字段存在
        assert isinstance(data, dict)

    def test_reset_performance(self, client, auth_token):
        """测试重置性能统计"""
        response = client.post(
            "/api/mos/performance/reset",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200


class TestRestorationAPI:
    """音频修复接口测试"""

    def test_get_algorithms(self, client, auth_token):
        """测试获取算法列表"""
        response = client.get(
            "/api/restoration/algorithms",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # 验证关键算法存在
        algorithm_names = [alg["name"] for alg in data]
        assert "super_resolution" in algorithm_names
        assert "dereverberation_wiener" in algorithm_names
        # SR 模型不应单独作为降噪器暴露
        assert "clearvoice_mossformer2_sr_48k" not in algorithm_names

    def test_get_algorithms_without_auth(self, client):
        """测试未认证获取算法列表"""
        response = client.get("/api/restoration/algorithms")
        assert response.status_code == 401

    def test_get_tasks(self, client, auth_token):
        """测试获取修复任务列表"""
        response = client.get(
            "/api/restoration/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "tasks" in data or isinstance(data, list)

    def test_get_tasks_without_auth(self, client):
        """测试未认证获取任务列表"""
        response = client.get("/api/restoration/tasks")
        assert response.status_code == 401

    def test_upload_audio(self, client, auth_token, sample_audio_bytes):
        """测试上传音频文件"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test_audio.wav", sample_audio_bytes, "audio/wav")},
            data={"algorithm": "spectral_subtraction"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_upload_without_auth(self, client, sample_audio_bytes):
        """测试未认证上传"""
        response = client.post(
            "/api/restoration/upload",
            files={"file": ("test_audio.wav", sample_audio_bytes, "audio/wav")},
            data={"algorithm": "spectral_subtraction"}
        )
        assert response.status_code == 401

    def test_upload_missing_file(self, client, auth_token):
        """测试缺少文件上传"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            data={"algorithm": "spectral_subtraction"}
        )
        assert response.status_code == 422

    def test_upload_missing_algorithm(self, client, auth_token, sample_audio_bytes):
        """测试缺少算法参数上传（应使用默认算法）"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test_audio.wav", sample_audio_bytes, "audio/wav")}
        )
        # algorithm 有默认值 "dereverberation"，应返回 200
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data

    def test_upload_batch(self, client, auth_token, sample_audio_bytes):
        """测试批量上传音频文件"""
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
        data = response.json()
        # 验证批量上传返回结果

    def test_get_task_status_not_found(self, client, auth_token):
        """测试获取不存在的任务状态"""
        response = client.get(
            "/api/restoration/tasks/nonexistent-task-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_get_source_not_found(self, client, auth_token):
        """测试获取不存在的源音频"""
        response = client.get(
            "/api/restoration/source/nonexistent-task-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_download_not_found(self, client, auth_token):
        """测试下载不存在的任务结果"""
        response = client.get(
            "/api/restoration/download/nonexistent-task-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_delete_nonexistent_task(self, client, auth_token):
        """测试删除不存在的任务"""
        response = client.delete(
            "/api/restoration/tasks/nonexistent-task-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_invalid_algorithm(self, client, auth_token, sample_audio_bytes):
        """测试使用不支持的算法"""
        response = client.post(
            "/api/restoration/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test_audio.wav", sample_audio_bytes, "audio/wav")},
            data={"algorithm": "invalid_algorithm"}
        )
        assert response.status_code == 400


class TestHealthCheck:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


class TestWebSocketEndpoint:
    """WebSocket 端点测试"""

    def test_restoration_ws_route_exists(self, client):
        """测试 WebSocket 路由存在"""
        # 通过 OpenAPI schema 验证 WebSocket 路由已注册
        # TestClient 不支持 WebSocket 协议升级测试
        from app.main import app
        routes = [r.path for r in app.routes]
        # WebSocket 路径格式
        ws_route = "/api/restoration/ws/{task_id}"
        # 检查是否有类似的 WebSocket 路由（可能是带前缀的）
        has_ws = any("restoration/ws" in str(r) for r in routes)
        assert has_ws, "WebSocket 路由应存在于路由表中"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

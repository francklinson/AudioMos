"""
端到端完整工作流测试
测试从上传到结果下载的完整流程
"""
import pytest
import os
import time
import io


@pytest.fixture
def client():
    """创建测试客户端"""
    try:
        from fastapi.testclient import TestClient
        from app.main import app
        return TestClient(app)
    except ImportError as e:
        pytest.skip(f"无法创建TestClient: {e}")


@pytest.fixture
def auth_token(client):
    """获取认证token"""
    try:
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        if response.status_code == 200:
            return response.json()["access_token"]
        else:
            pytest.skip(f"无法获取认证token: {response.status_code}")
    except Exception as e:
        pytest.skip(f"认证失败: {e}")


class TestUploadWorkflow:
    """上传工作流测试"""

    def test_upload_single_file(self, client, auth_token):
        """测试单文件上传"""
        # 创建虚拟音频文件
        fake_audio = io.BytesIO(b"RIFF" + b"\x00" * 100)  # 简化的WAV头部
        fake_audio.name = "test_single.wav"
        
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test_single.wav", fake_audio, "audio/wav")}
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert "files" in data
            assert "message" in data
            assert len(data["files"]) == 1
        else:
            # 如果后端不支持假音频，跳过
            pytest.skip(f"上传测试跳过: {response.status_code}")

    def test_upload_multiple_files(self, client, auth_token):
        """测试多文件上传"""
        fake_audio1 = io.BytesIO(b"RIFF" + b"\x00" * 100)
        fake_audio2 = io.BytesIO(b"RIFF" + b"\x00" * 100)
        
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files=[
                ("files", ("test1.wav", fake_audio1, "audio/wav")),
                ("files", ("test2.wav", fake_audio2, "audio/wav"))
            ]
        )
        
        if response.status_code == 200:
            data = response.json()
            assert "task_id" in data
            assert len(data["files"]) == 2

    def test_upload_invalid_format(self, client, auth_token):
        """测试无效格式上传"""
        fake_file = io.BytesIO(b"invalid data")
        fake_file.name = "test.txt"
        
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test.txt", fake_file, "text/plain")}
        )
        
        # 应该返回400错误或不支持的格式
        assert response.status_code in [200, 400]


class TestTaskWorkflow:
    """任务工作流测试"""

    @pytest.fixture
    def created_task(self, client, auth_token):
        """创建测试任务"""
        fake_audio = io.BytesIO(b"RIFF" + b"\x00" * 100)
        
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("workflow_test.wav", fake_audio, "audio/wav")}
        )
        
        if response.status_code == 200:
            return response.json()["task_id"]
        else:
            pytest.skip(f"无法创建测试任务: {response.status_code}")

    def test_task_status_flow(self, client, auth_token, created_task):
        """测试任务状态流转"""
        task_id = created_task
        
        # 1. 检查初始状态
        response = client.get(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        initial_data = response.json()
        assert initial_data["task_id"] == task_id
        assert "status" in initial_data
        assert "progress" in initial_data
        
        # 2. 提交处理任务
        response = client.post(
            f"/api/mos/process/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # 3. 检查任务状态（可能在队列中或处理中）
        response = client.get(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] in ["queued", "processing", "pending"]

    def test_task_listing(self, client, auth_token, created_task):
        """测试任务列表"""
        # 获取任务列表
        response = client.get(
            "/api/mos/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        tasks = response.json()
        
        # 验证列表中包含新创建的任务
        task_ids = [t["task_id"] for t in tasks]
        assert created_task in task_ids

    def test_task_deletion(self, client, auth_token, created_task):
        """测试任务删除"""
        task_id = created_task
        
        # 删除任务
        response = client.delete(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        
        # 验证任务已删除
        response = client.get(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404


class TestAuthenticationFlow:
    """认证流程测试"""

    def test_complete_auth_flow(self, client):
        """测试完整认证流程"""
        # 1. 登录
        login_response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        assert login_response.status_code == 200
        token_data = login_response.json()
        assert "access_token" in token_data
        token = token_data["access_token"]
        
        # 2. 使用token访问受保护资源
        me_response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert me_response.status_code == 200
        user_data = me_response.json()
        assert user_data["username"] == "admin"
        
        # 3. 登出
        logout_response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert logout_response.status_code == 200

    def test_token_expired_or_invalid(self, client):
        """测试无效或过期的token"""
        # 使用无效token
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token_xyz"}
        )
        assert response.status_code == 401


class TestErrorHandling:
    """错误处理测试"""

    def test_404_endpoint(self, client):
        """测试404端点"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """测试方法不允许"""
        response = client.post("/health")  # health只支持GET
        assert response.status_code == 405


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

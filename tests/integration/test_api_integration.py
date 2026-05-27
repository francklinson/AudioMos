"""
API集成测试
测试FastAPI后端接口的完整功能
"""
import pytest
import os

# 创建全局测试客户端实例
_test_client = None

def get_test_client():
    """获取或创建测试客户端"""
    global _test_client
    if _test_client is None:
        try:
            from fastapi.testclient import TestClient
            from app.main import app
            _test_client = TestClient(app)
        except Exception as e:
            pytest.skip(f"无法创建TestClient: {e}")
    return _test_client


@pytest.fixture
def client():
    """创建测试客户端"""
    return get_test_client()


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


class TestRootEndpoints:
    """根路径端点测试"""

    def test_root_endpoint(self, client):
        """测试根路径"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "AudioMOS API"

    def test_health_check(self, client):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestAuthIntegration:
    """认证集成测试"""

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

    def test_login_wrong_password(self, client):
        """测试错误密码登录"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "wrong_password"}
        )
        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """测试不存在的用户登录"""
        response = client.post(
            "/api/auth/login",
            data={"username": "nonexistent", "password": "password"}
        )
        assert response.status_code == 401

    def test_login_missing_username(self, client):
        """测试缺少用户名"""
        response = client.post(
            "/api/auth/login",
            data={"password": "password"}
        )
        assert response.status_code == 422

    def test_login_missing_password(self, client):
        """测试缺少密码"""
        response = client.post(
            "/api/auth/login",
            data={"username": "admin"}
        )
        assert response.status_code == 422

    def test_get_current_user(self, client, auth_token):
        """测试获取当前用户信息"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "admin"
        assert data["is_active"] is True

    def test_get_current_user_no_auth(self, client):
        """测试未认证获取用户信息"""
        response = client.get("/api/auth/me")
        assert response.status_code == 401

    def test_get_current_user_invalid_token(self, client):
        """测试无效token"""
        response = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == 401

    def test_logout(self, client, auth_token):
        """测试登出"""
        response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestMosAPIIntegration:
    """MOS评分API集成测试"""

    def test_get_tasks_unauthorized(self, client):
        """测试未授权获取任务列表"""
        response = client.get("/api/mos/tasks")
        assert response.status_code == 401

    def test_get_tasks_authorized(self, client, auth_token):
        """测试已授权获取任务列表"""
        response = client.get(
            "/api/mos/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_upload_no_auth(self, client):
        """测试未认证上传"""
        response = client.post("/api/mos/upload")
        assert response.status_code == 401

    def test_upload_no_files(self, client, auth_token):
        """测试上传无文件"""
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 422  # 缺少文件参数

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

    def test_delete_task_not_found(self, client, auth_token):
        """测试删除不存在的任务"""
        response = client.delete(
            "/api/mos/tasks/nonexistent_task_id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404


class TestMosAPIWorkflow:
    """MOS API工作流测试"""

    @pytest.fixture
    def test_task(self, client, auth_token):
        """创建测试任务"""
        import io
        
        # 创建虚拟音频文件
        fake_audio = io.BytesIO(b"fake audio data")
        fake_audio.name = "test.wav"
        
        response = client.post(
            "/api/mos/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"files": ("test.wav", fake_audio, "audio/wav")}
        )
        
        if response.status_code == 200:
            return response.json()["task_id"]
        else:
            pytest.skip(f"无法创建测试任务: {response.status_code}")

    def test_task_lifecycle(self, client, auth_token, test_task):
        """测试任务生命周期"""
        task_id = test_task
        
        # 1. 获取任务状态
        response = client.get(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert "status" in data
        assert "progress" in data
        
        # 2. 获取任务列表（应包含新任务）
        response = client.get(
            "/api/mos/tasks",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        tasks = response.json()
        task_ids = [t["task_id"] for t in tasks]
        assert task_id in task_ids
        
        # 3. 删除任务
        response = client.delete(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        
        # 4. 验证任务已删除
        response = client.get(
            f"/api/mos/tasks/{task_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404


class TestAPIPerformance:
    """API性能测试"""

    def test_health_check_performance(self, client):
        """测试健康检查性能"""
        import time
        
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # 应在1秒内响应

    def test_login_performance(self, client):
        """测试登录性能"""
        import time
        
        start = time.time()
        response = client.post(
            "/api/auth/login",
            data={"username": "admin", "password": "tp123456"}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0  # 应在2秒内响应


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
后端API测试
测试FastAPI后端接口
"""
import pytest


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
        assert isinstance(data, list)

    def test_upload_without_auth(self, client):
        """测试未认证上传"""
        response = client.post("/api/mos/upload")
        assert response.status_code == 401


class TestHealthCheck:
    """健康检查接口测试"""

    def test_health_check(self, client):
        """测试健康检查接口"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

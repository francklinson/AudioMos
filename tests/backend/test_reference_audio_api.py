"""
参考音频管理API测试
测试参考音频上传、查询、删除、指纹管理等接口
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
    assert response.status_code == 200
    return response.json()["access_token"]


@pytest.fixture
def sample_audio_bytes():
    """生成简单的测试音频数据（1秒静音，16kHz，单声道）"""
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


class TestReferenceAudioAPI:
    """参考音频管理接口测试"""

    def test_list_reference_audios(self, client, auth_token):
        """测试获取参考音频列表"""
        response = client.get(
            "/api/reference-audio/list",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "items" in data
        assert isinstance(data["items"], list)

    def test_list_without_auth(self, client):
        """测试未认证获取列表"""
        response = client.get("/api/reference-audio/list")
        assert response.status_code == 401

    def test_upload_reference_audio(self, client, auth_token, sample_audio_bytes):
        """测试上传参考音频"""
        response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test_ref.wav", sample_audio_bytes, "audio/wav")},
            data={"description": "测试参考音频"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "filename" in data
        assert "original_name" in data
        assert data["original_name"] == "test_ref.wav"
        assert data["description"] == "测试参考音频"

    def test_upload_without_auth(self, client, sample_audio_bytes):
        """测试未认证上传"""
        response = client.post(
            "/api/reference-audio/upload",
            files={"file": ("test_ref.wav", sample_audio_bytes, "audio/wav")}
        )
        assert response.status_code == 401

    def test_upload_invalid_format(self, client, auth_token):
        """测试上传不支持的文件格式"""
        response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("test.txt", b"invalid content", "text/plain")}
        )
        assert response.status_code == 400

    def test_upload_batch(self, client, auth_token, sample_audio_bytes):
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
        assert "results" in data or "success" in data
        if "results" in data:
            assert "success" in data["results"]
            assert "failed" in data["results"]
            assert data["results"]["total"] == 2
        else:
            assert "failed" in data
            assert data["total"] == 2

    def test_get_detail_not_found(self, client, auth_token):
        """测试获取不存在的参考音频详情"""
        response = client.get(
            "/api/reference-audio/detail/nonexistent-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_update_reference_audio(self, client, auth_token, sample_audio_bytes):
        """测试更新参考音频描述"""
        # 先上传
        upload_response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("update_test.wav", sample_audio_bytes, "audio/wav")}
        )
        assert upload_response.status_code == 200
        audio_id = upload_response.json()["id"]
        
        # 更新描述
        update_response = client.put(
            f"/api/reference-audio/update/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"description": "更新后的描述", "ground_truth_text": "测试文本"}
        )
        assert update_response.status_code == 200
        data = update_response.json()
        assert data["description"] == "更新后的描述"
        assert data["ground_truth_text"] == "测试文本"

    def test_delete_reference_audio(self, client, auth_token, sample_audio_bytes):
        """测试删除参考音频"""
        # 先上传
        upload_response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("delete_test.wav", sample_audio_bytes, "audio/wav")}
        )
        assert upload_response.status_code == 200
        audio_id = upload_response.json()["id"]
        
        # 删除
        delete_response = client.delete(
            f"/api/reference-audio/delete/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert delete_response.status_code == 200
        
        # 验证已删除
        detail_response = client.get(
            f"/api/reference-audio/detail/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert detail_response.status_code == 404

    def test_delete_nonexistent_audio(self, client, auth_token):
        """测试删除不存在的参考音频"""
        response = client.delete(
            "/api/reference-audio/delete/nonexistent-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_download_nonexistent_audio(self, client, auth_token):
        """测试下载不存在的参考音频"""
        response = client.get(
            "/api/reference-audio/download/nonexistent-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 404

    def test_check_status(self, client, auth_token):
        """测试检查参考音频状态"""
        response = client.get(
            "/api/reference-audio/check/status",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_count" in data
        assert "total_size" in data

    def test_fingerprint_status(self, client, auth_token):
        """测试查询指纹数据库状态"""
        response = client.get(
            "/api/reference-audio/fingerprint/status",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # 根据实际实现验证返回字段

    def test_fingerprint_build(self, client, auth_token):
        """测试建立/重建指纹数据库"""
        response = client.post(
            "/api/reference-audio/fingerprint/build",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # 根据实际实现验证返回字段


class TestReferenceAudioWorkflow:
    """参考音频完整工作流测试"""

    def test_complete_workflow(self, client, auth_token, sample_audio_bytes):
        """测试完整的参考音频管理流程"""
        # 1. 上传参考音频
        upload_response = client.post(
            "/api/reference-audio/upload",
            headers={"Authorization": f"Bearer {auth_token}"},
            files={"file": ("workflow_test.wav", sample_audio_bytes, "audio/wav")},
            data={"description": "工作流测试音频"}
        )
        assert upload_response.status_code == 200
        audio_id = upload_response.json()["id"]
        
        # 2. 获取列表，确认已添加
        list_response = client.get(
            "/api/reference-audio/list",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert list_response.status_code == 200
        items = list_response.json()["items"]
        assert any(item["id"] == audio_id for item in items)
        
        # 3. 获取详情
        detail_response = client.get(
            f"/api/reference-audio/detail/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert detail_response.status_code == 200
        assert detail_response.json()["id"] == audio_id
        
        # 4. 更新描述
        update_response = client.put(
            f"/api/reference-audio/update/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"description": "更新后的描述"}
        )
        assert update_response.status_code == 200
        assert update_response.json()["description"] == "更新后的描述"
        
        # 5. 删除
        delete_response = client.delete(
            f"/api/reference-audio/delete/{audio_id}",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        assert delete_response.status_code == 200
        
        # 6. 验证已删除
        list_response = client.get(
            "/api/reference-audio/list",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        items = list_response.json()["items"]
        assert not any(item["id"] == audio_id for item in items)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
安全模块单元测试
测试认证和Token管理功能
"""
import pytest
import os
from datetime import timedelta


class TestPasswordHash:
    """密码哈希测试类"""

    def test_password_hash_creation(self):
        """测试密码哈希创建"""
        try:
            from app.core.security import get_password_hash, verify_password
            
            password = "test_password_123"
            hashed = get_password_hash(password)
            
            # 验证哈希值
            assert hashed is not None
            assert isinstance(hashed, str)
            assert hashed.startswith("plain:")
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_password_verification(self):
        """测试密码验证"""
        try:
            from app.core.security import get_password_hash, verify_password
            
            password = "test_password_123"
            hashed = get_password_hash(password)
            
            # 验证正确密码
            assert verify_password(password, hashed) is True
            
            # 验证错误密码
            assert verify_password("wrong_password", hashed) is False
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_plain_prefix_password(self):
        """测试带plain前缀的密码验证"""
        try:
            from app.core.security import verify_password
            
            password = "test_password"
            hashed = "plain:test_password"
            
            assert verify_password(password, hashed) is True
            assert verify_password("wrong", hashed) is False
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")


class TestToken:
    """Token测试类"""

    def test_token_creation(self):
        """测试Token创建"""
        try:
            from app.core.security import create_access_token
            
            data = {"sub": "test_user"}
            token = create_access_token(data)
            
            # 验证Token创建
            assert token is not None
            assert isinstance(token, str)
            assert len(token) > 0
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_token_decode(self):
        """测试Token解码"""
        try:
            from app.core.security import create_access_token, decode_token
            
            data = {"sub": "test_user"}
            token = create_access_token(data)
            decoded = decode_token(token)
            
            # 验证解码结果
            assert decoded is not None
            assert decoded.username == "test_user"
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_token_decode_invalid(self):
        """测试无效Token解码"""
        try:
            from app.core.security import decode_token
            
            # 解码无效Token
            decoded = decode_token("invalid_token")
            assert decoded is None
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_token_expiration(self):
        """测试Token过期"""
        try:
            from app.core.security import create_access_token, decode_token
            from datetime import timedelta
            import time
            
            data = {"sub": "test_user"}
            # 创建过期时间为负数的Token（立即过期）
            token = create_access_token(data, expires_delta=timedelta(seconds=-1))
            
            # 尝试解码过期Token
            decoded = decode_token(token)
            # 过期Token应该返回None
            assert decoded is None
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")


class TestUserModel:
    """用户模型测试类"""

    def test_user_creation(self):
        """测试用户创建"""
        try:
            from app.core.security import User
            
            user = User(username="test_user", is_active=True)
            
            assert user.username == "test_user"
            assert user.is_active is True
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_user_in_db(self):
        """测试数据库用户模型"""
        try:
            from app.core.security import UserInDB
            
            user = UserInDB(
                username="test_user",
                hashed_password="plain:test_password",
                is_active=True
            )
            
            assert user.username == "test_user"
            assert user.hashed_password == "plain:test_password"
            assert user.is_active is True
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")


class TestUserAuthentication:
    """用户认证测试类"""

    def test_get_user(self):
        """测试获取用户"""
        try:
            from app.core.security import get_user
            
            # 测试获取存在的用户
            user = get_user("admin")
            assert user is not None
            assert user.username == "admin"
            
            # 测试获取不存在的用户
            user = get_user("nonexistent")
            assert user is None
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_authenticate_user_success(self):
        """测试用户认证成功"""
        try:
            from app.core.security import authenticate_user
            
            # 从配置获取默认密码
            user = authenticate_user("admin", "tp123456")
            assert user is not None
            assert user.username == "admin"
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")

    def test_authenticate_user_failure(self):
        """测试用户认证失败"""
        try:
            from app.core.security import authenticate_user
            
            # 错误密码
            user = authenticate_user("admin", "wrong_password")
            assert user is None
            
            # 不存在的用户
            user = authenticate_user("nonexistent", "password")
            assert user is None
        except ImportError as e:
            pytest.skip(f"安全模块导入失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

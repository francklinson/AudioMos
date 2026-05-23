"""
安全认证模块
提供JWT Token生成和验证功能
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from pydantic import BaseModel

from .config import settings


class Token(BaseModel):
    """Token模型"""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token数据模型"""
    username: Optional[str] = None


class User(BaseModel):
    """用户模型"""
    username: str
    is_active: bool = True


class UserInDB(User):
    """数据库用户模型"""
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    使用简单的字符串比较(生产环境应使用bcrypt)
    
    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码
        
    Returns:
        是否验证通过
    """
    # 简化版本：直接比较或前缀匹配
    if hashed_password.startswith("plain:"):
        return hashed_password[6:] == plain_password
    return hashed_password == plain_password


def get_password_hash(password: str) -> str:
    """
    获取密码哈希
    简化版本：添加前缀标识
    
    Args:
        password: 明文密码
        
    Returns:
        带前缀的密码
    """
    return f"plain:{password}"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    创建JWT访问令牌
    
    Args:
        data: 要编码的数据
        expires_delta: 过期时间增量
        
    Returns:
        JWT令牌字符串
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.auth.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, 
        settings.auth.secret_key, 
        algorithm="HS256"
    )
    return encoded_jwt


def decode_token(token: str) -> Optional[TokenData]:
    """
    解码JWT令牌
    
    Args:
        token: JWT令牌字符串
        
    Returns:
        TokenData对象,如果验证失败返回None
    """
    try:
        payload = jwt.decode(
            token, 
            settings.auth.secret_key, 
            algorithms=["HS256"]
        )
        username: str = payload.get("sub")
        if username is None:
            return None
        return TokenData(username=username)
    except JWTError:
        return None


# 简单的内存用户存储(生产环境应使用数据库)
# 使用延迟加载，避免在导入时settings未加载完成
_users_db = None

def get_users_db():
    global _users_db
    if _users_db is None:
        _users_db = {
            settings.auth.admin_username: {
                "username": settings.auth.admin_username,
                "hashed_password": get_password_hash(settings.auth.admin_password),
                "is_active": True
            }
        }
    return _users_db


def get_user(username: str) -> Optional[UserInDB]:
    """
    获取用户信息
    
    Args:
        username: 用户名
        
    Returns:
        UserInDB对象,如果不存在返回None
    """
    db = get_users_db()
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    验证用户
    
    Args:
        username: 用户名
        password: 密码
        
    Returns:
        UserInDB对象,如果验证失败返回None
    """
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

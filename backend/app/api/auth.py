"""
认证API路由
提供登录和Token管理接口
"""
from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from app.core.security import (
    Token, authenticate_user, create_access_token, 
    decode_token, get_users_db, User
)
from app.core.config import settings
from app.core.logging_config import logger


router = APIRouter(prefix="/auth", tags=["认证"])

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """
    获取当前用户
    
    Args:
        token: JWT令牌
        
    Returns:
        User对象
        
    Raises:
        HTTPException: 如果验证失败
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = decode_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception
    
    users_db = get_users_db()
    user = users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(**user)


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    获取当前活跃用户
    
    Args:
        current_user: 当前用户
        
    Returns:
        User对象
        
    Raises:
        HTTPException: 如果用户未激活
    """
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="用户未激活")
    return current_user


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    用户登录
    
    Args:
        form_data: 登录表单数据(username, password)
        
    Returns:
        Token对象
        
    Raises:
        HTTPException: 如果认证失败
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.warning(f"登录失败: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.auth.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"用户登录成功: {user.username}")
    return Token(access_token=access_token)


@router.get("/me", response_model=User)
async def get_user_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """
    获取当前用户信息
    
    Args:
        current_user: 当前用户
        
    Returns:
        User对象
    """
    return current_user


@router.post("/logout")
async def logout(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """
    用户登出
    
    Args:
        current_user: 当前用户
        
    Returns:
        登出消息
    """
    logger.info(f"用户登出: {current_user.username}")
    return {"message": "登出成功"}

#!/usr/bin/env python3
"""
AudioMOS 后端服务启动脚本
"""
import sys
import os
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.core.config import settings
from app.core.logging_config import logger
from app.core.network_utils import validate_and_fix_host, print_network_info


def _generate_selfsigned_cert():
    """生成开发用自签名 SSL 证书（首次运行时自动创建）"""
    cert_dir = os.path.join(os.path.dirname(__file__), ".ssl")
    cert_file = os.path.join(cert_dir, "dev.crt")
    key_file = os.path.join(cert_dir, "dev.key")

    if os.path.isfile(cert_file) and os.path.isfile(key_file):
        return cert_file, key_file

    os.makedirs(cert_dir, exist_ok=True)
    logger.info("生成开发用自签名 SSL 证书...")
    import subprocess
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", key_file, "-out", cert_file,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost/O=AudioMos-Dev",
    ], check=True, capture_output=True)
    logger.info(f"SSL 证书已生成: {cert_dir}")
    return cert_file, key_file


def main():
    """
    主函数 - 启动服务
    """
    parser = argparse.ArgumentParser(description="AudioMOS 后端服务")
    parser.add_argument("--https", action="store_true", help="启用 HTTPS（开发用自签名证书）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("启动 AudioMOS 后端服务...")
    logger.info("=" * 60)

    # 获取配置的 host
    configured_host = settings.server.host
    configured_port = settings.server.port

    # 如果配置的是 auto，进行自动检测
    if configured_host.lower() == "auto":
        actual_host, warning = validate_and_fix_host(configured_host, "后端服务")
        if warning:
            logger.warning(warning)
        logger.info(f"监听地址: {actual_host}:{configured_port}")
    else:
        # 直接使用配置的值 (像 VersTTS 一样)
        actual_host = configured_host
        logger.info(f"监听地址: {actual_host}:{configured_port}")

    # SSL 配置
    ssl_kwargs = {}
    if args.https:
        cert_file, key_file = _generate_selfsigned_cert()
        ssl_kwargs["ssl_certfile"] = cert_file
        ssl_kwargs["ssl_keyfile"] = key_file
        logger.info("HTTPS 模式已启用（自签名证书）")

    # 启动服务
    uvicorn.run(
        "app.main:app",
        host=actual_host,
        port=configured_port,
        reload=settings.server.debug,
        log_level="info",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()

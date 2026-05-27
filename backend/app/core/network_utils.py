"""
网络配置工具模块
提供自动检测最佳网络配置的功能，确保最大兼容性
"""
import socket
import sys
from typing import Optional, Tuple


def check_host_support(host: str, port: int = 8000) -> bool:
    """
    检查系统是否支持指定的 host 地址
    
    Args:
        host: 主机地址 (如 "0.0.0.0", "127.0.0.1")
        port: 测试端口
    
    Returns:
        bool: 是否支持
    """
    try:
        # 测试 1: DNS 解析
        socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        
        # 测试 2: 尝试绑定 (如果端口未被占用)
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        test_socket.bind((host, 0))  # 使用端口 0 让系统自动分配
        test_socket.close()
        
        return True
    except (socket.gaierror, OSError, socket.error):
        return False


def get_best_host(prefer_public: bool = False) -> str:
    """
    获取最佳的 host 配置
    
    优先级:
    1. 如果 prefer_public=True 且支持 0.0.0.0，使用 0.0.0.0
    2. 如果支持 127.0.0.1，使用 127.0.0.1 (最可靠)
    3. 如果支持 0.0.0.0，使用 0.0.0.0
    4. 最后尝试 localhost
    
    Args:
        prefer_public: 是否优先使用公网可访问的地址
    
    Returns:
        str: 最佳的 host 地址
    """
    test_port = 8000
    
    # 测试顺序
    if prefer_public:
        candidates = ["0.0.0.0", "127.0.0.1", "localhost"]
    else:
        candidates = ["127.0.0.1", "0.0.0.0", "localhost"]
    
    for host in candidates:
        if check_host_support(host, test_port):
            return host
    
    # 如果都不支持，返回最安全的 127.0.0.1
    return "127.0.0.1"


def get_server_ip() -> Optional[str]:
    """
    获取服务器的实际 IP 地址
    
    Returns:
        str: 服务器 IP 地址，或 None
    """
    try:
        # 方法 1: 通过 UDP 连接外部地址获取本地 IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        try:
            # 连接到一个公共 DNS 服务器
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        finally:
            s.close()
        return ip
    except Exception:
        pass
    
    try:
        # 方法 2: 通过 gethostname 获取
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip != "127.0.0.1":
            return ip
    except Exception:
        pass
    
    return None


def get_best_host_for_server() -> str:
    """
    获取适合服务器部署的最佳 host
    
    优先级:
    1. 如果支持 0.0.0.0，使用 0.0.0.0 (允许外部访问)
    2. 获取服务器实际 IP
    3. 使用 127.0.0.1 (仅本地)
    
    Returns:
        str: 最佳 host 地址
    """
    # 优先尝试 0.0.0.0 (最灵活，支持外部访问)
    if check_host_support("0.0.0.0"):
        return "0.0.0.0"
    
    # 尝试获取服务器实际 IP
    server_ip = get_server_ip()
    if server_ip and check_host_support(server_ip):
        return server_ip
    
    # 最后使用 127.0.0.1
    return "127.0.0.1"


def validate_and_fix_host(host: str, service_name: str = "service") -> Tuple[str, str]:
    """
    验证 host 配置，如果不支持则自动修复
    
    Args:
        host: 配置的 host 地址，支持 "auto" 自动选择
        service_name: 服务名称 (用于日志)
    
    Returns:
        Tuple[str, str]: (实际使用的 host, 警告信息)
    """
    warning = ""
    
    # 如果配置的是 "auto"，自动选择最佳 host
    if host.lower() == "auto":
        new_host = get_best_host_for_server()
        if new_host == "0.0.0.0":
            warning = f"ℹ️  {service_name} 使用 auto 模式，自动选择 host: {new_host} (支持外部访问)"
        elif new_host not in ["127.0.0.1", "localhost"]:
            warning = f"ℹ️  {service_name} 使用 auto 模式，自动选择 host: {new_host} (服务器实际IP)"
        else:
            warning = f"⚠️  {service_name} 使用 auto 模式，但无法获取公网IP，使用: {new_host} (仅本地访问)"
        return new_host, warning
    
    # 如果配置的是 0.0.0.0，检查是否支持
    if host == "0.0.0.0":
        if not check_host_support("0.0.0.0"):
            # 尝试获取服务器实际 IP
            server_ip = get_server_ip()
            if server_ip and check_host_support(server_ip):
                warning = (
                    f"⚠️  当前系统不支持 0.0.0.0，"
                    f"已将 {service_name} 的 host 自动调整为服务器实际IP: {server_ip}\n"
                    f"   如需使用 0.0.0.0，请检查 /etc/hosts 或系统网络配置"
                )
                return server_ip, warning
            else:
                new_host = "127.0.0.1"
                warning = (
                    f"⚠️  当前系统不支持 0.0.0.0，且无法获取服务器IP，"
                    f"已将 {service_name} 的 host 自动调整为 {new_host}\n"
                    f"   注意: 此配置仅允许本机访问！"
                )
                return new_host, warning
    
    # 检查配置的 host 是否支持
    if not check_host_support(host):
        new_host = get_best_host_for_server()
        warning = (
            f"⚠️  配置的 host '{host}' 在当前系统不可用，"
            f"已将 {service_name} 的 host 自动调整为 {new_host}"
        )
        return new_host, warning
    
    return host, warning


def print_network_info():
    """打印网络配置信息"""
    print("=" * 60)
    print("网络配置检测")
    print("=" * 60)
    
    # 检测各种地址支持情况
    hosts_to_test = ["127.0.0.1", "0.0.0.0", "localhost"]
    print("\n地址支持检测:")
    for host in hosts_to_test:
        supported = check_host_support(host)
        status = "✅ 支持" if supported else "❌ 不支持"
        print(f"  {host:15} {status}")
    
    # 获取推荐配置
    best_host = get_best_host(prefer_public=False)
    public_host = get_best_host(prefer_public=True)
    
    print(f"\n推荐配置:")
    print(f"  安全模式 (仅本地): {best_host}")
    print(f"  公网模式 (如支持): {public_host}")
    
    # 获取服务器 IP
    server_ip = get_server_ip()
    if server_ip:
        print(f"\n服务器 IP: {server_ip}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 直接运行此文件进行网络检测
    print_network_info()

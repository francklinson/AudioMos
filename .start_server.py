import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_script_dir, 'backend'))
sys.path.insert(0, '.')

import uvicorn
from app.core.logging_config import logger

host = os.environ.get('AUDIOMOS_HOST', '0.0.0.0')
port = int(os.environ.get('AUDIOMOS_PORT', '8002'))
enable_https = os.environ.get('AUDIOMOS_HTTPS', '1').strip() in ('1', 'true', 'yes')


def _generate_selfsigned_cert():
    """生成开发用自签名 SSL 证书"""
    cert_dir = os.path.join(_script_dir, 'backend', '.ssl')
    cert_file = os.path.join(cert_dir, 'dev.crt')
    key_file = os.path.join(cert_dir, 'dev.key')

    if os.path.isfile(cert_file) and os.path.isfile(key_file):
        return cert_file, key_file

    os.makedirs(cert_dir, exist_ok=True)
    logger.info('生成开发用自签名 SSL 证书...')
    import subprocess
    subprocess.run([
        'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
        '-keyout', key_file, '-out', cert_file,
        '-days', '365', '-nodes',
        '-subj', '/CN=localhost/O=AudioMos-Dev',
    ], check=True, capture_output=True)
    logger.info(f'SSL 证书已生成: {cert_dir}')
    return cert_file, key_file


logger.info('=' * 60)
logger.info('AudioMOS 服务启动')
logger.info('=' * 60)

ssl_kwargs = {}
if enable_https:
    cert_file, key_file = _generate_selfsigned_cert()
    ssl_kwargs['ssl_certfile'] = cert_file
    ssl_kwargs['ssl_keyfile'] = key_file
    logger.info(f'监听地址: https://{host}:{port}')
else:
    logger.info(f'监听地址: http://{host}:{port}')

uvicorn.run(
    'app.main:app',
    host=host,
    port=port,
    reload=False,
    access_log=True,
    **ssl_kwargs,
)

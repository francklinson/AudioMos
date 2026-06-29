import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(_script_dir, 'backend'))
sys.path.insert(0, '.')

import uvicorn
from app.core.logging_config import logger

host = os.environ.get('AUDIOMOS_HOST', '0.0.0.0')
port = int(os.environ.get('AUDIOMOS_PORT', '8002'))

logger.info('=' * 60)
logger.info('AudioMOS 服务启动')
logger.info('=' * 60)
logger.info(f'监听地址: {host}:{port}')

uvicorn.run(
    'app.main:app',
    host=host,
    port=port,
    reload=False,
    access_log=True
)

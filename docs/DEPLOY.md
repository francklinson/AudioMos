# AudioMOS 服务器部署说明

## 常见问题及解决方案

### 1. 报错: `getaddrinfo ENOTFOUND 0.0.0.0` 或 `Name or service not known`

**问题原因**: 某些服务器无法解析 `0.0.0.0` 这个特殊 IP 地址

**解决方案**:

#### 方案 A: 修改配置文件 (推荐)
编辑 `config/config.yaml`,将 `host` 从 `0.0.0.0` 改为 `127.0.0.1` 或服务器实际 IP:

```yaml
server:
  backend:
    host: "127.0.0.1"  # 或服务器实际IP,如 "192.168.1.100"
    port: 8077
  frontend:
    host: "127.0.0.1"  # 或服务器实际IP
    port: 8006
```

#### 方案 B: 使用环境变量
启动前设置环境变量覆盖配置:

```bash
export AUDIOMOS_BACKEND_HOST="127.0.0.1"
export AUDIOMOS_FRONTEND_HOST="127.0.0.1"
./start.sh start
```

#### 方案 C: 检查服务器 hosts 文件
确保 `/etc/hosts` 文件包含以下内容:

```
127.0.0.1   localhost
127.0.0.1   127.0.0.1
```

### 2. 模型文件缺失

**检查模型文件**:
```bash
./start.sh models
```

**如果模型缺失,需要复制以下目录到服务器**:
```
models/
├── timm/                          # 83MB - UTMOS依赖
├── utmos/                         # 781MB - UTMOS权重
├── wav2vec2/                      # 361MB - UTMOS依赖
├── scoreq/                        # 1.4GB - ScoreQ模型
├── tcf/                           # 212MB - 音色还原度模型
├── wenet/                         # WeNet语音识别模型
├── dnsmos/                        # DNSMOS模型(可选)
└── nisqa/                         # NISQA模型(可选)
```

### 3. 端口冲突

**修改端口**:
编辑 `config/config.yaml` 修改 `port` 值,或使用环境变量:

```bash
export AUDIOMOS_BACKEND_PORT="8080"
export AUDIOMOS_FRONTEND_PORT="3000"
./start.sh start
```

### 4. 防火墙设置

确保服务器防火墙开放相应端口:

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 8077/tcp
sudo ufw allow 8006/tcp

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8077/tcp
sudo firewall-cmd --permanent --add-port=8006/tcp
sudo firewall-cmd --reload
```

## 部署步骤

### 1. 准备环境

```bash
# 安装 Python 3.10+
python3 --version

# 安装 Node.js 16+
node --version
npm --version
```

### 2. 复制项目到服务器

```bash
# 使用 rsync 或 scp 复制项目
rsync -avz --exclude='.venv' --exclude='node_modules' \
  /local/path/to/AudioMOS/ server:/remote/path/to/AudioMOS/

# 复制模型文件(重要!)
rsync -avz /local/path/to/AudioMOS/models/ server:/remote/path/to/AudioMOS/models/
```

### 3. 修改配置

```bash
cd /remote/path/to/AudioMOS

# 编辑配置文件
vim config/config.yaml

# 修改以下内容:
# - server.backend.host: 改为服务器IP或127.0.0.1
# - server.frontend.host: 改为服务器IP或127.0.0.1
# - auth.secret_key: 修改为随机密钥
# - auth.admin_password: 修改默认密码
```

### 4. 启动服务

```bash
# 检查模型文件
./start.sh models

# 启动服务
./start.sh start

# 查看状态
./start.sh status
```

### 5. 使用 Nginx 反向代理 (推荐)

如果服务绑定到 `127.0.0.1`,可以使用 Nginx 反向代理:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8006;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8077/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 故障排查

### 查看日志

```bash
# 后端日志
tail -f logs/backend.log

# 前端日志
tail -f logs/frontend.log
```

### 检查服务状态

```bash
./start.sh status
```

### 停止服务

```bash
./start.sh stop
```

### 重启服务

```bash
./start.sh restart
```

## 注意事项

1. **不要直接使用 `0.0.0.0`**: 某些云服务器或容器环境无法解析此地址
2. **模型文件必须复制**: 服务器上需要有完整的 `models/` 目录
3. **防火墙配置**: 确保服务器防火墙允许访问配置的端口
4. **生产环境**: 建议使用 Nginx 反向代理,并配置 HTTPS

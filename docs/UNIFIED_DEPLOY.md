# AudioMOS 前后端一体部署方案

## 概述

本方案采用前后端一体架构，类似 VersTTS 的部署方式：
- **单服务部署**：只需启动一个服务
- **单端口访问**：避免跨域问题
- **静态文件托管**：前端构建后由后端托管
- **0.0.0.0 支持**：部署更简单，无需处理 Vite 代理问题

## 架构对比

### 原方案（前后端分离）
```
浏览器 → 前端服务(8006) → Vite代理 → 后端服务(8077)
         (React SPA)       (/api/*)    (FastAPI)
```
- 需要两个端口
- Vite 代理有 0.0.0.0 连接问题
- 需要处理跨域

### 新方案（前后端一体）
```
浏览器 → 后端服务(8077) → 静态文件服务
         (FastAPI)         (frontend/*)
```
- 只需一个端口
- 无跨域问题
- 无 Vite 代理问题

## 部署步骤

### 1. 安装依赖

```bash
# 安装后端依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..
```

### 2. 构建前端

```bash
# 使用生产配置构建前端
cd frontend
npm run build:prod
cd ..

# 或者使用启动脚本自动构建
python start_unified.py --build
```

构建输出目录：`frontend/static/`

### 3. 启动服务

```bash
# 基本启动（默认 0.0.0.0:8077）
python start_unified.py

# 指定端口
python start_unified.py --port 8080

# 指定地址
python start_unified.py --host 0.0.0.0 --port 8077

# 强制重新构建前端
python start_unified.py --build
```

### 4. 访问服务

- **前端页面**：http://localhost:8077
- **API 文档**：http://localhost:8077/docs
- **健康检查**：http://localhost:8077/health

## 配置文件

### 后端配置（config/config.yaml）

```yaml
server:
  backend:
    host: "0.0.0.0"  # 支持外部访问
    port: 8077
```

### 前端生产配置（frontend/vite.config.prod.ts）

- 使用相对路径 `./`
- 构建输出到 `frontend/static`
- API 请求使用相对路径 `/api/*`

## 目录结构

```
AudioMos/
├── backend/
│   ├── app/
│   │   └── main.py          # 托管静态文件
│   └── static/              # 前端构建输出
│       ├── index.html
│       └── assets/
├── frontend/
│   ├── src/
│   ├── vite.config.ts       # 开发配置
│   └── vite.config.prod.ts  # 生产配置
├── start_unified.py         # 一体启动脚本
└── config/
    └── config.yaml
```

## 开发模式 vs 生产模式

### 开发模式（保留）

使用原有的前后端分离方式，支持热更新：

```bash
# 终端1：启动后端
cd backend
python run.py

# 终端2：启动前端
cd frontend
npm run dev
```

### 生产模式（推荐）

使用前后端一体方式：

```bash
# 构建并启动
python start_unified.py --build

# 或只启动（如果已构建）
python start_unified.py
```

## 常见问题

### 1. 前端构建失败

```bash
# 确保 node_modules 已安装
cd frontend
npm install
npm run build:prod
```

### 2. 静态文件未找到

检查 `frontend/static/index.html` 是否存在：
```bash
ls -la frontend/static/
```

### 3. API 请求 404

确保 API 路径以 `/api` 开头，例如：
- ✅ `/api/auth/login`
- ❌ `/auth/login`

### 4. 端口被占用

```bash
# 查找占用端口的进程
lsof -i :8077

# 杀死进程
kill -9 <PID>

# 或使用其他端口
python start_unified.py --port 8080
```

## 优势总结

| 特性 | 原方案 | 新方案 |
|------|--------|--------|
| 服务数量 | 2个 | 1个 |
| 端口数量 | 2个 | 1个 |
| 跨域配置 | 需要 | 不需要 |
| Vite 代理 | 需要 | 不需要 |
| 0.0.0.0 支持 | 有问题 | 完美支持 |
| 部署复杂度 | 较高 | 较低 |
| 开发体验 | 好（热更新）| 一般 |
| 生产部署 | 复杂 | 简单 |

## 建议

- **开发阶段**：使用原方案（前后端分离），热更新体验好
- **生产部署**：使用新方案（前后端一体），部署更简单

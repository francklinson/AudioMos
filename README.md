# AudioMOS 音频质量评分系统

AudioMOS 是一个基于深度学习的音频质量评估系统，支持多种 MOS (Mean Opinion Score) 评分指标，包括有参考和无参考的音频质量评估算法。

## 功能特性

### 支持的评分指标

#### 有参考指标（需要参考音频）
- **PESQ** - 语音质量感知评估
- **STOI** - 短时客观可懂度
- **SISDR** - 尺度不变信噪比
- **WER** - 词错误率（基于语音识别）
- **TCF** - 音色还原度（基于说话人验证模型）

#### 无参考指标
- **DNSMOS** - 微软深度噪声抑制 MOS 评分
- **NISQA** - 神经网络语音质量评估
- **Scoreq** - 基于深度学习的语音质量评估
- **UTMOS** - 东京大学 SaruLab MOS 预测系统

### 系统特性
- 任务队列系统 - 支持多用户并发提交，后台顺序处理
- WebSocket 实时进度推送
- 音频自动切分和对齐
- 结果导出为 Excel 格式
- 用户认证和权限管理
- 支持 .wav 和 .mp3 格式音频

## 技术栈

### 后端
- **FastAPI** - 高性能 Web 框架
- **Python 3.12**
- **PyTorch** - 深度学习框架
- **CUDA** - GPU 加速

### 前端
- **React 18** - 用户界面框架
- **TypeScript** - 类型安全
- **Ant Design** - UI 组件库
- **Vite** - 构建工具

## 项目结构

```
AudioMOS/
├── backend/              # FastAPI 后端服务
│   ├── app/
│   │   ├── api/         # API 路由
│   │   │   ├── auth.py  # 认证接口
│   │   │   └── mos.py   # MOS 评分接口
│   │   ├── core/        # 核心模块
│   │   │   ├── config.py        # 配置管理
│   │   │   ├── task_queue.py    # 任务队列
│   │   │   ├── security.py      # 安全认证
│   │   │   └── mos_optimizer.py # MOS 优化
│   │   └── main.py      # 应用入口
│   └── run.py           # 启动脚本
├── frontend/            # React 前端应用
│   ├── src/
│   │   ├── pages/       # 页面组件
│   │   │   ├── Home.tsx # 主页面
│   │   │   └── Login.tsx# 登录页面
│   │   ├── contexts/    # React 上下文
│   │   └── services/    # API 服务
│   └── package.json
├── app/                 # 算法模块
│   ├── algorithms/      # MOS 评分算法
│   │   ├── dnsmos/      # DNSMOS 算法
│   │   ├── nisqa/       # NISQA 算法
│   │   ├── scoreq/      # Scoreq 算法
│   │   ├── tcf/         # 音色还原度
│   │   ├── utmos/       # UTMOS 算法
│   │   ├── speechmetrics/ # 语音指标库
│   │   └── wenet/       # 语音识别
│   └── core/            # 音频处理核心
│       ├── audio_cut.py      # 音频切分
│       ├── audio_processor.py # 音频对齐
│       └── mos_calculator.py  # MOS 计算
├── config/              # 配置文件
│   └── config.yaml      # 主配置
├── data/                # 数据目录
│   ├── ref/            # 参考音频
│   ├── uploads/        # 上传文件
│   ├── temp/           # 临时文件
│   └── results/        # 结果文件
├── requirements.txt     # Python 依赖
└── start.sh            # 启动脚本
```

## 快速开始

### 环境要求
- Python 3.12+
- Node.js 18+
- CUDA 12.8+ (推荐，用于 GPU 加速)

### 安装步骤

#### 1. 克隆项目
```bash
git clone <repository-url>
cd AudioMOS
```

#### 2. 创建 Python 虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 3. 安装 Python 依赖
```bash
pip install -r requirements.txt
```

#### 4. 安装前端依赖
```bash
cd frontend
npm install
```

#### 5. 配置系统
编辑 `config/config.yaml` 文件：
```yaml
auth:
  secret_key: "your-secret-key"  # 修改 JWT 密钥
  admin_username: "admin"         # 修改管理员账号
  admin_password: "your-password" # 修改管理员密码

paths:
  ref_dir: "./data/ref"  # 参考音频目录
```

#### 6. 准备参考音频（可选）
如需使用有参考指标，将参考音频放入 `data/ref/` 目录。

#### 7. 启动服务

**方式一：使用启动脚本**
```bash
./start.sh
```

**方式二：手动启动**
```bash
# 启动后端
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 启动前端（新终端）
cd frontend
npm run dev
```

### 访问系统
- 前端界面：http://localhost:5173
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs

## 使用说明

### 1. 登录系统
- 默认账号：admin
- 默认密码：tp123456
- 首次使用请修改默认密码

### 2. 上传音频
- 支持 .wav 和 .mp3 格式
- 支持批量上传
- 最大文件大小：100MB

### 3. 选择评分指标
- 有参考指标：需要参考音频
- 无参考指标：无需参考音频
- 可自由选择需要计算的指标

### 4. 查看结果
- 实时查看处理进度
- 任务完成后下载 Excel 结果文件
- 查看详细的评分数据

## 配置说明

### 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `server.host` | 服务器地址 | 0.0.0.0 |
| `server.port` | 服务器端口 | 8000 |
| `auth.secret_key` | JWT 密钥 | - |
| `auth.access_token_expire_minutes` | Token 过期时间 | 60 |
| `paths.ref_dir` | 参考音频目录 | ./data/ref |
| `paths.upload_dir` | 上传文件目录 | ./data/uploads |
| `paths.result_dir` | 结果文件目录 | ./data/results |
| `paths.temp_dir` | 临时文件目录 | ./data/temp |
| `cuda.enabled` | 是否启用 CUDA | true |
| `cuda.device_id` | GPU 设备 ID | 0 |

### 环境变量
可通过环境变量覆盖配置：
- `AUDIOMOS_HOST` - 服务器地址
- `AUDIOMOS_PORT` - 服务器端口
- `AUDIOMOS_SECRET_KEY` - JWT 密钥
- `AUDIOMOS_CUDA_ENABLED` - 是否启用 CUDA

## API 接口

### 认证接口
- `POST /api/auth/login` - 用户登录
- `GET /api/auth/me` - 获取当前用户信息
- `POST /api/auth/logout` - 用户登出

### MOS 评分接口
- `POST /api/mos/upload` - 上传音频文件
- `POST /api/mos/process/{task_id}` - 提交处理任务
- `GET /api/mos/tasks/{task_id}` - 获取任务状态
- `GET /api/mos/tasks` - 获取任务列表
- `GET /api/mos/download/{task_id}` - 下载结果文件
- `GET /api/mos/results/{task_id}` - 获取详细结果
- `DELETE /api/mos/tasks/{task_id}` - 删除任务
- `WS /api/mos/ws/{task_id}` - WebSocket 实时进度

## 开发指南

### 后端开发
```bash
cd backend
python -m uvicorn app.main:app --reload
```

### 前端开发
```bash
cd frontend
npm run dev
```

### 代码规范
- 后端：遵循 PEP 8 规范
- 前端：使用 ESLint + Prettier

## 注意事项

1. **安全性**
   - 生产环境务必修改默认密码和 JWT 密钥
   - 建议配置 HTTPS
   - 限制文件上传大小和类型

2. **性能优化**
   - 启用 CUDA 可大幅提升处理速度
   - 大文件建议分批处理
   - 定期清理临时文件

3. **存储空间**
   - 上传的音频文件会占用存储空间
   - 定期清理不需要的结果文件
   - 临时文件会自动清理

## 故障排除

### 常见问题

1. **CUDA 不可用**
   - 检查 CUDA 是否正确安装
   - 检查 PyTorch 版本是否与 CUDA 匹配

2. **模型加载失败**
   - 检查模型文件是否存在
   - 检查磁盘空间是否充足

3. **前端无法连接后端**
   - 检查后端服务是否启动
   - 检查 CORS 配置

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。


---

**注意**：本项目仅供学习和研究使用，商业使用请遵守相关算法和模型的许可协议。

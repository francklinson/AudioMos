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

### 音频处理功能
- **音频修复** - 去混响、超分辨率、降噪增强
  - ClearVoice FRCRN/MossFormer 系列模型
  - SpeechBrain SepFormer/MetricGAN+
  - 传统方法（谱减法、维纳滤波）
- **降噪测评** - 多种降噪算法对比评估
  - 支持 10+ 种降噪算法
  - PESQ/STOI/SISDR/DNSMOS 等多维度评估

### 系统特性
- **智能模型加载** - 启动时预加载核心模型，其他模型延迟加载
- **显存优化** - 自动显存清理，防止 OOM
- **任务队列系统** - 支持多用户并发提交，后台顺序处理
- **WebSocket 实时进度推送**
- **音频自动切分和对齐** - 基于 Shazam 指纹匹配
- **结果导出为 Excel 格式**
- **用户认证和权限管理**
- **支持 .wav 和 .mp3 格式音频**

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
│   │   │   ├── auth.py       # 认证接口
│   │   │   ├── mos.py        # MOS 评分接口
│   │   │   ├── restoration.py # 音频修复接口
│   │   │   └── denoise.py    # 降噪测评接口
│   │   ├── core/        # 核心模块
│   │   │   ├── config.py           # 配置管理
│   │   │   ├── task_queue.py       # 任务队列
│   │   │   ├── security.py         # 安全认证
│   │   │   ├── logging_config.py   # 日志配置
│   │   │   └── reference_matcher.py # 参考音频匹配
│   │   └── main.py      # 应用入口
│   └── run.py           # 启动脚本
├── frontend/            # React 前端应用
│   ├── src/
│   │   ├── pages/       # 页面组件
│   │   │   ├── Home.tsx     # MOS评分主页面
│   │   │   ├── Restoration.tsx # 音频修复页面
│   │   │   ├── Denoise.tsx   # 降噪测评页面
│   │   │   └── Login.tsx     # 登录页面
│   │   ├── contexts/    # React 上下文
│   │   └── services/    # API 服务
│   └── package.json
├── app/                 # 算法模块
│   ├── algorithms/      # 算法实现
│   │   ├── dnsmos/      # DNSMOS 算法
│   │   ├── nisqa/       # NISQA 算法
│   │   ├── scoreq/      # Scoreq 算法
│   │   ├── tcf/         # 音色还原度
│   │   ├── utmos/       # UTMOS 算法
│   │   ├── speechmetrics/ # 语音指标库 (STOI/SISDR/PESQ)
│   │   ├── wenet/       # 语音识别 (WER)
│   │   ├── denoise/     # 降噪算法库
│   │   │   ├── clearervoice_denoiser.py  # ClearVoice降噪
│   │   │   ├── speechbrain_denoiser.py   # SpeechBrain降噪
│   │   │   └── traditional_denoiser.py   # 传统降噪方法
│   │   └── restoration/ # 音频修复算法
│   │       ├── dereverberation.py    # 去混响
│   │       ├── super_resolution.py   # 超分辨率
│   │       └── denoise_adapter.py    # 降噪适配器
│   └── core/            # 音频处理核心
│       ├── calculator/       # MOS计算
│       │   └── mos_calculator.py
│       ├── reference_matcher.py    # 参考音频匹配(Shazam指纹)
│       ├── reference_pipeline.py   # 参考音频处理流程
│       └── audio_processor.py      # 音频处理器
├── tests/               # 测试套件
│   └── complete_test_suite.py  # 完整测试用例
├── config/              # 配置文件
│   └── config.yaml      # 主配置
├── data/                # 数据目录
│   ├── ref/            # 参考音频
│   ├── uploads/        # 上传文件
│   ├── temp/           # 临时文件
│   └── results/        # 结果文件
├── models/              # 模型文件
│   ├── tcf/            # TCF音色还原度模型
│   ├── utmos/          # UTMOS模型
│   ├── wenet/          # WeNet语音识别模型
│   └── ...
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
- **前端界面**：http://localhost:8002
- **API 文档**：http://localhost:8002/docs

### 默认账号
- 用户名：`admin`
- 密码：`tp123456`

## 使用说明

### 1. 登录系统
使用默认账号登录，首次使用建议修改默认密码。

### 2. MOS 评分
1. **上传音频** - 支持 .wav 和 .mp3 格式，批量上传
2. **选择指标** - 有参考指标（需参考音频）或无参考指标
3. **提交任务** - 系统自动处理，实时查看进度
4. **查看结果** - 下载 Excel 结果文件，查看详细评分数据

### 3. 音频修复
1. 选择修复算法（去混响、降噪、超分辨率等）
2. 上传待修复音频
3. 系统自动处理并生成修复后的音频
4. 支持试听对比原音频和修复后音频

### 4. 降噪测评
1. 上传带噪声音频和干净参考音频
2. 选择要测评的降噪算法
3. 系统自动计算 PESQ/STOI/SISDR/DNSMOS 等指标
4. 生成对比报告和评分排名

### 5. 参考音频管理
- 上传参考音频用于有参考指标计算
- 系统自动提取音频指纹（Shazam算法）
- 支持混合音频自动分段匹配

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

#### 认证接口
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/login` | 用户登录 |
| GET | `/api/auth/me` | 获取当前用户信息 |

#### MOS 评分接口
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/mos/upload` | 上传音频文件 |
| POST | `/api/mos/process/{task_id}` | 提交处理任务 |
| GET | `/api/mos/tasks/{task_id}` | 获取任务状态 |
| GET | `/api/mos/tasks` | 获取任务列表 |
| GET | `/api/mos/download/{task_id}` | 下载结果文件 |
| GET | `/api/mos/results/{task_id}` | 获取详细结果 |
| DELETE | `/api/mos/tasks/{task_id}` | 删除任务 |
| WS | `/api/mos/ws/{task_id}` | WebSocket 实时进度 |

#### 音频修复接口
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/restoration/algorithms` | 获取修复算法列表 |
| POST | `/api/restoration/upload` | 上传音频文件 |
| POST | `/api/restoration/process/{task_id}` | 提交修复任务 |
| GET | `/api/restoration/tasks/{task_id}` | 获取任务状态 |
| GET | `/api/restoration/tasks` | 获取任务列表 |
| GET | `/api/restoration/source/{task_id}` | 获取原始音频 |
| GET | `/api/restoration/result/{task_id}` | 下载修复后音频 |

#### 降噪测评接口
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/denoise/algorithms` | 获取降噪算法列表 |
| POST | `/api/denoise/tasks` | 创建测评任务 |
| GET | `/api/denoise/tasks/{task_id}` | 获取任务状态 |
| GET | `/api/denoise/tasks` | 获取任务列表 |
| GET | `/api/denoise/download/{task_id}` | 下载测评报告 |

#### 参考音频接口
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/reference-audio/list` | 获取参考音频列表 |
| POST | `/api/reference-audio/upload` | 上传参考音频 |
| DELETE | `/api/reference-audio/{audio_id}` | 删除参考音频 |
| GET | `/api/reference-audio/check/status` | 检查参考音频状态 |

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

## 测试

项目包含完整的测试套件：

```bash
# 运行完整测试
python3 tests/complete_test_suite.py
```

测试覆盖：
- 认证系统测试
- API 接口测试
- 性能测试（响应时间、并发）
- GPU 显存监控
- 端到端流程测试

## 模型加载策略

系统采用智能模型加载策略优化显存使用：

### 启动时预加载
- MOS 计算模型（DNSMOS、NISQA、ScoreQ、UTMOS、TCF）
- 轻量级降噪模型（FRCRN、MossFormerGAN）
- 常用音频修复算法

### 延迟加载
- 大型降噪模型（MossFormer2 系列）
- 其他音频修复算法

### 显存优化
- 自动显存清理（`torch.cuda.empty_cache()`）
- 支持设置 `PYTORCH_CUDA_ALLOC_CONF` 环境变量优化显存分配

## 故障排除

### 常见问题

1. **CUDA OOM（显存不足）**
   - 确保关闭其他占用显存的程序
   - 设置环境变量：`export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - 系统会自动清理显存，如仍不足请减少并发任务

2. **模型加载失败**
   - 检查模型文件是否完整（运行 `./start.sh` 会自动检查）
   - 检查磁盘空间是否充足
   - 查看日志：`logs/unified.log`

3. **参考音频匹配失败**
   - 确保参考音频已正确上传
   - 检查音频格式（支持 8kHz-48kHz）
   - 混合音频需要包含足够的参考音频片段（至少10个hash匹配）

4. **服务启动失败**
   - 检查端口 8002 是否被占用
   - 检查配置文件 `config/config.yaml` 是否正确
   - 查看启动日志排查错误

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request。


---

**注意**：本项目仅供学习和研究使用，商业使用请遵守相关算法和模型的许可协议。

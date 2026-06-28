# 工作记录 2026-06-28: 前端简化改造（React → HTML+JS+CSS）

## 概述
将项目前端从 React + TypeScript + Vite 架构简化为纯 HTML+JS+CSS 静态文件方案，消除 Node.js 构建依赖，解决生产环境因 Vite DNS 解析导致的绑定异常问题。

## 背景
- 生产环境部署时出现绑定IP错误，排查后发现是前端构建工具链问题
- 原前端基于 React 18 + TypeScript + Vite 5 + Ant Design 5，依赖复杂
- 部署服务器需同时维护 Python + Node.js 两套运行时
- node_modules 约 200MB+，构建过程脆弱（`--legacy-peer-deps`）
- 实际上本项目前端功能为典型的 CRUD + 文件操作，React 大材小用

## 改动内容

### 新增文件
1. **`backend/static/index.html`** (24KB)
   - 单页应用骨架，引入 Bootstrap 5.3 + Bootstrap Icons + Chart.js (CDN)
   - 包含登录页、主应用、4个 Tab 面板（MOS评分/降噪测评/音频修复/参考音频）
   - 4 个 Modal（结果查看/编辑/匹配测试/确认对话框）

2. **`backend/static/css/app.css`** (14KB)
   - 渐变主题风格（与原 React 版一致）
   - 登录页动画（浮动图标、脉冲徽标、滑入卡片）
   - 上传区域拖拽效果、算法卡片选择、任务列表、波形图
   - 响应式布局（移动端适配）

3. **`backend/static/js/app.js`** (47KB, ~1000行)
   - 完整 API 层（fetch 封装 + 认证拦截 + 401 自动跳转登录）
   - 认证系统（登录/登出/token 管理）
   - MOS评分页：文件拖拽上传、指标选择、任务轮询、结果查看(表格+Chart.js图表)、下载/删除
   - 降噪测评页：算法卡片选择、文件上传、任务管理、Excel/HTML/Markdown 三格式报告下载
   - 音频修复页：算法选择、文件上传、任务管理、试听对比(波形图+播放器)
   - 参考音频页：列表展示、上传/播放/编辑/删除、指纹数据库管理、内容匹配测试
   - 音频播放器：play/pause、进度条跳转
   - Canvas 波形图绘制
   - Toast 通知系统、确认对话框

### 修改文件
4. **`start.sh`**
   - 新增简化前端检测逻辑：检测到 `js/app.js` + `css/app.css` 时自动跳过 Node.js 构建
   - 原有 React 构建流程保留作为 fallback

### 删除文件
5. `backend/static/assets/index-BmSyKMt1.css` (旧 React 构建产物)
6. `backend/static/assets/index-DGQ2fIYi.js` (旧 React 构建产物)
7. `backend/static/vite.svg` (旧 React favicon)

## 架构对比

| 对比项 | 改造前 (React) | 改造后 (HTML+JS+CSS) |
|--------|---------------|---------------------|
| 前构建步骤 | npm install → tsc → vite build | 无 |
| 运行时依赖 | Node.js ≥ 18 + npm | 无 |
| node_modules | ~200MB+ | 0 |
| 服务器运行时 | Python + Node.js 两套 | 仅 Python |
| 部署流程 | 构建→复制→启动 | 直接启动 |
| 总代码量 | 多层目录+配置文件~2000行+依赖 | 3个文件~85KB |
| 前端框架 | React 18 + Ant Design 5 | Bootstrap 5 (CDN) |
| 语言 | TypeScript | 原生 JS |

## 测试结果 (58 项全部通过)

```
[1]  静态文件服务     ✅ 4/4  首页、CSS、JS、SPA回退
[2]  认证系统         ✅ 4/4  登录成功/失败、获取用户、未授权拦截
[3]  MOS评分API       ✅ 1/1  任务列表接口
[4]  降噪测评API      ✅ 2/2  算法列表、任务列表
[5]  音频修复API      ✅ 2/2  算法列表、任务列表
[6]  参考音频API      ✅ 3/3  列表、状态、指纹数据库
[7]  健康检查         ✅ 1/1
[8]  页面内容         ✅ 23/23 HTML结构完整性
[9]  JS函数           ✅ 15/15 功能函数就绪
[10] 功能集成         ✅ 3/3  参考音频数据验证
```

## 注意
- 使用 BootCDN 加载 Bootstrap 5.3，部署服务器需有外网访问权限；如无外网可下载到本地引用
- Chart.js 同样从 CDN 加载，用于 MOS 结果可视化
- 原 React 代码保留在 `frontend/` 目录作为参考，不再构建使用
- 如需回退到 React 版：删除 `js/app.js` 和 `css/app.css`，执行 `export AUDIOMOS_REBUILD_FRONTEND=1` 后启动

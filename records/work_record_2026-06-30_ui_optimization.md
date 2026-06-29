# UI优化工作记录

**日期**: 2026-06-30
**时间**: 00:22
**负责人**: zhouchenghao

---

## 任务概述

根据需求文档中的3个待办任务进行UI和系统优化：
1. 总任务/已完成/处理中状态显示更新不及时
2. 整理日志输出，查缺补漏，增强可读性
3. 处理进度在页面中友好显示

---

## [待办1] 统计卡片动画更新

### 问题
统计卡片（总任务/已完成/处理中）的数值更新时没有视觉反馈，用户无法感知状态变化。

### 修改

**CSS** (`frontend/static/css/app.css`):
- 新增 `stat-updated` 动画类：`@keyframes statFlash` 缩放+高亮动画
- 新增 `stat-updated-at` 样式：显示最后更新时间

**JavaScript** (`frontend/static/js/app.js`):
- 新增 `state.mosPrevStats` 跟踪前一次统计值
- `updateMosStats()` 函数增加变化检测逻辑
- 数值变化时添加 `stat-updated` 类触发CSS动画
- 每个stat-card底部添加"更新于 HH:MM:SS"时间标签

### 影响文件
- `frontend/static/css/app.css` (新增约50行)
- `frontend/static/js/app.js` (updateMosStats函数重写)

---

## [待办2] 日志输出整理

### 问题
- 日志格式不够清晰，缺少函数名和行号
- 缺少HTTP请求耗时日志
- 控制台日志格式冗长（含完整时间戳）

### 修改

**日志配置** (`backend/app/core/logging_config.py`):
- 文件日志格式改为：`时间 | 级别 | 模块.函数:行号 | 消息`
- 控制台日志格式改为：`时间 | 级别 | 消息`（仅含时分秒，更简洁）
- 新增 `log_request()` 工具函数：统一HTTP请求日志记录格式
- 文件日志始终启用，控制台日志也始终启用(非DEBUG时用INFO级别)

**请求中间件** (`backend/app/main.py`):
- 新增 `log_request_duration` 中间件
- 自动记录每个API请求的方法、路径、状态码、耗时
- 仅记录 `/api/` 和 `/health` 路径，跳过静态文件

### 影响文件
- `backend/app/core/logging_config.py` (重写日志格式)
- `backend/app/main.py` (新增中间件)

---

## [待办3] 处理进度友好显示

### 问题
- 进度仅显示一个百分比数字，不够直观
- 无分步骤显示（用户不知道当前在做什么）
- 轮询间隔固定5秒，不够灵活
- 已有WebSocket端点(`/api/mos/ws/{task_id}`)但前端未使用

### 修改

**后端** (`backend/app/api/mos.py`):
- 新增 `PROGRESS_STEPS` 映射：progress值到步骤名（uploading/matching/splitting/computing/generating/done）
- 新增 `_get_step_name()`：根据progress值推算当前步骤
- `update_task_progress()` 改为结构化消息格式：`[步骤名]描述`
- 新增WebSocket进度推送：`asyncio.ensure_future(manager.send_progress(...))`

**JavaScript** (`frontend/static/js/app.js`):
- 新增 `state.mosStepNames` / `state.mosStepOrder`：步骤名和显示名称映射
- 新增 `connectMosWs()` / `disconnectMosWs()`：WebSocket连接管理（自动重连+5分钟超时）
- 新增 `renderMosProgressSteps()`：渲染分步进度UI（已完成✓ / 处理中⏳ / 待处理○）
- 新增 `renderMosTasks()` 增强：
  * 进度条使用 `progress-enhanced` 样式（带光泽动画和百分比标签）
  * 解析 `[步骤名]描述` 格式展示分步进度
  * 处理中任务自动建立WebSocket连接
- 轮询改为 `setTimeout` 递归模式（`schedulePoll`/`doPoll`）：
  * 有处理中任务时2秒轮询
  * 无活动任务时5秒轮询
  * 等待API响应后才安排下一次轮询，避免并发堆积
- `loadMosTasks()` 改为返回 tasks 数组供轮询判断

**CSS** (`frontend/static/css/app.css`):
- 新增 `progress-detail` 样式：进度详情面板
- 新增 `progress-step` 样式：分步进度项（active/completed状态）
- 新增 `progress-enhanced` 样式：增强进度条（8px高+光泽动画+百分比标签）
- 新增 `@keyframes progressShine`：进度条光泽扫描动画

### 影响文件
- `backend/app/api/mos.py` (update_task_progress重写 + WebSocket推送)
- `frontend/static/js/app.js` (大量新增)
- `frontend/static/css/app.css` (新增约80行)

---

## 修复: 进度步骤提前显示✓的问题

### 问题
`renderMosProgressSteps()` 根据 `progress` 百分比推断已完成步骤（如 `progress>=40` 认为uploading/matching/splitting已完成），但后端进度可能跳跃（5%→50%），导致未实际执行的步骤被错误打上 ✓。

### 根因
用 `progress` 值替代真正的步骤状态做推断。

### 修复
改为根据后端报告的**当前步骤名**在 `mosStepOrder` 数组中的索引位置确定：
- 当前步骤 → ⏳ 活跃
- 之前的步骤 → ✓ 已完成（确实已执行完毕）
- 之后的步骤 → ○ 待处理
- **不再依赖 progress 值做任何推断**

### 影响文件
- `frontend/static/js/app.js` (`renderMosProgressSteps` 重写)

---

## 效果说明

### 前端展示改进
1. **统计卡片**: 数值变化时自动放大高亮闪烁 + 显示最后更新时间
2. **进度条**: 更粗(8px) + 光泽扫描动画 + 百分比标签
3. **分步进度**: 实时显示6个步骤的完成状态（上传→匹配→切分→计算→报告→完成）
4. **轮询加速**: 有任务处理中时2秒刷新，空闲时5秒

### 日志改进
1. **文件日志**: 格式更清晰，含函数名和行号便于定位
2. **控制台日志**: 更简洁，仅显示时分秒
3. **请求日志**: 自动记录API耗时，便于性能监控

### WebSocket实时推送
- 任务处理过程中，后端主动推送进度到前端
- 前端实时更新进度条和分步状态
- 无需等待轮询周期，即时反馈

---

## 涉及文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `frontend/static/css/app.css` | 修改 | 新增动画/进度/统计样式(~130行) |
| `frontend/static/js/app.js` | 修改 | 新增WebSocket/步骤/动态轮询 |
| `backend/app/api/mos.py` | 修改 | 结构化进度+WebSocket推送 |
| `backend/app/core/logging_config.py` | 修改 | 优化日志格式+请求日志函数 |
| `backend/app/main.py` | 修改 | 新增请求耗时中间件 |

---

## 后续建议

1. **降噪测评/音频修复模块**可复用相同的WebSocket进度推送模式
2. 可考虑将计算过程中的每个指标完成情况实时推送到前端（如PESQ完成✅/STOI完成✅等）
3. 日志级别可通过配置文件动态调整，生产环境建议使用INFO

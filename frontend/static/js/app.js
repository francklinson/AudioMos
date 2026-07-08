/* ===================== AudioMOS 前端应用 ===================== */

// ======================== 工具函数 ========================
const $ = id => document.getElementById(id);
const qs = (s, c) => (c || document).querySelector(s);
const qsa = (s, c) => (c || document).querySelectorAll(s);

// 任务列表分页
const MOS_PAGE_SIZE = 10;           // 每页条数
let _mosAllTasks = [];              // 全量任务（最新在前）
let _mosPage = 1;                   // 当前页码

function formatTime(t) {
  if (!t || isNaN(t)) return '00:00';
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function formatSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let size = bytes;
  while (size >= 1024 && i < units.length - 1) { size /= 1024; i++; }
  return `${size.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

function formatDate(ts) {
  if (!ts) return '-';
  const d = new Date(ts);
  return `${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function statusBadge(status) {
  const map = {
    completed: '<span class="badge badge-status-completed">已完成</span>',
    processing: '<span class="badge badge-status-processing">处理中</span>',
    failed: '<span class="badge badge-status-failed">失败</span>',
    pending: '<span class="badge badge-status-pending">排队中</span>',
    queued: '<span class="badge badge-status-queued">排队中</span>',
  };
  return map[status] || `<span class="badge bg-secondary">${status}</span>`;
}

function shortId(id) { return id ? id.slice(0, 8) : ''; }

// ======================== Toast ========================
function showToast(msg, type = 'info') {
  let container = $('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const icons = { success: 'bi-check-circle-fill', error: 'bi-x-circle-fill', warning: 'bi-exclamation-triangle-fill', info: 'bi-info-circle-fill' };
  const el = document.createElement('div');
  el.className = `toast-custom ${type}`;
  el.innerHTML = `<i class="bi ${icons[type] || icons.info}"></i> ${msg}`;
  container.appendChild(el);
  setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity .3s'; setTimeout(() => el.remove(), 300); }, 3000);
}

// ======================== API ========================
function getToken() { return localStorage.getItem('token'); }
function setToken(t) { localStorage.setItem('token', t); }
function clearToken() { localStorage.removeItem('token'); }

function getAuthHeaders() {
  const t = getToken();
  return t ? { 'Authorization': `Bearer ${t}` } : {};
}

async function api(path, opts = {}) {
  const { body, formData, method = 'GET', raw } = opts;
  const headers = { ...(opts.headers || {}) };
  if (!formData) {
    headers['Content-Type'] = 'application/json';
  }
  Object.assign(headers, getAuthHeaders());
  // 确保方法大写
  const meth = method.toUpperCase();
  const resp = await fetch(path, {
    method: meth,
    headers,
    body: formData ? body : (body ? JSON.stringify(body) : undefined),
  });
  if (resp.status === 401) {
    clearToken();
    // 不在此处自动跳转——让 checkAuth 或调用者处理，避免轮询时意外退出
    throw new Error('未授权');
  }
  if (!resp.ok) {
    let msg = `请求失败 (${resp.status})`;
    try { const e = await resp.json(); msg = e.detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  if (raw) return resp;
  const text = await resp.text();
  return text ? JSON.parse(text) : null;
}

function apiUrl(path) { return path; }

// ======================== 认证 ========================
function showApp(show) {
  // 隐藏加载状态
  const loading = $('loading-section');
  if (loading) loading.classList.add('d-none');

  $('login-section').classList.toggle('d-none', show);
  $('login-section').classList.toggle('d-flex', !show);
  $('app-section').classList.toggle('d-none', !show);
}

let _authing = false;

/**
 * 带超时的 fetch 封装：超过 timeoutMs 毫秒则 reject
 */
function fetchWithTimeout(url, options = {}, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(timer));
}

function checkAuth() {
  if (_authing) return;
  const token = getToken();
  if (token) {
    _authing = true;

    // 有 token 时立即显示主页面，避免"验证会话中..."闪烁
    showApp(true);

    const fallbackTimer = setTimeout(() => {
      if (_authing) {
        _authing = false;
        showToast('与服务器连接超时，部分功能可能不可用', 'error');
      }
    }, 8000);

    api(apiUrl('/api/auth/me')).then(user => {
      clearTimeout(fallbackTimer);
      state.user = user;
      $('user-display').textContent = user.username;
      loadAllData();
    }).catch((err) => {
      clearTimeout(fallbackTimer);
      // 区分认证错误和网络错误
      if (err.name === 'AbortError' || err.message === '未授权') {
        // 401 或超时 → 清除 token，踢回登录页
        clearToken();
        showApp(false);
        showToast(err.name === 'AbortError' ? '连接服务器超时，请重新登录' : '登录已过期，请重新登录', 'error');
      } else {
        // 网络错误 — 已在主页面，保留界面，后台重试
        showToast('连接服务器失败 (' + err.message + ')', 'error');
        setTimeout(() => { _authing = false; checkAuth(); }, 3000);
        return;
      }
    }).finally(() => { if (_authing) _authing = false; });
  } else {
    showApp(false);
  }
}

async function handleLogin(e) {
  e.preventDefault();
  e.stopPropagation();

  // 手动验证：确保字段非空
  const username = $('username-input').value.trim();
  const password = $('password-input').value;
  if (!username || !password) {
    showToast('请输入用户名和密码', 'warning');
    return;
  }

  const btn = $('login-btn');
  const text = $('login-btn-text');
  const spinner = $('login-spinner');
  btn.disabled = true;
  text.textContent = '登录中...';
  spinner.classList.remove('d-none');
  try {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);
    const data = await fetchWithTimeout(apiUrl('/api/auth/login'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: params.toString(),
    }, 10000);
    if (!data.ok) {
      let msg = '用户名或密码错误';
      try { const e = await data.json(); msg = e.detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const result = await data.json();
    setToken(result.access_token);
    showToast('登录成功', 'success');
    checkAuth();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    btn.disabled = false;
    text.textContent = '登录';
    spinner.classList.add('d-none');
  }
}

function handleLogout() {
  api(apiUrl('/api/auth/logout'), { method: 'POST' }).catch(() => {});
  clearToken();
  showApp(false);
  showToast('已退出登录', 'info');
}

// ======================== 全局状态 ========================
const state = {
  user: null,
  mosPollTimer: null,
  mosPrevStats: { total: -1, completed: -1, processing: -1 },
  mosWsConnections: {},       // { taskId: WebSocket }
  mosStepNames: {             // 进度步骤名称映射
    'uploading': '上传文件中',
    'matching': '匹配参考音频',
    'splitting': '切分音频',
    'computing': '计算MOS得分',
    'generating': '生成报告',
    'done': '处理完成',
  },
  mosStepOrder: ['uploading', 'matching', 'splitting', 'computing', 'generating', 'done'],
  restorationWsConnections: {},   // { taskId: WebSocket }
  restorationStepNames: {
    'queued': '排队等待',
    'loading': '加载模型',
    'reading': '读取音频',
    'processing': '执行修复',
    'saving': '保存结果',
    'done': '处理完成',
  },
  restorationStepOrder: ['queued', 'loading', 'reading', 'processing', 'saving', 'done'],
  _pollInterval: 5000,        // 默认轮询间隔
};

// ======================== 页面初始化数据加载 ========================
function loadAllData() {
  loadMosTasks();
  loadRestorationAlgorithms();
  loadRestorationTasks();
  loadRefAudioList();
  loadAsrAlgorithms();
  loadAsrDatasets();
}

// ==================== MOS 评分 ====================
let mosFiles = [];
let mosMetrics = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos'];
const MOS_ALL_METRICS = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos'];
const MOS_REF_METRICS = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf'];
const MOS_DEFAULT_METRICS = ['pesq', 'stoi', 'sisdr', 'wer', 'tcf', 'dnsmos', 'nisqa', 'scoreq', 'utmos'];

function initMosPage() {
  // 上传区域（文件 input 用 absolute 覆盖层，直接点它即可打开选择器，不加额外 click 防止冲突）
  const zone = $('mos-upload-zone');
  const input = $('mos-file-input');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    handleMosFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', () => handleMosFiles(input.files));

  // 指标配置
  $('mos-toggle-config').addEventListener('click', () => {
    const pane = $('mos-metrics-config');
    const btn = $('mos-toggle-config');
    pane.classList.toggle('d-none');
    btn.innerHTML = pane.classList.contains('d-none') ? '<i class="bi bi-gear"></i> 展开' : '<i class="bi bi-gear"></i> 收起';
  });

  qsa('.mos-metric').forEach(cb => cb.addEventListener('change', updateMosMetrics));
  $('mos-select-all').addEventListener('click', () => { qsa('.mos-metric').forEach(c => c.checked = true); updateMosMetrics(); });
  $('mos-deselect-all').addEventListener('click', () => { qsa('.mos-metric').forEach(c => c.checked = false); updateMosMetrics(); });
  $('mos-reset-default').addEventListener('click', () => {
    qsa('.mos-metric').forEach(c => c.checked = MOS_DEFAULT_METRICS.includes(c.value));
    updateMosMetrics();
  });

  $('mos-submit-btn').addEventListener('click', uploadMosFiles);
}

function handleMosFiles(files) {
  mosFiles = Array.from(files).filter(f => /\.(wav|mp3)$/i.test(f.name));
  const skipped = files.length - mosFiles.length;
  if (skipped > 0) showToast(`已过滤 ${skipped} 个非音频文件`, 'warning');
  updateMosSubmitBtn();
}

function updateMosMetrics() {
  mosMetrics = [];
  qsa('.mos-metric:checked').forEach(cb => mosMetrics.push(cb.value));
  // 更新标签显示
  qsa('#mos-metrics-summary .badge').forEach(b => {
    const m = b.dataset.metric;
    b.classList.toggle('bg-primary', mosMetrics.includes(m));
    b.classList.toggle('bg-secondary', !mosMetrics.includes(m));
  });
  updateMosSubmitBtn();
}

function updateMosSubmitBtn() {
  const btn = $('mos-submit-btn');
  const text = $('mos-submit-text');
  if (mosFiles.length === 0) {
    btn.disabled = true;
    text.textContent = '请选择音频文件';
  } else if (mosMetrics.length === 0) {
    btn.disabled = true;
    text.textContent = '请至少选择一项计算指标';
  } else {
    btn.disabled = false;
    text.textContent = `开始上传并处理 (${mosFiles.length} 个文件)`;
  }
}

async function uploadMosFiles() {
  const btn = $('mos-submit-btn');
  btn.disabled = true;
  $('mos-submit-text').textContent = '上传处理中...';
  const fileCount = mosFiles.length;
  try {
    // 先上传
    const fd = new FormData();
    mosFiles.forEach(f => fd.append('files', f));
    if (mosMetrics.length > 0) fd.append('metrics', JSON.stringify(mosMetrics));
    const uploadResult = await api(apiUrl('/api/mos/upload'), { method: 'POST', formData: true, body: fd });
    const taskId = uploadResult.task_id;
    if (!taskId) throw new Error('后端未返回任务ID');
    // 启动处理
    await api(apiUrl(`/api/mos/process/${taskId}`), { method: 'POST' });
    mosFiles = [];
    $('mos-file-input').value = '';
    updateMosSubmitBtn();
    // 等待任务列表刷新后再提示
    await loadMosTasks();
    showToast(`已提交任务，正在处理 ${fileCount} 个文件`, 'success');
  } catch (err) {
    showToast('上传失败: ' + err.message, 'error');
    updateMosSubmitBtn();
  }
}

async function loadMosTasks() {
  try {
    const data = await api(apiUrl('/api/mos/tasks'));
    const tasks = data.tasks || data || [];
    renderMosTasks(tasks);
    updateMosStats(tasks);
    return tasks;
  } catch (_) { return []; }
}

/** 判断任务是否为处理中/排队中 */
function _isActiveTask(t) {
  return t.status === 'processing' || t.status === 'pending' || t.status === 'queued';
}

/** 构建单个任务卡片的HTML */
function _buildMosTaskHtml(t) {
  const status = t.status || 'pending';
  const progress = t.progress || 0;
  // 优先使用后端返回的 file_summary，降级到文件名列表
  const fileSummary = t.file_summary || (
    t.files ? (Array.isArray(t.files) ? t.files.join(', ') : t.files) :
    (t.file_name || '')
  );
  const fileCount = t.file_count || 0;
  const isProcessing = _isActiveTask(t);
  const msg = t.message || '';
  const stepMatch = msg.match(/^\[(\w+)\](.+)/);
  const stepDesc = stepMatch ? stepMatch[2].trim() : msg;
  return `<div class="task-item" id="mos-task-${t.task_id}">
    <div class="task-header">
      <div class="task-info">
        <div class="task-title">
          <span class="task-file-count badge bg-secondary me-1">${fileCount}个文件</span>
          <span class="task-file">${_escapeHtml(fileSummary)}</span>
        </div>
        <div class="task-meta text-muted small">
          创建: ${formatDate(t.created_at || t.create_time)}
        </div>
      </div>
      <div class="task-actions">
        ${statusBadge(status)}
        ${status === 'completed' ? `<button class="btn btn-sm btn-outline-info" onclick="showMosResult('${t.task_id}')"><i class="bi bi-eye"></i> 查看</button>
          <button class="btn btn-sm btn-outline-success" onclick="downloadMosResult('${t.task_id}')"><i class="bi bi-download"></i> 下载</button>` : ''}
        <button class="btn btn-sm btn-outline-danger" onclick="deleteMosTask('${t.task_id}')"><i class="bi bi-trash"></i></button>
      </div>
    </div>
    ${isProcessing ? `<div class="mt-2 progress-wrap">
      <div class="progress progress-enhanced" style="height:8px">
        <div class="progress-bar progress-bar-striped progress-bar-animated" style="width:${progress}%"></div>
        <span class="progress-percent">${progress}%</span>
      </div>
    </div>
    <div class="progress-detail" id="mos-progress-detail-${shortId(t.task_id)}">
      <div class="progress-step active">
        <span class="step-icon"><div class="spinner-border spinner-sm" role="status"></div></span>
        <span>${_escapeHtml(stepDesc || '处理中...')}</span>
      </div>
    </div>` : ''}
    <div class="task-detail small text-muted">${t.message && !stepMatch ? _escapeHtml(t.message) : ''}</div>
  </div>`;
}

/** 增量更新已存在的任务卡片的进度和状态（不替换DOM） */
function _updateMosTaskElement(el, t) {
  const taskId = el.id.replace('mos-task-', '');
  const status = t.status || 'pending';
  const progress = t.progress || 0;
  const msg = t.message || '';
  const stepMatch = msg.match(/^\[(\w+)\](.+)/);
  const stepDesc = stepMatch ? stepMatch[2].trim() : msg;

  // 1. 更新状态徽章
  const badgeContainer = el.querySelector('.task-actions');
  if (badgeContainer) {
    const oldBadge = badgeContainer.querySelector('.badge');
    const newBadgeHtml = statusBadge(status);
    if (!oldBadge || oldBadge.outerHTML !== newBadgeHtml) {
      if (oldBadge) oldBadge.outerHTML = newBadgeHtml;
      else badgeContainer.insertAdjacentHTML('afterbegin', newBadgeHtml);
    }
  }

  // 2. 更新/移除查看/下载按钮（状态切换时：processing→completed 或反之）
  const hasViewBtn = el.querySelector('.btn-outline-info');
  const shouldHaveViewBtn = status === 'completed';
  if (shouldHaveViewBtn && !hasViewBtn) {
    const actionsDiv = el.querySelector('.task-actions');
    if (actionsDiv) {
      // 在删除按钮之前插入查看/下载按钮，保持 DOM 顺序一致性
      const delBtn = actionsDiv.querySelector('.btn-outline-danger');
      if (delBtn) {
        delBtn.insertAdjacentHTML('beforebegin',
          `<button class="btn btn-sm btn-outline-info" onclick="showMosResult('${taskId}')"><i class="bi bi-eye"></i> 查看</button>
         <button class="btn btn-sm btn-outline-success" onclick="downloadMosResult('${taskId}')"><i class="bi bi-download"></i> 下载</button>`);
      } else {
        actionsDiv.insertAdjacentHTML('beforeend',
          `<button class="btn btn-sm btn-outline-info" onclick="showMosResult('${taskId}')"><i class="bi bi-eye"></i> 查看</button>
         <button class="btn btn-sm btn-outline-success" onclick="downloadMosResult('${taskId}')"><i class="bi bi-download"></i> 下载</button>`);
      }
    }
  } else if (!shouldHaveViewBtn && hasViewBtn) {
    // 从completed变成非completed（罕见但处理）
    const btns = el.querySelectorAll('.btn-outline-info, .btn-outline-success');
    btns.forEach(b => b.remove());
  }

  // 3. 更新进度条
  const progressWrap = el.querySelector('.progress-wrap');
  if (status === 'processing' || status === 'pending') {
    if (!progressWrap) {
      // 尚未有进度条 → 插入
      const detailDiv = el.querySelector('.task-detail');
      const newWrap = document.createElement('div');
      newWrap.className = 'mt-2 progress-wrap';
      newWrap.innerHTML = `<div class="progress progress-enhanced" style="height:8px">
        <div class="progress-bar progress-bar-striped progress-bar-animated" style="width:${progress}%"></div>
        <span class="progress-percent">${progress}%</span>
      </div>`;
      el.insertBefore(newWrap, detailDiv);

      const newDetail = document.createElement('div');
      newDetail.className = 'progress-detail';
      newDetail.id = `mos-progress-detail-${shortId(taskId)}`;
      newDetail.innerHTML = `<div class="progress-step active"><span class="step-icon"><div class="spinner-border spinner-sm" role="status"></div></span><span>${_escapeHtml(stepDesc || '处理中...')}</span></div>`;
      el.insertBefore(newDetail, detailDiv);
    } else {
      // 已有进度条 → 只改宽度和百分比文字
      const bar = progressWrap.querySelector('.progress-bar');
      const pct = progressWrap.querySelector('.progress-percent');
      if (bar) bar.style.width = progress + '%';
      if (pct) pct.textContent = progress + '%';
    }
  } else {
    // 非处理中 → 移除进度条
    if (progressWrap) progressWrap.remove();
    const detailDiv = el.querySelector('.progress-detail');
    if (detailDiv) detailDiv.remove();
  }

  // 4. 更新步骤详情 — 仅在无WebSocket连接时通过轮询更新
  //    有WebSocket时由ws实时推送，轮询不覆盖（防止实时步骤跳跃）
  if (status === 'processing' || status === 'pending') {
    const hasWs = state.mosWsConnections && !!state.mosWsConnections[taskId];
    if (!hasWs) {
      renderMosProgressSteps(taskId, progress, msg);
    }
  }
}

/** HTML转义（防止文件名/消息中包含特殊字符破坏HTML） */
function _escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/**
 * 渲染MOS任务列表 — 智能增量更新
 *
 * 关键设计：避免全量innerHTML替换，防止WebSocket实时进度被轮询重置。
 * - 首次渲染：创建当前页所有任务卡片
 * - 后续轮询：增量更新已有卡片的进度/状态
 * - 新任务出现时：插入到当前页顶部
 * - 分页：按时间倒序分页显示，每页 MOS_PAGE_SIZE 条
 */
function renderMosTasks(tasks) {
  const container = $('mos-task-list');
  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无任务</div>';
    _mosAllTasks = [];
    return;
  }

  // 缓存全量任务（最新在前）
  _mosAllTasks = [...tasks].sort((a, b) => {
    const ta = a.created_at || a.create_time || '';
    const tb = b.created_at || b.create_time || '';
    return ta > tb ? -1 : ta < tb ? 1 : 0;
  });

  const needsFullRender = !container.querySelector('.task-item');

  if (needsFullRender) {
    // ====== 全量渲染 ======
    _renderCurrentPage();
    return;
  }

  // ====== 增量更新 ======
  const existingIds = new Set();
  const taskContainer = $('mos-task-container');

  _mosAllTasks.forEach(t => {
    existingIds.add(t.task_id);
    const el = $(`mos-task-${t.task_id}`);
    if (el) {
      _updateMosTaskElement(el, t);
    } else {
      // 新任务 → 插入到列表顶部
      (taskContainer || container).insertAdjacentHTML('afterbegin', _buildMosTaskHtml(t));
      if (_isActiveTask(t)) connectMosWs(t.task_id);
    }
  });

  // 移除已删除的任务
  container.querySelectorAll('.task-item[id^="mos-task-"]').forEach(el => {
    const id = el.id.replace('mos-task-', '');
    if (!existingIds.has(id)) {
      disconnectMosWs(id);
      el.remove();
    }
  });

  // 翻页到第一页展示最新任务
  if (_mosPage !== 1) {
    _mosPage = 1;
    _renderCurrentPage();
  } else {
    _syncPaginationControls();
  }
}

function _renderCurrentPage() {
  const container = $('mos-task-list');
  const totalPages = Math.ceil(_mosAllTasks.length / MOS_PAGE_SIZE) || 1;
  if (_mosPage > totalPages) _mosPage = totalPages;
  if (_mosPage < 1) _mosPage = 1;

  const start = (_mosPage - 1) * MOS_PAGE_SIZE;
  const pageTasks = _mosAllTasks.slice(start, start + MOS_PAGE_SIZE);

  let html = '<div id="mos-task-container">';
  pageTasks.forEach(t => { html += _buildMosTaskHtml(t); });
  html += '</div>';
  html += _buildPaginationHtml();
  container.innerHTML = html;
  pageTasks.forEach(t => { if (_isActiveTask(t)) connectMosWs(t.task_id); });
}

function _buildPaginationHtml() {
  const totalCount = _mosAllTasks.length;
  const totalPages = Math.ceil(totalCount / MOS_PAGE_SIZE) || 1;
  if (totalPages <= 1) return '';

  if (_mosPage > totalPages) _mosPage = totalPages;
  if (_mosPage < 1) _mosPage = 1;

  let html = '<div class="d-flex justify-content-center align-items-center gap-2 py-2" id="mos-pagination">';
  html += `<button class="btn btn-sm btn-outline-secondary" onclick="_changeMosPage(-1)" ${_mosPage <= 1 ? 'disabled' : ''}><i class="bi bi-chevron-left"></i> 上一页</button>`;

  // 页码按钮
  const rangeStart = Math.max(1, _mosPage - 2);
  const rangeEnd = Math.min(totalPages, _mosPage + 2);
  for (let p = rangeStart; p <= rangeEnd; p++) {
    if (p === _mosPage) {
      html += `<span class="btn btn-sm btn-primary disabled">${p}</span>`;
    } else {
      html += `<button class="btn btn-sm btn-outline-secondary" onclick="_goMosPage(${p})">${p}</button>`;
    }
  }

  html += `<button class="btn btn-sm btn-outline-secondary" onclick="_changeMosPage(1)" ${_mosPage >= totalPages ? 'disabled' : ''}>下一页 <i class="bi bi-chevron-right"></i></button>`;
  html += `<span class="text-muted small ms-1">共 ${totalCount} 条</span>`;
  html += '</div>';
  return html;
}

function _syncPaginationControls() {
  const pagination = $('mos-pagination');
  if (!pagination) return;
  const totalCount = _mosAllTasks.length;
  const totalPages = Math.ceil(totalCount / MOS_PAGE_SIZE) || 1;
  if (totalPages <= 1) { pagination.remove(); return; }
  // 替换整个分页控件
  pagination.outerHTML = _buildPaginationHtml();
}

function _changeMosPage(delta) {
  _mosPage += delta;
  const totalPages = Math.ceil(_mosAllTasks.length / MOS_PAGE_SIZE) || 1;
  if (_mosPage < 1) _mosPage = 1;
  if (_mosPage > totalPages) _mosPage = totalPages;
  _renderCurrentPage();
}

function _goMosPage(page) {
  _mosPage = page;
  _renderCurrentPage();
}

/** WebSocket推送的处理进度步骤 — 根据后端报告的步骤名确定完成状态 */
function renderMosProgressSteps(taskId, progress, message) {
  const shortId_ = shortId(taskId);
  const container = $(`mos-progress-detail-${shortId_}`);
  if (!container) return;
  // 如果消息是纯文本（非步骤格式），直接更新文本
  const stepMatch = message ? message.match(/^\[(\w+)\](.+)/) : null;
  if (!stepMatch) {
    const activeStep = container.querySelector('.progress-step.active');
    if (activeStep) {
      activeStep.innerHTML = `<span class="step-icon"><div class="spinner-border spinner-sm" role="status"></div></span><span>${message || '处理中...'}</span>`;
    }
    return;
  }
  const currentStep = stepMatch[1].toLowerCase();
  const stepDesc = stepMatch[2].trim();

  // 关键：根据步骤名在 order 中的位置确定完成状态
  // 步骤顺序: ['uploading','matching','splitting','computing','generating','done']
  // 当前步骤 → 活跃(⏳)，之前的步骤 → 已完成(✓)，之后的步骤 → 待处理(○)
  // 绝不靠 progress 值来猜"是否完成"——后端报告了哪个步骤就是哪个
  const order = state.mosStepOrder;
  const currentIdx = order.indexOf(currentStep);
  const hasCurrent = currentIdx >= 0;

  let html = '';
  order.forEach(step => {
    const stepLabel = state.mosStepNames[step] || step;
    const idx = order.indexOf(step);

    const isCompleted = hasCurrent && idx < currentIdx;
    const isActive = step === currentStep;

    let iconHtml;
    if (isActive) {
      iconHtml = '<div class="spinner-border spinner-sm" role="status"></div>';
    } else if (isCompleted) {
      iconHtml = '<i class="bi bi-check-circle-fill text-success" style="font-size:.85rem"></i>';
    } else {
      iconHtml = '<i class="bi bi-circle" style="font-size:.85rem;color:#ddd"></i>';
    }

    html += `<div class="progress-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}">
      <span class="step-icon">${iconHtml}</span>
      <span>${stepLabel}</span>
      ${isActive ? `<span class="ms-1 small text-muted">— ${stepDesc}</span>` : ''}
    </div>`;
  });

  container.innerHTML = html;
}

/** 连接到WebSocket获取实时进度 */
function connectMosWs(taskId) {
  // 避免重复连接
  if (state.mosWsConnections[taskId]) return;
  try {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/mos/ws/${taskId}`;
    const ws = new WebSocket(wsUrl);
    state.mosWsConnections[taskId] = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status && data.progress !== undefined) {
          // 更新任务列表中的进度条
          const taskEl = $(`mos-task-${taskId}`);
          if (taskEl) {
            const progressBar = taskEl.querySelector('.progress-bar');
            const percentEl = taskEl.querySelector('.progress-percent');
            if (progressBar) progressBar.style.width = data.progress + '%';
            if (percentEl) percentEl.textContent = data.progress + '%';
          }
          // 更新详细步骤
          if (data.progress < 100) {
            renderMosProgressSteps(taskId, data.progress, data.message || '');
          } else {
            // 任务完成 - 关闭WebSocket
            disconnectMosWs(taskId);
          }
        }
      } catch (_) {}
    };

    ws.onerror = () => { disconnectMosWs(taskId); };
    ws.onclose = () => { disconnectMosWs(taskId); };

    // 5秒后自动断开（防止残留连接）
    setTimeout(() => {
      if (state.mosWsConnections[taskId]) {
        disconnectMosWs(taskId);
      }
    }, 300000); // 5分钟超时
  } catch (_) {
    // WebSocket不可用时静默失败，靠轮询兜底
  }
}

function disconnectMosWs(taskId) {
  if (state.mosWsConnections[taskId]) {
    try {
      state.mosWsConnections[taskId].close();
    } catch (_) {}
    delete state.mosWsConnections[taskId];
  }
}

function updateMosStats(tasks) {
  if (!tasks) return;
  const total = tasks.length;
  const completed = tasks.filter(t => t.status === 'completed').length;
  const processing = tasks.filter(t => t.status === 'processing' || t.status === 'pending' || t.status === 'queued').length;

  // 检测变化并触发动画
  const prev = state.mosPrevStats;
  const changed = [];
  if (prev.total !== -1 && prev.total !== total) changed.push('mos-stat-total');
  if (prev.completed !== -1 && prev.completed !== completed) changed.push('mos-stat-completed');
  if (prev.processing !== -1 && prev.processing !== processing) changed.push('mos-stat-processing');

  state.mosPrevStats = { total, completed, processing };

  // 更新值
  const totalEl = $('mos-stat-total');
  const compEl = $('mos-stat-completed');
  const procEl = $('mos-stat-processing');
  totalEl.textContent = total;
  compEl.textContent = completed;
  procEl.textContent = processing;

  // 触发脉冲动画
  changed.forEach(id => {
    const el = $(id);
    if (!el) return;
    el.classList.remove('stat-updated');
    // 强制回流使动画可重复触发
    void el.offsetWidth;
    el.classList.add('stat-updated');
    // 动画结束后移除类
    setTimeout(() => el.classList.remove('stat-updated'), 700);
  });

  // 更新"最后更新"时间
  const now = new Date();
  const timeStr = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}:${String(now.getSeconds()).padStart(2,'0')}`;
  // 为每个stat-card添加更新时间
  const statCards = qsa('#mos-stat-cards .stat-card');
  statCards.forEach(card => {
    let updatedEl = card.querySelector('.stat-updated-at');
    if (!updatedEl) {
      updatedEl = document.createElement('div');
      updatedEl.className = 'stat-updated-at';
      card.appendChild(updatedEl);
    }
    updatedEl.textContent = `更新于 ${timeStr}`;
  });
}

/** 获取MOS结果音频试听URL */
function getMosAudioUrl(taskId, filename) {
  const token = getToken();
  return `/api/mos/audio/${taskId}/${encodeURIComponent(filename)}?token=${encodeURIComponent(token || '')}`;
}

/** 按文件名中的数字自然排序 */
function sortByFileNameNumeric(arr, key = '文件名') {
  return [...arr].sort((a, b) => {
    const nameA = String(a[key] || a['file'] || '');
    const nameB = String(b[key] || b['file'] || '');
    const numA = parseInt(nameA.match(/\d+/)?.[0] || '0', 10);
    const numB = parseInt(nameB.match(/\d+/)?.[0] || '0', 10);
    if (numA !== numB) return numA - numB;
    return nameA.localeCompare(nameB);
  });
}

async function showMosResult(taskId) {
  const modal = new bootstrap.Modal($('mos-result-modal'));
  const body = $('mos-result-body');
  body.innerHTML = '<div class="text-center py-4"><div class="spinner-border" role="status"></div></div>';
  modal.show();
  try {
    const data = await api(apiUrl(`/api/mos/results/${taskId}`));
    const results = data.results || (Array.isArray(data) ? data : [data]);
    if (!results || results.length === 0) {
      body.innerHTML = '<div class="text-center text-muted py-4">暂无结果数据</div>';
      return;
    }

    // 按文件名数字排序
    const sorted = sortByFileNameNumeric(results);
    const cols = data.columns || Object.keys(results[0]);

    // 构建横表：每行一个文件，每列一个指标
    const filenameKey = cols.includes('文件名') ? '文件名' : 'file';
    const metricCols = cols.filter(c => c !== filenameKey);

    let html = `<div class="mb-2"><strong>任务:</strong> ${shortId(taskId)} | <strong>文件数:</strong> ${sorted.length}</div>`;
    html += '<div class="table-responsive"><table class="table table-striped table-hover result-table" style="font-size:0.85rem">';
    html += '<thead><tr><th>#</th><th>文件名</th>';
    metricCols.forEach(c => { html += `<th>${c}</th>`; });
    html += '<th style="width:180px">试听</th></tr></thead><tbody>';

    sorted.forEach((r, i) => {
      const fname = r[filenameKey] || r['file'] || '';
      const audioUrl = getMosAudioUrl(taskId, fname);
      html += `<tr><td>${i + 1}</td><td><strong>${fname}</strong></td>`;
      metricCols.forEach(c => {
        const v = r[c];
        html += `<td>${typeof v === 'number' ? v.toFixed(4) : (v ?? '-')}</td>`;
      });
      html += `<td>
        <audio controls style="height:30px;width:160px" preload="none">
          <source src="${audioUrl}">
        </audio>
      </td></tr>`;
    });

    html += '</tbody></table></div>';
    body.innerHTML = html;
  } catch (err) {
    body.innerHTML = `<div class="alert alert-danger">加载结果失败: ${err.message}</div>`;
  }
}

function drawMosChart(results) {
  const container = $('mos-result-body');
  if (!results || results.length === 0) return;
  // 为第一个文件构建指标图表
  const r = results[0];
  const scores = r.scores || r.metrics || r;
  const entries = Object.entries(scores).filter(([k, v]) => v !== null && v !== undefined && k !== 'file_name' && k !== 'file' && typeof v === 'number');
  if (entries.length < 2) return;
  const canvas = document.createElement('canvas');
  canvas.id = 'mos-chart-canvas';
  canvas.style.maxHeight = '200px';
  canvas.style.marginTop = '1rem';
  container.appendChild(canvas);
  try {
    new Chart(canvas, {
      type: 'bar',
      data: {
        labels: entries.map(([k]) => k.toUpperCase()),
        datasets: [{
          label: 'MOS 评分',
          data: entries.map(([, v]) => v),
          backgroundColor: ['#667eea', '#764ba2', '#52c41a', '#faad14', '#ff4d4f', '#1890ff', '#722ed1', '#13c2c2', '#eb2f96'],
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, max: Math.max(5, ...entries.map(([, v]) => v) + 1) } }
      }
    });
  } catch (_) {}
}

async function downloadMosResult(taskId) {
  try {
    const resp = await api(apiUrl(`/api/mos/download/${taskId}`), { method: 'GET', raw: true });
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mos_result_${shortId(taskId)}.xlsx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast('下载失败: ' + err.message, 'error');
  }
}

async function deleteMosTask(taskId) {
  showConfirm('确定要删除此任务吗？', async () => {
    try {
      await api(apiUrl(`/api/mos/tasks/${taskId}`), { method: 'DELETE' });
      showToast('删除成功', 'success');
      loadMosTasks();
    } catch (err) {
      showToast('删除失败: ' + err.message, 'error');
    }
  });
}

// ==================== 音频修复 ====================
let restorationAlgs = [];
let restorationSelectedAlg = null;
let restorationFile = null;

async function loadRestorationAlgorithms() {
  try {
    const data = await api(apiUrl('/api/restoration/algorithms'));
    restorationAlgs = data.algorithms || data || [];
    renderRestorationAlgorithms(restorationAlgs);
  } catch (_) {}
}

function renderRestorationAlgorithms(algs) {
  const container = $('restoration-algorithms');
  if (!algs || algs.length === 0) {
    container.innerHTML = '<div class="text-muted">暂无可用算法</div>';
    return;
  }
  // 默认选择第一个
  if (!restorationSelectedAlg && algs.length > 0) restorationSelectedAlg = algs[0].name;
  container.innerHTML = algs.map(a => {
    const type = a.type || '深度学习';
    const typeClass = type.includes('传统') ? 'bg-secondary' : 'bg-info';
    const selected = restorationSelectedAlg === a.name ? 'selected' : '';
    return `<div class="col-md-4 col-sm-6">
      <div class="algorithm-card ${selected}" data-alg="${a.name}" onclick="selectRestorationAlgorithm('${a.name}')">
        <div class="alg-name">${a.display_name || a.name}</div>
        <span class="alg-type ${typeClass} text-white">${type}</span>
        <div class="alg-desc mt-1">${a.description || ''}</div>
        ${a.advantages ? `<small class="text-success d-block mt-1"><i class="bi bi-check-circle"></i> ${a.advantages}</small>` : ''}
      </div>
    </div>`;
  }).join('');
  updateRestorationSubmitBtn();
}

function selectRestorationAlgorithm(name) {
  restorationSelectedAlg = name;
  qsa('#restoration-algorithms .algorithm-card').forEach(c => {
    c.classList.toggle('selected', c.dataset.alg === name);
  });
  updateRestorationSubmitBtn();
}

function initRestorationPage() {
  const zone = $('restoration-upload-zone');
  const input = $('restoration-file-input');

  // 拖拽上传
  if (zone) {
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        restorationFile = files[0];
        $('restoration-file-info').classList.remove('d-none');
        $('restoration-file-name').textContent = restorationFile.name;
        $('restoration-file-size').textContent = formatSize(restorationFile.size);
        updateRestorationSubmitBtn();
      }
    });
  }

  input.addEventListener('change', () => {
    const files = input.files;
    restorationFile = files.length > 0 ? files[0] : null;
    if (restorationFile) {
      $('restoration-file-info').classList.remove('d-none');
      $('restoration-file-name').textContent = restorationFile.name;
      $('restoration-file-size').textContent = formatSize(restorationFile.size);
    } else {
      $('restoration-file-info').classList.add('d-none');
    }
    updateRestorationSubmitBtn();
  });
  $('restoration-submit-btn').addEventListener('click', submitRestorationTask);
}

function updateRestorationSubmitBtn() {
  const btn = $('restoration-submit-btn');
  btn.disabled = !restorationSelectedAlg || !restorationFile;
}

function showRestorationAlert(type, msg) {
  const errorEl = $('restoration-error');
  const successEl = $('restoration-success');
  if (errorEl) { errorEl.classList.add('d-none'); errorEl.textContent = ''; }
  if (successEl) { successEl.classList.add('d-none'); successEl.textContent = ''; }
  const target = type === 'success' ? successEl : errorEl;
  if (target) {
    target.textContent = msg;
    target.classList.remove('d-none');
    setTimeout(() => { target.classList.add('d-none'); target.textContent = ''; }, 4000);
  }
}

async function submitRestorationTask() {
  const btn = $('restoration-submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-border spinner-border-sm"></div> 处理中...';
  try {
    // 上传
    const fd = new FormData();
    fd.append('file', restorationFile);
    fd.append('algorithm', restorationSelectedAlg);
    const uploadData = await api(apiUrl('/api/restoration/upload'), { method: 'POST', formData: true, body: fd });
    const taskId = uploadData.task_id || uploadData.task_ids?.[0];
    if (taskId) {
      await api(apiUrl(`/api/restoration/process/${taskId}`), { method: 'POST' });
      showToast('修复任务已提交', 'success');
    }
    loadRestorationTasks();
    $('restoration-tab-tasks').click();
  } catch (err) {
    showRestorationAlert('error', '提交失败: ' + err.message);
    showToast('提交失败: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> 开始修复';
    // 清空文件选择，允许立即选择新文件
    $('restoration-file-input').value = '';
    restorationFile = null;
    $('restoration-file-info').classList.add('d-none');
    updateRestorationSubmitBtn();
  }
}

async function loadRestorationTasks() {
  try {
    const data = await api(apiUrl('/api/restoration/tasks'));
    const tasks = data.tasks || data || [];
    renderRestorationTasks(tasks);
  } catch (_) {}
}

function _isRestorationActive(t) {
  const s = t.status;
  return s === 'processing' || s === 'pending' || s === 'queued';
}

function _restorationActionsHtml(t) {
  const status = t.status || 'pending';
  return `${statusBadge(status)}
    ${status === 'completed' ? `
      <button class="btn btn-sm btn-outline-info" onclick="toggleRestorationCompare('${t.task_id}', this)"><i class="bi bi-play-circle"></i> 试听对比</button>
      <button class="btn btn-sm btn-outline-success" onclick="downloadRestorationResult('${t.task_id}')"><i class="bi bi-download"></i></button>
    ` : ''}
    <button class="btn btn-sm btn-outline-danger" onclick="deleteRestorationTask('${t.task_id}')"><i class="bi bi-trash"></i></button>`;
}

function _buildRestorationTaskHtml(t) {
  const status = t.status || 'pending';
  const progress = t.progress || 0;
  const fileName = t.filename || t.file_name || shortId(t.task_id);
  const algName = t.algorithm || '';
  const isActive = _isRestorationActive(t);
  return `<div class="task-item" id="restoration-task-${t.task_id}">
    <div class="task-header">
      <div>
        <span class="task-file">${_escapeHtml(fileName)}</span>
        ${algName ? `<span class="badge bg-info ms-2">${_escapeHtml(algName)}</span>` : ''}
        <span class="task-id ms-2">${shortId(t.task_id)}</span>
      </div>
      <div class="task-actions">
        ${_restorationActionsHtml(t)}
      </div>
    </div>
    <div class="restoration-progress-wrap mt-2" ${isActive ? '' : 'style="display:none"'}>
      <div class="progress" style="height:6px"><div class="progress-bar progress-bar-striped progress-bar-animated" style="width:${progress}%"></div></div>
      <div id="restoration-progress-${shortId(t.task_id)}" class="progress-detail"></div>
    </div>
    <div class="task-detail">创建: ${formatDate(t.created_at || t.create_time)}${t.duration ? ` | 耗时: ${t.duration.toFixed(1)}s` : ''}</div>
    <div class="restoration-compare-${t.task_id} d-none mt-2"></div>
  </div>`;
}

function _updateRestorationTaskElement(el, t) {
  const isActive = _isRestorationActive(t);
  // 重建 actions（状态徽章 + 按钮，处理 active→completed 转换）
  const actionsEl = el.querySelector('.task-actions');
  if (actionsEl) actionsEl.innerHTML = _restorationActionsHtml(t);
  // 进度区显隐 + 进度条宽度
  const wrap = el.querySelector('.restoration-progress-wrap');
  if (wrap) {
    wrap.style.display = isActive ? '' : 'none';
    if (isActive) {
      const bar = wrap.querySelector('.progress-bar');
      if (bar) bar.style.width = (t.progress || 0) + '%';
    }
  }
}

function renderRestorationTasks(tasks) {
  $('restoration-task-count').textContent = tasks ? tasks.length : 0;
  const container = $('restoration-task-list');
  if (!tasks || tasks.length === 0) {
    // 清空时断开所有 WS
    Object.keys(state.restorationWsConnections).forEach(id => disconnectRestorationWs(id));
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无任务 <button class="btn btn-sm btn-primary ms-2" onclick="document.getElementById(\'restoration-tab-upload\').click()">创建第一个修复任务</button></div>';
    return;
  }

  // 判断是否需要全量渲染（首次或所有卡片都被移除）
  const needsFullRender = !container.querySelector('.task-item');
  if (needsFullRender) {
    container.innerHTML = tasks.map(t => _buildRestorationTaskHtml(t)).join('');
    tasks.forEach(t => { if (_isRestorationActive(t)) connectRestorationWs(t.task_id); });
    return;
  }

  // 增量更新（避免全量 innerHTML 替换破坏 WS 实时更新的 DOM 与已展开的对比面板）
  const existingIds = new Set();
  tasks.forEach(t => {
    existingIds.add(t.task_id);
    const el = $(`restoration-task-${t.task_id}`);
    if (el) {
      _updateRestorationTaskElement(el, t);
    } else {
      // 新任务 → 插入到列表顶部
      container.insertAdjacentHTML('afterbegin', _buildRestorationTaskHtml(t));
      if (_isRestorationActive(t)) connectRestorationWs(t.task_id);
    }
  });

  // 移除已删除的任务（本地删除或别的客户端删除）
  container.querySelectorAll('.task-item[id^="restoration-task-"]').forEach(el => {
    const id = el.id.replace('restoration-task-', '');
    if (!existingIds.has(id)) {
      disconnectRestorationWs(id);
      el.remove();
    }
  });
}

/** WebSocket推送的处理进度步骤 — 根据后端报告的步骤名确定完成状态 */
function renderRestorationProgressSteps(taskId, progress, message) {
  const container = $(`restoration-progress-${shortId(taskId)}`);
  if (!container) return;
  const stepMatch = message ? message.match(/^\[(\w+)\](.+)/) : null;
  if (!stepMatch) {
    container.innerHTML = `<div class="progress-step active"><span class="step-icon"><div class="spinner-border spinner-sm" role="status"></div></span><span>${_escapeHtml(message || '处理中...')}</span></div>`;
    return;
  }
  const currentStep = stepMatch[1].toLowerCase();
  const stepDesc = stepMatch[2].trim();
  const order = state.restorationStepOrder;
  const currentIdx = order.indexOf(currentStep);
  const hasCurrent = currentIdx >= 0;
  let html = '';
  order.forEach(step => {
    const stepLabel = state.restorationStepNames[step] || step;
    const idx = order.indexOf(step);
    const isCompleted = hasCurrent && idx < currentIdx;
    const isActive = step === currentStep;
    let iconHtml;
    if (isActive) iconHtml = '<div class="spinner-border spinner-sm" role="status"></div>';
    else if (isCompleted) iconHtml = '<i class="bi bi-check-circle-fill text-success" style="font-size:.85rem"></i>';
    else iconHtml = '<i class="bi bi-circle" style="font-size:.85rem;color:#ddd"></i>';
    html += `<div class="progress-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}">
      <span class="step-icon">${iconHtml}</span>
      <span>${stepLabel}</span>
      ${isActive ? `<span class="ms-1 small text-muted">— ${_escapeHtml(stepDesc)}</span>` : ''}
    </div>`;
  });
  container.innerHTML = html;
}

/** 连接到WebSocket获取实时进度 */
function connectRestorationWs(taskId) {
  if (state.restorationWsConnections[taskId]) return;
  try {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/restoration/ws/${taskId}`;
    const ws = new WebSocket(wsUrl);
    state.restorationWsConnections[taskId] = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.status && data.progress !== undefined) {
          const el = $(`restoration-task-${taskId}`);
          if (el) {
            const bar = el.querySelector('.progress-bar');
            if (bar) bar.style.width = data.progress + '%';
            renderRestorationProgressSteps(taskId, data.progress, data.message || '');
          }
          if (data.status === 'completed') {
            disconnectRestorationWs(taskId);
            // 拉取终态（result_file 等仅在后端 tasks dict 中）
            loadRestorationTasks();
          } else if (data.status === 'failed') {
            disconnectRestorationWs(taskId);
            showRestorationAlert('error', data.message || '修复失败');
            loadRestorationTasks();
          }
        }
      } catch (_) {}
    };

    ws.onerror = () => { disconnectRestorationWs(taskId); };
    ws.onclose = () => { disconnectRestorationWs(taskId); };

    // 5分钟超时自动断开（防止残留连接）
    setTimeout(() => {
      if (state.restorationWsConnections[taskId]) {
        disconnectRestorationWs(taskId);
      }
    }, 300000);
  } catch (_) {
    // WebSocket不可用时静默失败
  }
}

function disconnectRestorationWs(taskId) {
  if (state.restorationWsConnections[taskId]) {
    try { state.restorationWsConnections[taskId].close(); } catch (_) {}
    delete state.restorationWsConnections[taskId];
  }
}

// WaveSurfer 实例缓存（避免重复创建）
const _wavesurferInstances = {};

async function toggleRestorationCompare(taskId, btn) {
  const container = qs(`.restoration-compare-${taskId}`);
  if (!container) return;
  if (!container.classList.contains('d-none')) {
    container.classList.add('d-none');
    if (btn) btn.innerHTML = '<i class="bi bi-play-circle"></i> 试听对比';
    // 暂停所有相关波形播放
    const origKey = `orig-${taskId}`;
    const procKey = `proc-${taskId}`;
    if (_wavesurferInstances[origKey]) _wavesurferInstances[origKey].pause();
    if (_wavesurferInstances[procKey]) _wavesurferInstances[procKey].pause();
    return;
  }
  container.classList.remove('d-none');
  if (btn) btn.innerHTML = '<i class="bi bi-pause-circle"></i> 收起对比';
  if (!container.dataset.loaded) {
    container.dataset.loaded = 'true';
    const token = getToken();
    const srcUrl = apiUrl(`/api/restoration/source/${taskId}?token=${token}`);
    const resultUrl = apiUrl(`/api/restoration/download/${taskId}?token=${token}`);
    const origWaveId = `wave-original-${shortId(taskId)}`;
    const procWaveId = `wave-processed-${shortId(taskId)}`;
    
    container.innerHTML = `<div class="audio-compare">
      <div class="audio-side">
        <div class="audio-side-header original"><i class="bi bi-soundwave"></i> 原始音频</div>
        <div class="waveform-controls">
          <button class="waveform-play-btn original-btn" data-wave-id="${origWaveId}" onclick="toggleWaveformPlay('${taskId}', 'orig')">
            <i class="bi bi-play-fill"></i>
          </button>
          <span class="waveform-time" id="time-${origWaveId}">00:00 / 00:00</span>
        </div>
        <div id="${origWaveId}" class="waveform-container"></div>
      </div>
      <div class="arrow-indicator"><i class="bi bi-arrow-right"></i></div>
      <div class="audio-side">
        <div class="audio-side-header processed"><i class="bi bi-soundwave"></i> 修复后音频</div>
        <div class="waveform-controls">
          <button class="waveform-play-btn processed-btn" data-wave-id="${procWaveId}" onclick="toggleWaveformPlay('${taskId}', 'proc')">
            <i class="bi bi-play-fill"></i>
          </button>
          <span class="waveform-time" id="time-${procWaveId}">00:00 / 00:00</span>
        </div>
        <div id="${procWaveId}" class="waveform-container"></div>
      </div>
    </div>`;
    
    // 创建 WaveSurfer 实例
    createWaveSurfer(origWaveId, srcUrl, '#fa8c16', '#d46b08', taskId, 'orig');
    createWaveSurfer(procWaveId, resultUrl, '#73d13d', '#389e0d', taskId, 'proc');
  }
}

/**
 * 切换波形播放/暂停
 */
function toggleWaveformPlay(taskId, type) {
  const key = `${type}-${taskId}`;
  const ws = _wavesurferInstances[key];
  if (!ws) return;
  
  if (ws.isPlaying()) {
    ws.pause();
  } else {
    // 播放前暂停另一侧
    const otherKey = type === 'orig' ? `proc-${taskId}` : `orig-${taskId}`;
    if (_wavesurferInstances[otherKey]) {
      _wavesurferInstances[otherKey].pause();
    }
    ws.play();
  }
}

/**
 * 格式化时间 mm:ss
 */
function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

/**
 * 创建 WaveSurfer 波形控件
 * @param {string} containerId - 容器元素 ID
 * @param {string} audioUrl - 音频 URL
 * @param {string} waveColor - 波形颜色（未播放部分）
 * @param {string} progressColor - 进度颜色（已播放部分）
 * @param {string} taskId - 任务 ID
 * @param {string} type - 类型标识 ('orig' 或 'proc')
 */
function createWaveSurfer(containerId, audioUrl, waveColor, progressColor, taskId, type) {
  const container = document.getElementById(containerId);
  if (!container || typeof WaveSurfer === 'undefined') {
    // WaveSurfer 未加载，回退到简单提示
    if (container) container.innerHTML = '<div class="waveform-fallback"><i class="bi bi-hourglass-split"></i> 波形加载中...</div>';
    return;
  }
  
  const key = `${type}-${taskId}`;
  const timeEl = document.getElementById(`time-${containerId}`);
  const playBtn = document.querySelector(`button[data-wave-id="${containerId}"]`);
  
  // 清理旧实例（如有）
  if (_wavesurferInstances[key]) {
    _wavesurferInstances[key].destroy();
    delete _wavesurferInstances[key];
  }
  
  // 创建新实例
  const wavesurfer = WaveSurfer.create({
    container: `#${containerId}`,
    waveColor: waveColor,
    progressColor: progressColor,
    cursorColor: progressColor,
    cursorWidth: 2,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    height: 64,
    normalize: true,
    hideScrollbar: true,
    fillParent: true,
    responsive: true,
    backend: 'WebAudio',
    url: audioUrl,
  });
  
  _wavesurferInstances[key] = wavesurfer;
  
  // 播放/暂停事件 - 更新按钮图标
  wavesurfer.on('play', () => {
    if (playBtn) playBtn.innerHTML = '<i class="bi bi-pause-fill"></i>';
    // 同步暂停另一侧波形（避免同时播放）
    const otherKey = type === 'orig' ? `proc-${taskId}` : `orig-${taskId}`;
    if (_wavesurferInstances[otherKey]) {
      _wavesurferInstances[otherKey].pause();
      const otherBtn = document.querySelector(`button[data-wave-id="${otherKey.replace(type === 'orig' ? 'proc' : 'orig', type)}"]`);
      // 更新另一侧按钮图标
      const otherContainerId = type === 'orig' 
        ? `wave-processed-${shortId(taskId)}` 
        : `wave-original-${shortId(taskId)}`;
      const otherPlayBtn = document.querySelector(`button[data-wave-id="${otherContainerId}"]`);
      if (otherPlayBtn) otherPlayBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
    }
  });
  
  wavesurfer.on('pause', () => {
    if (playBtn) playBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
  });
  
  wavesurfer.on('finish', () => {
    if (playBtn) playBtn.innerHTML = '<i class="bi bi-play-fill"></i>';
  });
  
  // 时间更新
  wavesurfer.on('ready', () => {
    container.classList.add('waveform-ready');
    const duration = wavesurfer.getDuration();
    if (timeEl) timeEl.textContent = `00:00 / ${formatTime(duration)}`;
  });
  
  wavesurfer.on('audioprocess', (currentTime) => {
    const duration = wavesurfer.getDuration();
    if (timeEl) timeEl.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
  });
  
  wavesurfer.on('seeking', (currentTime) => {
    const duration = wavesurfer.getDuration();
    if (timeEl) timeEl.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
  });
  
  // 加载错误处理
  wavesurfer.on('error', (err) => {
    console.warn(`WaveSurfer 加载失败 (${type}):`, err);
    container.innerHTML = `<div class="waveform-error">
      <i class="bi bi-exclamation-triangle"></i> 波形加载失败
      <button class="btn btn-sm btn-outline-secondary ms-2" onclick="retryWaveform('${containerId}','${audioUrl}','${waveColor}','${progressColor}','${taskId}','${type}')">
        <i class="bi bi-arrow-clockwise"></i> 重试
      </button>
    </div>`;
  });
}

/**
 * 重试加载波形
 */
function retryWaveform(containerId, audioUrl, waveColor, progressColor, taskId, type) {
  const container = document.getElementById(containerId);
  if (container) container.innerHTML = '<div class="waveform-fallback"><i class="bi bi-hourglass-split"></i> 加载中...</div>';
  createWaveSurfer(containerId, audioUrl, waveColor, progressColor, taskId, type);
}

async function downloadRestorationResult(taskId) {
  try {
    const token = getToken();
    const resp = await fetch(apiUrl(`/api/restoration/download/${taskId}?token=${token}`));
    if (!resp.ok) throw new Error('下载失败');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `修复结果_${shortId(taskId)}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast('下载失败: ' + err.message, 'error');
  }
}

async function deleteRestorationTask(taskId) {
  showConfirm('确定要删除此任务吗？', async () => {
    try {
      disconnectRestorationWs(taskId);
      await api(apiUrl(`/api/restoration/tasks/${taskId}`), { method: 'DELETE' });
      showToast('删除成功', 'success');
      loadRestorationTasks();
    } catch (err) {
      showRestorationAlert('error', '删除失败: ' + err.message);
      showToast('删除失败: ' + err.message, 'error');
    }
  });
}

// ==================== 音频播放器 ====================
// 使用全局 audio 元素
const audioEl = $('audio-player');
let activeAudioControls = null;

function toggleAudioPlay(btn) {
  const controls = btn.closest('.audio-controls');
  const url = controls.dataset.url;
  if (!url) return;
  if (activeAudioControls === controls && !audioEl.paused) {
    audioEl.pause();
    setPlayBtnIcon(btn, false);
    return;
  }
  // 停止之前的
  if (activeAudioControls && activeAudioControls !== controls) {
    const prevBtn = activeAudioControls.querySelector('.play-btn');
    if (prevBtn) setPlayBtnIcon(prevBtn, false);
  }
  activeAudioControls = controls;
  audioEl.src = url;
  audioEl.play().then(() => {
    setPlayBtnIcon(btn, true);
  }).catch(() => {});
  // 更新时间
  audioEl.ontimeupdate = () => {
    const pct = audioEl.currentTime / (audioEl.duration || 1) * 100;
    const fill = controls.querySelector('.progress-fill');
    if (fill) fill.style.width = pct + '%';
    const displays = controls.querySelectorAll('.time-display');
    if (displays[0]) displays[0].textContent = formatTime(audioEl.currentTime);
    if (displays[1]) displays[1].textContent = formatTime(audioEl.duration);
  };
  audioEl.onended = () => {
    setPlayBtnIcon(btn, false);
    activeAudioControls = null;
  };
}

function setPlayBtnIcon(btn, playing) {
  if (!btn) return;
  btn.innerHTML = playing ? '<i class="bi bi-pause-fill"></i>' : '<i class="bi bi-play-fill"></i>';
}

function seekAudio(bar, e) {
  if (!audioEl.src) return;
  const rect = bar.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audioEl.currentTime = pct * audioEl.duration;
}

// 波形绘制共用一个 AudioContext（浏览器限制约 6 个，避免泄漏）
let _waveformAudioCtx = null;

async function drawWaveform(canvasId, url, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const blob = await resp.blob();
    if (!_waveformAudioCtx) {
      _waveformAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const buf = await _waveformAudioCtx.decodeAudioData(await blob.arrayBuffer());
    const data = buf.getChannelData(0);
    const w = canvas.clientWidth || canvas.width || 300;
    const h = canvas.clientHeight || canvas.height || 60;
    canvas.width = w * 2; canvas.height = h * 2;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const cctx = canvas.getContext('2d');
    cctx.scale(2, 2);
    cctx.clearRect(0, 0, w, h);
    // 简化的波形绘制
    const step = Math.ceil(data.length / w);
    cctx.beginPath();
    cctx.strokeStyle = color;
    cctx.lineWidth = 1.5;
    const mid = h / 2;
    for (let i = 0; i < w; i++) {
      let max = 0;
      for (let j = 0; j < step && i * step + j < data.length; j++) {
        const abs = Math.abs(data[i * step + j]);
        if (abs > max) max = abs;
      }
      cctx.moveTo(i, mid);
      cctx.lineTo(i, mid - max * mid * 0.8);
    }
    cctx.stroke();
    cctx.beginPath();
    for (let i = 0; i < w; i++) {
      let max = 0;
      for (let j = 0; j < step && i * step + j < data.length; j++) {
        const abs = Math.abs(data[i * step + j]);
        if (abs > max) max = abs;
      }
      cctx.moveTo(i, mid);
      cctx.lineTo(i, mid + max * mid * 0.8);
    }
    cctx.stroke();
  } catch (_) {}
}

// ==================== 参考音频 ====================
async function loadRefAudioList() {
  try {
    const data = await api(apiUrl('/api/reference-audio/list'));
    const items = data.items || data.audio_list || (Array.isArray(data) ? data : []);
    renderRefAudioList(items);
    updateRefStats(items);
  } catch (_) {}
}

function renderRefAudioList(items) {
  const container = $('ref-audio-list');
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无参考音频</div>';
    return;
  }
  container.innerHTML = items.map(item => {
    const id = item.id || item.audio_id;
    const fname = item.filename || item.file_name || item.original_name || '未知';
    const fsize = item.file_size || item.size || 0;
    return `<div class="ref-audio-item">
      <div class="ref-audio-info">
        <div class="ref-audio-name">${fname}</div>
        <div class="ref-audio-meta">${item.duration ? item.duration.toFixed(1) + 's' : ''} ${item.sample_rate ? '| ' + item.sample_rate + 'Hz' : ''} ${fsize ? '| ' + formatSize(fsize) : ''}${item.description ? ' | ' + item.description : ''}</div>
      </div>
      <div class="task-actions">
        <button class="btn btn-sm btn-outline-primary" onclick="playRefAudio('${id}')"><i class="bi bi-play-circle"></i></button>
        <button class="btn btn-sm btn-outline-info" onclick="editRefAudio('${id}')"><i class="bi bi-pencil"></i></button>
        <button class="btn btn-sm btn-outline-success" onclick="downloadRefAudio('${id}')"><i class="bi bi-download"></i></button>
        <button class="btn btn-sm btn-outline-danger" onclick="deleteRefAudio('${id}')"><i class="bi bi-trash"></i></button>
      </div>
    </div>`;
  }).join('');
}

function updateRefStats(items) {
  $('ref-stat-count').textContent = items ? items.length : 0;
  if (items && items.length > 0) {
    const totalSize = items.reduce((sum, i) => sum + (i.file_size || i.size || 0), 0);
    $('ref-stat-size').textContent = formatSize(totalSize);
  }
}

function initReferencePage() {
  $('ref-file-input').addEventListener('change', () => {
    const files = $('ref-file-input').files;
    if (files.length > 0) {
      $('ref-upload-info').classList.remove('d-none');
      $('ref-upload-count').textContent = `已选择 ${files.length} 个文件`;
    } else {
      $('ref-upload-info').classList.add('d-none');
    }
  });
  $('ref-upload-btn').addEventListener('click', uploadRefAudio);

  // 编辑保存
  $('ref-edit-save-btn').addEventListener('click', saveRefAudioEdit);
}

async function uploadRefAudio() {
  const files = $('ref-file-input').files;
  if (files.length === 0) return;
  const btn = $('ref-upload-btn');
  const spinner = $('ref-upload-spinner');
  btn.disabled = true;
  spinner.classList.remove('d-none');
  try {
    if (files.length === 1) {
      const fd = new FormData();
      fd.append('file', files[0]);
      await api(apiUrl('/api/reference-audio/upload'), { method: 'POST', formData: true, body: fd });
    } else {
      const fd = new FormData();
      for (const f of files) fd.append('files', f);
      await api(apiUrl('/api/reference-audio/upload-batch'), { method: 'POST', formData: true, body: fd });
    }
    showToast('上传成功', 'success');
    $('ref-file-input').value = '';
    $('ref-upload-info').classList.add('d-none');
    loadRefAudioList();
  } catch (err) {
    showToast('上传失败: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    spinner.classList.add('d-none');
  }
}

async function deleteRefAudio(id) {
  showConfirm('确定要删除此参考音频吗？', async () => {
    try {
      await api(apiUrl(`/api/reference-audio/delete/${id}`), { method: 'DELETE' });
      showToast('删除成功', 'success');
      loadRefAudioList();
    } catch (err) {
      showToast('删除失败: ' + err.message, 'error');
    }
  });
}

async function editRefAudio(id) {
  try {
    const data = await api(apiUrl(`/api/reference-audio/detail/${id}`));
    const item = data.audio || data || {};
    const audioId = item.audio_id || item.id || id;
    $('ref-edit-modal').dataset.audioId = audioId;
    $('ref-edit-filename').value = item.file_name || item.filename || '';
    $('ref-edit-duration').value = item.duration ? item.duration.toFixed(2) + 's' : '';
    $('ref-edit-samplerate').value = item.sample_rate ? item.sample_rate + ' Hz' : '';
    $('ref-edit-description').value = item.description || '';
    $('ref-edit-groundtruth').value = item.ground_truth_text || '';
    new bootstrap.Modal($('ref-edit-modal')).show();
  } catch (err) {
    showToast('加载详情失败: ' + err.message, 'error');
  }
}

async function saveRefAudioEdit() {
  const audioId = $('ref-edit-modal').dataset.audioId;
  if (!audioId) return;
  try {
    await api(apiUrl(`/api/reference-audio/update/${audioId}`), {
      method: 'PUT',
      body: {
        description: $('ref-edit-description').value,
        ground_truth_text: $('ref-edit-groundtruth').value,
      }
    });
    showToast('保存成功', 'success');
    bootstrap.Modal.getInstance($('ref-edit-modal')).hide();
    loadRefAudioList();
  } catch (err) {
    showToast('保存失败: ' + err.message, 'error');
  }
}

function playRefAudio(id) {
  const token = getToken();
  const url = apiUrl(`/api/reference-audio/download/${id}?token=${encodeURIComponent(token || '')}`);
  if (audioEl.src !== url || audioEl.paused) {
    audioEl.src = url;
    audioEl.play().catch(() => {});
  } else {
    audioEl.pause();
  }
}

async function downloadRefAudio(id) {
  try {
    const token = getToken();
    const resp = await fetch(apiUrl(`/api/reference-audio/download/${id}?token=${token}`));
    if (!resp.ok) throw new Error('下载失败');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reference_${id.slice(0, 8)}.wav`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast('下载失败: ' + err.message, 'error');
  }
}

// ==================== 确认对话框 ====================
let confirmCallback = null;

function showConfirm(msg, cb) {
  confirmCallback = cb;
  $('confirm-modal-body').textContent = msg;
  $('confirm-modal-ok').onclick = () => {
    bootstrap.Modal.getInstance($('confirm-modal')).hide();
    if (confirmCallback) confirmCallback();
    confirmCallback = null;
  };
  new bootstrap.Modal($('confirm-modal')).show();
}

// ==================== 轮询 =====================
let _pollTimerId = null;
let _pollIntervalCurrent = 5000;

function startPolling() {
  if (_pollTimerId) return;
  schedulePoll();
}

function schedulePoll() {
  if (_pollTimerId) clearTimeout(_pollTimerId);
  _pollTimerId = setTimeout(doPoll, _pollIntervalCurrent);
}

async function doPoll() {
  const token = getToken();
  if (!token) {
    if ($('app-section') && !$('app-section').classList.contains('d-none')) {
      showApp(false);
      showToast('会话已过期，请重新登录', 'error');
    }
    schedulePoll();
    return;
  }

  // 检查是否有处理中的任务，动态调整轮询间隔
  const curTasks = await loadMosTasks();

  let hasActive = false;
  if (curTasks && curTasks.length > 0) {
    hasActive = curTasks.some(t =>
      t.status === 'processing' || t.status === 'pending' || t.status === 'queued'
    );
  }

  // 有活动任务时2秒轮询，无活动时5秒
  _pollIntervalCurrent = hasActive ? 2000 : 5000;

  loadRestorationTasks();

  schedulePoll();
}

// ==================== 初始化 ====================

// 立即执行：认证检查和页面切换（不依赖 Bootstrap / CDN 脚本）
// app.js 位于页面底部，此时上方 DOM 已全部就绪
// 不等 DOMContentLoaded — 那会阻塞在远程 CDN 脚本加载上（可能几秒）
checkAuth();

// 事件绑定 — DOM 已就绪
$('login-form').addEventListener('submit', handleLogin);
$('logout-btn').addEventListener('click', handleLogout);

// 初始化各页面
initMosPage();
initRestorationPage();
initReferencePage();
initAsrPage();

// 启动轮询
startPolling();

// ==================== ASR 语音识别评测 ====================
let asrAlgorithms = [];
let asrSelectedAlgorithm = '';
let asrFile = null;
let asrDatasets = [];
let asrBenchmarkAlgos = new Set();

async function loadAsrAlgorithms() {
  try {
    const data = await api(apiUrl('/api/asr/algorithms'));
    asrAlgorithms = data.algorithms || data || [];
    const sel = $('asr-algorithm-select');
    sel.innerHTML = '';
    asrAlgorithms.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.name;
      opt.textContent = `${a.display_name || a.name}${a.initialized ? ' ✅' : ''}`;
      sel.appendChild(opt);
    });
    if (asrAlgorithms.length > 0) {
      asrSelectedAlgorithm = asrAlgorithms[0].name;
      updateAsrAlgoInfo();
    }
    // 渲染 benchmark 算法多选
    renderAsrBenchmarkAlgos();
  } catch (e) {
    console.error('加载ASR算法列表失败:', e);
  }
}

function updateAsrAlgoInfo() {
  const info = $('asr-algorithm-info');
  const algo = asrAlgorithms.find(a => a.name === asrSelectedAlgorithm);
  if (algo) {
    info.innerHTML = `<strong>${algo.display_name || algo.name}</strong> | 架构: ${algo.architecture || '-'} | 参数: ${algo.params || '-'} | AISHELL-1 CER: ${algo.cer_aishell1 || '-'} | ${algo.streaming ? '支持流式' : '非流式'} | ${algo.license || '-'}`;
  } else {
    info.textContent = '';
  }
}

async function loadAsrDatasets() {
  try {
    const data = await api(apiUrl('/api/asr/datasets'));
    // 后端返回纯数组，兼容可能包裹在 {datasets: [...]} 中的情况
    asrDatasets = Array.isArray(data) ? data : (data.datasets || []);
    const sel = $('asr-dataset-select');
    sel.innerHTML = '';
    let firstAvailable = null;
    asrDatasets.forEach(ds => {
      const opt = document.createElement('option');
      // 使用注册 key 作为 value，确保与后端 lookup 一致
      opt.value = ds.key || ds.name;
      opt.textContent = `${ds.name}${ds.available ? '' : ' (不可用)'} - ${ds.description || ''}`;
      if (ds.available && !firstAvailable) firstAvailable = ds.key || ds.name;
      sel.appendChild(opt);
    });
    // 自动选中第一个可用的数据集（优先内置测试集）
    if (firstAvailable) {
      sel.value = firstAvailable;
    }
    // 更新按钮状态
    updateAsrBenchmarkBtn();
  } catch (e) {
    console.error('加载ASR数据集失败:', e);
  }
}

function renderAsrBenchmarkAlgos() {
  const container = $('asr-benchmark-algorithms');
  container.innerHTML = '';
  asrAlgorithms.forEach(a => {
    const col = document.createElement('div');
    col.className = 'col-md-4 col-lg-3';
    const checked = asrBenchmarkAlgos.has(a.name) ? 'checked' : '';
    col.innerHTML = `
      <div class="form-check">
        <input class="form-check-input asr-bench-algo" type="checkbox" value="${a.name}" id="asr-ba-${a.name}" ${checked}>
        <label class="form-check-label" for="asr-ba-${a.name}">
          ${a.display_name || a.name} <small class="text-muted">(${a.params || '?'})</small>
        </label>
      </div>`;
    container.appendChild(col);
  });
  // 绑定事件
  container.querySelectorAll('.asr-bench-algo').forEach(cb => {
    cb.addEventListener('change', () => {
      if (cb.checked) asrBenchmarkAlgos.add(cb.value);
      else asrBenchmarkAlgos.delete(cb.value);
      updateAsrBenchmarkBtn();
    });
  });
  // 渲染完成后刷新按钮状态（此时数据集可能已加载）
  updateAsrBenchmarkBtn();
}

function updateAsrBenchmarkBtn() {
  const dataset = $('asr-dataset-select').value;
  $('asr-benchmark-btn').disabled = asrBenchmarkAlgos.size === 0 || !dataset;
}

function initAsrPage() {
  // 算法选择
  $('asr-algorithm-select').addEventListener('change', e => {
    asrSelectedAlgorithm = e.target.value;
    updateAsrAlgoInfo();
  });

  // 数据集选择变更时更新按钮状态
  $('asr-dataset-select').addEventListener('change', updateAsrBenchmarkBtn);

  // 文件上传（input 已通过 CSS 透明覆盖 zone，用户直接点 input，禁止重复触发）
  const zone = $('asr-upload-zone');
  const input = $('asr-file-input');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) { asrFile = e.dataTransfer.files[0]; updateAsrFileInfo(); }
  });
  input.addEventListener('change', () => { if (input.files.length) { asrFile = input.files[0]; updateAsrFileInfo(); } });

  // 提交识别
  $('asr-submit-btn').addEventListener('click', submitAsrRecognition);

  // 提交benchmark
  $('asr-benchmark-btn').addEventListener('click', submitAsrBenchmark);
}

function updateAsrFileInfo() {
  if (!asrFile) return;
  $('asr-file-info').classList.remove('d-none');
  $('asr-file-name').textContent = asrFile.name;
  $('asr-file-size').textContent = formatSize(asrFile.size);
  $('asr-submit-btn').disabled = false;
}

async function submitAsrRecognition() {
  if (!asrFile || !asrSelectedAlgorithm) { showToast('请选择算法和音频文件', 'warning'); return; }
  const btn = $('asr-submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 识别中...';

  try {
    const fd = new FormData();
    fd.append('audio_file', asrFile);
    fd.append('algorithm', asrSelectedAlgorithm);
    fd.append('language', 'zh');

    const refText = $('asr-reference-text').value.trim();
    if (refText) fd.append('reference_text', refText);

    const data = await api(apiUrl('/api/asr/transcribe'), { method: 'POST', formData: true, body: fd });
    const taskId = data.task_id;

    // 轮询任务状态
    pollAsrTask(taskId);
  } catch (e) {
    showToast('识别失败: ' + e.message, 'error');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> <span>开始识别</span>';
  }
}

async function pollAsrTask(taskId) {
  const poll = async () => {
    try {
      const data = await api(apiUrl(`/api/asr/tasks/${taskId}`));
      if (data.status === 'completed') {
        showAsrResult(data);
        $('asr-submit-btn').disabled = false;
        $('asr-submit-btn').innerHTML = '<i class="bi bi-play-fill"></i> <span>开始识别</span>';
        return;
      } else if (data.status === 'failed') {
        showToast('识别失败: ' + (data.error || '未知错误'), 'error');
        $('asr-submit-btn').disabled = false;
        $('asr-submit-btn').innerHTML = '<i class="bi bi-play-fill"></i> <span>开始识别</span>';
        return;
      }
      setTimeout(poll, 2000);
    } catch (e) {
      setTimeout(poll, 3000);
    }
  };
  poll();
}

function showAsrResult(taskData) {
  const section = $('asr-result-section');
  const content = $('asr-result-content');
  section.classList.remove('d-none');

  const r = taskData.result || {};
  let html = `<div class="mb-3"><strong>识别文本:</strong> <span class="fs-5">${r.text || '-'}</span></div>`;
  html += `<div class="row g-2 mb-3">`;
  html += `<div class="col-md-3"><div class="stat-card"><div class="stat-value">${r.rtf?.toFixed(3) || '-'}</div><div class="stat-label">RTF</div></div></div>`;
  html += `<div class="col-md-3"><div class="stat-card"><div class="stat-value">${r.processing_time?.toFixed(2) || '-'}s</div><div class="stat-label">耗时</div></div></div>`;
  html += `<div class="col-md-3"><div class="stat-card"><div class="stat-value">${r.confidence ? (r.confidence * 100).toFixed(1) + '%' : '-'}</div><div class="stat-label">置信度</div></div></div>`;
  html += `<div class="col-md-3"><div class="stat-card"><div class="stat-value">${r.language || '-'}</div><div class="stat-label">语言</div></div></div>`;
  html += `</div>`;

  if (taskData.cer !== undefined && taskData.cer !== null) {
    html += `<div class="alert alert-info">CER (字错误率): <strong>${(taskData.cer * 100).toFixed(2)}%</strong></div>`;
  }

  if (r.segments && r.segments.length > 0) {
    html += `<h6>分段结果</h6><div class="table-responsive"><table class="table table-sm table-striped"><thead><tr><th>起始</th><th>结束</th><th>文本</th></tr></thead><tbody>`;
    r.segments.forEach(s => {
      html += `<tr><td>${s.start?.toFixed(2)}s</td><td>${s.end?.toFixed(2)}s</td><td>${s.text}</td></tr>`;
    });
    html += `</tbody></table></div>`;
  }

  content.innerHTML = html;
}

async function submitAsrBenchmark() {
  if (asrBenchmarkAlgos.size === 0) { showToast('请至少选择一个算法', 'warning'); return; }
  const dataset = $('asr-dataset-select').value;
  const maxSamples = parseInt($('asr-max-samples').value) || 100;

  const btn = $('asr-benchmark-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 评测中...';

  $('asr-benchmark-progress').classList.remove('d-none');
  $('asr-benchmark-result').classList.add('d-none');

  try {
    const data = await api(apiUrl('/api/asr/benchmark/run'), {
      method: 'POST',
      body: {
        algorithms: Array.from(asrBenchmarkAlgos),
        dataset: dataset,
        max_samples: maxSamples,
      },
    });

    const benchId = data.bench_id;
    pollAsrBenchmark(benchId);
  } catch (e) {
    showToast('启动评测失败: ' + e.message, 'error');
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-speedometer2"></i> <span>开始评测</span>';
  }
}

async function pollAsrBenchmark(benchId) {
  const poll = async () => {
    try {
      const data = await api(apiUrl(`/api/asr/benchmark/${benchId}`));
      $('asr-bench-bar').style.width = data.progress + '%';
      $('asr-bench-progress-text').textContent = data.progress.toFixed(0) + '%';
      $('asr-bench-status').textContent = data.status === 'running' ? '评测中...' : data.status;

      if (data.status === 'completed') {
        $('asr-benchmark-progress').classList.add('d-none');
        showAsrBenchmarkResult(data);
        $('asr-benchmark-btn').disabled = false;
        $('asr-benchmark-btn').innerHTML = '<i class="bi bi-speedometer2"></i> <span>开始评测</span>';
        return;
      } else if (data.status === 'failed') {
        showToast('评测失败', 'error');
        $('asr-benchmark-btn').disabled = false;
        $('asr-benchmark-btn').innerHTML = '<i class="bi bi-speedometer2"></i> <span>开始评测</span>';
        return;
      }
      setTimeout(poll, 3000);
    } catch (e) {
      setTimeout(poll, 5000);
    }
  };
  poll();
}

function showAsrBenchmarkResult(benchData) {
  $('asr-benchmark-result').classList.remove('d-none');

  // 排名表
  const ranking = [];
  for (const [name, result] of Object.entries(benchData.results || {})) {
    ranking.push({ name, ...result.metrics });
  }
  ranking.sort((a, b) => (a.cer ?? 999) - (b.cer ?? 999));

  let rankHtml = '<table class="table table-striped table-hover"><thead><tr><th>排名</th><th>算法</th><th>CER</th><th>WER</th><th>RTF</th><th>评测句数</th></tr></thead><tbody>';
  ranking.forEach((r, i) => {
    const cls = i === 0 ? 'table-warning' : i === 1 ? 'table-light' : '';
    rankHtml += `<tr class="${cls}"><td>${i + 1}</td><td>${r.name || '-'}</td><td>${(r.cer * 100).toFixed(2)}%</td><td>${(r.wer * 100).toFixed(2)}%</td><td>${r.rtf?.toFixed(3) || '-'}</td><td>${r.num_utterances || '-'}</td></tr>`;
  });
  rankHtml += '</tbody></table>';
  $('asr-ranking-table').innerHTML = rankHtml;

  // 详细指标
  let detailHtml = '';
  for (const [name, result] of Object.entries(benchData.results || {})) {
    const m = result.metrics || {};
    detailHtml += `<div class="card mb-2"><div class="card-body"><h6>${name}</h6>`;
    detailHtml += `<div class="row g-2">`;
    detailHtml += `<div class="col"><div class="stat-card"><div class="stat-value">${(m.cer * 100).toFixed(2)}%</div><div class="stat-label">CER</div></div></div>`;
    detailHtml += `<div class="col"><div class="stat-card"><div class="stat-value">${(m.wer * 100).toFixed(2)}%</div><div class="stat-label">WER</div></div></div>`;
    detailHtml += `<div class="col"><div class="stat-card"><div class="stat-value">${m.rtf?.toFixed(3) || '-'}</div><div class="stat-label">RTF</div></div></div>`;
    detailHtml += `<div class="col"><div class="stat-card"><div class="stat-value">${m.processing_time?.toFixed(1) || '-'}s</div><div class="stat-label">总耗时</div></div></div>`;
    detailHtml += `</div>`;
    if (result.errors && result.errors.length > 0) {
      detailHtml += `<div class="text-danger small mt-2">错误: ${result.errors.slice(0, 3).join('; ')}</div>`;
    }
    detailHtml += `</div></div>`;
  }
  $('asr-detail-metrics').innerHTML = detailHtml;
}

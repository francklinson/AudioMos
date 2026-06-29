/* ===================== AudioMOS 前端应用 ===================== */

// ======================== 工具函数 ========================
const $ = id => document.getElementById(id);
const qs = (s, c) => (c || document).querySelector(s);
const qsa = (s, c) => (c || document).querySelectorAll(s);

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
  denoisePollTimer: null,
  restorationPollTimer: null,
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
  _pollInterval: 5000,        // 默认轮询间隔
};

// ======================== 页面初始化数据加载 ========================
function loadAllData() {
  loadMosTasks();
  loadDenoiseAlgorithms();
  loadDenoiseTasks();
  loadRestorationAlgorithms();
  loadRestorationTasks();
  loadRefAudioList();
  loadFingerprintStatus();
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
  const files = t.files ? (Array.isArray(t.files) ? t.files.join(', ') : t.files) : (t.file_name || shortId(t.task_id));
  const isProcessing = _isActiveTask(t);
  const msg = t.message || '';
  const stepMatch = msg.match(/^\[(\w+)\](.+)/);
  const stepDesc = stepMatch ? stepMatch[2].trim() : msg;
  return `<div class="task-item" id="mos-task-${t.task_id}">
    <div class="task-header">
      <div>
        <span class="task-id">${shortId(t.task_id)}</span>
        <span class="task-file ms-2">${_escapeHtml(files)}</span>
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
    <div class="task-detail">创建: ${formatDate(t.created_at || t.create_time)}${t.message && !stepMatch ? ` | ${_escapeHtml(t.message)}` : ''}</div>
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
      actionsDiv.insertAdjacentHTML('beforeend',
        `<button class="btn btn-sm btn-outline-info" onclick="showMosResult('${taskId}')"><i class="bi bi-eye"></i> 查看</button>
         <button class="btn btn-sm btn-outline-success" onclick="downloadMosResult('${taskId}')"><i class="bi bi-download"></i> 下载</button>`);
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
 * - 首次渲染：创建所有任务卡片
 * - 后续轮询：仅增量更新已有卡片的进度/状态，保持WebSocket更新的DOM存活
 * - 新任务出现时：添加到列表
 * - 已删除任务：从DOM移除
 */
function renderMosTasks(tasks) {
  const container = $('mos-task-list');
  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无任务</div>';
    return;
  }

  // 判断是否需要全量渲染（首次或所有卡片都被移除）
  const needsFullRender = !container.querySelector('.task-item');

  if (needsFullRender) {
    // ====== 首次全量渲染 ======
    let html = '';
    tasks.forEach(t => { html += _buildMosTaskHtml(t); });
    container.innerHTML = html;
    tasks.forEach(t => { if (_isActiveTask(t)) connectMosWs(t.task_id); });
    return;
  }

  // ====== 增量更新 ======
  const existingIds = new Set();

  tasks.forEach(t => {
    existingIds.add(t.task_id);
    const el = $(`mos-task-${t.task_id}`);
    if (el) {
      // 已有卡片 → 增量更新
      _updateMosTaskElement(el, t);
    } else {
      // 新任务 → 插入到列表顶部
      container.insertAdjacentHTML('afterbegin', _buildMosTaskHtml(t));
      if (_isActiveTask(t)) connectMosWs(t.task_id);
    }
  });

  // 移除已删除的任务（本地删除或别的客户端删除）
  container.querySelectorAll('.task-item[id^="mos-task-"]').forEach(el => {
    const id = el.id.replace('mos-task-', '');
    if (!existingIds.has(id)) {
      disconnectMosWs(id);
      el.remove();
    }
  });
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

// ==================== 降噪测评 ====================
let denoiseSelectedAlgorithms = [];
let denoiseNoisyFiles = null;
let denoiseRefFiles = null;

async function loadDenoiseAlgorithms() {
  try {
    const data = await api(apiUrl('/api/denoise/algorithms'));
    const algs = data.algorithms || data || [];
    renderDenoiseAlgorithms(algs);
  } catch (_) {}
}

function renderDenoiseAlgorithms(algs) {
  const container = $('denoise-algorithms');
  if (!algs || algs.length === 0) {
    container.innerHTML = '<div class="text-muted">暂无可用算法</div>';
    return;
  }
  container.innerHTML = algs.map((a, i) => {
    const type = a.type || '深度学习';
    const typeClass = type.includes('传统') ? 'bg-secondary' : 'bg-info';
    const selected = denoiseSelectedAlgorithms.includes(a.name) ? 'selected' : '';
    const disabled = a.initialized === false ? 'opacity-50' : '';
    return `<div class="col-md-4 col-sm-6">
      <div class="algorithm-card ${selected} ${disabled}" data-alg="${a.name}" onclick="toggleDenoiseAlgorithm('${a.name}')">
        <div class="alg-name">${a.display_name || a.name}</div>
        <span class="alg-type ${typeClass} text-white">${type}</span>
        ${a.initialized === false ? '<span class="badge bg-warning ms-1">未初始化</span>' : ''}
        <div class="alg-desc mt-1">${a.description || ''}</div>
        ${a.advantages ? `<small class="text-success d-block mt-1"><i class="bi bi-check-circle"></i> ${a.advantages}</small>` : ''}
      </div>
    </div>`;
  }).join('');
  updateDenoiseSubmitBtn();
}

function toggleDenoiseAlgorithm(name) {
  const idx = denoiseSelectedAlgorithms.indexOf(name);
  if (idx >= 0) denoiseSelectedAlgorithms.splice(idx, 1);
  else denoiseSelectedAlgorithms.push(name);
  qsa('#denoise-algorithms .algorithm-card').forEach(c => {
    c.classList.toggle('selected', denoiseSelectedAlgorithms.includes(c.dataset.alg));
  });
  updateDenoiseSubmitBtn();
}

function initDenoisePage() {
  $('denoise-file-input').addEventListener('change', () => {
    denoiseNoisyFiles = $('denoise-file-input').files;
    updateDenoiseFileInfo();
    updateDenoiseSubmitBtn();
  });
  $('denoise-has-ref').addEventListener('change', () => {
    $('denoise-ref-upload').classList.toggle('d-none', !$('denoise-has-ref').checked);
  });
  $('denoise-ref-input').addEventListener('change', () => {
    denoiseRefFiles = $('denoise-ref-input').files;
  });
  $('denoise-submit-btn').addEventListener('click', submitDenoiseTask);
}

function updateDenoiseFileInfo() {
  if (denoiseNoisyFiles && denoiseNoisyFiles.length > 0) {
    $('denoise-file-info').classList.remove('d-none');
    $('denoise-file-count').textContent = `已选择 ${denoiseNoisyFiles.length} 个文件`;
  } else {
    $('denoise-file-info').classList.add('d-none');
  }
}

function updateDenoiseSubmitBtn() {
  const btn = $('denoise-submit-btn');
  btn.disabled = denoiseSelectedAlgorithms.length === 0 || !denoiseNoisyFiles || denoiseNoisyFiles.length === 0;
}

async function submitDenoiseTask() {
  const btn = $('denoise-submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-border spinner-border-sm"></div> 提交中...';
  try {
    const fd = new FormData();
    for (const f of denoiseNoisyFiles) fd.append('files', f);
    if (denoiseRefFiles) { for (const f of denoiseRefFiles) fd.append('reference_files', f); }
    fd.append('algorithms', JSON.stringify(denoiseSelectedAlgorithms));
    const data = await api(apiUrl('/api/denoise/upload'), { method: 'POST', formData: true, body: fd });
    // 启动处理
    for (const taskId of (data.task_ids || [data.task_id])) {
      await api(apiUrl(`/api/denoise/process/${taskId}`), { method: 'POST' });
    }
    showToast('降噪测评任务已提交', 'success');
    loadDenoiseTasks();
    // 切换到任务列表
    $('denoise-tab-tasks').click();
  } catch (err) {
    showToast('提交失败: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> 开始测评';
  }
}

async function loadDenoiseTasks() {
  try {
    const data = await api(apiUrl('/api/denoise/tasks'));
    const tasks = data.tasks || data || [];
    renderDenoiseTasks(tasks);
  } catch (_) {}
}

function renderDenoiseTasks(tasks) {
  // 更新计数
  $('denoise-task-count').textContent = tasks.length;
  const container = $('denoise-task-list');
  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无任务</div>';
    return;
  }
  container.innerHTML = tasks.map(t => {
    const status = t.status || 'pending';
    const progress = t.progress || 0;
    return `<div class="task-item">
      <div class="task-header">
        <div>
          <span class="task-id">${shortId(t.task_id)}</span>
          <span class="task-file ms-2">${t.message || ''}</span>
        </div>
        <div class="task-actions">
          ${statusBadge(status)}
          ${status === 'completed' ? `
            <button class="btn btn-sm btn-outline-success" onclick="downloadDenoiseReport('${t.task_id}','excel')"><i class="bi bi-file-earmark-excel"></i> Excel</button>
            <button class="btn btn-sm btn-outline-info" onclick="downloadDenoiseReport('${t.task_id}','html')"><i class="bi bi-file-earmark-code"></i> HTML</button>
            <button class="btn btn-sm btn-outline-secondary" onclick="downloadDenoiseReport('${t.task_id}','markdown')"><i class="bi bi-file-earmark-text"></i> Markdown</button>
          ` : ''}
          <button class="btn btn-sm btn-outline-danger" onclick="deleteDenoiseTask('${t.task_id}')"><i class="bi bi-trash"></i></button>
        </div>
      </div>
      ${status === 'processing' || status === 'pending' ? `<div class="mt-2"><div class="progress" style="height:6px"><div class="progress-bar progress-bar-striped progress-bar-animated" style="width:${progress}%"></div></div></div>` : ''}
      <div class="task-detail">创建: ${formatDate(t.created_at || t.create_time)}${t.message ? ` | ${t.message}` : ''}</div>
    </div>`;
  }).join('');
}

async function downloadDenoiseReport(taskId, format) {
  try {
    const resp = await api(apiUrl(`/api/denoise/download/${taskId}?format=${format}`), { method: 'GET', raw: true });
    const blob = await resp.blob();
    const ext = format === 'excel' ? 'xlsx' : format;
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `降噪测评报告_${shortId(taskId)}.${ext}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showToast('下载失败: ' + err.message, 'error');
  }
}

async function deleteDenoiseTask(taskId) {
  showConfirm('确定要删除此任务吗？', async () => {
    try {
      await api(apiUrl(`/api/denoise/tasks/${taskId}`), { method: 'DELETE' });
      showToast('删除成功', 'success');
      loadDenoiseTasks();
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
  $('restoration-file-input').addEventListener('change', () => {
    const files = $('restoration-file-input').files;
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
    showToast('提交失败: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-play-fill"></i> 开始修复';
  }
}

async function loadRestorationTasks() {
  try {
    const data = await api(apiUrl('/api/restoration/tasks'));
    const tasks = data.tasks || data || [];
    renderRestorationTasks(tasks);
  } catch (_) {}
}

function renderRestorationTasks(tasks) {
  $('restoration-task-count').textContent = tasks.length;
  const container = $('restoration-task-list');
  if (!tasks || tasks.length === 0) {
    container.innerHTML = '<div class="text-center text-muted py-4"><i class="bi bi-inbox"></i> 暂无任务 <button class="btn btn-sm btn-primary ms-2" onclick="document.getElementById(\'restoration-tab-upload\').click()">创建第一个修复任务</button></div>';
    return;
  }
  container.innerHTML = tasks.map(t => {
    const status = t.status || 'pending';
    const progress = t.progress || 0;
    const fileName = t.file_name || t.message || shortId(t.task_id);
    const algName = t.algorithm || '';
    return `<div class="task-item">
      <div class="task-header">
        <div>
          <span class="task-file">${fileName}</span>
          ${algName ? `<span class="badge bg-info ms-2">${algName}</span>` : ''}
          <span class="task-id ms-2">${shortId(t.task_id)}</span>
        </div>
        <div class="task-actions">
          ${statusBadge(status)}
          ${status === 'completed' ? `
            <button class="btn btn-sm btn-outline-info" onclick="toggleRestorationCompare('${t.task_id}', this)"><i class="bi bi-play-circle"></i> 试听对比</button>
            <button class="btn btn-sm btn-outline-success" onclick="downloadRestorationResult('${t.task_id}')"><i class="bi bi-download"></i></button>
          ` : ''}
          <button class="btn btn-sm btn-outline-danger" onclick="deleteRestorationTask('${t.task_id}')"><i class="bi bi-trash"></i></button>
        </div>
      </div>
      ${(status === 'processing' || status === 'pending') ? `<div class="mt-2"><div class="progress" style="height:6px"><div class="progress-bar progress-bar-striped progress-bar-animated" style="width:${progress}%"></div></div></div>` : ''}
      <div class="task-detail">创建: ${formatDate(t.created_at || t.create_time)}${t.duration ? ` | 耗时: ${t.duration.toFixed(1)}s` : ''}</div>
      <div class="restoration-compare-${t.task_id} d-none mt-2"></div>
    </div>`;
  }).join('');
}

async function toggleRestorationCompare(taskId, btn) {
  const container = qs(`.restoration-compare-${taskId}`);
  if (!container) return;
  if (!container.classList.contains('d-none')) {
    container.classList.add('d-none');
    if (btn) btn.innerHTML = '<i class="bi bi-play-circle"></i> 试听对比';
    return;
  }
  container.classList.remove('d-none');
  if (btn) btn.innerHTML = '<i class="bi bi-pause-circle"></i> 收起对比';
  if (!container.dataset.loaded) {
    container.dataset.loaded = 'true';
    const token = getToken();
    const srcUrl = apiUrl(`/api/restoration/source/${taskId}?token=${token}`);
    const resultUrl = apiUrl(`/api/restoration/download/${taskId}?token=${token}`);
    container.innerHTML = `<div class="audio-compare">
      <div class="audio-side"><div class="audio-side-header original"><i class="bi bi-soundwave"></i> 原始音频</div>
        <canvas class="waveform-canvas" id="wave-original-${shortId(taskId)}"></canvas>
        <div class="audio-controls" data-url="${srcUrl}">
          <button class="play-btn original-btn" onclick="toggleAudioPlay(this)"><i class="bi bi-play-fill"></i></button>
          <span class="time-display">00:00</span>
          <div class="progress-bar-custom" onclick="seekAudio(this, event)"><div class="progress-fill original-fill"></div></div>
          <span class="time-display duration">00:00</span>
        </div>
      </div>
      <div class="arrow-indicator"><i class="bi bi-arrow-right"></i></div>
      <div class="audio-side"><div class="audio-side-header processed"><i class="bi bi-soundwave"></i> 修复后音频</div>
        <canvas class="waveform-canvas" id="wave-processed-${shortId(taskId)}"></canvas>
        <div class="audio-controls" data-url="${resultUrl}">
          <button class="play-btn processed-btn" onclick="toggleAudioPlay(this)"><i class="bi bi-play-fill"></i></button>
          <span class="time-display">00:00</span>
          <div class="progress-bar-custom" onclick="seekAudio(this, event)"><div class="progress-fill processed-fill"></div></div>
          <span class="time-display duration">00:00</span>
        </div>
      </div>
    </div>`;
    // 绘制波形
    drawWaveform(`wave-original-${shortId(taskId)}`, srcUrl, '#d46b08');
    drawWaveform(`wave-processed-${shortId(taskId)}`, resultUrl, '#389e0d');
  }
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
      await api(apiUrl(`/api/restoration/tasks/${taskId}`), { method: 'DELETE' });
      showToast('删除成功', 'success');
      loadRestorationTasks();
    } catch (err) {
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

async function drawWaveform(canvasId, url, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const blob = await resp.blob();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
    const data = buf.getChannelData(0);
    const w = canvas.width || canvas.clientWidth;
    const h = canvas.height || canvas.clientHeight;
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
    ctx.close();
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

// ==================== 指纹数据库 ====================
async function loadFingerprintStatus() {
  try {
    const data = await api(apiUrl('/api/reference-audio/fingerprint/status'));
    renderFingerprintStatus(data);
  } catch (_) {
    $('ref-fingerprint-status').innerHTML = '<div class="text-center text-muted py-3">无法加载指纹状态</div>';
  }
}

function renderFingerprintStatus(data) {
  const container = $('ref-fingerprint-status');
  if (!data) {
    container.innerHTML = '<div class="text-center text-muted py-3">暂无数据</div>';
    return;
  }
  const ready = data.is_ready || data.ready || data.status === 'ready';
  $('ref-stat-fingerprint').innerHTML = ready
    ? '<span class="text-success">已就绪</span>'
    : '<span class="text-secondary">未建立</span>';
  container.innerHTML = `<div class="mb-2">
    <span class="badge ${ready ? 'bg-success' : 'bg-secondary'}">${ready ? '已就绪' : '未建立'}</span>
  </div>
  <div class="row g-2 mb-2">
    <div class="col-3"><small class="text-muted d-block">参考音频数</small><strong>${data.total_refs || data.ref_count || 0}</strong></div>
    <div class="col-3"><small class="text-muted d-block">Hash总数</small><strong>${data.total_hashes || data.hash_count || 0}</strong></div>
    <div class="col-3"><small class="text-muted d-block">唯一Hash</small><strong>${data.unique_hashes || data.unique_hash_count || 0}</strong></div>
    <div class="col-3"><small class="text-muted d-block">构建耗时</small><strong>${data.build_time ? data.build_time.toFixed(2) + 's' : '-'}</strong></div>
  </div>
  <div class="mt-2">
    <button class="btn btn-sm btn-outline-primary me-2" onclick="buildFingerprint()"><i class="bi bi-arrow-repeat"></i> 重建指纹数据库</button>
    <button class="btn btn-sm btn-outline-info" onclick="openMatchTest()"><i class="bi bi-search"></i> 测试内容匹配</button>
  </div>`;
}

async function buildFingerprint() {
  try {
    await api(apiUrl('/api/reference-audio/fingerprint/build'), { method: 'POST' });
    showToast('指纹数据库重建成功', 'success');
    loadFingerprintStatus();
  } catch (err) {
    showToast('重建失败: ' + err.message, 'error');
  }
}

function openMatchTest() {
  $('ref-match-taskid').value = '';
  $('ref-match-result').innerHTML = '';
  new bootstrap.Modal($('ref-match-modal')).show();
  $('ref-match-test-btn').onclick = testMatch;
}

async function testMatch() {
  const taskId = $('ref-match-taskid').value.trim();
  if (!taskId) { showToast('请输入任务ID', 'warning'); return; }
  const btn = $('ref-match-test-btn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner-border spinner-border-sm"></div> 测试中...';
  try {
    const fd = new FormData();
    fd.append('test_audio_id', taskId);
    const data = await api(apiUrl('/api/reference-audio/fingerprint/match-test'), { method: 'POST', formData: true, body: fd });
    const results = data.results || data.matches || [data].filter(Boolean);
    if (!results || results.length === 0) {
      $('ref-match-result').innerHTML = '<div class="text-muted">无匹配结果</div>';
      return;
    }
    $('ref-match-result').innerHTML = results.map(r => `<div class="card mb-2">
      <div class="card-body py-2">
        <div class="row g-2">
          <div class="col-6"><small class="text-muted">参考音频</small><br>${r.ref_file || r.reference_file || '-'}</div>
          <div class="col-3"><small class="text-muted">偏移</small><br>${r.offset != null ? r.offset.toFixed(2) + 's' : '-'}</div>
          <div class="col-3"><small class="text-muted">置信度</small><br>${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '-'}</div>
        </div>
        <div class="row g-2 mt-1">
          <div class="col-4"><small class="text-muted">Hash匹配</small><br>${r.hash_matches ?? '-'}</div>
          <div class="col-4"><small class="text-muted">DTW距离</small><br>${r.dtw_distance != null ? r.dtw_distance.toFixed(4) : '-'}</div>
          <div class="col-4"><small class="text-muted">是否匹配</small><br>${r.is_match || r.match_found ? '<span class="text-success">是</span>' : '<span class="text-danger">否</span>'}</div>
        </div>
      </div>
    </div>`).join('');
  } catch (err) {
    $('ref-match-result').innerHTML = `<div class="alert alert-danger">测试失败: ${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i class="bi bi-search"></i> 测试匹配';
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

  if ($('denoise-algorithms')) loadDenoiseTasks();
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

// 启动轮询
startPolling();

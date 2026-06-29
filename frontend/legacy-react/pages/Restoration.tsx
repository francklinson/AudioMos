/**
 * 音频修复页面
 * 提供降噪、去混响、超分辨率等音频修复功能
 * 支持处理前后音频试听对比 + 波形图可视化
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { restorationApi } from '../services/api';
import './Restoration.css';

// ── 类型定义 ──────────────────────────────────────────────

interface Algorithm {
  name: string;
  display_name: string;
  description: string;
  type: string;
  advantages: string[];
  limitations: string[];
  initialized: boolean;
}

interface TaskInfo {
  task_id: string;
  algorithm: string;
  filename: string;
  status: string;
  created_at: string;
  progress: number;
  message: string;
  result_file: string | null;
  processing_time: number | null;
  metadata: Record<string, any> | null;
}

// ── 波形图组件 ────────────────────────────────────────────

interface AudioWaveformProps {
  audioUrl: string;
  color?: string;
  height?: number;
}

const AudioWaveform: React.FC<AudioWaveformProps> = ({
  audioUrl,
  color = '#4fc3f7',
  height = 80,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!audioUrl) return;
    setLoading(true);

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    let cancelled = false;

    fetch(audioUrl)
      .then((res) => res.arrayBuffer())
      .then((buffer) => audioContext.decodeAudioData(buffer))
      .then((audioBuffer) => {
        if (cancelled) return;
        drawWaveform(audioBuffer);
        setLoading(false);
      })
      .catch(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      audioContext.close().catch(() => {});
    };
  }, [audioUrl]);

  const drawWaveform = (audioBuffer: AudioBuffer) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const data = audioBuffer.getChannelData(0);
    const width = canvas.width;
    const h = canvas.height;
    const step = Math.ceil(data.length / width);
    const amp = h / 2;

    ctx.clearRect(0, 0, width, h);

    // 中间线
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    ctx.lineTo(width, amp);
    ctx.stroke();

    // 上半波形
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[Math.floor(i * step + j)];
        if (datum === undefined) continue;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, amp - max * amp * 0.9);
    }
    ctx.lineTo(width, amp);
    ctx.closePath();
    ctx.fill();

    // 下半波形 (同一颜色，稍暗)
    ctx.fillStyle = color + '66';
    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[Math.floor(i * step + j)];
        if (datum === undefined) continue;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, amp - min * amp * 0.9);
    }
    ctx.lineTo(width, amp);
    ctx.closePath();
    ctx.fill();

    // 包络线
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < width; i++) {
      let min = 1.0, max = -1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[Math.floor(i * step + j)];
        if (datum === undefined) continue;
        if (datum < min) min = datum;
        if (datum > max) max = datum;
      }
      ctx.lineTo(i, amp - max * amp * 0.9);
    }
    ctx.stroke();

    ctx.strokeStyle = color + '88';
    ctx.beginPath();
    ctx.moveTo(0, amp);
    for (let i = 0; i < width; i++) {
      let min = 1.0;
      for (let j = 0; j < step; j++) {
        const datum = data[Math.floor(i * step + j)];
        if (datum === undefined) continue;
        if (datum < min) min = datum;
      }
      ctx.lineTo(i, amp - min * amp * 0.9);
    }
    ctx.stroke();
  };

  return (
    <div className="waveform-container">
      {loading && <div className="waveform-loading">加载波形...</div>}
      <canvas
        ref={canvasRef}
        width={600}
        height={height}
        className="waveform-canvas"
        style={{ display: loading ? 'none' : 'block' }}
      />
    </div>
  );
};

// ── 音频播放器组件 ─────────────────────────────────────────

interface AudioPlayerProps {
  audioUrl: string;
  label: string;
  color?: string;
  waveformColor?: string;
}

const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioUrl,
  label,
  color = '#4fc3f7',
  waveformColor = '#4fc3f7',
}) => {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (playing) {
      audio.pause();
    } else {
      audio.play();
    }
    setPlaying(!playing);
  };

  const handleTimeUpdate = () => {
    const audio = audioRef.current;
    if (audio) setCurrentTime(audio.currentTime);
  };

  const handleLoaded = () => {
    const audio = audioRef.current;
    if (audio) setDuration(audio.duration);
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    audio.currentTime = ratio * duration;
  };

  const formatTime = (t: number) => {
    const m = Math.floor(t / 60);
    const s = Math.floor(t % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="audio-player" style={{ borderColor: color }}>
      <div className="player-label">{label}</div>
      <AudioWaveform audioUrl={audioUrl} color={waveformColor} height={64} />
      <div className="player-controls">
        <button
          className="play-btn"
          onClick={togglePlay}
          style={{ color, borderColor: color }}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <span className="time-display">{formatTime(currentTime)}</span>
        <div className="seek-bar" onClick={handleSeek}>
          <div
            className="seek-fill"
            style={{
              width: duration ? `${(currentTime / duration) * 100}%` : '0%',
              background: color,
            }}
          />
        </div>
        <span className="time-display">{formatTime(duration)}</span>
      </div>
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoaded}
        onEnded={() => setPlaying(false)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        preload="metadata"
      />
    </div>
  );
};

// ── 主页面组件 ────────────────────────────────────────────

const Restoration: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [activeTab, setActiveTab] = useState<'upload' | 'tasks'>('upload');
  const [expandedTask, setExpandedTask] = useState<string | null>(null);

  // 加载算法列表
  const loadAlgorithms = useCallback(async () => {
    try {
      const data = await restorationApi.getAlgorithms();
      setAlgorithms(data);
      if (data.length > 0 && !selectedAlgorithm) {
        setSelectedAlgorithm(data[0].name);
      }
    } catch (err: any) {
      console.error('加载算法失败:', err);
    }
  }, [selectedAlgorithm]);

  // 加载任务列表
  const loadTasks = useCallback(async () => {
    try {
      const data = await restorationApi.getTasks();
      setTasks(data || []);
    } catch (err: any) {
      console.error('加载任务失败:', err);
    }
  }, []);

  useEffect(() => {
    loadAlgorithms();
    loadTasks();
  }, []);

  // 轮询任务状态
  useEffect(() => {
    const hasProcessing = tasks.some(
      (t) => t.status === 'processing' || t.status === 'pending'
    );
    if (!hasProcessing) return;

    const interval = setInterval(() => loadTasks(), 2000);
    return () => clearInterval(interval);
  }, [tasks, loadTasks]);

  // 上传并处理
  const handleSubmit = async () => {
    if (!file) {
      setError('请选择音频文件');
      return;
    }
    if (!selectedAlgorithm) {
      setError('请选择修复算法');
      return;
    }

    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const uploadResult = await restorationApi.uploadFile(file, selectedAlgorithm);
      await restorationApi.processTask(uploadResult.task_id);
      setSuccess('任务已提交，正在处理...');
      setFile(null);
      loadTasks();
      setActiveTab('tasks');
    } catch (err: any) {
      setError(err.response?.data?.detail || '操作失败');
    } finally {
      setLoading(false);
    }
  };

  // 下载结果
  const handleDownload = async (taskId: string) => {
    try {
      await restorationApi.downloadResult(taskId);
    } catch (err: any) {
      setError('下载失败');
    }
  };

  // 删除任务
  const handleDelete = async (taskId: string) => {
    if (!window.confirm('确定要删除该任务吗？')) return;
    try {
      await restorationApi.deleteTask(taskId);
      if (expandedTask === taskId) setExpandedTask(null);
      loadTasks();
    } catch (err: any) {
      setError('删除失败');
    }
  };

  // 切换展开任务（显示试听对比）
  const toggleExpand = (taskId: string) => {
    setExpandedTask(expandedTask === taskId ? null : taskId);
  };

  // 获取状态文本
  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: '等待处理',
      processing: '处理中',
      completed: '已完成',
      failed: '失败',
    };
    return statusMap[status] || status;
  };

  // 获取状态样式类名
  const getStatusClass = (status: string) => `status-${status}`;

  return (
    <div className="restoration-container">
      <div className="restoration-header">
        <h1>🔧 音频修复</h1>
        <p className="subtitle">
          去混响 · 超分辨率 · 语音降噪 · 语音分离 — 共 {algorithms.length} 个算法可选
        </p>
      </div>

      {/* 错误提示 */}
      {error && (
        <div className="alert alert-error">
          <span>{error}</span>
          <button onClick={() => setError('')}>×</button>
        </div>
      )}

      {/* 成功提示 */}
      {success && (
        <div className="alert alert-success">
          <span>{success}</span>
          <button onClick={() => setSuccess('')}>×</button>
        </div>
      )}

      {/* 标签切换 */}
      <div className="tab-bar">
        <button
          className={`tab ${activeTab === 'upload' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📤 新建修复
        </button>
        <button
          className={`tab ${activeTab === 'tasks' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('tasks')}
        >
          📋 任务列表 ({tasks.length})
        </button>
      </div>

      {/* 上传标签页 */}
      {activeTab === 'upload' && (
        <div className="tab-content">
          <div className="section">
            <h3>🤖 选择修复算法</h3>
            <div className="algorithm-intro">
              <p>📌 请根据您的处理需求选择合适的算法：</p>
              <ul>
                <li><strong>🏠 去混响</strong> - 去除房间回声和混响效果，提升语音清晰度，适合录音室/会议室录音</li>
                <li><strong>🔊 超分辨率</strong> - 将低采样率音频重建为高采样率（16k→48k），恢复高频细节</li>
                <li><strong>🎙️ 语音降噪</strong> - 多种深度学习降噪算法，去除背景噪声、风扇声、空调声等</li>
                <li><strong>👥 语音分离</strong> - 分离多人对话，提取特定说话人语音</li>
              </ul>
            </div>
            <div className="algorithm-grid">
              {algorithms.map((algo) => (
                <div
                  key={algo.name}
                  className={`algorithm-card ${selectedAlgorithm === algo.name ? 'selected' : ''}`}
                  onClick={() => setSelectedAlgorithm(algo.name)}
                >
                  <div className="algo-header">
                    <h4>{algo.display_name}</h4>
                    <span className={`algo-type ${algo.type === '深度学习' ? 'type-dl' : 'type-traditional'}`}>
                      {algo.type}
                    </span>
                  </div>
                  <p className="algo-desc">{algo.description}</p>
                  {algo.advantages.length > 0 && (
                    <div className="algo-pros">
                      <strong>✅ 优势:</strong>
                      <ul>
                        {algo.advantages.map((adv, i) => (
                          <li key={i}>{adv}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {algo.limitations.length > 0 && (
                    <div className="algo-cons">
                      <strong>⚠️ 局限:</strong>
                      <ul>
                        {algo.limitations.map((lim, i) => (
                          <li key={i}>{lim}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="section">
            <h3>上传音频文件</h3>
            <div className="upload-area">
              <input
                type="file"
                accept=".wav,.mp3,.flac"
                onChange={(e) => {
                  setFile(e.target.files?.[0] || null);
                  setError('');
                }}
                id="file-input"
              />
              <label htmlFor="file-input" className="file-label">
                {file ? (
                  <span>📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)</span>
                ) : (
                  <span>📁 点击选择音频文件 (.wav, .mp3, .flac)</span>
                )}
              </label>
            </div>
          </div>

          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={loading || !file}
          >
            {loading ? '处理中...' : '🚀 开始修复'}
          </button>

          <div className="info-section">
            <h3>📖 功能说明</h3>
            <div className="info-grid">
              <div className="info-card">
                <h4>🎙️ 语音降噪</h4>
                <p>使用深度学习或传统信号处理方法去除音频中的背景噪声。</p>
              </div>
              <div className="info-card">
                <h4>👥 语音分离</h4>
                <p>从混合音频中分离出不同说话人的语音，支持2人场景。</p>
              </div>
              <div className="info-card">
                <h4>🏠 去混响</h4>
                <p>去除房间混响效果，提升语音清晰度。</p>
              </div>
              <div className="info-card">
                <h4>🔊 超分辨率</h4>
                <p>将低采样率音频重建为高采样率，恢复高频细节。</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 任务列表标签页 */}
      {activeTab === 'tasks' && (
        <div className="tab-content">
          {tasks.length === 0 ? (
            <div className="empty-state">
              <p>暂无修复任务</p>
              <button className="btn-secondary" onClick={() => setActiveTab('upload')}>
                创建第一个修复任务
              </button>
            </div>
          ) : (
            <div className="task-list">
              {tasks.map((task) => {
                const isExpanded = expandedTask === task.task_id;
                const isCompleted = task.status === 'completed';
                const isProcessing = task.status === 'processing';
                const isFailed = task.status === 'failed';
                const sourceUrl = restorationApi.getSourceAudioUrl(task.task_id);
                const resultUrl = isCompleted
                  ? restorationApi.getResultAudioUrl(task.task_id)
                  : '';

                return (
                  <div key={task.task_id} className={`task-card ${isExpanded ? 'task-expanded' : ''}`}>
                    <div className="task-info">
                      <div className="task-header">
                        <span className="task-filename">📄 {task.filename}</span>
                        <span className={`task-status ${getStatusClass(task.status)}`}>
                          {getStatusText(task.status)}
                        </span>
                      </div>
                      <div className="task-meta">
                        <span>算法: {task.algorithm}</span>
                        <span>创建: {task.created_at}</span>
                        {task.processing_time && (
                          <span>处理耗时: {task.processing_time.toFixed(2)}s</span>
                        )}
                      </div>

                      {isProcessing && (
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${(task.progress || 0) * 100}%` }}
                          />
                        </div>
                      )}

                      {task.metadata && Object.keys(task.metadata).length > 0 && (
                        <div className="task-metadata">
                          {Object.entries(task.metadata).map(([key, value]) => (
                            <span key={key} className="metadata-item">
                              {key}: {String(value)}
                            </span>
                          ))}
                        </div>
                      )}

                      {task.message && <p className="task-message">{task.message}</p>}

                      {/* ── 展开按钮：完成/失败的处理任务始终可见 ── */}
                      {(isCompleted || isFailed) && (
                        <button
                          className="btn-expand"
                          onClick={() => toggleExpand(task.task_id)}
                        >
                          {isExpanded ? '🔼 收起详情' : '🔽 展开试听对比'}
                        </button>
                      )}

                      {/* ── 展开区域 ── */}
                      {isExpanded && isCompleted && (
                        <div className="comparison-panel">
                          <div className="comparison-grid">
                            <div className="comparison-side">
                              <AudioPlayer
                                audioUrl={sourceUrl}
                                label="📌 原始带噪音频"
                                color="#ff9800"
                                waveformColor="#ff9800"
                              />
                            </div>
                            <div className="comparison-arrow">→</div>
                            <div className="comparison-side">
                              <AudioPlayer
                                audioUrl={resultUrl}
                                label="✅ 修复后音频"
                                color="#4caf50"
                                waveformColor="#4caf50"
                              />
                            </div>
                          </div>
                        </div>
                      )}

                      {isExpanded && isFailed && (
                        <div className="comparison-panel">
                          <p style={{ color: '#ef5350' }}>
                            ⚠️ 处理失败: {task.message || '未知错误'}
                          </p>
                        </div>
                      )}
                    </div>

                    <div className="task-actions">
                      {isCompleted && (
                        <button
                          className="btn-download"
                          onClick={() => handleDownload(task.task_id)}
                        >
                          ⬇️ 下载
                        </button>
                      )}
                      <button
                        className="btn-delete"
                        onClick={() => handleDelete(task.task_id)}
                      >
                        🗑️ 删除
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* 使用提示 */}
      <div className="tips-section">
        <h4>💡 使用提示</h4>
        <ul>
          <li><strong>传统方法</strong>（谱减法、维纳滤波）：无需下载模型，即时可用</li>
          <li><strong>ClearVoice系列</strong>（FRCRN、MossFormer2、MossFormerGAN）：降噪效果最优</li>
          <li>处理完成后，任务卡片会出现「🔽 展开试听对比」按钮，点击可查看波形图并试听前后差异</li>
          <li>支持格式：.wav, .mp3, .flac；推荐16kHz或48kHz单声道</li>
        </ul>
      </div>
    </div>
  );
};

export default Restoration;

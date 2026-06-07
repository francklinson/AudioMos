/**
 * 音频修复页面
 * 提供去混响、超分辨率等音频修复功能
 */
import React, { useState, useEffect, useCallback } from 'react';
import { restorationApi } from '../services/api';
import './Restoration.css';

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

const Restoration: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [activeTab, setActiveTab] = useState<'upload' | 'tasks'>('upload');

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
    const hasProcessing = tasks.some((t) => t.status === 'processing' || t.status === 'pending');
    if (!hasProcessing) return;

    const interval = setInterval(() => {
      loadTasks();
    }, 2000);

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
      // 上传文件
      const uploadResult = await restorationApi.uploadFile(file, selectedAlgorithm);
      // 提交处理
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
      loadTasks();
    } catch (err: any) {
      setError('删除失败');
    }
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
  const getStatusClass = (status: string) => {
    return `status-${status}`;
  };

  return (
    <div className="restoration-container">
      <div className="restoration-header">
        <h1>🔧 音频修复</h1>
        <p className="subtitle">去混响 · 超分辨率 · 音频增强</p>
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
          {/* 算法选择 */}
          <div className="section">
            <h3>选择修复算法</h3>
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

          {/* 文件上传 */}
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

          {/* 提交按钮 */}
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={loading || !file}
          >
            {loading ? '处理中...' : '🚀 开始修复'}
          </button>

          {/* 说明文档 */}
          <div className="info-section">
            <h3>📖 功能说明</h3>
            <div className="info-card">
              <h4>去混响 (Dereverberation)</h4>
              <p>使用深度学习模型去除音频中的房间混响效果，提升语音清晰度。适用于会议室录音、远场语音等场景。</p>
            </div>
            <div className="info-card">
              <h4>超分辨率 (Super Resolution)</h4>
              <p>将低采样率音频重建为高采样率（带宽扩展），恢复丢失的高频成分。适用于老旧录音修复、电话音频增强等场景。</p>
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
              {tasks.map((task) => (
                <div key={task.task_id} className="task-card">
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

                    {/* 进度条 */}
                    {task.status === 'processing' && (
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{ width: `${(task.progress || 0) * 100}%` }}
                        />
                      </div>
                    )}

                    {/* 元数据 */}
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
                  </div>

                  <div className="task-actions">
                    {task.status === 'completed' && (
                      <button className="btn-download" onClick={() => handleDownload(task.task_id)}>
                        ⬇️ 下载
                      </button>
                    )}
                    <button className="btn-delete" onClick={() => handleDelete(task.task_id)}>
                      🗑️ 删除
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 使用提示 */}
      <div className="tips-section">
        <h4>💡 使用提示</h4>
        <ul>
          <li>去混响: 建议使用16kHz单声道音频，处理时间取决于音频长度</li>
          <li>超分辨率: 输入低采样率音频(8kHz)，输出高采样率音频(48kHz)</li>
          <li>深度学习算法首次使用时会下载模型，可能需要几分钟</li>
          <li>传统信号处理方法无需下载模型，即刻可用</li>
        </ul>
      </div>
    </div>
  );
};

export default Restoration;

import React, { useState, useEffect } from 'react';
import { denoiseApi } from '../services/api';
import './Denoise.css';

interface Algorithm {
  name: string;
  description: string;
  type: string;
  pros: string[];
  cons: string[];
  initialized: boolean;
}

interface Task {
  task_id: string;
  status: string;
  progress: number;
  message: string;
  created_at: string;
  updated_at: string;
}

const Denoise: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<Algorithm[]>([]);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([]);
  const [noisyFiles, setNoisyFiles] = useState<FileList | null>(null);
  const [referenceFiles, setReferenceFiles] = useState<FileList | null>(null);
  const [hasReference, setHasReference] = useState(false);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [currentTask, setCurrentTask] = useState<Task | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState<'upload' | 'tasks'>('upload');

  // 加载算法列表
  useEffect(() => {
    fetchAlgorithms();
    fetchTasks();
  }, []);

  // 轮询任务状态
  useEffect(() => {
    if (!currentTask || currentTask.status === 'completed' || currentTask.status === 'failed') {
      return;
    }

    const interval = setInterval(async () => {
      try {
        const status = await denoiseApi.getTaskStatus(currentTask.task_id);
        setCurrentTask(status);
        
        if (status.status === 'completed' || status.status === 'failed') {
          fetchTasks();
        }
      } catch (err) {
        console.error('获取任务状态失败:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [currentTask]);

  const fetchAlgorithms = async () => {
    try {
      const data = await denoiseApi.getAlgorithms();
      setAlgorithms(data);
    } catch (err: any) {
      setError('获取算法列表失败: ' + err.message);
    }
  };

  const fetchTasks = async () => {
    try {
      const data = await denoiseApi.getTasks();
      setTasks(data);
    } catch (err: any) {
      console.error('获取任务列表失败:', err);
    }
  };

  const handleAlgorithmToggle = (name: string) => {
    setSelectedAlgorithms(prev => 
      prev.includes(name) 
        ? prev.filter(a => a !== name)
        : [...prev, name]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!noisyFiles || noisyFiles.length === 0) {
      setError('请上传带噪音频文件');
      return;
    }
    
    if (selectedAlgorithms.length === 0) {
      setError('请至少选择一个降噪算法');
      return;
    }

    setLoading(true);
    setError('');

    try {
      // 上传文件
      const uploadResult = await denoiseApi.uploadFiles(
        noisyFiles,
        hasReference ? referenceFiles : null,
        selectedAlgorithms
      );

      // 提交处理任务
      await denoiseApi.processTask(uploadResult.task_id);

      setCurrentTask({
        task_id: uploadResult.task_id,
        status: 'queued',
        progress: 0,
        message: '任务已加入队列',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });
      
      setActiveTab('tasks');
      fetchTasks();
    } catch (err: any) {
      setError('提交任务失败: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async (taskId: string, format: 'excel' | 'html' | 'markdown') => {
    try {
      await denoiseApi.downloadReport(taskId, format);
    } catch (err: any) {
      setError('下载报告失败: ' + err.message);
    }
  };

  const handleDeleteTask = async (taskId: string) => {
    if (!window.confirm('确定要删除这个任务吗？')) return;
    
    try {
      await denoiseApi.deleteTask(taskId);
      fetchTasks();
      if (currentTask?.task_id === taskId) {
        setCurrentTask(null);
      }
    } catch (err: any) {
      setError('删除任务失败: ' + err.message);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return '#4CAF50';
      case 'failed': return '#f44336';
      case 'processing': return '#2196F3';
      case 'queued': return '#FF9800';
      default: return '#9E9E9E';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return '已完成';
      case 'failed': return '失败';
      case 'processing': return '处理中';
      case 'queued': return '排队中';
      default: return '等待中';
    }
  };

  return (
    <div className="denoise-container">
      <div className="denoise-header">
        <h1>🎧 降噪算法测评</h1>
        <p>对比测试多种业界先进的音频降噪算法</p>
      </div>

      <div className="denoise-tabs">
        <button 
          className={activeTab === 'upload' ? 'active' : ''}
          onClick={() => setActiveTab('upload')}
        >
          📤 新建测评
        </button>
        <button 
          className={activeTab === 'tasks' ? 'active' : ''}
          onClick={() => setActiveTab('tasks')}
        >
          📋 任务列表 ({tasks.length})
        </button>
      </div>

      {error && (
        <div className="error-message">
          {error}
          <button onClick={() => setError('')}>✕</button>
        </div>
      )}

      {activeTab === 'upload' && (
        <form className="denoise-form" onSubmit={handleSubmit}>
          {/* 算法选择 */}
          <div className="form-section">
            <h3>🤖 选择降噪算法</h3>
            <div className="algorithms-grid">
              {algorithms.map(algo => (
                <div 
                  key={algo.name}
                  className={`algorithm-card ${selectedAlgorithms.includes(algo.name) ? 'selected' : ''} ${!algo.initialized ? 'unavailable' : ''}`}
                  onClick={() => algo.initialized && handleAlgorithmToggle(algo.name)}
                >
                  <div className="algorithm-header">
                    <input 
                      type="checkbox"
                      checked={selectedAlgorithms.includes(algo.name)}
                      onChange={() => {}}
                      disabled={!algo.initialized}
                    />
                    <span className="algorithm-name">{algo.name}</span>
                    <span className={`algorithm-type ${algo.type === '深度学习' ? 'dl' : 'traditional'}`}>
                      {algo.type}
                    </span>
                  </div>
                  <p className="algorithm-desc">{algo.description}</p>
                  <div className="algorithm-pros-cons">
                    {algo.pros.length > 0 && (
                      <div className="pros">
                        <strong>✓ 优势:</strong> {algo.pros.join(', ')}
                      </div>
                    )}
                    {algo.cons.length > 0 && (
                      <div className="cons">
                        <strong>✗ 局限:</strong> {algo.cons.join(', ')}
                      </div>
                    )}
                  </div>
                  {!algo.initialized && (
                    <div className="unavailable-badge">未初始化</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* 文件上传 */}
          <div className="form-section">
            <h3>📁 上传音频文件</h3>
            
            <div className="file-upload-section">
              <div className="file-upload">
                <label>带噪音频文件 (必填)</label>
                <input 
                  type="file" 
                  accept=".wav,.mp3"
                  multiple
                  onChange={(e) => setNoisyFiles(e.target.files)}
                />
                <span className="file-hint">
                  {noisyFiles ? `已选择 ${noisyFiles.length} 个文件` : '支持 .wav, .mp3 格式'}
                </span>
              </div>

              <div className="reference-toggle">
                <label>
                  <input 
                    type="checkbox"
                    checked={hasReference}
                    onChange={(e) => setHasReference(e.target.checked)}
                  />
                  我有参考音频(干净语音)
                </label>
                <span className="hint">提供参考音频可计算PESQ、STOI等有参考指标</span>
              </div>

              {hasReference && (
                <div className="file-upload">
                  <label>参考音频文件 (可选)</label>
                  <input 
                    type="file" 
                    accept=".wav,.mp3"
                    multiple
                    onChange={(e) => setReferenceFiles(e.target.files)}
                  />
                  <span className="file-hint">
                    {referenceFiles ? `已选择 ${referenceFiles.length} 个文件` : '与带噪音频文件一一对应'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* 提交按钮 */}
          <button 
            type="submit" 
            className="submit-btn"
            disabled={loading}
          >
            {loading ? '提交中...' : '🚀 开始测评'}
          </button>
        </form>
      )}

      {activeTab === 'tasks' && (
        <div className="tasks-section">
          {/* 当前任务状态 */}
          {currentTask && currentTask.status !== 'completed' && currentTask.status !== 'failed' && (
            <div className="current-task">
              <h3>⏳ 当前任务</h3>
              <div className="task-card active">
                <div className="task-header">
                  <span className="task-id">{currentTask.task_id.slice(0, 8)}</span>
                  <span 
                    className="task-status"
                    style={{ backgroundColor: getStatusColor(currentTask.status) }}
                  >
                    {getStatusText(currentTask.status)}
                  </span>
                </div>
                <div className="progress-bar">
                  <div 
                    className="progress-fill"
                    style={{ width: `${currentTask.progress}%` }}
                  />
                </div>
                <p className="task-message">{currentTask.message}</p>
              </div>
            </div>
          )}

          {/* 任务列表 */}
          <div className="tasks-list">
            <h3>📋 历史任务</h3>
            {tasks.length === 0 ? (
              <div className="empty-state">暂无任务</div>
            ) : (
              tasks.map(task => (
                <div key={task.task_id} className="task-card">
                  <div className="task-header">
                    <span className="task-id">{task.task_id.slice(0, 8)}</span>
                    <span 
                      className="task-status"
                      style={{ backgroundColor: getStatusColor(task.status) }}
                    >
                      {getStatusText(task.status)}
                    </span>
                  </div>
                  <p className="task-message">{task.message}</p>
                  <div className="task-meta">
                    <span>创建时间: {new Date(task.created_at).toLocaleString()}</span>
                  </div>
                  
                  {task.status === 'completed' && (
                    <div className="task-actions">
                      <button onClick={() => handleDownload(task.task_id, 'excel')}>
                        📊 Excel报告
                      </button>
                      <button onClick={() => handleDownload(task.task_id, 'html')}>
                        🌐 HTML报告
                      </button>
                      <button onClick={() => handleDownload(task.task_id, 'markdown')}>
                        📝 Markdown
                      </button>
                    </div>
                  )}
                  
                  <button 
                    className="delete-btn"
                    onClick={() => handleDeleteTask(task.task_id)}
                  >
                    🗑️ 删除
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* 说明文档 */}
      <div className="denoise-info">
        <h3>📖 使用说明</h3>
        <div className="info-grid">
          <div className="info-card">
            <h4>📈 评估指标说明</h4>
            <ul>
              <li><strong>PESQ</strong>: 感知语音质量 (1-4.5, 越高越好)</li>
              <li><strong>STOI</strong>: 短时客观可懂度 (0-1, 越高越好)</li>
              <li><strong>SI-SDR</strong>: 尺度不变信噪比 (dB, 越高越好)</li>
              <li><strong>DNSMOS</strong>: 深度噪声抑制MOS分</li>
              <li><strong>RTF</strong>: 实时因子 (&lt;1表示实时处理)</li>
            </ul>
          </div>
          <div className="info-card">
            <h4>🔧 支持的算法</h4>
            <ul>
              <li><strong>深度学习</strong>: MetricGAN+, SepFormer, FRCRN, MossFormer</li>
              <li><strong>传统方法</strong>: 谱减法, 维纳滤波</li>
            </ul>
          </div>
          <div className="info-card">
            <h4>💡 使用建议</h4>
            <ul>
              <li>提供参考音频可获得更全面的评估指标</li>
              <li>建议同时测试多种算法进行对比</li>
              <li>大文件处理可能需要较长时间，请耐心等待</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Denoise;

import axios, { AxiosInstance, AxiosError } from 'axios';

// 根据环境选择API基础URL
const getBaseURL = () => {
  // 测试环境使用相对路径，让 MSW 可以拦截
  if (process.env.NODE_ENV === 'test' || import.meta.env?.MODE === 'test') {
    return '';
  }
  return import.meta.env.VITE_API_URL || '';
};

const API_BASE_URL = getBaseURL();

// 创建axios实例
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器 - 添加token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器 - 处理错误
api.interceptors.response.use(
  (response) => {
    console.log('[API] Response:', response.config?.url, response.status);
    return response;
  },
  (error: AxiosError) => {
    console.error('[API] Error:', error.config?.url, error.response?.status, error.message);
    if (error.response?.status === 401) {
      // Token过期或无效,清除并跳转登录
      console.warn('[API] 401 detected, redirecting to /login');
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// 认证相关API
export const authApi = {
  login: async (username: string, password: string) => {
    // OAuth2PasswordRequestForm 要求 application/x-www-form-urlencoded 格式
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);

    const response = await api.post('/api/auth/login', params, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });
    return response.data;
  },
  
  getCurrentUser: async () => {
    const response = await api.get('/api/auth/me');
    return response.data;
  },
  
  logout: async () => {
    const response = await api.post('/api/auth/logout');
    return response.data;
  },
};

// MOS评分相关API
export const mosApi = {
  uploadFiles: async (files: File[], metrics?: string[]) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    // 添加计算项目配置
    if (metrics && metrics.length > 0) {
      formData.append('metrics', JSON.stringify(metrics));
    }

    const response = await api.post('/api/mos/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  startProcess: async (taskId: string) => {
    const response = await api.post(`/api/mos/process/${taskId}`);
    return response.data;
  },
  
  getTaskStatus: async (taskId: string) => {
    const response = await api.get(`/api/mos/tasks/${taskId}`);
    return response.data;
  },
  
  getTasks: async () => {
    const response = await api.get('/api/mos/tasks');
    return response.data;
  },
  
  downloadResult: async (taskId: string) => {
    const response = await api.get(`/api/mos/download/${taskId}`, {
      responseType: 'blob',
    });
    return response.data;
  },
  
  deleteTask: async (taskId: string) => {
    const response = await api.delete(`/api/mos/tasks/${taskId}`);
    return response.data;
  },

  getTaskResults: async (taskId: string) => {
    const response = await api.get(`/api/mos/results/${taskId}`);
    return response.data;
  },

  /** 获取结果详情中音频文件的试听URL */
  getAudioUrl: (taskId: string, filename: string) => {
    const token = localStorage.getItem('token');
    return `${api.defaults.baseURL}/api/mos/audio/${taskId}/${encodeURIComponent(filename)}?token=${encodeURIComponent(token || '')}`;
  },
};

// 降噪测评相关API
export const denoiseApi = {
  getAlgorithms: async () => {
    const response = await api.get('/api/denoise/algorithms');
    return response.data;
  },
  
  uploadFiles: async (files: FileList, referenceFiles: FileList | null, algorithms: string[]) => {
    const formData = new FormData();
    
    // 添加带噪音频文件
    Array.from(files).forEach((file) => {
      formData.append('files', file);
    });
    
    // 添加参考音频文件
    if (referenceFiles) {
      Array.from(referenceFiles).forEach((file) => {
        formData.append('reference_files', file);
      });
    }
    
    // 添加算法选择
    formData.append('algorithms', JSON.stringify(algorithms));
    
    const response = await api.post('/api/denoise/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  processTask: async (taskId: string) => {
    const response = await api.post(`/api/denoise/process/${taskId}`);
    return response.data;
  },
  
  getTaskStatus: async (taskId: string) => {
    const response = await api.get(`/api/denoise/tasks/${taskId}`);
    return response.data;
  },
  
  getTasks: async () => {
    const response = await api.get('/api/denoise/tasks');
    return response.data;
  },
  
  downloadReport: async (taskId: string, format: 'excel' | 'html' | 'markdown' = 'excel') => {
    const response = await api.get(`/api/denoise/download/${taskId}?format=${format}`, {
      responseType: 'blob',
    });
    
    // 创建下载链接
    const blob = new Blob([response.data]);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    
    // 设置文件名
    const extension = format === 'excel' ? 'xlsx' : format;
    link.download = `降噪测评报告_${taskId.slice(0, 8)}.${extension}`;
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    
    return response.data;
  },
  
  deleteTask: async (taskId: string) => {
    const response = await api.delete(`/api/denoise/tasks/${taskId}`);
    return response.data;
  },
};

// 音频修复相关API
export const restorationApi = {
  getAlgorithms: async () => {
    const response = await api.get('/api/restoration/algorithms');
    return response.data;
  },

  uploadFile: async (file: File, algorithm: string) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algorithm);

    const response = await api.post('/api/restoration/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  uploadBatch: async (files: File[], algorithm: string) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    formData.append('algorithm', algorithm);

    const response = await api.post('/api/restoration/upload-batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  processTask: async (taskId: string) => {
    const response = await api.post(`/api/restoration/process/${taskId}`);
    return response.data;
  },

  getTaskStatus: async (taskId: string) => {
    const response = await api.get(`/api/restoration/tasks/${taskId}`);
    return response.data;
  },

  getTasks: async () => {
    const response = await api.get('/api/restoration/tasks');
    return response.data;
  },

  downloadResult: async (taskId: string) => {
    const response = await api.get(`/api/restoration/download/${taskId}`, {
      responseType: 'blob',
    });

    const blob = new Blob([response.data]);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `修复结果_${taskId.slice(0, 8)}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);

    return response.data;
  },

  deleteTask: async (taskId: string) => {
    const response = await api.delete(`/api/restoration/tasks/${taskId}`);
    return response.data;
  },

  /** 获取原始音频的试听URL */
  getSourceAudioUrl: (taskId: string) => {
    const token = localStorage.getItem('token');
    return `${api.defaults.baseURL}/api/restoration/source/${taskId}?token=${encodeURIComponent(token || '')}`;
  },

  /** 获取处理后音频的试听URL */
  getResultAudioUrl: (taskId: string) => {
    const token = localStorage.getItem('token');
    return `${api.defaults.baseURL}/api/restoration/download/${taskId}?token=${encodeURIComponent(token || '')}`;
  },
};

// 参考音频管理相关API
export const referenceAudioApi = {
  /** 获取参考音频列表 */
  list: async () => {
    const response = await api.get('/api/reference-audio/list');
    return response.data;
  },

  /** 上传单个参考音频 */
  upload: async (file: File, description?: string, setAsDefault?: boolean) => {
    const formData = new FormData();
    formData.append('file', file);
    if (description) formData.append('description', description);
    if (setAsDefault) formData.append('set_as_default', 'true');

    const response = await api.post('/api/reference-audio/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /** 批量上传参考音频 */
  uploadBatch: async (files: File[], setFirstAsDefault?: boolean) => {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    if (setFirstAsDefault) formData.append('set_first_as_default', 'true');

    const response = await api.post('/api/reference-audio/upload-batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /** 获取参考音频详情 */
  getDetail: async (audioId: string) => {
    const response = await api.get(`/api/reference-audio/detail/${audioId}`);
    return response.data;
  },

  /** 下载参考音频 */
  download: async (audioId: string) => {
    const response = await api.get(`/api/reference-audio/download/${audioId}`, {
      responseType: 'blob',
    });
    const blob = new Blob([response.data]);
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `reference_${audioId.slice(0, 8)}.wav`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
    return response.data;
  },

  /** 更新参考音频信息（描述、ground truth文本） */
  update: async (audioId: string, data: { description?: string; ground_truth_text?: string }) => {
    const response = await api.put(`/api/reference-audio/update/${audioId}`, data);
    return response.data;
  },

  /** 删除参考音频 */
  delete: async (audioId: string) => {
    const response = await api.delete(`/api/reference-audio/delete/${audioId}`);
    return response.data;
  },

  /** 检查参考音频状态 */
  checkStatus: async () => {
    const response = await api.get('/api/reference-audio/check/status');
    return response.data;
  },

  /** 建立/重建指纹数据库 */
  buildFingerprint: async () => {
    const response = await api.post('/api/reference-audio/fingerprint/build');
    return response.data;
  },

  /** 获取指纹数据库状态 */
  getFingerprintStatus: async () => {
    const response = await api.get('/api/reference-audio/fingerprint/status');
    return response.data;
  },

  /** 测试内容匹配 */
  testMatch: async (testAudioId: string) => {
    const formData = new FormData();
    formData.append('test_audio_id', testAudioId);
    const response = await api.post('/api/reference-audio/fingerprint/match-test', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },
};

export default api;

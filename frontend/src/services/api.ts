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

export default api;

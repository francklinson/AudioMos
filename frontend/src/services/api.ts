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
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Token过期或无效,清除并跳转登录
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// 认证相关API
export const authApi = {
  login: async (username: string, password: string) => {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await api.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
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

export default api;

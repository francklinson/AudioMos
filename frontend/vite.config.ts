import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import fs from 'fs'
import yaml from 'js-yaml'

// 读取后端配置
function getBackendConfig() {
  // 优先使用环境变量
  const backendHost = process.env.AUDIOMOS_BACKEND_HOST || 'localhost'
  const backendPort = process.env.AUDIOMOS_BACKEND_PORT || '8000'
  
  // 尝试从配置文件读取
  const configPaths = [
    path.resolve(__dirname, '../config/config.yaml'),
    path.resolve(__dirname, './config/config.yaml'),
  ]
  
  for (const configPath of configPaths) {
    if (fs.existsSync(configPath)) {
      try {
        const config = yaml.load(fs.readFileSync(configPath, 'utf8')) as any
        if (config?.server?.backend) {
          return {
            host: process.env.AUDIOMOS_BACKEND_HOST || config.server.backend.host || 'localhost',
            port: process.env.AUDIOMOS_BACKEND_PORT || config.server.backend.port || 8000,
          }
        }
      } catch (e) {
        console.warn('读取配置文件失败:', e)
      }
    }
  }
  
  return {
    host: backendHost,
    port: parseInt(backendPort),
  }
}

// 读取前端配置
function getFrontendConfig() {
  // 优先使用环境变量
  const frontendPort = process.env.AUDIOMOS_FRONTEND_PORT || '3000'
  const frontendHost = process.env.AUDIOMOS_FRONTEND_HOST || '0.0.0.0'
  
  // 尝试从配置文件读取
  const configPaths = [
    path.resolve(__dirname, '../config/config.yaml'),
    path.resolve(__dirname, './config/config.yaml'),
  ]
  
  for (const configPath of configPaths) {
    if (fs.existsSync(configPath)) {
      try {
        const config = yaml.load(fs.readFileSync(configPath, 'utf8')) as any
        if (config?.server?.frontend) {
          return {
            host: process.env.AUDIOMOS_FRONTEND_HOST || config.server.frontend.host || '0.0.0.0',
            port: process.env.AUDIOMOS_FRONTEND_PORT || config.server.frontend.port || 3000,
          }
        }
      } catch (e) {
        console.warn('读取配置文件失败:', e)
      }
    }
  }
  
  return {
    host: frontendHost,
    port: parseInt(frontendPort),
  }
}

const backendConfig = getBackendConfig()
const frontendConfig = getFrontendConfig()

console.log(`前端配置: host=${frontendConfig.host}, port=${frontendConfig.port}`)
console.log(`后端代理: http://${backendConfig.host}:${backendConfig.port}`)

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      host: frontendConfig.host,
      port: frontendConfig.port,
      proxy: {
        '/api': {
          target: `http://${backendConfig.host}:${backendConfig.port}`,
          changeOrigin: true,
        },
      },
    },
    define: {
      // 将配置注入到前端
      __BACKEND_URL__: JSON.stringify(`http://${backendConfig.host}:${backendConfig.port}`),
    },
  }
})

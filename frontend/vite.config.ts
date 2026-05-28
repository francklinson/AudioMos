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

// 处理后端代理地址：如果 host 是 0.0.0.0 或 auto，使用 localhost 进行代理
// 因为 Node.js 的 HTTP 代理无法直接连接到 0.0.0.0
function getProxyTarget(host: string, port: number): string {
  if (host === '0.0.0.0' || host === 'auto' || host === '::') {
    return `http://localhost:${port}`
  }
  return `http://${host}:${port}`
}

const proxyTarget = getProxyTarget(backendConfig.host, backendConfig.port)

console.log(`前端配置: host=${frontendConfig.host}, port=${frontendConfig.port}`)
console.log(`后端配置: host=${backendConfig.host}, port=${backendConfig.port}`)
console.log(`后端代理: ${proxyTarget}`)

// 处理 host 配置，确保兼容性
// 如果 host 是 0.0.0.0，在 Vite 中使用 true 表示监听所有接口
// 这样可以避免 Node.js dns.lookup 解析 0.0.0.0 失败的问题
function getViteHost(host: string): string | boolean {
  if (host === '0.0.0.0' || host === '::') {
    // 使用 true 表示监听所有接口，Vite 内部会正确处理
    return true
  }
  if (host === 'auto') {
    // auto 模式也监听所有接口
    return true
  }
  return host
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  // 获取 Vite 可用的 host 配置
  const viteHost = getViteHost(frontendConfig.host)
  
  console.log(`Vite server config: host=${viteHost === true ? '0.0.0.0 (all interfaces)' : viteHost}, port=${frontendConfig.port}`)
  
  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      host: viteHost,
      port: frontendConfig.port,
      proxy: {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
    define: {
      // 将配置注入到前端
      // 注意：这里使用 window.location 相关的逻辑在前端动态构建 URL
      // 避免硬编码 host，确保在不同部署环境下都能正常工作
      __BACKEND_URL__: JSON.stringify(`http://${backendConfig.host === '0.0.0.0' || backendConfig.host === 'auto' ? 'localhost' : backendConfig.host}:${backendConfig.port}`),
    },
  }
})

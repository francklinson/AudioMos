import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

/**
 * 生产环境构建配置 - 用于前后端一体部署
 * 
 * 特点:
 * 1. 使用相对路径，不依赖具体 host
 * 2. API 请求使用相对路径 /api/*
 * 3. 构建输出到 backend/static 目录
 */

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  // 使用相对路径
  base: './',
  build: {
    // 输出到 backend/static 目录
    outDir: '../backend/static',
    // 清空输出目录
    emptyOutDir: true,
    // 生成 source map
    sourcemap: true,
    // 配置 rollup 选项
    rollupOptions: {
      output: {
        // 确保资源文件使用相对路径
        entryFileNames: 'assets/[name]-[hash].js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: (assetInfo) => {
          const info = assetInfo.name.split('.')
          const ext = info[info.length - 1]
          if (/\.(png|jpe?g|gif|svg|webp|ico)$/i.test(assetInfo.name)) {
            return 'assets/images/[name]-[hash][extname]'
          }
          if (/\.(woff2?|eot|ttf|otf)$/i.test(assetInfo.name)) {
            return 'assets/fonts/[name]-[hash][extname]'
          }
          return 'assets/[name]-[hash][extname]'
        },
      },
    },
  },
  define: {
    // 生产环境使用相对路径
    __BACKEND_URL__: JSON.stringify(''),
  },
})

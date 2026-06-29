# AudioMOS 前端

## 目录结构

```
frontend/
├── static/          # ✅ 活动前端 — 手写 HTML+JS+CSS 静态页面（当前使用）
│   ├── index.html
│   ├── css/app.css
│   ├── js/app.js
│   └── assets/
├── legacy-react/    # ⚠️ 已弃用 — 旧 React 源码，仅保留参考
│   ├── App.tsx, main.tsx, ...
│   └── pages/, components/, contexts/, services/
└── README.md
```

## 说明

- **`static/`**: 当前正在使用的前端。FastAPI 后端直接提供此目录的静态文件。
- **`legacy-react/`**: 旧版 React+Vite 前端，已不再使用和构建（package.json 等构建配置已移除）。


import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

/**
 * The dev server proxies both backend paths, so the browser only ever talks to
 * one origin and no CORS preflight or cross-origin WebSocket handshake is
 * involved during development.
 *
 *   /api    — prediction socket and the translation stream
 *   /models — the MediaPipe .task bundles and label_map.json, served straight
 *             from the Android app's assets so browser and phone load
 *             byte-identical models
 */
const BACKEND = 'http://localhost:8000'

/**
 * Root deployment by default ("/") for Vercel, custom domains and dev server.
 * Can be overridden via VITE_BASE (e.g. for GitHub Pages: VITE_BASE=/iSuara/ npm run build).
 */
const base = process.env.VITE_BASE ?? '/'

export default defineConfig({
  base,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND, changeOrigin: true, ws: true },
      '/models': { target: BACKEND, changeOrigin: true },
    },
  },
  // MediaPipe ships large prebuilt wasm; excluding it from dep optimization
  // stops Vite from trying to rewrite the bundle's own worker loading.
  optimizeDeps: { exclude: ['@mediapipe/tasks-vision'] },
})

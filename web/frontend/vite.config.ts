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
 * GitHub Pages serves a project site from https://<user>.github.io/<repo>/, so
 * every asset URL needs that prefix. Dev keeps "/" — the dev server is at the
 * origin root — which is why runtime asset paths use import.meta.env.BASE_URL
 * rather than hardcoding either value.
 *
 * Override for a different repo name or a user/organisation site (where the
 * base is "/"):  VITE_BASE=/ npm run build
 */
const base = process.env.VITE_BASE ?? (process.env.NODE_ENV === 'production' ? '/iSuara/' : '/')

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

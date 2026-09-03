/**
 * Where the API lives.
 *
 * Empty by default, which makes every call same-origin — what the Vite dev
 * proxy expects locally, and what a single-host deployment expects. Set
 * VITE_API_BASE at build time to point a statically hosted frontend at a
 * backend on another origin:
 *
 *   VITE_API_BASE=https://isuara-api.onrender.com npm run build
 *
 * This is a URL, not a credential, so baking it into the bundle is correct.
 * Nothing secret may ever go in a VITE_ variable: Vite inlines them into the
 * JavaScript it ships, where anyone can read them. The Gemini keys stay on the
 * server and are never sent to the browser.
 *
 * When this is set, requests become cross-origin, so the backend must name the
 * frontend's origin in ALLOWED_ORIGINS or the browser will block the response.
 */

/** Base URL for /api and /ws, with any trailing slash removed. */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/+$/, '')

/** Absolute URL for an API path such as `/api/health`. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

/**
 * WebSocket URL for a path such as `/ws/predict`.
 *
 * Derives the scheme from whichever origin is actually being addressed: the
 * page's own when same-origin, the configured base's otherwise — so an https
 * frontend talking to an https backend gets wss, which is the only combination
 * a browser permits from a secure page.
 */
export function wsUrl(path: string): string {
  const origin = API_BASE || window.location.origin
  return `${origin.replace(/^http/, 'ws')}${path}`
}

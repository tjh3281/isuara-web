/// <reference types="vite/client" />

// Declares import.meta.env, which the runtime asset paths use to pick up the
// deployment base path (see vite.config.ts). Also gives asset imports like the
// bundled .ttf their module types.

interface ImportMetaEnv {
  /**
   * Origin of the API, e.g. https://isuara-api.onrender.com. Empty or unset
   * means same-origin, which is what the dev proxy and a single-host deployment
   * both want. See lib/apiBase.ts.
   *
   * A URL, never a credential. Vite inlines every VITE_ variable into the
   * shipped JavaScript, so anything secret placed here would be public.
   */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

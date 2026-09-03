/**
 * The label map — the 98 gloss names and their per-language renderings.
 *
 * Loaded at runtime from /models/label_map.json, which the backend serves out of
 * the Android app's assets, rather than bundled into the JS. One file, one
 * source of truth: retraining the model and dropping in a new label map must not
 * require rebuilding the website to match.
 */

import type { LanguageDef } from './language'

/**
 * The model's no-sign class. This is a control-flow sentinel, not a word: it
 * gates sentence building and the auto-translate trigger, so it must never be
 * compared against a translated string.
 */
export const IDLE = 'Idle'

export interface LabelMap {
  labels: string[]
  /** gloss -> { en, zh, ... }. Absent for glosses with no translations. */
  translations: Record<string, Record<string, string>>
}

interface RawLabelMap {
  actions_ordered: string[]
  translations?: Record<string, Record<string, string>>
}

export async function loadLabelMap(): Promise<LabelMap> {
  // BASE_URL rather than a leading slash — see landmarkExtractor.ts. This is a
  // bundled static asset, so it loads with or without the backend.
  const response = await fetch(`${import.meta.env.BASE_URL}models/label_map.json`)
  if (!response.ok) {
    throw new Error(`label_map.json: HTTP ${response.status}`)
  }
  const raw: RawLabelMap = await response.json()

  const translations: Record<string, Record<string, string>> = {}
  for (const gloss of raw.actions_ordered) {
    const entry = raw.translations?.[gloss]
    if (!entry) continue
    const byLang: Record<string, string> = {}
    for (const [key, value] of Object.entries(entry)) {
      if (typeof value === 'string' && value.trim() !== '') byLang[key] = value
    }
    if (Object.keys(byLang).length > 0) translations[gloss] = byLang
  }

  return { labels: raw.actions_ordered, translations }
}

/**
 * The gloss rendered in `language`, or null when nothing extra should be shown:
 * Malay is canonical, "Idle" is a control-flow sentinel rather than a sign, and
 * an unmapped gloss should render no second row at all.
 *
 * Callers keep the Malay gloss as the value they store and send onward — this is
 * display only.
 */
export function glossIn(
  labelMap: LabelMap | null,
  gloss: string,
  language: LanguageDef,
): string | null {
  const key = language.labelMapKey
  if (!key || !labelMap) return null
  if (gloss === IDLE) return null

  // A low-confidence word is displayed with a trailing "?"; look up the word
  // itself and put the marker back afterwards.
  const clean = gloss.endsWith('?') ? gloss.slice(0, -1) : gloss
  const translated = labelMap.translations[clean]?.[key]
  if (!translated) return null
  return clean === gloss ? translated : `${translated}?`
}

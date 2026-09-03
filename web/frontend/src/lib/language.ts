/**
 * Display/speech language for glosses and translated sentences — a port of
 * service/Language.kt.
 *
 * Malay is canonical: sign glosses, the sentence buffer and the LLM prompt are
 * all Malay internally, and `labelMapKey` is null for it because no lookup is
 * needed. The other entries name their key in the `translations` object of
 * label_map.json.
 */

export type LanguageCode = 'MALAY' | 'ENGLISH' | 'MANDARIN' | 'TAMIL'

export interface LanguageDef {
  code: LanguageCode
  /** Key in label_map.json's `translations`, or null for canonical Malay. */
  labelMapKey: 'en' | 'zh' | 'ta' | null
  /** Key in a Translation object. */
  translationKey: 'ms' | 'en' | 'zh' | 'ta'
  /** BCP-47 tag for speech synthesis. */
  locale: string
  menuLabel: string
  /** Compact label for the language chip. */
  shortLabel: string
}

export const LANGUAGES: Record<LanguageCode, LanguageDef> = {
  MALAY: {
    code: 'MALAY',
    labelMapKey: null,
    translationKey: 'ms',
    locale: 'ms-MY',
    menuLabel: 'Malay only',
    shortLabel: 'MS',
  },
  ENGLISH: {
    code: 'ENGLISH',
    labelMapKey: 'en',
    translationKey: 'en',
    locale: 'en-US',
    menuLabel: 'English',
    shortLabel: 'EN',
  },
  MANDARIN: {
    code: 'MANDARIN',
    labelMapKey: 'zh',
    translationKey: 'zh',
    locale: 'zh-CN',
    menuLabel: 'Mandarin',
    shortLabel: '中',
  },
  TAMIL: {
    code: 'TAMIL',
    labelMapKey: 'ta',
    translationKey: 'ta',
    locale: 'ta-IN',
    menuLabel: 'Tamil',
    shortLabel: 'த',
  },
}

export const LANGUAGE_ORDER: LanguageCode[] = ['MALAY', 'ENGLISH', 'MANDARIN', 'TAMIL']

/** True when a second row should be rendered beneath the Malay one. */
export function isSecondary(language: LanguageDef): boolean {
  return language.labelMapKey !== null
}

/** One sentence rendered in every supported language. */
export interface Translation {
  ms: string
  en: string
  zh: string
  ta: string
}

export function forLanguage(translation: Translation, language: LanguageDef): string {
  return translation[language.translationKey]
}

/** Fallback used when no translator is configured or the call fails. */
export function ofRawGlosses(text: string): Translation {
  return { ms: text, en: text, zh: text, ta: text }
}

const STORAGE_KEY = 'isuara.language'

/**
 * The persisted language choice — the web equivalent of LanguagePreference.kt.
 *
 * Wrapped in try/catch because a private window or blocked site data makes the
 * accessor itself throw, and losing a display preference must never stop the
 * app from starting.
 */
export function loadLanguage(): LanguageCode {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored && stored in LANGUAGES) return stored as LanguageCode
  } catch {
    /* ignore */
  }
  return 'MALAY'
}

export function saveLanguage(code: LanguageCode): void {
  try {
    localStorage.setItem(STORAGE_KEY, code)
  } catch {
    /* ignore */
  }
}

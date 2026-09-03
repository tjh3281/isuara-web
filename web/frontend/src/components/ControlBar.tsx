/**
 * The bottom Row of ui/CameraScreen.kt: language chip, clear, Translate, speak.
 *
 * All four children are 56.dp (uniformButtonHeight) with 12.dp between them.
 * Icons are the Compose originals — Icons.Default.Delete and
 * Icons.Default.PlayArrow — so the row reads identically to the phone.
 *
 * The language control is a compact chip rather than a spinner for the same
 * reason it is on Android: this row already carries three other controls, and
 * the selected language is a one- or two-character label anyway.
 */

import { useEffect, useRef, useState } from 'react'

import { LANGUAGE_ORDER, LANGUAGES, type LanguageCode, type LanguageDef } from '../lib/language'

interface Props {
  language: LanguageDef
  onLanguageChange: (code: LanguageCode) => void
  onClear: () => void
  onTranslate: () => void
  onSpeak: () => void
  canTranslate: boolean
  canSpeak: boolean
  translationEnabled: boolean
}

export function ControlBar({
  language,
  onLanguageChange,
  onClear,
  onTranslate,
  onSpeak,
  canTranslate,
  canSpeak,
  translationEnabled,
}: Props) {
  const [menuOpen, setMenuOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!menuOpen) return
    const onPointerDown = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [menuOpen])

  return (
    <div className="controls">
      <div className="controls__language" ref={menuRef}>
        <button
          type="button"
          className="chip"
          onClick={() => setMenuOpen((open) => !open)}
          aria-haspopup="menu"
          aria-expanded={menuOpen}
        >
          {language.shortLabel}
        </button>
        {menuOpen && (
          <div className="menu" role="menu">
            {LANGUAGE_ORDER.map((code) => (
              <button
                key={code}
                type="button"
                role="menuitemradio"
                aria-checked={code === language.code}
                className={`menu__item${code === language.code ? ' menu__item--active' : ''}`}
                onClick={() => {
                  onLanguageChange(code)
                  setMenuOpen(false)
                }}
              >
                {LANGUAGES[code].menuLabel}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Icons.Default.Delete */}
      <button type="button" className="chip" onClick={onClear} title="Clear">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
          <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" />
        </svg>
        <span className="sr-only">Clear</span>
      </button>

      <button
        type="button"
        className="button"
        onClick={onTranslate}
        disabled={!canTranslate}
        title={
          translationEnabled
            ? 'Restructure the glosses into a sentence'
            : 'No Gemini API key configured — the raw glosses are used instead'
        }
      >
        Translate
      </button>

      {/* Icons.Default.PlayArrow */}
      <button type="button" className="fab" onClick={onSpeak} disabled={!canSpeak} title="Speak">
        <svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">
          <path d="M8 5v14l11-7z" />
        </svg>
        <span className="sr-only">Speak</span>
      </button>
    </div>
  )
}

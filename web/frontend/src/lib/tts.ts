/**
 * TtsService — port of service/TtsService.kt onto the Web Speech API.
 *
 * Speaks in the language the signer selected. As on Android, a voice for a given
 * language is frequently absent (Malay especially — Chrome on desktop often has
 * no ms-MY voice at all), so an unavailable voice falls back to Malay, then
 * Indonesian, which is close enough to be understood, and warns once rather than
 * going silently mute, which would read as a bug.
 */

import { forLanguage, type LanguageDef, type Translation } from './language'

const MALAY = 'ms'
const INDONESIAN = 'id'

export class TtsService {
  private voices: SpeechSynthesisVoice[] = []
  private warned = new Set<string>()

  constructor() {
    if (!this.supported) {
      console.warn('[tts] speechSynthesis unavailable in this browser')
      return
    }
    this.loadVoices()
    // Chrome populates the voice list asynchronously and fires this once ready;
    // reading it synchronously at startup usually returns an empty array.
    speechSynthesis.addEventListener('voiceschanged', this.loadVoices)
  }

  get supported(): boolean {
    return typeof window !== 'undefined' && 'speechSynthesis' in window
  }

  private loadVoices = (): void => {
    this.voices = speechSynthesis.getVoices()
  }

  /**
   * The best available voice for a BCP-47 tag.
   *
   * Matches the exact tag first, then the bare language subtag, so a browser
   * offering only "zh-TW" still speaks Mandarin when "zh-CN" was asked for.
   */
  private findVoice(locale: string): SpeechSynthesisVoice | null {
    const wanted = locale.toLowerCase()
    const base = wanted.split('-')[0]
    return (
      this.voices.find((v) => v.lang.toLowerCase() === wanted) ??
      this.voices.find((v) => v.lang.toLowerCase().startsWith(base + '-')) ??
      this.voices.find((v) => v.lang.toLowerCase() === base) ??
      null
    )
  }

  /** Malay, falling back to Indonesian (very close) then whatever exists. */
  private malayVoice(): SpeechSynthesisVoice | null {
    const malay = this.findVoice(MALAY)
    if (malay) return malay

    const indonesian = this.findVoice(INDONESIAN)
    if (indonesian) {
      this.warnOnce('ms', 'Malay TTS not available, using Indonesian (close to Malay)')
      return indonesian
    }

    this.warnOnce('ms', 'Malay & Indonesian TTS not available, using the default voice')
    return null
  }

  private warnOnce(key: string, message: string): void {
    if (this.warned.has(key)) return
    this.warned.add(key)
    console.warn(`[tts] ${message}`)
  }

  /**
   * Speak `text` in `language`, interrupting anything already playing.
   *
   * If the requested voice is unavailable this speaks `fallbackText` — the Malay
   * rendering — rather than reading Malay words with, say, a Mandarin voice.
   */
  speak(text: string, language: LanguageDef, fallbackText: string = text): void {
    if (!this.supported || !text.trim()) return
    speechSynthesis.cancel()

    let spoken = text
    let voice: SpeechSynthesisVoice | null

    if (language.translationKey === 'ms') {
      voice = this.malayVoice()
    } else {
      voice = this.findVoice(language.locale)
      if (!voice) {
        this.warnOnce(
          language.code,
          `${language.menuLabel} TTS unavailable, speaking Malay instead`,
        )
        spoken = fallbackText
        voice = this.malayVoice()
      }
    }

    if (!spoken.trim()) return

    const utterance = new SpeechSynthesisUtterance(spoken)
    if (voice) {
      utterance.voice = voice
      utterance.lang = voice.lang
    } else {
      utterance.lang = language.locale
    }
    // Matches the 0.9 rate the Android build sets — signed sentences are short
    // and a default-rate voice clips them past comprehension.
    utterance.rate = 0.9
    speechSynthesis.speak(utterance)
  }

  /** Speaks a translation in the selected language, falling back to its Malay. */
  speakTranslation(translation: Translation, language: LanguageDef): void {
    this.speak(forLanguage(translation, language), language, translation.ms)
  }

  stop(): void {
    if (this.supported) speechSynthesis.cancel()
  }

  close(): void {
    this.stop()
    if (this.supported) speechSynthesis.removeEventListener('voiceschanged', this.loadVoices)
  }
}

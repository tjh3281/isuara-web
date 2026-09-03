/**
 * iSuara on the web.
 *
 * Mirrors MainActivity + CameraScreen, structurally as well as visually: a
 * full-height Column on black, holding a portrait 3:4 camera box and then a
 * weight(1f) panel with the recognised word, the confidence bar, the sentence
 * card and the button row. There is deliberately no header, no landing screen
 * and no start button — the phone shows the camera the moment permission is
 * granted, so the web build does too.
 *
 * The auto-translate rule is ported exactly: when the model reports Idle while a
 * sentence is buffered, wait two seconds — long enough to be sure the signer has
 * stopped rather than paused — then translate and speak. Afterwards the sentence
 * is not cleared; the predictor is told to clear it at the moment the next sign
 * arrives, so the result stays readable instead of vanishing.
 */

import { useCallback, useEffect, useRef, useState } from 'react'

import { AvatarScreen } from './components/AvatarScreen'
import { BottomNav, type Tab } from './components/BottomNav'
import { CameraView } from './components/CameraView'
import { ControlBar } from './components/ControlBar'
import { SentencePanel } from './components/SentencePanel'
import { useSignPipeline } from './hooks/useSignPipeline'
import { IDLE } from './lib/labelMap'
import {
  LANGUAGES,
  loadLanguage,
  ofRawGlosses,
  saveLanguage,
  type LanguageCode,
  type Translation,
} from './lib/language'
import { TtsService } from './lib/tts'
import {
  IDLE_PROGRESS,
  translate as runTranslate,
  type DebateProgress,
} from './lib/translateClient'

/** How long the signer must hold still before the sentence is considered done. */
const AUTO_TRANSLATE_DELAY_MS = 2000

export default function App() {
  const pipeline = useSignPipeline()
  const { state, predictor, labelMap } = pipeline

  const ttsRef = useRef<TtsService | null>(null)
  if (!ttsRef.current) ttsRef.current = new TtsService()
  const tts = ttsRef.current

  const [languageCode, setLanguageCode] = useState<LanguageCode>(loadLanguage)
  const [translation, setTranslation] = useState<Translation | null>(null)
  const [isTranslating, setIsTranslating] = useState(false)
  const [progress, setProgress] = useState<DebateProgress>(IDLE_PROGRESS)
  const [showLandmarks, setShowLandmarks] = useState(false)
  const [tab, setTab] = useState<Tab>('camera')

  // Probed once by the pipeline when the camera starts; null means no backend,
  // which is the normal case for the statically-hosted build.
  const health = pipeline.health
  const language = LANGUAGES[languageCode]

  // Start the camera on mount, as MainActivity does once permission is granted.
  // Guarded by a ref rather than an empty dep array so React 18's StrictMode
  // double-invoke cannot open two streams.
  const startedRef = useRef(false)
  useEffect(() => {
    if (startedRef.current) return
    startedRef.current = true
    void pipeline.start()
  }, [pipeline])

  useEffect(() => () => tts.close(), [tts])

  /**
   * Runs the debate and shows the result.
   *
   * `speak` is false for the manual button, matching the Android build: pressing
   * Translate shows the sentence, pressing Speak says it. The automatic path
   * does both, because nobody is holding the phone to press anything.
   */
  const translateSentence = useCallback(
    async (speak: boolean) => {
      const words = predictor.getSentenceWords()
      if (words.length === 0 || isTranslating) return

      setIsTranslating(true)
      setTranslation(null)

      const rawSentence = words.join(' ')
      try {
        // The expression rides along so rule 8 can shift register. Weak
        // readings are dropped server-side rather than hedged, so sending
        // whatever we have is safe.
        const reading = pipeline.emotion
        const result = await runTranslate(
          words,
          setProgress,
          reading
            ? {
                descriptor: reading.descriptor,
                confidence: reading.confidence,
                isHighArousal: reading.isHighArousal,
              }
            : null,
        )
        setTranslation(result)
        if (speak) tts.speakTranslation(result, language)
      } catch (e) {
        // No translator configured, no network, or every interpreter failed —
        // fall back to the raw glosses, which are Malay, and speak them as Malay.
        console.warn('[translate] falling back to raw glosses', e)
        setTranslation(ofRawGlosses(rawSentence))
        if (speak) tts.speak(rawSentence, LANGUAGES.MALAY)
      } finally {
        setIsTranslating(false)
        // The candidates stay on screen after the run so the accordion can still
        // be opened to see who said what; only the stage returns to idle.
        setProgress((p) => ({ ...p, stage: 'IDLE' }))
      }
    },
    [isTranslating, language, predictor, tts, pipeline.emotion],
  )

  // ── auto-translate ──
  const isTranslatingRef = useRef(isTranslating)
  isTranslatingRef.current = isTranslating

  useEffect(() => {
    if (
      state.currentWord !== IDLE ||
      state.sentence.length === 0 ||
      isTranslatingRef.current ||
      state.isWaitingForNewSentence
    ) {
      return
    }

    const timer = setTimeout(async () => {
      await translateSentence(true)
      // Hold the text on screen until the next sign, rather than resetting now.
      predictor.prepareForNewSentence()
    }, AUTO_TRANSLATE_DELAY_MS)

    return () => clearTimeout(timer)
    // Deliberately keyed on the recognised word: a new word restarts the wait,
    // which is what makes a pause mid-sentence not trigger a translation.
  }, [
    state.currentWord,
    state.sentence.length,
    state.isWaitingForNewSentence,
    predictor,
    translateSentence,
  ])

  // Clear the previous result once a genuinely new sentence starts.
  useEffect(() => {
    if (state.sentence.length === 1 && !isTranslatingRef.current) setTranslation(null)
  }, [state.sentence.length])

  const handleLanguageChange = (code: LanguageCode) => {
    setLanguageCode(code)
    saveLanguage(code)
  }

  const handleClear = () => {
    predictor.resetAll()
    setTranslation(null)
    setIsTranslating(false)
    setProgress(IDLE_PROGRESS)
    tts.stop()
  }

  const handleSpeak = () => {
    if (translation) {
      tts.speakTranslation(translation, language)
      return
    }
    // Untranslated glosses are Malay, so speak them as Malay.
    const raw = state.sentence.join(' ')
    if (raw) tts.speak(raw, LANGUAGES.MALAY)
  }

  return (
    <div className="app">
      {/*
        Both tabs stay mounted and the inactive one is hidden rather than
        unmounted. Tearing down the camera tab would drop the camera stream and
        the sentence buffer on every tab switch, and tearing down the avatar tab
        would discard the WebGL context and the loaded clips.
      */}
      <div className="screen" hidden={tab !== 'camera'}>
        {/*
          No counterpart on the phone — the app cannot reach a state where the
          model is missing. Disappears entirely once the backend loads it.
        */}
        {health?.modelError && (
          <div className="notice">
            <strong>Sign recognition unavailable.</strong> Camera, tracking,
            translation and speech still work.
            <details>
              <summary>Why</summary>
              <pre>{health.modelError}</pre>
            </details>
          </div>
        )}

        <CameraView
          videoRef={pipeline.videoRef}
          keypoints={state.keypoints}
          bufferProgress={state.bufferProgress}
          fps={pipeline.fps}
          facingMode={pipeline.facingMode}
          showLandmarks={showLandmarks}
          status={pipeline.status}
          emotion={pipeline.emotion}
          onToggleLandmarks={() => setShowLandmarks((v) => !v)}
          onSwitchCamera={pipeline.switchCamera}
        />

        {/* Column(fillMaxWidth).weight(1f).padding(16.dp) */}
        <div className="panel">
          <SentencePanel
            state={state}
            labelMap={labelMap}
            language={language}
            translation={translation}
            isTranslating={isTranslating}
            progress={progress}
          />
          <ControlBar
            language={language}
            onLanguageChange={handleLanguageChange}
            onClear={handleClear}
            onTranslate={() => void translateSentence(false)}
            onSpeak={handleSpeak}
            canTranslate={state.sentence.length > 0 && !isTranslating}
            canSpeak={state.sentence.length > 0 || translation !== null}
            translationEnabled={health?.translationEnabled ?? false}
          />
        </div>
      </div>

      <div className="screen" hidden={tab !== 'avatar'}>
        <AvatarScreen />
      </div>

      <BottomNav tab={tab} onChange={setTab} />
    </div>
  )
}

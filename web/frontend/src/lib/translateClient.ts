/**
 * Client for the debate translator.
 *
 * The backend streams newline-delimited JSON so the UI can show which stage the
 * multi-agent pipeline is in — the browser equivalent of collecting
 * Translator.stage as a StateFlow on Android. A debate takes seconds; without
 * the stages it would look frozen.
 *
 * Failures are reported by throwing, never by returning an error string, so a
 * caller distinguishes a real translation from a failure by catching. Callers
 * fall back to the raw Malay glosses, exactly as CameraScreen.kt does.
 */

import { apiUrl } from './apiBase'
import type { Translation } from './language'

export type TranslationStage =
  | 'IDLE'
  | 'CONSULTING'
  | 'COLLECTED'
  | 'JUDGING'
  | 'DECIDING'

export const STAGE_LABELS: Record<TranslationStage, string> = {
  IDLE: '',
  CONSULTING: 'Consulting interpreters…',
  COLLECTED: 'Interpretations received',
  JUDGING: 'Weighing interpretations…',
  DECIDING: 'Choosing best translation…',
}

export interface Health {
  status: string
  model: string
  /** Verbatim load failure, so a broken setup is diagnosable in the browser. */
  modelError: string
  classes: number
  translationEnabled: boolean
  keySlots: number
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch(apiUrl('/api/health'))
  if (!response.ok) throw new Error(`health: HTTP ${response.status}`)
  return response.json()
}

/** One agent's slot in the debate — mirrors CandidateView in DebateProgress.kt. */
export interface CandidateView {
  index: number
  model: string
  sentence: string | null
  failed: boolean
}

/** The judge's decision — mirrors JudgeVerdict.kt. */
export interface JudgeVerdict {
  choice: number
  reason: string
}

/** Everything the UI needs to show the debate as it happens. */
export interface DebateProgress {
  stage: TranslationStage
  candidates: CandidateView[]
  verdict: JudgeVerdict | null
}

export const IDLE_PROGRESS: DebateProgress = {
  stage: 'IDLE',
  candidates: [],
  verdict: null,
}

/**
 * Display labels for the debate agents.
 *
 * LABELS ONLY — nothing here changes which model answers. Every request still
 * goes to the Gemini model the corresponding key is for; AGENT_MODELS in
 * web/backend/app/translator.py is what actually runs, and the server logs and
 * the wire protocol both carry the real id.
 *
 * Written down plainly because a reader seeing "DeepSeek" in the interface and
 * `gemini-3.1-flash-lite` in the logs should be able to find out why in one
 * place rather than assuming one of the two is a bug.
 */
const DISPLAY_NAMES: Record<string, string> = {
  'gemini-3.1-flash-lite': 'DeepSeek-V4-Flash',
  'gemini-2.5-flash': 'MiniMax-M2.7',
  'gemini-3.5-flash': 'Kimi-K2.6',
}

/** The label for a model id — its alias if it has one, otherwise the bare name. */
export function shortModelName(model: string): string {
  const slash = model.lastIndexOf('/')
  const bare = slash >= 0 ? model.slice(slash + 1) : model
  return DISPLAY_NAMES[bare] ?? bare
}

/** An emotion reading to steer register — mirrors EmotionReading.kt. */
export interface EmotionReading {
  descriptor: string
  confidence: number
  isHighArousal: boolean
}

/**
 * Runs a translation, invoking `onProgress` as the debate unfolds.
 *
 * Progress is reported incrementally rather than only at the end because agents
 * resolve at very different speeds — the Android notes ~9s to ~126s across its
 * three models — so waiting for all of them before showing anything means
 * minutes of dead air.
 *
 * Resolves with the chosen translation, or throws if every interpreter failed.
 */
export async function translate(
  words: string[],
  onProgress: (progress: DebateProgress) => void,
  emotion?: EmotionReading | null,
  signal?: AbortSignal,
): Promise<Translation> {
  const response = await fetch(apiUrl('/api/translate'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ words, emotion: emotion ?? null }),
    signal,
  })

  if (!response.ok || !response.body) {
    throw new Error(`translate: HTTP ${response.status}`)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let translation: Translation | null = null
  let error: string | null = null

  let progress: DebateProgress = { stage: 'CONSULTING', candidates: [], verdict: null }

  const handleLine = (line: string) => {
    if (!line.trim()) return
    const event = JSON.parse(line) as {
      stage?: TranslationStage
      candidates?: CandidateView[]
      candidate?: CandidateView
      verdict?: JudgeVerdict
      translation?: Translation
      error?: string
    }

    if (event.stage) progress = { ...progress, stage: event.stage }
    if (event.candidates) progress = { ...progress, candidates: event.candidates }
    if (event.candidate) {
      // Replace the matching slot in place, so the row keeps its position while
      // the others are still pending.
      const next = progress.candidates.map((c) =>
        c.index === event.candidate!.index ? event.candidate! : c,
      )
      progress = { ...progress, candidates: next }
    }
    if (event.verdict) progress = { ...progress, verdict: event.verdict }
    if (event.translation) translation = event.translation
    if (event.error) error = event.error

    onProgress(progress)
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // The last element is a partial line until the next chunk completes it.
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      lines.forEach(handleLine)
    }
    handleLine(buffer)
  } finally {
    reader.releaseLock()
  }

  if (error) throw new Error(error)
  if (!translation) throw new Error('translation stream ended with no result')
  return translation
}

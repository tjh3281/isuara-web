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
  const response = await fetch('/api/health')
  if (!response.ok) throw new Error(`health: HTTP ${response.status}`)
  return response.json()
}

/**
 * Runs a translation, invoking `onStage` as the pipeline advances.
 *
 * Resolves with the chosen translation, or throws if every interpreter failed.
 */
export async function translate(
  words: string[],
  onStage: (stage: TranslationStage) => void,
  signal?: AbortSignal,
): Promise<Translation> {
  const response = await fetch('/api/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ words }),
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

  const handleLine = (line: string) => {
    if (!line.trim()) return
    const event = JSON.parse(line) as {
      stage?: TranslationStage
      translation?: Translation
      error?: string
    }
    if (event.stage) onStage(event.stage)
    if (event.translation) translation = event.translation
    if (event.error) error = event.error
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
    onStage('IDLE')
    reader.releaseLock()
  }

  if (error) throw new Error(error)
  if (!translation) throw new Error('translation stream ended with no result')
  return translation
}

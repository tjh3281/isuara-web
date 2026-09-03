/**
 * Everything between the camera and the buttons: the sign being recognised
 * right now, the sentence built so far, and the translated result.
 *
 * The layout follows CameraScreen.kt's centre panel — current word, confidence
 * bar, then a card holding the gloss buffer and the translation — because that
 * ordering reflects the pipeline, and a demo audience reads it top to bottom as
 * the stages it is watching.
 */

import { glossIn, type LabelMap } from '../lib/labelMap'
import { forLanguage, isSecondary, type LanguageDef, type Translation } from '../lib/language'
import { STAGE_LABELS, type DebateProgress } from '../lib/translateClient'
import { DebatePanel } from './DebatePanel'
import type { PredictionState } from '../lib/signPredictor'

interface Props {
  state: PredictionState
  labelMap: LabelMap | null
  language: LanguageDef
  translation: Translation | null
  isTranslating: boolean
  progress: DebateProgress
}

export function SentencePanel({
  state,
  labelMap,
  language,
  translation,
  isTranslating,
  progress,
}: Props) {
  const secondaryWord = glossIn(labelMap, state.currentWord, language)
  const sentence = state.sentence.join(' ')

  // The second gloss row falls back to the Malay word for any gloss the label
  // map has no translation for, so the row stays aligned with the first.
  const secondarySentence =
    sentence && isSecondary(language)
      ? state.sentence.map((w) => glossIn(labelMap, w, language) ?? w).join(' ')
      : null

  // A fragment, not a wrapper: App owns the .panel element so the DOM mirrors
  // the single Compose Column that holds the word, the bar, the card and the
  // button row.
  return (
    <>
      <div className="current">
        {state.currentWord && (
          <>
            <p className={`current__word${state.isConfident ? ' current__word--confident' : ''}`}>
              {state.currentWord.toUpperCase()}
            </p>
            {secondaryWord && <p className="current__secondary">{secondaryWord}</p>}
          </>
        )}
      </div>

      <div className="confidence">
        <div
          className={`confidence__fill${state.isConfident ? ' confidence__fill--confident' : ''}`}
          style={{ width: `${Math.min(state.confidence, 1) * 100}%` }}
        />
      </div>

      <div className="card">
        <p className={`card__glosses${sentence ? '' : ' card__glosses--empty'}`}>
          {sentence || 'Waiting for signs…'}
        </p>
        {secondarySentence && <p className="card__glosses-secondary">{secondarySentence}</p>}

        {(isTranslating || translation) && (
          <div className="card__result">
            {isTranslating && progress.candidates.length === 0 ? (
              // Before the slots are announced there is nothing to expand, so
              // show the plain stage line rather than an empty accordion.
              <p className="card__stage">
                <span className="spinner" aria-hidden="true" />
                {STAGE_LABELS[progress.stage] || 'Refining grammar…'}
              </p>
            ) : (
              <>
                {translation && (
                  <>
                    <p className="card__translation">{translation.ms}</p>
                    {isSecondary(language) && (
                      <p className="card__translation-secondary">
                        {forLanguage(translation, language)}
                      </p>
                    )}
                  </>
                )}
                <DebatePanel progress={progress} active={isTranslating} />
              </>
            )}
          </div>
        )}
      </div>
    </>
  )
}
